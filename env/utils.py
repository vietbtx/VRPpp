from argparse import Namespace
from collections import defaultdict
import colorsys
from functools import lru_cache
import math
import os
import pickle
import re
from joblib import Memory
import numpy as np
import pandas as pd
import torch
import random
import plotly.graph_objects as go
import plotly.io as pio
from tensorboard.backend.event_processing import event_accumulator

try:
    from scour.scour import scourString
    from IPython.display import SVG
    pio.kaleido.scope.mathjax = None
except:
    pass


memory = Memory('__pycache__', verbose=0)


def read_number(s, mode=int):
    try:
        return mode(s)
    except:
        return read_number(s, float) if mode == int else s

@lru_cache
def distance_between(a_x, a_y, b_x, b_y):
    xd = a_x - b_x
    yd = a_y - b_y
    return math.sqrt(xd * xd + yd * yd)

@lru_cache
def angle_between(a_x, a_y, b_x, b_y):
    return math.atan2(b_y - a_y, b_x - a_x)

@memory.cache
def read_instance(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(": ", 1)
            if len(parts) == 2:
                data[parts[0].strip()] = read_number(parts[1].strip())
            else:
                parts = line.split()
                parts = [read_number(part) for part in parts]
                if parts[0] == "NODE_COORD_SECTION":
                    data['NODE_COORD_SECTION'] = []
                elif parts[0] == "DEMAND_SECTION":
                    data['DEMAND_SECTION'] = []
                elif parts[0] == "STATIONS_COORD_SECTION":
                    data['STATIONS_COORD_SECTION'] = []
                elif parts[0] == "DEPOT_SECTION":
                    data['DEPOT'] = 1
                    break
                elif len(parts) == 3:
                    data['NODE_COORD_SECTION'].append(parts)
                elif len(parts) == 2:
                    data['DEMAND_SECTION'].append(parts)
                elif len(parts) == 1:
                    if 'STATIONS_COORD_SECTION' not in data:
                        continue
                    data['STATIONS_COORD_SECTION'].append(parts[0])
    return data

def find_next_id(instance, solution, demand, energy, consumption, check=True):
    prev_node = instance.nodes[solution[-1]]
    distances = {}
    for node in instance.demands:
        if node.id in solution:
            continue
        distance = node.distance_to(prev_node)
        if 0 < distance*consumption <= energy and node.demand <= demand:
            if check:
                next_keys = find_next_id(instance, solution + [node.id], demand-node.demand, energy-distance*consumption, consumption, False)
            if not check or len(next_keys) > 0:
                distances[node.id] = distance
    keys = list(distances.keys())
    if len(keys) == 0 and prev_node.is_demand:
        for node in instance.depots + instance.stations:
            distance = node.distance_to(prev_node)
            if 0 < distance*consumption <= energy:
                distances[node.id] = distance
    keys = list(distances.keys())
    if len(keys) > 0:
        keys.sort(key=lambda x: distances[x])
    return keys

@memory.cache(ignore=["instance"])
def generate_init_tours(instance, name, init_mode='clockhand', round_int=False):
    from .InitSolution import InitClockHand
    print(f"Running initial solution: {name}")
    if init_mode == 'clockhand':
        init = InitClockHand.from_instance(instance, round_int)
        logs = []
        n_samples = 32
        for angle in range(-n_samples, n_samples):
            angle = angle / n_samples * math.pi
            solution = init.clock_hand_partition(angle, init.demand_nodes)
            solution = [node.id for tour in solution for node in tour.nodes]
            solution = [id for i, id in enumerate(solution) if solution[i] != solution[i-1]] + [init.depot.id]
            score = instance.evaluation(solution)
            logs.append((solution, score))
        solution, _ = min(logs, key=lambda x: x[1])
    elif init_mode == 'dbca':
        solution = instance.make_env('VNS').init_solution()
    elif init_mode == 'default':
        solution = instance.make_env().init_solution()
        if instance.mode == "EVRP" and instance.args.algo != "VNS":
            vrp_repairer = instance.make_env("VNS")
            vrp_repairer.step(solution)
            solution = vrp_repairer.get_best_solution()
    score = instance.evaluation(solution)
    print(f"Init completed! Instance {name} - cost = {score:.3f}")
    return solution

def angle_comparator(nodes):
    node_points = [[node.x, node.y] for node in nodes][:-1]
    depot = node_points[0]
    center = np.mean(node_points, 0)
    angle = angle_between(depot[0], depot[1], center[0], center[1])
    return angle

def sort_tours_by_center(tours):
    tours = sorted(tours, key=lambda nodes: angle_comparator(nodes))
    return tours

def convert_solution_to_tours(nodes, solution):
    tours = []
    tour = []
    for node_id in solution:
        node = nodes[node_id]
        tour.append(node)
        if node.is_depot:
            if len(tour) > 1:
                tours.append(tour)
            tour = [node]
    tours = sort_tours_by_center(tours)
    return tours

def plot_solution(nodes, tours, title=None):
    fig = go.Figure()
    fig.update_layout(width=320, height=320)
    if title is not None:
        fig.update_layout(title_text=title, title_x=0.5)
    
    colors = ['#dee1fe', '#c6cafd', '#7C85FB', '#636EFA', '#1929F8', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    all_pos = []
    for k, tour in enumerate(tours):
        pos_x, pos_y = [], []
        for node in tour:
            pos_x.append(node.x)
            pos_y.append(node.y)
        all_pos.append((pos_x, pos_y, f"EVRP{k+1}"))
    
    pos_x_2, pos_y_2 = [], []
    for pos_x, pos_y, name in all_pos:
        pos_x_2 += [pos_x[0], pos_x[1], pos_x[0], pos_x[-2]]
        pos_y_2 += [pos_y[0], pos_y[1], pos_y[0], pos_y[-2]]
    pos_x_2.append(pos_x[0])
    pos_y_2.append(pos_y[0])

    fig.add_trace(go.Scatter(x=pos_x_2, y=pos_y_2, mode='lines', name=name, line_color=colors[1], line_width=2))
    
    for pos_x, pos_y, name in all_pos:
        fig.add_trace(go.Scatter(x=pos_x[1:-1], y=pos_y[1:-1], mode='lines', name=name, line_color=colors[2], line_width=2))
    
    station_x, station_y = [], []
    for node in nodes:
        if node.is_depot:
            depot = node
        elif node.is_station:
            station_x.append(node.x)
            station_y.append(node.y)
    if len(station_x) > 1:
        fig.add_trace(go.Scatter(x=station_x, y=station_y, mode='markers', name="station", marker_color="black", marker_size=10, marker_symbol="square"))
    fig.add_trace(go.Scatter(x=[depot.x], y=[depot.y], mode='markers', name="depot", marker_color="red", marker_size=12, marker_symbol="hexagram"))

    fig.update_layout(template='plotly_white', margin=dict(l=4, r=4, t=4, b=4, pad=4, autoexpand=True))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False)
    return fig

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def swap_and_flatten(arr):
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

def load_scores(path="logs"):
    os.makedirs(path, exist_ok=True)
    if os.path.isfile(f'{path}/best_scores.pkl'):
        with open(f'{path}/best_scores.pkl', 'rb') as f:
            return pickle.load(f)
    return {}

def save_scores(best_scores, path="logs"):
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/best_scores.pkl', 'wb') as f:
        pickle.dump(best_scores, f)

def compare(key):
    items = re.split(r'(\d+)', key)
    items = tuple(read_number(x) for x in items)
    return items
    
def sort_instances(names):
    return sorted(names, key=compare)

def read_tensorboard_event(folder):
    for file in os.listdir(folder):
        if "events" in file:
            event = event_accumulator.EventAccumulator(os.path.join(folder, file))
            event.Reload()
            return event

def compute_gap(scores, heuristic_scores):
    gaps = []
    for name, score in scores.items():
        baseline_score = heuristic_scores[name]
        gap = (baseline_score - score) / baseline_score
        gaps.append(gap)
    average_gap = np.mean(gaps)
    return average_gap

@memory.cache
def load_all_scores(log_folder, seeds, keys):
    data = defaultdict(list)
    for seed in seeds:
        folder = f"{log_folder}/seed_{seed}"
        data["Folder"].append(folder)
        event = read_tensorboard_event(folder)
        steps = event.Scalars("steps")
        running_time = steps[-1].wall_time - steps[0].wall_time
        for key in keys:
            scores = [x.value for x in event.Scalars(f"scores/{key}")]
            data[key].append(scores)
        data["Running Time"].append(round(running_time, 2))
    df = pd.DataFrame(data).set_index(["Folder", "Running Time"])
    return df

@memory.cache
def load_all_evrp_rand_scores(log_folders, instance_keys):
    data = defaultdict(list)
    for folder in log_folders:
        data["Folder"].append(folder)
        event = read_tensorboard_event(folder)
        steps = event.Scalars("steps")
        running_time = steps[-1].wall_time - steps[0].wall_time
        for key in instance_keys:
            scores = [x.value for x in event.Scalars(f"scores/{key}")]
            data[key].append(scores)
        data["Running Time"].append(round(running_time, 2))
    df = pd.DataFrame(data).set_index(["Folder", "Running Time"])
    return df

def instance_fig(df, key, folder="dataset/train/data_evrp_wcci", return_seed=False):
    from env.VRPInstance import VRPInstance
    best_folder, _ = df[key].idxmin()
    data = load_scores(best_folder)[key]
    solution = data.solution
    args = Namespace(round_int=False, algo=None)
    if return_seed:
        seed = best_folder.split("_")[-1]
        key = f"seed_{seed}/{key}"
    instance = VRPInstance.from_path(f"{folder}/{key}", args, f"sub_{key}")
    instance.solution = solution
    fig = instance.plot()
    if return_seed:
        return fig, data.score, seed
    else:
        return fig, data.score

def get_gap_df(df, name, running_time, heuristic_scores):
    results = defaultdict(list)
    n = max(df.applymap(lambda x: len(x)).max())
    for i in range(n):
        score_df = df.applymap(lambda x: min(x[:i+1])).min()
        gap = compute_gap(score_df, heuristic_scores)
        results[name].append(gap)
        results["Running Time"].append((i+1)*running_time/n)
    results = pd.DataFrame(results)
    return results

def get_gap_evrp_rand_df(df, name, running_time, heuristic_scores):
    results = defaultdict(list)
    n = max(df.applymap(lambda x: len(x)).max())
    for i in range(n):
        score_df = df.applymap(lambda x: min(x[:i+1]))
        data = {}
        for index, row in score_df.iterrows():
            seed = index[0].split("/")[-1]
            row.index = [seed + "_" + x.split(".evrp")[0] for x in row.keys()]
            data.update(row)
        gap = compute_gap(data, heuristic_scores)
        results[name].append(gap)
        results["Running Time"].append((i+1)*running_time/n)
    results = pd.DataFrame(results)
    return results

def show_plotly_figure(fig):
    options = Namespace()
    options.enable_viewboxing = True
    options.enable_id_stripping = True
    options.enable_comment_stripping = True
    options.shorten_ids = True
    options.indent = None
    svg = fig.to_image("svg")
    svg = scourString(svg, options)
    return SVG(svg)