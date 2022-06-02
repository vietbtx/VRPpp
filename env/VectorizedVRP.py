

from functools import partial
from multiprocessing import Array, Pipe, Pool, Process
import os
from typing import List

from .VRPEnv import VRP
from .VRPInstance import VRPInstance
from .utils import sort_instances


def get_filenames(args):
    file_names = os.listdir(args.data_folder)
    file_names = sort_instances(file_names)
    file_names = [os.path.join(args.data_folder, file_name) for file_name in file_names]
    return file_names


def init_instances(args):
    file_names = get_filenames(args)
    func = partial(VRPInstance.from_path, args=args)
    with Pool(args.n_envs) as p:
        p.map(func, file_names)
    # for file_name in file_names:
    #     func(file_name)


def read_instances(args):
    file_names = get_filenames(args)
    instances: List[VRPInstance] = []
    for file_name in file_names:
        instance = VRPInstance.from_path(file_name, args=args)
        instances.append(instance)
    return instances

    
def worker(remote, parent_remote, args, arr, worker_id):
    parent_remote.close()
    args.seed = worker_id + args.seed*1024
    instances = read_instances(args)
    game = VRP(instances, args)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            reward, done, info = game.step(data, arr, worker_id)
            remote.send((reward, done, info))
        elif cmd == 'get_current_state':
            state = game.state()
            remote.send(state)
        elif cmd == 'update_solution':
            info = game.update_solution(data)
            remote.send(info)
        elif cmd == 'close':
            remote.close()
            break


class VectorizedVRP(object):

    def __init__(self, args):
        instances = read_instances(args)
        self.env = VRP(instances, args)
        self.waiting = False
        self.closed = False
        self.arr = Array('i', [0]*args.n_envs)
        self.ps, self.remotes, self.work_remotes = [], [], []
        for worker_id in range(args.n_envs):
            remote, work_remote = Pipe()
            self.ps.append(Process(target=worker, args=(work_remote, remote, args, self.arr, worker_id)))
            self.work_remotes.append(work_remote)
            self.remotes.append(remote)
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.n_envs = args.n_envs

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        rewards, dones, infos = zip(*results)
        return rewards, dones, infos

    def step(self, actions):
        for i in range(self.n_envs):
            self.arr[i] = 0
        self.step_async(actions)
        results = self.step_wait()
        return results

    def current_states_async(self):
        for remote in self.remotes:
            remote.send(('get_current_state', None))
        self.waiting = True
    
    def update_solution_async(self, scores):
        for remote in self.remotes:
            remote.send(('update_solution', scores))
        self.waiting = True
    
    def update_solution_wait(self):
        states = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return states

    def current_states_wait(self):
        states = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return states

    def get_current_states(self):
        self.current_states_async()
        return self.current_states_wait()
    
    def update_solution(self, scores):
        self.update_solution_async(scores)
        return self.update_solution_wait()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
