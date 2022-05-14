from multiprocessing import Pool
import os
from random import randint
from time import sleep

def run(cmd, use_sleep=True):
    print(cmd)
    if use_sleep:
        sleep(randint(0, 200))
    os.system(cmd)

if __name__ == "__main__":
    all_cmd = [
        # "python -u main.py --seed=1 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=2 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=3 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=4 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=5 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=6 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=7 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=8 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=9 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=10 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=11 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=12 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=13 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=14 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=15 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=16 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=17 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=18 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=19 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",
        # "python -u main.py --seed=20 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24",

        # "python -u main.py --seed=1 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=2 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=3 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=4 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=5 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=6 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=7 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=8 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=9 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=10 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=11 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=12 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=13 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=14 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=15 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=16 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=17 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=18 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=19 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",
        # "python -u main.py --seed=20 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128",

        # "python -u main.py --seed=1 --data-folder=dataset/train/data_cvrp --algo=HGS --round-int --max-count=48",
        # "python -u main.py --seed=2 --data-folder=dataset/train/data_cvrp --algo=HGS --round-int --max-count=48",
        # "python -u main.py --seed=3 --data-folder=dataset/train/data_cvrp --algo=HGS --round-int --max-count=48",

        # "python -u main.py --seed=1 --data-folder=dataset/train/data_cvrp --algo=VNS --round-int --max-count=128",
        # "python -u main.py --seed=2 --data-folder=dataset/train/data_cvrp --algo=VNS --round-int --max-count=128",
        # "python -u main.py --seed=3 --data-folder=dataset/train/data_cvrp --algo=VNS --round-int --max-count=128",
        
        # "python -u main.py --seed=1 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24 --imitation-rate=0",
        # "python -u main.py --seed=2 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24 --imitation-rate=0",
        # "python -u main.py --seed=3 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24 --imitation-rate=0",
        # "python -u main.py --seed=4 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24 --imitation-rate=0",
        # "python -u main.py --seed=5 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24 --imitation-rate=0",
        # "python -u main.py --seed=6 --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24 --imitation-rate=0",

        # "python -u main.py --seed=1 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128 --imitation-rate=0",
        # "python -u main.py --seed=2 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128 --imitation-rate=0",
        # "python -u main.py --seed=3 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128 --imitation-rate=0",
        # "python -u main.py --seed=4 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128 --imitation-rate=0",
        # "python -u main.py --seed=5 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128 --imitation-rate=0",
        # "python -u main.py --seed=6 --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128 --imitation-rate=0",

        # "python -u main.py --seed=1 --data-folder=dataset/train/data_cvrp --algo=HGS --round-int --max-count=48 --imitation-rate=0",
        # "python -u main.py --seed=2 --data-folder=dataset/train/data_cvrp --algo=HGS --round-int --max-count=48 --imitation-rate=0",
        # "python -u main.py --seed=3 --data-folder=dataset/train/data_cvrp --algo=HGS --round-int --max-count=48 --imitation-rate=0",

        # "python -u main.py --seed=1 --data-folder=dataset/train/data_cvrp --algo=VNS --round-int --max-count=128 --imitation-rate=0",
        # "python -u main.py --seed=2 --data-folder=dataset/train/data_cvrp --algo=VNS --round-int --max-count=128 --imitation-rate=0",
        # "python -u main.py --seed=3 --data-folder=dataset/train/data_cvrp --algo=VNS --round-int --max-count=128 --imitation-rate=0",
        
        # "python -u main.py --seed=1 --data-folder=dataset/train/uniform_N500 --algo=HGS --max-count=4",
        # "python -u main.py --seed=2 --data-folder=dataset/train/uniform_N500 --algo=HGS --max-count=4",
        # "python -u main.py --seed=3 --data-folder=dataset/train/uniform_N500 --algo=HGS --max-count=4",

        # "python -u main.py --seed=1 --data-folder=dataset/train/uniform_N1000 --algo=HGS --max-count=2",
        # "python -u main.py --seed=2 --data-folder=dataset/train/uniform_N1000 --algo=HGS --max-count=2",
        # "python -u main.py --seed=3 --data-folder=dataset/train/uniform_N1000 --algo=HGS --max-count=2",

        # "python -u main.py --seed=1 --data-folder=dataset/train/uniform_N2000 --algo=HGS --max-count=1",
        # "python -u main.py --seed=2 --data-folder=dataset/train/uniform_N2000 --algo=HGS --max-count=1",
        # "python -u main.py --seed=3 --data-folder=dataset/train/uniform_N2000 --algo=HGS --max-count=1",

        "python -u main.py --seed=1 --data-folder=dataset/train/realworld --algo=HGS --max-count=16",
        "python -u main.py --seed=2 --data-folder=dataset/train/realworld --algo=HGS --max-count=16",
        "python -u main.py --seed=3 --data-folder=dataset/train/realworld --algo=HGS --max-count=16",
    ]

    folders = set([cmd.split("--data-folder=")[1].split(" --max-count")[0] for cmd in all_cmd])

    for folder in folders:
        run(f"python -u init_solution.py --data-folder={folder}", False)

    with Pool(3) as p:
        p.map(run, all_cmd)