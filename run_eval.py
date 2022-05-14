from multiprocessing import Pool
import os
from random import randint
from time import sleep

def run(cmd):
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    all_cmd = [
        # "python -u evaluate.py --seed=1 --data-folder=dataset/test/data_evrp_random/seed_1 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_1/model.pt",
        # "python -u evaluate.py --seed=2 --data-folder=dataset/test/data_evrp_random/seed_2 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_2/model.pt",
        # "python -u evaluate.py --seed=3 --data-folder=dataset/test/data_evrp_random/seed_3 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_3/model.pt",
        # "python -u evaluate.py --seed=4 --data-folder=dataset/test/data_evrp_random/seed_4 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_4/model.pt",
        # "python -u evaluate.py --seed=5 --data-folder=dataset/test/data_evrp_random/seed_5 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_5/model.pt",
        # "python -u evaluate.py --seed=6 --data-folder=dataset/test/data_evrp_random/seed_6 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_6/model.pt",
        # "python -u evaluate.py --seed=7 --data-folder=dataset/test/data_evrp_random/seed_7 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_7/model.pt",
        # "python -u evaluate.py --seed=8 --data-folder=dataset/test/data_evrp_random/seed_8 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_8/model.pt",
        # "python -u evaluate.py --seed=9 --data-folder=dataset/test/data_evrp_random/seed_9 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_9/model.pt",
        # "python -u evaluate.py --seed=10 --data-folder=dataset/test/data_evrp_random/seed_10 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_10/model.pt",
        # "python -u evaluate.py --seed=11 --data-folder=dataset/test/data_evrp_random/seed_11 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_11/model.pt",
        # "python -u evaluate.py --seed=12 --data-folder=dataset/test/data_evrp_random/seed_12 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_12/model.pt",
        # "python -u evaluate.py --seed=13 --data-folder=dataset/test/data_evrp_random/seed_13 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_13/model.pt",
        # "python -u evaluate.py --seed=14 --data-folder=dataset/test/data_evrp_random/seed_14 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_14/model.pt",

        # "python -u evaluate.py --seed=1 --data-folder=dataset/test/data_evrp_random/seed_1 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_1/model.pt",
        # "python -u evaluate.py --seed=2 --data-folder=dataset/test/data_evrp_random/seed_2 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_2/model.pt",
        # "python -u evaluate.py --seed=3 --data-folder=dataset/test/data_evrp_random/seed_3 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_3/model.pt",
        # "python -u evaluate.py --seed=4 --data-folder=dataset/test/data_evrp_random/seed_4 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_4/model.pt",
        # "python -u evaluate.py --seed=5 --data-folder=dataset/test/data_evrp_random/seed_5 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_5/model.pt",
        # "python -u evaluate.py --seed=6 --data-folder=dataset/test/data_evrp_random/seed_6 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_6/model.pt",
        # "python -u evaluate.py --seed=7 --data-folder=dataset/test/data_evrp_random/seed_7 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_7/model.pt",
        # "python -u evaluate.py --seed=8 --data-folder=dataset/test/data_evrp_random/seed_8 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_8/model.pt",
        # "python -u evaluate.py --seed=9 --data-folder=dataset/test/data_evrp_random/seed_9 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_9/model.pt",
        # "python -u evaluate.py --seed=10 --data-folder=dataset/test/data_evrp_random/seed_10 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_10/model.pt",
        # "python -u evaluate.py --seed=11 --data-folder=dataset/test/data_evrp_random/seed_11 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_11/model.pt",
        # "python -u evaluate.py --seed=12 --data-folder=dataset/test/data_evrp_random/seed_12 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_12/model.pt",
        # "python -u evaluate.py --seed=13 --data-folder=dataset/test/data_evrp_random/seed_13 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_13/model.pt",
        # "python -u evaluate.py --seed=14 --data-folder=dataset/test/data_evrp_random/seed_14 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_14/model.pt",

        # "python -u evaluate.py --seed=1 --data-folder=dataset/test/data_evrp_random/seed_1 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_1/model.pt",
        # "python -u evaluate.py --seed=2 --data-folder=dataset/test/data_evrp_random/seed_2 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_2/model.pt",
        # "python -u evaluate.py --seed=3 --data-folder=dataset/test/data_evrp_random/seed_3 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_3/model.pt",
        # "python -u evaluate.py --seed=4 --data-folder=dataset/test/data_evrp_random/seed_4 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_4/model.pt",
        # "python -u evaluate.py --seed=5 --data-folder=dataset/test/data_evrp_random/seed_5 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_5/model.pt",
        # "python -u evaluate.py --seed=6 --data-folder=dataset/test/data_evrp_random/seed_6 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_6/model.pt",
        # "python -u evaluate.py --seed=7 --data-folder=dataset/test/data_evrp_random/seed_7 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_1/model.pt",
        # "python -u evaluate.py --seed=8 --data-folder=dataset/test/data_evrp_random/seed_8 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_2/model.pt",
        # "python -u evaluate.py --seed=9 --data-folder=dataset/test/data_evrp_random/seed_9 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_3/model.pt",
        # "python -u evaluate.py --seed=10 --data-folder=dataset/test/data_evrp_random/seed_10 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_4/model.pt",
        # "python -u evaluate.py --seed=11 --data-folder=dataset/test/data_evrp_random/seed_11 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_5/model.pt",
        # "python -u evaluate.py --seed=12 --data-folder=dataset/test/data_evrp_random/seed_12 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_6/model.pt",
        # "python -u evaluate.py --seed=13 --data-folder=dataset/test/data_evrp_random/seed_13 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_1/model.pt",
        # "python -u evaluate.py --seed=14 --data-folder=dataset/test/data_evrp_random/seed_14 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_2/model.pt",

        # "python -u evaluate.py --seed=1 --data-folder=dataset/test/data_evrp_random/seed_1 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_1/model.pt",
        # "python -u evaluate.py --seed=2 --data-folder=dataset/test/data_evrp_random/seed_2 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_2/model.pt",
        # "python -u evaluate.py --seed=3 --data-folder=dataset/test/data_evrp_random/seed_3 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_3/model.pt",
        # "python -u evaluate.py --seed=4 --data-folder=dataset/test/data_evrp_random/seed_4 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_4/model.pt",
        # "python -u evaluate.py --seed=5 --data-folder=dataset/test/data_evrp_random/seed_5 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_5/model.pt",
        # "python -u evaluate.py --seed=6 --data-folder=dataset/test/data_evrp_random/seed_6 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_6/model.pt",
        # "python -u evaluate.py --seed=7 --data-folder=dataset/test/data_evrp_random/seed_7 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_1/model.pt",
        # "python -u evaluate.py --seed=8 --data-folder=dataset/test/data_evrp_random/seed_8 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_2/model.pt",
        # "python -u evaluate.py --seed=9 --data-folder=dataset/test/data_evrp_random/seed_9 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_3/model.pt",
        # "python -u evaluate.py --seed=10 --data-folder=dataset/test/data_evrp_random/seed_10 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_4/model.pt",
        # "python -u evaluate.py --seed=11 --data-folder=dataset/test/data_evrp_random/seed_11 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_5/model.pt",
        # "python -u evaluate.py --seed=12 --data-folder=dataset/test/data_evrp_random/seed_12 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_6/model.pt",
        # "python -u evaluate.py --seed=13 --data-folder=dataset/test/data_evrp_random/seed_13 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_1/model.pt",
        # "python -u evaluate.py --seed=14 --data-folder=dataset/test/data_evrp_random/seed_14 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_2/model.pt",

        "python -u evaluate.py --seed=1 --data-folder=dataset/test/uniform_N500 --algo=HGS --max-count=4 --model-path=logs/dataset_train_uniform_N500_HGS_0.1/seed_1/model.pt",
        "python -u evaluate.py --seed=2 --data-folder=dataset/test/uniform_N500 --algo=HGS --max-count=4 --model-path=logs/dataset_train_uniform_N500_HGS_0.1/seed_2/model.pt",
        "python -u evaluate.py --seed=3 --data-folder=dataset/test/uniform_N500 --algo=HGS --max-count=4 --model-path=logs/dataset_train_uniform_N500_HGS_0.1/seed_3/model.pt",
        
    ]

    folders = set([cmd.split("--data-folder=")[1].split(" --max-count")[0] for cmd in all_cmd])

    for folder in folders:
        run(f"python -u init_solution.py --data-folder={folder}")

    # with Pool(3) as p:
    #     p.map(run, all_cmd)
    for cmd in all_cmd:
        run(cmd)