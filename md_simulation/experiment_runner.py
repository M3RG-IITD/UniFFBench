import random
import time

## Keep this True for experiment running. False for debug mode.
HACKED_REPLICA = True
if HACKED_REPLICA:
    time.sleep(random.randint(10, 300))

import argparse
import datetime
import os
import subprocess
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import yaml
from aim import Run
from ase.io import read
from experiments.utils.utils import _get_next_version
from loguru import logger
from matsciml.interfaces.ase import MatSciMLCalculator

from file_utils import InProgressExperimentTracker
from models._equiformerv2 import load_pretrained_equiformerv2
from models._mattersim import load_pretrained_mattersim
from models._orb import load_pretrained_orb
# from models._orb import load_pretrained_orb_v3
from models._sevennet import load_pretrained_sevennet
from models.matgl_pretrained import load_pretrained_matgl
from models.pretrained_mace import load_pretrained_mace
from npt_simulation import run

def _get_next_version(root_dir: str) -> str:
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    existing_versions = []
    for d in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
            existing_versions.append(int(d.split("_")[1]))

    if len(existing_versions) == 0:
        return "version_0"

    return f"version_{max(existing_versions) + 1}"


def update_completion_file(completions_file):
    def parse_line(line):
        parts = line.split(",")
        return int(parts[0]), datetime.datetime.fromisoformat(parts[1])

    if os.path.isfile(completions_file):
        with open(completions_file, "r") as f:
            lines = f.read().split("\n")
            completed = [parse_line(line) for line in lines if line]
            completed.sort()
            if len(completed) > 0:
                index = completed[-1][0] + 1
            else:
                index = 0
    else:
        completed = []
        index = 0

    current_time = datetime.datetime.now()
    with open(completions_file, "a+") as f:
        f.write(f"{index},{current_time.isoformat()}\n")

    if len(completed) > 1:
        time_diffs = [
            (completed[i][1] - completed[i - 1][1]).total_seconds()
            for i in range(1, len(completed))
        ]
        average_time_diff = sum(time_diffs) / len(time_diffs)
    else:
        average_time_diff = None

    return index, average_time_diff


def get_calculator():
    return MatSciMLCalculator


def get_model(model_name):
    if model_name in ["chgnet_dgl", "m3gnet_dgl"]:
        return load_pretrained_matgl(model_name)
    if model_name in ["mattersim"]:
        return load_pretrained_mattersim()
    if model_name in ["orb"]:
        return load_pretrained_orb()
    if model_name in ["orb_v3"]:
        return load_pretrained_orb_v3()
    if model_name in ["sevennet"]:
        return load_pretrained_sevennet()
    if model_name in ["equiformerv2"]:
        return load_pretrained_equiformerv2()
    if model_name in ["mace_pyg"]:
        return load_pretrained_mace(model_name)


def calculator_from_model(args):
    calc = get_calculator()
    model = get_model(args.model_name)
    calc = calc(model, matsciml_model=False)
    return calc


def log(self, d, step=0):
    for key, value in d.items():
        if not isinstance(value, str):
            self.track(value, name=key, step=step)
        else:
            self.set(key, value)


def setup_logger(
    project: str,
    entity: str,
    config=None,
) -> None:
    experiment_logger = Run(repo=config.aim_repo, experiment=f"{config.project}")

    config_dict = {k: str(v) for k, v in config.__dict__.items()}
    for k in ["command", "entity", "experiment_times_file"]:
        config_dict.pop(k)

    experiment_logger.log = MethodType(log, experiment_logger)
    experiment_logger.log(config_dict)
    return experiment_logger


def log_hardware_environment(experiment_logger):
    sys_info = (subprocess.check_output("lscpu", shell=True).strip()).decode()
    sys_info = sys_info.split("\n")
    try:
        model = [_ for _ in sys_info if "Model name:" in _]
        cpu_type = model[0].split("  ")[-1]
        if not args.debug:
            experiment_logger.log({"cpu_type": cpu_type})
    except Exception:
        pass


def dump_cli_args(source_folder):
    with open(results_dir.joinpath("cli_args.yaml"), "a") as f:
        yaml.safe_dump({"file_name": source_folder}, f, indent=2)


def get_source_folder(args):
    cif_files_dir = args.input_dir
    dirs = os.listdir(cif_files_dir)
    dirs.sort()
    source_folder = dirs[args.index]
    source_folder_path = os.path.join(cif_files_dir, source_folder)
    logger.info("Reading folder number:", source_folder)
    return source_folder, source_folder_path


def main(args):
    aim_setup = {"project": args.project, "entity": args.entity, "config": args}
    if not args.debug:
        experiment_logger = setup_logger(**aim_setup)
        pod_name = subprocess.run(
            ["cat", "/etc/hostname"], capture_output=True, text=True, check=True
        ).stdout.strip("\n")
        experiment_logger.log({"pod_name": pod_name})

        log_hardware_environment(experiment_logger)
    else:
        experiment_logger = None

    source_folder, source_folder_path = get_source_folder(args)
    dump_cli_args(source_folder=source_folder)

    calculator = calculator_from_model(args)

    assert os.path.isdir(
        source_folder_path
    ), f"Source folder path is not a directory: {source_folder_path}"

    for file in os.listdir(source_folder_path):
        file_path = os.path.join(source_folder_path, file)
        # print("filepath kya hai:",file_path)
        # print("file_namekya hai", file)

        temperature, pressure = file.split("_")[2:4]
        print(temperature,pressure)
        # pressure = pressure.strip("_")
        # temperature, pressure = float(temperature), float(pressure[0:-4])
        temperature, pressure = float(temperature), float(pressure)
        atoms = read(file_path)
        atoms.calc = calculator
        run(atoms, args, temperature, pressure, file, experiment_logger)

    if not args.debug:
        # Finish the aim run
        experiment_logger.close()


if __name__ == "__main__":
    # Seed for the Python random module
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)  # if you are using multi-GPU.
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("--index", type=int, default=0, help="index of folder")
    parser.add_argument("--runsteps", type=int, default=50000)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_atoms", type=int, default=200)
    parser.add_argument("--trajdump_interval", type=int, default=10)
    parser.add_argument("--minimize_steps", type=int, default=1000)
    parser.add_argument("--thermo_interval", type=int, default=10)
    parser.add_argument("--log_dir_base", type=Path, default="./simulation_results") #default="./simulation_results")
    parser.add_argument("--replica", action="store_true")
    parser.add_argument("--project", type=str, default="debug")
    parser.add_argument("--entity", type=str, default="sajidmannan")
    parser.add_argument("--aim_repo", type=str, default="./simulation_results/aim_repo")
    parser.add_argument("--experiment_times_file", type=str, default="./simulation_results/aim_repo/experiment_times.log")

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.replica:
        os.makedirs(str(args.log_dir_base.joinpath(args.model_name)), exist_ok=True)
        os.makedirs(str(args.log_dir_base.joinpath(args.model_name)), exist_ok=True)
        args.experiment_times_file = str(
            args.log_dir_base.joinpath(args.model_name, "experiment_times.txt")
        )
        completions_file = str(
            args.log_dir_base.joinpath(args.model_name, "completed.txt")
        )
        args.index, args.avg_completion_time = update_completion_file(completions_file)

    in_progress = InProgressExperimentTracker(
        track_file=str(args.log_dir_base.joinpath(args.model_name, "in_progress.json"))
    )

    if args.index > 2684:
        time.sleep(1_000_000)
        os._exit(0)

    log_dir_base = args.log_dir_base.joinpath(args.model_name, str(args.index))
    results_dir = log_dir_base.joinpath(_get_next_version(log_dir_base))
    results_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir = results_dir
    if args.debug:
        args.results_dir = "./debug_logs"

    with open(results_dir.joinpath("cli_args.yaml"), "w") as f:
        command = "python experiment_runner.py " + " ".join(
            f"--{k} {v}" for k, v in vars(args).items()
        )
        args.command = command
        yaml.safe_dump({k: str(v) for k, v in args.__dict__.items()}, f, indent=2)

    with open(results_dir.joinpath("cpu_spec.txt"), "w") as f:
        result = subprocess.run(
            "lscpu", shell=True, stdout=f, stderr=subprocess.PIPE, text=True
        )

    try:
        total_time_start = time.time()
        in_progress.start_run(args.index)
        main(args)
        in_progress.complete_run(args.index)
        total_time_end = time.time() - total_time_start
        with open(args.experiment_times_file, "a+") as f:
            f.write(str(total_time_end) + "\n")
    except Exception:
        import traceback

        traceback.format_exc()
        with open(results_dir.joinpath("error.txt"), "w") as f:
            f.write("\n" + str(traceback.format_exc()))
            print(traceback.format_exc())

# python experiment_runner.py --model_name mattersim --input_dir /store/nosnap/mlip-eval/uip-data/amcsd_processed_final/all --index 0 --debug
# python experiment_runner.py --model_name orb --input_dir /store/nosnap/mlip-eval/uip-data/amcsd_processed_final/all --index 0 --debug
# python experiment_runner.py --model_name sevennet --input_dir /store/nosnap/mlip-eval/uip-data/amcsd_processed_final/all --index 0 --debug
# python experiment_runner.py --model_name equiformerv2 --input_dir /store/nosnap/mlip-eval/uip-data/amcsd_processed_final/all --index 0 --debug
