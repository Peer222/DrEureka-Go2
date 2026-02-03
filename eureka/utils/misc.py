from typing import List
import subprocess
import logging
import time

from utils.extract_task_code import file_to_string  # type: ignore


def filter_traceback(s):
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
    return ""  # Return an empty string if no Traceback is found


def block_until_training(
    rl_filepath,
    log_status=False,
    iter_num=-1,
    response_id=-1,
):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "running" in rl_log or "Traceback" in rl_log:
            if log_status and "running" in rl_log:
                logging.info(
                    f"Iteration {iter_num}: Code Run {response_id} successfully training!"
                )
            if log_status and "Traceback" in rl_log:
                logging.info(
                    f"Iteration {iter_num}: Code Run {response_id} execution error!"
                )
            break


def block_until_free_gpu(
    processes: List[subprocess.Popen],
    used_gpus: List[int],
    num_gpus: int,
    processes_per_gpu: int,
    check_frequency: int = 60,
) -> int:
    queues = {i: 0 for i in range(num_gpus)}
    free_gpu = -1
    while True:
        for gpu_idx, p in zip(used_gpus, processes):
            if p.poll() == None:
                queues[gpu_idx] += 1

        for gpu_idx, num_processes in queues.items():
            if num_processes < processes_per_gpu:
                free_gpu = gpu_idx
                break
        if free_gpu >= 0:
            logging.info(f"{queues}: Free GPU {free_gpu} -> Start next evaluation...")
            break
        time.sleep(check_frequency)
    return free_gpu


def construct_run_log(stdout_str):
    run_log = {}
    lines: List[str] = stdout_str.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("│") and line.endswith("│"):
            line = line[1:-1].split("│")
            key, val = line[0].strip(), line[1].strip()
            if key == "train/episode/rew success/mean":
                key = "fitness_score"
            elif key == "timesteps" or key == "iterations":
                key = key
            elif "train/episode/rew" in key:
                key = key.split("/")[2]
            elif key == "train/episode/episode length/mean":
                key = "episode_length"
            elif "loss" in key:
                key = key.split("/")[0].split("mean ")[-1]

            run_log[key] = run_log.get(key, []) + [float(val)]

    if "fitness_score" not in run_log.keys():
        logging.warning("'fitness_score' is missing in run log!")
        return None

    return run_log
