from typing import List, Dict
import subprocess
import logging
import time
from pathlib import Path
import cv2

from utils.extract_task_code import file_to_string  # type: ignore



def set_seed(seed):
    # From isaacgymenvs.utils.utils
    import os
    import random
    import numpy as np
    import torch

    """ set seed across modules """
    if seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def filter_traceback(s: str):
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
        if line.startswith("Exception"):
            return "\n" + line
    return ""  # Return an empty string if no Traceback is found


def block_until_training(
    rl_filepath: Path,
    log_status=False,
    iter_num=-1,
    response_id=-1,
    check_frequency: int = 20,
):
    # Ensure that the RL training has started before moving on
    startup_time_needed = 0
    while startup_time_needed < 30 * 60:
        rl_log = file_to_string(rl_filepath)
        if "running" in rl_log or "Traceback" in rl_log or "Exception" in rl_log:
            if log_status and "running" in rl_log:
                logging.info(
                    f"Iteration {iter_num}: Code Run {response_id} successfully training!"
                )
            if log_status and ("Traceback" in rl_log or "Exception" in rl_log):
                logging.info(
                    f"Iteration {iter_num}: Code Run {response_id} execution error!"
                )
            break
        time.sleep(check_frequency)
        startup_time_needed += check_frequency
    logging.info(f"Startup time needed: {(startup_time_needed / 60):.2f}")


def block_until_free_gpu(
    processes: List[subprocess.Popen],
    used_gpus: List[int],
    num_gpus: int,
    processes_per_gpu: int,
    check_frequency: int = 60,
) -> int:
    free_gpu = -1
    while True:
        queues = {i: 0 for i in range(num_gpus)}
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


def prepare_video_message(frame_dir: Path, fps: int) -> List[Dict[str, str]]:
    frames = frame_dir.glob("*.jpg")
    video_message = []
    i = -1
    for i, frame in enumerate(frames):
        video_message.append({"type": "text", "text": f"<{i/fps:.2f} seconds>"})
        video_message.append(
            {"type": "image_url", "image_url": {"url": f"file://{str(frame.absolute())}"}}
        )
    logging.info(f"Number of frames: {i+1}")
    return video_message


def extract_frames(video_path: Path, frame_dir: Path, fps: int, max_video_length: int):
    frame_dir.mkdir(exist_ok=True)
    old_frames = frame_dir.glob("*.jpg")
    for frame in old_frames:
        frame.unlink()

    video = cv2.VideoCapture(video_path)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        video.release()
        cv2.destroyAllWindows()
        logging.warning(f"Video {video_path} has 0 fps! Maybe it does not exist")
        return
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / original_fps
    max_frames = (max_video_length / duration) * frame_count
    step = original_fps // fps
    logging.info(f"Final Video: {original_fps=}, {fps=}, {step=}")
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if i % step == 0 and frame_count - i < max_frames:
            cv2.imwrite(frame_dir / f"{video_path.stem}-{i:04d}.jpg", frame)
        i += 1

    video.release()
    cv2.destroyAllWindows()
