import hydra
import os
import numpy as np
import json
import pandas as pd
import re
import logging
import openai
import wandb
import omegaconf
import subprocess
from pathlib import Path
import shutil
import time
import requests
from ml_logger import logger

# relative imports for editor
from utils.misc import *  # type: ignore
from utils.extract_task_code import *  # type: ignore
import plots_plus
from typing import Literal

EUREKA_ROOT_DIR = Path.cwd()
ROOT_DIR = EUREKA_ROOT_DIR / ".."


def generate_samples(cfg, messages, stats):
    openai.api_key = "..."
    vllm_host = "http://0.0.0.0:8000"
    openai.api_base = f"{vllm_host}/v1"

    responses = []

    for s in range(cfg.sample):
        response = None
        for attempt in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model=f"{cfg.model_path}{cfg.model}",
                    messages=messages,
                    n=1,
                )
                break
            except Exception as e:
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()

        responses.extend(response["choices"])  # type: ignore
        stats["prompt_tokens"].append(response["usage"]["prompt_tokens"])  # type: ignore
        stats["completion_tokens"].append(response["usage"]["completion_tokens"])  # type: ignore
        stats["total_tokens"].append(response["usage"]["total_tokens"])  # type: ignore

    # split thinking and non thinking content
    for i, response in enumerate(responses):
        text = response["message"]["content"]
        thinking_content = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        response["message"]["thinking"] = (
            thinking_content.group(1) if thinking_content and len(thinking_content.group(1)) else "None"
        )
        response["message"]["answer"] = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL
        )
        if response["message"]["thinking"] != "None":
            res = requests.post(
                f"{vllm_host}/tokenize",
                headers={"Content-Type": "application/json"},
                json={
                    "model": f"{cfg.model_path}{cfg.model}",
                    "prompt": response["message"]["answer"],
                },
            )
            stats["thinking_tokens"].append(
                stats["completion_tokens"][i] - res.json()["count"]
            )
            stats["answer_tokens"].append(res.json()["count"])
        else:
            stats["thinking_tokens"].append(0)
            stats["answer_tokens"].append(stats["completion_tokens"][i])
    return responses, stats


def add_failure_values(stats):
    stats["execution"].append(0)
    stats["fitness_score_max"].append(0)
    stats["fitness_score_mean"].append(0)
    stats["fitness_score_min"].append(0)
    stats["episode_length"].append(0)
    # stats["prompt_tokens"].append(DUMMY_FAILURE) # correct values added already before evaluation failure
    # stats["completion_tokens"].append(DUMMY_FAILURE)
    # stats["total_tokens"].append(DUMMY_FAILURE)
    stats["reward_total_max"].append(0)
    stats["reward_total_mean"].append(0)
    stats["reward_total_min"].append(0)
    stats["num_reward_functions"].append(0)
    stats["reward_names"].append([])
    # "reward_correlation": [],  # not implemented
    return stats


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {str(EUREKA_ROOT_DIR)}")

    TIMESTAMP = workspace_dir.name
    (workspace_dir / "rewards").mkdir()
    (workspace_dir / "logs").mkdir()
    (workspace_dir / "chats").mkdir()
    (workspace_dir / "graphics").mkdir()

    logging.info(f"Using LLM: {cfg.model_path}{cfg.model}")
    logging.info("Task: " + cfg.env.task)
    logging.info("Task description: " + cfg.env.description)

    maximum_fitness_score = -1
    maximum_sample = -1
    maximum_iteration = -1

    full_stats = pd.DataFrame()
    full_metrics = []

    logger.log_params(Cfg=vars(cfg))
    if cfg.use_wandb:
        config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        run = wandb.init(
            dir=workspace_dir,
            entity="peer222-luh",
            project="master-thesis",
            name=f"{cfg.model}_{TIMESTAMP}",
            group="eureka",
            config=config,  # type: ignore
        )

    env_name = cfg.env.env_name.lower()
    task_rew_file = ROOT_DIR / env_name / cfg.env.reward_template_file
    task_obs_file = EUREKA_ROOT_DIR / "envs" / f"{env_name}.py"
    output_file = ROOT_DIR / env_name / cfg.env.reward_output_file

    task_rew_code_string = file_to_string(task_rew_file)  # type: ignore
    task_obs_code_string = file_to_string(task_obs_file)  # type: ignore
    no_think_flag = ""
    if not cfg.thinking_enabled:
        no_think_flag = " /no_think"

    # Loading all text prompts
    prompt_dir = EUREKA_ROOT_DIR / "prompts"
    initial_system = file_to_string(prompt_dir / "initial_system.txt")  # type: ignore
    code_output_tip = file_to_string(prompt_dir / "code_output_tip.txt")  # type: ignore
    code_feedback = file_to_string(prompt_dir / "code_feedback.txt")  # type: ignore
    initial_user = file_to_string(prompt_dir / "initial_user.txt")  # type: ignore
    reward_signature = file_to_string(prompt_dir / "reward_signatures" / f"{env_name}.txt")  # type: ignore
    policy_feedback = file_to_string(prompt_dir / "policy_feedback.txt")  # type: ignore
    execution_error_feedback = file_to_string(prompt_dir / "execution_error_feedback.txt")  # type: ignore

    initial_system = (
        initial_system.format(task_reward_signature_string=reward_signature)
        + code_output_tip
        + no_think_flag
    )
    initial_user = initial_user.format(
        task_obs_code_string=task_obs_code_string, task_description=cfg.env.description
    )
    messages = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]

    # Eureka generation loop
    for iter in range(cfg.iteration):
        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples")

        stats = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "thinking_tokens": [],
            "answer_tokens": [],
            "total_tokens": [],
            "execution": [],
            "fitness_score_max": [],
            "fitness_score_mean": [],
            "fitness_score_min": [],
            "episode_length": [],
            "reward_total_max": [],
            "reward_total_mean": [],
            "reward_total_min": [],
            "num_reward_functions": [],
            "reward_names": [],
        }
        iteration_metrics = []

        # Get Eureka response
        samples, stats = generate_samples(cfg, messages, stats)

        # Launch all evaluations
        evaluation_runs = []
        for sample_idx, sample in enumerate(samples):
            logging.info(f"Iteration {iter}: Processing Code Run {sample_idx}")

            code_string = parse_generated_reward_functions(sample["message"]["answer"])  # type: ignore
            # Add the Eureka Reward Signature to the environment code
            cur_task_rew_code_string = task_rew_code_string.replace(
                "# INSERT EUREKA REWARD HERE", code_string
            )
            with open(output_file, "w") as file:  # TODO maybe adapt for concurrency
                file.writelines(cur_task_rew_code_string + "\n")

            ### saving messages, llm response and generated reward code
            with open(f"chats/iteration-{iter}_sample-{sample_idx}.md", "w") as file:
                for message in messages:
                    file.write(f"\n\n## {message['role']}:\n\n")
                    file.write(f"{message['content']}\n")
                file.write(f"\n\n ---------- \n## {sample['message']['role']}:\n\n")
                file.write(
                    f'***Thinking:***\n\n{sample["message"]["thinking"]}\n\n***Final Answer:***\n\n{sample["message"]["answer"]}'
                )
            shutil.copy(output_file, f"rewards/iteration-{iter}_sample-{sample_idx}.py")
            ###

            # Find the freest GPU to run GPU-accelerated RL
            # set_freest_gpu()  # TODO not working on luis slurm cluster
            # TODO support parallel executions again

            rl_logpath = f"logs/iteration-{iter}_sample-{sample_idx}.txt"
            run_gpu =  sample_idx % cfg.num_gpus
            with open(rl_logpath, "w") as f:
                # Execute the python file with flags
                command = f"python -u {ROOT_DIR}/{env_name}/{cfg.env.train_script} --iterations {cfg.env.train_iterations} --headless --dr-config off --reward-config eureka --wandb-group eureka/{TIMESTAMP}/{iter}/{sample_idx}"
                command = command.split(" ")
                if not cfg.use_wandb:
                    command.append("--no-wandb")
                logging.info(command)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = run_gpu
                evaluation_runs.append(subprocess.Popen(command, stdout=f, stderr=f, env=env))

            # needed so that rewards are not overridden
            block_until_training(  # type: ignore
                rl_logpath,
                success_keyword=cfg.env.success_keyword,
                failure_keyword=cfg.env.failure_keyword,
                log_status=True,
                iter_num=iter,
                response_id=sample_idx,
            )
            block_until_queue_finished(evaluation_runs, cfg.num_gpus, cfg.processes_per_gpu)  # type: ignore

        # Gather evaluation results and construct reward reflection
        contents = []  # Logs and other feedback for LLM
        for response_id, evaluation_run in enumerate(evaluation_runs):
            evaluation_run.communicate()
            rl_logpath = f"logs/iteration-{iter}_sample-{response_id}.txt"
            try:
                with open(rl_logpath, "r") as f:
                    stdout_str = f.read()
            except:
                # TODO bugfixing by LLM not implemented
                content = execution_error_feedback.format(
                    traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                )
                content += code_output_tip
                contents.append(content)
                add_failure_values(stats)
                logging.error(f"ERROR: Could not open/read {rl_logpath}")
                continue

            content = ""
            traceback_msg = filter_traceback(stdout_str)  # type: ignore

            if traceback_msg == "":
                stats["execution"].append(1)
                run_log = construct_run_log(stdout_str)  # type: ignore
                if run_log is None:
                    logging.warning(
                        f"WARNING: Stopped execution without error message: Skipping Run {response_id}!"
                    )
                    contents.append("Unknown error")
                    add_failure_values(stats)
                    continue

                logged_train_iterations = np.array(run_log["iterations"]).shape[0]
                step_size = max(logged_train_iterations // cfg.feedback_series_size, 1)
                epoch_freq = cfg.env.train_iterations // cfg.feedback_series_size
                logging.info(f"{logged_train_iterations=}; {step_size=}; {epoch_freq=}")

                content += policy_feedback.format(epoch_freq=epoch_freq)

                # Add reward components log to the feedback
                metrics = {}
                reward_names = []
                num_rewards = 0
                for metric in sorted(run_log.keys()):
                    if metric not in ["timesteps", "iterations"]:
                        metric_cur = [
                            "{:.2f}".format(x) for x in run_log[metric][::step_size]
                        ]
                        metric_cur_max = max(run_log[metric])
                        metric_cur_mean = sum(run_log[metric]) / len(run_log[metric])
                        metric_cur_min = min(run_log[metric])

                        metric_name = metric
                        if "fitness_score" == metric:
                            stats["fitness_score_max"].append(metric_cur_max)
                            stats["fitness_score_mean"].append(metric_cur_mean)
                            stats["fitness_score_min"].append(metric_cur_min)
                            metrics["fitness_score"] = run_log[metric]
                            metric_name = "task score"
                        elif "episode_length" == metric:
                            stats["episode_length"].append(metric_cur_max)
                        elif "total" in metric:
                            stats["reward_total_max"].append(metric_cur_max)
                            stats["reward_total_mean"].append(metric_cur_mean)
                            stats["reward_total_min"].append(metric_cur_min)
                            metrics["total"] = run_log[metric]
                        elif "rew" in metric:
                            num_rewards += 1
                            rew_name = metric.split("rew ")[-1]
                            reward_names.append(rew_name)
                            metrics[rew_name] = run_log[metric]
                        elif "loss" in metric:
                            # losses should not be included in llm feedback
                            metrics[metric] = run_log[metric]
                            continue

                        content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f}  \n"

                content += code_feedback
                stats["num_reward_functions"].append(num_rewards)
                stats["reward_names"].append(reward_names)
                iteration_metrics.append(metrics)
            else:
                # Otherwise, provide execution traceback error feedback
                add_failure_values(stats)
                logging.warning(
                    f"WARNING: Failed code execution of {response_id}: {traceback_msg}"
                )
                content += execution_error_feedback.format(traceback_msg=traceback_msg)
            content += code_output_tip
            contents.append(content)

        full_metrics.append(iteration_metrics)
        logging.info(stats)
        stats = pd.DataFrame(stats)
        stats["iteration"] = iter
        stats["version"] = f"{cfg.model}_{TIMESTAMP}"
        full_stats = pd.concat([full_stats, stats])
        full_stats.to_csv("stats.csv")
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(stats["fitness_score_max"])
        best_content = contents[best_sample_idx]

        best_fitness_score = np.max(stats["fitness_score_max"])
        execution_rate = np.sum(stats["execution"]) / cfg.sample

        # Update the best Eureka Output
        if best_fitness_score > maximum_fitness_score:
            maximum_fitness_score = best_fitness_score
            maximum_sample = best_sample_idx
            maximum_iteration = iter

        logging.info(
            f"Iteration {iter}: Max Fitness Score: {best_fitness_score:.2f}, Execution Rate: {execution_rate:.2f}, Max All Time: {maximum_fitness_score:.2f}"
        )
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")

        message_type = "answer"
        if cfg.thinking_enabled and cfg.include_thinking_in_prompt:
            message_type = "content"
        if len(messages) == 2:
            messages += [
                {
                    "role": "assistant",
                    "content": samples[best_sample_idx]["message"][message_type],
                }
            ]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {
                "role": "assistant",
                "content": samples[best_sample_idx]["message"][message_type],
            }
            messages[-1] = {"role": "user", "content": best_content}

    ###
    full_stats["version"] = f"{cfg.model}_{TIMESTAMP}"
    full_stats.to_csv("stats.csv")
    if cfg.use_wandb:
        table = wandb.Table(dataframe=full_stats)
        run.log({"Stats": table})  # type: ignore
    with open("metrics.json", "w") as f:
        json.dump(full_metrics, f)

    ###
    if maximum_fitness_score < 0:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info(
            "Please double check the output iteration-*_sample-*.txt files for repeating errors!"
        )
        exit()
    max_reward_code_path = Path(
        f"rewards/iteration-{maximum_iteration}_sample-{maximum_sample}.py"
    )
    logging.info(
        f"\n Task: {cfg.env.task}, \n Max Training Fitness Score {maximum_fitness_score:.2f} \n Best Reward Code Path: {str(max_reward_code_path)}"
    )

    ### Defaults to best reward configuration
    best_reward = file_to_string(max_reward_code_path)  # type: ignore
    with open(output_file, "w") as file:
        file.writelines(best_reward + "\n")

    ### Get run directory of best-performing policy
    max_reward_log_path = Path("logs") / f"{max_reward_code_path.stem}.txt"
    with open(max_reward_log_path, "r") as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith("Dashboard: "):
            run_dir = line.split(": ")[1].strip()
            run_dir = run_dir.replace(
                "http://app.dash.ml/", f"{ROOT_DIR}/{env_name}/runs/"
            )
            logging.info("Best policy run directory: " + run_dir)

    # graphics
    full_stats["version"] = cfg.model  # plot only model name as version for clarity
    metrics_df = plots_plus.utils.convert_metric_series(
        full_metrics, cfg.env.train_iterations
    )
    plots_plus.eureka.create_plots(cfg.model, full_stats, metrics_df, workspace_dir / "graphics")


if __name__ == "__main__":
    main()
