from functools import lru_cache
import hydra
import sys
import csv
import traceback
import numpy as np
from typing import List, Dict
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
import submitit
import ast

# relative imports for editor
from utils.misc import *  # type: ignore
from utils.extract_task_code import *  # type: ignore
import plots_plus

EUREKA_ROOT_DIR = Path.cwd()
ROOT_DIR = EUREKA_ROOT_DIR / ".."


@lru_cache(maxsize=None)
def load_prompting(cfg):
    env_name = cfg.env.env_name.lower()
    class Prompts:
        # loading task/environment description
        task_obs_file = EUREKA_ROOT_DIR / "envs" / f"{env_name}.py"
        task_obs_code_string = file_to_string(task_obs_file)
        # Loading all text prompts
        prompt_dir = EUREKA_ROOT_DIR / "prompts"
        initial_system = file_to_string(prompt_dir / "initial_system.txt")
        code_output_tip = file_to_string(prompt_dir / "code_output_tip.txt")
        code_feedback = file_to_string(prompt_dir / "code_feedback_evolved.txt")
        initial_user = file_to_string(prompt_dir / "initial_user.txt")
        reward_signature = file_to_string(prompt_dir / "reward_signatures" / f"{env_name}_evolved.txt")
        policy_feedback = file_to_string(prompt_dir / "policy_feedback_evolved.txt")
        # not implemented
        # execution_error_feedback = file_to_string(prompt_dir / "execution_error_feedback.txt")
    return Prompts


def construct_dialog(cfg, database_df: pd.DataFrame) -> List[Dict[str, str]]:
    prompts = load_prompting(cfg)

    no_think_flag = ""
    if not cfg.thinking_enabled:
        no_think_flag = " /no_think"

    initial_system = (
        prompts.initial_system.format(task_reward_signature_string=prompts.reward_signature)
        + prompts.code_output_tip
        + no_think_flag
    )
    initial_user = prompts.initial_user.format(
        task_obs_code_string=prompts.task_obs_code_string, task_description=cfg.env.description
    )
    messages = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]
    if len(database_df):
        topk_database = database_df
        if cfg.topk_sampling:
            topk_database = topk_database.sort_values(by="fitness_score", ascending=False).iloc[:cfg.topk_sampling]
        history_samples = topk_database.sample(cfg.num_examples, weights="fitness_score")
        history_samples: pd.DataFrame = history_samples.sort_values(by="fitness_score", ascending=True)

        for i, (index, history_sample) in enumerate(history_samples.iterrows()):
            prefix = f"Version {i + 1}:\n\n" if cfg.num_examples > 1 else ""
            messages += [
                    {
                        "role": "assistant",
                        "content": prefix + history_sample["generation"],
                    }
                ]
            epoch_freq = cfg.env.train_iterations // cfg.feedback_series_size
            feedback = prompts.policy_feedback.format(version=i + 1, epoch_freq=epoch_freq) + history_sample["training_summarization"]
            messages += [{"role": "user", "content": feedback}]
        instructions = prompts.code_feedback + prompts.code_output_tip
        messages += [{"role": "user", "content": instructions}]
    return messages


def generate_samples(iteration: int, cfg, database_df: pd.DataFrame, stats):
    openai.api_key = "..."
    vllm_host = f"http://0.0.0.0:{cfg.port}"
    openai.api_base = f"{vllm_host}/v1"

    responses = []
    all_messages = []
    num_prev_prompts = len(stats["prompt_tokens"])

    for s in range(cfg.sample):
        messages = construct_dialog(cfg, database_df)
        all_messages.append(messages)
        start_time = time.time()
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
        logging.info(f"Generation {s} took {(time.time() - start_time):.1f} seconds")

    # split thinking and non thinking content
    for i, (response, messages) in enumerate(zip(responses, all_messages)):
        text: str = response["message"]["content"]
        thinking_content = re.search(r"(<think>)?(.*?)</think>", text, flags=re.DOTALL)
        response["message"]["thinking"] = (
            thinking_content.group(2).strip()
            if thinking_content and len(thinking_content.group(2))
            else "None"
        )
        response["message"]["answer"] = text.split("</think>")[-1].strip()
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
                stats["completion_tokens"][i + num_prev_prompts] - res.json()["count"]
            )
            stats["answer_tokens"].append(res.json()["count"])
        else:
            stats["thinking_tokens"].append(0)
            stats["answer_tokens"].append(
                stats["completion_tokens"][i + num_prev_prompts]
            )
        
        ### saving messages, llm response and generated reward code
        with open(f"chats/iteration-{iteration}_sample-{i}.md", "w") as file:
            for message in messages:
                file.write(f"\n\n## {message['role']}:\n\n")
                file.write(f"{message['content']}\n")
            file.write(f"\n\n ---------- \n## {response['message']['role']}:\n\n")
            file.write(
                f'***Thinking:***\n\n{response["message"]["thinking"]}\n\n***Final Answer:***\n\n{response["message"]["answer"]}'
            )
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
    log_dir = workspace_dir / "logs"  # dir to save run logs
    log_dir.mkdir()
    reward_dir = workspace_dir / "rewards"  # dir to save run logs
    reward_dir.mkdir()
    (workspace_dir / "chats").mkdir()
    (workspace_dir / "graphics").mkdir()

    env_name = cfg.env.env_name.lower()
    output_file = ROOT_DIR / env_name / cfg.env.reward_output_file
    task_rew_file = ROOT_DIR / env_name / cfg.env.reward_template_file
    task_rew_code_string = file_to_string(task_rew_file)

    logging.info(f"Using LLM: {cfg.model_path}{cfg.model}")
    logging.info("Task: " + cfg.env.task)
    logging.info("Task description: " + cfg.env.description)

    maximum_fitness_score = -1
    maximum_iteration = -1
    maximum_sample = -1

    full_stats = pd.DataFrame()
    full_metrics = []

    database_df = pd.DataFrame(columns=["iteration", "sample", "fitness_score", "generation", "training_summarization", "reward_code_path"])

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
            group="eureka-evolved",
            config=config,  # type: ignore
        )

    last_complete_iteration = -1
    if cfg.resume:
        # TODO rewrite resume
        raise NotImplementedError("Resume not implemented")

    if cfg.use_submitit:
        submitit_executor = submitit.SlurmExecutor(folder="submitit")
        submitit_executor.update_parameters(
            stderr_to_stdout=True,
            cpus_per_task=4,
            mem="48G",
            partition="tnt",
            gres="gpu:rtx_3090:1",
            job_name="run",
            time="12:00:00",
            additional_parameters={
                "reservation": "tnt"
            }
        )

        def train(train_cfg: Dict):
            try:
                if cfg.env.env_name.lower() == "globe_walking_go2":
                    from globe_walking_go2.scripts.train import train_go2

                    train_go2(**train_cfg)
                elif cfg.env.env_name.lower() == "forward_locomotion_go2":
                    from forward_locomotion_go2.scripts.train import train_mc

                    train_cfg["command_config"] = "off"
                    train_mc(**train_cfg)
                else:
                    raise NotImplementedError(
                        f"Not implemented environment: {cfg.env.env_name.lower()}"
                    )
            except Exception as e:
                print(f"Exception: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()

    # Eureka generation loop
    for iter in range(last_complete_iteration + 1, cfg.iteration):
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
        samples, stats = generate_samples(iter, cfg, database_df, stats)

        # Launch all evaluations
        evaluation_runs = []
        used_gpus = []
        free_eval_gpu: int = 0
        for sample_idx, sample in enumerate(samples):
            logging.info(f"Iteration {iter}: Processing Code Run {sample_idx}")

            code_string = parse_generated_reward_functions(sample["message"]["answer"])  # type: ignore
            # Add the Eureka Reward Signature to the environment code
            cur_task_rew_code_string = task_rew_code_string.replace(
                "# INSERT EUREKA REWARD HERE", code_string
            )

            with open(reward_dir / f"iteration-{iter}_sample-{sample_idx}.py", "w") as file:
                file.writelines(cur_task_rew_code_string + "\n")

            rl_logpath = log_dir / f"iteration-{iter}_sample-{sample_idx}.log"
            if cfg.use_submitit:
                train_cfg = {
                    "reward_struct": cur_task_rew_code_string,
                    "iterations": cfg.env.train_iterations,
                    "reward_config": "eureka",
                    "dr_config": "off",
                    "no_wandb": not cfg.use_run_wandb,
                    # also used as result directory path
                    "wandb_group": f"eureka-evolved/{TIMESTAMP}/{iter}/{sample_idx}",
                    "headless": True,
                    "device": "cuda:0",
                }
                job = submitit_executor.submit(train, train_cfg)  # type: ignore
                evaluation_runs.append(job)
            else:
                # TODO make dynamic loading of reward-struct work here as well
                shutil.copy(
                    reward_dir / f"iteration-{iter}_sample-{sample_idx}.py", output_file
                )
                with open(rl_logpath, "w") as f:
                    # Execute the python file with flags
                    command = f"python -u {ROOT_DIR}/{env_name}/{cfg.env.train_script} --iterations {cfg.env.train_iterations} --headless --dr-config off --reward-config eureka --wandb-group eureka-evolved/{TIMESTAMP}/{iter}/{sample_idx} --device cuda:{free_eval_gpu}"
                    command = command.split(" ")
                    if not cfg.use_run_wandb:
                        command.append("--no-wandb")
                    logging.info(command)
                    evaluation_runs.append(
                        subprocess.Popen(command, stdout=f, stderr=f)
                    )
                    used_gpus.append(free_eval_gpu)
                # needed so that rewards are not overridden
                block_until_training(
                    rl_logpath,
                    log_status=True,
                    iter_num=iter,
                    response_id=sample_idx,
                )
                free_eval_gpu: int = block_until_free_gpu(
                    evaluation_runs, used_gpus, cfg.num_gpus, cfg.processes_per_gpu
                )

        # evaluation of runs
        for response_id, evaluation_run in enumerate(evaluation_runs):
            rl_logpath = log_dir / f"iteration-{iter}_sample-{response_id}.log"
            if cfg.use_submitit:
                try:
                    evaluation_run.result()
                    log = evaluation_run.stdout()
                    with open(rl_logpath, "w") as log_file:
                        log_file.write(log)
                except Exception as e:
                    logging.info(f"Job {iter}-{response_id} failed! \n{e}\n")
            else:
                evaluation_run.communicate()

            with open(rl_logpath, "r") as f:
                stdout_str = f.read()

            traceback_msg = filter_traceback(stdout_str)  # type: ignore
            logging.info(f"{iter}-{response_id} traceback message: ${traceback_msg}$")
            if traceback_msg == "":
                run_log = construct_run_log(stdout_str)  # type: ignore
                if run_log is None:
                    logging.warning(
                        f"WARNING: Stopped execution without error message: Skipping Run {response_id}!"
                    )
                    add_failure_values(stats)
                    iteration_metrics.append({})
                    continue
                stats["execution"].append(1)

                logged_train_iterations = np.array(run_log["iterations"]).shape[0]
                step_size = max(logged_train_iterations // cfg.feedback_series_size, 1)

                # Add reward components log to the feedback
                metrics = {}
                reward_names = []
                num_rewards = 0
                fitness_score = 0
                training_summarization = ""
                for metric in sorted(run_log.keys()):
                    if metric not in ["timesteps", "iterations", "time elapsed/mean", "time iter/mean"]:
                        metric_cur = [
                            "{:.2f}".format(x) for x in run_log[metric][::step_size]
                        ]
                        metric_cur_max = max(run_log[metric])
                        metric_cur_mean = sum(run_log[metric]) / len(run_log[metric])
                        metric_cur_min = min(run_log[metric])

                        metric_name = metric
                        if "fitness_score" == metric:
                            fitness_score = max(metric_cur_max, 0)
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

                        training_summarization += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f}  \n"

                stats["num_reward_functions"].append(num_rewards)
                stats["reward_names"].append(reward_names)
                iteration_metrics.append(metrics)
                logging.info(f"{iter}-{response_id}: {fitness_score=}")

                # add successful run to database
                database_df.loc[iter * cfg.sample + response_id] = [
                    iter,
                    response_id,
                    fitness_score,
                    samples[response_id]["message"]["answer"],
                    training_summarization,
                    reward_dir / f"iteration-{iter}_sample-{response_id}.py"
                ]
                database_df.to_csv("database.csv", quoting=csv.QUOTE_ALL, escapechar='\\')
            else:
                # Otherwise, provide execution traceback error feedback
                add_failure_values(stats)
                iteration_metrics.append({})
                logging.warning(
                    f"WARNING: Failed code execution of {response_id}: {traceback_msg}"
                )

        full_metrics.append(iteration_metrics)
        with open("metrics.json", "w") as f:
            json.dump(full_metrics, f)
        logging.info(stats)
        stats = pd.DataFrame(stats)
        stats["iteration"] = iter
        stats["sample"] = stats.index
        stats["version"] = f"{cfg.model}_{TIMESTAMP}"
        full_stats = pd.concat([full_stats, stats], ignore_index=True)
        full_stats.to_csv("stats.csv", index=False)
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(stats["fitness_score_max"])

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

        if cfg.thinking_enabled and cfg.include_thinking_in_prompt:
            raise NotImplementedError("Including thinking content in conversation not supported")

    ###
    full_stats["version"] = f"{cfg.model}_{TIMESTAMP}"
    full_stats.to_csv("stats.csv")
    if cfg.use_wandb:
        table = wandb.Table(dataframe=full_stats)
        run.log({"Stats": table})  # type: ignore
    with open("metrics.json", "w") as f:
        json.dump(full_metrics, f)

    if cfg.resume:
        logging.info("Resumed training finished!")
        return

    ###
    if maximum_fitness_score < 0:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info(
            "Please double check the output iteration-*_sample-*.log files for repeating errors!"
        )
        exit()
    max_reward_code_path = Path(
        reward_dir / f"iteration-{maximum_iteration}_sample-{maximum_sample}.py"
    )
    logging.info(
        f"\n Task: {cfg.env.task}, \n Max Training Fitness Score {maximum_fitness_score:.2f} \n Best Reward Code Path: {str(max_reward_code_path)}"
    )

    ### Defaults to best reward configuration
    best_reward = file_to_string(max_reward_code_path)  # type: ignore
    with open(output_file, "w") as file:
        file.writelines(best_reward + "\n")

    ### Get run directory of best-performing policy
    max_reward_log_path = Path("logs") / f"{max_reward_code_path.stem}.log"
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
    plots_plus.eureka.create_plots(
        cfg.model, full_stats, metrics_df, workspace_dir / "graphics"
    )


if __name__ == "__main__":
    main()
