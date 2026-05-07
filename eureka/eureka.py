import hydra
import sys
import traceback
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
import submitit
import ast

# relative imports for editor
from utils.misc import *  # type: ignore
from utils.extract_task_code import *  # type: ignore
import plots_plus

EUREKA_ROOT_DIR = Path.cwd()
ROOT_DIR = EUREKA_ROOT_DIR / ".."


def generate_samples(cfg, messages, stats):
    openai.api_key = "..."
    vllm_host = f"http://0.0.0.0:{cfg.port}"
    openai.api_base = f"{vllm_host}/v1"

    responses = []
    num_prev_prompts = len(stats["prompt_tokens"])

    custom_params = {}
    if cfg.use_custom_params:
        custom_params = {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "presence_penalty": cfg.presence_penalty,
            "extra_body": {
                "top_k": cfg.top_k,
                "repetition_penalty": cfg.repetition_penalty,
                # "chat_template_kwargs": {"enable_thinking": cfg.thinking_enabled},
            },
        }

    for s in range(cfg.sample):
        start_time = time.time()
        response = None
        for attempt in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model=f"{cfg.model_path}{cfg.model}",
                    messages=messages,
                    n=1,
                    max_tokens=cfg.max_completion_tokens,
                    **custom_params,
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
    for i, response in enumerate(responses):
        text: str = response["message"]["content"]
        thinking_content = re.search(r"(<think>)?(.*?)</think>", text, flags=re.DOTALL)
        response["message"]["thinking"] = (
            thinking_content.group(2).strip()
            if thinking_content and len(thinking_content.group(2).strip())
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
    return responses, stats


def add_failure_values(stats):
    stats["execution"].append(0)
    stats["fitness_score_last"].append(0)
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
    set_seed(cfg.seed)
    logging.info(f"Seed: {cfg.seed}")
    workspace_dir = Path.cwd()
    log_dir = Path("logs")  # dir to save run logs
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
    maximum_iteration = -1
    maximum_sample = -1

    full_stats = pd.DataFrame()
    full_metrics = []
    ancestors = ()

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

    last_complete_iteration = -1
    if cfg.resume:
        resumed_base_dir = Path(cfg.stats_file).parent
        logging.info(f"Resume from stats file: {cfg.stats_file}")
        full_stats = pd.read_csv(cfg.stats_file)
        if "Unnamed: 0" in full_stats.columns:
            full_stats.drop("Unnamed: 0", axis=1, inplace=True)
        full_stats["reward_names"] = full_stats["reward_names"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        last_complete_iteration = len(full_stats) // cfg.sample - 1
        logging.info(f"Last complete iteration: {last_complete_iteration}")
        best_idx = full_stats[
            full_stats["iteration"] <= last_complete_iteration  # type: ignore
        ].idxmax(numeric_only=True)[cfg.optimization_target]
        logging.info(f"Best Index: {best_idx}")
        maximum_fitness_score = full_stats.iloc[best_idx][cfg.optimization_target]
        maximum_iteration = full_stats.iloc[best_idx]["iteration"]
        maximum_sample = full_stats.iloc[best_idx]["sample"]

        best_current_idx = full_stats[
            full_stats["iteration"] == last_complete_iteration  # type: ignore
        ].idxmax(numeric_only=True)[cfg.optimization_target]
        logging.info(f"Best Index of last iteration: {best_current_idx}")
        best_current_sample = full_stats.iloc[best_current_idx]["sample"]
        ancestors = ([last_complete_iteration, best_current_sample])
        if last_complete_iteration >= 0:
            with open(
                resumed_base_dir
                / "chats"
                / f"iteration-{last_complete_iteration}_sample-{best_current_sample}.md",
                "r",
            ) as f:
                text = f.read()
            chat_message = re.search(r"\n\n\*\*\*Final Answer:\*\*\*\n\n(.*)", text, flags=re.DOTALL)
            if chat_message:
                llm_reward_generation = chat_message.group(1).strip()
                logging.info(f"{llm_reward_generation=}")
                messages.append(
                    {"role": "assistant", "content": llm_reward_generation}
                )
                with open(resumed_base_dir / "logs" / f"iteration-{last_complete_iteration}_sample-{best_current_sample}.log", "r") as f:
                    run_log = construct_run_log(f.read())  # type: ignore
                if run_log is None:
                    raise ValueError(f"Could not parse run log on resume!")

                logged_train_iterations = np.array(run_log["iterations"]).shape[0]
                step_size = max(logged_train_iterations // cfg.feedback_series_size, 1)
                epoch_freq = cfg.env.train_iterations // cfg.feedback_series_size

                reward_reflection = policy_feedback.format(epoch_freq=epoch_freq)
                reward_reflection += construct_numeric_feedback(run_log, step_size)
                reward_reflection += code_feedback

                messages.append({"role": "user", "content": reward_reflection + code_output_tip})
            else:
                raise ValueError("Failed to parse chat message for resume")
        # load metrics for completeness and plot generation
        if (resumed_base_dir / "metrics.json").exists():
            with open(resumed_base_dir / "metrics.json", "r") as f:
                full_metrics = json.load(f)

    if cfg.use_submitit:
        submitit_executor = submitit.SlurmExecutor(folder="submitit")
        submitit_executor.update_parameters(
            stderr_to_stdout=True,
            cpus_per_task=4,
            mem="48G",
            partition="tnt",
            gres="gpu:rtx_3090:1",
            job_name="run",
            time="4:00:00",
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
            "fitness_score_last": [],
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
        used_gpus = []
        free_eval_gpu: int = 0
        for sample_idx, sample in enumerate(samples):
            logging.info(f"Iteration {iter}: Processing Code Run {sample_idx}")

            code_string = parse_generated_reward_functions(sample["message"]["answer"])  # type: ignore
            # Add the Eureka Reward Signature to the environment code
            cur_task_rew_code_string = task_rew_code_string.replace(
                "# INSERT EUREKA REWARD HERE", code_string
            )
            with open(f"rewards/iteration-{iter}_sample-{sample_idx}.py", "w") as file:
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

            rl_logpath = log_dir / f"iteration-{iter}_sample-{sample_idx}.log"
            if cfg.use_submitit:
                train_cfg = {
                    "reward_struct": cur_task_rew_code_string,
                    "iterations": cfg.env.train_iterations,
                    "reward_config": "eureka",
                    "dr_config": "off",
                    "no_wandb": not cfg.use_run_wandb,
                    # also used as result directory path
                    "wandb_group": f"eureka/{TIMESTAMP}/{iter}/{sample_idx}",
                    "headless": True,
                    "device": "cuda:0",
                    "seed": cfg.seed,
                }
                job = submitit_executor.submit(train, train_cfg)  # type: ignore
                evaluation_runs.append(job)
            else:
                # TODO make dynamic loading of reward-struct work here as well
                shutil.copy(
                    f"rewards/iteration-{iter}_sample-{sample_idx}.py", output_file
                )
                with open(rl_logpath, "w") as f:
                    # Execute the python file with flags
                    command = f"python -u {ROOT_DIR}/{env_name}/{cfg.env.train_script} --iterations {cfg.env.train_iterations} --headless --dr-config off --reward-config eureka --wandb-group eureka/{TIMESTAMP}/{iter}/{sample_idx} --device cuda:{free_eval_gpu} --seed {cfg.seed}"
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

        # Gather evaluation results and construct reward reflection
        contents = []  # Logs and other feedback for LLM
        for response_id, evaluation_run in enumerate(evaluation_runs):
            rl_logpath = log_dir / f"iteration-{iter}_sample-{response_id}.log"
            if cfg.use_submitit:
                try:
                    evaluation_run.result()
                    log = evaluation_run.stdout()
                except Exception as e:
                    logging.info(f"Job {iter}-{response_id} failed! \n{e}\n")
                    log = f"Exception: Job {iter}-{response_id} failed! \n{e}\n"
                with open(rl_logpath, "w") as log_file:
                    log_file.write(log)
            else:
                evaluation_run.communicate()

            with open(rl_logpath, "r") as f:
                stdout_str = f.read()

            content = ""
            traceback_msg = filter_traceback(stdout_str)  # type: ignore

            if traceback_msg == "":
                run_log = construct_run_log(stdout_str)  # type: ignore
                if run_log is None:
                    logging.warning(
                        f"WARNING: Stopped execution without error message: Skipping Run {response_id}!"
                    )
                    contents.append("Unknown error")
                    add_failure_values(stats)
                    iteration_metrics.append({})
                    continue
                stats["execution"].append(1)

                logged_train_iterations = np.array(run_log["iterations"]).shape[0]
                step_size = max(logged_train_iterations // cfg.feedback_series_size, 1)
                epoch_freq = cfg.env.train_iterations // cfg.feedback_series_size

                content += policy_feedback.format(epoch_freq=epoch_freq)

                # Add reward components log to the feedback
                metrics = {}
                reward_names = []
                num_rewards = 0
                fitness_score = 0
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
                            fitness_score = max(metric_cur_max, 0)
                            stats["fitness_score_last"].append(run_log[metric][-1])
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
                logging.info(f"{iter}-{response_id}: {fitness_score=}")
            else:
                # Otherwise, provide execution traceback error feedback
                add_failure_values(stats)
                iteration_metrics.append({})
                logging.warning(
                    f"WARNING: Failed code execution of {response_id}: {traceback_msg}"
                )
                content += execution_error_feedback.format(traceback_msg=traceback_msg)
            content += code_output_tip
            contents.append(content)

        full_metrics.append(iteration_metrics)
        with open("metrics.json", "w") as f:
            json.dump(full_metrics, f)
        logging.info(stats)
        stats = pd.DataFrame(stats)
        stats["iteration"] = iter
        stats["sample"] = stats.index
        stats["version"] = f"{cfg.model}_{TIMESTAMP}"
        stats["seed"] = cfg.seed
        stats["ancestors"] = [ancestors] * len(stats)
        full_stats = pd.concat([full_stats, stats], ignore_index=True)
        full_stats.to_csv("stats.csv", index=False)
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(stats["fitness_score_max"])
        best_content = contents[best_sample_idx]
        ancestors = ([iter, best_sample_idx])

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
            "Please double check the output iteration-*_sample-*.log files for repeating errors!"
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
