import isaacgym

assert isaacgym
import torch
import pickle as pkl
import shutil
from pathlib import Path
import pandas as pd

from forward_locomotion_go2.go2_gym.envs import *  # type: ignore
from forward_locomotion_go2.go2_gym.envs.base.legged_robot_config import Cfg
from forward_locomotion_go2.go2_gym.envs.go2.go2_config import config_go2
from forward_locomotion_go2.go2_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv
from forward_locomotion_go2.go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from forward_locomotion_go2.go2_gym import MINI_GYM_ROOT_DIR
from forward_locomotion_go2.go2_gym_learn.ppo.actor_critic import ActorCritic

from plots_plus import rollout
from dataclasses import dataclass
from typing import Literal
import tyro
from ml_logger import logger


def load_env(checkpoint_path: Path, headless=False, dr_config="off", save_video=True, device="cuda:0"):
    # Will be overwritten by the loaded config from parameters.pkl
    Cfg.commands = Cfg.commands_original
    Cfg.rewards = Cfg.rewards_eureka
    Cfg.domain_rand = Cfg.domain_rand_off

    # prepare environment
    config_go2(Cfg)

    with open(checkpoint_path / ".." / "parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
    def set_cfg_recursive(cfg, load):
        for key, value in load.items():
            if not hasattr(cfg, key):
                continue
            if key in ["commands_original", "commands_constrained", "domain_rand_original", "domain_rand_eureka", "domain_rand_off", "rewards_original", "rewards_eureka"]:
                # Don't overwrite presets from Cfg
                continue
            if isinstance(value, dict):
                set_cfg_recursive(getattr(cfg, key), value)
            else:
                setattr(cfg, key, value)

    set_cfg_recursive(Cfg, cfg)
    Cfg.commands.command_curriculum = False

    if dr_config == "original":
        Cfg.domain_rand = Cfg.domain_rand_original
    elif dr_config == "eureka":
        Cfg.domain_rand = Cfg.domain_rand_eureka
    elif dr_config == "off":
        Cfg.domain_rand = Cfg.domain_rand_off
    
    Cfg.env.record_video = save_video

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    if device == "cpu":
        Cfg.sim.use_gpu_pipeline = False
        Cfg.env.record_video = True
    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=Cfg)  # type: ignore
    env = HistoryWrapper(env)

    actor_critic = ActorCritic(
        num_obs=Cfg.env.num_observations,
        num_privileged_obs=Cfg.env.num_privileged_obs,
        num_obs_history=Cfg.env.num_observations * \
                        Cfg.env.num_observation_history,
        num_actions=Cfg.env.num_actions)

    weights = torch.load(checkpoint_path / "ac_weights_final.pt", map_location="cpu")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    policy = actor_critic.act_inference

    return env, policy


def play_go2(
    run_path: Path,
    headless=True,
    dr_config="off",
    save_video=False,
    num_rollouts: int = 1,
    device: Literal["cpu", "cuda:0", "cuda:1"] = "cuda:0"
):
    print("Start play", flush=True)
    checkpoint_path = run_path / "checkpoints"
    env, policy = load_env(checkpoint_path, headless=headless, dr_config=dr_config, device=device)
    print("Loaded env and policy", flush=True)

    all_stats_df = pd.DataFrame()
    for rollout_index in range(num_rollouts):
        if save_video:
            import imageio
            video_dir_path = checkpoint_path / "../videos"
            video_dir_path.mkdir(exist_ok=True)
            mp4_writer = imageio.get_writer(video_dir_path / f"final-{rollout_index}.mp4", fps=50)
        obs = env.reset()

        episode_length = 0
        episode_reward = 0
        time_steps = []
        accumulated_rewards = []
        positions = []
        linear_velocities = []
        angular_velocities = []
        global_linear_velocities = []
        global_angular_velocities = []

        foot_contact_forces = []
        joint_positions = []
        torques = []
        out_of_limits = []

        robot_idx = 0
        done = torch.tensor(0)
        while True:
            if save_video:
                img = env.render(mode="rgb_array")
                mp4_writer.append_data(img)  # type: ignore
            with torch.no_grad():
                actions = policy(obs)
            time_steps.append(episode_length * env.dt)
            accumulated_rewards.append(
                {f"rew_{k}": v.item() for k, v in env.episode_sums.items()}
            )
            positions.append(env.root_states[robot_idx, 0:3].tolist())
            linear_velocities.append(env.base_lin_vel[robot_idx, :].tolist())
            angular_velocities.append(env.base_ang_vel[robot_idx, :].tolist())
            global_linear_velocities.append(env.root_states[robot_idx, 7:10].tolist())
            global_angular_velocities.append(env.root_states[robot_idx, 10:13].tolist())
            foot_contact_forces.append(
                torch.norm(
                    env.contact_forces[robot_idx, env.feet_indices, :], dim=-1
                ).tolist()
            )
            joint_positions.append(
                (env.dof_pos[robot_idx, :] * 57.2958).tolist()
            )  # radiant to degrees
            torques.append(env.torques[robot_idx, :].tolist())
            out_of_limits.append(
                (
                    -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.0)
                    + (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.0)
                )
                .count_nonzero()
                .item()
            )

            if done.any():  # type: ignore
                break
            obs, rew, done, info = env.step(actions)
            episode_reward += rew
            episode_length += 1

        if save_video:
            mp4_writer.close()  # type: ignore

            # rounding performed to reduce file size
            time_steps_df = pd.DataFrame(time_steps, columns=["time_(s)"]).round(2)
            accumulated_rewards_df = pd.DataFrame(accumulated_rewards).round(2)
            positions_df = pd.DataFrame(positions, columns=["x", "y", "z"]).round(2)
            linear_velocities_df = pd.DataFrame(
                linear_velocities, columns=["linear_x", "linear_y", "linear_z"]
            ).round(2)
            angular_velocities_df = pd.DataFrame(
                angular_velocities, columns=["angular_x", "angular_y", "angular_z"]
            ).round(2)
            global_linear_velocities_df = pd.DataFrame(
                global_angular_velocities,
                columns=["global_linear_x", "global_linear_y", "global_linear_z"],
            ).round(2)
            global_angular_velocities_df = pd.DataFrame(
                global_angular_velocities,
                columns=["global_angular_x", "global_angular_y", "global_angular_z"],
            ).round(2)
            foot_contact_forces_df = pd.DataFrame(
                foot_contact_forces,
                columns=["front_left", "front_right", "rear_left", "rear_right"],
            ).round(1)
            joint_positions_df = pd.DataFrame(
                joint_positions, columns=[f"position_{n}" for n in env.dof_names]
            ).round(1)
            torques_df = pd.DataFrame(
                torques, columns=[f"torque_{n}" for n in env.dof_names]
            ).round(1)
            out_of_limits_df = pd.DataFrame(out_of_limits, columns=["out_of_limits"])

            stats_df = pd.concat(
                [
                    time_steps_df,
                    accumulated_rewards_df,
                    positions_df,
                    linear_velocities_df,
                    angular_velocities_df,
                    global_linear_velocities_df,
                    global_angular_velocities_df,
                    foot_contact_forces_df,
                    joint_positions_df,
                    torques_df,
                    out_of_limits_df,
                ],
                axis=1,
            )
            stats_df["rollout"] = rollout_index
            # save only every second data point for reduced file size
            stats_df = stats_df.iloc[::2]
            all_stats_df = pd.concat([all_stats_df, stats_df])

    if save_video:
        all_stats_df.to_csv(checkpoint_path / ".." / "rollout_stats.csv")
        rollout.create_plots(
            all_stats_df, checkpoint_path / ".." / "graphics" / "rollouts", env="forward_locomotion_go2"
        )


if __name__ == "__main__":

    @dataclass
    class Args:
        run: Path
        """run directory from which checkpoints are loaded"""
        dr_config: Literal["mini", "full", "eureka", "off", "load"]
        """Domain randomization config"""
        load_reward: bool = False
        """Load reward file from associated eureka run"""
        num_rollouts: int = 1
        """Number of rollouts that are performed"""
        headless: bool = False
        """Play in headless mode"""
        no_video: bool = False
        """If set, no video is recorded"""
        device: Literal["cpu", "cuda:0", "cuda:1"] = "cuda:0"
        """Device that is used for simulation and policy"""

    args = tyro.cli(Args)

    if args.load_reward:
        iteration_idx = args.run.parent.stem
        sample_idx = args.run.stem[0]
        reward_path = (
            args.run.parents[1]
            / "rewards"
            / f"iteration-{iteration_idx}_sample-{sample_idx}.py"
        )
        shutil.copyfile(
            reward_path,
            Path(MINI_GYM_ROOT_DIR) / "go2_gym" / "rewards" / "eureka_reward.py",
        )
        print(f"{reward_path=}")

    play_go2(
        run_path=args.run,
        dr_config=args.dr_config,
        num_rollouts=args.num_rollouts,
        headless=args.headless,
        save_video=not args.no_video,
        device=args.device,
    )
