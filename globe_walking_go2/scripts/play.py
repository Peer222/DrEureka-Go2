import isaacgym
assert isaacgym

import torch
import pickle as pkl
import yaml
import shutil
from pathlib import Path
import pandas as pd

from globe_walking_go2.go2_gym.envs import *  # type: ignore
from globe_walking_go2.go2_gym.envs.base.legged_robot_config import Cfg, set_seed
from globe_walking_go2.go2_gym.envs.go2.go2_config import config_go2
from globe_walking_go2.go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from globe_walking_go2.go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from globe_walking_go2.go2_gym import MINI_GYM_ROOT_DIR

from plots_plus import rollout
from typing import Literal, Optional


def load_policy(checkpoint_path: Path):
    body = torch.jit.load(checkpoint_path / "body_latest.jit", map_location="cpu")  # type: ignore
    adaptation_module = torch.jit.load(checkpoint_path / "adaptation_module_latest.jit", map_location="cpu")  # type: ignore

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        action = body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def load_env(checkpoint_path: Path, headless=False, dr_config="off", save_video=True, device="cuda:0", reward_struct: Optional[str] = None):
    # Will be overwritten by the loaded config from parameters.pkl
    Cfg.env = Cfg.env_mini  # type: ignore
    Cfg.sensors = Cfg.sensors_mini
    Cfg.terrain = Cfg.terrain_mini  # type: ignore
    Cfg.domain_rand = Cfg.domain_rand_off  # type: ignore
    Cfg.sim.physx = Cfg.sim.physx_mini  # type: ignore

    config_go2(Cfg)  # type: ignore
    if (checkpoint_path / "config.yaml").exists():
        with open(checkpoint_path / "config.yaml", "rb") as file:
            cfg = yaml.safe_load(file)
            cfg = cfg["Cfg"]
    elif (checkpoint_path / "../parameters.pkl").exists():
        with open(checkpoint_path / "../parameters.pkl", "rb") as file:
            pkl_cfg = pkl.load(file)
            cfg = pkl_cfg["Cfg"]
    else:
        raise Exception(
            f"Missing {checkpoint_path / 'config.yaml'} or {checkpoint_path / '../parameters.pkl'}"
        )

    def set_cfg_recursive(cfg, load):
        for key, value in load.items():
            if not hasattr(cfg, key):
                continue
            if dr_config != "load" and key in [
                "env_mini",
                "env_full",
                "sensors_mini",
                "sensors_full",
                "terrain_mini",
                "terrain_full",
                "domain_rand_mini",
                "domain_rand_full",
                "domain_rand_eureka",
                "physx_mini",
                "physx_full",
            ]:
                # Don't overwrite presets from Cfg
                continue
            if key in ["pos", "ball_init_pos"]:
                # Backwards compatibility
                continue
            if isinstance(value, dict):
                set_cfg_recursive(getattr(cfg, key), value)
            else:
                # if value != getattr(cfg, key):
                #     print(f"Overwriting {key} from {getattr(cfg, key)} to {value}")
                setattr(cfg, key, value)

    set_cfg_recursive(Cfg, cfg)
    Cfg.multi_gpu = False

    if dr_config == "eureka":
        Cfg.domain_rand = Cfg.domain_rand_eureka  # type: ignore
    elif dr_config == "off":
        Cfg.domain_rand = Cfg.domain_rand_off  # type: ignore
    elif dr_config == "load":
        pass  # Load from the loaded config
    else:
        raise ValueError("Invalid domain randomization configuration")
    Cfg.domain_rand.randomize = False

    Cfg.env.record_video = save_video
    if __name__ == "__main__":
        print("Recording in HD")
        Cfg.env.recording_width_px = 1080
        Cfg.env.recording_height_px = 720

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.num_border_boxes = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    # Cfg.control.control_type = "actuator_net"

    # The following are a series of tests to verify that DR is working as expected
    if False:
        # For visualizing multiple envs
        Cfg.env.num_envs = 3
        Cfg.terrain.center_robots = False
    if False:
        # Put quadruped on ground, get rid of ball
        Cfg.domain_rand.ball_radius_range = [0.0, 0.0]
    if False:
        # Put the ball somewhere fall away
        Cfg.ball.init_pos_range = [100.0, 100.0, 0.2]
    if False:
        # Drop quadruped and ball (to test restitution)
        Cfg.ball.ball_init_pos = [1.0, 1.0, 1.0]
        Cfg.init_state.pos[-1] = 10.0

    # Extreme values for testing
    if False:
        print("> Friction Test: quadruped should slip off easily")
        Cfg.domain_rand.robot_friction_range = [0.0, 0.0]
        Cfg.domain_rand.ball_friction_range = [0.0, 0.0]
    if False:
        print("> Mass test: ball should be immovable")
        Cfg.domain_rand.ball_mass_range = [1000.0, 1000.0]
        Cfg.domain_rand.terrain_tile_roughness_range = [
            0.0,
            0.0,
        ]  # Disable terrain so balls don't move due to gravity
        Cfg.domain_rand.ball_push_vel_range = [0.0, 0.0]
        Cfg.domain_rand.gravity_range = [0.0, 0.0]
    if False:
        print(
            "> Radius test: balls should be big and vary in size, quadrupeds should spawn perfectly on top"
        )
        Cfg.domain_rand.ball_radius_range = [0.0, 2.0]
    if False:
        print("> Restitution test 1: UNSTABLE IN PUBLIC ISAACGYM")
        # To make the effect clear, set Cfg.ball.ball_init_pos = [1.0, 1.0, 1.0] as well
        Cfg.domain_rand.robot_restitution_range = [10.0, 10.0]
        Cfg.domain_rand.ball_restitution_range = [1.0, 1.0]
        Cfg.domain_rand.ball_compliance_range = [0.0, 0.0]
        Cfg.domain_rand.ball_drag_range = [0.0, 0.0]
        Cfg.domain_rand.ball_push_vel_range = [0.0, 0.0]
        Cfg.domain_rand.gravity_range = [0.0, 0.0]
        Cfg.domain_rand.terrain_tile_roughness_range = [0.0, 0.0]

    if False:
        print("> Restitution test 2: UNSTABLE IN PUBLIC ISAACGYM")
        ball_restitution_range = [0.0, 0.0]
        Cfg.domain_rand.terrain_ground_restitution_range = [0.0, 0.0]
    if False:
        print("> Compliance test: NOT IMPLEMENTED IN PUBLIC ISAACGYM")
        Cfg.domain_rand.ball_compliance_range = [10.0, 10.0]
    if False:
        print("> Drag test: ball should move less")
        Cfg.domain_rand.ball_drag_range = [500.0, 500.0]
    if False:
        print("> Push test: quadruped and ball should get pushed around violently")
        Cfg.domain_rand.push_robot_interval_s = 1
        Cfg.domain_rand.robot_push_vel_range = [10.0, 10.0]
        Cfg.domain_rand.push_ball_interval_s = 1
        Cfg.domain_rand.ball_push_vel_range = [10.0, 10.0]
    if False:
        print("> Gravity test: quadrupeds and balls should shift around")
        Cfg.domain_rand.gravity_range = [-3.0, 3.0]
        Cfg.domain_rand.gravity_rand_interval_s = 1
        Cfg.terrain.x_init_range = 0.0
        Cfg.terrain.y_init_range = 0.0
    if False:
        print(
            "> Payload test: quadruped leg should be more bent, unable to support itself"
        )
        Cfg.domain_rand.robot_payload_mass_range = [10.0, 10.0]
    if False:
        print("> CoM test: quadruped should tilt to one side")
        Cfg.domain_rand.robot_com_displacement_range = [0.5, 0.5]
        Cfg.terrain.x_init_range = 0.0
        Cfg.terrain.y_init_range = 0.0
    if False:
        print("> Inertia test: the ball should be harder to rotate")
        Cfg.domain_rand.ball_inertia_multiplier_range = [1000.0, 1000.0]
    if False:
        print("> Spring coefficient test: robot's feet should bounce off the ball")
        Cfg.domain_rand.ball_spring_coefficient_range = [0.7, 0.7]

    if device == "cpu":
        Cfg.sim.use_gpu_pipeline = False

    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=Cfg, reward_struct=reward_struct)  # type: ignore
    env = HistoryWrapper(env)

    policy = load_policy(checkpoint_path)
    return env, policy


def play_go2(
    run_path: Path,
    headless=True,
    dr_config="off",
    save_video=False,
    num_rollouts: int = 1,
    device: Literal["cpu", "cuda:0", "cuda:1"] = "cuda:0",
    reward_struct: Optional[str] = None,
    seed: int = 0,
    file_prefix: str = "",
):
    set_seed(seed)
    print("Start play", flush=True)
    checkpoint_path = run_path / "checkpoints"
    env, policy = load_env(checkpoint_path, headless=headless, dr_config=dr_config, save_video=save_video, device=device, reward_struct=reward_struct)
    print("Loaded env and policy", flush=True)

    all_stats_df = pd.DataFrame()
    for rollout_index in range(num_rollouts):
        if save_video:
            import imageio
            video_dir_path = checkpoint_path / "../videos"
            video_dir_path.mkdir(exist_ok=True)
            mp4_writer = imageio.get_writer(video_dir_path / f"{file_prefix}final-{rollout_index}.mp4", fps=int(1 / env.dt))
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

        robot_idx = env.robot_actor_idxs.item()
        done = torch.tensor(0)
        while True:
            if done.any():  # type: ignore
                break
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
                global_linear_velocities,
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
        all_stats_df.to_csv(checkpoint_path / ".." / f"{file_prefix}rollout_stats.csv")
        try:
            rollout.create_plots(
                all_stats_df, checkpoint_path / ".." / "graphics" / f"{file_prefix}rollouts", env="globe_walking_go2"
            )
        except Exception as e:
            print(f"A bug occured during plotting: {e}")


if __name__ == "__main__":
    from dataclasses import dataclass
    import tyro

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
        device: Literal["cpu", "cuda:0", "cuda:1"] = "cpu"
        """Computation device [cpu, cuda:0]"""
        seed: int = 0
        """Seed"""
        file_prefix: str = ""
        """Prefix that is added to video and graphics file/dir names"""

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
        seed=args.seed,
        file_prefix=args.file_prefix,
    )
