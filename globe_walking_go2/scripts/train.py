import os 
import argparse
from pathlib import Path
from typing import Optional


def update_config(Cfg, reward_config, dr_config):
    from globe_walking_go2.go2_gym.envs.go2.go2_config import config_go2

    if reward_config == "eureka":
        Cfg.reward_container_name = "EurekaReward"
    elif reward_config == "eureka_original":
        Cfg.reward_container_name = "EurekaOriginalReward"

    if dr_config == "eureka":
        Cfg.env = Cfg.env_full
        Cfg.sensors = Cfg.sensors_full
        Cfg.terrain = Cfg.terrain_full
        Cfg.domain_rand = Cfg.domain_rand_eureka
        Cfg.sim.physx = Cfg.sim.physx_full
    elif dr_config == "off":
        Cfg.env = Cfg.env_mini
        Cfg.sensors = Cfg.sensors_mini
        Cfg.terrain = Cfg.terrain_mini
        Cfg.domain_rand = Cfg.domain_rand_off
        Cfg.sim.physx = Cfg.sim.physx_mini
    else:
        raise ValueError(f"Invalid dr_config: {dr_config}")

    config_go2(Cfg)  # type: ignore
    return Cfg

def train_go2(iterations, reward_config, dr_config, headless=True, resume_path=None, no_wandb=False, wandb_group=None, wandb_project=None, wandb_entity=None, seed=0, device="cuda:0", num_eval_rollouts: int = 1, reward_struct: Optional[str] = None):
    import isaacgym
    assert isaacgym
    import torch
    import wandb
    from ml_logger import logger
    from plots_plus.train import create_plots

    from globe_walking_go2.go2_gym.envs.base.legged_robot_config import Cfg, set_seed
    from globe_walking_go2.go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
    from globe_walking_go2.go2_gym.envs.wrappers.history_wrapper import HistoryWrapper

    from globe_walking_go2.go2_gym_learn.ppo_cse import Runner
    from globe_walking_go2.go2_gym_learn.ppo_cse.actor_critic import AC_Args
    from globe_walking_go2.go2_gym_learn.ppo_cse.ppo import PPO_Args
    from globe_walking_go2.go2_gym_learn.ppo_cse import RunnerArgs

    from globe_walking_go2.scripts.play import play_go2
    from globe_walking_go2.go2_gym import MINI_GYM_ROOT_DIR

    set_seed(seed, torch_deterministic=False)

    Cfg = update_config(Cfg, reward_config, dr_config)
    if resume_path:
        RunnerArgs.resume = True
        RunnerArgs.load_run = resume_path
        RunnerArgs.resume_checkpoint = os.path.join(RunnerArgs.load_run, "checkpoints", "ac_weights_last.pt")

    # setup logging
    run_dir = Path(f"{MINI_GYM_ROOT_DIR}/../runs").resolve()
    time_now = logger.utcnow(f'{wandb_group}_%Y-%m-%d_%H:%M:%S')
    logger.configure(time_now, root=str(run_dir))
    run_dir = run_dir / str(time_now)
    print(f"{run_dir=}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if not no_wandb:
        wandb.init(
            dir=run_dir,
            project=wandb_project,
            entity=wandb_entity,
            name=str(time_now),
            group=wandb_group,
            config={
                "AC_Args": vars(AC_Args),
                "PPO_Args": vars(PPO_Args),
                "RunnerArgs": vars(RunnerArgs),
                "Cfg": vars(Cfg),
                "HEADLESS": headless,
            },
        )

    print(f"{device=}")
    if headless:
        print("Running headless... disable video recording for training")
        Cfg.env.record_video = False  # type: ignore
    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                    Cfg=vars(Cfg))

    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=Cfg, reward_struct=reward_struct)  # type: ignore
    env = HistoryWrapper(env)
    print("Start training...", flush=True)

    runner = Runner(env, device=device, multi_gpu=Cfg.multi_gpu)
    runner.learn(num_learning_iterations=int(iterations), init_at_random_ep_len=True, eval_freq=100, no_wandb=no_wandb)

    # log video of trained policy rollout
    print(f"Start rollout... Running headless on cpu with video rendering", flush=True)
    # clean environment/gpu
    env.close()
    del env
    torch.cuda.empty_cache()
    # run on cpu to prevent segmentation faults?
    play_go2(run_path=run_dir, dr_config=dr_config, save_video=True, headless=True, num_rollouts=num_eval_rollouts, device="cpu", reward_struct=reward_struct)
    print(f"Rollout complete! Start plotting...", flush=True)

    create_plots(run_dir / "outputs.log", run_dir / "graphics")
    print(f"Successfully completed!", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="peer222-luh")
    parser.add_argument("--wandb-project", type=str, default="master-thesis")
    parser.add_argument("--wandb-group", type=str, default="globe-walking-go2/x")

    parser.add_argument("--dr-config", type=str, required=True, choices=["eureka", "off"])
    # More options need to be added in LeggedRobot as well
    parser.add_argument("--reward-config", type=str, required=True, choices=["eureka", "eureka_original"])

    parser.add_argument("--num-eval-rollouts", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    resume_path = None
    train_go2(iterations=args.iterations, reward_config=args.reward_config, dr_config=args.dr_config, headless=args.headless, resume_path=resume_path, no_wandb=args.no_wandb, wandb_group=args.wandb_group, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity, seed=args.seed, device=args.device, num_eval_rollouts=args.num_eval_rollouts)
