import os 
import argparse


def train_go2(iterations, dr_config, headless=True, resume_path=None, no_wandb=False, wandb_group=None, wandb_project=None, wandb_entity=None, seed=0, device="cuda:0"):
    import isaacgym
    assert isaacgym
    import torch
    import wandb

    from globe_walking_go2.go2_gym.envs.base.legged_robot_config import Cfg, set_seed
    from globe_walking_go2.go2_gym.envs.go2.go2_config import config_go2
    from globe_walking_go2.go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv

    from globe_walking_go2.go2_gym_learn.ppo_cse import Runner
    from globe_walking_go2.go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from globe_walking_go2.go2_gym_learn.ppo_cse.actor_critic import AC_Args
    from globe_walking_go2.go2_gym_learn.ppo_cse.ppo import PPO_Args
    from globe_walking_go2.go2_gym_learn.ppo_cse import RunnerArgs
    from globe_walking_go2.scripts.play import play_go2

    from ml_logger import logger

    set_seed(seed, torch_deterministic=False)

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

    if resume_path:
        RunnerArgs.resume = True
        RunnerArgs.load_run = resume_path
        RunnerArgs.resume_checkpoint = os.path.join(RunnerArgs.load_run, "checkpoints", "ac_weights_last.pt")


    run_dir = Path(f"{MINI_GYM_ROOT_DIR}/../runs").resolve()
    time_now = logger.utcnow(f'{wandb_group}_%Y-%m-%d_%H:%M:%S')
    logger.configure(time_now, root=str(run_dir))
    run_dir = run_dir / str(time_now)

    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                    Cfg=vars(Cfg))

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

    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=Cfg)  # type: ignore

    env = HistoryWrapper(env)
    runner = Runner(env, device=device, multi_gpu=Cfg.multi_gpu)
    runner.learn(num_learning_iterations=int(iterations), init_at_random_ep_len=True, eval_freq=100)

    # log video of trained policy rollout
    play_go2(run_path=run_dir, dr_config="off", save_video=True, headless=True)


if __name__ == '__main__':
    from pathlib import Path
    from globe_walking_go2.go2_gym import MINI_GYM_ROOT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="peer222-luh")
    parser.add_argument("--wandb-project", type=str, default="master-thesis")
    parser.add_argument("--wandb-group", type=str, default="globe-walking-go2/x")

    parser.add_argument("--dr-config", type=str, required=True, choices=["eureka", "off"])
    parser.add_argument("--reward-config", type=str, required=True, choices=["eureka", "original"])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    assert args.reward_config == "eureka", "Only Eureka reward is available" # TODO

    resume_path = None
    train_go2(iterations=args.iterations, dr_config=args.dr_config, headless=args.headless, resume_path=resume_path, no_wandb=args.no_wandb, wandb_group=args.wandb_group, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity, seed=args.seed, device=args.device)
