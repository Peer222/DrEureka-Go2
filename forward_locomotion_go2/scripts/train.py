import argparse
from pathlib import Path

def train_mc(iterations, command_config, reward_config, dr_config, eureka_target_velocity=None,
             headless=True, no_wandb=False, wandb_group=None, wandb_project=None, wandb_entity=None, seed=0):

    import isaacgym
    assert isaacgym
    import wandb
    from ml_logger import logger

    from forward_locomotion_go2.go2_gym import MINI_GYM_ROOT_DIR
    from forward_locomotion_go2.go2_gym.envs.base.legged_robot_config import Cfg, set_seed
    from forward_locomotion_go2.go2_gym.envs.go2.go2_config import config_go2
    from forward_locomotion_go2.go2_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv
    from forward_locomotion_go2.go2_gym_learn.ppo import Runner
    from forward_locomotion_go2.go2_gym.envs.wrappers.history_wrapper import HistoryWrapper

    from forward_locomotion_go2.go2_gym_learn.ppo.actor_critic import AC_Args
    from forward_locomotion_go2.go2_gym_learn.ppo.ppo import PPO_Args
    from forward_locomotion_go2.go2_gym_learn.ppo import RunnerArgs

    set_seed(seed, torch_deterministic=False)

    if command_config == "original":
        Cfg.commands = Cfg.commands_original
        assert reward_config == "original"
    elif command_config == "constrained":
        Cfg.commands = Cfg.commands_constrained
        assert reward_config == "original"
    elif command_config == "off":
        Cfg.commands = Cfg.commands_original  # Will be turned off below
    else:
        raise NotImplementedError

    if reward_config == "original":
        Cfg.rewards = Cfg.rewards_original
        assert eureka_target_velocity is None
    elif reward_config == "eureka":
        Cfg.rewards = Cfg.rewards_eureka
        if eureka_target_velocity is not None:
            Cfg.rewards.target_velocity = eureka_target_velocity
    elif reward_config == "eureka_original":
        Cfg.rewards = Cfg.rewards_eureka_original
        if eureka_target_velocity is not None:
            Cfg.rewards.target_velocity = eureka_target_velocity
    else:
        raise NotImplementedError

    if dr_config == "original":
        Cfg.domain_rand = Cfg.domain_rand_original
    elif dr_config == "eureka":
        Cfg.domain_rand = Cfg.domain_rand_eureka
    elif dr_config == "off":
        Cfg.domain_rand = Cfg.domain_rand_off
    else:
        raise NotImplementedError
    
    config_go2(Cfg)
    if command_config == "original" or command_config == "constrained":
        Cfg.commands.command_curriculum = True
        Cfg.env.observe_command = True
        Cfg.env.num_observations = 42
    else:
        Cfg.commands.command_curriculum = False

    device = 'cuda:0'
    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=Cfg)  # type: ignore

    run_dir = Path(f"{MINI_GYM_ROOT_DIR}/../runs").resolve()
    time_now = logger.utcnow(f'{wandb_group}/%Y-%m-%d_%H:%M:%S')
    logger.configure(time_now, root=str(run_dir))
    run_dir = run_dir / str(time_now)

    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                    Cfg=vars(Cfg))

    run_name = run_dir.stem
    name_prefix = wandb_group + "/" if wandb_group is not None else ""
    wandb.init(
        dir=run_dir,
        project=wandb_project,
        entity=wandb_entity,
        name=f"{name_prefix}{run_name}",
        group=wandb_group,
        config={
            "AC_Args": vars(AC_Args),
            "PPO_Args": vars(PPO_Args),
            "RunnerArgs": vars(RunnerArgs),
            "Cfg": vars(Cfg),
            "HEADLESS": headless,
        },
    )

    env = HistoryWrapper(env)
    runner = Runner(env, device=device)
    runner.learn(num_learning_iterations=int(iterations), init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="peer222-luh")
    parser.add_argument("--wandb-project", type=str, default="master-thesis")
    parser.add_argument("--wandb-group", type=str, default="forward-locomotion-go2")

    parser.add_argument("--command-config", type=str, default="off", choices=["original", "constrained", "off"])
    parser.add_argument("--reward-config", type=str, required=True, choices=["original", "eureka", "eureka_original"])
    parser.add_argument("--dr-config", type=str, required=True, choices=["original", "eureka", "eureka_original", "off"]) # TODO eureka original

    parser.add_argument("--eureka-target-velocity", type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_mc(iterations=args.iterations, command_config=args.command_config, reward_config=args.reward_config, dr_config=args.dr_config, eureka_target_velocity=args.eureka_target_velocity,
              headless=args.headless, no_wandb=args.no_wandb, wandb_group=args.wandb_group, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity, seed=args.seed)
