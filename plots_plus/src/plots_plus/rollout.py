import pandas as pd
from pathlib import Path
import plots_plus


def create_plots(
    rollout_stats_df: pd.DataFrame, graphics_dir: Path, env: str = "globe_walking_go2"
):
    graphics_dir.mkdir(exist_ok=True, parents=True)

    metrics = rollout_stats_df.columns

    fitness_score_df = rollout_stats_df[["time_(s)", "rollout", "rew_fitness_score"]]
    fitness_score_df = fitness_score_df.rename({"rew_fitness_score": "fitness_score"}, axis=1)
    plots_plus.lineplot(
        fitness_score_df,
        x="time_(s)",
        y="fitness_score",
        style="rollout",
        colorpalette=plots_plus.colors.REWARD_COLOR_MAP,
        filepath=graphics_dir / "fitness_score.png",
    )

    rewards = {
        m: m.split("rew_")[-1] for m in metrics if "rew" in m and "fitness_score" not in m
    }
    rewards_df = rollout_stats_df[["time_(s)", "rollout", *rewards.keys()]]  # type: ignore
    rewards_df = rewards_df.rename(rewards, axis=1)
    rewards_df = plots_plus.utils.rotate_df(
        rewards_df,
        ["time_(s)", "rollout"],
        list(rewards.values()),
        "reward",
    )
    plots_plus.lineplot(
        rewards_df,
        x="time_(s)",
        y="reward",
        hue="type",
        style="rollout",
        colorpalette=plots_plus.colors.REWARD_COLOR_MAP,
        filepath=graphics_dir / "rewards.png",
    )

    for rollout_idx in rollout_stats_df["rollout"].unique():

        contact_forces = ["front_left", "front_right", "rear_left", "rear_right"]
        contact_forces_df = rollout_stats_df[["time_(s)", "rollout", *contact_forces]]  # type: ignore
        contact_forces_df = contact_forces_df[contact_forces_df["rollout"] == rollout_idx]  # type: ignore
        contact_forces_df = plots_plus.utils.rotate_df(
            contact_forces_df,
            ["time_(s)", "rollout"],
            contact_forces,
            ylabel="force_(N)",
        )
        plots_plus.lineplot(
            contact_forces_df,
            x="time_(s)",
            y="force_(N)",
            hue="type",
            ylim=(-5, None),
            colorpalette=plots_plus.colors.CONTACT_FORCES_COLOR_MAP,
            filepath=graphics_dir / str(rollout_idx) / "contact_forces.png",
        )

        torques = {m: m.split("torque_")[-1] for m in metrics if "torque_" in m}
        torques_df = rollout_stats_df[["time_(s)", "rollout", *torques.keys()]]  # type: ignore
        torques_df = torques_df.rename(torques, axis=1)[
            torques_df["rollout"] == rollout_idx
        ]
        torques_df = plots_plus.utils.rotate_df(
            torques_df,
            ["time_(s)", "rollout"],
            list(torques.values()),
            ylabel="torque_(Nm)",
        )
        plots_plus.lineplot(
            torques_df,
            x="time_(s)",
            y="torque_(Nm)",
            hue="type",
            colorpalette=plots_plus.colors.JOINT_COLOR_MAP,
            filepath=graphics_dir / str(rollout_idx) / "torques.png",
        )
        joint_positions = {
            m: m.split("position_")[-1] for m in metrics if "position_" in m
        }
        joint_positions_df = rollout_stats_df[["time_(s)", "rollout", *joint_positions.keys()]]  # type: ignore
        joint_positions_df = joint_positions_df.rename(joint_positions, axis=1)[
            joint_positions_df["rollout"] == rollout_idx
        ]
        joint_positions_df = plots_plus.utils.rotate_df(
            joint_positions_df,
            ["time_(s)", "rollout"],
            list(joint_positions.values()),
            ylabel="joint_position_(deg)",
        )
        plots_plus.lineplot(
            joint_positions_df,
            x="time_(s)",
            y="joint_position_(deg)",
            hue="type",
            colorpalette=plots_plus.colors.JOINT_COLOR_MAP,
            filepath=graphics_dir / str(rollout_idx) / "joint_positions.png",
        )

    global_linear_velocities = {
        "global_linear_x": "x",
        "global_linear_y": "y",
        "global_linear_z": "z",
    }
    global_linear_df: pd.DataFrame = rollout_stats_df[["time_(s)", "rollout", *global_linear_velocities.keys()]]  # type: ignore
    global_linear_df = global_linear_df.rename(global_linear_velocities, axis=1)
    global_linear_df = plots_plus.utils.rotate_df(
        global_linear_df,
        ["time_(s)", "rollout"],
        list(global_linear_velocities.values()),
        ylabel="global_linear_velocity_(m/s)",
    )
    plots_plus.lineplot(
        global_linear_df,
        x="time_(s)",
        y="global_linear_velocity_(m/s)",
        hue="type",
        colorpalette=plots_plus.colors.VELOCITY_COLOR_MAP,
        filepath=graphics_dir / "global_linear_velocities.png",
    )
    global_angular_velocities = {
        "global_angular_x": "x",
        "global_angular_y": "y",
        "global_angular_z": "z",
    }
    global_angular_df: pd.DataFrame = rollout_stats_df[["time_(s)", "rollout", *global_angular_velocities.keys()]]  # type: ignore
    global_angular_df = global_angular_df.rename(global_angular_velocities, axis=1)
    global_angular_df = plots_plus.utils.rotate_df(
        global_angular_df,
        ["time_(s)", "rollout"],
        list(global_angular_velocities.values()),
        ylabel="global_angular_velocity_(m/s)",
    )
    plots_plus.lineplot(
        global_angular_df,
        x="time_(s)",
        y="global_angular_velocity_(m/s)",
        hue="type",
        colorpalette=plots_plus.colors.VELOCITY_COLOR_MAP,
        filepath=graphics_dir / "global_angular_velocities.png",
    )

    linear_velocities = {"linear_x": "x", "linear_y": "y", "linear_z": "z"}
    linear_df: pd.DataFrame = rollout_stats_df[["time_(s)", "rollout", *linear_velocities.keys()]]  # type: ignore
    linear_df = linear_df.rename(linear_velocities, axis=1)
    linear_df = plots_plus.utils.rotate_df(
        linear_df,
        ["time_(s)", "rollout"],
        list(linear_velocities.values()),
        ylabel="linear_velocity_(m/s)",
    )
    plots_plus.lineplot(
        linear_df,
        x="time_(s)",
        y="linear_velocity_(m/s)",
        hue="type",
        colorpalette=plots_plus.colors.VELOCITY_COLOR_MAP,
        filepath=graphics_dir / "linear_velocities.png",
    )
    angular_velocities = {"angular_x": "x", "angular_y": "y", "angular_z": "z"}
    angular_df: pd.DataFrame = rollout_stats_df[["time_(s)", "rollout", *angular_velocities.keys()]]  # type: ignore
    angular_df = angular_df.rename(angular_velocities, axis=1)
    angular_df = plots_plus.utils.rotate_df(
        angular_df,
        ["time_(s)", "rollout"],
        list(angular_velocities.values()),
        ylabel="angular_velocity_(m/s)",
    )
    plots_plus.lineplot(
        angular_df,
        x="time_(s)",
        y="angular_velocity_(m/s)",
        hue="type",
        colorpalette=plots_plus.colors.VELOCITY_COLOR_MAP,
        filepath=graphics_dir / "angular_velocities.png",
    )

    positions = ["x", "y", "z"]
    positions_df: pd.DataFrame = rollout_stats_df[["time_(s)", "rollout", *positions]]  # type: ignore
    positions_df.loc[:, "x"] -= positions_df["x"].iloc[0]
    positions_df.loc[:, "y"] -= positions_df["y"].iloc[0]
    positions_df.loc[:, "z"] -= positions_df["z"].iloc[0]
    # reduce plotted positions to 10 per second
    positions_df: pd.DataFrame = positions_df[((positions_df["time_(s)"] * 25) % 10) == 0]  # type: ignore
    if "forward_locomotion" in env:
        x_max = positions_df["x"].abs().max()
        y_max = positions_df["y"].abs().max()
    else:
        x_max = max(positions_df["x"].abs().max(), positions_df["y"].abs().max())
        y_max = x_max
    plots_plus.scatterplot(
        positions_df,
        x="x",
        y="y",
        hue="z",
        style="rollout",
        xlim=(-x_max, x_max),
        ylim=(-y_max, y_max),
        colorpalette=plots_plus.colors.CONTINUOUS_COLOR_MAP,
        filepath=graphics_dir / "positions_z-hue.png",
    )
    plots_plus.scatterplot(
        positions_df,
        x="x",
        y="y",
        hue="time_(s)",
        style="rollout",
        xlim=(-x_max, x_max),
        ylim=(-y_max, y_max),
        colorpalette=plots_plus.colors.CONTINUOUS_COLOR_MAP,
        filepath=graphics_dir / "positions_time-hue.png",
    )

    out_of_limits_df: pd.DataFrame = rollout_stats_df[["time_(s)", "rollout", "out_of_limits"]]  # type: ignore
    plots_plus.lineplot(
        out_of_limits_df,
        x="time_(s)",
        y="out_of_limits",
        hue="rollout",
        ylim=(-0.1, None),
        alpha=0.7,
        filepath=graphics_dir / "out_of_limits.png",
    )


__all__ = ["create_plots"]


def __dir__():
    return __all__


if __name__ == "__main__":
    import tyro
    from dataclasses import dataclass

    @dataclass
    class Args:
        statspath: Path
        """Path to rollout statistics file"""
        result_dir: Path
        """directory in which graphics are stored"""
        max_rollout_duration: float = 40
        """maximal length of the rollout in seconds that is plotted"""

    args = tyro.cli(Args)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    rollout_stats_df = pd.read_csv(args.statspath)
    rollout_stats_df: pd.DataFrame = rollout_stats_df[rollout_stats_df["time_(s)"] <= args.max_rollout_duration]  # type: ignore

    create_plots(
        rollout_stats_df,
        args.result_dir,
    )
