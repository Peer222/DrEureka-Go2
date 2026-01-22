from pathlib import Path
import tyro
from dataclasses import dataclass
import subprocess
import shutil


@dataclass
class Args:
    run_dir: Path
    """Base Eureka Run Directory (not wandb)"""
    include_subdirs: bool = False
    """include wandb files from evaluations"""
    delete_on_success: bool = False
    """Delete wandb directory after successful syncing"""


def upload(args: Args):
    wandb_eureka_run = args.run_dir / "wandb" / "latest-run"
    result = subprocess.run(
        ["wandb", "sync", str(wandb_eureka_run)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if len(result.stderr):
        print(result.stderr)
    else:
        print(result.stdout)
    if args.delete_on_success and not len(result.stderr):
        print("DELETING...")
        shutil.rmtree(str(wandb_eureka_run.parent))

    if args.include_subdirs:
        wandb_subdirs = args.run_dir.rglob("*/*/wandb")
        for subdir in wandb_subdirs:
            print(subdir)
            result = subprocess.run(
                ["wandb", "sync", str(subdir / "latest-run")],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if len(result.stderr):
                print(result.stderr)
            else:
                print(result.stdout)
            if args.delete_on_success and not len(result.stderr):
                shutil.rmtree(str(subdir))


if __name__ == "__main__":
    args = tyro.cli(Args)
    upload(args)
