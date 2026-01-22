from dataclasses import dataclass
import tyro
from huggingface_hub import snapshot_download


if __name__ == "__main__":

    @dataclass
    class Args:
        """Set environment variable: HF_HUB_DISABLE_XET=1 if download fails due to file system error"""

        model_name: str
        "huggingface model name / repo id"

    args = tyro.cli(Args)

    snapshot_download(
        repo_id=args.model_name,
        local_dir=f"/bigwork/nhwpduep/master_thesis/models/{args.model_name}",
        # local_dir_use_symlinks=False, # deprecated
        # resume_download=True, # deprecated
        max_workers=1,
    )
