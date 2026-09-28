import glob
import os
from pathlib import Path


def load_latest_model(log_dir: Path) -> Path:
    """Return the path of the most recent checkpoint in log_dir/checkpoints."""
    checkpoint_files = glob.glob(os.path.join(log_dir, "checkpoints/checkpoint_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    checkpoint_files.sort(
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]), reverse=True
    )
    latest_checkpoint = checkpoint_files[0]

    print(f"Loading model from: {latest_checkpoint}")
    return Path(latest_checkpoint)
