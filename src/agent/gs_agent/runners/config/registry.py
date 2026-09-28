from pathlib import Path

from gs_agent.runners.config.schema import RunnerArgs

RUNNER_SINGLE_HAND_RETARGETING_MLP = RunnerArgs(
    total_iterations=100000,  # More iterations for trajectory following
    log_interval=100,
    save_interval=2500,
    save_path=Path("./logs/ppo_single_hand_retargeting"),
)

# BC runner for single hand retargeting distillation
RUNNER_BC_SINGLE_HAND_RETARGETING = RunnerArgs(
    total_iterations=100000,
    log_interval=100,
    save_interval=1000,
    save_path=Path("./logs/bc_single_hand_retargeting"),
)