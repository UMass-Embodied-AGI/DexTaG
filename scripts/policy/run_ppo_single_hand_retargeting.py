#!/usr/bin/env python3
"""Example: Train PPO on WUJI Hand Retargeting task with object manipulation using Genesis RL.

This script trains a policy to follow reference hand trajectories from demonstrations
while manipulating objects, following the ManipTrans DexHandImitator architecture.
"""
import genesis as gs
gs.init(performance_mode=True, backend=getattr(gs.constants.backend, "cuda"))


import glob
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import gs_env.sim.envs as gs_envs
import numpy as np
import torch
from gs_agent.algos.config.registry import (
    PPO_HAND_IMITATOR_TACTILE_MAP_XARM,
)
from gs_agent.algos.ppo import PPO
from gs_agent.runners.config.registry import RUNNER_SINGLE_HAND_RETARGETING_MLP
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry, OBJECT_REGISTRY
from utils import apply_overrides_generic, config_to_yaml, seed_everything


def create_gs_env(
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    args: Any = None,
    eval_mode: bool = False,
) -> Any:
    """Instantiate the env class named by args.env_name."""
    if torch.cuda.is_available() and device == "cuda":
        device_tensor = torch.device("cuda")
    else:
        device_tensor = torch.device("cpu")
    print(f"Using device: {device_tensor}")

    env_class = getattr(gs_envs, args.env_name)

    return env_class(
        args=args,
        num_envs=num_envs,
        show_viewer=show_viewer,
        device=device_tensor,  # type: ignore
        eval_mode=eval_mode,
    )


def create_ppo_runner_from_registry(
    env: Any,
    exp_name: str | None = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
) -> OnPolicyRunner:
    """Create PPO runner using configuration from the registry."""
    # Environment setup
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create PPO algorithm
    ppo = PPO(
        env=wrapped_env,
        cfg=algo_cfg,
        device=wrapped_env.device,
    )

    # Create PPO runner
    if exp_name is not None:
        # Avoid mutating a frozen Pydantic model; create a copied config with updated save_path
        runner_args = runner_args.model_copy(update={"save_path": Path(f"./logs/{exp_name}")})
    runner = OnPolicyRunner(
        algorithm=ppo,
        runner_args=runner_args,
        device=wrapped_env.device,
    )
    return runner


def evaluate_policy(
    exp_name: str,
    show_viewer: bool = False,
    num_ckpt: int | None = None,
    device: str = "cuda",
    env_args: Any = None,
    algo_cfg: Any = None,
    save_trajectory: bool = False,
    num_runs_per_traj: int = 5,
) -> None:
    """Evaluate the policy."""
    print("=" * 80)
    print("EVALUATION MODE: Disabling domain randomization and observation noise")
    print("=" * 80)

    # Disable observation noise for evaluation
    env_args = env_args.model_copy(
        update={
            "obs_noises": {},  # Disable observation noise
        }
    )

    # Disable domain randomization for evaluation
    from gs_env.sim.robots.config.schema import DomainRandomizationArgs

    robot_args = env_args.robot_args.model_copy(
        update={
            "dr_args": DomainRandomizationArgs(
                kp_range=(1.0, 1.0),
                kd_range=(1.0, 1.0),
                motor_strength_range=(1.0, 1.0),
                motor_offset_range=(0.0, 0.0),
                friction_range=(1.0, 1.0),
                mass_range=(0.0, 0.0),
                com_displacement_range=(0.0, 0.0),
            )
        }
    )
    env_args = env_args.model_copy(update={"robot_args": robot_args})

    # Find the experiment directory without creating a new runner
    log_pattern = f"logs/{exp_name}/*"
    log_dirs = glob.glob(log_pattern)
    if not log_dirs:
        raise FileNotFoundError(f"No experiment directories found matching pattern: {log_pattern}")

    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    # Pick the most recent run dir that actually contains a checkpoint.
    exp_dir = next(
        (d for d in log_dirs if list(Path(d).glob("checkpoints/*.pt"))),
        None,
    )
    if exp_dir is None:
        raise FileNotFoundError(
            f"No run directory with checkpoints found under logs/{exp_name}/"
        )
    print(f"Loading policy from experiment: {exp_dir}")

    # Load checkpoint - either specific one or latest
    if num_ckpt is not None:
        ckpt_path = Path(exp_dir) / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    else:
        ckpt_path = load_latest_model(Path(exp_dir))
        num_ckpt = int(ckpt_path.stem.split("_")[-1])

    print(f"Loading checkpoint: {ckpt_path}")

    # Create environment for evaluation
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=1,
        device=device,
        args=env_args,
        eval_mode=True,
    )

    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create PPO algorithm and load checkpoint
    ppo = PPO(
        env=wrapped_env,
        cfg=algo_cfg,
        device=wrapped_env.device,
    )
    ppo.load(ckpt_path, load_optimizer=False)

    # Get inference policy
    inference_policy = ppo.get_inference_policy()

    print("Starting evaluation...")

    def evaluate() -> None:
        nonlocal wrapped_env, inference_policy, show_viewer, exp_name, save_trajectory
        print(f"Running {num_runs_per_traj} pass(es) over all trajectories (Ctrl+C to stop early)")

        num_traj = wrapped_env.env._num_trajectories
        traj_ids = list(wrapped_env.env._traj_ids)
        traj_lengths = np.asarray(wrapped_env.env._traj_lengths)
        print(
            f"Eval setup: {num_traj} trajectories x {num_runs_per_traj} runs each "
            f"= {num_traj * num_runs_per_traj} total episodes "
            f"({num_runs_per_traj} iterations of {num_traj} episodes)."
        )

        step_count = 0
        total_reward = 0.0
        episode_count = 1
        episode_length = 0
        iteration_idx = 0
        episodes_in_iter = 0

        # Per-trajectory accumulators — cumulative across iterations (no reset).
        per_traj_lengths: dict[int, list[int]] = {i: [] for i in range(num_traj)}
        per_traj_rewards: dict[int, list[float]] = {i: [] for i in range(num_traj)}
        per_traj_success: dict[int, list[bool]] = {i: [] for i in range(num_traj)}
        per_traj_completion: dict[int, list[float]] = {i: [] for i in range(num_traj)}

        # Build per-reward-term tracking from loaded reward functions
        reward_scales = {
            name: func.scale for name, func in wrapped_env.env._reward_functions.items()
        }
        print(f"Tracking {len(reward_scales)} reward terms: {list(reward_scales.keys())}")
        # Per-step accumulators (unscaled) for the current episode
        step_rewards: dict[str, list[float]] = {name: [] for name in reward_scales}
        # Per-trajectory per-reward-term averages — cumulative.
        per_traj_reward_terms: dict[int, dict[str, list[float]]] = {
            i: {name: [] for name in reward_scales} for i in range(num_traj)
        }

        # Trajectory recording
        if save_trajectory:
            traj_save_dir = Path("./deploy/logs") / exp_name / "trajectories"
            traj_save_dir.mkdir(parents=True, exist_ok=True)
            traj_record: dict[str, list] = {
                "base_target_pos": [],       # (T, 7) arm control commands
                "hand_target_pos": [],       # (T, 20) hand control commands
                "arm_dof_pos": [],           # (T, 7) actual arm joint positions
                "hand_dof_pos": [],          # (T, 20) actual hand joint positions
                "object_pos": [],            # (T, 3) world-frame object position
                "object_quat": [],           # (T, 4) WXYZ world-frame object rotation
                "tactile_map": [],           # (T, 24, 32) current tactile map
                "target_tactile_map": [],    # (T, 24, 32) target tactile map
            }
            traj_episode_idx = 0

        # Reset environment
        obs, _ = wrapped_env.get_observations()

        # Create a wrapper that always uses deterministic=True
        class DeterministicWrapper(torch.nn.Module):
            def __init__(self, policy: Any) -> None:
                super().__init__()
                self.policy = policy

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                action, _ = self.policy(obs, deterministic=True)
                return action

        # Wrap and trace the policy with deterministic=True baked in
        wrapped_policy = DeterministicWrapper(inference_policy)
        inference_policy = torch.jit.trace(wrapped_policy, obs)

        while True:
            current_traj_idx = int(wrapped_env.env.env_traj_idx[0].item())

            # Get action from policy
            with torch.no_grad():
                action = inference_policy(obs)  # type: ignore[misc]

            # Step environment
            obs, _, reward, terminated, truncated, info = wrapped_env.step(action)

            # Track per-reward-term unscaled values
            reward_terms = info.get("reward_terms", {})
            for name, scale in reward_scales.items():
                if name in reward_terms:
                    scaled_val = reward_terms[name][0].cpu().item()
                    unscaled_val = scaled_val / scale if scale != 0 else 0.0
                    step_rewards[name].append(unscaled_val)

            # Record trajectory data
            if save_trajectory:
                env = wrapped_env.env
                traj_record["base_target_pos"].append(env._robot._last_base_target[0].cpu().numpy())
                traj_record["hand_target_pos"].append(env._robot._last_hand_target[0].cpu().numpy())
                traj_record["arm_dof_pos"].append(env.arm_dof_pos[0].cpu().numpy())
                traj_record["hand_dof_pos"].append(env.hand_dof_pos[0].cpu().numpy())
                # env.object_pos/quat are kept in sync by update_buffers() each step.
                traj_record["object_pos"].append(env.object_pos[0].cpu().numpy())
                traj_record["object_quat"].append(env.object_quat[0].cpu().numpy())
                traj_record["tactile_map"].append(
                    env.tactile_map_flat[0].reshape(24, 32).cpu().numpy()
                )
                traj_record["target_tactile_map"].append(
                    env.target_tactile_map_flat[0].reshape(24, 32).cpu().numpy()
                )

            # Accumulate reward
            total_reward += reward.item()
            step_count += 1
            episode_length += 1

            if torch.any(terminated) or torch.any(truncated):
                # Bucket this episode's stats under the trajectory it belonged to.
                per_traj_lengths[current_traj_idx].append(episode_length)
                per_traj_rewards[current_traj_idx].append(total_reward)
                success = bool(torch.any(info["termination"]["succeeded"]).cpu().item())
                per_traj_success[current_traj_idx].append(success)
                completion = min(
                    1.0, episode_length / float(traj_lengths[current_traj_idx])
                )
                per_traj_completion[current_traj_idx].append(completion)

                # Per-reward-term episode averages (unscaled)
                for name in reward_scales:
                    vals = step_rewards[name]
                    per_traj_reward_terms[current_traj_idx][name].append(
                        float(np.mean(vals)) if vals else 0.0
                    )
                    step_rewards[name] = []

                # Save trajectory for this episode
                if save_trajectory:
                    traj_data = {k: np.stack(v) for k, v in traj_record.items()}
                    np.savez(traj_save_dir / f"episode_{traj_episode_idx:04d}_{traj_ids[current_traj_idx]}.npz", **traj_data)
                    print(f"  Saved trajectory ({traj_data['base_target_pos'].shape[0]} steps) to episode_{traj_episode_idx:04d}_{traj_ids[current_traj_idx]}.npz")
                    traj_episode_idx += 1
                    traj_record = {k: [] for k in traj_record}

                print(
                    f"Episode {episode_count} (iter {iteration_idx}, "
                    f"traj[{current_traj_idx}]={traj_ids[current_traj_idx]}) "
                    f"ends with length {episode_length}/"
                    f"{int(traj_lengths[current_traj_idx])} steps "
                    f"({completion:.1%}), total reward {total_reward:.2f}, "
                    f"success {success}"
                )

                # Reset episode-specific counters
                episode_length = 0
                total_reward = 0.0
                episode_count += 1
                episodes_in_iter += 1

                # Iteration boundary: one full cycle through all trajectories.
                # Print cumulative summary (each traj has iteration_idx+1 samples).
                if episodes_in_iter >= num_traj:
                    iteration_idx += 1
                    episodes_in_iter = 0

                    print(f"\n{'='*80}")
                    print(
                        f"Iteration {iteration_idx} done — cumulative summary "
                        f"(each trajectory run {iteration_idx} time"
                        f"{'s' if iteration_idx != 1 else ''})"
                    )
                    print(f"{'='*80}")

                    # Per-trajectory rows
                    print(
                        f"  {'idx':>3}  {'traj_id':<24}  {'success':>8}  "
                        f"{'completion':>11}  {'avg_len':>8}  {'traj_len':>8}  "
                        f"{'avg_reward':>11}"
                    )
                    for i in range(num_traj):
                        succ = (
                            float(np.mean(per_traj_success[i]))
                            if per_traj_success[i] else float("nan")
                        )
                        comp = (
                            float(np.mean(per_traj_completion[i]))
                            if per_traj_completion[i] else float("nan")
                        )
                        avg_len = (
                            float(np.mean(per_traj_lengths[i]))
                            if per_traj_lengths[i] else float("nan")
                        )
                        avg_r = (
                            float(np.mean(per_traj_rewards[i]))
                            if per_traj_rewards[i] else float("nan")
                        )
                        print(
                            f"  {i:>3}  {traj_ids[i]:<24}  {succ:>8.4f}  "
                            f"{comp:>10.2%}  {avg_len:>8.1f}  "
                            f"{int(traj_lengths[i]):>8d}  {avg_r:>11.2f}"
                        )

                    # Overall (flatten all trajectories)
                    all_lengths = [v for vs in per_traj_lengths.values() for v in vs]
                    all_rewards = [v for vs in per_traj_rewards.values() for v in vs]
                    all_success = [v for vs in per_traj_success.values() for v in vs]
                    all_completion = [
                        v for vs in per_traj_completion.values() for v in vs
                    ]
                    print(
                        f"\n  Overall: success={np.mean(all_success):.4f}  "
                        f"completion={np.mean(all_completion):.2%}  "
                        f"avg_len={np.mean(all_lengths):.1f}  "
                        f"avg_reward={np.mean(all_rewards):.2f}  "
                        f"(n={len(all_lengths)} episodes)"
                    )
                    print(f"  Avg unscaled reward terms (overall):")
                    for name in reward_scales:
                        all_vals = [
                            v for vs in per_traj_reward_terms.values() for v in vs[name]
                        ]
                        if all_vals:
                            print(f"    {name}: {np.mean(all_vals):.4f}")
                    print(f"{'='*80}\n")

                    if iteration_idx >= num_runs_per_traj:
                        print(
                            f"Reached {num_runs_per_traj} iterations "
                            f"({num_runs_per_traj} runs per trajectory). Stopping."
                        )
                        return


    try:
        if platform.system() == "Darwin" and show_viewer:
            import threading

            threading.Thread(target=evaluate).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            evaluate()
    except KeyboardInterrupt:
        pass


def train_policy(
    exp_name: str | None = None,
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    use_wandb: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    env_args: Any = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
    resume_checkpoint: str | None = None,
) -> None:
    """Train the policy using PPO.

    Args:
        resume_checkpoint: Path to checkpoint file to resume training from, or 'latest' to auto-find
    """

    # Create environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=num_envs,
        device=device,
        args=env_args,
    )

    # Get configuration and runner from registry
    runner = create_ppo_runner_from_registry(
        env,
        exp_name=exp_name,
        algo_cfg=algo_cfg,
        runner_args=runner_args,
    )

    # Load checkpoint if resuming training
    start_iteration = 0
    resume_from_dir = None
    ckpt_path = None
    if resume_checkpoint is not None:
        if resume_checkpoint == "latest":
            # Find latest checkpoint from experiment (graceful: start fresh if none found)
            if exp_name is None:
                raise ValueError("exp_name must be specified when using resume_checkpoint='latest'")
            log_pattern = f"logs/{exp_name}/*"
            log_dirs = glob.glob(log_pattern)
            if log_dirs:
                # Sort by directory name (descending) to pick the latest timestamped run.
                log_dirs.sort(key=lambda x: os.path.basename(x), reverse=True)
                for exp_dir in log_dirs:
                    try:
                        ckpt_path = load_latest_model(Path(exp_dir))
                        print(f"Resuming from latest checkpoint: {ckpt_path}")
                        break
                    except FileNotFoundError:
                        continue
                if ckpt_path is None:
                    print("No checkpoint files found in any experiment dir. Starting fresh.")
            else:
                print("No previous experiment found. Starting fresh training.")
        else:
            # Use specified checkpoint path
            ckpt_path = Path(resume_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            print(f"Resuming from checkpoint: {ckpt_path}")

    if ckpt_path is not None:
        # Load the checkpoint
        runner.load_checkpoint(ckpt_path)

        # Use the iteration counter from the checkpoint itself (already incremented)
        start_iteration = runner.algorithm.current_iter
        print(f"Resuming training from iteration {start_iteration}")

        # Sync environment's training_step with the loaded iteration counter
        env.training_step = start_iteration
        print(f"Synced environment training_step to {start_iteration}")

        # When resuming, use the same directory as the checkpoint
        resume_from_dir = ckpt_path.parent.parent  # Go up from checkpoints/ to experiment dir
        print(f"Resuming will continue in directory: {resume_from_dir}")

        # Update the runner's checkpoint directory to use the resumed location
        runner.save_dir = resume_from_dir
        runner.checkpoint_dir = resume_from_dir / "checkpoints"
        print(f"Checkpoints will be saved to: {runner.checkpoint_dir}")

    # Set up logging with proper configuration
    if exp_name is not None:
        save_path = Path(f"./logs/{exp_name}")
    else:
        save_path = runner_args.save_path

    # Use existing directory when resuming, or create new timestamped one for fresh training
    if resume_from_dir is not None:
        logger_folder = resume_from_dir
    else:
        logger_folder = save_path / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logger_configure(
        folder=str(logger_folder),
        format_strings=["stdout", "csv", "wandb"],
        entity=wandb_entity,
        project=wandb_project,
        exp_name=exp_name,
        mode="online" if use_wandb and (not show_viewer) else "disabled",
    )

    # Save configuration files to YAML
    print("Saving configuration files to YAML...")
    config_dir = logger_folder / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_to_yaml(env_args, config_dir / "env_args.yaml")
    config_to_yaml(algo_cfg, config_dir / "algo_cfg.yaml")
    config_to_yaml(runner_args, config_dir / "runner_args.yaml")

    # Train using Runner
    print("Starting training...")

    def train() -> None:
        nonlocal runner, logger, start_iteration
        train_summary_info = runner.train(metric_logger=logger, start_iteration=start_iteration)
        print("Training completed successfully!")
        print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
        print(f"Total iterations: {train_summary_info['total_iterations']}.")
        print(f"Total steps: {train_summary_info['total_steps']}.")
        print(f"Total reward: {train_summary_info['final_reward']:.2f}.")

    try:
        if platform.system() == "Darwin" and show_viewer:
            import threading

            threading.Thread(target=train).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            train()
    except KeyboardInterrupt:
        pass


def main(
    num_envs: int = 4096,
    show_viewer: bool = False,
    device: str = "cuda",
    eval: bool = False,
    exp_name: str | None = None,
    num_ckpt: int | None = None,
    use_wandb: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    env_name: str = "single_hand_retargeting_tactile_map_xarm",
    resume_checkpoint: str | None = None,
    object_name: str | None = None,
    save_trajectory: bool = False,
    num_runs_per_traj: int = 5,
    **cfg_overrides: Any,
) -> None:
    """Entry point.

    You can override configs via dot-notation, e.g.:
    --object_name=marker_pen  # Object type from OBJECT_REGISTRY (marker_pen or hammer)
    --reward_args.WristPositionTrackingReward.scale=0.2
    --algo.lr=1e-3
    --runner.total_iterations=5000
    --resume_checkpoint=latest  # Resume from latest checkpoint in exp_name
    --resume_checkpoint=path/to/checkpoint.pt  # Resume from specific checkpoint
    --exp_name=my_experiment  # Run name: logs/<exp_name>/<timestamp>/ and W&B run name (required for --eval)
    --wandb_project=my_project  # Wandb project name
    --wandb_entity=my_team  # Wandb team/user name
    """
    # Bucket overrides into env / algo / runner
    env_overrides: dict[str, Any] = {}
    algo_overrides: dict[str, Any] = {}
    runner_overrides: dict[str, Any] = {}

    for k, v in cfg_overrides.items():
        if k.startswith("cfgs.env.") or k.startswith("env.") or k.startswith("reward_args."):
            env_overrides[k] = v
            continue
        if k.startswith("cfgs.algo.") or k.startswith("algo."):
            algo_overrides[k] = v
            continue
        if k.startswith("cfgs.runner.") or k.startswith("runner."):
            runner_overrides[k] = v
            continue

    env_args = EnvArgsRegistry[env_name]

    # Apply object config before env overrides so --env.* can still override
    if object_name is not None:
        if object_name not in OBJECT_REGISTRY:
            raise ValueError(f"Unknown object '{object_name}'. Available: {list(OBJECT_REGISTRY.keys())}")
        env_args = env_args.model_copy(update={"object_config": OBJECT_REGISTRY[object_name]})
        print(f"Using object config: {object_name}")

    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))

    algo_cfg = apply_overrides_generic(
        PPO_HAND_IMITATOR_TACTILE_MAP_XARM, algo_overrides, prefixes=("cfgs.", "algo.")
    )
    runner_args = apply_overrides_generic(
        RUNNER_SINGLE_HAND_RETARGETING_MLP, runner_overrides, prefixes=("cfgs.", "runner.")
    )

    seed_everything(env_args.gs_init_args.seed)
    print(f"Seed: {env_args.gs_init_args.seed}")

    if eval:
        # Evaluation mode - don't create runner to avoid creating empty log dir
        num_envs = 1
        print("Evaluation mode: Loading trained policy")
        evaluate_policy(
            exp_name=exp_name,
            show_viewer=show_viewer,
            num_ckpt=num_ckpt,
            device=device,
            env_args=env_args,
            algo_cfg=algo_cfg,
            save_trajectory=save_trajectory,
            num_runs_per_traj=num_runs_per_traj,
        )
    else:
        # Training mode
        print("Training mode: Starting policy training")
        train_policy(
            exp_name=exp_name,
            show_viewer=show_viewer,
            num_envs=num_envs,
            device=device,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            env_args=env_args,
            algo_cfg=algo_cfg,
            runner_args=runner_args,
            resume_checkpoint=resume_checkpoint,
        )


if __name__ == "__main__":
    fire.Fire(main)
