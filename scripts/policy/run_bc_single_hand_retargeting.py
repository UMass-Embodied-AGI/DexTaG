#!/usr/bin/env python3
"""Example: Train BC to distill policy from teacher to student for single hand retargeting.

This script distills a PPO-trained teacher policy (with privileged observations)
to a student policy (with deployment-compatible observations).
"""
import genesis as gs
gs.init(performance_mode=True, backend=getattr(gs.constants.backend, "cuda"))


import glob
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import gs_env.sim.envs as gs_envs
import numpy as np
import torch
from gs_agent.algos.bc import BC
from gs_agent.algos.config.registry import BC_SINGLE_HAND_RETARGETING_TEMPORAL
from gs_agent.algos.config.schema import BCArgs
from gs_agent.runners.config.registry import RUNNER_BC_SINGLE_HAND_RETARGETING
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import SingleHandRetargetingEnvArgs
from utils import (
    apply_overrides_generic,
    config_to_yaml,
    seed_everything,
    yaml_to_config,
)


def create_gs_env(
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    args: Any = None,
    eval_mode: bool = False,
) -> gs_envs.SingleHandRetargetingEnvTactileMap:
    """Create Genesis environment for single hand retargeting."""
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


def create_bc_runner_from_registry(
    env: gs_envs.SingleHandRetargetingEnvTactileMap,
    teacher_exp_name: str,
    exp_name: str | None = None,
    teacher_ckpt: int | None = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
) -> tuple[OnPolicyRunner, Path]:
    """Create BC runner to distill from teacher to student.

    Args:
        env: Student environment
        teacher_exp_name: Name of teacher experiment directory
        exp_name: Name of BC experiment directory
        teacher_ckpt: Checkpoint number to load from teacher. If None, loads latest.
        algo_cfg: BC algorithm configuration
        runner_args: Runner configuration
    """
    # Environment setup
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Find teacher experiment directory
    teacher_exp_dir = str(_resolve_teacher_exp_dir(teacher_exp_name))
    print(f"Loading teacher from experiment: {teacher_exp_dir}")

    # Load teacher checkpoint
    if teacher_ckpt is not None:
        teacher_ckpt_path = (
            Path(teacher_exp_dir) / "checkpoints" / f"checkpoint_{teacher_ckpt:04d}.pt"
        )
        if not teacher_ckpt_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint {teacher_ckpt_path} not found")
    else:
        teacher_ckpt_path = load_latest_model(Path(teacher_exp_dir))
        teacher_ckpt = int(teacher_ckpt_path.stem.split("_")[-1])

    print(f"Loading teacher checkpoint: {teacher_ckpt_path}")

    # Load teacher environment config
    teacher_config_path = Path(teacher_exp_dir) / "configs" / "env_args.yaml"
    if not teacher_config_path.exists():
        raise FileNotFoundError(
            f"Teacher config not found at {teacher_config_path}. "
            "Make sure the teacher experiment has configs/env_args.yaml"
        )
    print(f"Loading teacher config from: {teacher_config_path}")

    # Update algo_cfg with teacher path and config
    algo_cfg = algo_cfg.model_copy(
        update={
            "teacher_path": teacher_ckpt_path,
            "teacher_config_path": teacher_config_path,
        }
    )

    # Create BC algorithm
    bc = BC(
        env=wrapped_env,
        cfg=algo_cfg,
        device=wrapped_env.device,
    )

    # Create BC runner
    if exp_name is not None:
        runner_args = runner_args.model_copy(update={"save_path": Path(f"./logs/{exp_name}")})
    runner = OnPolicyRunner(
        algorithm=bc,
        runner_args=runner_args,
        device=wrapped_env.device,
    )

    return runner, teacher_config_path


def evaluate_policy(
    exp_name: str,
    show_viewer: bool = False,
    num_ckpt: int | None = None,
    device: str = "cuda",
    env_overrides: dict[str, Any] | None = None,
    algo_cfg: Any = None,
    save_trajectory: bool = False,
    num_runs_per_traj: int = 5,
) -> None:
    """Evaluate a trained BC policy."""
    if env_overrides is None:
        env_overrides = {}

    print("=" * 80)
    print("EVALUATION MODE: Keeping student obs noise + delay intact")
    print("=" * 80)

    # Locate experiment directory
    exp_dir = _latest_run_dir(exp_name)
    print(f"Loading policy from experiment: {exp_dir}")

    # Resolve checkpoint
    if num_ckpt is not None:
        ckpt_path = Path(exp_dir) / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    else:
        ckpt_path = load_latest_model(Path(exp_dir))
        num_ckpt = int(ckpt_path.stem.split("_")[-1])
    print(f"Loading checkpoint: {ckpt_path}")

    # Load configs
    print(f"Loading configs from experiment: {exp_dir}")
    env_args = yaml_to_config(Path(exp_dir) / "configs" / "env_args.yaml", SingleHandRetargetingEnvArgs)
    algo_cfg = yaml_to_config(Path(exp_dir) / "configs" / "algo_cfg.yaml", BCArgs)

    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))

    # Build eval environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=1,
        device=device,
        args=env_args,
        eval_mode=True,
    )
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Recreate BC algo and load weights
    bc = BC(env=wrapped_env, cfg=algo_cfg, device=wrapped_env.device)
    bc.load(ckpt_path, load_optimizer=False)
    inference_policy = bc.get_inference_policy()

    print("Starting evaluation...")

    def evaluate() -> None:
        nonlocal wrapped_env, inference_policy, show_viewer, exp_name, env, save_trajectory

        print(f"Running {num_runs_per_traj} pass(es) over all trajectories (Ctrl+C to stop early)")

        num_traj = wrapped_env.env._num_trajectories
        traj_ids = list(wrapped_env.env._traj_ids)
        traj_lengths = np.asarray(wrapped_env.env._traj_lengths)
        print(
            f"Eval setup: {num_traj} trajectories x {num_runs_per_traj} runs each "
            f"= {num_traj * num_runs_per_traj} total episodes "
            f"({num_runs_per_traj} iterations of {num_traj} episodes)."
        )

        # Trajectory recording
        if save_trajectory:
            traj_save_dir = Path("./deploy/logs") / exp_name / "trajectories"
            traj_save_dir.mkdir(parents=True, exist_ok=True)
            traj_record: dict[str, list] = {
                "base_target_pos": [],   # (T, 7) arm control commands
                "hand_target_pos": [],   # (T, 20) hand control commands
                "arm_dof_pos": [],       # (T, 7) actual arm joint positions
                "hand_dof_pos": [],      # (T, 20) actual hand joint positions
                "object_pos": [],        # (T, 3) world-frame object position
                "object_quat": [],       # (T, 4) WXYZ world-frame object rotation
            }
            traj_episode_idx = 0

        # Get an example observation and trace the policy
        obs, _ = wrapped_env.get_observations()
        traced_policy = torch.jit.trace(inference_policy, obs)

        step_count = 0
        total_reward = 0.0
        episode_count = 1
        episode_length = 0
        iteration_idx = 0
        episodes_in_iter = 0

        # Per-step accumulators (current episode) for BC-specific tracking errors.
        wrist_pos_errors: list[float] = []
        wrist_rot_errors: list[float] = []
        object_pos_errors: list[float] = []
        object_rot_errors: list[float] = []

        # Per-trajectory accumulators — cumulative across iterations (no reset).
        per_traj_lengths: dict[int, list[int]] = {i: [] for i in range(num_traj)}
        per_traj_rewards: dict[int, list[float]] = {i: [] for i in range(num_traj)}
        per_traj_success: dict[int, list[bool]] = {i: [] for i in range(num_traj)}
        per_traj_completion: dict[int, list[float]] = {i: [] for i in range(num_traj)}
        per_traj_wrist_pos_errors: dict[int, list[float]] = {i: [] for i in range(num_traj)}
        per_traj_wrist_rot_errors: dict[int, list[float]] = {i: [] for i in range(num_traj)}
        per_traj_object_pos_errors: dict[int, list[float]] = {i: [] for i in range(num_traj)}
        per_traj_object_rot_errors: dict[int, list[float]] = {i: [] for i in range(num_traj)}

        while True:
            current_traj_idx = int(wrapped_env.env.env_traj_idx[0].item())

            with torch.no_grad():
                action = traced_policy(obs)  # type: ignore[misc]

            obs, _, reward, terminated, truncated, info = wrapped_env.step(action)

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

            cur_idx = wrapped_env.env.progress_buf[0].cpu().item()

            # Track wrist errors
            target_wrist_pos = wrapped_env.env._traj_data["wrist_pos"][0, cur_idx].cpu()
            current_wrist_pos = wrapped_env.env.base_pos[0].cpu()
            wrist_pos_error = torch.norm(target_wrist_pos - current_wrist_pos).item()
            wrist_pos_errors.append(wrist_pos_error)

            target_wrist_quat = wrapped_env.env._traj_data["wrist_quat"][0, cur_idx].cpu()
            current_wrist_quat = wrapped_env.env.base_quat[0].cpu()
            wrist_rot_error = 1 - torch.abs(torch.sum(target_wrist_quat * current_wrist_quat)).item()
            wrist_rot_errors.append(wrist_rot_error)

            # Track object errors
            target_obj_pos = wrapped_env.env._traj_data["obj_pos"][0, cur_idx].cpu()
            current_obj_pos = wrapped_env.env.object_pos[0].cpu()
            obj_pos_error = torch.norm(target_obj_pos - current_obj_pos).item()
            object_pos_errors.append(obj_pos_error)

            target_obj_quat = wrapped_env.env._traj_data["obj_quat"][0, cur_idx].cpu()
            current_obj_quat = wrapped_env.env.object_quat[0].cpu()
            obj_rot_error = 1 - torch.abs(torch.sum(target_obj_quat * current_obj_quat)).item()
            object_rot_errors.append(obj_rot_error)

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

                per_traj_wrist_pos_errors[current_traj_idx].append(
                    float(np.mean(wrist_pos_errors)) if wrist_pos_errors else 0.0
                )
                per_traj_wrist_rot_errors[current_traj_idx].append(
                    float(np.mean(wrist_rot_errors)) if wrist_rot_errors else 0.0
                )
                per_traj_object_pos_errors[current_traj_idx].append(
                    float(np.mean(object_pos_errors)) if object_pos_errors else 0.0
                )
                per_traj_object_rot_errors[current_traj_idx].append(
                    float(np.mean(object_rot_errors)) if object_rot_errors else 0.0
                )
                wrist_pos_errors = []
                wrist_rot_errors = []
                object_pos_errors = []
                object_rot_errors = []

                # Save trajectory for this episode
                if save_trajectory:
                    traj_data = {k: np.stack(v) for k, v in traj_record.items()}
                    np.savez(traj_save_dir / f"episode_{traj_episode_idx:04d}.npz", **traj_data)
                    print(f"  Saved trajectory ({traj_data['base_target_pos'].shape[0]} steps) to episode_{traj_episode_idx:04d}.npz")
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
                    all_wrist_pos = [
                        v for vs in per_traj_wrist_pos_errors.values() for v in vs
                    ]
                    all_wrist_rot = [
                        v for vs in per_traj_wrist_rot_errors.values() for v in vs
                    ]
                    all_object_pos = [
                        v for vs in per_traj_object_pos_errors.values() for v in vs
                    ]
                    all_object_rot = [
                        v for vs in per_traj_object_rot_errors.values() for v in vs
                    ]
                    print(
                        f"\n  Overall: success={np.mean(all_success):.4f}  "
                        f"completion={np.mean(all_completion):.2%}  "
                        f"avg_len={np.mean(all_lengths):.1f}  "
                        f"avg_reward={np.mean(all_rewards):.2f}  "
                        f"(n={len(all_lengths)} episodes)"
                    )
                    print(
                        f"  Tracking errors (overall): "
                        f"wrist_pos={np.mean(all_wrist_pos):.4f} m  "
                        f"wrist_rot={np.mean(all_wrist_rot):.4f}  "
                        f"object_pos={np.mean(all_object_pos):.4f} m  "
                        f"object_rot={np.mean(all_object_rot):.4f}"
                    )
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
    teacher_exp_name: str,
    exp_name: str | None = None,
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    use_wandb: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    teacher_ckpt: int | None = None,
    env_args: Any = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
    resume_checkpoint: str | None = None,
) -> None:
    """Train BC policy to distill from teacher to student.

    Args:
        teacher_exp_name: Name of teacher experiment directory
        exp_name: Name of BC experiment directory
        show_viewer: Whether to show viewer
        num_envs: Number of parallel environments
        device: Device to use
        use_wandb: Whether to use wandb logging
        teacher_ckpt: Checkpoint number to load from teacher. If None, loads latest.
        env_args: Student environment configuration
        algo_cfg: BC algorithm configuration
        runner_args: Runner configuration
        resume_checkpoint: Path to checkpoint file to resume training from, or 'latest' to auto-find
    """
    # Create student environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=num_envs,
        device=device,
        args=env_args,
    )

    # Create BC runner
    runner, teacher_config_path = create_bc_runner_from_registry(
        env=env,
        teacher_exp_name=teacher_exp_name,
        exp_name=exp_name,
        teacher_ckpt=teacher_ckpt,
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
                # Fall back through older runs if the newest has no checkpoints yet.
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

        # Use the BC iteration counter from the checkpoint itself (already incremented)
        bc_algo = runner.algorithm
        assert isinstance(bc_algo, BC)
        start_iteration = bc_algo.current_iter
        print(f"Resuming training from iteration {start_iteration}")

        # The env curriculum continues from the teacher's iterations plus BC iterations
        env.training_step = bc_algo.training_step
        print(f"Synced environment training_step to {env.training_step}")

        # When resuming, use the same directory as the checkpoint
        resume_from_dir = ckpt_path.parent.parent  # Go up from checkpoints/ to experiment dir
        print(f"Resuming will continue in directory: {resume_from_dir}")

        # Update the runner's checkpoint directory to use the resumed location
        runner.save_dir = resume_from_dir
        runner.checkpoint_dir = resume_from_dir / "checkpoints"
        print(f"Checkpoints will be saved to: {runner.checkpoint_dir}")

    # Set up logging
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
        exp_name=exp_name or "bc_single_hand_retargeting",
        mode="online" if use_wandb and (not show_viewer) else "disabled",
    )

    # Save configuration files
    print("Saving configuration files to YAML...")
    config_dir = logger_folder / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_to_yaml(env_args, config_dir / "env_args.yaml")
    config_to_yaml(algo_cfg, config_dir / "algo_cfg.yaml")
    config_to_yaml(runner_args, config_dir / "runner_args.yaml")

    # Copy teacher config for reference
    shutil.copy(teacher_config_path, config_dir / "teacher_env_args.yaml")
    print(f"Saved teacher config to: {config_dir / 'teacher_env_args.yaml'}")
    print("Note: BC uses teacher observations for teacher policy, student observations for student.")

    # Train
    print("Starting BC training...")
    print(f"Teacher experiment: {teacher_exp_name}")
    print(f"Student observation dim: {env.actor_obs_dim}")

    def train() -> None:
        nonlocal runner, logger, start_iteration
        train_summary_info = runner.train(metric_logger=logger, start_iteration=start_iteration)
        print("Training completed successfully!")
        print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
        print(f"Total iterations: {train_summary_info['total_iterations']}.")
        print(f"Total steps: {train_summary_info['total_steps']}.")
        print(f"Final reward: {train_summary_info['final_reward']:.2f}.")

    try:
        if platform.system() == "Darwin" and show_viewer:
            import threading
            threading.Thread(target=train).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            train()
    except KeyboardInterrupt:
        pass


def _latest_run_dir(exp_name: str) -> Path:
    """Return the newest run folder under logs/<exp_name>/ that contains checkpoints.

    A resumed run keeps saving into its original folder but leaves an empty,
    newer-timestamped folder behind, so the newest folder alone can't be trusted.
    """
    base = Path(f"logs/{exp_name}")
    run_dirs = sorted((d for d in base.iterdir() if d.is_dir()), reverse=True) if base.is_dir() else []
    for run_dir in run_dirs:
        if any((run_dir / "checkpoints").glob("*.pt")):
            return run_dir
    raise FileNotFoundError(f"No run folder with checkpoints found under {base}/")


def _resolve_teacher_exp_dir(teacher_exp_name: str) -> Path:
    """Resolve teacher experiment directory from name.

    Supports both: "exp_name" (newest run with checkpoints) and "exp_name/timestamp" (exact dir).
    """
    direct_path = Path(f"logs/{teacher_exp_name}")
    if direct_path.exists() and (direct_path / "checkpoints").exists():
        return direct_path
    return _latest_run_dir(teacher_exp_name)


def _build_student_env_args(
    teacher_exp_dir: Path,
    env_overrides: dict[str, Any],
) -> SingleHandRetargetingEnvArgs:
    """Build student env args by loading teacher config and applying student overrides.

    Inherits object_config, reward_args, joint_mapping, etc. from the teacher,
    then applies the student-specific changes (obs terms, depth camera, etc.)
    from EnvArgsRegistry["single_hand_retargeting_bc_student"].
    """
    teacher_config_path = teacher_exp_dir / "configs" / "env_args.yaml"
    if not teacher_config_path.exists():
        raise FileNotFoundError(
            f"Teacher config not found at {teacher_config_path}. "
            "Make sure the teacher experiment has configs/env_args.yaml"
        )
    print(f"Loading teacher env config from: {teacher_config_path}")
    teacher_env_args = yaml_to_config(teacher_config_path, SingleHandRetargetingEnvArgs)

    # Student-specific overrides from the registry
    student_registry = EnvArgsRegistry["single_hand_retargeting_bc_student"]
    student_overrides = {
        "is_student": student_registry.is_student,
        "proprio_terms_student": student_registry.proprio_terms_student,
        "target_motion_terms_student": student_registry.target_motion_terms_student,
        "actor_obs_terms": student_registry.actor_obs_terms,
        "critic_obs_terms": student_registry.critic_obs_terms,
        "use_wrist_depth_camera": student_registry.use_wrist_depth_camera,
        "wrist_depth_camera_resolution": student_registry.wrist_depth_camera_resolution,
        "use_wrist_pointcloud": student_registry.use_wrist_pointcloud,
        "wrist_pointcloud_num_points": student_registry.wrist_pointcloud_num_points,
        "pointcloud_depth_noise_std": student_registry.pointcloud_depth_noise_std,
        "pointcloud_rect_dropout_num": student_registry.pointcloud_rect_dropout_num,
        "pointcloud_rect_dropout_width": student_registry.pointcloud_rect_dropout_width,
        "pointcloud_rect_dropout_height": student_registry.pointcloud_rect_dropout_height,
        "student_obs_delay_max": student_registry.student_obs_delay_max,
        "obs_noises": student_registry.obs_noises,
    }
    env_args = teacher_env_args.model_copy(update=student_overrides)

    # Apply any CLI env overrides on top
    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))
    return env_args


def main(
    teacher_exp_name: str = "ppo_single_hand_retargeting",
    exp_name: str | None = None,
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    use_wandb: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    teacher_ckpt: int | None = None,
    eval: bool = False,
    num_ckpt: int | None = None,
    resume_checkpoint: str | None = None,
    save_trajectory: bool = False,
    num_runs_per_traj: int = 5,
    **cfg_overrides: Any,
) -> None:
    """Entry point for BC training and evaluation.

    The student env config is built by loading the teacher's saved env_args.yaml
    and applying student-specific overrides (obs terms, depth camera, etc.).
    This automatically inherits object_config, reward_args, joint_mapping, etc.

    Args:
        teacher_exp_name: Name of teacher experiment directory (PPO-trained)
        exp_name: Name of BC experiment directory
        show_viewer: Whether to show the viewer
        num_envs: Number of parallel environments
        device: Device to use (cuda/cpu)
        use_wandb: Whether to use wandb logging
        wandb_project: Wandb project name
        wandb_entity: Wandb team/user name
        teacher_ckpt: Teacher checkpoint number. If None, loads latest.
        eval: If True, evaluate instead of train
        num_ckpt: BC checkpoint number for evaluation
        resume_checkpoint: Path to checkpoint file to resume training from, or 'latest' to auto-find
        save_trajectory: Save each eval episode to deploy/logs/<exp_name>/trajectories/
        num_runs_per_traj: Number of eval passes over every trajectory

    You can override configs via dot-notation, e.g.:
        --algo.lr=1e-4
        --runner.total_iterations=2000
        --resume_checkpoint=latest
        --resume_checkpoint=path/to/checkpoint.pt
    """
    # Bucket overrides
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

    # Resolve teacher experiment and build student env args from teacher config
    teacher_exp_dir = _resolve_teacher_exp_dir(teacher_exp_name)
    print(f"Teacher experiment directory: {teacher_exp_dir}")
    env_args = _build_student_env_args(teacher_exp_dir, env_overrides)

    algo_cfg = apply_overrides_generic(
        BC_SINGLE_HAND_RETARGETING_TEMPORAL, algo_overrides, prefixes=("cfgs.", "algo.")
    )

    runner_args = apply_overrides_generic(
        RUNNER_BC_SINGLE_HAND_RETARGETING, runner_overrides, prefixes=("cfgs.", "runner.")
    )

    seed_everything(env_args.gs_init_args.seed)
    print(f"Seed: {env_args.gs_init_args.seed}")

    if eval:
        print("Evaluation mode: Loading trained BC policy")
        assert exp_name is not None, "exp_name is required for evaluation"
        evaluate_policy(
            exp_name=exp_name,
            show_viewer=show_viewer,
            num_ckpt=num_ckpt,
            device=device,
            env_overrides=env_overrides,
            algo_cfg=algo_cfg,
            save_trajectory=save_trajectory,
            num_runs_per_traj=num_runs_per_traj,
        )
    else:
        print("Training mode: Starting BC policy distillation")
        train_policy(
            teacher_exp_name=teacher_exp_name,
            exp_name=exp_name,
            show_viewer=show_viewer,
            num_envs=num_envs,
            device=device,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            teacher_ckpt=teacher_ckpt,
            env_args=env_args,
            algo_cfg=algo_cfg,
            runner_args=runner_args,
            resume_checkpoint=resume_checkpoint,
        )


if __name__ == "__main__":
    fire.Fire(main)
