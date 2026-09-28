# DexTaG: Tactile-as-Guidance in Reinforcement Learning for Dexterous Manipulation

**[Project page](https://dextag.github.io/)**

[Han Yang](https://hanyangclarence.github.io/)<sup>1</sup>,
[Yian Wang](https://wangyian-me.github.io/)<sup>1</sup>,
[Yunlong Song](https://yunlong-song.com/)<sup>2</sup>,
[Zhenjia Xu](https://zhenjiaxu.com/)<sup>2</sup>,
[Chuang Gan](https://people.csail.mit.edu/ganchuang/)<sup>1</sup>

<sup>1</sup>UMass Amherst &nbsp;&nbsp; <sup>2</sup>Genesis AI

![DexTaG: tactile glove demonstrations guide RL training, and the policy runs on a real xArm7 with a WUJI hand](media/teaser.png)

This is the official code for DexTaG. DexTaG learns dexterous tool use, such as
picking up and reorienting a marker pen or a hammer, from demonstrations recorded
with a tactile motion-capture glove. The glove's tactile readings guide
reinforcement learning toward how the human actually held the object. The learned
policy is then distilled into a controller that runs on a real xArm7 with a WUJI
hand, without any tactile input.

This repository contains the simulation training, built on
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis):

- **Retargeter (teacher):** trained with reinforcement learning on all recorded
  demonstrations of one object, with rewards for matching the glove's tactile readings.
- **Student controller:** distilled from the retargeter. It follows a target object
  trajectory using only what the real robot can sense.

The simulated tactile sensor comes from our
[Genesis fork](https://github.com/hanyangclarence/Genesis), which the install step
pulls in automatically.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Data and checkpoints

**Demonstration data:** [dextag_trajectories.zip](https://drive.google.com/file/d/17gkLgZad7WZxZLt9SvIQ_VP64L53H0eP/view?usp=sharing)
(209 MB). Each demonstration is a preprocessed hand-object trajectory recorded with
the tactile glove, paired frame by frame with the glove's real 24x32 tactile map.
The zip holds these trajectories and each object's collision mesh, in
`marker_pen/pickles_filtered/` and `hammer_mixed/pickles_filtered/`. Unzip it
anywhere and pass the object's folder as `--env.object_config.trajectory_path`.

**Pretrained retargeters:**
[marker pen](https://drive.google.com/file/d/1TX5gRLLzj9hAsjoBLCIatOT3ONt2fQrl/view?usp=sharing)
and [hammer](https://drive.google.com/file/d/1PjSGp8IZ8ohfCW-SgcIfSr4WMnR1m4iN/view?usp=sharing)
(21 MB each). Unzip them at the repository root; they unpack to
`logs/mp0917_oldgf_s3/` and `logs/hm0917_oldgf_s3/`. Use those names as `--exp_name`
to evaluate them (step 2), or as `--teacher_exp_name` to distill a student (step 3).

## Usage

Run every command from the repository root, and point
`--env.object_config.trajectory_path` at an object's trajectory folder from the
demonstration data.

### 1. Train the retargeter

```bash
python scripts/policy/run_ppo_single_hand_retargeting.py \
    --exp_name=my_run --object_name=marker_pen \
    --env.object_config.trajectory_path=/path/to/pickles_filtered \
    --num_envs=3072 --runner.total_iterations=15000
```

`--object_name` is `marker_pen` or `hammer`. Checkpoints are written to
`logs/<exp_name>/`, and `--resume_checkpoint=latest` resumes a run.

### 2. Evaluate and render

```bash
python scripts/policy/run_ppo_single_hand_retargeting.py \
    --exp_name=my_run --object_name=marker_pen \
    --env.object_config.trajectory_path=/path/to/pickles_filtered \
    --eval=True --save_trajectory=True --num_runs_per_traj=1 --use_wandb=False
```

This prints success and completion rates and saves rollouts to
`deploy/logs/<exp_name>/trajectories/`. To render one as a scene video and a
tactile-map video:

```bash
python scripts/visualization/render_rollout_with_tactile_frames.py \
    --npz deploy/logs/my_run/trajectories/episode_0000_<id>.npz \
    --output-dir /tmp/viz --object-mesh-dir /path/to/pickles_filtered \
    --object-id marker_pen_scanned
```

### 3. Distill the student controller

```bash
python scripts/policy/run_bc_single_hand_retargeting.py \
    --teacher_exp_name=my_run --exp_name=my_run_bc \
    --env.object_config.trajectory_path=/path/to/pickles_filtered --num_envs=2048
```

The student's environment is built from the retargeter's saved configuration
(object, rewards, and hand setup), so those settings don't need to be repeated.
To evaluate the student, rerun the same command with
`--eval=True --save_trajectory=True --num_runs_per_traj=1 --use_wandb=False`.

## Acknowledgements

This code is built on [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) and
started from [GenesisPlayground](https://github.com/yun-long/GenesisPlayground).

## License

MIT. See [LICENSE](LICENSE).
