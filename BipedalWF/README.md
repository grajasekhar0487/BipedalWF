# BipedalWF

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered tasks are:
- `Bipedalwf-Balance-WF-Tron2A` — balance and velocity-command locomotion for the WF-TRON2A robot.


## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project's `pyproject.toml` uses editable relative paths to an Isaac Lab source checkout. Update
`[tool.uv.sources]` if either directory moves.

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

Train with the native Newton MJWarp visualizer (`newton_gl`):

```bash
uv run isaaclab train --task Bipedalwf-Balance-WF-Tron2A --rl_library rsl_rl \
  --num_envs 16 --device cuda --viz newton_gl physics=newton_mjwarp
uv run isaaclab play --task Bipedalwf-Balance-WF-Tron2A --rl_library rsl_rl \
  --checkpoint latest --device cuda --viz newton_gl physics=newton_mjwarp
```

To use the Isaac Sim Kit viewport with PhysX, install the Isaac Sim extra and select both the physics preset and Kit
visualizer:

```bash
uv run --extra isaacsim isaaclab train --task Bipedalwf-Balance-WF-Tron2A --rl_library rsl_rl \
  --num_envs 16 --device cuda --viz kit physics=isaacsim_physx
uv run --extra isaacsim isaaclab play --task Bipedalwf-Balance-WF-Tron2A --rl_library rsl_rl \
  --checkpoint latest --device cuda --viz kit physics=isaacsim_physx
```

For Newton physics displayed in the Kit viewport, install the Isaac Sim extra and select `--viz kit` with
`physics=newton_mjwarp`. The standalone `newton_gl` viewer does not require Isaac Sim. Replace `cuda` with `cpu` on
either backend when needed; rendering requires a graphical desktop. The examples use 16 environments to keep the live
viewport manageable. Increase `--num_envs` for faster training without rendering. The task samples forward/reverse,
lateral, and yaw-rate commands. Early checkpoints may fall frequently; let PPO train for multiple iterations and use
the generated policy checkpoint for playback.

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `isaaclab list_envs`.

## Project structure

Task families live in `src/BipedalWF/tasks`. Within a manager-based family, shared MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/<robot>`. The WF-TRON2A USD and its
referenced description assets live under `src/BipedalWF/assets/data/robots/WF_TRON2A`.

## Project assets

Keep project-owned USD files and related data under `src/BipedalWF/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from BipedalWF.assets import BIPEDALWF_ASSETS_DIR

robot_usd_path = BIPEDALWF_ASSETS_DIR / "robots" / "my_robot.usd"
```

Keep asset configuration in the `BipedalWF.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
