# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GR00T-WholeBodyControl (SONIC) — humanoid robot whole-body control from NVIDIA GEAR Lab. Three main subsystems:

- **gear_sonic/** — Python VR teleoperation stack (PICO XR headset body/hand tracking, ZMQ messaging, IK solvers, MuJoCo sim)
- **gear_sonic_deploy/** — C++ real-time inference for hardware deployment (ONNX/TensorRT, DDS motor commands, multi-threaded 4-loop architecture)
- **decoupled_wbc/** — Decoupled whole-body controller (RL lower body + IK upper body), used in GR00T N1.5/N1.6

## Common Commands

### Install
```bash
pip install -e decoupled_wbc/          # core; use [full] or [dev] for extras
pip install -e gear_sonic/             # core; use [teleop] or [sim] for extras
```

### Lint & Format
```bash
make run-checks          # isort + black + ruff (check only)
make format              # isort + black (auto-fix)
./lint.sh                # check mode
./lint.sh --fix          # auto-fix mode
```

### Tests
```bash
pytest decoupled_wbc/tests/                    # all tests
pytest decoupled_wbc/tests/test_foo.py         # single file
pytest decoupled_wbc/tests/test_foo.py::test_bar  # single test
```

### C++ Build (gear_sonic_deploy)
```bash
cd gear_sonic_deploy && mkdir -p build && cd build && cmake .. && make
```

### Docs
```bash
sphinx-build -b html docs/source docs/build/html
```

## Code Style

- Python 3.10 required
- Black: 100 char line length
- Ruff: 115 char line length, rules E/F/I
- isort: black profile
- `external_dependencies/` is excluded from all linters

## Architecture Notes

### gear_sonic_deploy (C++ inference)
Four real-time threads: input (100 Hz), control/policy (50 Hz), planner (10 Hz), command writer (500 Hz). Uses Unitree SDK v2 DDS for motor commands. Supports keyboard, gamepad, ZMQ, and ROS2 input types.

### gear_sonic (Python teleop)
ZMQ-based pipeline: PICO XR headset → XRoboToolkit SDK → body/hand tracking → IK solvers → robot commands. Entry point: `gear_sonic/scripts/pico_manager_thread_server.py`.

### decoupled_wbc (Python control)
Hydra-driven config system. Tyro-based CLI for deployment configs (`decoupled_wbc/control/main/teleop/configs/configs.py`). Robot models defined in `control/robot_model/`, environments in `control/envs/g1/`. Data collection via `control/main/teleop/run_g1_data_exporter.py`.

### Robot support
Currently targets Unitree G1 humanoid (29 DOF + optional Inspire/Dex3 hands).
