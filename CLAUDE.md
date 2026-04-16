# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl-agent is an extension of veRL for training LLM agents via reinforcement learning. It implements GiGPO (Group-in-Group Policy Optimization), a NeurIPS 2025 algorithm for fine-grained credit assignment in long-horizon agent training. The key innovation is a step-independent multi-turn rollout mechanism that allows customizable per-step input structures, enabling scalable training for tasks requiring 50+ steps.

## Build and Development Commands

```bash
# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e .[test]

# Run linting/formatting
ruff check .
ruff format .

# Run pre-commit hooks
pre-commit run --all-files

# Run tests
pytest -s tests/utils/cpu_tests/           # CPU-only tests
pytest -s tests/trainer/                   # Trainer tests
pytest -s -x tests/test_protocol.py        # Single test file
```

## Training Commands

Main entry point: `python3 -m verl.trainer.main_ppo`

Example training run (ALFWorld with GiGPO):
```bash
bash examples/gigpo_trainer/run_alfworld.sh
```

Training scripts follow this pattern:
1. Run data preprocessing (e.g., `examples.data_preprocess.prepare`)
2. Execute `verl.trainer.main_ppo` with Hydra config overrides

Key config parameters:
- `algorithm.adv_estimator`: gigpo, grpo, ppo, dapo, gspo, rloo
- `actor_rollout_ref.model.path`: Model path (e.g., Qwen/Qwen2.5-1.5B-Instruct)
- `env.env_name`: Environment (alfworld/AlfredTWEnv, webshop, search, sokoban)
- `env.rollout.n`: Group size for group-based algorithms
- `trainer.n_gpus_per_node`: Number of GPUs

## Architecture

### Core Components

**Multi-Turn Rollout** (`agent_system/multi_turn_rollout/rollout_loop.py`):
- `TrajectoryCollector` orchestrates agent-environment interaction
- Constructs per-step observations using memory module and prompts
- Computes rewards incrementally during episode execution

**Memory Management** (`agent_system/memory/`):
- `SimpleMemory` provides default history management
- Determines what context to include at each step (not full history concatenation)
- Extensible via `BaseMemory` interface

**Environment Manager** (`agent_system/environments/env_manager.py`):
- Wraps gym-style environments for parallelized rollouts
- Supports "group environments" where multiple rollouts share identical initial states
- Per-environment prompts in `agent_system/environments/prompts/`

**DataProto Protocol** (`verl/protocol.py`):
- Unified data format for tensor batches + non-tensor metadata
- Used across all module boundaries

**Distributed Training** (`verl/single_controller/ray/`):
- Ray-based orchestration with FSDP or Megatron parallelism
- Workers: ActorRolloutRefWorker, CriticWorker, RolloutWorker

**RL Algorithms** (`verl/trainer/ppo/core_algos.py`):
- Pluggable advantage estimators: GiGPO, GRPO, PPO, DAPO, GSPO, RLOO
- Critic-free algorithms (GiGPO, GRPO) don't require value network

### Directory Structure

- `verl/`: Core framework (models, trainers, workers, utilities)
- `agent_system/`: Agent-specific extensions
  - `environments/`: Environment packages, managers, and prompts
  - `memory/`: History management
  - `multi_turn_rollout/`: Episode execution loop
- `examples/`: Training scripts organized by algorithm
- `gigpo/`: GiGPO algorithm implementation

### Configuration System

Uses Hydra with YAML base configs in `verl/trainer/config/`. Override via CLI:
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    env.env_name=alfworld/AlfredTWEnv \
    trainer.n_gpus_per_node=2
```

## Adding New Environments

1. Create environment package in `agent_system/environments/env_package/{name}/`
   - Implement gym-style interface with multi-process support
2. Add prompts in `agent_system/environments/prompts/{name}.py`
3. Register manager in `agent_system/environments/env_manager.py` extending `EnvironmentManagerBase`

Reference implementation: WebShop (`env_package/webshop/`, `prompts/webshop.py`, env_manager.py L304)

## Environment Notes

- **WebShop**: Requires Python ≤3.10 (use separate conda env `verl-agent-webshop`)
- **Search-R1**: Requires separate retriever server (`conda env: retriever`) using ~6GB GPU RAM
- **Ray**: Pinned to <2.50.0 for compatibility
- **vLLM**: Recommend 0.11.0; LoRA requires ≥0.7.3