# FORGE: Forward-learning Optimized RL with Guided Experience

FORGE is a training framework that combines parallel exploration, experience evolution, and experience-guided learning for efficient multi-turn agent training.

## Overview

FORGE addresses key limitations of existing approaches:

| Approach | Issue | FORGE Solution |
|----------|-------|----------------|
| GRPO/GiGPO | Sparse reward signal only | Add dense experience-guided signal |
| SFT during sleep | Doesn't improve over RL alone | Forward learning (library evolution, no SFT) |
| Token efficiency focus | Ignores GPU parallelism | Time efficiency via parallel sampling |

### Key Components

1. **Step-Level Experience Retrieval**: Retrieve relevant experiences at EACH step based on current observation (vs. task-level retrieval in single-turn methods)
2. **Observation Clustering (GiGPO-Inspired)**: Group similar observations across trajectories for efficient experience sharing - uses the same similarity metric as GiGPO's step grouping
3. **Experience Evolution**: Forward-learning library updates without gradient-based fine-tuning
4. **Experience Library**: Hierarchical storage with Golden Zone (successes) and Warning Zone (failures)
5. **Parallel Exploration**: Sample G trajectories in parallel (GPU parallelism makes G samples ≈ 1 sample in wall-clock time)

## Quick Start

```bash
# ALFWorld
bash examples/forge/run_alfworld.sh

# Sokoban (visual)
bash examples/forge/run_sokoban.sh

# WebShop
bash examples/forge/run_webshop.sh
```

## Configuration Reference

### Algorithm Configuration

```yaml
algorithm:
  adv_estimator: forge  # Use FORGE advantage estimator
  gamma: 0.95           # Discount factor for step rewards
  use_kl_in_reward: False  # KL penalty in rewards (usually disabled)
```

### FORGE-Specific Configuration

All FORGE configurations are under `algorithm.forge`:

#### Advantage Computation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_step_rewards` | bool | `True` | Enable GiGPO-compatible multi-level advantages. When True, combines episode-level and step-level advantages. |
| `step_advantage_w` | float | `1.0` | Weight for step-level advantages in combined score: `A = A_episode + step_advantage_w * A_step` |
| `mode` | str | `"mean_norm"` | Normalization mode. Options: `"mean_norm"` (subtract mean) or `"mean_std_norm"` (subtract mean, divide by std) |
| `norm_by_std` | bool | `True` | Whether to normalize advantages by standard deviation |
| `enable_similarity` | bool | `False` | Enable similarity-based step grouping (uses text similarity instead of exact match) |
| `similarity_thresh` | float | `0.95` | Threshold for similarity-based grouping (0.0-1.0) |

#### Experience Library

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `golden_capacity_per_level` | int | `100` | Max experiences per level in Golden Zone. Levels: `principle`, `method`, `example` |
| `warning_capacity_per_level` | int | `50` | Max experiences per level in Warning Zone. Levels: `mistake`, `pattern`, `diagnostic` |
| `use_embeddings` | bool | `False` | Use embedding-based retrieval (requires sentence-transformers) |
| `success_threshold` | float | `0.5` | Score threshold for classifying trajectories as successful |

#### Experience Evolution

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `similarity_threshold` | float | `0.85` | Threshold for merging similar experiences (avoids redundancy) |

#### Step-Level Retrieval (GiGPO-Inspired Observation Clustering)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retrieval_mode` | str | `"observation"` | How to retrieve experiences: `"observation"` (current state), `"task"` (task type only), `"hybrid"` (both), `"clustered"` (GiGPO-style clustering) |
| `enable_clustering` | bool | `True` | Use observation clusters for efficient experience sharing |
| `obs_similarity_threshold` | float | `0.85` | Threshold for observation similarity (matches GiGPO's step grouping threshold) |
| `top_k_golden` | int | `2` | Number of golden experiences to retrieve per step |
| `top_k_warning` | int | `1` | Number of warning experiences to retrieve per step |
| `max_experience_length` | int | `800` | Maximum characters for experience text in prompt |

**Key Insight**: This uses the same similarity metric as GiGPO's `build_step_group()` function. If two observations would be grouped together for advantage normalization in GiGPO, they will also share experiences in FORGE.

#### Self-Distillation (Optional, Disabled by Default)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `divergence_type` | str | `"jsd"` | Divergence for distillation. Options: `"kl"`, `"reverse_kl"`, `"jsd"` (Jensen-Shannon) |
| `distill_temperature` | float | `1.0` | Softmax temperature for distillation (higher = softer distributions) |
| `top_k_golden` | int | `3` | Number of golden experiences to retrieve for teacher prompt |
| `top_k_warning` | int | `2` | Number of warning experiences to retrieve for teacher prompt |
| `max_distill_weight` | float | `0.5` | Maximum weight for distillation loss component |

#### Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `library_save_freq` | int | `10` | Save library checkpoint every N training steps (0 = disabled) |
| `checkpoint_dir` | str | `"checkpoints/forge"` | Directory for library checkpoints |

### Environment Configuration

```yaml
env:
  env_name: alfworld/AlfredTWEnv  # Environment name
  seed: 0                          # Random seed
  max_steps: 50                    # Max steps per episode
  rollout:
    n: 8                           # Group size (parallel trajectories per task)
  resources_per_worker:
    num_cpus: 0.1                  # CPU allocation per env worker
```

### Model Configuration

```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-4B            # Model path
    use_remove_padding: True       # Remove padding for efficiency
    enable_gradient_checkpointing: True  # Memory optimization

  actor:
    optim:
      lr: 1e-6                     # Learning rate
    ppo_mini_batch_size: 32        # Mini-batch size for PPO updates
    ppo_micro_batch_size_per_gpu: 4  # Micro-batch per GPU
    use_kl_loss: True              # KL loss regularization
    kl_loss_coef: 0.01             # KL loss coefficient
    kl_loss_type: low_var_kl       # KL loss type
    use_invalid_action_penalty: True   # Penalize invalid actions
    invalid_action_penalty_coef: 0.1   # Invalid action penalty coefficient

  rollout:
    tensor_model_parallel_size: 2  # TP size for generation
    gpu_memory_utilization: 0.6    # vLLM GPU memory fraction
    val_kwargs:
      temperature: 0.4             # Validation temperature
      do_sample: True              # Enable sampling for validation
```

## Experience Library Structure

### Golden Zone (Successful Strategies)

```
Golden Zone
├── Principles (high-level guidelines)
│   └── "Examine objects before interacting with them"
├── Methods (reasoning patterns)
│   └── "Action sequence: go to desk -> pick up key -> go to door"
└── Examples (concrete facts)
    └── "Key decision: Open the drawer before looking inside"
```

### Warning Zone (Failure Patterns)

```
Warning Zone
├── Mistakes (actions to avoid)
│   └── "Avoid repeating 'go to desk' multiple times"
├── Patterns (recurring failures)
│   └── "If nothing happens, try a different approach"
└── Diagnostics (root cause insights)
    └── "Task failed due to exceeding step limit"
```

## Hyperparameter Tuning Guide

### For Different Environments

| Environment | Recommended Settings |
|-------------|---------------------|
| **ALFWorld** | `max_steps=50`, `gamma=0.95`, `step_advantage_w=1.0` |
| **WebShop** | `max_steps=15`, `gamma=0.95`, `step_advantage_w=1.0` |
| **Sokoban** | `max_steps=15`, `gamma=0.95`, `step_advantage_w=1.0` |
| **Search** | `max_steps=10`, `gamma=0.90`, `step_advantage_w=0.5` |

### Mode Selection

- **`mean_norm`** (default): Subtracts group mean only. More stable, works well for most cases.
- **`mean_std_norm`**: Subtracts mean and divides by std. Can help when reward scales vary significantly across tasks.

### Library Capacity

- **Small tasks** (< 100 training samples): `golden_capacity_per_level=50`
- **Medium tasks** (100-1000 samples): `golden_capacity_per_level=100` (default)
- **Large tasks** (> 1000 samples): `golden_capacity_per_level=200`

### Success Threshold

Adjust based on your reward distribution:
- **Binary rewards (0/1)**: `success_threshold=0.5`
- **Continuous rewards [0, 1]**: `success_threshold=0.7` (stricter)
- **Negative possible**: `success_threshold=0.0`

## Metrics

FORGE logs the following metrics:

| Metric | Description |
|--------|-------------|
| `forge/reward_mean` | Mean episode reward |
| `forge/reward_std` | Std of episode rewards |
| `forge/positive_ratio` | Fraction of successful trajectories |
| `forge/strategies_added` | New golden experiences added this step |
| `forge/warnings_added` | New warning experiences added this step |
| `forge/library_golden_size` | Total golden zone size |
| `forge/library_warning_size` | Total warning zone size |
| `forge/library_total_size` | Total library size |

## Comparison with Other Algorithms

| Feature | GRPO | GiGPO | Cyclic-FLEX | FORGE |
|---------|------|-------|-------------|-------|
| Episode advantages | ✓ | ✓ | ✓ | ✓ |
| Step advantages | ✗ | ✓ | ✓ | ✓ |
| Experience library | ✗ | ✗ | ✓ | ✓ |
| SFT during sleep | ✗ | ✗ | ✓ | ✗ |
| Forward learning | ✗ | ✗ | ✗ | ✓ |
| Critic-free | ✓ | ✓ | ✓ | ✓ |

## Troubleshooting

### Library Not Growing

If `forge/library_total_size` stays at 0:
- Check `success_threshold` - it may be too high
- Verify trajectories have non-zero rewards
- Check logs for "strategies_added" counts

### Memory Issues

If running out of GPU memory:
- Reduce `golden_capacity_per_level`
- Disable `use_embeddings`
- Reduce `ppo_micro_batch_size_per_gpu`

### Slow Training

If training is slower than expected:
- Reduce `library_save_freq` (or set to 0)
- Disable `enable_similarity` for step grouping
- Use smaller model or increase tensor parallelism

## Advanced Usage

### Loading Pre-trained Library

```python
from verl.trainer.ppo.forge import ExperienceLibrary

library = ExperienceLibrary()
library.load("path/to/library_checkpoint.pkl")
```

### Custom Experience Extraction

Override the `ExperienceEvolver` class for domain-specific extraction:

```python
from verl.trainer.ppo.forge import ExperienceEvolver

class CustomEvolver(ExperienceEvolver):
    def extract_strategies(self, trajectory, task_type):
        # Custom logic for your domain
        ...
```

## References

- **FLEX**: Forward Learning from Experience (experience library evolution)
- **OPSD**: On-Policy Self-Distillation (dense token-level signal)
- **GiGPO**: Group-in-Group Policy Optimization (multi-level advantages)

## Files

```
examples/forge/
├── README.md           # This file
├── run_alfworld.sh     # ALFWorld training script
├── run_sokoban.sh      # Sokoban training script
└── run_webshop.sh      # WebShop training script

verl/trainer/ppo/forge/
├── __init__.py              # Module exports
├── experience_library.py    # Golden/Warning zone storage
├── experience_evolution.py  # Forward learning updates
├── self_distillation.py     # Experience-guided distillation
└── core_forge.py            # Advantage computation
```
