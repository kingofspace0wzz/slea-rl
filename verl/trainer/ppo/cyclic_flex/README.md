# Cyclic-FLEX: Wake-Sleep Framework for Continuous Agent Evolution

This module implements the Cyclic-FLEX algorithm for training LLM agents via reinforcement learning with continuous adaptation capabilities.

## Overview

Cyclic-FLEX addresses the "Memory-Capacity Gap" in agent training by combining:

- **Wake Phase**: GRPO/GiGPO-style inference-time adaptation with experience accumulation
- **Sleep Phase**: Periodic weight consolidation via LoRA fine-tuning

This biologically-inspired wake-sleep cycle enables continuous agent evolution without hitting context length limitations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WAKE PHASE (Per-Step)                     │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │   Rollout    │→ │   Advantage   │→ │    Experience    │  │
│  │   Collect    │  │   Compute     │  │     Buffer       │  │
│  └──────────────┘  └───────────────┘  └──────────────────┘  │
└────────────────────────────│────────────────────────────────┘
                             ▼ Trigger: buffer.is_full()
┌─────────────────────────────────────────────────────────────┐
│                   SLEEP PHASE (Periodic)                     │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │   Dataset    │→ │     LoRA      │→ │     Buffer       │  │
│  │   Prepare    │  │   Training    │  │     Flush        │  │
│  └──────────────┘  └───────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
verl/trainer/ppo/cyclic_flex/
├── __init__.py              # Module exports
├── README.md                # This file
├── experience_buffer.py     # Wake phase experience storage
├── core_cyclic_flex.py      # Advantage computation functions
├── sleep_trainer.py         # LoRA-based weight consolidation
└── reflection_generator.py  # Strategic reflection generation
```

## Components

### 1. ExperienceBuffer (`experience_buffer.py`)

Stores successful experiences during the wake phase for later consolidation.

**Key Features:**
- FIFO eviction when capacity is reached
- Optional semantic retrieval via sentence-transformers
- Converts experiences to SFT dataset format for sleep phase
- Checkpoint save/load for persistence

**Usage:**
```python
from verl.trainer.ppo.cyclic_flex import ExperienceBuffer

buffer = ExperienceBuffer(
    capacity=100,
    min_score_threshold=0.5,
    use_semantic_retrieval=True,
)

# Add experience
buffer.add(
    query="Navigate to kitchen",
    response="go north\ntake lamp",
    score=1.0,
    uid="task_001",
    traj_uid="traj_001",
    reflection="Key: Check inventory before navigating.",
)

# Check if sleep phase should trigger
if buffer.is_full():
    dataset = buffer.to_sft_dataset(tokenizer)
    # ... run sleep phase
    buffer.flush()
```

### 2. Core Advantage Computation (`core_cyclic_flex.py`)

Implements GRPO-style group-relative advantage estimation with optional GiGPO step-level credit assignment.

**Functions:**

- `compute_cyclic_flex_advantage()`: Episode-level GRPO-style advantages
- `compute_cyclic_flex_with_step_rewards()`: Combined episode + step-level advantages (GiGPO-compatible)
- `compute_reflection_candidates()`: Identifies winner/loser pairs for contrastive reflection

**Usage:**
```python
from verl.trainer.ppo.cyclic_flex.core_cyclic_flex import compute_cyclic_flex_advantage

advantages, returns = compute_cyclic_flex_advantage(
    token_level_rewards=rewards,      # (batch_size, seq_len)
    response_mask=mask,               # (batch_size, seq_len)
    index=uids,                       # Group IDs
    traj_index=traj_uids,             # Trajectory IDs
)
```

### 3. SleepTrainer (`sleep_trainer.py`)

Handles weight consolidation via LoRA-based supervised fine-tuning.

**Key Features:**
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Configurable training parameters (epochs, learning rate, etc.)
- Optional weight merging after training
- Checkpoint management for sleep cycles

**Usage:**
```python
from verl.trainer.ppo.cyclic_flex import SleepTrainer

trainer = SleepTrainer(
    lora_rank=16,
    lora_alpha=32,
    learning_rate=2e-5,
    epochs=2,
)

# Prepare dataloader from experience buffer
dataloader = trainer.prepare_dataloader(buffer, tokenizer)

# Run consolidation
metrics = trainer.run_consolidation(
    model=model,
    dataloader=dataloader,
    device="cuda",
)
```

### 4. ReflectionGenerator (`reflection_generator.py`)

Generates strategic reflections from trajectory comparisons.

**Key Features:**
- Environment-specific prompts (ALFWorld, WebShop, Search, Sokoban)
- Simple heuristic mode for fast reflection generation
- LLM-based mode for detailed strategic insights

**Usage:**
```python
from verl.trainer.ppo.cyclic_flex import ReflectionGenerator

generator = ReflectionGenerator(
    use_llm=False,
    env_name="alfworld",
)

reflections = generator.generate_reflections(
    batch=batch,
    scores=scores,
    tokenizer=tokenizer,
    model=model,
)
```

## Configuration

Add to your training config (e.g., via Hydra):

```yaml
algorithm:
  adv_estimator: cyclic_flex

  cyclic_flex:
    # Wake Phase
    use_step_rewards: true        # Enable GiGPO-style step advantages
    step_advantage_w: 1.0         # Weight for step-level advantages
    retrieval_k: 5                # Experiences to retrieve for ICL

    # Sleep Phase Triggers
    buffer_capacity: 100          # Triggers sleep when full
    consolidation_freq: 0         # Steps between consolidation (0 = buffer-triggered only)
    min_positive_experiences: 10  # Minimum experiences for sleep
    max_cycles: 10                # Maximum sleep cycles

    # LoRA Training
    sleep_epochs: 2
    lora_rank: 16
    lora_alpha: 32
    lora_dropout: 0.05
    learning_rate: 2e-5

    # Reflection
    use_llm_reflection: false     # Use LLM for reflection generation
    reflection_env: "default"     # Environment for reflection prompts
```

## Integration with verl-agent

Cyclic-FLEX is integrated into the verl-agent training pipeline via:

1. **AdvantageEstimator enum**: Added `CYCLIC_FLEX` option in `ray_trainer.py`
2. **Trainer integration**: Wake/sleep logic in `RayPPOTrainer.fit()`
3. **Config system**: Full configuration in `ppo_trainer.yaml`

To use Cyclic-FLEX:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=cyclic_flex \
    algorithm.cyclic_flex.buffer_capacity=100 \
    algorithm.cyclic_flex.use_step_rewards=true
```

Or use the provided example scripts:

```bash
bash examples/cyclic_flex_trainer/run_alfworld.sh
```

## Metrics

During training, Cyclic-FLEX logs the following metrics:

### Wake Phase
- `cyclic_flex/buffer_size`: Current number of experiences in buffer
- `cyclic_flex/buffer_utilization`: buffer_size / capacity
- `cyclic_flex/experiences_added`: Experiences added this step

### Sleep Phase
- `cyclic_flex/sleep_triggered`: 1 when sleep phase runs
- `cyclic_flex/sleep_cycle`: Current sleep cycle number
- `cyclic_flex/sleep_num_experiences`: Experiences consolidated
- `sleep/train_loss`: Average training loss during sleep
- `sleep/final_loss`: Final training loss

## Theoretical Background

### Dual-State MDP Formulation

Cyclic-FLEX models the agent state as a tuple:

$$S_t = (x_t, \theta_t, E_t)$$

Where:
- $x_t$: Current observation
- $\theta_t$: Model weights
- $E_t$: Experience buffer contents

### Wake Phase Objective

During wake, we accumulate experiences while optimizing:

$$\max_\pi \mathbb{E}[\sum_t r_t | \theta, E]$$

Using GRPO-style group-relative advantages for policy updates.

### Sleep Phase Objective

During sleep, we consolidate experiences via SFT:

$$\min_\theta \mathbb{E}_{(x,\epsilon) \sim E}[-\log \pi_\theta(\epsilon | x)]$$

Where $\epsilon$ is the reflection/strategy, forcing internalization of reasoning patterns.

## References

- Training-Free GRPO: [arXiv:2510.08191](https://arxiv.org/abs/2510.08191)
- GiGPO (verl-agent): NeurIPS 2025
- Early Experience: [arXiv:2511.06449](https://arxiv.org/abs/2511.06449)
- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

## Testing

Run unit tests:

```bash
pytest tests/trainer/ppo/cyclic_flex/ -v
```

Or with unittest:

```bash
python -m unittest discover tests/trainer/ppo/cyclic_flex/ -v
```

## File Locations

| Component | Path |
|-----------|------|
| Core module | `verl/trainer/ppo/cyclic_flex/` |
| Configuration | `verl/trainer/config/ppo_trainer.yaml` |
| Example scripts | `examples/cyclic_flex_trainer/` |
| Unit tests | `tests/trainer/ppo/cyclic_flex/` |
