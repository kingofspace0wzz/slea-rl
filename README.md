<h1 align="center">SLEA-RL</h1>

<h3 align="center">
<b>Step-Level Experience Augmented RL for Agentic Training</b>
<br>
<sub><i>formerly known as FORGE</i></sub>
</h3>

<p align="center">
  <a href="https://github.com/kingofspace0wzz/slea-rl">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
</p>

`SLEA-RL` is a reinforcement learning framework for training LLM agents with **step-level experience augmentation**. It maintains a hierarchical experience library that grows via forward learning (no SFT), retrieves relevant experiences at every step of a multi-turn rollout, and combines episode- and step-level advantages for fine-grained credit assignment on long-horizon agent tasks.

The framework is built on top of [`verl-agent`](https://github.com/langfengQ/verl-agent) and [`veRL`](https://github.com/volcengine/verl), and inherits all of their infrastructure: step-independent multi-turn rollouts, parallelized gym-style environments, pluggable memory modules, and FSDP/Megatron distributed training.

## Key Ideas

| Problem | SLEA-RL |
|---|---|
| GRPO/GiGPO rely on sparse reward signal only | Adds a dense, experience-guided signal at every step |
| SFT-during-sleep does not improve over RL alone | Forward learning library evolution — no gradient updates to the library |
| Token-efficiency focus ignores GPU parallelism | Time-efficient parallel sampling (G trajectories ≈ 1 in wall-clock) |
| Task-level retrieval misses per-step context | Step-level experience retrieval via GiGPO-style observation clustering |

### Components

1. **Step-Level Experience Retrieval** — relevant experiences are retrieved based on the *current observation* at each step, not just the task.
2. **Observation Clustering** — similar observations across trajectories are grouped using the same similarity metric as GiGPO's step grouping, so two steps that would share an advantage also share experience.
3. **Experience Evolution** — the library updates via forward learning (no fine-tuning): merging, deduplication, and pruning of the Golden/Warning zones.
4. **Hierarchical Experience Library** — Golden Zone (successes: principles → methods → examples) and Warning Zone (failures: mistakes → patterns → diagnostics).
5. **Parallel Exploration** — groups of G trajectories are sampled in parallel; GPU parallelism makes this nearly free in wall-clock time.
6. **Optional Self-Distillation** — use the library as privileged information for JSD/KL distillation against the student policy.

# Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Supported Environments](#supported-environments)
- [Other RL Algorithms](#other-rl-algorithms)
- [FAQ](#faq)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

# Installation

## Core install
```bash
conda create -n slea-rl python==3.12 -y
conda activate slea-rl

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

## Supported Environments
> ⚠️ **Important:**
> Install each environment in its *own* conda environment to avoid package conflicts.

### 1. ALFWorld
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
alfworld-download -f
```

### 2. WebShop
WebShop requires Python ≤3.10, so use a dedicated environment:
```bash
conda create -n slea-rl-webshop python==3.10 -y
conda activate slea-rl-webshop

cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all

cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```

### 3. Search
```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2

cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```

Set up the retrieval server in its own environment:
```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz

bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log
```

### 4. Sokoban
```bash
pip install matplotlib
pip install gym==0.26.2
pip install gym_sokoban==0.0.6
```

### 5. Gym Cards
```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```

### 6. AppWorld (Experimental)
```bash
cd repo_root/
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
pip install -e .

conda create -n appworld python=3.12 -y
conda activate appworld
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
appworld download data
```

# Quick Start

Run SLEA-RL on any supported environment:

```bash
# ALFWorld
bash examples/forge/run_alfworld.sh

# Sokoban (visual)
bash examples/forge/run_sokoban.sh

# WebShop
bash examples/forge/run_webshop.sh

# Search
bash examples/forge/run_search.sh
```

> The script directory `examples/forge/` is kept for compatibility with the original FORGE naming; the underlying algorithm is SLEA-RL.

# Configuration

The advantage estimator is selected via Hydra:

```yaml
algorithm:
  adv_estimator: forge      # SLEA-RL advantage estimator
  gamma: 0.95
  use_kl_in_reward: False

  forge:
    use_step_rewards: True     # combine episode- and step-level advantages
    step_advantage_w: 1.0
    mode: "mean_norm"          # or "mean_std_norm"
    enable_similarity: False
    similarity_thresh: 0.95

    # Experience library
    golden_capacity_per_level: 100    # principle / method / example
    warning_capacity_per_level: 50    # mistake / pattern / diagnostic
    use_embeddings: False
    success_threshold: 0.5

    # Step-level retrieval
    retrieval_mode: "observation"     # observation / task / hybrid / clustered
    enable_clustering: True
    obs_similarity_threshold: 0.85
    top_k_golden: 2
    top_k_warning: 1
    max_experience_length: 800

    # Self-distillation (optional)
    divergence_type: "jsd"            # kl / reverse_kl / jsd
    distill_temperature: 1.0
    max_distill_weight: 0.5

    # Checkpointing
    library_save_freq: 10
    checkpoint_dir: "checkpoints/forge"
```

Full reference: [`examples/forge/README.md`](./examples/forge/README.md).

## Recommended settings per environment

| Environment | max_steps | gamma | step_advantage_w |
|---|---|---|---|
| ALFWorld | 50 | 0.95 | 1.0 |
| WebShop  | 15 | 0.95 | 1.0 |
| Sokoban  | 15 | 0.95 | 1.0 |
| Search   | 10 | 0.90 | 0.5 |

## Metrics logged during training

| Metric | Description |
|---|---|
| `forge/reward_mean` / `reward_std` | Episode reward statistics |
| `forge/positive_ratio` | Fraction of successful trajectories |
| `forge/strategies_added` / `warnings_added` | New experiences per step |
| `forge/library_golden_size` / `library_warning_size` / `library_total_size` | Library growth |

# Other RL Algorithms

Inherited from `verl-agent`, SLEA-RL ships with the following baselines for comparison:

| Algorithm | Entry script |
|---|---|
| GiGPO | `examples/gigpo_trainer/run_alfworld.sh` |
| GiGPO (dynamic) | `examples/gigpo_dynamic_trainer/run_alfworld.sh` |
| GRPO | `examples/grpo_trainer/run_alfworld.sh` |
| PPO | `examples/ppo_trainer/run_alfworld.sh` |
| DAPO | `examples/dapo_trainer/run_alfworld.sh` |
| GSPO | `examples/gspo_trainer/run_alfworld.sh` |
| RLOO | `examples/rloo_trainer/run_alfworld.sh` |

LoRA, Qwen3-VL, and prompt-only GPT-4o agent baselines are also supported — see the corresponding scripts in `examples/`.

# FAQ

### 1. Customize the memory module
The default history manager is [`SimpleMemory`](./agent_system/memory/memory.py), invoked in [`env_manager.py`](./agent_system/environments/env_manager.py) (`build_text_obs()`). Subclass `BaseMemory` to add dynamic summarization, selective retention, or external knowledge.

### 2. Data preparation
For most environments (ALFWorld, WebShop, Sokoban, Search), data only tags the modality — `""` for text-only or `"<image>"` for vision-language. Agent input comes from `env.step()` feedback. Search-R1 passes tasks via `env_kwargs` in [`rollout_loop.py`](./agent_system/multi_turn_rollout/rollout_loop.py#L301).

### 3. Customize prompts
Prompts live in [`agent_system/environments/prompts/`](./agent_system/environments/prompts/). WebShop prompt template:

```
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}. Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}.
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: [{available_actions}].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation inside <think> </think> tags,
then choose an admissible action and wrap it in <action> </action> tags.
```

### 4. Add a new environment
1. Create your environment package (gym-style + multi-process) in [`agent_system/environments/env_package/`](./agent_system/environments/env_package/).
2. Add prompts in [`agent_system/environments/prompts/`](./agent_system/environments/prompts/).
3. Register a manager in [`env_manager.py`](./agent_system/environments/env_manager.py) subclassing [`EnvironmentManagerBase`](./agent_system/environments/base.py#L19).

Reference: the WebShop integration.

# Acknowledgement

SLEA-RL is built on top of [`verl-agent`](https://github.com/langfengQ/verl-agent) (Feng et al., NeurIPS 2025) and [`veRL`](https://github.com/volcengine/verl). The supported environments are adapted from [ALFWorld](https://github.com/alfworld/alfworld), [Sokoban](https://github.com/mpSchrader/gym-sokoban), [SkyRL-Gym](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [Gym Cards](https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards), [WebShop](https://github.com/princeton-nlp/WebShop), and [AppWorld](https://github.com/stonybrooknlp/appworld). The observation-clustering similarity metric is reused from GiGPO's step grouping.

# Citation

If you use SLEA-RL in your research, please cite this repository alongside the upstream GiGPO and verl-agent work:

```bibtex
@misc{slea-rl-2026,
  title  = {SLEA-RL: Step-Level Experience Augmented RL for Agentic Training},
  author = {Wang, Kuan},
  year   = {2026},
  url    = {https://github.com/kingofspace0wzz/slea-rl}
}

@article{feng2025group,
  title   = {Group-in-Group Policy Optimization for LLM Agent Training},
  author  = {Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal = {arXiv preprint arXiv:2505.10978},
  year    = {2025}
}
```
