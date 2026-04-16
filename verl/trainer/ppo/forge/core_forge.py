# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
FORGE Core Algorithms

Advantage computation for FORGE, building on GiGPO's multi-level advantage structure.

FORGE combines:
1. Episode-level advantages (GRPO-style)
2. Step-level advantages (GiGPO-style)
3. Experience-weighted adjustments (FORGE-specific)

The advantage computation is mostly reused from GiGPO,
with optional experience-based weighting.
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import uuid

from verl import DataProto


def compute_forge_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    step_advantage_w: float = 1.0,
    epsilon: float = 1e-6,
    mode: str = "mean_norm",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute FORGE advantages.

    This reuses GiGPO's multi-level advantage computation:
    - Episode-level normalization (Eq. 3 in GiGPO)
    - Step-level grouping and normalization (Eq. 6-7 in GiGPO)
    - Combined advantage (Eq. 8 in GiGPO)

    The main difference from GiGPO is in the training loop,
    where FORGE adds experience evolution and distillation.

    Args:
        token_level_rewards: (bs, response_length) - Token-level rewards
        step_rewards: (bs,) - Step-level rewards (discounted returns)
        response_mask: (bs, response_length) - Response mask
        anchor_obs: (bs,) - Anchor observations for step grouping
        index: (bs,) - Episode group UIDs
        traj_index: (bs,) - Trajectory UIDs
        step_advantage_w: Weight for step-level advantages
        epsilon: Small value for numerical stability
        mode: "mean_norm" or "mean_std_norm"
        enable_similarity: Enable similarity-based step grouping
        similarity_thresh: Threshold for similarity grouping

    Returns:
        (advantages, returns) - Both shape (bs, response_length)
    """
    # Import GiGPO functions to reuse
    from gigpo.core_gigpo import (
        episode_norm_reward,
        build_step_group,
        step_norm_reward,
    )

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute episode-level advantages
    episode_advantages = episode_norm_reward(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        epsilon=epsilon,
        remove_std=remove_std,
    )

    # Build step groups
    step_group_uids = build_step_group(
        anchor_obs=anchor_obs,
        index=index,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )

    # Compute step-level advantages
    step_advantages = step_norm_reward(
        step_rewards=step_rewards,
        response_mask=response_mask,
        index=step_group_uids,
        epsilon=epsilon,
        remove_std=remove_std,
    )

    # Combine advantages
    advantages = episode_advantages + step_advantage_w * step_advantages

    return advantages, advantages


def compute_forge_episode_only_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_by_std: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute FORGE advantages without step-level component.

    This is similar to GRPO and can be used when step rewards are not available.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) - Episode group UIDs
        traj_index: (bs,) - Trajectory UIDs
        epsilon: Small value for numerical stability
        norm_by_std: Whether to normalize by std

    Returns:
        (advantages, returns)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()

    with torch.no_grad():
        bsz = scores.shape[0]

        # Collect scores per group
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])

        # Compute mean and std per group
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"No score in prompt index: {idx}")

        # Normalize scores
        for i in range(bsz):
            if norm_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        # Broadcast to token level
        advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages


def compute_experience_weighted_advantage(
    advantages: torch.Tensor,
    batch: DataProto,
    library_stats: Dict[str, Any],
    experience_weight: float = 0.1,
) -> torch.Tensor:
    """
    Optional: Weight advantages based on experience relevance.

    This can be used to give more weight to trajectories that
    are similar to successful experiences in the library.

    Args:
        advantages: Base advantages (bs, response_length)
        batch: DataProto batch
        library_stats: Library statistics
        experience_weight: Weight for experience-based adjustment

    Returns:
        Adjusted advantages
    """
    if library_stats.get('total_size', 0) == 0:
        return advantages

    # Simple weighting based on library size
    # More sophisticated approaches could use embedding similarity
    library_factor = min(1.0, library_stats['total_size'] / 100)

    # Apply slight boost to advantages when library is mature
    adjusted = advantages * (1.0 + experience_weight * library_factor)

    return adjusted


def compute_step_discounted_returns(batch: DataProto, gamma: float) -> torch.Tensor:
    """
    Compute discounted returns for step-level advantages.

    This is the same as GiGPO's step return computation.

    Args:
        batch: DataProto with rewards and trajectory info
        gamma: Discount factor

    Returns:
        Step-level discounted returns (bs,)
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)

    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)

    for uid in unique_traj_uids:
        traj_indices = np.where(traj_uids == uid)[0]
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]

        # Calculate discounted returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0

        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return

        returns_by_traj[uid] = traj_returns

    # Reconstruct returns in original order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]
        all_returns[i] = returns_by_traj[uid][idx_in_traj]

    device = batch.batch['input_ids'].device
    return torch.tensor(all_returns, dtype=torch.float32, device=device)


def prepare_forge_batch(
    batch: DataProto,
    gamma: float,
    use_step_rewards: bool = True,
) -> DataProto:
    """
    Prepare batch for FORGE training by computing step rewards.

    Args:
        batch: Input DataProto batch
        gamma: Discount factor
        use_step_rewards: Whether to compute step rewards

    Returns:
        Updated batch with step_rewards
    """
    if use_step_rewards and 'step_rewards' not in batch.batch:
        step_rewards = compute_step_discounted_returns(batch, gamma)
        batch.batch['step_rewards'] = step_rewards

    return batch
