# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
Core Cyclic-FLEX advantage computation functions.

This module implements the advantage estimation for Cyclic-FLEX, combining:
1. GRPO-style group-relative normalization
2. GiGPO-style step-level credit assignment (optional)
3. Experience-weighted learning bonus (optional)

The advantage computation follows the wake phase of the Cyclic-FLEX algorithm,
where trajectories are compared within groups and high-performing strategies
are identified for later consolidation in the sleep phase.
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from verl.trainer.ppo.cyclic_flex.experience_buffer import ExperienceBuffer


def compute_cyclic_flex_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    experience_buffer: Optional['ExperienceBuffer'] = None,
    experience_weight: float = 0.1,
    epsilon: float = 1e-6,
    norm_by_std: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages for Cyclic-FLEX (episode-level).

    This function implements GRPO-style group normalization where advantages
    are computed relative to other trajectories within the same group (uid).

    Args:
        token_level_rewards: Per-token rewards, shape (batch_size, response_length)
        response_mask: Valid token mask, shape (batch_size, response_length)
        index: Group ID (uid) for each sample, shape (batch_size,)
        traj_index: Trajectory ID for each sample, shape (batch_size,)
        experience_buffer: Optional buffer for experience-based scoring (not used in basic version)
        experience_weight: Weight for experience-based bonus (reserved for future use)
        epsilon: Small value for numerical stability
        norm_by_std: Whether to normalize by standard deviation (True) or just mean (False)

    Returns:
        advantages: Advantage estimates, shape (batch_size, response_length)
        returns: Return estimates (same as advantages for GRPO-style), shape (batch_size, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    device = token_level_rewards.device

    # Compute episode-level scores (sum of token rewards)
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)

    # Group scores by uid
    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()

    with torch.no_grad():
        batch_size = scores.shape[0]

        # Collect scores by group, avoiding duplicates from same trajectory
        for i in range(batch_size):
            uid = index[i]
            tid = traj_index[i]

            # Track unique (uid, traj_uid) pairs to avoid double counting
            pair = (uid, tid)
            if pair not in seen_pairs:
                id2scores[uid].append(scores[i])
                seen_pairs.add(pair)

            id2indices[uid].append(i)

        # Compute group statistics
        id2mean = {}
        id2std = {}

        for uid in id2scores:
            group_scores = torch.stack(id2scores[uid])

            if len(group_scores) == 1:
                # Single sample in group: normalize to 0
                id2mean[uid] = group_scores[0]
                id2std[uid] = torch.tensor(1.0, device=device)
            else:
                id2mean[uid] = group_scores.mean()
                id2std[uid] = group_scores.std()

        # Normalize scores within each group
        normalized_scores = torch.zeros_like(scores)

        for i in range(batch_size):
            uid = index[i]
            if norm_by_std:
                # Mean-std normalization (default for GRPO)
                normalized_scores[i] = (scores[i] - id2mean[uid]) / (id2std[uid] + epsilon)
            else:
                # Mean-only normalization
                normalized_scores[i] = scores[i] - id2mean[uid]

        # Expand to token level and apply mask
        advantages = normalized_scores.unsqueeze(-1).expand(-1, response_length) * response_mask
        returns = advantages.clone()

    return advantages, returns


def compute_cyclic_flex_with_step_rewards(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    step_advantage_w: float = 1.0,
    epsilon: float = 1e-6,
    norm_by_std: bool = True,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Cyclic-FLEX advantages with GiGPO-style step-level credit assignment.

    This combines episode-level advantages with step-level advantages based on
    anchor observation grouping (from GiGPO). This enables finer-grained credit
    assignment for multi-turn agent tasks.

    Args:
        token_level_rewards: Per-token rewards, shape (batch_size, response_length)
        step_rewards: Per-step discounted returns, shape (batch_size,)
        response_mask: Valid token mask, shape (batch_size, response_length)
        anchor_obs: Anchor observations for step grouping, shape (batch_size,)
        index: Group ID (uid) for each sample, shape (batch_size,)
        traj_index: Trajectory ID for each sample, shape (batch_size,)
        step_advantage_w: Weight for step-level advantages (vs episode-level)
        epsilon: Small value for numerical stability
        norm_by_std: Whether to normalize by standard deviation
        enable_similarity: Whether to use similarity-based grouping (vs exact match)
        similarity_thresh: Threshold for similarity grouping (if enabled)

    Returns:
        advantages: Combined advantage estimates, shape (batch_size, response_length)
        returns: Return estimates, shape (batch_size, response_length)
    """
    # Import GiGPO functions for step-level computation
    try:
        from gigpo.core_gigpo import (
            episode_norm_reward,
            build_step_group,
            step_norm_reward,
        )
    except ImportError:
        # Fallback to episode-only if GiGPO not available
        print("Warning: gigpo module not found, falling back to episode-only advantages")
        return compute_cyclic_flex_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            norm_by_std=norm_by_std,
        )

    # Episode-level advantages (Eq. 3 in GiGPO paper)
    episode_advantages = episode_norm_reward(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        epsilon=epsilon,
        remove_std=not norm_by_std,
    )

    # Build step-level groups based on anchor observations (Eq. 6 in GiGPO paper)
    step_group_uids = build_step_group(
        anchor_obs=anchor_obs,
        index=index,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
        summarize=False,
    )

    # Step-level advantages (Eq. 7 in GiGPO paper)
    step_advantages = step_norm_reward(
        step_rewards=step_rewards,
        response_mask=response_mask,
        index=step_group_uids,
        epsilon=epsilon,
        remove_std=not norm_by_std,
    )

    # Combined advantages (Eq. 8 in GiGPO paper)
    advantages = episode_advantages + step_advantage_w * step_advantages
    returns = advantages.clone()

    return advantages, returns


def compute_reflection_candidates(
    batch_scores: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    top_k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify candidates for reflection generation based on group comparisons.

    For each group, identifies the best and worst trajectories for contrastive
    reflection generation (comparing what worked vs what didn't).

    Args:
        batch_scores: Scores for each sample, shape (batch_size,)
        index: Group ID (uid) for each sample
        traj_index: Trajectory ID for each sample
        top_k: Number of top/bottom trajectories to select per group

    Returns:
        winner_indices: Indices of winning trajectories
        loser_indices: Indices of losing trajectories (paired with winners)
    """
    winners = []
    losers = []

    # Group by uid
    id2data = defaultdict(list)
    for i in range(len(batch_scores)):
        uid = index[i]
        score = batch_scores[i].item() if torch.is_tensor(batch_scores[i]) else batch_scores[i]
        id2data[uid].append((i, score, traj_index[i]))

    for uid, data in id2data.items():
        if len(data) < 2:
            continue

        # Sort by score
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

        # Get unique trajectories (avoid comparing same trajectory)
        seen_trajs = set()
        top_items = []
        bottom_items = []

        for idx, score, tid in sorted_data:
            if tid not in seen_trajs:
                if len(top_items) < top_k:
                    top_items.append((idx, score))
                seen_trajs.add(tid)

        seen_trajs = set()
        for idx, score, tid in reversed(sorted_data):
            if tid not in seen_trajs:
                if len(bottom_items) < top_k:
                    bottom_items.append((idx, score))
                seen_trajs.add(tid)

        # Pair winners with losers
        for (win_idx, win_score), (lose_idx, lose_score) in zip(top_items, bottom_items):
            if win_score > lose_score:  # Only if there's actual difference
                winners.append(win_idx)
                losers.append(lose_idx)

    return np.array(winners), np.array(losers)


def compute_experience_bonus(
    queries: list,
    experience_buffer: 'ExperienceBuffer',
    batch_size: int,
    bonus_weight: float = 0.1,
) -> torch.Tensor:
    """
    Compute experience-based bonus for trajectories similar to past successes.

    This encourages the model to follow strategies that have worked before,
    providing a form of implicit curriculum learning.

    Args:
        queries: List of query strings for each sample
        experience_buffer: Buffer containing past successful experiences
        batch_size: Number of samples in batch
        bonus_weight: Weight for the experience bonus

    Returns:
        bonus: Experience bonus for each sample, shape (batch_size,)
    """
    bonus = torch.zeros(batch_size)

    if experience_buffer is None or len(experience_buffer) == 0:
        return bonus

    for i, query in enumerate(queries):
        # Retrieve similar experiences
        similar_exps = experience_buffer.retrieve(query, k=3)

        if similar_exps:
            # Bonus based on average score of similar past experiences
            avg_score = np.mean([exp.score for exp in similar_exps])
            bonus[i] = bonus_weight * avg_score

    return bonus
