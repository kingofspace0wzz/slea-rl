# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0
"""
Tests for the core Cyclic-FLEX advantage computation functions.
"""

import unittest
import torch
import numpy as np

from verl.trainer.ppo.cyclic_flex.core_cyclic_flex import (
    compute_cyclic_flex_advantage,
    compute_reflection_candidates,
    compute_experience_bonus,
)


class TestComputeCyclicFlexAdvantage(unittest.TestCase):
    """Tests for the compute_cyclic_flex_advantage function."""

    def test_basic_advantage_computation(self):
        """Test basic GRPO-style advantage computation."""
        batch_size = 4
        response_length = 10

        # Create test data
        token_level_rewards = torch.zeros(batch_size, response_length)
        # Set final token rewards as episode scores
        token_level_rewards[:, -1] = torch.tensor([1.0, 0.5, 0.8, 0.2])

        response_mask = torch.ones(batch_size, response_length)

        # All same group
        index = np.array([0, 0, 0, 0])
        traj_index = np.array([0, 1, 2, 3])

        advantages, returns = compute_cyclic_flex_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
        )

        # Check shapes
        self.assertEqual(advantages.shape, (batch_size, response_length))
        self.assertEqual(returns.shape, (batch_size, response_length))

        # Within a group, advantages should be normalized
        # Mean score is 0.625, so first trajectory (1.0) should have positive advantage
        self.assertGreater(advantages[0, 0].item(), 0)
        # Last trajectory (0.2) should have negative advantage
        self.assertLess(advantages[3, 0].item(), 0)

    def test_multiple_groups(self):
        """Test advantage computation with multiple groups."""
        batch_size = 6
        response_length = 5

        token_level_rewards = torch.zeros(batch_size, response_length)
        token_level_rewards[:, -1] = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.9, 0.1])

        response_mask = torch.ones(batch_size, response_length)

        # Two groups
        index = np.array([0, 0, 0, 1, 1, 1])
        traj_index = np.array([0, 1, 2, 3, 4, 5])

        advantages, returns = compute_cyclic_flex_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
        )

        # Group 0: scores [1.0, 0.5, 0.8], mean = 0.767
        # Group 1: scores [0.2, 0.9, 0.1], mean = 0.4

        # First trajectory in group 0 (score 1.0) should be positive
        self.assertGreater(advantages[0, 0].item(), 0)

        # First trajectory in group 1 (score 0.2) should be negative
        self.assertLess(advantages[3, 0].item(), 0)

    def test_single_sample_group(self):
        """Test handling of groups with single sample."""
        batch_size = 3
        response_length = 5

        token_level_rewards = torch.zeros(batch_size, response_length)
        token_level_rewards[:, -1] = torch.tensor([1.0, 0.5, 0.8])

        response_mask = torch.ones(batch_size, response_length)

        # Each sample in its own group
        index = np.array([0, 1, 2])
        traj_index = np.array([0, 1, 2])

        advantages, returns = compute_cyclic_flex_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
        )

        # Single samples should normalize to 0
        self.assertAlmostEqual(advantages[0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(advantages[1, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(advantages[2, 0].item(), 0.0, places=5)

    def test_response_mask_applied(self):
        """Test that response mask is properly applied."""
        batch_size = 2
        response_length = 5

        token_level_rewards = torch.zeros(batch_size, response_length)
        token_level_rewards[:, -1] = torch.tensor([1.0, 0.5])

        # Mask out some tokens
        response_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ], dtype=torch.float)

        index = np.array([0, 0])
        traj_index = np.array([0, 1])

        advantages, returns = compute_cyclic_flex_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
        )

        # Masked positions should be 0
        self.assertEqual(advantages[0, 3].item(), 0.0)
        self.assertEqual(advantages[0, 4].item(), 0.0)
        self.assertEqual(advantages[1, 2].item(), 0.0)

    def test_norm_by_std_disabled(self):
        """Test advantage computation without std normalization."""
        batch_size = 4
        response_length = 5

        token_level_rewards = torch.zeros(batch_size, response_length)
        token_level_rewards[:, -1] = torch.tensor([1.0, 0.5, 0.8, 0.2])

        response_mask = torch.ones(batch_size, response_length)
        index = np.array([0, 0, 0, 0])
        traj_index = np.array([0, 1, 2, 3])

        advantages, _ = compute_cyclic_flex_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            norm_by_std=False,
        )

        # Mean score is 0.625
        # First trajectory should have advantage = 1.0 - 0.625 = 0.375
        self.assertAlmostEqual(advantages[0, 0].item(), 0.375, places=4)


class TestComputeReflectionCandidates(unittest.TestCase):
    """Tests for the compute_reflection_candidates function."""

    def test_basic_candidate_selection(self):
        """Test basic winner/loser selection."""
        batch_scores = torch.tensor([0.9, 0.3, 0.7, 0.1])
        index = np.array([0, 0, 0, 0])  # All same group
        traj_index = np.array([0, 1, 2, 3])

        winners, losers = compute_reflection_candidates(
            batch_scores=batch_scores,
            index=index,
            traj_index=traj_index,
            top_k=1,
        )

        # Should identify trajectory 0 (score 0.9) as winner
        # and trajectory 3 (score 0.1) as loser
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(losers), 1)
        self.assertEqual(winners[0], 0)
        self.assertEqual(losers[0], 3)

    def test_multiple_groups_candidates(self):
        """Test candidate selection with multiple groups."""
        batch_scores = torch.tensor([0.9, 0.3, 0.8, 0.1, 0.7, 0.2])
        index = np.array([0, 0, 0, 1, 1, 1])
        traj_index = np.array([0, 1, 2, 3, 4, 5])

        winners, losers = compute_reflection_candidates(
            batch_scores=batch_scores,
            index=index,
            traj_index=traj_index,
            top_k=1,
        )

        # Should have one winner/loser pair per group
        self.assertEqual(len(winners), 2)
        self.assertEqual(len(losers), 2)

    def test_single_sample_group_ignored(self):
        """Test that single-sample groups are ignored."""
        batch_scores = torch.tensor([0.9, 0.5])
        index = np.array([0, 1])  # Each in own group
        traj_index = np.array([0, 1])

        winners, losers = compute_reflection_candidates(
            batch_scores=batch_scores,
            index=index,
            traj_index=traj_index,
        )

        # No pairs should be selected from single-sample groups
        self.assertEqual(len(winners), 0)
        self.assertEqual(len(losers), 0)

    def test_no_difference_group_skipped(self):
        """Test that groups with no score difference are skipped."""
        batch_scores = torch.tensor([0.5, 0.5])
        index = np.array([0, 0])
        traj_index = np.array([0, 1])

        winners, losers = compute_reflection_candidates(
            batch_scores=batch_scores,
            index=index,
            traj_index=traj_index,
        )

        # Equal scores should not produce pairs
        self.assertEqual(len(winners), 0)


class TestComputeExperienceBonus(unittest.TestCase):
    """Tests for the compute_experience_bonus function."""

    def test_empty_buffer(self):
        """Test bonus computation with empty buffer."""
        queries = ["query1", "query2"]
        batch_size = 2

        bonus = compute_experience_bonus(
            queries=queries,
            experience_buffer=None,
            batch_size=batch_size,
        )

        # Empty buffer should return zero bonus
        self.assertEqual(bonus.shape, (batch_size,))
        self.assertTrue(torch.all(bonus == 0))

    def test_bonus_shape(self):
        """Test that bonus has correct shape."""
        from unittest.mock import MagicMock, patch

        # Create mock buffer
        mock_buffer = MagicMock()
        mock_buffer.__len__ = MagicMock(return_value=5)

        # Mock retrieve to return empty
        mock_buffer.retrieve = MagicMock(return_value=[])

        queries = ["q1", "q2", "q3"]
        batch_size = 3

        bonus = compute_experience_bonus(
            queries=queries,
            experience_buffer=mock_buffer,
            batch_size=batch_size,
        )

        self.assertEqual(bonus.shape, (batch_size,))


class TestAdvantageTensorOperations(unittest.TestCase):
    """Tests for tensor operations in advantage computation."""

    def test_advantage_dtype(self):
        """Test that advantages maintain proper dtype."""
        batch_size = 2
        response_length = 5

        token_level_rewards = torch.zeros(batch_size, response_length, dtype=torch.float32)
        token_level_rewards[:, -1] = torch.tensor([1.0, 0.5])

        response_mask = torch.ones(batch_size, response_length, dtype=torch.float32)
        index = np.array([0, 0])
        traj_index = np.array([0, 1])

        advantages, returns = compute_cyclic_flex_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
        )

        self.assertEqual(advantages.dtype, torch.float32)
        self.assertEqual(returns.dtype, torch.float32)

    def test_advantage_device_consistency(self):
        """Test that advantages are on the same device as input."""
        batch_size = 2
        response_length = 5

        token_level_rewards = torch.zeros(batch_size, response_length)
        token_level_rewards[:, -1] = torch.tensor([1.0, 0.5])

        response_mask = torch.ones(batch_size, response_length)
        index = np.array([0, 0])
        traj_index = np.array([0, 1])

        advantages, returns = compute_cyclic_flex_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
        )

        self.assertEqual(advantages.device, token_level_rewards.device)
        self.assertEqual(returns.device, token_level_rewards.device)


if __name__ == "__main__":
    unittest.main()
