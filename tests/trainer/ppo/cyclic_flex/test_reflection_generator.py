# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0
"""
Tests for the ReflectionGenerator in Cyclic-FLEX.
"""

import unittest
import torch
import numpy as np
from unittest.mock import MagicMock

from verl.trainer.ppo.cyclic_flex.reflection_generator import (
    ReflectionGenerator,
    REFLECTION_PROMPTS,
)


class TestReflectionPrompts(unittest.TestCase):
    """Tests for the reflection prompt templates."""

    def test_all_environments_have_prompts(self):
        """Test that all expected environments have prompts."""
        expected_envs = ['alfworld', 'webshop', 'search', 'sokoban', 'default']

        for env in expected_envs:
            self.assertIn(env, REFLECTION_PROMPTS)
            self.assertIn('system', REFLECTION_PROMPTS[env])
            self.assertIn('template', REFLECTION_PROMPTS[env])

    def test_prompt_template_placeholders(self):
        """Test that templates have required placeholders."""
        required_placeholders = ['{task}', '{win_score', '{winner_summary}', '{lose_score', '{loser_summary}']

        for env, prompts in REFLECTION_PROMPTS.items():
            template = prompts['template']
            for placeholder in required_placeholders:
                self.assertIn(placeholder, template,
                             f"Missing {placeholder} in {env} template")


class TestReflectionGenerator(unittest.TestCase):
    """Tests for the ReflectionGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = ReflectionGenerator(
            use_llm=False,
            env_name="alfworld",
        )

    def test_initialization(self):
        """Test generator initializes correctly."""
        self.assertFalse(self.generator.use_llm)
        self.assertEqual(self.generator.env_name, "alfworld")
        self.assertEqual(self.generator.prompts, REFLECTION_PROMPTS['alfworld'])

    def test_initialization_default_env(self):
        """Test generator falls back to default for unknown env."""
        generator = ReflectionGenerator(env_name="unknown_env")

        self.assertEqual(generator.prompts, REFLECTION_PROMPTS['default'])

    def test_initialization_with_path_env(self):
        """Test generator handles env names with paths."""
        generator = ReflectionGenerator(env_name="alfworld/AlfredTWEnv")

        self.assertEqual(generator.prompts, REFLECTION_PROMPTS['alfworld'])


class TestSimpleReflection(unittest.TestCase):
    """Tests for simple reflection generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = ReflectionGenerator(
            use_llm=False,
            env_name="alfworld",
        )

    def test_simple_reflection_high_score(self):
        """Test simple reflection for high-scoring trajectory."""
        # Create mock batch
        batch = MagicMock()

        reflection = self.generator._generate_simple_reflection(
            winner_score=1.0,
            loser_score=0.2,
            batch=batch,
            winner_idx=0,
            loser_idx=1,
        )

        self.assertIn("successful", reflection)
        self.assertIn("1.00", reflection)
        self.assertIn("0.80", reflection)  # Score diff
        self.assertIn("alfworld", reflection.lower())

    def test_simple_reflection_medium_score(self):
        """Test simple reflection for medium-scoring trajectory."""
        batch = MagicMock()

        reflection = self.generator._generate_simple_reflection(
            winner_score=0.6,
            loser_score=0.4,
            batch=batch,
            winner_idx=0,
            loser_idx=1,
        )

        self.assertIn("partially successful", reflection)

    def test_simple_reflection_low_score(self):
        """Test simple reflection for low-scoring trajectory."""
        batch = MagicMock()

        reflection = self.generator._generate_simple_reflection(
            winner_score=0.3,
            loser_score=0.1,
            batch=batch,
            winner_idx=0,
            loser_idx=1,
        )

        self.assertIn("attempted", reflection)

    def test_environment_specific_hints(self):
        """Test that reflections include environment-specific hints."""
        env_hints = {
            'alfworld': 'systematic exploration',
            'webshop': 'product search',
            'search': 'query formulation',
            'sokoban': 'box traps',
        }

        for env, hint in env_hints.items():
            generator = ReflectionGenerator(use_llm=False, env_name=env)
            batch = MagicMock()

            reflection = generator._generate_simple_reflection(
                winner_score=1.0,
                loser_score=0.5,
                batch=batch,
                winner_idx=0,
                loser_idx=1,
            )

            self.assertIn(hint, reflection.lower(),
                         f"Missing hint '{hint}' for env '{env}'")


class TestGenerateReflections(unittest.TestCase):
    """Tests for the main generate_reflections method."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = ReflectionGenerator(
            use_llm=False,
            env_name="default",
        )

    def create_mock_batch(self, batch_size):
        """Helper to create mock batch."""
        batch = MagicMock()
        batch.__len__ = MagicMock(return_value=batch_size)
        batch.non_tensor_batch = {
            'uid': np.arange(batch_size // 2).repeat(2),  # Groups of 2
            'traj_uid': np.arange(batch_size),
        }
        return batch

    def test_generate_reflections_basic(self):
        """Test basic reflection generation."""
        batch_size = 4
        batch = self.create_mock_batch(batch_size)

        scores = torch.tensor([0.9, 0.3, 0.8, 0.2])

        reflections = self.generator.generate_reflections(
            batch=batch,
            scores=scores,
        )

        self.assertEqual(len(reflections), batch_size)

        # Winners should have reflections
        self.assertTrue(len(reflections[0]) > 0)  # Winner of group 0
        self.assertTrue(len(reflections[2]) > 0)  # Winner of group 1

        # Losers should have empty reflections
        self.assertEqual(reflections[1], "")
        self.assertEqual(reflections[3], "")

    def test_generate_reflections_single_sample_groups(self):
        """Test that single-sample groups get no reflections."""
        batch = MagicMock()
        batch.__len__ = MagicMock(return_value=3)
        batch.non_tensor_batch = {
            'uid': np.array([0, 1, 2]),  # Each in own group
            'traj_uid': np.array([0, 1, 2]),
        }

        scores = torch.tensor([0.9, 0.5, 0.3])

        reflections = self.generator.generate_reflections(
            batch=batch,
            scores=scores,
        )

        # No reflections for single-sample groups
        self.assertTrue(all(r == "" for r in reflections))

    def test_generate_reflections_no_improvement(self):
        """Test handling when winner equals loser score."""
        batch = MagicMock()
        batch.__len__ = MagicMock(return_value=2)
        batch.non_tensor_batch = {
            'uid': np.array([0, 0]),
            'traj_uid': np.array([0, 1]),
        }

        # Equal scores
        scores = torch.tensor([0.5, 0.5])

        reflections = self.generator.generate_reflections(
            batch=batch,
            scores=scores,
        )

        # No reflections when no improvement
        self.assertTrue(all(r == "" for r in reflections))


class TestBatchReflectionsSimple(unittest.TestCase):
    """Tests for the generate_batch_reflections_simple method."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = ReflectionGenerator(use_llm=False)

    def test_batch_reflections_simple(self):
        """Test simple batch reflection generation."""
        scores = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1])
        uids = np.array([0, 0, 1, 1, 2, 2])

        reflections, indices = self.generator.generate_batch_reflections_simple(
            scores=scores,
            uids=uids,
        )

        # Should have reflections for best in each group
        self.assertEqual(len(indices), 3)  # 3 groups

        # Best trajectories: 0 (score 0.9), 2 (score 0.8), 4 (score 0.7)
        self.assertIn(0, indices)
        self.assertIn(2, indices)
        self.assertIn(4, indices)

    def test_batch_reflections_filters_negative(self):
        """Test that negative scores don't get reflections."""
        scores = torch.tensor([0.9, 0.3, -0.5, -0.2])
        uids = np.array([0, 0, 1, 1])

        reflections, indices = self.generator.generate_batch_reflections_simple(
            scores=scores,
            uids=uids,
        )

        # Only group 0 should have reflection (group 1 has all negative)
        self.assertEqual(len(indices), 1)
        self.assertIn(0, indices)

    def test_batch_reflections_content(self):
        """Test that reflections have expected content."""
        scores = torch.tensor([0.85, 0.3])
        uids = np.array([0, 0])

        reflections, indices = self.generator.generate_batch_reflections_simple(
            scores=scores,
            uids=uids,
        )

        # Check reflection content
        self.assertIn("0.85", reflections[0])
        self.assertIn("Effective", reflections[0])


class TestTrajectoryExtraction(unittest.TestCase):
    """Tests for trajectory summary extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = ReflectionGenerator(use_llm=True)

    def test_extract_from_responses(self):
        """Test extracting summary from response tokens."""
        batch = MagicMock()
        batch.__getitem__ = MagicMock()

        item = MagicMock()
        item.batch = {
            'responses': torch.tensor([1, 2, 3, 4, 5])
        }
        item.non_tensor_batch = {}
        batch.__getitem__.return_value = item

        tokenizer = MagicMock()
        tokenizer.decode = MagicMock(return_value="Decoded response text")

        summary = self.generator._extract_trajectory_summary(
            batch=batch,
            idx=0,
            tokenizer=tokenizer,
        )

        self.assertEqual(summary, "Decoded response text")

    def test_extract_fallback_to_prompt(self):
        """Test fallback to raw_prompt when responses unavailable."""
        batch = MagicMock()

        item = MagicMock()
        item.batch = {}  # No responses
        item.non_tensor_batch = {
            'raw_prompt': "Original prompt text"
        }
        batch.__getitem__ = MagicMock(return_value=item)

        summary = self.generator._extract_trajectory_summary(
            batch=batch,
            idx=0,
            tokenizer=None,
        )

        self.assertEqual(summary, "Original prompt text")

    def test_extract_truncation(self):
        """Test that summaries are truncated to max length."""
        generator = ReflectionGenerator(
            use_llm=True,
            max_trajectory_length=20,
        )

        batch = MagicMock()
        item = MagicMock()
        item.batch = {}
        item.non_tensor_batch = {
            'raw_prompt': "This is a very long prompt that exceeds the maximum length"
        }
        batch.__getitem__ = MagicMock(return_value=item)

        summary = generator._extract_trajectory_summary(
            batch=batch,
            idx=0,
            tokenizer=None,
        )

        self.assertEqual(len(summary), 20)


if __name__ == "__main__":
    unittest.main()
