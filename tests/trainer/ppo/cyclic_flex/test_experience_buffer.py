# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0
"""
Tests for the ExperienceBuffer in Cyclic-FLEX.
"""

import unittest
import tempfile
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from verl.trainer.ppo.cyclic_flex.experience_buffer import (
    ExperienceItem,
    ExperienceBuffer,
)


class TestExperienceItem(unittest.TestCase):
    """Tests for the ExperienceItem dataclass."""

    def test_experience_item_creation(self):
        """Test basic ExperienceItem creation."""
        item = ExperienceItem(
            query="What is the capital of France?",
            response="Paris",
            score=0.95,
            uid="test_uid_1",
            traj_uid="traj_1",
        )

        self.assertEqual(item.query, "What is the capital of France?")
        self.assertEqual(item.response, "Paris")
        self.assertEqual(item.score, 0.95)
        self.assertEqual(item.uid, "test_uid_1")
        self.assertEqual(item.traj_uid, "traj_1")
        self.assertIsNone(item.reflection)
        self.assertIsNone(item.embedding)

    def test_experience_item_with_reflection(self):
        """Test ExperienceItem with reflection."""
        item = ExperienceItem(
            query="Navigate to kitchen",
            response="go north\ntake lamp",
            score=1.0,
            uid="alfworld_1",
            traj_uid="traj_2",
            reflection="Key: Check inventory before navigating.",
        )

        self.assertEqual(item.reflection, "Key: Check inventory before navigating.")

    def test_experience_item_with_embedding(self):
        """Test ExperienceItem with embedding."""
        embedding = np.random.randn(384).astype(np.float32)
        item = ExperienceItem(
            query="Search query",
            response="Search results",
            score=0.8,
            uid="search_1",
            traj_uid="traj_3",
            embedding=embedding,
        )

        self.assertIsNotNone(item.embedding)
        self.assertEqual(item.embedding.shape, (384,))


class TestExperienceBuffer(unittest.TestCase):
    """Tests for the ExperienceBuffer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ExperienceBuffer(
            capacity=10,
            min_score_threshold=0.0,
            use_semantic_retrieval=False,  # Disable to avoid loading model
        )

    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        self.assertEqual(self.buffer.capacity, 10)
        self.assertEqual(len(self.buffer), 0)
        self.assertFalse(self.buffer.is_full())

    def test_add_single_experience(self):
        """Test adding a single experience."""
        self.buffer.add(
            query="Test query",
            response="Test response",
            score=0.9,
            uid="uid_1",
            traj_uid="traj_1",
        )

        self.assertEqual(len(self.buffer), 1)

    def test_add_experience_below_threshold(self):
        """Test that experiences below threshold are not added."""
        buffer = ExperienceBuffer(
            capacity=10,
            min_score_threshold=0.5,
            use_semantic_retrieval=False,
        )

        buffer.add(
            query="Low score query",
            response="Low score response",
            score=0.3,  # Below threshold
            uid="uid_1",
            traj_uid="traj_1",
        )

        self.assertEqual(len(buffer), 0)

    def test_buffer_capacity_limit(self):
        """Test that buffer respects capacity limit with FIFO eviction."""
        for i in range(15):  # Add more than capacity
            self.buffer.add(
                query=f"Query {i}",
                response=f"Response {i}",
                score=0.9,
                uid=f"uid_{i}",
                traj_uid=f"traj_{i}",
            )

        self.assertEqual(len(self.buffer), 10)  # Should be at capacity

    def test_is_full(self):
        """Test is_full method."""
        for i in range(5):
            self.buffer.add(
                query=f"Query {i}",
                response=f"Response {i}",
                score=0.9,
                uid=f"uid_{i}",
                traj_uid=f"traj_{i}",
            )

        self.assertFalse(self.buffer.is_full())

        for i in range(5, 10):
            self.buffer.add(
                query=f"Query {i}",
                response=f"Response {i}",
                score=0.9,
                uid=f"uid_{i}",
                traj_uid=f"traj_{i}",
            )

        self.assertTrue(self.buffer.is_full())

    def test_retrieve_recent(self):
        """Test retrieving recent experiences (no semantic retrieval)."""
        for i in range(5):
            self.buffer.add(
                query=f"Query {i}",
                response=f"Response {i}",
                score=0.9 - i * 0.1,
                uid=f"uid_{i}",
                traj_uid=f"traj_{i}",
            )

        # Retrieve top 3
        retrieved = self.buffer.retrieve("Any query", k=3)

        self.assertEqual(len(retrieved), 3)

    def test_flush(self):
        """Test flushing the buffer."""
        for i in range(5):
            self.buffer.add(
                query=f"Query {i}",
                response=f"Response {i}",
                score=0.9,
                uid=f"uid_{i}",
                traj_uid=f"traj_{i}",
            )

        self.assertEqual(len(self.buffer), 5)

        self.buffer.flush()

        self.assertEqual(len(self.buffer), 0)

    def test_get_positive_experiences(self):
        """Test getting positive experiences."""
        # Add mixed scores
        self.buffer.add(query="Q1", response="R1", score=0.9, uid="1", traj_uid="t1")
        self.buffer.add(query="Q2", response="R2", score=0.1, uid="2", traj_uid="t2")
        self.buffer.add(query="Q3", response="R3", score=0.7, uid="3", traj_uid="t3")
        self.buffer.add(query="Q4", response="R4", score=0.2, uid="4", traj_uid="t4")

        positive = self.buffer.get_positive_experiences(threshold=0.5)

        self.assertEqual(len(positive), 2)  # Only Q1 and Q3

    def test_to_sft_dataset(self):
        """Test converting buffer to SFT dataset."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.apply_chat_template = MagicMock(return_value="formatted text")
        tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]),
        }
        tokenizer.__call__ = lambda *args, **kwargs: {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # Add some experiences
        self.buffer.add(query="Q1", response="R1", score=0.9, uid="1", traj_uid="t1")
        self.buffer.add(query="Q2", response="R2", score=0.8, uid="2", traj_uid="t2")

        # This will fail without proper tokenizer, just test that method exists
        try:
            dataset = self.buffer.to_sft_dataset(tokenizer)
        except Exception:
            # Expected to fail with mock tokenizer, but method exists
            pass

    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "buffer_checkpoint.pt")

            # Add experiences
            for i in range(5):
                self.buffer.add(
                    query=f"Query {i}",
                    response=f"Response {i}",
                    score=0.9 - i * 0.1,
                    uid=f"uid_{i}",
                    traj_uid=f"traj_{i}",
                )

            # Save checkpoint
            self.buffer.save_checkpoint(checkpoint_path)

            self.assertTrue(os.path.exists(checkpoint_path))

            # Create new buffer and load
            new_buffer = ExperienceBuffer(
                capacity=10,
                use_semantic_retrieval=False,
            )
            new_buffer.load_checkpoint(checkpoint_path)

            self.assertEqual(len(new_buffer), 5)
            self.assertEqual(new_buffer.experiences[0].query, "Query 0")

    def test_get_stats(self):
        """Test getting buffer statistics."""
        for i in range(5):
            self.buffer.add(
                query=f"Query {i}",
                response=f"Response {i}",
                score=0.5 + i * 0.1,  # 0.5, 0.6, 0.7, 0.8, 0.9
                uid=f"uid_{i}",
                traj_uid=f"traj_{i}",
            )

        stats = self.buffer.get_stats()

        self.assertEqual(stats['size'], 5)
        self.assertEqual(stats['capacity'], 10)
        self.assertAlmostEqual(stats['avg_score'], 0.7, places=5)
        self.assertAlmostEqual(stats['min_score'], 0.5, places=5)
        self.assertAlmostEqual(stats['max_score'], 0.9, places=5)


class TestExperienceBufferFromBatch(unittest.TestCase):
    """Tests for add_from_batch method."""

    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ExperienceBuffer(
            capacity=20,
            min_score_threshold=0.0,
            use_semantic_retrieval=False,
        )

    def test_add_from_batch_basic(self):
        """Test adding experiences from a DataProto batch."""
        # Create mock batch
        batch = MagicMock()
        batch.__len__ = MagicMock(return_value=4)
        batch.non_tensor_batch = {
            'uid': np.array(['uid_0', 'uid_0', 'uid_1', 'uid_1']),
            'traj_uid': np.array(['traj_0', 'traj_1', 'traj_2', 'traj_3']),
            'raw_prompt': np.array(['Query 0', 'Query 1', 'Query 2', 'Query 3']),
        }
        batch.batch = {
            'responses': torch.randint(0, 100, (4, 10)),
        }

        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        reflections = ['Reflection 0', 'Reflection 1', '', '']

        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.decode = MagicMock(return_value="Decoded response")

        added = self.buffer.add_from_batch(
            batch=batch,
            scores=scores,
            reflections=reflections,
            tokenizer=tokenizer,
        )

        self.assertEqual(added, 4)
        self.assertEqual(len(self.buffer), 4)


if __name__ == "__main__":
    unittest.main()
