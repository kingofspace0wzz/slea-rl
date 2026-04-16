# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0
"""
Tests for the SleepTrainer in Cyclic-FLEX.
"""

import unittest
import tempfile
import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from verl.trainer.ppo.cyclic_flex.sleep_trainer import (
    SleepTrainer,
    SleepConfig,
    PEFT_AVAILABLE,
)


class TestSleepConfig(unittest.TestCase):
    """Tests for the SleepConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SleepConfig()

        self.assertEqual(config.lora_rank, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.lora_dropout, 0.05)
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.epochs, 2)
        self.assertEqual(config.batch_size, 4)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SleepConfig(
            lora_rank=32,
            lora_alpha=64,
            learning_rate=1e-4,
            epochs=3,
        )

        self.assertEqual(config.lora_rank, 32)
        self.assertEqual(config.lora_alpha, 64)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.epochs, 3)


class TestSleepTrainer(unittest.TestCase):
    """Tests for the SleepTrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.trainer = SleepTrainer(
            lora_rank=8,
            lora_alpha=16,
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
        )

    def test_initialization(self):
        """Test trainer initializes correctly."""
        self.assertEqual(self.trainer.lora_rank, 8)
        self.assertEqual(self.trainer.lora_alpha, 16)
        self.assertEqual(self.trainer.epochs, 1)
        self.assertEqual(self.trainer.cycle_count, 0)
        self.assertEqual(self.trainer.total_training_steps, 0)

    def test_from_config(self):
        """Test creating trainer from config."""
        config = SleepConfig(
            lora_rank=32,
            epochs=3,
        )

        trainer = SleepTrainer.from_config(config)

        self.assertEqual(trainer.lora_rank, 32)
        self.assertEqual(trainer.epochs, 3)

    def test_get_state(self):
        """Test getting trainer state."""
        self.trainer.cycle_count = 5
        self.trainer.total_training_steps = 100

        state = self.trainer.get_state()

        self.assertEqual(state['cycle_count'], 5)
        self.assertEqual(state['total_training_steps'], 100)

    def test_load_state(self):
        """Test loading trainer state."""
        state = {
            'cycle_count': 3,
            'total_training_steps': 50,
        }

        self.trainer.load_state(state)

        self.assertEqual(self.trainer.cycle_count, 3)
        self.assertEqual(self.trainer.total_training_steps, 50)

    def test_empty_dataloader_handling(self):
        """Test handling of empty dataloader."""
        metrics = self.trainer.run_consolidation(
            model=MagicMock(),
            dataloader=None,
            device="cpu",
        )

        self.assertTrue(metrics.get('sleep/skipped', False))

    def test_empty_dataloader_list_handling(self):
        """Test handling of empty dataloader (as list)."""
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=0)

        metrics = self.trainer.run_consolidation(
            model=MagicMock(),
            dataloader=mock_dataloader,
            device="cpu",
        )

        self.assertTrue(metrics.get('sleep/skipped', False))


class TestSleepTrainerConsolidation(unittest.TestCase):
    """Tests for the consolidation process."""

    def setUp(self):
        """Set up test fixtures with a simple model."""
        self.trainer = SleepTrainer(
            lora_rank=4,
            lora_alpha=8,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            merge_after_training=False,  # Don't merge for testing
            save_adapter=False,  # Don't save for testing
        )

    def test_simple_model_training(self):
        """Test training on a simple model (no LoRA)."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

        # Create simple dataset
        input_ids = torch.randint(0, 5, (4, 10))
        attention_mask = torch.ones(4, 10)
        labels = torch.randint(0, 5, (4, 10))

        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Mock the model's forward to return a loss
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, input_ids, attention_mask, labels):
                # Return a mock output with loss
                class Output:
                    loss = torch.tensor(1.0, requires_grad=True)
                return Output()

        mock_model = MockModel()

        # Run consolidation without LoRA
        with patch.object(self.trainer, 'apply_lora', return_value=mock_model):
            with patch('verl.trainer.ppo.cyclic_flex.sleep_trainer.PEFT_AVAILABLE', False):
                metrics = self.trainer.run_consolidation(
                    model=mock_model,
                    dataloader=dataloader,
                    device="cpu",
                    verbose=False,
                )

        # Check metrics
        self.assertIn('sleep/train_loss', metrics)
        self.assertIn('sleep/cycle', metrics)
        self.assertEqual(metrics['sleep/cycle'], 1)


class TestSleepTrainerDataloader(unittest.TestCase):
    """Tests for dataloader preparation."""

    def setUp(self):
        """Set up test fixtures."""
        self.trainer = SleepTrainer(batch_size=2)

    def test_prepare_dataloader_empty_buffer(self):
        """Test preparing dataloader from empty buffer."""
        # Create mock buffer that returns empty dataset
        mock_buffer = MagicMock()
        mock_buffer.to_sft_dataset = MagicMock(return_value={
            'input_ids': torch.tensor([]),
            'attention_mask': torch.tensor([]),
            'labels': torch.tensor([]),
        })

        mock_tokenizer = MagicMock()

        dataloader = self.trainer.prepare_dataloader(mock_buffer, mock_tokenizer)

        self.assertIsNone(dataloader)


class TestSleepTrainerCheckpointing(unittest.TestCase):
    """Tests for checkpoint functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.trainer = SleepTrainer(
            save_adapter=True,
        )

    def test_checkpoint_directory_creation(self):
        """Test that checkpoint directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer.checkpoint_dir = tmpdir
            self.trainer.cycle_count = 0

            # Create a mock model
            mock_model = MagicMock()
            mock_model.state_dict = MagicMock(return_value={'weight': torch.tensor([1.0])})

            with patch('verl.trainer.ppo.cyclic_flex.sleep_trainer.PEFT_AVAILABLE', False):
                checkpoint_path = self.trainer._save_checkpoint(mock_model)

            self.assertTrue(os.path.exists(checkpoint_path))


if __name__ == "__main__":
    unittest.main()
