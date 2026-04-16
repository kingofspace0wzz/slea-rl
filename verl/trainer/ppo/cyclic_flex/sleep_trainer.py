# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
Sleep Trainer for Cyclic-FLEX weight consolidation.

The SleepTrainer handles the sleep phase of the Cyclic-FLEX algorithm,
converting accumulated experiences from the buffer into weight updates
via LoRA-based supervised fine-tuning.

Key features:
- LoRA-based efficient fine-tuning (minimal parameter updates)
- Short training cycles to avoid catastrophic forgetting
- Support for merging LoRA weights into base model
- Checkpoint management for sleep cycles
"""

import os
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. Sleep phase will use full fine-tuning.")


@dataclass
class SleepConfig:
    """Configuration for sleep phase training."""
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training settings
    learning_rate: float = 2e-5
    epochs: int = 2
    batch_size: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Consolidation settings
    merge_after_training: bool = True
    save_adapter: bool = True

    # Checkpoint settings
    checkpoint_dir: str = "checkpoints/cyclic_flex"


class SleepTrainer:
    """Handles sleep phase consolidation via SFT/LoRA.

    The sleep trainer converts experiences accumulated during the wake phase
    into weight updates. It uses LoRA (Low-Rank Adaptation) for efficient
    fine-tuning, minimizing the risk of catastrophic forgetting.

    Training objective:
        L(θ) = -E[(x,ε)∈D_sleep] [log π_θ(ε | x)]

    Where ε is the reflection/strategy, not just the final answer.
    This forces the model to internalize reasoning strategies.
    """

    def __init__(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        learning_rate: float = 2e-5,
        epochs: int = 2,
        batch_size: int = 4,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        merge_after_training: bool = True,
        save_adapter: bool = True,
        checkpoint_dir: str = "checkpoints/cyclic_flex",
    ):
        """Initialize the sleep trainer.

        Args:
            lora_rank: LoRA rank (r parameter)
            lora_alpha: LoRA scaling factor (α parameter)
            lora_dropout: Dropout rate for LoRA layers
            target_modules: List of module names to apply LoRA to
            learning_rate: Learning rate for SFT
            epochs: Number of training epochs per sleep cycle
            batch_size: Batch size for training
            warmup_ratio: Proportion of steps for learning rate warmup
            max_grad_norm: Maximum gradient norm for clipping
            weight_decay: Weight decay for AdamW optimizer
            merge_after_training: Whether to merge LoRA weights into base model
            save_adapter: Whether to save LoRA adapter checkpoints
            checkpoint_dir: Directory for saving checkpoints
        """
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.merge_after_training = merge_after_training
        self.save_adapter = save_adapter
        self.checkpoint_dir = checkpoint_dir

        self.cycle_count = 0
        self.total_training_steps = 0

    @classmethod
    def from_config(cls, config: SleepConfig) -> 'SleepTrainer':
        """Create SleepTrainer from config object."""
        return cls(
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.batch_size,
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=config.max_grad_norm,
            weight_decay=config.weight_decay,
            merge_after_training=config.merge_after_training,
            save_adapter=config.save_adapter,
            checkpoint_dir=config.checkpoint_dir,
        )

    def prepare_dataloader(
        self,
        buffer: 'ExperienceBuffer',
        tokenizer,
    ) -> DataLoader:
        """Convert experience buffer to training dataloader.

        Args:
            buffer: ExperienceBuffer containing accumulated experiences
            tokenizer: Tokenizer for the model

        Returns:
            DataLoader ready for training
        """
        data = buffer.to_sft_dataset(tokenizer)

        if data['input_ids'].numel() == 0:
            print("Warning: Empty dataset for sleep phase")
            return None

        dataset = TensorDataset(
            data['input_ids'],
            data['attention_mask'],
            data['labels'],
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def apply_lora(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply LoRA adapters to the model for efficient fine-tuning.

        Args:
            model: Base model to adapt

        Returns:
            Model with LoRA adapters applied
        """
        if not PEFT_AVAILABLE:
            print("Warning: peft not available, skipping LoRA application")
            return model

        # Check if already a PeftModel
        if isinstance(model, PeftModel):
            print("Model already has LoRA adapters")
            return model

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        return get_peft_model(model, lora_config)

    def run_consolidation(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute SFT consolidation on the accumulated experiences.

        This is the core of the sleep phase, where experiences are compressed
        into weight updates.

        Args:
            model: Model to fine-tune
            dataloader: DataLoader with training data
            device: Device to train on
            verbose: Whether to print progress

        Returns:
            Dictionary of training metrics
        """
        if dataloader is None or len(dataloader) == 0:
            return {
                'sleep/skipped': True,
                'sleep/reason': 'empty_dataloader',
            }

        # Apply LoRA if not already applied
        if PEFT_AVAILABLE and not isinstance(model, PeftModel):
            model = self.apply_lora(model)

        model.train()
        model = model.to(device)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        if verbose:
            print(f"Sleep Phase: Training {trainable_params:,} / {total_params:,} parameters ({100*trainable_params/total_params:.2f}%)")

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Training loop
        total_steps = len(dataloader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        metrics = {
            'losses': [],
            'learning_rates': [],
        }
        global_step = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                # Learning rate schedule (linear warmup then constant)
                if global_step < warmup_steps:
                    lr = self.learning_rate * (global_step + 1) / warmup_steps
                else:
                    lr = self.learning_rate

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                metrics['losses'].append(loss.item())
                metrics['learning_rates'].append(lr)

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            if verbose:
                print(f"  Sleep Epoch {epoch + 1}/{self.epochs}, Loss: {avg_epoch_loss:.4f}")

        self.total_training_steps += global_step

        # Merge LoRA weights if configured
        if self.merge_after_training and PEFT_AVAILABLE and isinstance(model, PeftModel):
            if verbose:
                print("  Merging LoRA weights into base model...")
            model = model.merge_and_unload()

        # Save adapter checkpoint
        self.cycle_count += 1
        if self.save_adapter:
            self._save_checkpoint(model)

        # Compute final metrics
        final_metrics = {
            'sleep/train_loss': sum(metrics['losses']) / len(metrics['losses']) if metrics['losses'] else 0,
            'sleep/final_loss': metrics['losses'][-1] if metrics['losses'] else 0,
            'sleep/cycle': self.cycle_count,
            'sleep/num_samples': len(dataloader.dataset),
            'sleep/num_steps': global_step,
            'sleep/trainable_params': trainable_params,
            'sleep/total_params': total_params,
        }

        return final_metrics

    def _save_checkpoint(self, model: torch.nn.Module) -> str:
        """Save checkpoint for this sleep cycle.

        Args:
            model: Model to checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"sleep_cycle_{self.cycle_count}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save adapter or full model state
        if PEFT_AVAILABLE and isinstance(model, PeftModel):
            model.save_pretrained(checkpoint_path)
        else:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_path, "model_state.pt")
            )

        # Save metadata
        metadata = {
            'cycle': self.cycle_count,
            'total_training_steps': self.total_training_steps,
            'config': {
                'lora_rank': self.lora_rank,
                'lora_alpha': self.lora_alpha,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
            }
        }
        torch.save(metadata, os.path.join(checkpoint_path, "metadata.pt"))

        print(f"  Saved sleep checkpoint to {checkpoint_path}")
        return checkpoint_path

    def get_state(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing."""
        return {
            'cycle_count': self.cycle_count,
            'total_training_steps': self.total_training_steps,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load trainer state from checkpoint."""
        self.cycle_count = state.get('cycle_count', 0)
        self.total_training_steps = state.get('total_training_steps', 0)


class DistributedSleepTrainer(SleepTrainer):
    """Sleep trainer with support for distributed training via Ray/FSDP.

    This is a placeholder for future distributed sleep phase implementation.
    The main challenges are:
    1. Coordinating LoRA application across FSDP shards
    2. Gathering experiences from all workers
    3. Synchronizing weight updates

    For now, sleep phase runs on a single GPU with the full model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distributed = False  # Flag for future implementation

    def run_consolidation_distributed(
        self,
        actor_rollout_wg,  # Ray worker group
        buffer: 'ExperienceBuffer',
        tokenizer,
    ) -> Dict[str, Any]:
        """Run consolidation using distributed workers.

        This is a stub for future implementation. Currently falls back
        to single-GPU training.
        """
        # For now, just use the standard consolidation
        # Future: coordinate with Ray workers for distributed sleep
        print("Warning: Distributed sleep not yet implemented, using single-GPU")
        return {}
