# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
Cyclic-FLEX: A Wake-Sleep Framework for Continuous Agent Evolution

This module implements Cyclic-FLEX, which combines:
- Wake Phase: GRPO/GiGPO-style inference-time adaptation with experience accumulation
- Sleep Phase: Periodic weight consolidation via LoRA fine-tuning

Key Components:
- ExperienceBuffer: Short-term memory for storing successful trajectories
- SleepTrainer: LoRA-based weight consolidation during sleep phase
- ReflectionGenerator: Distills lessons from trajectory comparisons
- core_cyclic_flex: Advantage computation functions
"""

from verl.trainer.ppo.cyclic_flex.experience_buffer import ExperienceBuffer, ExperienceItem
from verl.trainer.ppo.cyclic_flex.sleep_trainer import SleepTrainer
from verl.trainer.ppo.cyclic_flex.reflection_generator import ReflectionGenerator
from verl.trainer.ppo.cyclic_flex import core_cyclic_flex

__all__ = [
    'ExperienceBuffer',
    'ExperienceItem',
    'SleepTrainer',
    'ReflectionGenerator',
    'core_cyclic_flex',
]
