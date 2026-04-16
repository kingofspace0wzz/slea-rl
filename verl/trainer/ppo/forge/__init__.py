# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
FORGE: Forward-learning Optimized RL with Guided Experience

A training framework that combines:
1. Parallel exploration (time-efficient sampling)
2. Experience evolution (forward learning, no SFT)
3. Experience-guided self-distillation (dense token-level signal)

Key Components:
- ExperienceLibrary: Hierarchical storage with Golden/Warning zones
- ExperienceEvolver: Forward learning updates to library
- ForgeDistillationManager: Self-distillation with library as privileged info
- core_forge: Advantage computation (reuses GiGPO patterns)

The key insight is using the experience library as privileged information
for self-distillation, enabling dense learning signals without requiring
reference trajectories or ground truth.
"""

from verl.trainer.ppo.forge.experience_library import (
    Experience,
    ExperienceLibrary,
    ExperienceZone,
    GoldenZone,
    WarningZone,
)
from verl.trainer.ppo.forge.experience_evolution import (
    ExperienceEvolver,
    evolve_library_from_batch,
)
from verl.trainer.ppo.forge.self_distillation import (
    ForgeDistillationManager,
    TeacherStudentDistillation,
    compute_distillation_loss,
    compute_forge_distillation_loss_batch,
)
from verl.trainer.ppo.forge.step_experience import (
    StepExperienceManager,
    MultiTurnExperienceTracker,
    create_experience_augmented_observation,
    # Observation Clustering (GiGPO-inspired)
    ObservationCluster,
    ObservationIndex,
    are_observations_similar,
    compute_observation_signature,
    build_observation_clusters_from_batch,
)
from verl.trainer.ppo.forge import core_forge

__all__ = [
    # Experience Library
    'Experience',
    'ExperienceLibrary',
    'ExperienceZone',
    'GoldenZone',
    'WarningZone',
    # Evolution
    'ExperienceEvolver',
    'evolve_library_from_batch',
    # Step-level Experience (Multi-Turn Specific)
    'StepExperienceManager',
    'MultiTurnExperienceTracker',
    'create_experience_augmented_observation',
    # Observation Clustering (GiGPO-inspired)
    'ObservationCluster',
    'ObservationIndex',
    'are_observations_similar',
    'compute_observation_signature',
    'build_observation_clusters_from_batch',
    # Distillation (Optional, can be disabled)
    'ForgeDistillationManager',
    'TeacherStudentDistillation',
    'compute_distillation_loss',
    'compute_forge_distillation_loss_batch',
    # Core algorithms
    'core_forge',
]
