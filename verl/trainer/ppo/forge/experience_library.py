# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
FORGE Experience Library

A hierarchical experience library following FLEX's design:
- Golden Zone: Successful strategies organized by abstraction level
- Warning Zone: Failure patterns for avoiding common mistakes

The library serves as privileged information for self-distillation.
"""

import json
import os
import pickle
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class Experience:
    """A single experience entry in the library."""
    uid: str
    content: str  # The strategy/warning text
    level: str  # 'principle', 'method', or 'example' for golden; 'mistake', 'pattern', 'diagnostic' for warning
    source_trajectory: Optional[str] = None  # Original trajectory text (truncated)
    task_type: Optional[str] = None  # Task category for retrieval
    score: float = 0.0  # Quality score
    usage_count: int = 0  # How many times this experience has been retrieved
    embedding: Optional[np.ndarray] = None  # For similarity-based retrieval
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'uid': self.uid,
            'content': self.content,
            'level': self.level,
            'source_trajectory': self.source_trajectory,
            'task_type': self.task_type,
            'score': self.score,
            'usage_count': self.usage_count,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Experience':
        return cls(
            uid=data['uid'],
            content=data['content'],
            level=data['level'],
            source_trajectory=data.get('source_trajectory'),
            task_type=data.get('task_type'),
            score=data.get('score', 0.0),
            usage_count=data.get('usage_count', 0),
            metadata=data.get('metadata', {}),
        )


class ExperienceZone:
    """Base class for experience zones (Golden or Warning)."""

    def __init__(
        self,
        name: str,
        levels: List[str],
        capacity_per_level: int = 100,
    ):
        self.name = name
        self.levels = levels
        self.capacity_per_level = capacity_per_level
        self.experiences: Dict[str, List[Experience]] = {level: [] for level in levels}

    def add(self, experience: Experience) -> bool:
        """Add an experience to the appropriate level."""
        if experience.level not in self.levels:
            print(f"[ExperienceZone] Warning: Unknown level '{experience.level}' for zone '{self.name}'")
            return False

        level_list = self.experiences[experience.level]

        # Check capacity
        if len(level_list) >= self.capacity_per_level:
            # Find the lowest scored experience
            worst_idx = min(range(len(level_list)), key=lambda i: level_list[i].score)
            worst = level_list[worst_idx]

            # Only replace if new experience is better than the worst
            if experience.score <= worst.score:
                return False  # Don't add - new experience is not better

            # Remove the worst experience to make room
            removed = level_list.pop(worst_idx)
            print(f"[ExperienceZone] Replaced experience (score={removed.score:.2f}) "
                  f"with new (score={experience.score:.2f}) in {self.name}/{experience.level}")

        level_list.append(experience)
        return True

    def get_all(self, level: Optional[str] = None) -> List[Experience]:
        """Get all experiences, optionally filtered by level."""
        if level is not None:
            return self.experiences.get(level, [])

        all_exp = []
        for level_list in self.experiences.values():
            all_exp.extend(level_list)
        return all_exp

    def size(self) -> int:
        """Total number of experiences."""
        return sum(len(level_list) for level_list in self.experiences.values())

    def clear(self):
        """Clear all experiences."""
        for level in self.levels:
            self.experiences[level] = []


class GoldenZone(ExperienceZone):
    """
    Golden Zone: Successful strategies organized by abstraction level.

    Levels:
    - principle: High-level guidelines and principles
    - method: Reasoning patterns and methods
    - example: Concrete examples and facts
    """

    def __init__(self, capacity_per_level: int = 100):
        super().__init__(
            name='golden',
            levels=['principle', 'method', 'example'],
            capacity_per_level=capacity_per_level,
        )


class WarningZone(ExperienceZone):
    """
    Warning Zone: Failure patterns for avoiding common mistakes.

    Levels:
    - mistake: Common mistakes to avoid
    - pattern: Recurring failure patterns
    - diagnostic: Diagnostic insights from failures
    """

    def __init__(self, capacity_per_level: int = 50):
        super().__init__(
            name='warning',
            levels=['mistake', 'pattern', 'diagnostic'],
            capacity_per_level=capacity_per_level,
        )


class ExperienceLibrary:
    """
    FORGE Experience Library

    Hierarchical storage of experiences with:
    - Golden Zone: Successful strategies
    - Warning Zone: Failure patterns

    Supports:
    - Text-based retrieval
    - Embedding-based similarity search (optional)
    - Experience quality scoring
    - Library evolution without gradient updates
    """

    def __init__(
        self,
        golden_capacity_per_level: int = 100,
        warning_capacity_per_level: int = 50,
        use_embeddings: bool = False,
        embedding_model: str = 'all-MiniLM-L6-v2',
    ):
        self.golden = GoldenZone(capacity_per_level=golden_capacity_per_level)
        self.warning = WarningZone(capacity_per_level=warning_capacity_per_level)

        self.use_embeddings = use_embeddings and HAS_SENTENCE_TRANSFORMERS
        self.embedding_model = None

        if self.use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"[ExperienceLibrary] Initialized embedding model: {embedding_model}")
            except Exception as e:
                print(f"[ExperienceLibrary] Failed to load embedding model: {e}")
                self.use_embeddings = False

        # Statistics tracking
        self.total_additions = 0
        self.total_retrievals = 0

    def add_golden(
        self,
        content: str,
        level: str,
        source_trajectory: Optional[str] = None,
        task_type: Optional[str] = None,
        score: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Add a successful strategy to the golden zone."""
        exp = Experience(
            uid=str(uuid.uuid4()),
            content=content,
            level=level,
            source_trajectory=source_trajectory,
            task_type=task_type,
            score=score,
            metadata=metadata or {},
        )

        if self.use_embeddings and self.embedding_model is not None:
            exp.embedding = self.embedding_model.encode(content, convert_to_numpy=True)

        self.golden.add(exp)
        self.total_additions += 1
        return exp.uid

    def add_warning(
        self,
        content: str,
        level: str,
        source_trajectory: Optional[str] = None,
        task_type: Optional[str] = None,
        score: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Add a failure pattern to the warning zone."""
        exp = Experience(
            uid=str(uuid.uuid4()),
            content=content,
            level=level,
            source_trajectory=source_trajectory,
            task_type=task_type,
            score=score,
            metadata=metadata or {},
        )

        if self.use_embeddings and self.embedding_model is not None:
            exp.embedding = self.embedding_model.encode(content, convert_to_numpy=True)

        self.warning.add(exp)
        self.total_additions += 1
        return exp.uid

    def retrieve(
        self,
        query: str,
        task_type: Optional[str] = None,
        top_k_golden: int = 3,
        top_k_warning: int = 2,
        use_similarity: bool = True,
    ) -> Dict[str, List[Experience]]:
        """
        Retrieve relevant experiences for the given query.

        Args:
            query: The query text (e.g., current state description)
            task_type: Optional task type filter
            top_k_golden: Number of golden experiences to retrieve
            top_k_warning: Number of warning experiences to retrieve
            use_similarity: Whether to use embedding similarity (if available)

        Returns:
            Dict with 'golden' and 'warning' lists of experiences
        """
        self.total_retrievals += 1

        golden_experiences = self._retrieve_from_zone(
            self.golden, query, task_type, top_k_golden, use_similarity
        )
        warning_experiences = self._retrieve_from_zone(
            self.warning, query, task_type, top_k_warning, use_similarity
        )

        # Update usage counts
        for exp in golden_experiences + warning_experiences:
            exp.usage_count += 1

        return {
            'golden': golden_experiences,
            'warning': warning_experiences,
        }

    def _retrieve_from_zone(
        self,
        zone: ExperienceZone,
        query: str,
        task_type: Optional[str],
        top_k: int,
        use_similarity: bool,
    ) -> List[Experience]:
        """Retrieve from a specific zone."""
        all_exp = zone.get_all()

        # Filter by task type if specified
        if task_type:
            all_exp = [e for e in all_exp if e.task_type is None or e.task_type == task_type]

        if not all_exp:
            return []

        # Use embedding similarity if available
        if use_similarity and self.use_embeddings and self.embedding_model is not None:
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

            # Compute similarities
            scored = []
            for exp in all_exp:
                if exp.embedding is not None:
                    sim = np.dot(query_embedding, exp.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(exp.embedding) + 1e-8
                    )
                    scored.append((exp, sim))
                else:
                    scored.append((exp, exp.score))

            # Sort by similarity
            scored.sort(key=lambda x: x[1], reverse=True)
            return [exp for exp, _ in scored[:top_k]]
        else:
            # Sort by score
            all_exp.sort(key=lambda x: x.score, reverse=True)
            return all_exp[:top_k]

    def format_for_prompt(
        self,
        retrieved: Dict[str, List[Experience]],
        max_length: int = 2000,
    ) -> str:
        """
        Format retrieved experiences for inclusion in a prompt.

        Args:
            retrieved: Output from retrieve()
            max_length: Maximum character length for the formatted text

        Returns:
            Formatted string for prompt inclusion
        """
        parts = []

        # Format golden experiences
        if retrieved.get('golden'):
            parts.append("=== Successful Strategies ===")
            for i, exp in enumerate(retrieved['golden'], 1):
                level_name = exp.level.capitalize()
                parts.append(f"[{level_name} {i}] {exp.content}")

        # Format warnings
        if retrieved.get('warning'):
            parts.append("\n=== Common Pitfalls to Avoid ===")
            for i, exp in enumerate(retrieved['warning'], 1):
                level_name = exp.level.capitalize()
                parts.append(f"[{level_name} {i}] {exp.content}")

        text = "\n".join(parts)

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length - 3] + "..."

        return text

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            'golden_size': self.golden.size(),
            'warning_size': self.warning.size(),
            'total_size': self.golden.size() + self.warning.size(),
            'total_additions': self.total_additions,
            'total_retrievals': self.total_retrievals,
            'golden_by_level': {
                level: len(exps) for level, exps in self.golden.experiences.items()
            },
            'warning_by_level': {
                level: len(exps) for level, exps in self.warning.experiences.items()
            },
        }

    def is_empty(self) -> bool:
        """Check if the library is empty."""
        return self.golden.size() == 0 and self.warning.size() == 0

    def save(self, path: str):
        """Save the library to disk."""
        data = {
            'golden': [exp.to_dict() for exp in self.golden.get_all()],
            'warning': [exp.to_dict() for exp in self.warning.get_all()],
            'statistics': {
                'total_additions': self.total_additions,
                'total_retrievals': self.total_retrievals,
            }
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[ExperienceLibrary] Saved to {path}")

    def load(self, path: str):
        """Load the library from disk."""
        if not os.path.exists(path):
            print(f"[ExperienceLibrary] No checkpoint found at {path}")
            return

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Load golden experiences
        for exp_dict in data.get('golden', []):
            exp = Experience.from_dict(exp_dict)
            if self.use_embeddings and self.embedding_model is not None:
                exp.embedding = self.embedding_model.encode(exp.content, convert_to_numpy=True)
            self.golden.add(exp)

        # Load warning experiences
        for exp_dict in data.get('warning', []):
            exp = Experience.from_dict(exp_dict)
            if self.use_embeddings and self.embedding_model is not None:
                exp.embedding = self.embedding_model.encode(exp.content, convert_to_numpy=True)
            self.warning.add(exp)

        # Load statistics
        stats = data.get('statistics', {})
        self.total_additions = stats.get('total_additions', 0)
        self.total_retrievals = stats.get('total_retrievals', 0)

        print(f"[ExperienceLibrary] Loaded from {path}: {self.golden.size()} golden, {self.warning.size()} warning")
