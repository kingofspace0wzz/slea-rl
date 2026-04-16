# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
Experience Buffer for Cyclic-FLEX wake-sleep cycles.

The ExperienceBuffer serves as short-term memory during the wake phase,
accumulating successful trajectories and their reflections. When the buffer
reaches capacity, it triggers the sleep phase for weight consolidation.
"""

import pickle
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import torch

from verl import DataProto


@dataclass
class ExperienceItem:
    """A single unit of learned knowledge compatible with DataProto.

    Attributes:
        query: The input problem/observation (raw prompt text)
        trajectory_tokens: Token IDs of the winning trajectory response
        reflection: Distilled lesson/strategy from this experience
        score: Reward or advantage score achieved
        anchor_obs: Anchor observation for GiGPO-style step grouping
        env_name: Environment source (alfworld, webshop, search, sokoban)
        metadata: Additional information (timestamps, trajectory uid, etc.)
        embedding: Optional precomputed embedding for retrieval
    """
    query: str
    trajectory_tokens: Optional[torch.Tensor]
    reflection: str
    score: float
    anchor_obs: str = ""
    env_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'trajectory_tokens': self.trajectory_tokens.cpu().numpy().tolist() if self.trajectory_tokens is not None else None,
            'reflection': self.reflection,
            'score': self.score,
            'anchor_obs': self.anchor_obs,
            'env_name': self.env_name,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperienceItem':
        """Create from dictionary."""
        traj_tokens = None
        if data.get('trajectory_tokens') is not None:
            traj_tokens = torch.tensor(data['trajectory_tokens'])
        return cls(
            query=data['query'],
            trajectory_tokens=traj_tokens,
            reflection=data['reflection'],
            score=data['score'],
            anchor_obs=data.get('anchor_obs', ''),
            env_name=data.get('env_name', ''),
            metadata=data.get('metadata', {}),
        )


class ExperienceBuffer:
    """Managed container for experiences with capacity limits and semantic retrieval.

    The buffer implements:
    - FIFO eviction when capacity is exceeded
    - Semantic similarity retrieval for ICL context
    - Conversion to SFT dataset format for sleep phase
    - Checkpoint save/load for persistence

    Attributes:
        capacity: Maximum number of experiences before triggering sleep (C_MAX)
        retrieval_k: Number of experiences to retrieve for ICL context
        items: List of stored ExperienceItem objects
    """

    def __init__(
        self,
        capacity: int = 100,
        retrieval_k: int = 5,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_score_threshold: float = 0.0,
    ):
        """Initialize the experience buffer.

        Args:
            capacity: Maximum experiences before sleep phase triggers
            retrieval_k: Number of experiences to retrieve for ICL
            embedding_model_name: Model for semantic embeddings
            min_score_threshold: Minimum score for experience to be stored
        """
        self.capacity = capacity
        self.retrieval_k = retrieval_k
        self.embedding_model_name = embedding_model_name
        self.min_score_threshold = min_score_threshold

        self.items: List[ExperienceItem] = []
        self._embedding_model = None  # Lazy load
        self._embeddings_cache: Optional[np.ndarray] = None

        # Statistics
        self.total_added = 0
        self.total_evicted = 0

    def add(self, item: ExperienceItem) -> bool:
        """Add an experience item to the buffer.

        Args:
            item: ExperienceItem to add

        Returns:
            True if item was added, False if filtered out
        """
        # Filter by minimum score
        if item.score < self.min_score_threshold:
            return False

        self.items.append(item)
        self._embeddings_cache = None  # Invalidate cache
        self.total_added += 1

        # FIFO eviction if over capacity
        while len(self.items) > self.capacity:
            self.items.pop(0)
            self.total_evicted += 1

        return True

    def add_from_batch(
        self,
        batch: DataProto,
        reflections: List[str],
        scores: torch.Tensor,
        env_name: str,
        only_best_per_group: bool = True,
    ) -> int:
        """Add experiences extracted from a training batch.

        Extracts high-scoring trajectories from the batch and stores them
        as experiences with their associated reflections.

        Args:
            batch: DataProto containing trajectories
            reflections: List of reflection strings for each trajectory
            scores: Tensor of scores for each trajectory
            env_name: Name of the environment
            only_best_per_group: If True, only add the best trajectory from each uid group

        Returns:
            Number of experiences actually added
        """
        added = 0
        batch_size = len(batch)

        # Get uid for grouping
        uids = batch.non_tensor_batch.get('uid', np.arange(batch_size))

        if only_best_per_group:
            # Group by uid and find best trajectory in each group
            uid_to_best = {}  # uid -> (index, score)
            for i in range(batch_size):
                uid = uids[i]
                score_val = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]

                if uid not in uid_to_best or score_val > uid_to_best[uid][1]:
                    uid_to_best[uid] = (i, score_val)

            # Only process the best trajectory from each group
            indices_to_add = [idx for idx, _ in uid_to_best.values()]
        else:
            indices_to_add = list(range(batch_size))

        for i in indices_to_add:
            item = batch[i]
            score_val = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]

            # Extract query from raw_prompt
            raw_prompt = item.non_tensor_batch.get('raw_prompt', [''])[0] if 'raw_prompt' in item.non_tensor_batch else ''
            if isinstance(raw_prompt, (list, np.ndarray)):
                raw_prompt = raw_prompt[0] if len(raw_prompt) > 0 else ''

            # Extract trajectory tokens
            trajectory_tokens = None
            if 'responses' in item.batch:
                trajectory_tokens = item.batch['responses'].clone()

            # Extract anchor observation for step-level grouping
            anchor_obs = ''
            if 'anchor_obs' in item.non_tensor_batch:
                anchor_obs_data = item.non_tensor_batch['anchor_obs']
                if isinstance(anchor_obs_data, (list, np.ndarray)):
                    anchor_obs = str(anchor_obs_data[0]) if len(anchor_obs_data) > 0 else ''
                else:
                    anchor_obs = str(anchor_obs_data)

            # Create experience item
            exp = ExperienceItem(
                query=str(raw_prompt),
                trajectory_tokens=trajectory_tokens,
                reflection=reflections[i] if i < len(reflections) else "",
                score=score_val,
                anchor_obs=anchor_obs,
                env_name=env_name,
                metadata={
                    'traj_uid': item.non_tensor_batch.get('traj_uid', [None])[0] if 'traj_uid' in item.non_tensor_batch else None,
                    'uid': item.non_tensor_batch.get('uid', [None])[0] if 'uid' in item.non_tensor_batch else None,
                    'batch_idx': i,
                }
            )

            if self.add(exp):
                added += 1

        return added

    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return len(self.items) >= self.capacity

    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                print("Warning: sentence-transformers not installed. Using random retrieval.")
                return None
        return self._embedding_model

    def _compute_embeddings(self) -> Optional[np.ndarray]:
        """Compute embeddings for all items if model available."""
        if self._embeddings_cache is not None:
            return self._embeddings_cache

        model = self._get_embedding_model()
        if model is None:
            return None

        texts = [item.query for item in self.items]
        if not texts:
            return None

        self._embeddings_cache = model.encode(texts, convert_to_numpy=True)
        return self._embeddings_cache

    def retrieve(self, query: str, k: Optional[int] = None) -> List[ExperienceItem]:
        """Retrieve top-k relevant experiences using semantic similarity.

        Args:
            query: Query string to match against
            k: Number of experiences to retrieve (defaults to self.retrieval_k)

        Returns:
            List of most relevant ExperienceItem objects
        """
        if not self.items:
            return []

        k = k or self.retrieval_k
        k = min(k, len(self.items))

        # Try semantic retrieval
        embeddings = self._compute_embeddings()
        if embeddings is not None:
            model = self._get_embedding_model()
            query_emb = model.encode([query], convert_to_numpy=True)[0]
            similarities = np.dot(embeddings, query_emb)
            top_indices = np.argsort(similarities)[-k:][::-1]
            return [self.items[i] for i in top_indices]

        # Fallback to random sampling
        import random
        return random.sample(self.items, k)

    def retrieve_by_env(self, env_name: str, k: Optional[int] = None) -> List[ExperienceItem]:
        """Retrieve experiences filtered by environment.

        Args:
            env_name: Environment name to filter by
            k: Maximum number to retrieve

        Returns:
            List of ExperienceItem objects from specified environment
        """
        k = k or self.retrieval_k
        env_items = [item for item in self.items if item.env_name == env_name]

        if not env_items:
            return []

        # Return highest scoring items from this environment
        sorted_items = sorted(env_items, key=lambda x: x.score, reverse=True)
        return sorted_items[:k]

    def format_experiences_for_prompt(
        self,
        experiences: List[ExperienceItem],
        include_reflection: bool = True,
        max_length: int = 2000,
    ) -> str:
        """Format experiences as context for ICL prompting.

        Args:
            experiences: List of experiences to format
            include_reflection: Whether to include reflection text
            max_length: Maximum total character length

        Returns:
            Formatted string for prompt injection
        """
        if not experiences:
            return ""

        lines = ["Here are some relevant past experiences that may help:"]
        current_length = len(lines[0])

        for i, exp in enumerate(experiences, 1):
            if include_reflection and exp.reflection:
                line = f"\n[Experience {i}] (Score: {exp.score:.2f})\nStrategy: {exp.reflection}"
            else:
                line = f"\n[Experience {i}] (Score: {exp.score:.2f})"

            if current_length + len(line) > max_length:
                break

            lines.append(line)
            current_length += len(line)

        return "\n".join(lines)

    def to_sft_dataset(self, tokenizer) -> Dict[str, torch.Tensor]:
        """Convert buffer to SFT-ready format for sleep phase consolidation.

        The format trains on reflections to internalize strategies:
        - Input: Query
        - Target: Reflection + Response trajectory

        Args:
            tokenizer: Tokenizer for encoding

        Returns:
            Dictionary with 'input_ids', 'labels', 'attention_mask' tensors
        """
        if not self.items:
            return {
                'input_ids': torch.empty(0),
                'labels': torch.empty(0),
                'attention_mask': torch.empty(0),
            }

        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for item in self.items:
            # Skip items without trajectory tokens
            if item.trajectory_tokens is None:
                continue

            # Format prompt
            prompt = f"Query: {item.query[:1000]}"  # Truncate long queries

            # Format target with reflection
            if item.reflection:
                target_prefix = f"\n\nStrategy: {item.reflection}\n\nResponse: "
            else:
                target_prefix = "\n\nResponse: "

            # Tokenize prompt (will be masked in labels)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            target_prefix_tokens = tokenizer.encode(target_prefix, add_special_tokens=False)

            # Get trajectory tokens
            traj = item.trajectory_tokens
            if traj.dim() > 1:
                traj = traj.squeeze(0)
            traj = traj.cpu().tolist()

            # Combine: prompt + target_prefix + trajectory
            full_input = prompt_tokens + target_prefix_tokens + traj

            # Labels: -100 for prompt (masked), actual tokens for target
            labels = [-100] * len(prompt_tokens) + target_prefix_tokens + traj

            input_ids_list.append(torch.tensor(full_input, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(full_input), dtype=torch.long))

        if not input_ids_list:
            return {
                'input_ids': torch.empty(0),
                'labels': torch.empty(0),
                'attention_mask': torch.empty(0),
            }

        # Pad to same length
        max_len = max(len(x) for x in input_ids_list)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        padded_inputs = []
        padded_labels = []
        padded_masks = []

        for inp, lbl, mask in zip(input_ids_list, labels_list, attention_mask_list):
            pad_len = max_len - len(inp)
            padded_inputs.append(torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
            padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]))
            padded_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))

        return {
            'input_ids': torch.stack(padded_inputs),
            'labels': torch.stack(padded_labels),
            'attention_mask': torch.stack(padded_masks),
        }

    def flush(self) -> int:
        """Clear buffer after sleep phase.

        Returns:
            Number of items that were flushed
        """
        count = len(self.items)
        self.items.clear()
        self._embeddings_cache = None
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        env_counts = {}
        for item in self.items:
            env_counts[item.env_name] = env_counts.get(item.env_name, 0) + 1

        scores = [item.score for item in self.items]

        return {
            'size': len(self.items),
            'capacity': self.capacity,
            'utilization': len(self.items) / self.capacity if self.capacity > 0 else 0,
            'total_added': self.total_added,
            'total_evicted': self.total_evicted,
            'env_distribution': env_counts,
            'avg_score': np.mean(scores) if scores else 0,
            'min_score': np.min(scores) if scores else 0,
            'max_score': np.max(scores) if scores else 0,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save buffer state to file.

        Args:
            path: File path for checkpoint
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        data = {
            'items': [item.to_dict() for item in self.items],
            'capacity': self.capacity,
            'retrieval_k': self.retrieval_k,
            'min_score_threshold': self.min_score_threshold,
            'total_added': self.total_added,
            'total_evicted': self.total_evicted,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_checkpoint(self, path: str) -> None:
        """Load buffer state from file.

        Args:
            path: File path to load from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.items = [ExperienceItem.from_dict(d) for d in data['items']]
        self.capacity = data.get('capacity', self.capacity)
        self.retrieval_k = data.get('retrieval_k', self.retrieval_k)
        self.min_score_threshold = data.get('min_score_threshold', self.min_score_threshold)
        self.total_added = data.get('total_added', 0)
        self.total_evicted = data.get('total_evicted', 0)
        self._embeddings_cache = None

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"ExperienceBuffer(size={len(self)}, capacity={self.capacity}, utilization={len(self)/self.capacity:.1%})"
