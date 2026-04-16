# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
FORGE Step-Level Experience Module

Key innovation for multi-turn tasks:
- Experience retrieval at EACH STEP based on current observation
- Observation-indexed library for efficient retrieval
- Experience-augmented prompts during rollout

This differentiates from single-turn experience RL (e.g., RLEP):
- RLEP: Task-level experience replay
- FORGE: Step-level experience retrieval with observation-based indexing

Integration with GiGPO's step grouping:
- Uses observation clustering to group similar states across trajectories
- Experiences are indexed by observation clusters, not individual observations
- Similar observations → same experience pool (efficiency + generalization)
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

from verl.trainer.ppo.forge.experience_library import ExperienceLibrary, Experience


# =============================================================================
# Observation Clustering (Adapted from GiGPO's step grouping)
# =============================================================================

def are_observations_similar(obs_a: str, obs_b: str, threshold: float = 0.85) -> bool:
    """
    Check if two observations are similar enough to share experiences.

    This mirrors GiGPO's `are_similar()` function for step grouping.
    The intuition: if two observations would share advantage normalization
    in GiGPO, they should also share experience retrieval in FORGE.

    Args:
        obs_a: First observation text
        obs_b: Second observation text
        threshold: Similarity threshold (0.0-1.0)

    Returns:
        True if observations are similar enough to cluster together
    """
    if obs_a == obs_b:
        return True
    # Truncate for efficiency on long observations
    a_truncated = obs_a[:500]
    b_truncated = obs_b[:500]
    return SequenceMatcher(None, a_truncated, b_truncated).ratio() >= threshold


def compute_observation_signature(observation: str, method: str = 'truncate') -> str:
    """
    Compute a signature for an observation for clustering/hashing.

    Different methods trade off between precision and generalization:
    - 'truncate': Use first N characters (fast, good for structured obs)
    - 'hash': Hash the full observation (exact match only)
    - 'normalize': Normalize whitespace and case, then truncate

    Args:
        observation: Raw observation text
        method: Signature computation method

    Returns:
        Signature string for clustering
    """
    if method == 'hash':
        return str(hash(observation))

    # Normalize observation
    normalized = observation.strip()

    if method == 'normalize':
        # Remove extra whitespace, lowercase
        normalized = ' '.join(normalized.lower().split())

    # Truncate to reasonable length for clustering
    return normalized[:300]


class ObservationCluster:
    """
    A cluster of similar observations that share experiences.

    This implements the observation-based indexing that differentiates
    FORGE from single-turn experience RL. In single-turn RL, experiences
    are indexed by task. In multi-turn FORGE, experiences are indexed by
    observation clusters within tasks.

    Analogy to GiGPO:
    - GiGPO groups observations for advantage normalization
    - FORGE groups observations for experience sharing
    """

    def __init__(
        self,
        cluster_id: str,
        representative_obs: str,
        similarity_threshold: float = 0.85,
    ):
        """
        Args:
            cluster_id: Unique identifier for this cluster
            representative_obs: The canonical observation for this cluster
            similarity_threshold: Threshold for membership
        """
        self.cluster_id = cluster_id
        self.representative_obs = representative_obs
        self.similarity_threshold = similarity_threshold

        # Experiences associated with this observation cluster
        self.golden_experience_uids: List[str] = []
        self.warning_experience_uids: List[str] = []

        # Statistics
        self.member_count = 1  # How many observations belong to this cluster
        self.retrieval_count = 0

    def matches(self, observation: str) -> bool:
        """Check if an observation belongs to this cluster."""
        return are_observations_similar(
            observation,
            self.representative_obs,
            self.similarity_threshold
        )

    def add_experience(self, experience_uid: str, is_golden: bool = True):
        """Associate an experience with this observation cluster."""
        if is_golden:
            if experience_uid not in self.golden_experience_uids:
                self.golden_experience_uids.append(experience_uid)
        else:
            if experience_uid not in self.warning_experience_uids:
                self.warning_experience_uids.append(experience_uid)


class ObservationIndex:
    """
    Index of observation clusters for efficient experience retrieval.

    This is the core data structure for step-level experience retrieval.
    Instead of searching the entire library for each observation, we:
    1. Find the observation's cluster (or create a new one)
    2. Retrieve experiences associated with that cluster

    Benefits:
    - O(clusters) lookup instead of O(experiences)
    - Similar observations share experiences (generalization)
    - Aligns with GiGPO's step grouping philosophy
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        enable_similarity: bool = True,
        max_clusters: int = 500,
    ):
        """
        Args:
            similarity_threshold: Threshold for clustering observations
            enable_similarity: If False, use exact match only (like GiGPO default)
            max_clusters: Maximum number of clusters to maintain
        """
        self.similarity_threshold = similarity_threshold
        self.enable_similarity = enable_similarity
        self.max_clusters = max_clusters

        # Cluster storage: signature -> list of clusters
        # Multiple clusters can have the same signature (for similarity-based clustering)
        self._clusters_by_signature: Dict[str, List[ObservationCluster]] = defaultdict(list)
        self._all_clusters: Dict[str, ObservationCluster] = {}  # cluster_id -> cluster

        # Statistics
        self.total_lookups = 0
        self.cluster_hits = 0
        self.new_clusters_created = 0

    def find_or_create_cluster(
        self,
        observation: str,
        task_type: Optional[str] = None,
    ) -> ObservationCluster:
        """
        Find the cluster for an observation, or create a new one.

        This mirrors GiGPO's clustering logic:
        - If enable_similarity=False: exact match only
        - If enable_similarity=True: similarity-based clustering

        Args:
            observation: Observation text
            task_type: Optional task type for scoped clustering

        Returns:
            The matching or newly created cluster
        """
        self.total_lookups += 1

        # Compute signature for initial lookup
        signature = compute_observation_signature(observation)
        if task_type:
            signature = f"{task_type}:{signature}"

        # Look for existing cluster
        candidate_clusters = self._clusters_by_signature.get(signature, [])

        if not self.enable_similarity:
            # Exact match mode (like GiGPO default)
            for cluster in candidate_clusters:
                if cluster.representative_obs == observation:
                    self.cluster_hits += 1
                    cluster.member_count += 1
                    return cluster
        else:
            # Similarity-based matching (like GiGPO with enable_similarity=True)
            for cluster in candidate_clusters:
                if cluster.matches(observation):
                    self.cluster_hits += 1
                    cluster.member_count += 1
                    return cluster

        # No matching cluster found, create new one
        if len(self._all_clusters) >= self.max_clusters:
            self._evict_least_used_cluster()

        cluster_id = f"obs_cluster_{len(self._all_clusters)}_{hash(observation) % 10000}"
        new_cluster = ObservationCluster(
            cluster_id=cluster_id,
            representative_obs=observation,
            similarity_threshold=self.similarity_threshold,
        )

        self._clusters_by_signature[signature].append(new_cluster)
        self._all_clusters[cluster_id] = new_cluster
        self.new_clusters_created += 1

        return new_cluster

    def _evict_least_used_cluster(self):
        """Remove the least-used cluster to make room for new ones."""
        if not self._all_clusters:
            return

        # Find least used
        least_used = min(
            self._all_clusters.values(),
            key=lambda c: (c.retrieval_count, c.member_count)
        )

        # Remove from signature index
        signature = compute_observation_signature(least_used.representative_obs)
        if signature in self._clusters_by_signature:
            self._clusters_by_signature[signature] = [
                c for c in self._clusters_by_signature[signature]
                if c.cluster_id != least_used.cluster_id
            ]

        # Remove from main index
        del self._all_clusters[least_used.cluster_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_clusters': len(self._all_clusters),
            'total_lookups': self.total_lookups,
            'cluster_hits': self.cluster_hits,
            'hit_rate': self.cluster_hits / max(1, self.total_lookups),
            'new_clusters_created': self.new_clusters_created,
        }


# =============================================================================
# Step Experience Manager (Enhanced with Observation Clustering)
# =============================================================================

class StepExperienceManager:
    """
    Manages step-level experience retrieval for multi-turn rollouts.

    ENHANCED VERSION with GiGPO-style observation clustering:
    1. Observation-based retrieval: Finds experiences relevant to current state
    2. Observation clustering: Groups similar observations across trajectories
    3. Cluster-indexed experiences: Similar states share the same experience pool
    4. Prompt augmentation: Injects experiences into observation prompt

    The key insight from GiGPO is that many observations across different
    trajectories represent the "same state" and should share information.
    Just as GiGPO uses this for advantage normalization, FORGE uses it
    for experience sharing.
    """

    def __init__(
        self,
        library: ExperienceLibrary,
        top_k_golden: int = 2,
        top_k_warning: int = 1,
        max_experience_length: int = 800,
        retrieval_mode: str = 'observation',  # 'observation', 'task', 'hybrid', 'clustered'
        enable_clustering: bool = True,
        similarity_threshold: float = 0.85,
        warmup_steps: int = 5,
        min_library_size: int = 10,
        max_experience_tokens: int = 200,
    ):
        """
        Args:
            library: Experience library
            top_k_golden: Number of golden experiences to retrieve per step
            top_k_warning: Number of warning experiences to retrieve per step
            max_experience_length: Max characters for experience text in prompt
            retrieval_mode: How to retrieve experiences
                - 'observation': Based on current observation (multi-turn specific)
                - 'task': Based on task type only (like single-turn)
                - 'hybrid': Both task and observation
                - 'clustered': Use observation clusters (GiGPO-style)
            enable_clustering: Whether to use observation clustering
            similarity_threshold: Threshold for observation similarity (0.0-1.0)
            warmup_steps: Number of training steps before enabling experience injection
            min_library_size: Minimum library size before enabling retrieval
            max_experience_tokens: Approximate max tokens for experience text (to guard prompt budget)
        """
        self.library = library
        self.top_k_golden = top_k_golden
        self.top_k_warning = top_k_warning
        self.max_experience_length = max_experience_length
        self.retrieval_mode = retrieval_mode
        self.enable_clustering = enable_clustering
        self.similarity_threshold = similarity_threshold
        self.warmup_steps = warmup_steps
        self.min_library_size = min_library_size
        self.max_experience_tokens = max_experience_tokens

        # Training step counter (incremented externally)
        self.current_step = 0

        # Observation index for cluster-based retrieval
        self.observation_index = ObservationIndex(
            similarity_threshold=similarity_threshold,
            enable_similarity=enable_clustering,
        )

        # Statistics
        self.total_retrievals = 0
        self.cache_hits = 0
        self.skipped_warmup = 0

        # Simple cache for recent retrievals (observation hash -> experiences)
        self._cache: Dict[str, Dict[str, List[Experience]]] = {}
        self._cache_max_size = 1000

    def retrieve_for_observation(
        self,
        observation: str,
        task_type: Optional[str] = None,
        step_idx: int = 0,
    ) -> Dict[str, List[Experience]]:
        """
        Retrieve relevant experiences for the current observation.

        This is the KEY INNOVATION for multi-turn:
        - Single-turn: Retrieve once per task
        - Multi-turn: Retrieve at EACH STEP based on observation

        Enhanced with GiGPO-style clustering:
        - Similar observations across trajectories share experience pools
        - Cluster membership is computed using the same similarity metric
          that GiGPO uses for step grouping

        Args:
            observation: Current observation text from environment
            task_type: Optional task type for filtering
            step_idx: Current step index (can influence retrieval strategy)

        Returns:
            Dict with 'golden' and 'warning' lists
        """
        self.total_retrievals += 1

        # Warmup: don't inject experiences until library is mature enough
        if self.current_step < self.warmup_steps:
            self.skipped_warmup += 1
            return {'golden': [], 'warning': []}

        library_size = self.library.golden.size() + self.library.warning.size()
        if library_size < self.min_library_size:
            return {'golden': [], 'warning': []}

        if self.library.is_empty():
            return {'golden': [], 'warning': []}

        # Check cache
        cache_key = self._compute_cache_key(observation, task_type)
        if cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key]

        # Use clustered retrieval if enabled
        if self.enable_clustering and self.retrieval_mode == 'clustered':
            retrieved = self._retrieve_via_cluster(observation, task_type)
        else:
            # Standard retrieval modes
            if self.retrieval_mode == 'observation':
                query = observation
            elif self.retrieval_mode == 'task':
                query = task_type or observation[:200]
            else:  # hybrid
                query = f"{task_type or ''} {observation}"

            # Retrieve from library
            retrieved = self.library.retrieve(
                query=query,
                task_type=task_type,
                top_k_golden=self.top_k_golden,
                top_k_warning=self.top_k_warning,
                use_similarity=self.library.use_embeddings,
            )

        # Update cache
        if len(self._cache) < self._cache_max_size:
            self._cache[cache_key] = retrieved

        return retrieved

    def _retrieve_via_cluster(
        self,
        observation: str,
        task_type: Optional[str] = None,
    ) -> Dict[str, List[Experience]]:
        """
        Retrieve experiences using observation clustering.

        This method:
        1. Finds the observation cluster (or creates a new one)
        2. Retrieves experiences associated with that cluster
        3. Falls back to standard retrieval if cluster has no experiences

        The cluster membership uses the same similarity metric as GiGPO's
        step grouping, ensuring consistency between advantage normalization
        and experience sharing.
        """
        # Find or create cluster for this observation
        cluster = self.observation_index.find_or_create_cluster(observation, task_type)
        cluster.retrieval_count += 1

        # Check if cluster has associated experiences
        if cluster.golden_experience_uids or cluster.warning_experience_uids:
            # Retrieve experiences by UID from library
            golden_exps = []
            warning_exps = []

            for uid in cluster.golden_experience_uids[:self.top_k_golden]:
                exp = self._find_experience_by_uid(uid, is_golden=True)
                if exp:
                    golden_exps.append(exp)

            for uid in cluster.warning_experience_uids[:self.top_k_warning]:
                exp = self._find_experience_by_uid(uid, is_golden=False)
                if exp:
                    warning_exps.append(exp)

            if golden_exps or warning_exps:
                return {'golden': golden_exps, 'warning': warning_exps}

        # Fallback: standard retrieval and associate results with cluster
        retrieved = self.library.retrieve(
            query=observation,
            task_type=task_type,
            top_k_golden=self.top_k_golden,
            top_k_warning=self.top_k_warning,
            use_similarity=self.library.use_embeddings,
        )

        # Associate retrieved experiences with this cluster for future lookups
        for exp in retrieved.get('golden', []):
            cluster.add_experience(exp.uid, is_golden=True)
        for exp in retrieved.get('warning', []):
            cluster.add_experience(exp.uid, is_golden=False)

        return retrieved

    def _find_experience_by_uid(
        self,
        uid: str,
        is_golden: bool = True,
    ) -> Optional[Experience]:
        """Find an experience by its UID."""
        zone = self.library.golden if is_golden else self.library.warning
        for exp in zone.get_all():
            if exp.uid == uid:
                return exp
        return None

    def associate_experience_with_observation(
        self,
        observation: str,
        experience_uid: str,
        is_golden: bool = True,
        task_type: Optional[str] = None,
    ):
        """
        Explicitly associate an experience with an observation cluster.

        This can be called after training to link successful/failed
        experiences to the observations where they are most relevant.

        Args:
            observation: The observation text
            experience_uid: UID of the experience to associate
            is_golden: Whether it's a golden or warning experience
            task_type: Optional task type for scoping
        """
        cluster = self.observation_index.find_or_create_cluster(observation, task_type)
        cluster.add_experience(experience_uid, is_golden=is_golden)

    def augment_observation(
        self,
        observation: str,
        task_type: Optional[str] = None,
        step_idx: int = 0,
        format_style: str = 'concise',  # 'concise', 'detailed', 'minimal'
    ) -> str:
        """
        Augment observation with retrieved experiences.

        This creates the experience-augmented prompt for the current step.

        Args:
            observation: Original observation text
            task_type: Task type for retrieval
            step_idx: Current step
            format_style: How to format experiences
                - 'concise': Brief summary (recommended for multi-turn)
                - 'detailed': Full experience text
                - 'minimal': Just key points

        Returns:
            Augmented observation string
        """
        retrieved = self.retrieve_for_observation(observation, task_type, step_idx)

        if not retrieved['golden'] and not retrieved['warning']:
            return observation

        # Format experiences based on style
        exp_text = self._format_experiences(retrieved, format_style)

        if not exp_text:
            return observation

        # Augment observation
        # For multi-turn, we put experiences at the beginning as context
        augmented = f"{exp_text}\n\n{observation}"

        return augmented

    def _format_experiences(
        self,
        retrieved: Dict[str, List[Experience]],
        style: str,
    ) -> str:
        """Format retrieved experiences for prompt inclusion."""
        parts = []
        current_length = 0

        # Format golden experiences (tips for success)
        if retrieved.get('golden'):
            if style == 'minimal':
                tips = [exp.content[:100] for exp in retrieved['golden'][:2]]
                parts.append(f"[Tips: {'; '.join(tips)}]")
            elif style == 'concise':
                parts.append("[Helpful Strategies]")
                for i, exp in enumerate(retrieved['golden'], 1):
                    text = exp.content[:200] if len(exp.content) > 200 else exp.content
                    parts.append(f"- {text}")
                    current_length += len(text)
                    if current_length > self.max_experience_length * 0.6:
                        break
            else:  # detailed
                parts.append("[Successful Strategies]")
                for exp in retrieved['golden']:
                    parts.append(f"[{exp.level}] {exp.content}")

        # Format warnings (pitfalls to avoid)
        if retrieved.get('warning') and current_length < self.max_experience_length * 0.8:
            if style == 'minimal':
                warns = [exp.content[:80] for exp in retrieved['warning'][:1]]
                parts.append(f"[Avoid: {'; '.join(warns)}]")
            elif style == 'concise':
                parts.append("[Common Pitfalls]")
                for exp in retrieved['warning'][:self.top_k_warning]:
                    text = exp.content[:150] if len(exp.content) > 150 else exp.content
                    parts.append(f"- {text}")
            else:
                parts.append("[Warnings]")
                for exp in retrieved['warning']:
                    parts.append(f"- {exp.content}")

        text = "\n".join(parts)

        # Enforce character length limit
        if len(text) > self.max_experience_length:
            text = text[:self.max_experience_length - 3] + "..."

        # Enforce approximate token budget (rough: 1 token ≈ 4 chars)
        max_chars_from_tokens = self.max_experience_tokens * 4
        if len(text) > max_chars_from_tokens:
            text = text[:max_chars_from_tokens - 3] + "..."

        return text

    def _compute_cache_key(self, observation: str, task_type: Optional[str]) -> str:
        """Compute cache key from observation."""
        # Use first 200 chars + task type as key
        obs_key = observation[:200].strip().lower()
        return f"{task_type or 'none'}:{hash(obs_key)}"

    def clear_cache(self):
        """Clear the retrieval cache."""
        self._cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = {
            'total_retrievals': self.total_retrievals,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.total_retrievals),
            'cache_size': len(self._cache),
            'library_size': self.library.golden.size() + self.library.warning.size(),
        }

        # Add clustering statistics if enabled
        if self.enable_clustering:
            stats['observation_clustering'] = self.observation_index.get_statistics()

        return stats


def create_experience_augmented_observation(
    observation: str,
    library: ExperienceLibrary,
    task_type: Optional[str] = None,
    top_k_golden: int = 2,
    top_k_warning: int = 1,
    max_length: int = 800,
) -> str:
    """
    Convenience function for augmenting a single observation.

    Args:
        observation: Current observation text
        library: Experience library
        task_type: Task type for filtering
        top_k_golden: Number of golden experiences
        top_k_warning: Number of warnings
        max_length: Max experience text length

    Returns:
        Augmented observation string
    """
    if library.is_empty():
        return observation

    manager = StepExperienceManager(
        library=library,
        top_k_golden=top_k_golden,
        top_k_warning=top_k_warning,
        max_experience_length=max_length,
    )

    return manager.augment_observation(observation, task_type)


class MultiTurnExperienceTracker:
    """
    Tracks experience usage across a multi-turn episode.

    Used for:
    1. Analyzing which experiences helped
    2. Updating experience quality scores
    3. Identifying gaps in the library
    4. Linking experiences to observation clusters
    """

    def __init__(self):
        self.episode_retrievals: List[Dict[str, Any]] = []
        self.used_experience_uids: List[str] = []
        self.observation_to_experience: Dict[str, List[str]] = defaultdict(list)

    def record_retrieval(
        self,
        step_idx: int,
        observation: str,
        retrieved: Dict[str, List[Experience]],
    ):
        """Record what was retrieved at each step."""
        experience_uids = [
            exp.uid for exp in retrieved.get('golden', []) + retrieved.get('warning', [])
        ]

        self.episode_retrievals.append({
            'step': step_idx,
            'obs_preview': observation[:100],
            'obs_signature': compute_observation_signature(observation),
            'golden_count': len(retrieved.get('golden', [])),
            'warning_count': len(retrieved.get('warning', [])),
            'experience_uids': experience_uids,
        })
        self.used_experience_uids.extend(experience_uids)

        # Track observation -> experience mapping
        obs_sig = compute_observation_signature(observation)
        self.observation_to_experience[obs_sig].extend(experience_uids)

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of experience usage in the episode."""
        return {
            'total_steps': len(self.episode_retrievals),
            'total_retrievals': sum(
                r['golden_count'] + r['warning_count'] for r in self.episode_retrievals
            ),
            'unique_experiences_used': len(set(self.used_experience_uids)),
            'steps_with_experiences': sum(
                1 for r in self.episode_retrievals
                if r['golden_count'] > 0 or r['warning_count'] > 0
            ),
            'unique_observations': len(self.observation_to_experience),
        }

    def get_experience_observation_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping from observation signatures to experience UIDs.

        This can be used to update the experience library's observation index
        after training, linking experiences to the observations where they
        were retrieved.
        """
        return dict(self.observation_to_experience)

    def reset(self):
        """Reset for new episode."""
        self.episode_retrievals = []
        self.used_experience_uids = []
        self.observation_to_experience = defaultdict(list)


# =============================================================================
# Batch Processing for Training Integration
# =============================================================================

def build_observation_clusters_from_batch(
    anchor_observations: List[str],
    step_indices: List[int],
    enable_similarity: bool = False,
    similarity_threshold: float = 0.85,
) -> Dict[str, List[int]]:
    """
    Build observation clusters from a batch of trajectories.

    This mirrors GiGPO's `build_step_group()` function but for experience indexing.
    The key insight is that observations that are grouped together for advantage
    normalization should also share experiences.

    Args:
        anchor_observations: List of observation texts from all trajectories
        step_indices: List of step indices corresponding to each observation
        enable_similarity: Whether to use similarity-based clustering
        similarity_threshold: Threshold for similarity-based clustering

    Returns:
        Dict mapping cluster_id to list of indices in the input lists
    """
    clusters: Dict[str, List[int]] = defaultdict(list)

    # Group by step index first (like GiGPO)
    by_step: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for i, (obs, step_idx) in enumerate(zip(anchor_observations, step_indices)):
        by_step[step_idx].append((i, obs))

    # Within each step, cluster by observation similarity
    for step_idx, obs_list in by_step.items():
        if not enable_similarity:
            # Exact match clustering
            exact_clusters: Dict[str, List[int]] = defaultdict(list)
            for orig_idx, obs in obs_list:
                obs_key = compute_observation_signature(obs, method='hash')
                exact_clusters[f"step{step_idx}:{obs_key}"].append(orig_idx)
            clusters.update(exact_clusters)
        else:
            # Similarity-based clustering
            step_clusters: List[Dict[str, Any]] = []
            for orig_idx, obs in obs_list:
                # Try to find a matching cluster
                matched = False
                for cluster in step_clusters:
                    if are_observations_similar(obs, cluster['rep'], similarity_threshold):
                        cluster['indices'].append(orig_idx)
                        matched = True
                        break

                if not matched:
                    # Create new cluster
                    step_clusters.append({
                        'rep': obs,
                        'indices': [orig_idx],
                    })

            # Convert to output format
            for i, cluster in enumerate(step_clusters):
                cluster_id = f"step{step_idx}:cluster{i}"
                clusters[cluster_id] = cluster['indices']

    return clusters
