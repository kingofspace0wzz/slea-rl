# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
FORGE Experience Evolution

Forward learning module that evolves the experience library without gradient updates.
Follows FLEX's paradigm: learning happens through semantic extraction and library updates.

Key operations:
1. Strategy extraction from successful trajectories
2. Failure diagnosis from failed trajectories
3. Experience merging to avoid redundancy
4. Quality scoring for prioritization
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from verl.trainer.ppo.forge.experience_library import ExperienceLibrary, Experience


class ExperienceEvolver:
    """
    Forward-learning experience evolution.

    Updates the experience library by:
    1. Extracting strategies from successful trajectories
    2. Diagnosing failures from failed trajectories
    3. Merging similar experiences
    4. Pruning low-quality experiences

    No gradient updates - purely semantic evolution.
    """

    def __init__(
        self,
        library: ExperienceLibrary,
        extraction_model: Optional[Any] = None,  # Optional LLM for extraction
        similarity_threshold: float = 0.85,
        min_score_threshold: float = 0.5,
        max_strategies_per_step: int = 10,
        max_warnings_per_step: int = 5,
        top_k_trajectories: int = 5,
    ):
        """
        Args:
            library: The experience library to evolve
            extraction_model: Optional LLM model for semantic extraction
            similarity_threshold: Threshold for merging similar experiences
            min_score_threshold: Minimum score for successful trajectory
            max_strategies_per_step: Maximum new strategies to add per evolution step
            max_warnings_per_step: Maximum new warnings to add per evolution step
            top_k_trajectories: Only extract from top-k scored trajectories
        """
        self.library = library
        self.extraction_model = extraction_model
        self.similarity_threshold = similarity_threshold
        self.min_score_threshold = min_score_threshold
        self.max_strategies_per_step = max_strategies_per_step
        self.max_warnings_per_step = max_warnings_per_step
        self.top_k_trajectories = top_k_trajectories

        # Track evolution statistics
        self.strategies_extracted = 0
        self.warnings_extracted = 0
        self.experiences_merged = 0

    def evolve_from_trajectories(
        self,
        successful_trajectories: List[Dict[str, Any]],
        failed_trajectories: List[Dict[str, Any]],
        task_type: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Main evolution function. Updates library based on trajectories.

        Args:
            successful_trajectories: List of successful trajectory data
            failed_trajectories: List of failed trajectory data
            task_type: Optional task type for categorization

        Returns:
            Statistics dict with counts of added experiences
        """
        stats = {
            'strategies_added': 0,
            'warnings_added': 0,
            'strategies_merged': 0,
            'warnings_merged': 0,
        }

        # Only use top-k trajectories by score to avoid flooding
        if len(successful_trajectories) > self.top_k_trajectories:
            successful_trajectories = sorted(
                successful_trajectories,
                key=lambda t: t.get('score', t.get('reward', 0.0)),
                reverse=True,
            )[:self.top_k_trajectories]

        if len(failed_trajectories) > self.top_k_trajectories:
            failed_trajectories = sorted(
                failed_trajectories,
                key=lambda t: t.get('score', t.get('reward', 0.0)),
            )[:self.top_k_trajectories]  # Worst failures first

        # Extract strategies from successes (rate-limited)
        for traj in successful_trajectories:
            if stats['strategies_added'] >= self.max_strategies_per_step:
                break
            strategies = self.extract_strategies(traj, task_type)
            for strategy, level, score in strategies:
                if self._is_novel(strategy, score=score, is_golden=True):
                    added = self.library.add_golden(
                        content=strategy,
                        level=level,
                        source_trajectory=self._truncate_trajectory(traj),
                        task_type=task_type,
                        score=score,
                    )
                    if added:
                        stats['strategies_added'] += 1
                        self.strategies_extracted += 1
                    else:
                        stats['strategies_merged'] += 1
                else:
                    stats['strategies_merged'] += 1
                    self.experiences_merged += 1

        # Extract warnings from failures (rate-limited)
        for traj in failed_trajectories:
            if stats['warnings_added'] >= self.max_warnings_per_step:
                break
            warnings = self.extract_warnings(traj, task_type)
            for warning, level, score in warnings:
                if self._is_novel(warning, score=score, is_golden=False):
                    added = self.library.add_warning(
                        content=warning,
                        level=level,
                        source_trajectory=self._truncate_trajectory(traj),
                        task_type=task_type,
                        score=score,
                    )
                    if added:
                        stats['warnings_added'] += 1
                        self.warnings_extracted += 1
                    else:
                        stats['warnings_merged'] += 1
                else:
                    stats['warnings_merged'] += 1
                    self.experiences_merged += 1

        return stats

    def extract_strategies(
        self,
        trajectory: Dict[str, Any],
        task_type: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Extract strategies from a successful trajectory.

        Returns:
            List of (strategy_text, level, score) tuples
        """
        strategies = []

        # Get trajectory text and score
        traj_text = trajectory.get('text', trajectory.get('trajectory', ''))
        traj_score = trajectory.get('score', trajectory.get('reward', 1.0))

        if isinstance(traj_score, (list, np.ndarray)):
            traj_score = float(np.sum(traj_score))

        # Use LLM extraction if available
        if self.extraction_model is not None:
            strategies.extend(self._llm_extract_strategies(traj_text, task_type))
        else:
            # Rule-based extraction
            strategies.extend(self._rule_extract_strategies(traj_text, traj_score))

        return strategies

    def extract_warnings(
        self,
        trajectory: Dict[str, Any],
        task_type: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Extract warnings from a failed trajectory.

        Returns:
            List of (warning_text, level, score) tuples
        """
        warnings = []

        # Get trajectory text and score
        traj_text = trajectory.get('text', trajectory.get('trajectory', ''))
        traj_score = trajectory.get('score', trajectory.get('reward', 0.0))

        if isinstance(traj_score, (list, np.ndarray)):
            traj_score = float(np.sum(traj_score))

        # Use LLM extraction if available
        if self.extraction_model is not None:
            warnings.extend(self._llm_extract_warnings(traj_text, task_type))
        else:
            # Rule-based extraction
            warnings.extend(self._rule_extract_warnings(traj_text, traj_score))

        return warnings

    def _rule_extract_strategies(
        self,
        traj_text: str,
        traj_score: float,
    ) -> List[Tuple[str, str, float]]:
        """Rule-based strategy extraction."""
        strategies = []

        # Extract successful action sequences
        actions = self._extract_actions(traj_text)
        if actions:
            # Create method-level strategy
            action_summary = f"Action sequence that led to success: {' -> '.join(actions[:5])}"
            if len(actions) > 5:
                action_summary += f" (and {len(actions) - 5} more steps)"
            strategies.append((action_summary, 'method', traj_score))

        # Extract key decision points
        decisions = self._extract_key_decisions(traj_text)
        for decision in decisions:
            strategies.append((decision, 'example', traj_score * 0.8))

        # Create principle if score is high
        if traj_score >= 0.8:
            principle = self._generate_principle(traj_text)
            if principle:
                strategies.append((principle, 'principle', traj_score))

        return strategies

    def _rule_extract_warnings(
        self,
        traj_text: str,
        traj_score: float,
    ) -> List[Tuple[str, str, float]]:
        """Rule-based warning extraction."""
        warnings = []

        # Extract failed actions
        actions = self._extract_actions(traj_text)
        if actions:
            # Create mistake-level warning
            last_actions = actions[-3:] if len(actions) >= 3 else actions
            warning = f"Avoid this action pattern: {' -> '.join(last_actions)}"
            warnings.append((warning, 'mistake', abs(traj_score)))

        # Detect common failure patterns
        patterns = self._detect_failure_patterns(traj_text)
        for pattern in patterns:
            warnings.append((pattern, 'pattern', 0.5))

        # Create diagnostic insight
        diagnostic = self._generate_diagnostic(traj_text)
        if diagnostic:
            warnings.append((diagnostic, 'diagnostic', 0.3))

        return warnings

    def _extract_actions(self, text: str) -> List[str]:
        """Extract action strings from trajectory text."""
        # Common patterns for actions in agent environments
        action_patterns = [
            r'Action:\s*(.+?)(?:\n|$)',
            r'action:\s*(.+?)(?:\n|$)',
            r'>\s*(.+?)(?:\n|$)',
            r'\[Action\]\s*(.+?)(?:\n|$)',
        ]

        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend([m.strip() for m in matches if m.strip()])

        return actions

    def _extract_key_decisions(self, text: str) -> List[str]:
        """Extract key decision points from trajectory."""
        decisions = []

        # Look for reasoning patterns
        reasoning_patterns = [
            r'I (?:should|need to|will)\s+(.+?)(?:\.|$)',
            r'(?:Therefore|So|Thus),?\s+(.+?)(?:\.|$)',
            r'The (?:best|correct|right) (?:approach|action|step) is\s+(.+?)(?:\.|$)',
        ]

        for pattern in reasoning_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:  # Limit to 2 per pattern
                if len(match) > 10:  # Filter too short
                    decisions.append(f"Key decision: {match.strip()}")

        return decisions

    def _detect_failure_patterns(self, text: str) -> List[str]:
        """Detect common failure patterns in trajectory."""
        patterns = []

        # Check for repetitive actions
        actions = self._extract_actions(text)
        if actions:
            # Detect loops (same action repeated)
            for i in range(len(actions) - 2):
                if actions[i] == actions[i + 1] == actions[i + 2]:
                    patterns.append(f"Avoid repeating the same action '{actions[i]}' multiple times")
                    break

        # Check for common error indicators
        error_indicators = [
            (r'invalid', "Avoid invalid actions - check available options first"),
            (r'nothing happens', "If nothing happens, try a different approach"),
            (r'can\'t|cannot', "When an action is blocked, find alternatives"),
            (r'error|failed|failure', "This approach leads to errors"),
        ]

        text_lower = text.lower()
        for pattern, warning in error_indicators:
            if re.search(pattern, text_lower):
                patterns.append(warning)
                break  # Limit to one

        return patterns

    def _generate_principle(self, text: str) -> Optional[str]:
        """Generate a high-level principle from successful trajectory."""
        # Simple heuristic-based principle generation
        text_lower = text.lower()

        if 'examine' in text_lower or 'look' in text_lower:
            return "Examine objects before interacting with them"
        elif 'pick' in text_lower and 'put' in text_lower:
            return "Pick up items before placing them in target locations"
        elif 'open' in text_lower:
            return "Open containers before accessing their contents"
        elif 'go to' in text_lower or 'navigate' in text_lower:
            return "Navigate to the correct location before performing actions"

        return None

    def _generate_diagnostic(self, text: str) -> Optional[str]:
        """Generate diagnostic insight from failed trajectory."""
        text_lower = text.lower()

        if 'max' in text_lower and 'step' in text_lower:
            return "Task failed due to exceeding step limit - find more efficient paths"
        elif 'invalid' in text_lower:
            return "Task failed due to invalid actions - verify action validity before executing"

        return "Task failed - review the approach and try alternative strategies"

    def _is_novel(self, content: str, score: float, is_golden: bool) -> bool:
        """
        Check if content is novel (not too similar to existing experiences).

        Uses SequenceMatcher (same as GiGPO's observation similarity) for
        more robust duplicate detection than word overlap alone.

        If a similar experience exists and the new one has a higher score,
        update the existing experience's score instead of adding a duplicate.
        """
        zone = self.library.golden if is_golden else self.library.warning
        existing = zone.get_all()

        if not existing:
            return True

        content_lower = content.lower().strip()
        # Truncate for efficiency in comparison
        content_truncated = content_lower[:300]

        for exp in existing:
            exp_lower = exp.content.lower().strip()
            exp_truncated = exp_lower[:300]

            # Use SequenceMatcher for more robust similarity (matches GiGPO approach)
            similarity = SequenceMatcher(None, content_truncated, exp_truncated).ratio()
            if similarity > self.similarity_threshold:
                # Update existing experience score if new one is better
                if score > exp.score:
                    exp.score = score
                return False

            # Also check word overlap as a secondary filter
            overlap = self._compute_text_overlap(content_lower, exp_lower)
            if overlap > self.similarity_threshold:
                if score > exp.score:
                    exp.score = score
                return False

        return True

    def _compute_text_overlap(self, text1: str, text2: str) -> float:
        """Compute simple text overlap ratio."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _truncate_trajectory(self, trajectory: Dict[str, Any], max_length: int = 500) -> str:
        """Truncate trajectory text for storage."""
        text = trajectory.get('text', trajectory.get('trajectory', str(trajectory)))
        if len(text) > max_length:
            return text[:max_length - 3] + "..."
        return text

    def _llm_extract_strategies(
        self,
        traj_text: str,
        task_type: Optional[str],
    ) -> List[Tuple[str, str, float]]:
        """Use LLM to extract strategies (if extraction_model is set)."""
        # Placeholder for LLM-based extraction
        # In practice, this would call the LLM with a prompt like:
        # "Extract key strategies from this successful trajectory..."
        return self._rule_extract_strategies(traj_text, 1.0)

    def _llm_extract_warnings(
        self,
        traj_text: str,
        task_type: Optional[str],
    ) -> List[Tuple[str, str, float]]:
        """Use LLM to extract warnings (if extraction_model is set)."""
        # Placeholder for LLM-based extraction
        return self._rule_extract_warnings(traj_text, 0.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            'strategies_extracted': self.strategies_extracted,
            'warnings_extracted': self.warnings_extracted,
            'experiences_merged': self.experiences_merged,
            'library_stats': self.library.get_statistics(),
        }


def evolve_library_from_batch(
    library: ExperienceLibrary,
    batch,  # DataProto
    scores: 'torch.Tensor',
    tokenizer,
    success_threshold: float = 0.5,
    task_type: Optional[str] = None,
) -> Dict[str, int]:
    """
    Convenience function to evolve library from a training batch.

    Args:
        library: Experience library to update
        batch: DataProto batch from rollout
        scores: Trajectory scores (bs,)
        tokenizer: Tokenizer for decoding
        success_threshold: Score threshold for success
        task_type: Environment/task type

    Returns:
        Evolution statistics
    """
    import torch
    import numpy as np

    evolver = ExperienceEvolver(library)

    # Separate successful and failed trajectories
    successful = []
    failed = []

    # Get unique trajectories
    traj_uids = batch.non_tensor_batch.get('traj_uid', np.arange(len(scores)))
    unique_uids, unique_idx = np.unique(traj_uids, return_index=True)

    for i, uid in enumerate(unique_uids):
        idx = unique_idx[i]
        score = scores[idx].item() if isinstance(scores[idx], torch.Tensor) else scores[idx]

        # Decode trajectory text
        if 'responses' in batch.batch:
            response_ids = batch.batch['responses'][idx]
            traj_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        else:
            traj_text = str(batch.non_tensor_batch.get('text', [''])[idx])

        traj_data = {
            'text': traj_text,
            'score': score,
            'uid': uid,
        }

        if score >= success_threshold:
            successful.append(traj_data)
        else:
            failed.append(traj_data)

    # Evolve library
    stats = evolver.evolve_from_trajectories(
        successful_trajectories=successful,
        failed_trajectories=failed,
        task_type=task_type,
    )

    return stats
