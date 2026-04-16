# Copyright 2025 Anthropic
# Licensed under the Apache License, Version 2.0

"""
Reflection Generator for Cyclic-FLEX experience distillation.

The ReflectionGenerator distills lessons from trajectory comparisons,
creating concise strategic insights that can be:
1. Stored in the experience buffer for retrieval
2. Used as training targets in the sleep phase
3. Injected as context during future wake phase rollouts

Inspired by the "Self-Reflection" mechanism in Training-Free GRPO and
Early Experience papers.
"""

from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
import torch
from verl import DataProto


# Environment-specific reflection prompts
REFLECTION_PROMPTS = {
    'alfworld': {
        'system': """You are an expert at analyzing agent behavior in embodied AI tasks.
Given a successful and unsuccessful attempt at the same task, identify the key strategic difference that led to success.""",

        'template': """Task: {task}

Successful attempt (Score: {win_score:.2f}):
{winner_summary}

Failed attempt (Score: {lose_score:.2f}):
{loser_summary}

Provide a concise strategic insight (1-2 sentences) explaining what made the successful attempt work.
Focus on actionable advice that could help solve similar tasks.""",
    },

    'webshop': {
        'system': """You are an expert at analyzing e-commerce shopping strategies.
Given a successful and unsuccessful shopping attempt, identify the key strategic difference.""",

        'template': """Shopping Task: {task}

Successful attempt (Score: {win_score:.2f}):
{winner_summary}

Failed attempt (Score: {lose_score:.2f}):
{loser_summary}

Provide a concise strategic insight (1-2 sentences) explaining the effective shopping strategy.
Focus on product selection, navigation, or search strategies.""",
    },

    'search': {
        'system': """You are an expert at analyzing information retrieval strategies.
Given a successful and unsuccessful search attempt to answer a question, identify the key difference.""",

        'template': """Question: {task}

Successful attempt (Score: {win_score:.2f}):
{winner_summary}

Failed attempt (Score: {lose_score:.2f}):
{loser_summary}

Provide a concise strategic insight (1-2 sentences) explaining the effective search strategy.
Focus on query formulation, information synthesis, or reasoning approaches.""",
    },

    'sokoban': {
        'system': """You are an expert at analyzing puzzle-solving strategies.
Given a successful and unsuccessful Sokoban puzzle attempt, identify the key strategic difference.""",

        'template': """Sokoban Puzzle

Successful attempt (Score: {win_score:.2f}):
{winner_summary}

Failed attempt (Score: {lose_score:.2f}):
{loser_summary}

Provide a concise strategic insight (1-2 sentences) explaining the effective puzzle-solving approach.
Focus on box pushing order, corner avoidance, or planning strategies.""",
    },

    'default': {
        'system': """You are an expert at analyzing agent decision-making.
Given a successful and unsuccessful attempt at the same task, identify the key strategic difference.""",

        'template': """Task: {task}

Successful attempt (Score: {win_score:.2f}):
{winner_summary}

Failed attempt (Score: {lose_score:.2f}):
{loser_summary}

Provide a concise strategic insight (1-2 sentences) explaining what made the successful attempt work.""",
    }
}


class ReflectionGenerator:
    """Generate strategic reflections from trajectory comparisons.

    The generator supports two modes:
    1. Simple heuristic: Generate basic score-based reflections
    2. LLM-based: Use the same model to generate detailed reflections

    In practice, simple reflections are sufficient for the sleep phase,
    while LLM-based reflections provide richer context for ICL.
    """

    def __init__(
        self,
        use_llm: bool = False,
        env_name: str = "default",
        max_trajectory_length: int = 500,
    ):
        """Initialize the reflection generator.

        Args:
            use_llm: Whether to use LLM for reflection generation
            env_name: Environment name for prompt selection
            max_trajectory_length: Maximum characters for trajectory summaries
        """
        self.use_llm = use_llm
        self.env_name = env_name
        self.max_trajectory_length = max_trajectory_length

        # Get environment-specific prompts
        env_key = env_name.lower().split('/')[0]  # Handle 'alfworld/AlfredTWEnv'
        if env_key not in REFLECTION_PROMPTS:
            env_key = 'default'
        self.prompts = REFLECTION_PROMPTS[env_key]

    def generate_reflections(
        self,
        batch: DataProto,
        scores: torch.Tensor,
        tokenizer=None,
        model=None,
    ) -> List[str]:
        """Generate reflections for trajectories in the batch.

        For high-scoring trajectories, generates strategic reflections by
        comparing with lower-scoring trajectories from the same group.

        Args:
            batch: DataProto with trajectories
            scores: Scores for each trajectory
            tokenizer: Tokenizer for decoding (optional, for LLM mode)
            model: Model for LLM-based reflection (optional)

        Returns:
            List of reflection strings (empty string for non-reflected items)
        """
        batch_size = len(batch)
        reflections = [""] * batch_size

        # Get group information
        uids = batch.non_tensor_batch.get('uid', np.arange(batch_size))
        traj_uids = batch.non_tensor_batch.get('traj_uid', np.arange(batch_size))

        # Group trajectories by uid
        uid_to_indices = defaultdict(list)
        for i in range(batch_size):
            uid = uids[i]
            uid_to_indices[uid].append(i)

        # Process each group
        for uid, indices in uid_to_indices.items():
            if len(indices) < 2:
                # Need at least 2 trajectories to compare
                continue

            # Sort by score within group
            group_scores = [(i, scores[i].item() if torch.is_tensor(scores[i]) else scores[i])
                           for i in indices]
            sorted_group = sorted(group_scores, key=lambda x: x[1], reverse=True)

            # Get winner and loser (different trajectories)
            winner_idx, winner_score = sorted_group[0]
            loser_idx, loser_score = sorted_group[-1]

            # Skip if no meaningful difference
            if winner_score <= loser_score:
                continue

            # Generate reflection for winner
            if self.use_llm and model is not None and tokenizer is not None:
                reflection = self._generate_llm_reflection(
                    batch, winner_idx, loser_idx, winner_score, loser_score, tokenizer, model
                )
            else:
                reflection = self._generate_simple_reflection(
                    winner_score, loser_score, batch, winner_idx, loser_idx
                )

            reflections[winner_idx] = reflection

        return reflections

    def _generate_simple_reflection(
        self,
        winner_score: float,
        loser_score: float,
        batch: DataProto,
        winner_idx: int,
        loser_idx: int,
    ) -> str:
        """Generate a simple heuristic-based reflection.

        Args:
            winner_score: Score of winning trajectory
            loser_score: Score of losing trajectory
            batch: DataProto containing trajectories
            winner_idx: Index of winning trajectory
            loser_idx: Index of losing trajectory

        Returns:
            Simple reflection string
        """
        score_diff = winner_score - loser_score

        # Base reflection on score
        if winner_score >= 1.0:
            quality = "successful"
        elif winner_score >= 0.5:
            quality = "partially successful"
        else:
            quality = "attempted"

        reflection = f"This {quality} approach (score: {winner_score:.2f}) "

        if score_diff > 0.5:
            reflection += f"significantly outperformed alternatives (by {score_diff:.2f}). "
        else:
            reflection += f"outperformed alternatives (by {score_diff:.2f}). "

        # Add environment-specific hints based on env_name
        env_key = self.env_name.lower().split('/')[0]

        if env_key == 'alfworld':
            reflection += "Key: systematic exploration and object interaction."
        elif env_key == 'webshop':
            reflection += "Key: effective product search and attribute matching."
        elif env_key == 'search':
            reflection += "Key: targeted query formulation and information synthesis."
        elif env_key == 'sokoban':
            reflection += "Key: careful planning to avoid box traps."
        else:
            reflection += "Key: systematic task decomposition."

        return reflection

    def _generate_llm_reflection(
        self,
        batch: DataProto,
        winner_idx: int,
        loser_idx: int,
        winner_score: float,
        loser_score: float,
        tokenizer,
        model,
    ) -> str:
        """Generate LLM-based reflection by comparing trajectories.

        Args:
            batch: DataProto containing trajectories
            winner_idx: Index of winning trajectory
            loser_idx: Index of losing trajectory
            winner_score: Score of winner
            loser_score: Score of loser
            tokenizer: Tokenizer for encoding/decoding
            model: Model for generation

        Returns:
            LLM-generated reflection string
        """
        # Extract trajectory summaries
        winner_summary = self._extract_trajectory_summary(batch, winner_idx, tokenizer)
        loser_summary = self._extract_trajectory_summary(batch, loser_idx, tokenizer)

        # Get task description
        task = ""
        if 'raw_prompt' in batch.non_tensor_batch:
            raw_prompt = batch.non_tensor_batch['raw_prompt'][winner_idx]
            if isinstance(raw_prompt, (list, np.ndarray)):
                raw_prompt = raw_prompt[0] if len(raw_prompt) > 0 else ""
            task = str(raw_prompt)[:200]  # Truncate

        # Format prompt
        prompt = self.prompts['template'].format(
            task=task,
            win_score=winner_score,
            winner_summary=winner_summary,
            lose_score=loser_score,
            loser_summary=loser_summary,
        )

        # Generate reflection
        try:
            messages = [
                {"role": "system", "content": self.prompts['system']},
                {"role": "user", "content": prompt},
            ]

            # Apply chat template
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            print(f"Warning: LLM reflection failed: {e}")
            return self._generate_simple_reflection(winner_score, loser_score, batch, winner_idx, loser_idx)

    def _extract_trajectory_summary(
        self,
        batch: DataProto,
        idx: int,
        tokenizer,
    ) -> str:
        """Extract a text summary of a trajectory.

        Args:
            batch: DataProto containing trajectories
            idx: Index of trajectory to summarize
            tokenizer: Tokenizer for decoding

        Returns:
            Truncated text summary of the trajectory
        """
        item = batch[idx]

        # Try to decode response tokens
        if 'responses' in item.batch and tokenizer is not None:
            try:
                response_tokens = item.batch['responses']
                if response_tokens.dim() > 1:
                    response_tokens = response_tokens.squeeze(0)

                text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                return text[:self.max_trajectory_length]
            except Exception:
                pass

        # Fallback to raw prompt snippet
        if 'raw_prompt' in item.non_tensor_batch:
            raw_prompt = item.non_tensor_batch['raw_prompt']
            if isinstance(raw_prompt, (list, np.ndarray)):
                raw_prompt = raw_prompt[0] if len(raw_prompt) > 0 else ""
            return str(raw_prompt)[:self.max_trajectory_length]

        return "[No trajectory available]"

    def generate_batch_reflections_simple(
        self,
        scores: torch.Tensor,
        uids: np.ndarray,
    ) -> Tuple[List[str], List[int]]:
        """Generate simple reflections for all positive-scoring items.

        A faster alternative to full reflection generation that just
        annotates high-scoring trajectories.

        Args:
            scores: Scores for each trajectory
            uids: Group IDs for each trajectory

        Returns:
            Tuple of (reflections list, indices of reflected items)
        """
        batch_size = len(scores)
        reflections = [""] * batch_size
        reflected_indices = []

        # Group by uid and find best in each group
        uid_to_best = {}
        for i in range(batch_size):
            uid = uids[i]
            score = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]

            if uid not in uid_to_best or score > uid_to_best[uid][1]:
                uid_to_best[uid] = (i, score)

        # Generate reflections for best trajectories
        for uid, (idx, score) in uid_to_best.items():
            if score > 0:
                reflections[idx] = f"Effective approach achieving score {score:.2f}."
                reflected_indices.append(idx)

        return reflections, reflected_indices
