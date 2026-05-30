#!/usr/bin/env python3
"""
Project EvapoContext: Dynamic Context Re-Ranker
File: src/engine.py

This module calculates the time-decay retention score for text chunks,
applying stateful pinning gates and dynamic pruning boundaries based on
real-time system resource pressure levels.
"""

import math
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np

logger = logging.getLogger("EvapoContextEngine")


class DynamicContextReRanker:
    """
    The mathematical core of Project EvapoContext context management.
    
    This engine filters conversational history text blocks based on a combination
    of semantic relevance, length, conversational age (rank), and current system stress.
    """

    def __init__(
        self,
        base_threshold: float = 0.20,
        pressure_factor: float = 0.55,
        default_normalization: str = "log_multiply",
        decay_mode: str = "log",
        token_weight: float = 1.0,
        soft_pin_multiplier: float = 2.0,
        budget_sorting_mode: str = "force"
    ):
        """
        Initializes the engine with configurable coefficients.
        
        Args:
            base_threshold: The baseline threshold score required for a chunk to 
                            survive when system pressure is 0% (optimal conditions).
            pressure_factor: Controls how aggressively the threshold increases as 
                             system pressure rises (quadratic scaling).
            default_normalization: Determines how token count affects the retention weight of a chunk:
                                   - "log_multiply": Larger chunks are treated as heavier
                                     (higher retention) and survive longer.
                                   - "log_divide": Larger chunks are penalized (treated as
                                     costly to keep) and pruned faster.
                                   - "constant": Token count has no impact on weight calculation.
            decay_mode: Temporal decay curve to apply:
                        - "inverse_square": 1 / r^2 (legacy)
                        - "inverse": 1 / r
                        - "sqrt": 1 / sqrt(r)
                        - "log": 1 / log2(r + 1) (default, gentle)
            token_weight: Scaling exponent or factor to adjust the influence of token size.
            soft_pin_multiplier: Multiplier applied to the retention score of soft-pinned chunks.
            budget_sorting_mode: Default sorting mode for context budgeting:
                                 - "force": Prioritize absolute highest retention chunks.
                                 - "efficiency": Prioritize high relevance per token (retention_score/tokens).
        """
        self.base_threshold = base_threshold
        self.pressure_factor = pressure_factor
        self.default_normalization = default_normalization
        self.decay_mode = decay_mode
        self.token_weight = token_weight
        self.soft_pin_multiplier = soft_pin_multiplier
        self.budget_sorting_mode = budget_sorting_mode
        logger.info(
            f"Initialized DynamicContextReRanker (Base Threshold: {self.base_threshold}, "
            f"Pressure Factor: {self.pressure_factor}, Normalization: {self.default_normalization}, "
            f"Decay Mode: {self.decay_mode}, Token Weight: {self.token_weight}, "
            f"Soft Pin Multiplier: {self.soft_pin_multiplier}, "
            f"Budget Sorting Mode: {self.budget_sorting_mode})"
        )

    def calculate_pruning_threshold(self, system_pressure: float) -> float:
        """
        Calculates the dynamic threshold boundary that chunks must exceed to survive.
        
        The mathematical formula is:
            Threshold_pruning = base_threshold + (system_pressure^2 * pressure_factor)
            
        Example calculation:
            1. Idle System (pressure = 0.0):
               Threshold = 0.20 + (0.0^2 * 0.55) = 0.20
               -> Chunks only need a score of 0.20 to survive.
               
            2. High Stress System (pressure = 0.8 / 80% RAM full):
               Threshold = 0.20 + (0.64 * 0.55) = 0.20 + 0.352 = 0.552
               -> Chunks now need a much higher score (0.552) to stay in memory.
               
            3. Critical System (pressure = 1.0 / 100% RAM full):
               Threshold = 0.20 + (1.0^2 * 0.55) = 0.75
               -> Chunks must score >= 0.75. Almost all non-pinned chunks will be pruned.
        """
        # Clamp pressure to [0.0, 1.0] in case raw metrics are slightly outside boundaries
        clamped_pressure = max(0.0, min(1.0, system_pressure))
        
        # Compute threshold using quadratic scaling (squaring the pressure makes the reaction
        # slow at first, then extremely sharp and aggressive when system stress gets critical)
        threshold = self.base_threshold + (math.pow(clamped_pressure, 2) * self.pressure_factor)
        
        # Round to 6 decimal places to ensure clean floating-point precision checks
        return round(threshold, 6)

    def calculate_retention_score(
        self,
        similarity: float,
        token_count: int,
        rank: int,
        system_pressure: float = 0.0,
        normalization_mode: Optional[str] = None
    ) -> float:
        """
        Calculates the retention score keeping a text chunk in memory.
        
        Formula:
            Retention_Score = Relevance_Weight / decay(r)
            
        Step-by-Step Logic:
            1. Enforce rank decay based on decay_mode:
               - "inverse_square": divide by r^2 (legacy)
               - "inverse": divide by r
               - "sqrt": divide by sqrt(r)
               - "log": divide by log2(r + 1) (default, gentle)
            2. Apply relevance weight normalization with token_weight:
               - "log_multiply": similarity * (1.0 + (ln(1 + tokens) / 10) * token_weight)
               - "log_divide": similarity / ln(1 + tokens * token_weight)
               - "constant": similarity (constant weight)
        """
        # Ensure rank (r) is 1-indexed to prevent division by zero or negative ranks
        r = float(max(1, rank))

        # Clamp similarity to ensure it stays positive [0.0, 1.0]
        sim = max(0.0, similarity)
        # Prevent zero or negative token counts
        tokens = max(1, token_count)

        # Decide how to scale the informational weight based on length (token_count)
        mode = normalization_mode or self.default_normalization
        
        if mode == "log_multiply":
            # Using natural log (log1p returns ln(1+x) which is safe against x=0)
            # Dividing by 10 stabilizes the growth, adding 1.0 ensures we scale up from 100%
            weight_product = sim * (1.0 + (math.log1p(tokens) / 10.0) * self.token_weight)
        elif mode == "log_divide":
            # We divide similarity by log of tokens. The larger the tokens, the lower the weight.
            # This makes large files/chunks have less survival score because they cost too much space.
            weight_product = sim / max(1.0, math.log1p(tokens) * self.token_weight)
        elif mode == "constant":
            # Constant mode does not adjust similarity by token count.
            weight_product = sim
        else:
            raise ValueError(f"Unknown token normalization mode: {mode}")

        # Calculate decay factor based on configured decay_mode
        decay_mode = self.decay_mode
        if decay_mode == "inverse_square":
            decay_factor = math.pow(r, 2)
        elif decay_mode == "inverse":
            decay_factor = r
        elif decay_mode == "sqrt":
            decay_factor = math.sqrt(r)
        elif decay_mode == "log":
            # Normalized log decay: log2(r + 1). At r=1, log2(2) = 1.0.
            decay_factor = math.log2(r + 1)
        else:
            raise ValueError(f"Unknown decay mode: {decay_mode}")

        # Final Retention Score calculation
        score = weight_product / decay_factor
        
        return round(score, 6)

    def calculate_retention_score_vector(
        self,
        query_vector: np.ndarray,
        chunk_vector: np.ndarray,
        token_count: int,
        rank: int,
        system_pressure: float,
        normalization_mode: Optional[str] = None
    ) -> float:
        """
        Calculates the retention score using raw numerical embedding vectors.
        
        Step-by-Step Logic:
            1. Ensure the vectors are single-precision floats (float32) for speed and
               consistency across Windows and macOS.
            2. Compute the dot product: q . c
            3. Compute the magnitudes (norms) of both vectors.
            4. Compute Cosine Similarity: dot_product / (magnitude_q * magnitude_c)
            5. Run the standard calculate_retention_score method using this similarity.
        """
        # Force single precision floats for cross-platform vector consistency
        q = np.array(query_vector, dtype=np.float32)
        c = np.array(chunk_vector, dtype=np.float32)

        # Compute vector magnitudes (Euclidean L2 Norms)
        q_norm = np.linalg.norm(q)
        c_norm = np.linalg.norm(c)
        
        # Handle zero vector edge cases to avoid division by zero
        if q_norm == 0 or c_norm == 0:
            similarity = 0.0
        else:
            # Cosine similarity formula: (A . B) / (||A|| * ||B||)
            similarity = float(np.dot(q, c) / (q_norm * c_norm))
            # Clamp value mathematically to [0, 1] range to avoid floating-point inaccuracies
            similarity = max(0.0, min(1.0, similarity))

        return self.calculate_retention_score(
            similarity=similarity,
            token_count=token_count,
            rank=rank,
            system_pressure=system_pressure,
            normalization_mode=normalization_mode
        )

    def optimize_context(
        self,
        chunks: List[Dict[str, Any]],
        system_pressure: float,
        chronological: bool = True,
        normalization_mode: Optional[str] = None,
        token_budget: Optional[int] = None,
        budget_sorting_mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Takes in a raw list of text chunks and returns only the ones that survive pruning.
        Optionally enforces a strict token budget.
        
        Step-by-Step Logic:
            1. Calculate the pruning threshold based on system_pressure.
            2. Loop through all chunks in the input list:
               - Check the Pinning level:
                 - "critical" (or is_pinned = True): Bypasses pruning.
                 - "soft": Bypasses pruning, but score is scaled by soft_pin_multiplier.
                 - "none" / regular: Evaluated against pruning threshold.
               - Determine the temporal rank (r) of the chunk.
               - Calculate the chunk's retention score using calculate_retention_score.
               - If the score is higher than or equal to the pruning threshold (or pinned),
                 the chunk survives!
            3. Separate survivors into Critical vs. Non-Critical.
            4. Sort the non-critical chunks according to budget_sorting_mode:
               - "force": Sort by raw retention score descending (default).
               - "efficiency": Sort by (retention score / tokens) descending to prioritize density.
            5. Enforce token budget: Always retain all critical pinned chunks first. Log a warning if 
               critical chunks exceed the budget, and fill the remaining space with the sorted chunks.
        """
        num_chunks = len(chunks)
        pruning_threshold = self.calculate_pruning_threshold(system_pressure)
        
        logger.debug(
            f"Optimizing context. Input chunks: {num_chunks} | "
            f"System Pressure: {system_pressure * 100:.1f}% | "
            f"Pruning Threshold Required: {pruning_threshold:.4f} | "
            f"Token Budget: {token_budget} | "
            f"Budget Sorting Mode: {budget_sorting_mode or self.budget_sorting_mode}"
        )
        
        critical_chunks = []
        other_chunks = []
        
        for idx, chunk in enumerate(chunks):
            # Check for Pinning Level (three-tier system)
            is_pinned_flag = chunk.get("is_pinned", False)
            pinning_level = chunk.get("pinning_level", "critical" if is_pinned_flag else "none")
            
            # Determine temporal rank of the chunk
            if "rank" in chunk:
                rank = int(chunk["rank"])
            elif "r" in chunk:
                rank = int(chunk["r"])
            else:
                if chronological:
                    rank = num_chunks - idx
                else:
                    rank = idx + 1
            
            # Retrieve semantic metrics
            similarity = float(chunk.get("similarity", chunk.get("score", 0.0)))
            token_count = int(chunk.get("token_count", chunk.get("tokens", 100)))
            
            # If critical pinned, it bypasses pruning entirely and gets infinite retention score
            if pinning_level == "critical":
                pinned_chunk = chunk.copy()
                pinned_chunk["retention_score"] = float("inf")
                pinned_chunk["force_score"] = float("inf")
                pinned_chunk["pruning_threshold"] = pruning_threshold
                pinned_chunk["rank_assigned"] = rank
                critical_chunks.append(pinned_chunk)
                continue
            
            # Compute retention score for soft pinned and regular chunks
            retention_score = self.calculate_retention_score(
                similarity=similarity,
                token_count=token_count,
                rank=rank,
                system_pressure=system_pressure,
                normalization_mode=normalization_mode
            )
            
            # If soft pinned, it bypasses pruning but gets scaled retention score
            if pinning_level == "soft":
                boosted_score = retention_score * self.soft_pin_multiplier
                pinned_chunk = chunk.copy()
                pinned_chunk["retention_score"] = boosted_score
                pinned_chunk["force_score"] = boosted_score
                pinned_chunk["pruning_threshold"] = pruning_threshold
                pinned_chunk["rank_assigned"] = rank
                other_chunks.append(pinned_chunk)
                continue
                
            # Pruning filter check: only keep the chunk if its retention score meets the threshold
            if retention_score >= pruning_threshold:
                processed_chunk = chunk.copy()
                processed_chunk["retention_score"] = retention_score
                processed_chunk["force_score"] = retention_score
                processed_chunk["pruning_threshold"] = pruning_threshold
                processed_chunk["rank_assigned"] = rank
                other_chunks.append(processed_chunk)
        
        # Sort non-critical surviving chunks based on budget_sorting_mode
        sort_mode = budget_sorting_mode or self.budget_sorting_mode
        if sort_mode == "efficiency":
            other_chunks.sort(
                key=lambda x: x["retention_score"] / max(1, int(x.get("token_count", x.get("tokens", 100)))),
                reverse=True
            )
        else:
            other_chunks.sort(key=lambda x: x["retention_score"], reverse=True)
        
        # Combine critical (first priority) and sorted other chunks
        survived_chunks = critical_chunks + other_chunks
        
        # Enforce Token Budget
        if token_budget is not None:
            # Check critical chunk overflow
            critical_tokens = sum(int(c.get("token_count", c.get("tokens", 100))) for c in critical_chunks)
            if critical_tokens > token_budget:
                logger.warning(
                    f"Critical pinned chunks size ({critical_tokens} tokens) "
                    f"exceeds total token budget ({token_budget} tokens)!"
                )
                
            final_chunks = []
            accumulated_tokens = 0
            for chunk in survived_chunks:
                tokens = int(chunk.get("token_count", chunk.get("tokens", 100)))
                # Critical pinned chunks are ALWAYS kept
                if chunk["retention_score"] == float("inf"):
                    final_chunks.append(chunk)
                    accumulated_tokens += tokens
                else:
                    if accumulated_tokens + tokens <= token_budget:
                        final_chunks.append(chunk)
                        accumulated_tokens += tokens
            survived_chunks = final_chunks
        
        logger.debug(f"Optimization finished. {len(survived_chunks)}/{num_chunks} chunks survived.")
        return survived_chunks

# Backward compatibility alias
GravitationalScoringEngine = DynamicContextReRanker
