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
        budget_sorting_mode: str = "force",
        threshold_ratio: float = 0.80
    ):
        """
        Initializes the engine with configurable coefficients.
        
        Args:
            base_threshold: The baseline threshold score required for a chunk to 
                            survive when system pressure is 0% (optimal conditions).
            pressure_factor: Controls how aggressively the threshold increases as 
                             system pressure rises (quadratic scaling).
            default_normalization: Determines how token count affects the retention weight of a chunk.
            decay_mode: Temporal decay curve to apply.
            token_weight: Scaling exponent or factor to adjust the influence of token size.
            soft_pin_multiplier: Multiplier applied to the retention score of soft-pinned chunks.
            budget_sorting_mode: Default sorting mode for context budgeting.
            threshold_ratio: Dynamic capping ratio for relevance-capped threshold adaptation.
        """
        self.base_threshold = base_threshold
        self.pressure_factor = pressure_factor
        self.default_normalization = default_normalization
        self.decay_mode = decay_mode
        self.token_weight = token_weight
        self.soft_pin_multiplier = soft_pin_multiplier
        self.budget_sorting_mode = budget_sorting_mode
        self.threshold_ratio = threshold_ratio
        logger.info(
            f"Initialized DynamicContextReRanker (Base Threshold: {self.base_threshold}, "
            f"Pressure Factor: {self.pressure_factor}, Normalization: {self.default_normalization}, "
            f"Decay Mode: {self.decay_mode}, Token Weight: {self.token_weight}, "
            f"Soft Pin Multiplier: {self.soft_pin_multiplier}, "
            f"Budget Sorting Mode: {self.budget_sorting_mode}, "
            f"Threshold Ratio: {self.threshold_ratio})"
        )

    def calculate_pruning_threshold(self, system_pressure: float, max_score: Optional[float] = None) -> float:
        """
        Calculates the dynamic threshold boundary that chunks must exceed to survive.
        
        The mathematical formula is:
            Telemetry_Threshold = base_threshold + (system_pressure^2 * pressure_factor)
            Threshold_Cap = max(base_threshold, max_score * threshold_ratio)
            Pruning_Threshold = min(Telemetry_Threshold, Threshold_Cap)
        """
        # Clamp pressure to [0.0, 1.0] in case raw metrics are slightly outside boundaries
        clamped_pressure = max(0.0, min(1.0, system_pressure))
        
        # Compute telemetry-driven threshold using quadratic scaling
        telemetry_threshold = self.base_threshold + (math.pow(clamped_pressure, 2) * self.pressure_factor)
        
        if max_score is not None:
            # Prevent context collapse when retrieved match quality is low
            cap = max(self.base_threshold, max_score * self.threshold_ratio)
            threshold = min(telemetry_threshold, cap)
        else:
            threshold = telemetry_threshold
            
        # Round to 6 decimal places to ensure clean floating-point precision checks
        return round(threshold, 6)

    def calculate_retention_score(
        self,
        similarity: float,
        token_count: int,
        rank: int,
        system_pressure: float = 0.0,
        normalization_mode: Optional[str] = None,
        turn_offset: Optional[int] = None
    ) -> float:
        """
        Calculates the retention score keeping a text chunk in memory.
        
        Formula:
            Retention_Score = Relevance_Weight / decay(age)
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

        # Choose age metric: Turn Offset takes precedence over Search Rank (r)
        # Ensure age is >= 1 to prevent division issues
        age = float(max(1, turn_offset)) if turn_offset is not None else r

        # Calculate decay factor based on configured decay_mode
        decay_mode = self.decay_mode
        if decay_mode == "inverse_square":
            decay_factor = math.pow(age, 2)
        elif decay_mode == "inverse":
            decay_factor = age
        elif decay_mode == "sqrt":
            decay_factor = math.sqrt(age)
        elif decay_mode == "log":
            # Normalized log decay: log2(age + 1). At age=1, log2(2) = 1.0.
            decay_factor = math.log2(age + 1)
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
        normalization_mode: Optional[str] = None,
        turn_offset: Optional[int] = None
    ) -> float:
        """
        Calculates the retention score using raw numerical embedding vectors.
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
            normalization_mode=normalization_mode,
            turn_offset=turn_offset
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
        """
        num_chunks = len(chunks)
        
        # Calculate max similarity score across all candidate chunks for threshold adaptation
        max_score = 0.0
        if chunks:
            max_score = max(float(c.get("similarity", c.get("score", 0.0))) for c in chunks)
            
        pruning_threshold = self.calculate_pruning_threshold(system_pressure, max_score=max_score)
        
        logger.debug(
            f"Optimizing context. Input chunks: {num_chunks} | "
            f"System Pressure: {system_pressure * 100:.1f}% | "
            f"Max Score: {max_score:.4f} | "
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
            turn_offset = chunk.get("turn_offset")
            
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
                normalization_mode=normalization_mode,
                turn_offset=turn_offset
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
