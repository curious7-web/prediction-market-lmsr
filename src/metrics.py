# src/metrics.py
from typing import List
import math

def brier_score(probs: List[float], outcome: int) -> float:
    """
    Brier score for a single multi-class outcome.
    Lower is better.
    """
    return sum(
        (p - (1.0 if i == outcome else 0.0)) ** 2
        for i, p in enumerate(probs)
    )

def log_loss(probs: List[float], outcome: int, eps: float = 1e-12) -> float:
    """
    Log loss for a single outcome.
    Lower is better.
    """
    p = max(eps, min(1.0 - eps, probs[outcome]))
    return -math.log(p)
