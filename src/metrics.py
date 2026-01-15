from typing import List
import math
import numpy as np


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


def cvar(values, alpha: float = 0.05) -> float:
    """
    Conditional Value at Risk (CVaR).

    Mean of the worst alpha-fraction of outcomes.
    Captures systematic tail risk rather than single worst case.
    """
    values = np.asarray(values)

    if values.size == 0:
        return float("nan")

    cutoff = np.quantile(values, alpha)
    tail = values[values <= cutoff]

    return float(tail.mean())
