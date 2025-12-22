# src/agents.py
from dataclasses import dataclass
from typing import List, Optional
import random

@dataclass
class Agent:
    name: str
    cash: float
    shares: List[float]

    def net_worth(self, prices: List[float]) -> float:
        return self.cash + sum(s * p for s, p in zip(self.shares, prices))

    def decide_trade(self, prices: List[float], t: int) -> List[float]:
        raise NotImplementedError


@dataclass
class TruthTeller(Agent):
    """
    Agent that believes the true probability distribution
    and trades toward it.
    """
    belief: List[float]
    intensity: float = 2.0

    def decide_trade(self, prices: List[float], t: int) -> List[float]:
        # Move market toward belief
        return [self.intensity * (b - p) for b, p in zip(self.belief, prices)]


@dataclass
class NoiseTrader(Agent):
    """
    Random trader to create market pressure / noise.
    """
    scale: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)

    def decide_trade(self, prices: List[float], t: int) -> List[float]:
        K = len(prices)
        raw = [random.uniform(-self.scale, self.scale) for _ in range(K)]
        mean = sum(raw) / K
        return [x - mean for x in raw]
