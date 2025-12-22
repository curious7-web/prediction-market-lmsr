# src/lmsr.py
import math
from typing import List

def logsumexp(xs: List[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

class LMSRMarket:
    """
    Binary or multi-outcome LMSR market.
    q[i] = outstanding shares for outcome i
    b    = liquidity parameter
    """

    def __init__(self, b: float, q: List[float]):
        if b <= 0:
            raise ValueError("b must be > 0")
        if len(q) < 2:
            raise ValueError("Need at least two outcomes")
        self.b = b
        self.q = q

    def cost(self, q: List[float]) -> float:
        return self.b * logsumexp([qi / self.b for qi in q])

    def prices(self) -> List[float]:
        exps = [math.exp(qi / self.b) for qi in self.q]
        z = sum(exps)
        return [e / z for e in exps]

    def trade_cost(self, delta: List[float]) -> float:
        new_q = [qi + di for qi, di in zip(self.q, delta)]
        return self.cost(new_q) - self.cost(self.q)

    def execute_trade(self, delta: List[float]) -> float:
        c = self.trade_cost(delta)
        self.q = [qi + di for qi, di in zip(self.q, delta)]
        return c
