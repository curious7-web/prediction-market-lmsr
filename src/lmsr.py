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

        self.b = float(b)
        self.q = list(q)
        self.wealth = {}  

    def cost(self, q: List[float]) -> float:
        return self.b * logsumexp([qi / self.b for qi in q])

    def prices(self) -> List[float]:
        exps = [math.exp(qi / self.b) for qi in self.q]
        z = sum(exps)
        return [e / z for e in exps]

    def price(self, outcome: int) -> float:
        """Convenience accessor used by agents."""
        return self.prices()[outcome]

    def trade_cost(self, delta: List[float]) -> float:
        new_q = [qi + di for qi, di in zip(self.q, delta)]
        return self.cost(new_q) - self.cost(self.q)

    def execute_trade(self, agent: str, outcome: int, quantity: float) -> None:
        """
        Execute a trade of `quantity` shares for `outcome`
        and charge the agent the LMSR cost.
        """
        delta = [0.0 for _ in self.q]
        delta[outcome] = quantity

        cost = self.trade_cost(delta)

        self.q = [qi + di for qi, di in zip(self.q, delta)]
        self.wealth[agent] = self.wealth.get(agent, 0.0) - cost

    def resolve(self, outcome: int) -> None:
        """
        Resolve the market and pay out $1 per share
        of the realized outcome.
        """
        for agent, pnl in self.wealth.items():
            self.wealth[agent] = pnl + self.q[outcome]
