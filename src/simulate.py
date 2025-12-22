from typing import List, Dict
import random

from lmsr import LMSRMarket
from agents import TruthTeller, NoiseTrader, Agent
from metrics import brier_score, log_loss

def sample_outcome(true_probs: List[float]) -> int:
    r = random.random()
    s = 0.0
    for i, p in enumerate(true_probs):
        s += p
        if r <= s:
            return i
    return len(true_probs) - 1

def run_episode(
    K: int = 2,
    b: float = 20.0,
    T: int = 100,
    true_probs: List[float] = None,
    seed: int = 7
) -> Dict:
    random.seed(seed)

    if true_probs is None:
        true_probs = [0.55, 0.45] if K == 2 else [1.0 / K] * K

    market = LMSRMarket(b=b, q=[0.0] * K)

    agents: List[Agent] = [
        TruthTeller(name="truth", cash=0.0, shares=[0.0]*K, belief=true_probs, intensity=2.0),
        NoiseTrader(name="noise", cash=0.0, shares=[0.0]*K, scale=1.5, seed=seed),
    ]

    price_path = []
    costs_paid = {a.name: 0.0 for a in agents}

    for t in range(T):
        prices = market.prices()
        price_path.append(prices)

        for a in agents:
            delta = a.decide_trade(prices, t)
            c = market.execute_trade(delta)

            # Trader pays c, receives delta shares
            a.cash -= c
            a.shares = [s + d for s, d in zip(a.shares, delta)]
            costs_paid[a.name] += c

    final_prices = market.prices()
    outcome = sample_outcome(true_probs)

    # settle: pays 1 per share of realized outcome
    for a in agents:
        a.cash += a.shares[outcome]

    final_wealth = {a.name: a.cash for a in agents}

    return {
        "K": K,
        "b": b,
        "T": T,
        "true_probs": true_probs,
        "final_prices": final_prices,
        "outcome": outcome,
        "final_wealth": final_wealth,
        "brier": brier_score(final_prices, outcome),
        "log_loss": log_loss(final_prices, outcome),
        "price_path": price_path,
        "costs_paid": costs_paid
    }

if __name__ == "__main__":
    res = run_episode()
    print("Final prices:", res["final_prices"])
    print("Outcome:", res["outcome"])
    print("Final wealth:", res["final_wealth"])
    print("Brier:", res["brier"], "LogLoss:", res["log_loss"])
