# src/simulate.py

from typing import List, Dict
import random

from lmsr import LMSRMarket
from agents import TruthTeller, NoiseTrader, Agent
from metrics import brier_score, log_loss


def sample_outcome(true_probs: List[float]) -> int:
    """Sample final outcome from true probability distribution."""
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
    """
    Run a single LMSR market simulation episode.
    Returns final wealth, regret, and scoring metrics.
    """
    random.seed(seed)

    if true_probs is None:
        true_probs = [0.55, 0.45] if K == 2 else [1.0 / K] * K

    # --- Initialize market ---
    market = LMSRMarket(b=b, q=[0.0] * K)

    # --- Agents ---
    agents: List[Agent] = [
        TruthTeller(
            name="truth",
            cash=0.0,
            shares=[0.0] * K,
            belief=true_probs,
            intensity=2.0,
        ),
        NoiseTrader(
            name="noise",
            cash=0.0,
            shares=[0.0] * K,
            scale=1.5,
            seed=seed,
        ),
    ]

    # No-trade baseline (does nothing)
    baseline_wealth = 0.0

    price_path = []
    costs_paid = {a.name: 0.0 for a in agents}

    # --- Trading loop ---
    for t in range(T):
        prices = market.prices()
        price_path.append(prices)

        for a in agents:
            delta = a.decide_trade(prices, t)
            cost = market.execute_trade(delta)

            a.cash -= cost
            a.shares = [s + d for s, d in zip(a.shares, delta)]
            costs_paid[a.name] += cost

    # --- Outcome realization ---
    final_prices = market.prices()
    outcome = sample_outcome(true_probs)

    # --- Settlement ---
    for a in agents:
        a.cash += a.shares[outcome]

    final_wealth = {a.name: a.cash for a in agents}

    # --- Regret vs baseline ---
    regret = {
        name: baseline_wealth - wealth
        for name, wealth in final_wealth.items()
    }

    return {
        "K": K,
        "b": b,
        "T": T,
        "true_probs": true_probs,
        "final_prices": final_prices,
        "outcome": outcome,
        "final_wealth": final_wealth,
        "regret": regret,
        "brier": brier_score(final_prices, outcome),
        "log_loss": log_loss(final_prices, outcome),
        "price_path": price_path,
        "costs_paid": costs_paid,
    }


# -------------------------
# Run directly
# -------------------------
if __name__ == "__main__":
    result = run_episode()

    print("Final prices:", result["final_prices"])
    print("Outcome:", result["outcome"])
    print("Final wealth:", result["final_wealth"])
    print("Regret vs no-trade baseline:", result["regret"])
    print("Brier:", result["brier"])
    print("LogLoss:", result["log_loss"])
