# src/simulate.py
import random
from typing import Dict, List, Tuple

from lmsr import LMSRMarket
from agents import TruthTeller, NoiseTrader, GeminiStrategyAgent, Agent, Trade
from metrics import brier_score, log_loss

def run_episode(
    seed: int = 7,
    p_true: float = 0.55,
    b: float = 25.0,
    steps: int = 60,
    include_gemini: bool = True,
) -> Tuple[List[float], int, Dict[str, float], Dict[str, float], float, float]:
    random.seed(seed)

    market = LMSRMarket(num_outcomes=2, b=b)

    agents: List[Agent] = [
        TruthTeller(init_wealth=0.0, p_true=p_true),
        NoiseTrader(init_wealth=0.0),
    ]
    if include_gemini:
        agents.append(GeminiStrategyAgent(init_wealth=0.0, belief_p1=p_true))

    # Run trading
    for t in range(steps):
        prices = market.prices()
        for ag in agents:
            trade = ag.decide_trade(prices=prices, step=t, max_steps=steps, b=b)
            if trade is None:
                continue

            # Execute trade via LMSR (agent pays cost difference)
            # LMSRMarket should expose: trade(outcome, shares, side) -> cost (positive=pay, negative=receive)
            cost = market.trade(outcome=trade.outcome, shares=trade.shares, side=trade.side)

            ag.wealth -= cost
            if trade.side == "BUY":
                ag.position[trade.outcome] += trade.shares
            else:
                ag.position[trade.outcome] -= trade.shares

    # Realized outcome
    outcome = 1 if random.random() < p_true else 0

    # Settle positions (payout 1 per share of realized outcome)
    wealth: Dict[str, float] = {}
    for ag in agents:
        payout = ag.position[outcome] * 1.0
        ag.wealth += payout
        wealth[ag.name] = ag.wealth

    final_prices = market.prices()

    # Regret vs no-trade baseline (=0 profit, since init wealth 0)
    regret = {name: -w for name, w in wealth.items()}

    # Proper scoring on market final prob for realized outcome
    p1_final = float(final_prices[1])
    brier = brier_score(p1_final, outcome)
    ll = log_loss(p1_final, outcome)

    return final_prices, outcome, wealth, regret, brier, ll

def main():
    final_prices, outcome, wealth, regret, brier, ll = run_episode(include_gemini=True)

    print("Final prices:", final_prices)
    print("Outcome:", outcome)
    print("Final wealth:", wealth)
    print("Regret vs no-trade baseline:", regret)
    print("Brier:", brier)
    print("LogLoss:", ll)

if __name__ == "__main__":
    main()
