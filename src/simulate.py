# src/simulate.py
import numpy as np

from .lmsr import LMSRMarket
from .agents import (
    TruthTeller,
    NoiseTrader,
    GeminiStrategyAgent,
    RobustGeminiAgent,
    AdversarialNoiseTrader,
    Trade,
)


def run_episode(
    seed: int,
    include_gemini: bool = False,
    include_robust: bool = False,
    include_adversary: bool = False,
    gemini_belief: float | None = None,
):
    np.random.seed(seed)

    # Hidden true probability
    true_p = 0.6
    outcome = int(np.random.rand() < true_p)

    # Market
    market = LMSRMarket(b=10.0, q=np.array([0.0, 0.0]))

    context = (
        "Binary prediction market with noise and strategic agents. "
        "Outcome uncertainty is high."
    )

    agents = [
        TruthTeller(true_p),
        NoiseTrader(),
    ]

    if include_adversary:
        agents.append(AdversarialNoiseTrader())

    if include_gemini:
        agents.append(
            GeminiStrategyAgent(
                market_context=context,
                belief=gemini_belief,
            )
        )

    if include_robust:
        agents.append(
            RobustGeminiAgent(
                market_context=context,
                belief=gemini_belief,
                risk_aversion=10.0,
            )
        )

    # Single trading round
    for agent in agents:
        trade = agent.act(market)
        if isinstance(trade, Trade):
            market.execute_trade(
                agent.name,
                trade.outcome,
                trade.quantity,
            )

    market.resolve(outcome)

    wealth = {
        agent.name: market.wealth.get(agent.name, 0.0)
        for agent in agents
    }

    return {
        "wealth": wealth,
        "final_price": market.price(1),
        "outcome": outcome,
    }


if __name__ == "__main__":
    r = run_episode(
        seed=0,
        include_gemini=True,
        include_robust=True,
        include_adversary=True,
    )
    print(r)
