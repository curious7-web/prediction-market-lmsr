import random
from dataclasses import dataclass
from typing import Optional

from .gemini_interface import GeminiClient


@dataclass
class Trade:
    outcome: int
    quantity: float


class Agent:
    def __init__(self, name: str):
        self.name = name
        self.wealth = 0.0

    def act(self, market):
        raise NotImplementedError


class TruthTeller(Agent):
    def __init__(self, true_p: float):
        super().__init__("truth")
        self.true_p = true_p

    def act(self, market):
        price = market.price(1)
        qty = self.true_p - price
        if abs(qty) > 1e-3:
            return Trade(outcome=1, quantity=qty)
        return None


class NoiseTrader(Agent):
    def __init__(self):
        super().__init__("noise")

    def act(self, market):
        qty = random.uniform(-0.5, 0.5)
        return Trade(outcome=1, quantity=qty)


class GeminiStrategyAgent(Agent):
    """
    Gemini agent that either:
    - uses an externally supplied belief (for stress tests), OR
    - queries Gemini once per episode, OR
    - falls back safely to 0.5
    """

    def __init__(
        self,
        market_context: str,
        belief: float | None = None,
    ):
        super().__init__("gemini")

        if belief is not None:
            
            self.belief = float(belief)
        else:
           
            try:
                self.client = GeminiClient()
                self.belief = self.client.propose_belief(market_context)
            except Exception as e:
                print("[Gemini] API unavailable — using fallback belief.")
                self.belief = 0.5

        
        self.belief = max(0.01, min(0.99, self.belief))

    def act(self, market):
        price = market.price(1)
        qty = self.belief - price

       
        qty = max(min(qty, 0.5), -0.5)

        if abs(qty) > 1e-3:
            return Trade(outcome=1, quantity=qty)

        return None
class RobustGeminiAgent(Agent):
    """
    Risk-aware Gemini agent.
    Trades toward belief but aggressively scales down exposure
    to control tail risk.
    """

    def __init__(
        self,
        market_context: str,
        belief: float | None = None,
        risk_aversion: float = 10.0,  
    ):
        super().__init__("robust_gemini")

        if belief is not None:
            self.belief = float(belief)
        else:
            try:
                self.client = GeminiClient()
                self.belief = self.client.propose_belief(market_context)
            except Exception:
                print("[RobustGemini] API unavailable — using fallback belief.")
                self.belief = 0.5

        self.belief = max(0.01, min(0.99, self.belief))
        self.risk_aversion = risk_aversion

    def act(self, market):
        price = market.price(1)
        delta = self.belief - price

        
        qty = delta / (1.0 + self.risk_aversion * abs(delta))

        
        qty = max(min(qty, 0.5), -0.5)

        if abs(qty) > 1e-3:
            return Trade(outcome=1, quantity=qty)

        return None
class AdversarialNoiseTrader(Agent):
    """
    Trades against directional pressure to exploit biased agents.
    """

    def __init__(self, strength: float = 0.3):
        super().__init__("adversary")
        self.strength = strength

    def act(self, market):
        price = market.price(1)

        
        direction = -1 if price > 0.5 else 1
        qty = direction * self.strength

        return Trade(outcome=1, quantity=qty)
