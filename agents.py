# agents.py
import random

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.cash = 0.0
        self.position_yes = 0.0
        self.position_no = 0.0

    def pnl(self, outcome_yes: bool):
        if outcome_yes:
            return self.cash + self.position_yes
        else:
            return self.cash + self.position_no


class BaselineAgent(BaseAgent):
    """
    Trades toward true probability with small step size.
    """

    def __init__(self, true_prob: float, step_size: float = 1.0):
        super().__init__("baseline")
        self.true_prob = true_prob
        self.step_size = step_size

    def act(self, market):
        p_yes, p_no = market.prices()
        diff = self.true_prob - p_yes

        delta_yes = self.step_size * diff
        delta_no = -delta_yes

        cost = market.trade(delta_yes, delta_no)
        self.cash -= cost
        self.position_yes += delta_yes
        self.position_no += delta_no


class NoiseAgent(BaseAgent):
    """
    Trades randomly or with biased belief.
    """

    def __init__(self, bias: float = 0.5, scale: float = 1.0):
        super().__init__("noise")
        self.bias = bias
        self.scale = scale

    def act(self, market):
        p_yes, _ = market.prices()
        direction = random.choice([-1, 1])

        delta_yes = direction * self.scale * (self.bias - p_yes)
        delta_no = -delta_yes

        cost = market.trade(delta_yes, delta_no)
        self.cash -= cost
        self.position_yes += delta_yes
        self.position_no += delta_no
