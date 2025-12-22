# market.py
import math

class LMSRMarket:
    """
    Binary LMSR market (YES / NO).
    q_yes and q_no track outstanding shares.
    """

    def __init__(self, b: float):
        if b <= 0:
            raise ValueError("Liquidity parameter b must be > 0")
        self.b = b
        self.q_yes = 0.0
        self.q_no = 0.0

    def _cost(self, q_yes: float, q_no: float) -> float:
        m = max(q_yes, q_no)
        return self.b * (
            math.log(
                math.exp((q_yes - m) / self.b)
                + math.exp((q_no - m) / self.b)
            )
            + m / self.b
        )

    def prices(self):
        """Instantaneous prices (p_yes, p_no)."""
        m = max(self.q_yes, self.q_no)
        exp_yes = math.exp((self.q_yes - m) / self.b)
        exp_no = math.exp((self.q_no - m) / self.b)
        z = exp_yes + exp_no
        return exp_yes / z, exp_no / z

    def trade(self, delta_yes: float, delta_no: float) -> float:
        """
        Execute trade and return cost paid by trader.
        """
        old_cost = self._cost(self.q_yes, self.q_no)
        self.q_yes += delta_yes
        self.q_no += delta_no
        new_cost = self._cost(self.q_yes, self.q_no)
        return new_cost - old_cost
