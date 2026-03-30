import numpy as np
from typing import Optional

class ThresholdPolicy:
    """
    Policy:
      - Deterministic if sigma is None: Z = 1[p >= tau]
      - Stochastic if sigma is float:   P(Z=1) = sigmoid( sigma * (p - tau) )
    Large sigma -> sharp step (near-deterministic). Small sigma -> softer boundary.
    """
    def __init__(self, tau: float = 0.5, sigma: Optional[float] = None):
        self.tau = float(tau)
        self.sigma = None if sigma is None else float(sigma)

    def decide_prob(self, p: np.ndarray) -> np.ndarray:
        if self.sigma is None:
            return (p >= self.tau).astype(float)
        logits = self.sigma * (p - self.tau)
        return 1.0 / (1.0 + np.exp(-logits))

    def sample(self, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        q = self.decide_prob(p)
        if self.sigma is None:
            return q
        return (rng.random(size=q.shape) < q).astype(float)
