"""Signed rating/recency preference target; confidence unchanged from IALS."""
import numpy as np
from .ials import IALS

class RatingRecencyALS(IALS):
    def __init__(self, half_life=730., **kwargs):
        super().__init__(**kwargs)
        self.half_life = half_life

    def targets(self, data):
        age_days = (data.cutoff - data.frame.timestamp.to_numpy()) / 86400.
        recency = .5 + .5 * np.exp2(-age_days / self.half_life)
        return (data.frame.rating.to_numpy() - 3.) / 2. * recency
