"""
agents.py
Defines household and workplace agents for the water consumption simulator.

Three agent types (matching the paper):
  - FamilyHouseholdAgent   : families with children (spikes morning, noon, evening)
  - WorkingAdultAgent      : working adults (spikes morning + evening, low daytime)
  - WorkplaceSchoolAgent   : offices / schools (steady daytime, low otherwise)

Each agent holds:
  - a 24-hour baseline consumption profile (L/h)
  - a set of active interventions that reduce consumption
  - derived "with intervention" profile used for comparison
"""

import math
from mesa import Agent

# ---------------------------------------------------------------------------
# Intervention catalogue
# ---------------------------------------------------------------------------
INTERVENTIONS = {
    "efficient_toilet":  {"label": "Low-flow Toilet",            "savings": 0.15, "category": "fixture"},
    "shower_reducer":    {"label": "Shower Flow Reducer",        "savings": 0.15, "category": "fixture"},
    "dishwasher":        {"label": "Efficient Dishwasher",       "savings": 0.02, "category": "appliance"},
    "smart_meter":       {"label": "Smart Meter + Feedback App", "savings": 0.08, "category": "digital"},
    "leak_detection":    {"label": "Leak Detection Sensor",      "savings": 0.10, "category": "digital"},
    "rainwater_tank":    {"label": "Rainwater Harvesting Tank",  "savings": 0.10, "category": "infrastructure"},
    "education_program": {"label": "Water Education Program",    "savings": 0.07, "category": "behaviour"},
    "pressure_limiter":  {"label": "Pressure Limiter Valve",     "savings": 0.10, "category": "infrastructure"},
}

# ---------------------------------------------------------------------------
# Hourly consumption shape helpers
# ---------------------------------------------------------------------------

def _gaussian(hour, mu, sigma, amplitude):
    return amplitude * math.exp(-0.5 * ((hour - mu) / sigma) ** 2)


def _make_profile(peaks, base_daily_liters):
    """
    Build a 24-element list of L/h values that sums to base_daily_liters.
    peaks: list of (center_hour, sigma, relative_weight)
    """
    raw = [0.0] * 24
    for h in range(24):
        for (mu, sigma, w) in peaks:
            raw[h] += _gaussian(h, mu, sigma, w)
    total = sum(raw) or 1
    return [v / total * base_daily_liters for v in raw]


# ---------------------------------------------------------------------------
# Base household agent — Mesa 3.x API
# unique_id is auto-assigned by Mesa; do NOT pass it manually.
# ---------------------------------------------------------------------------

class WaterAgent(Agent):
    """
    Abstract base for all water-consuming agents.
    Subclasses define AGENT_TYPE, BASE_DAILY_L, and PEAKS.
    """

    AGENT_TYPE = "base"
    BASE_DAILY_L = 200
    PEAKS = []

    def __init__(self, model, interventions=None):
        super().__init__(model)                         # Mesa 3: no unique_id arg
        self.interventions = set(interventions or [])
        self._baseline_profile = _make_profile(self.PEAKS, self.BASE_DAILY_L)
        # Per-agent compliance (0.7–1.0)
        self.compliance = 0.7 + model.random.random() * 0.3

    def _savings_factor(self):
        """Combined savings from all active interventions, capped at 70%."""
        total = sum(
            INTERVENTIONS[i]["savings"] * self.compliance
            for i in self.interventions
            if i in INTERVENTIONS
        )
        return min(total, 0.70)

    def hourly_baseline(self, hour):
        return self._baseline_profile[hour]

    def hourly_intervention(self, hour):
        return self._baseline_profile[hour] * (1.0 - self._savings_factor())

    def daily_baseline(self):
        return sum(self._baseline_profile)

    def daily_intervention(self):
        return self.daily_baseline() * (1.0 - self._savings_factor())

    def daily_savings(self):
        return self.daily_baseline() - self.daily_intervention()

    def savings_pct(self):
        return self._savings_factor() * 100.0

    def step(self):
        pass


# ---------------------------------------------------------------------------
# Concrete agent types
# ---------------------------------------------------------------------------

class FamilyHouseholdAgent(WaterAgent):
    """
    Family with children.
    Peaks: morning (7h), noon (12h), evening (19h).
    Base: 320 L/day (4-person household).
    """
    AGENT_TYPE = "family"
    BASE_DAILY_L = 320
    PEAKS = [
        (7,  1.2, 3.0),
        (12, 1.0, 2.2),
        (19, 1.3, 3.5),
    ]


class WorkingAdultAgent(WaterAgent):
    """
    Household of working adults (no children).
    Peaks: morning (7h), evening (20h). Low during office hours.
    Base: 220 L/day (2-person household).
    """
    AGENT_TYPE = "working"
    BASE_DAILY_L = 220
    PEAKS = [
        (7,  1.0, 3.5),
        (20, 1.2, 3.0),
    ]


class WorkplaceSchoolAgent(WaterAgent):
    """
    Office building or school.
    Steady daytime use (8–16h), near-zero outside hours.
    Base: 800 L/day (~30-person unit).
    """
    AGENT_TYPE = "workplace"
    BASE_DAILY_L = 800
    PEAKS = [
        (8,  1.5, 1.2),
        (10, 2.0, 1.5),
        (12, 2.5, 1.8),
        (14, 2.0, 1.5),
        (16, 1.5, 1.0),
    ]


AGENT_CLASSES = {
    "family":    FamilyHouseholdAgent,
    "working":   WorkingAdultAgent,
    "workplace": WorkplaceSchoolAgent,
}
