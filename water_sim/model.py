"""
model.py
WaterNetworkModel: the top-level Mesa 3.x model.

Mesa 3.x changes applied:
  - No mesa.time / RandomActivation import — replaced by model.agents.shuffle_do()
  - Agents are auto-tracked in model.agents (AgentSet); no manual schedule needed
  - unique_id is auto-assigned; agents are created with Agent(model, ...) only
  - DataCollector agent_reporters iterate model.agents automatically

Hydraulic layer:
  - A PipeNetwork is constructed alongside the agent population.
  - Agents are assigned to leaf nodes (round-robin by creation order).
  - Hazen-Williams head-loss equations yield per-node pressures [bar].
  - Pressure deviation is defined relative to the nominal (baseline, average
    demand) operating pressure at each node.
"""

from mesa import Model
from mesa.datacollection import DataCollector
from agents import AGENT_CLASSES, INTERVENTIONS
from hydraulics import PipeNetwork


class WaterNetworkModel(Model):
    """
    Simulates a water distribution network composed of heterogeneous
    household and workplace agents.

    Parameters
    ----------
    n_families    : int   – number of FamilyHouseholdAgent instances
    n_working     : int   – number of WorkingAdultAgent instances
    n_workplaces  : int   – number of WorkplaceSchoolAgent instances
    interventions : list  – intervention IDs available in this scenario
    adoption_rate : float – fraction of agents (0.0–1.0) that actually
                            receive the interventions; the rest get none.
                            Default 1.0 (all agents adopt).
    seed          : int   – random seed for reproducibility
    """

    def __init__(
        self,
        n_families=30,
        n_working=40,
        n_workplaces=10,
        interventions=None,
        adoption_rate=1.0,
        seed=42,
    ):
        super().__init__(seed=seed)

        self.n_families   = n_families
        self.n_working    = n_working
        self.n_workplaces = n_workplaces
        self.interventions  = list(interventions or [])
        self.adoption_rate  = max(0.0, min(1.0, adoption_rate))

        # ----- create agents ------------------------------------------------
        for agent_type, count in [
            ("family",    n_families),
            ("working",   n_working),
            ("workplace", n_workplaces),
        ]:
            cls = AGENT_CLASSES[agent_type]
            for _ in range(count):
                adopted = (
                    self.interventions
                    if self.random.random() < self.adoption_rate
                    else []
                )
                cls(self, adopted)

        # ----- hydraulic network -------------------------------------------
        # Build a branching pipe network and assign agents to leaf nodes.
        # The network topology is seeded identically for all scenarios so
        # that pressure comparisons are meaningful (same pipes, same layout).
        n_agents = n_families + n_working + n_workplaces
        self.pipe_network = PipeNetwork(
            n_agents=n_agents,
            n_branches=3,
            seed=seed,          # fixed topology per seed
        )
        self.pipe_network.assign_agents(list(self.agents))

        # Pre-compute nominal pressures: build a temporary zero-adoption
        # baseline model with the same topology seed so the pipe layout
        # is identical, then record its mean pressures across 24h.
        # This gives a fixed reference that is independent of the current
        # scenario's adoption rate and intervention mix.
        _ref = WaterNetworkModel.__new__(WaterNetworkModel)
        _ref.interventions = []
        _ref.adoption_rate = 0.0
        # reuse real Model init minimally via object.__setattr__
        import random as _rnd
        object.__setattr__(_ref, 'random', _rnd.Random(seed))
        object.__setattr__(_ref, '_agents', self.pipe_network.__class__.__mro__)
        # Simpler: build a reference pipe network with zero-demand agents
        _ref_net = PipeNetwork(n_agents=n_agents, n_branches=3, seed=seed)
        # Use zero-demand proxy agents (anonymous objects with hourly_baseline=0)
        class _ZeroAgent:
            def hourly_baseline(self, h): return 0.0
            def hourly_intervention(self, h): return 0.0
        _zero_agents = [_ZeroAgent() for _ in range(n_agents)]

        # Actually: nominal = full baseline demand (all agents, no interventions)
        # Build a sibling pipe network with the same layout but agents in their
        # pure baseline state (hourly_baseline ignores interventions already).
        # So we can just use self.pipe_network with scenario="baseline" directly
        # — hourly_baseline() is always intervention-free.
        self._nominal_pressures = self.pipe_network.compute_mean_pressures("baseline")

        # ----- data collector -----------------------------------------------
        self.datacollector = DataCollector(
            model_reporters={
                "Total Daily Baseline (L)":     lambda m: m.total_daily_baseline(),
                "Total Daily Intervention (L)": lambda m: m.total_daily_intervention(),
                "Total Daily Savings (L)":      lambda m: m.total_daily_savings(),
                "Network Savings (%)":          lambda m: m.network_savings_pct(),
                "Mean Pressure Deviation (bar)":
                    lambda m: m.pipe_network.pressure_stats("intervention")
                              ["_network"]["mean_deviation_bar"],
            },
            agent_reporters={
                "Daily Baseline (L)":     lambda a: a.daily_baseline(),
                "Daily Intervention (L)": lambda a: a.daily_intervention(),
                "Daily Savings (L)":      lambda a: a.daily_savings(),
                "Agent Type":             lambda a: a.AGENT_TYPE,
            },
        )

        # Collect initial state
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    # Agent helpers
    # ------------------------------------------------------------------

    def agents_by_type(self, agent_type):
        return self.agents.select(lambda a: a.AGENT_TYPE == agent_type)

    # ------------------------------------------------------------------
    # Consumption metrics
    # ------------------------------------------------------------------

    def total_daily_baseline(self):
        return sum(a.daily_baseline() for a in self.agents)

    def total_daily_intervention(self):
        return sum(a.daily_intervention() for a in self.agents)

    def total_daily_savings(self):
        return self.total_daily_baseline() - self.total_daily_intervention()

    def network_savings_pct(self):
        base = self.total_daily_baseline()
        return (self.total_daily_savings() / base * 100) if base > 0 else 0.0

    # ------------------------------------------------------------------
    # Hourly profiles
    # ------------------------------------------------------------------

    def hourly_profile(self, scenario="baseline"):
        """Returns dict: {agent_type: [avg L/h per agent for h in 0..23]}"""
        profiles = {}
        for atype in ("family", "working", "workplace"):
            group = list(self.agents_by_type(atype))
            if not group:
                profiles[atype] = [0.0] * 24
                continue
            fn = "hourly_baseline" if scenario == "baseline" else "hourly_intervention"
            profiles[atype] = [
                sum(getattr(a, fn)(h) for a in group) / len(group)
                for h in range(24)
            ]
        return profiles

    def network_hourly_total(self, scenario="baseline"):
        """Total L/h across ALL agents for each hour."""
        profiles = self.hourly_profile(scenario)
        return [
            sum(
                profiles[atype][h] * len(list(self.agents_by_type(atype)))
                for atype in profiles
            )
            for h in range(24)
        ]

    # ------------------------------------------------------------------
    # Pressure API (thin wrappers around PipeNetwork)
    # ------------------------------------------------------------------

    def node_pressures(self, hour, scenario="baseline"):
        """
        Pressure [bar] at every node for a specific hour and scenario.
        Returns dict: node_id -> pressure_bar
        """
        pressures, _, _ = self.pipe_network.solve(hour, scenario)
        return pressures

    def pressure_stats(self, scenario="baseline"):
        """
        Per-node pressure statistics over all 24 hours.
        Returns dict: node_id -> {mean_bar, min_bar, max_bar, deviation_bar}
        Plus '_network' key with aggregate scalars.
        """
        return self.pipe_network.pressure_stats(scenario, nominal=self._nominal_pressures)

    def mean_pressure_deviation(self, scenario="intervention"):
        """
        Scalar: mean pressure deviation [bar] from nominal across all
        demand (leaf) nodes, averaged over 24 hours.
        Negative = pressure drop relative to baseline operating conditions.
        """
        return self.pressure_stats(scenario)["_network"]["mean_deviation_bar"]

    def node_pressure_profile(self, node_id, scenario="baseline"):
        """24-hour pressure profile [bar] for a single node."""
        return [
            self.pipe_network.solve(h, scenario)[0][node_id]
            for h in range(24)
        ]

    # ------------------------------------------------------------------
    # Intervention breakdown
    # ------------------------------------------------------------------

    def intervention_savings_breakdown(self):
        result = {}
        total_base = self.total_daily_baseline()
        all_agents = list(self.agents)
        avg_compliance = sum(a.compliance for a in all_agents) / len(all_agents)
        for inv_id in self.interventions:
            inv = INTERVENTIONS.get(inv_id, {})
            savings_frac = inv.get("savings", 0)
            result[inv_id] = {
                "label":       inv.get("label", inv_id),
                "category":    inv.get("category", ""),
                "savings_L":   total_base * savings_frac * avg_compliance * self.adoption_rate,
                "savings_pct": savings_frac * avg_compliance * self.adoption_rate * 100,
            }
        return result

    def per_type_summary(self):
        summary = {}
        for atype in ("family", "working", "workplace"):
            group = list(self.agents_by_type(atype))
            if not group:
                continue
            base_vals = [a.daily_baseline() for a in group]
            intv_vals = [a.daily_intervention() for a in group]
            summary[atype] = {
                "count":                len(group),
                "avg_baseline_L":       sum(base_vals) / len(base_vals),
                "avg_intervention_L":   sum(intv_vals) / len(intv_vals),
                "avg_savings_pct":      sum(a.savings_pct() for a in group) / len(group),
                "total_baseline_L":     sum(base_vals),
                "total_intervention_L": sum(intv_vals),
            }
        return summary
