"""
hydraulics.py
Simplified water distribution network hydraulic solver.

Network topology
----------------
A branching tree network (no loops) rooted at a single reservoir/source
node. Each agent in the ABM is assigned to one leaf node. Junction nodes
aggregate demand from their subtree.

Pipe model — Hazen-Williams head loss
--------------------------------------
  h_f = 10.67 * L * Q^1.852 / (C^1.852 * D^4.87)

  h_f : head loss [m]
  L   : pipe length [m]
  Q   : flow rate [m³/s]
  C   : Hazen-Williams roughness coefficient (dimensionless)
  D   : pipe internal diameter [m]

Pressure at each node
---------------------
  H_node = H_source - Σ h_f  (sum of head losses on path from source)

  Converted to gauge pressure [bar]:
  P [bar] = ρ g H / 1e5    (ρ=1000 kg/m³, g=9.81 m/s²)

Nominal operating pressure
--------------------------
  Chosen so that under average baseline demand the source delivers
  H_source = 40 m (≈ 3.9 bar) — a typical urban water service pressure.
  All deviations are reported relative to this nominal.

Units in this module
--------------------
  - Demand Q: m³/s  (converted internally from L/h agent outputs)
  - Head H: m
  - Pressure P: bar
  - Length L: m
  - Diameter D: m
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO     = 1000.0    # water density kg/m³
G       = 9.81      # gravity m/s²
L_TO_M3S = 1 / 3_600_000  # L/h → m³/s

# ---------------------------------------------------------------------------
# Default pipe parameters
# ---------------------------------------------------------------------------
HW_C    = 105       # Hazen-Williams C (aging cast iron/steel, Bucharest ~1960s–80s network)
SOURCE_HEAD = 50.0  # m   — reservoir/pump delivery head (~4.9 bar, typical Bucharest residential zone)


def hazen_williams_head_loss(Q_m3s, length_m, diameter_m, C=HW_C):
    """
    Head loss [m] in a single pipe segment.
    Returns 0 if Q == 0.
    """
    if Q_m3s <= 0:
        return 0.0
    # h_f = 10.67 * L * Q^1.852 / (C^1.852 * D^4.87)
    return (10.67 * length_m * (Q_m3s ** 1.852)
            / (C ** 1.852 * diameter_m ** 4.87))


def head_to_pressure_bar(head_m):
    """Convert hydraulic head [m] to gauge pressure [bar]."""
    return (RHO * G * head_m) / 1e5


# ---------------------------------------------------------------------------
# Network topology builder
# ---------------------------------------------------------------------------

class PipeNetwork:
    """
    Branching pipe network.  Structure (fixed for all model runs)::

        SOURCE(0) --trunk-- MAIN_JCT(1)
                              /    |    \
                         ZA(2) ZB(3) ZC(4)
                         /|    /|    /|
                       L  L  L  L  L  L   (leaf nodes)
         L(5) L(6)     L(7) L(8)     L(9) L(10)  ...leaf nodes

    Leaf nodes are demand nodes; each agent is assigned to exactly one leaf.
    Agents are distributed to leaf nodes in creation order (round-robin).

    Parameters
    ----------
    n_agents   : int   total number of demand agents
    n_branches : int   number of zone branches off the main junction (default 3)
    n_leaves   : int   number of leaf nodes per zone branch (default varies)
    """

    def __init__(self, n_agents, n_branches=3, seed=42):
        rng = np.random.default_rng(seed)

        self.n_agents   = n_agents
        self.n_branches = n_branches

        # ── Build node list ──────────────────────────────────────────────
        # Node 0: source
        # Node 1: main junction
        # Nodes 2 .. 1+n_branches: zone junctions
        # Remaining nodes: leaf nodes (one per agent, grouped by zone)
        self.nodes = []          # list of node dicts
        self.edges = []          # list of (from, to, length_m, diameter_m)
        self.node_agents = {}    # node_id -> list of agent indices

        # source
        self._add_node("source", elevation=10.0)
        # main junction
        self._add_node("junction")

        # trunk pipe: source → main junction
        # Bucharest district feeder: 150–300 m, 80–150 mm
        self._add_pipe(0, 1,
                       length=rng.uniform(150, 300),
                       diameter=rng.uniform(0.080, 0.150))

        # zone junctions + leaves
        # Pipe diameters are sized so that at peak baseline demand the
        # network delivers ~3.0 bar at leaf nodes (SOURCE_HEAD=40m gives
        # ~3.9 bar; ~3 m of head loss across the full tree).
        #   trunk  : ~47mm  (carries full network flow)
        #   branch : ~30mm  (carries ~1/n_branches of total flow)
        #   lateral: ~13mm  (carries demand of a few leaf nodes)
        leaves_per_zone = self._distribute(n_agents, n_branches)
        leaf_id = 1 + n_branches + 1   # first leaf node id

        for b in range(n_branches):
            zone_id = 2 + b
            self._add_node("zone_junction")

            # branch pipe: main junction → zone junction
            # Bucharest zone distributor: 80–150 m, 50–80 mm
            self._add_pipe(1, zone_id,
                           length=rng.uniform(80, 150),
                           diameter=rng.uniform(0.050, 0.080))

            for _ in range(leaves_per_zone[b]):
                self._add_node("leaf")
                self.node_agents[leaf_id] = []   # filled later by model

                # lateral pipe: zone junction → leaf
                # Bucharest service connection: 25–50 m, 20–32 mm
                self._add_pipe(zone_id, leaf_id,
                               length=rng.uniform(25, 50),
                               diameter=rng.uniform(0.020, 0.032))
                leaf_id += 1

        # convenience: list of leaf node ids
        self.leaf_ids = [
            n["id"] for n in self.nodes if n["type"] == "leaf"
        ]

        # precompute path from each node to source (as list of edge indices)
        self._paths = self._compute_paths()

    # ------------------------------------------------------------------
    def _add_node(self, ntype, elevation=0.0):
        nid = len(self.nodes)
        self.nodes.append({"id": nid, "type": ntype, "elevation": elevation})
        return nid

    def _add_pipe(self, fr, to, length, diameter):
        self.edges.append({
            "from": fr, "to": to,
            "length": length, "diameter": diameter,
        })

    @staticmethod
    def _distribute(total, buckets):
        """Distribute total items as evenly as possible into buckets."""
        base, rem = divmod(total, buckets)
        return [base + (1 if i < rem else 0) for i in range(buckets)]

    def _compute_paths(self):
        """For each node, find the list of edge indices on the path from source."""
        # Build adjacency (parent → children via edges)
        children = {n["id"]: [] for n in self.nodes}
        for i, e in enumerate(self.edges):
            children[e["from"]].append((e["to"], i))

        paths = {0: []}   # source has empty path
        queue = [0]
        while queue:
            nid = queue.pop(0)
            for child_id, edge_idx in children[nid]:
                paths[child_id] = paths[nid] + [edge_idx]
                queue.append(child_id)
        return paths

    # ------------------------------------------------------------------
    def assign_agents(self, agent_list):
        """Round-robin assignment of agents to leaf nodes."""
        for leaf_id in self.leaf_ids:
            self.node_agents[leaf_id] = []
        for i, agent in enumerate(agent_list):
            leaf = self.leaf_ids[i % len(self.leaf_ids)]
            self.node_agents[leaf].append(agent)

    # ------------------------------------------------------------------
    def solve(self, hour, scenario="baseline"):
        """
        Compute steady-state pressure [bar] at every node for a given hour.

        Returns
        -------
        pressures : dict  node_id -> pressure_bar
        flows     : dict  edge_idx -> Q_m3s
        heads     : dict  node_id -> head_m
        """
        # Step 1: demand at each leaf node [m³/s]
        demand = {}
        for leaf_id in self.leaf_ids:
            agents = self.node_agents.get(leaf_id, [])
            if scenario == "baseline":
                q_lh = sum(a.hourly_baseline(hour) for a in agents)
            else:
                q_lh = sum(a.hourly_intervention(hour) for a in agents)
            demand[leaf_id] = q_lh * L_TO_M3S

        # Step 2: flow in each pipe = sum of downstream demands (tree property)
        flows = {}
        for edge_idx, edge in enumerate(self.edges):
            downstream_leaves = self._downstream_leaves(edge["to"])
            flows[edge_idx] = sum(demand.get(l, 0) for l in downstream_leaves)

        # Step 3: head at each node = SOURCE_HEAD − cumulative head loss
        heads = {0: SOURCE_HEAD}
        for nid in range(1, len(self.nodes)):
            path = self._paths.get(nid, [])
            cumulative_loss = sum(
                hazen_williams_head_loss(
                    flows[ei],
                    self.edges[ei]["length"],
                    self.edges[ei]["diameter"],
                )
                for ei in path
            )
            heads[nid] = SOURCE_HEAD - cumulative_loss

        # Step 4: convert to pressure [bar], correcting for node elevation
        pressures = {}
        for nid, h in heads.items():
            elev = self.nodes[nid]["elevation"]
            pressures[nid] = head_to_pressure_bar(max(h - elev, 0))

        return pressures, flows, heads

    def _downstream_leaves(self, start_node):
        """All leaf node ids reachable from start_node (inclusive)."""
        children_map = {n["id"]: [] for n in self.nodes}
        for e in self.edges:
            children_map[e["from"]].append(e["to"])

        result = []
        stack = [start_node]
        while stack:
            nid = stack.pop()
            if self.nodes[nid]["type"] == "leaf":
                result.append(nid)
            stack.extend(children_map[nid])
        return result

    # ------------------------------------------------------------------
    def compute_mean_pressures(self, scenario="baseline"):
        """
        Mean pressure [bar] at every node averaged over all 24 hours
        for the given scenario. Used as a reusable building block.
        """
        all_pressures = {nid: [] for nid in range(len(self.nodes))}
        for h in range(24):
            p, _, _ = self.solve(h, scenario=scenario)
            for nid, pval in p.items():
                all_pressures[nid].append(pval)
        return {nid: float(np.mean(vals)) for nid, vals in all_pressures.items()}

    # ------------------------------------------------------------------
    def pressure_stats(self, scenario="baseline", nominal=None):
        """
        Per-node pressure statistics over all 24 hours.

        Parameters
        ----------
        scenario : "baseline" or "intervention"
        nominal  : dict node_id -> pressure_bar to use as the reference
                   for deviation calculations.  If None, uses this
                   network's own baseline pressures (self-referential,
                   deviation = 0 — only useful for structural checks).
                   Pass the pressures from a zero-adoption baseline model
                   to get meaningful deviations.
        """
        if nominal is None:
            nominal = self.compute_mean_pressures("baseline")

        stats = {nid: {"pressures": []} for nid in range(len(self.nodes))}
        for h in range(24):
            p, _, _ = self.solve(h, scenario=scenario)
            for nid, pval in p.items():
                stats[nid]["pressures"].append(pval)

        result = {}
        for nid, s in stats.items():
            vals = s["pressures"]
            result[nid] = {
                "mean_bar":      float(np.mean(vals)),
                "min_bar":       float(np.min(vals)),
                "max_bar":       float(np.max(vals)),
                "deviation_bar": float(np.mean(vals)) - nominal.get(nid, 0.0),
            }

        leaf_devs = [result[lid]["deviation_bar"] for lid in self.leaf_ids]
        result["_network"] = {
            "mean_deviation_bar": float(np.mean(leaf_devs)),
            "max_deviation_bar":  float(np.min(leaf_devs)),
            "std_deviation_bar":  float(np.std(leaf_devs)),
        }
        return result
