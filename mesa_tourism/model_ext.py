# model.py — CityModelExt with single source of truth for capacity/queueing
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

from agent import TouristGroup, calculate_travel_time


# If you already have a Router elsewhere, import it. This is only a minimal stub.
class Router:
    """Minimal wrapper; pass an instance with a NetworkX MultiDiGraph in .G"""
    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
    def node_xy(self, node) -> Tuple[float, float]:
        d = self.G.nodes[node]
        # tolerant to string/float storage
        x = float(d.get("x", d.get("lon")))
        y = float(d.get("y", d.get("lat")))
        return x, y


# -----------------------------
# POI with capacity + FIFO queue
# (no inside counters here — model is the single source of truth)
# -----------------------------
@dataclass
class POI:
    id: int
    node: object
    name: str
    capacity: int
    queue: Deque[Tuple[int, int, bool]] = field(default_factory=deque)  # (agent_id, group_size, is_guided)

    # queue utilities
    def _find_queue_index(self, agent_id: int) -> int:
        for i, (aid, _, _) in enumerate(self.queue):
            if aid == agent_id:
                return i
        return -1

    def in_queue(self, agent_id: int) -> bool:
        return self._find_queue_index(agent_id) >= 0

    def enqueue_once(self, agent_id: int, group_size: int, is_guided: bool):
        if not self.in_queue(agent_id):
            self.queue.append((agent_id, group_size, is_guided))

    def remove_from_queue(self, agent_id: int):
        idx = self._find_queue_index(agent_id)
        if idx >= 0:
            self.queue = deque(list(self.queue)[:idx] + list(self.queue)[idx + 1:])


class CityModelExt(Model):
    """Simulation environment with optional OSM routing and POI capacity/queueing."""
    def __init__(
        self,
        width, height,
        num_tourists, guided_ratio, total_time_steps,
        guided_start_times, self_guided_start_window, self_guided_start_interval,
        guided_wait_time, self_guided_wait_time,
        guided_group_size, self_guided_group_size,
        itinerary,
        *,
        router: Optional[Router] = None,                # None → simplified mode on grid
        poi_capacity_range: Tuple[int, int] = (25, 60),
        poi_coords: Optional[List[Tuple[int, int]]] = None,
        clustered: bool = True,
        num_clusters: int = 2,
        cluster_radius: int = 8,
    ):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)

        self.num_tourists = num_tourists
        self.guided_ratio = guided_ratio
        self.total_time_steps = total_time_steps
        self.guided_wait_time = guided_wait_time           # (lo, hi)
        self.self_guided_wait_time = self_guided_wait_time # (lo, hi)
        self.schedule_time = 0

        self.router = router

        # --- Create POIs ---
        max_group_size = max(guided_group_size, self_guided_group_size)
        self.pois: List[POI] = []

        if self.router is not None:
            num_pois = 8 if not itinerary else len(itinerary)
            nodes = list(self.router.G.nodes())
            if len(nodes) < num_pois:
                raise RuntimeError("OSM graph too small for the requested number of POIs.")
            sampled_nodes = random.sample(nodes, num_pois)
            for i, node in enumerate(sampled_nodes):
                cap = max(random.randint(*poi_capacity_range), max_group_size)
                self.pois.append(POI(id=i, node=node, name=f"POI-{i+1}", capacity=cap))
            self.itinerary = self._project_pois_to_grid(self.pois, width, height)
        else:
            # Simplified: use provided coords or synthesize clustered POIs on the grid
            if poi_coords:
                coords = poi_coords[:]
            else:
                num_pois = 8 if not itinerary else len(itinerary)
                coords = self._synthesize_poi_coords(
                    width, height, n=num_pois,
                    clustered=clustered, num_clusters=num_clusters, r=cluster_radius
                )
            for i, (gx, gy) in enumerate(coords):
                cap = max(random.randint(*poi_capacity_range), max_group_size)
                self.pois.append(POI(id=i, node=(gx, gy), name=f"POI-{i+1}", capacity=cap))
            self.itinerary = coords

        # --- Allocate agents (as in your original logic) ---
        self.guided_groups: List[TouristGroup] = []
        self.self_guided_groups: List[TouristGroup] = []

        max_guided_travelers = int(num_tourists * guided_ratio)
        max_self_guided_travelers = num_tourists - max_guided_travelers

        num_guided_groups = max_guided_travelers // guided_group_size
        guided_travelers_assigned = 0
        guided_start_time_index = 0
        for _ in range(num_guided_groups):
            if guided_travelers_assigned + guided_group_size > max_guided_travelers:
                break
            start_time = guided_start_times[guided_start_time_index]
            guided_start_time_index = (guided_start_time_index + 1) % len(guided_start_times)
            agent = TouristGroup(
                len(self.guided_groups), self, guided=True, start_time=start_time,
                guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time,
                group_size=guided_group_size
            )
            self.grid.place_agent(agent, self.itinerary[0])
            self.guided_groups.append(agent)
            guided_travelers_assigned += guided_group_size

        num_self_guided_groups = max_self_guided_travelers // self_guided_group_size
        self_guided_travelers_assigned = 0
        lo, hi = self_guided_start_window
        for _ in range(num_self_guided_groups):
            start_time = np.random.randint(lo, hi)
            start_time -= (start_time % self_guided_start_interval)
            group_size = min(
                np.random.randint(2, self_guided_group_size + 1),
                max_self_guided_travelers - self_guided_travelers_assigned
            )
            if group_size <= 0:
                break
            agent = TouristGroup(
                len(self.self_guided_groups), self, guided=False, start_time=start_time,
                guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time,
                group_size=group_size
            )
            self.grid.place_agent(agent, random.choice(self.itinerary))
            self.self_guided_groups.append(agent)
            self_guided_travelers_assigned += group_size

        total_assigned = guided_travelers_assigned + self_guided_travelers_assigned
        if total_assigned != num_tourists:
            print(f"Mismatch in assigned travelers: {total_assigned} != {num_tourists}")

        # --- State used for capacity/queueing ---
        self._agent_last_traveling: Dict[int, bool] = {}
        self._agent_last_remaining: Dict[int, int] = {}
        self._agent_current_poi_idx: Dict[int, int] = {}  # POI index agent is at (inside or queued)
        self._agents_inside: Dict[int, int] = {}          # agent_id -> group_size
        self._agents_queued: Dict[int, int] = {}          # agent_id -> group_size
        self._agents_by_id = {ag.unique_id: ag for ag in (self.guided_groups + self.self_guided_groups)}

        for ag in (self.guided_groups + self.self_guided_groups):
            self._agent_last_traveling[ag.unique_id] = bool(ag.traveling)
            self._agent_last_remaining[ag.unique_id] = int(ag.remaining_time)

        # --- DataCollector: metrics from model truth ---
        self.datacollector = DataCollector(
            model_reporters={
                "Total Congestion": CityModelExt.compute_total_congestion,
                "Guided Congestion": CityModelExt.compute_guided_congestion,
                "Self-Guided Congestion": CityModelExt.compute_self_guided_congestion,
                "Guided Completion Rate": CityModelExt.compute_guided_completion_rate,
                "Self-Guided Completion Rate": CityModelExt.compute_self_guided_completion_rate,
                "Total Queued": CityModelExt.compute_total_queued,
            }
        )

    # --------- helpers ----------
    def _synthesize_poi_coords(self, width: int, height: int, n: int,
                               clustered: bool, num_clusters: int, r: int) -> List[Tuple[int, int]]:
        if not clustered:
            return [(random.randint(2, width - 3), random.randint(2, height - 3)) for _ in range(n)]
        centers = [(random.randint(10, width - 10), random.randint(10, height - 10)) for _ in range(num_clusters)]
        coords = []
        for i in range(n):
            cx, cy = centers[i % num_clusters]
            x = min(max(1, int(np.random.normal(cx, r))), width - 2)
            y = min(max(1, int(np.random.normal(cy, r))), height - 2)
            coords.append((x, y))
        return coords

    def _project_pois_to_grid(self, pois: List[POI], width: int, height: int) -> List[Tuple[int, int]]:
        if self.router is None:
            return [p.node for p in pois]  # already grid coords
        xs, ys = [], []
        for p in pois:
            x, y = self.router.node_xy(p.node)  # lon, lat
            xs.append(x); ys.append(y)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        def proj(x, y):
            gx = int((x - xmin) / max(1e-9, (xmax - xmin)) * (width - 1))
            gy = int((y - ymin) / max(1e-9, (ymax - ymin)) * (height - 1))
            return (gx, gy)

        return [proj(*self.router.node_xy(p.node)) for p in pois]

    # Single source of truth: compute inside for a given POI from model state
    def inside_count(self, poi_idx: int) -> int:
        return sum(sz for aid, sz in self._agents_inside.items()
                   if self._agent_current_poi_idx.get(aid) == poi_idx)

    # ---------- metrics from model truth ----------
    def compute_total_queued(self):
        return sum(self._agents_queued.values())

    def compute_total_congestion(self):
        inside = sum(self._agents_inside.values())
        queued = sum(self._agents_queued.values())
        return inside + queued

    def compute_guided_congestion(self):
        guided_inside = sum(sz for aid, sz in self._agents_inside.items() if self._is_guided_agent(aid))
        guided_queued = sum(sz for aid, sz in self._agents_queued.items() if self._is_guided_agent(aid))
        return guided_inside + guided_queued

    def compute_self_guided_congestion(self):
        self_inside = sum(sz for aid, sz in self._agents_inside.items() if not self._is_guided_agent(aid))
        self_queued = sum(sz for aid, sz in self._agents_queued.items() if not self._is_guided_agent(aid))
        return self_inside + self_queued

    def _is_guided_agent(self, agent_id: int) -> bool:
        return any(ag.unique_id == agent_id for ag in self.guided_groups)

    def compute_guided_completion_rate(self):
        completed_groups = [group for group in self.guided_groups if group.completed]
        return len(completed_groups) / len(self.guided_groups) if self.guided_groups else 0

    def compute_self_guided_completion_rate(self):
        completed_groups = [group for group in self.self_guided_groups if group.completed]
        return len(completed_groups) / len(self.self_guided_groups) if self.self_guided_groups else 0

    # ---------- core step ----------
    def step(self):
        self.schedule_time += 1

        # Freeze queued agents so agent.py doesn't "walk away" from the queue
        for gid in list(self._agents_queued.keys()):
            ag = self._agents_by_id.get(gid)
            if ag is None:
                self._agents_queued.pop(gid, None)
                continue
            ag.traveling = False
            if getattr(ag, "remaining_time", 0) <= 0:
                ag.remaining_time = 1

        # Advance agents
        for group in self.guided_groups + self.self_guided_groups:
            group.step()

        # Post-move capacity & queue logic
        for group in self.guided_groups + self.self_guided_groups:
            gid = group.unique_id
            was_trav = self._agent_last_traveling.get(gid, False)
            was_rem  = self._agent_last_remaining.get(gid, 0)
            now_trav = bool(group.traveling)
            now_rem  = int(group.remaining_time)

            # Where is the group now (grid coord → nearest POI idx)?
            try:
                current_coord = group.path[group.current_step]
            except Exception:
                current_coord = None
            poi_idx = self._nearest_poi_index_from_grid(current_coord) if current_coord is not None else None

            # a) Arrival (traveling -> not traveling) at a POI
            if was_trav and not now_trav and not group.completed and poi_idx is not None:
                self._handle_arrival(group, poi_idx)

            # b) Departure (finished dwell or started traveling from inside)
            leaving_dwell = (was_rem > 0) and (now_rem == 0)
            started_travel_from_poi = (gid in self._agents_inside) and (not was_trav) and now_trav
            if (leaving_dwell or started_travel_from_poi) and (gid in self._agents_inside):
                self._handle_departure(group)

            # c) Retry admission for queued groups at this POI
            if (gid in self._agents_queued) and (now_rem == 0) and (not now_trav) and (poi_idx is not None):
                self._try_admit_from_queue(group, poi_idx)

            # d) Final cleanup if the group finished its tour
            if group.completed:
                self._cleanup_completion(group)

            # update last-state trackers
            self._agent_last_traveling[gid] = now_trav
            self._agent_last_remaining[gid] = now_rem

        # Reconcile POI queues to model-side map (defensive; cheap)
        for p in self.pois:
            if p.queue:
                p.queue = deque((aid, sz, guided) for (aid, sz, guided) in p.queue
                                if self._agents_queued.get(aid, 0) == sz)

        # Collect metrics
        self.datacollector.collect(self)

    # ----- capacity/queue helpers (model is source of truth) -----
    def _handle_arrival(self, group: TouristGroup, poi_idx: int):
        gid = group.unique_id
        p = self.pois[poi_idx]
        self._agent_current_poi_idx[gid] = poi_idx
        self._agents_queued.pop(gid, None)  # clear any previous queue state

        inside_now = self.inside_count(poi_idx)
        if inside_now + group.group_size <= p.capacity:
            # admit
            low, high = (self.guided_wait_time if group.guided else self.self_guided_wait_time)
            group.remaining_time = int(np.random.randint(low, high + 1))
            self._agents_inside[gid] = group.group_size
            p.remove_from_queue(gid)  # in case of stale record
        else:
            # queue
            group.remaining_time = 0
            self._agents_queued[gid] = group.group_size
            p.enqueue_once(gid, group.group_size, group.guided)

    def _handle_departure(self, group: TouristGroup):
        gid = group.unique_id
        # no POI.release here; model is truth
        self._agents_inside.pop(gid, None)
        # agent will advance naturally next step

    def _cleanup_completion(self, group: TouristGroup):
        gid = group.unique_id
        poi_idx = self._agent_current_poi_idx.get(gid)
        if poi_idx is not None:
            p = self.pois[poi_idx]
            p.remove_from_queue(gid)
        self._agents_inside.pop(gid, None)
        self._agents_queued.pop(gid, None)

    def _try_admit_from_queue(self, group: TouristGroup, poi_idx: int):
        gid = group.unique_id
        p = self.pois[poi_idx]
        inside_now = self.inside_count(poi_idx)
        if inside_now + group.group_size <= p.capacity:
            low, high = (self.guided_wait_time if group.guided else self.self_guided_wait_time)
            group.remaining_time = int(np.random.randint(low, high + 1))
            self._agents_inside[gid] = group.group_size
            self._agents_queued.pop(gid, None)
            p.remove_from_queue(gid)
        else:
            self._agents_inside.pop(gid, None)  # still not inside

    # ----- geometry helper -----
    def _nearest_poi_index_from_grid(self, coord: Tuple[int, int]) -> int:
        px, py = coord
        best_idx, best_d2 = 0, float("inf")
        for i, (x, y) in enumerate(self.itinerary):
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_idx, best_d2 = i, d2
        return best_idx


# -----------------------------
# (Optional) plotting
# -----------------------------
def plot_state_at_time(model, title="Tourist Distribution at Time Step", timestep=None):
    """Visualizes POIs and current group positions."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # POIs and annotations
    for idx, (x, y) in enumerate(model.itinerary):
        poi = model.pois[idx]
        qsz = sum(sz for _, sz, _ in poi.queue)
        inside = model.inside_count(idx)   # from model truth
        ax.scatter(x, y, s=120, c='black', marker='X')
        ax.text(x + 1, y + 1, f'{poi.name}\nCap:{poi.capacity} In:{inside} Q:{qsz}',
                fontsize=8, color='black')

    # arrows between POIs (visual hint only)
    for i in range(len(model.itinerary) - 1):
        x1, y1 = model.itinerary[i]
        x2, y2 = model.itinerary[i + 1]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='gray', lw=1.5))

    def draw_group(x, y, group, color, label):
        ax.scatter(x, y, s=group.group_size * 20, c=color, alpha=0.6, label=label)

    labels_drawn = {"Guided": False, "Self-Guided": False}

    # Guided tourists
    for group in model.guided_groups:
        if model.schedule_time >= group.start_time and not group.completed:
            if group.traveling and group.current_step < len(group.path):
                p1 = group.path[group.current_step - 1]
                p2 = group.path[group.current_step]
                ratio = 1 - group.travel_time / calculate_travel_time(p1, p2)
                x = p1[0] + (p2[0] - p1[0]) * ratio
                y = p1[1] + (p2[1] - p1[1]) * ratio
            else:
                x, y = group.path[group.current_step]
            draw_group(x, y, group, 'orange', 'Guided Group' if not labels_drawn["Guided"] else "")
            labels_drawn["Guided"] = True

    # Self-guided tourists
    for group in model.self_guided_groups:
        if model.schedule_time >= group.start_time and not group.completed:
            if group.traveling and group.current_step < len(group.path):
                p1 = group.path[group.current_step - 1]
                p2 = group.path[group.current_step]
                ratio = 1 - group.travel_time / calculate_travel_time(p1, p2)
                x = p1[0] + (p2[0] - p1[0]) * ratio
                y = p1[1] + (p2[1] - p1[1]) * ratio
            else:
                x, y = group.path[group.current_step]
            draw_group(x, y, group, 'blue', 'Self-Guided Group' if not labels_drawn["Self-Guided"] else "")
            labels_drawn["Self-Guided"] = True

    ax.scatter([], [], s=100, c='black', marker='X', label='POI')
    ax.set_title(f"{title} (Minute {model.schedule_time})" if timestep is None else f"{title} (Minute {timestep})")
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)
    ax.set_xlabel("Grid X (10m units)")
    ax.set_ylabel("Grid Y (10m units)")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(f'output/state_t{model.schedule_time}.png', dpi=fig.dpi)
    plt.show()
