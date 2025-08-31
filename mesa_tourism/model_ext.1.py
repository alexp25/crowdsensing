# model.py — CityModelExt that accepts a Router instance (no internal OSM loading)

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

# If you already have a Router class elsewhere, you can import it instead.
class Router:
    """Minimal wrapper; pass an instance with a NetworkX MultiDiGraph in .G"""
    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
    def node_xy(self, node: int) -> Tuple[float, float]:
        d = self.G.nodes[node]
        return d["x"], d["y"]  # lon, lat

# -----------------------------------
# POI with capacity and FIFO queueing
# -----------------------------------
@dataclass
class POI:
    id: int
    node: int
    name: str
    capacity: int
    inside_total: int = 0
    queue: Deque[Tuple[int, int, bool]] = field(default_factory=deque)  # (agent_id, group_size, is_guided)

    # ---- queue utilities ----
    def _find_queue_index(self, agent_id: int):
        for i, (aid, _, _) in enumerate(self.queue):
            if aid == agent_id:
                return i
        return -1

    def in_queue(self, agent_id: int) -> bool:
        return self._find_queue_index(agent_id) >= 0

    def remove_from_queue(self, agent_id: int):
        idx = self._find_queue_index(agent_id)
        if idx >= 0:
            # deque has no pop(index) → rebuild without that item
            self.queue = deque(list(self.queue)[:idx] + list(self.queue)[idx+1:])

    # ---- capacity / admission ----
    def can_enter(self, group_size: int) -> bool:
        return (self.inside_total + group_size) <= self.capacity

    def try_enter(self, agent_id: int, group_size: int, is_guided: bool) -> bool:
        if self.can_enter(group_size):
            if self.in_queue(agent_id):
                self.remove_from_queue(agent_id)
            self.inside_total += group_size
            return True
        if not self.in_queue(agent_id):
            self.queue.append((agent_id, group_size, is_guided))
        return False
    

    def release(self, group_size: int):
        """Only free capacity; DO NOT auto-admit from queue here.
        The model will admit queued groups so it can also update its own maps."""
        self.inside_total = max(0, self.inside_total - group_size)



class CityModelExt(Model):
    """Represents the entire simulation environment with OSM routing + POI capacity/queueing."""
    def __init__(
        self,
        width, height,
        num_tourists, guided_ratio, total_time_steps,
        guided_start_times, self_guided_start_window, self_guided_start_interval,
        guided_wait_time, self_guided_wait_time,
        guided_group_size, self_guided_group_size,
        itinerary,                               # kept for compatibility but will be replaced with POIs projected to grid
        *,
        router: Router,                          # <<< REQUIRED: pass a Router instance
        poi_capacity_range: Tuple[int, int] = (250, 600),
        poi_coords: Optional[List[Tuple[int, int]]] = None,  # ← grid coords for simplified mode
        clustered: bool = True,                    # ← synthesize clustered POIs if coords not given
        num_clusters: int = 2,
        cluster_radius: int = 8
    ):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)

        self.num_tourists = num_tourists
        self.guided_ratio = guided_ratio
        self.total_time_steps = total_time_steps
        self.guided_wait_time = guided_wait_time
        self.self_guided_wait_time = self_guided_wait_time
        self.schedule_time = 0

        # Router can be None (simplified mode)
        self.router = router

        # --- Create POIs ---
        # if router provided → OSM mode; else → simplified mode on grid
        max_group_size = max(guided_group_size, self_guided_group_size)
        self.pois: List[POI] = []

        if self.router is not None:
            # OSM mode (unchanged logic, but keep capacity >= max group size)
            num_pois = 8 if not itinerary else len(itinerary)
            nodes = list(self.router.G.nodes())
            if len(nodes) < num_pois:
                raise RuntimeError("OSM graph too small for the requested number of POIs.")
            sampled_nodes = random.sample(nodes, num_pois)
            for i, node in enumerate(sampled_nodes):
                cap = max(random.randint(*poi_capacity_range), max_group_size)
                self.pois.append(POI(id=i, node=node, name=f"POI-{i+1}", capacity=cap))

            # project OSM nodes to grid for plotting
            self.itinerary = self._project_pois_to_grid(self.pois, width, height)

        else:
            # Simplified mode (no OSM): make POIs directly on the grid
            if poi_coords:
                coords = poi_coords[:]  # use provided
            else:
                # synthesize clustered or uniform POIs on the grid
                num_pois = 8 if not itinerary else len(itinerary)
                coords = self._synthesize_poi_coords(
                    width, height, n=num_pois,
                    clustered=clustered, num_clusters=num_clusters, r=cluster_radius
                )
            # Build POIs; store the grid coord in 'node' for identification (not used as OSM node)
            for i, (gx, gy) in enumerate(coords):
                cap = max(random.randint(*poi_capacity_range), max_group_size)
                # store tuple coord as 'node' so our helpers can still hang onto an identifier
                self.pois.append(POI(id=i, node=(gx, gy), name=f"POI-{i+1}", capacity=cap))

            # in simplified mode, the itinerary is exactly the grid coords
            self.itinerary = coords

        # --- Allocate agents (keep your original logic) ---
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
            # place on grid for plotting; start at first POI
            self.grid.place_agent(agent, self.itinerary[0])
            self.guided_groups.append(agent)
            guided_travelers_assigned += guided_group_size

        # Self-guided allocation
        num_self_guided_groups = max_self_guided_travelers // self_guided_group_size
        self_guided_travelers_assigned = 0
        for _ in range(num_self_guided_groups):
            start_time = np.random.randint(*self_guided_start_window)
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
            # place on grid for plotting; start on a random POI
            self.grid.place_agent(agent, random.choice(self.itinerary))
            self.self_guided_groups.append(agent)
            self_guided_travelers_assigned += group_size

        total_assigned = guided_travelers_assigned + self_guided_travelers_assigned
        if total_assigned != num_tourists:
            print(f"Mismatch in assigned travelers: {total_assigned} != {num_tourists}")

        # --- Track per-agent transitions to enforce capacity/queueing without touching agent.py ---
        # We detect arrivals (traveling True -> False), dwell completion (remaining_time >0 -> 0), and manage POI admission.
        self._agent_last_traveling: Dict[int, bool] = {}
        self._agent_last_remaining: Dict[int, int] = {}
        self._agent_current_poi_idx: Dict[int, int] = {}  # which POI the agent is inside/queued for
        self._agents_inside: Dict[int, int] = {}          # agent_id -> group_size (currently inside)
        self._agents_queued: Dict[int, int] = {}          # agent_id -> group_size (currently queued)


        self._agents_by_id = {ag.unique_id: ag for ag in (self.guided_groups + self.self_guided_groups)}

        # init last-state dicts
        for ag in (self.guided_groups + self.self_guided_groups):
            self._agent_last_traveling[ag.unique_id] = bool(ag.traveling)
            self._agent_last_remaining[ag.unique_id] = int(ag.remaining_time)

        # --- DataCollector now uses true inside+queue counts from POIs ---
        self.datacollector = DataCollector(
            model_reporters={
                "Total Congestion": self.compute_total_congestion,
                "Guided Congestion": self.compute_guided_congestion,
                "Self-Guided Congestion": self.compute_self_guided_congestion,
                "Guided Completion Rate": self.compute_guided_completion_rate,
                "Self-Guided Completion Rate": self.compute_self_guided_completion_rate,
                "Total Queued": CityModelExt.compute_total_queued,
            }
        )


        # --- helper: synthesize POI coords for simplified mode ---
    def _synthesize_poi_coords(self, width: int, height: int, n: int,
                               clustered: bool, num_clusters: int, r: int) -> List[Tuple[int, int]]:
        if not clustered:
            # uniform random placement, keep away from borders a bit
            return [(random.randint(2, width-3), random.randint(2, height-3)) for _ in range(n)]

        # clustered: pick cluster centers, then sample around them
        centers = [(random.randint(10, width-10), random.randint(10, height-10)) for _ in range(num_clusters)]
        coords = []
        for i in range(n):
            cx, cy = centers[i % num_clusters]
            # small jitter around each center
            x = min(max(1, int(np.random.normal(cx, r))), width-2)
            y = min(max(1, int(np.random.normal(cy, r))), height-2)
            coords.append((x, y))
        return coords

    # --- projection: OSM → grid (as before), or identity in simplified mode ---
    def _project_pois_to_grid(self, pois: List[POI], width: int, height: int) -> List[Tuple[int, int]]:
        if self.router is None:
            # simplified mode: nodes are already grid coords
            return [p.node for p in pois]  # type: ignore[return-value]

        xs, ys = [], []
        for p in pois:
            x, y = self.router.node_xy(p.node)  # lon, lat
            xs.append(x); ys.append(y)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        def proj(x, y):
            gx = int((x - xmin) / max(1e-9, xmax - xmin) * (width - 1))
            gy = int((y - ymin) / max(1e-9, ymax - ymin) * (height - 1))
            return (gx, gy)

        return [proj(*self.router.node_xy(p.node)) for p in pois]

    # ---------- OSM->grid projection for plotting ----------
    def _project_pois_to_grid(self, pois: List[POI], width: int, height: int) -> List[Tuple[int, int]]:
        xs, ys = [], []
        for p in pois:
            x, y = self.router.node_xy(p.node)  # lon, lat
            xs.append(x); ys.append(y)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        def proj(x, y):
            gx = int((x - xmin) / max(1e-9, xmax - xmin) * (width - 1))
            gy = int((y - ymin) / max(1e-9, ymax - ymin) * (height - 1))
            return (gx, gy)

        return [proj(*self.router.node_xy(p.node)) for p in pois]

    def compute_total_queued(self):
        # return sum(sum(sz for _, sz, _ in p.queue) for p in self.pois)
        return sum(self._agents_queued.values())

    # ---------- Admission helpers that map an agent's grid position to the nearest POI ----------
    def _nearest_poi_index_from_grid(self, coord: Tuple[int, int]) -> int:
        # Find nearest projected POI on the plot grid (simple L2 distance)
        px, py = coord
        best_idx, best_d2 = 0, float("inf")
        for i, (x, y) in enumerate(self.itinerary):
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_idx, best_d2 = i, d2
        return best_idx

    # ---------- Congestion metrics based on actual POI states ----------
    def compute_total_congestion(self):
        inside = sum(p.inside_total for p in self.pois)
        queued = sum(sum(sz for _, sz, _ in p.queue) for p in self.pois)
        return inside + queued

    def compute_guided_congestion(self):
        # Count how many guided travelers are inside + queued (from our per-agent maps)
        guided_inside = sum(sz for aid, sz in self._agents_inside.items()
                            if self._is_guided_agent(aid))
        guided_queued = sum(sz for aid, sz in self._agents_queued.items()
                            if self._is_guided_agent(aid))
        return guided_inside + guided_queued

    def compute_self_guided_congestion(self):
        self_inside = sum(sz for aid, sz in self._agents_inside.items()
                          if not self._is_guided_agent(aid))
        self_queued = sum(sz for aid, sz in self._agents_queued.items()
                          if not self._is_guided_agent(aid))
        return self_inside + self_queued

    def _is_guided_agent(self, agent_id: int) -> bool:
        for ag in self.guided_groups:
            if ag.unique_id == agent_id:
                return True
        return False

    def compute_guided_completion_rate(self):
        completed_groups = [group for group in self.guided_groups if group.completed]
        return len(completed_groups) / len(self.guided_groups) if len(self.guided_groups) > 0 else 0

    def compute_self_guided_completion_rate(self):
        completed_groups = [group for group in self.self_guided_groups if group.completed]
        return len(completed_groups) / len(self.self_guided_groups) if len(self.self_guided_groups) > 0 else 0

    # ---------- Core step with capacity/queueing enforcement ----------
    def step(self):
        self.schedule_time += 1
        
        # Freeze queued agents so agent.py doesn't "walk away" from the queue
        for gid in list(self._agents_queued.keys()):
            ag = self._agents_by_id.get(gid)
            if ag is None:
                # safety: clear any stale queue entries
                self._agents_queued.pop(gid, None)
                continue
            # Hold them in place for 1 minute; agent.py will decrement to 0 during its step
            ag.traveling = False
            if getattr(ag, "remaining_time", 0) <= 0:
                ag.remaining_time = 1

        # 1) Advance agents (they move & handle own waiting counters)
        for group in self.guided_groups + self.self_guided_groups:
            group.step()

        # 2) After agents moved, detect arrivals, dwell finishing, and apply capacity rules
        for group in self.guided_groups + self.self_guided_groups:
            gid = group.unique_id
            was_trav = self._agent_last_traveling.get(gid, False)
            was_rem = self._agent_last_remaining.get(gid, 0)

            now_trav = bool(group.traveling)
            now_rem = int(group.remaining_time)

            # Determine which POI this group is at (based on its current grid path position)
            # Agent path[current_step] is its current POI coordinate on the grid
            try:
                current_coord = group.path[group.current_step]
            except Exception:
                current_coord = None

            poi_idx = None
            if current_coord is not None:
                poi_idx = self._nearest_poi_index_from_grid(current_coord)

            # a) Just arrived at a POI: traveling True -> False
            if was_trav and not now_trav and not group.completed and poi_idx is not None:
                self._handle_arrival(group, poi_idx)

            # b) If previously dwelling and now finished: remaining_time >0 -> 0 (and not traveling)
            # b) Departure: agent leaves a POI
            leaving_dwell = (was_rem > 0) and (now_rem == 0)
            started_travel_from_poi = (gid in self._agents_inside) and (not was_trav) and now_trav
            if (leaving_dwell or started_travel_from_poi) and (gid in self._agents_inside):
                self._handle_departure(group)

            # c) If queued (remaining_time == 0, not traveling), try to admit again
            if (gid in self._agents_queued) and (now_rem == 0) and (not now_trav) and (poi_idx is not None):
                self._try_admit_from_queue(group, poi_idx)
                
            # d) Final cleanup if the group finished its tour
            if group.completed:
                self._cleanup_completion(group)

            # Update last state trackers
            self._agent_last_traveling[gid] = now_trav
            self._agent_last_remaining[gid] = now_rem

           
        # keep POI queues in sync with model's queued map
        for p in self.pois:
            if p.queue:
                p.queue = deque((aid, sz, guided) for (aid, sz, guided) in p.queue
                                if self._agents_queued.get(aid, 0) == sz)
                    
        # inside_total equals sum of groups we think are inside
        calc_inside = sum(self._agents_inside.values())
        poi_inside = sum(p.inside_total for p in self.pois)
        if calc_inside != poi_inside:
            print(f"[WARN] inside mismatch calc={calc_inside} poi={poi_inside} at t={self.schedule_time}")

        # POI queues should be a subset of model queued map
        poi_q = sum(sz for p in self.pois for (_, sz, _) in p.queue)
        model_q = sum(self._agents_queued.values())
        if poi_q != model_q:
            print(f"[WARN] queue mismatch poi={poi_q} model={model_q} at t={self.schedule_time}")

        # 3) Collect metrics
        self.datacollector.collect(self)

    # ----- capacity/queue helpers -----
    def _handle_arrival(self, group, poi_idx):
        gid = group.unique_id
        p = self.pois[poi_idx]
        admitted = p.try_enter(gid, group.group_size, group.guided)
        self._agent_current_poi_idx[gid] = poi_idx
        self._agents_queued.pop(gid, None)   # clear any previous queue state

        if admitted:
            low, high = (self.guided_wait_time if group.guided else self.self_guided_wait_time)
            group.remaining_time = int(np.random.randint(low, high + 1))
            self._agents_inside[gid] = group.group_size
        else:
            group.remaining_time = 0
            self._agents_queued[gid] = group.group_size

    def _handle_departure(self, group: TouristGroup):
        """When group finishes dwell (remaining_time hits 0), release capacity and clear inside flag."""
        gid = group.unique_id
        poi_idx = self._agent_current_poi_idx.get(gid, None)
        if poi_idx is not None:
            p = self.pois[poi_idx]
            p.release(group.group_size)
        if gid in self._agents_inside:
            del self._agents_inside[gid]
        # After release, the agent will proceed to next POI on its own in the next step.
        
        
    def _cleanup_completion(self, group):
        gid = group.unique_id
        poi_idx = self._agent_current_poi_idx.get(gid)

        # clean current poi (your existing code)
        if poi_idx is not None:
            p = self.pois[poi_idx]
            if gid in self._agents_inside:
                p.release(group.group_size)
            if gid in self._agents_queued or p.in_queue(gid):
                p.remove_from_queue(gid)

        # hard cleanup everywhere (optional but safe)
        for p in self.pois:
            p.remove_from_queue(gid)

        self._agents_inside.pop(gid, None)
        self._agents_queued.pop(gid, None)


    # Model: retry admission
    def _try_admit_from_queue(self, group, poi_idx):
        gid = group.unique_id
        p = self.pois[poi_idx]
        admitted = p.try_enter(gid, group.group_size, group.guided)
        if admitted:
            low, high = (self.guided_wait_time if group.guided else self.self_guided_wait_time)
            group.remaining_time = int(np.random.randint(low, high + 1))
            self._agents_inside[gid] = group.group_size
            self._agents_queued.pop(gid, None)
        else:
            self._agents_inside.pop(gid, None)   # still not inside


# -----------------------------
# (Optional) keep your plotting
# -----------------------------
def plot_state_at_time(model, title="Tourist Distribution at Time Step", timestep=None):
    """Visualizes the full simulation state including POIs, labeled paths, and current group positions."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot POIs and annotate them (projected)
    for idx, (x, y) in enumerate(model.itinerary):
        poi = model.pois[idx]
        ax.scatter(x, y, s=120, c='black', marker='X')
        ax.text(x + 1, y + 1, f'{poi.name}\nCap:{poi.capacity} In:{poi.inside_total} Q:{sum(sz for _, sz, _ in poi.queue)}',
                fontsize=8, color='black')

    # Draw arrows between POIs to show direction (guided path)
    for i in range(len(model.itinerary) - 1):
        x1, y1 = model.itinerary[i]
        x2, y2 = model.itinerary[i + 1]
        ax.annotate("",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='gray', lw=1.5))

    # Helper function to draw a tourist group
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

    # Legend & axes
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
