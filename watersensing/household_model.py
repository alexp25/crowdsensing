from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import random
import numpy as np


@dataclass
class HouseholdProfile:
    name: str
    share: float
    dwell_range: Tuple[int, int]
    flow_lpm_per_household: float
    start_window: Tuple[int, int]
    start_interval: int
    group_size_range: Tuple[int, int]
    start_windows: Optional[List[Tuple[int, int]]] = None


@dataclass
class HouseholdGroup:
    gid: int
    profile_name: str
    size: int
    start_time: int
    service_zone_index: int = 0
    in_service_zone: Optional[int] = None
    in_queue: bool = False
    in_transfer: bool = False
    transfer_time: int = 0
    remaining_dwell: int = 0
    completed: bool = False


@dataclass
class ServiceZone:
    service_zone_id: int
    x: int
    y: int
    capacity: int
    queue: Deque[int]
    inside: Dict[int, int]


class HouseholdWaterNetworkModel:
    """Agent-based household demand model with service-zone capacity and FIFO queues."""

    def __init__(
        self,
        total_households: int,
        profiles: List[HouseholdProfile],
        day_minutes: int = 24 * 60,
        num_service_zones: int = 8,
        grid_size: Tuple[int, int] = (100, 100),
        service_zone_capacity_range: Tuple[int, int] = (30, 65),
        seed: int = 42,
        validate_state: bool = False,
    ):
        self.total_households = total_households
        self.profiles = profiles
        self.day_minutes = day_minutes
        self.num_service_zones = num_service_zones
        self.width, self.height = grid_size
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.validate_state = validate_state

        self.time = 0
        self.groups: Dict[int, HouseholdGroup] = {}
        self.profile_by_name = {p.name: p for p in profiles}
        self.service_zones = self._build_service_zones(service_zone_capacity_range)
        self.metrics = {
            "active_demand": [],
            "queued_households": [],
            "consumption_lpm": [],
            "max_service_zone_load": [],
            "pressure_index": [],
            "pressure_deficit": [],
            "min_service_zone_pressure": [],
        }
        self.profile_active: Dict[str, List[int]] = {p.name: [] for p in profiles}

        self._create_groups()

    def _build_service_zones(self, service_zone_capacity_range: Tuple[int, int]) -> List[ServiceZone]:
        service_zones: List[ServiceZone] = []
        for i in range(self.num_service_zones):
            service_zones.append(
                ServiceZone(
                    service_zone_id=i,
                    x=self.rng.randint(1, self.width - 2),
                    y=self.rng.randint(1, self.height - 2),
                    capacity=self.rng.randint(*service_zone_capacity_range),
                    queue=deque(),
                    inside={},
                )
            )
        return service_zones

    def _create_groups(self):
        gid = 0
        allocated = 0
        for i, profile in enumerate(self.profiles):
            if i == len(self.profiles) - 1:
                profile_households = self.total_households - allocated
            else:
                profile_households = int(round(self.total_households * profile.share))
                allocated += profile_households

            remaining = max(0, profile_households)
            while remaining > 0:
                lo, hi = profile.group_size_range
                size = min(self.rng.randint(lo, hi), remaining)
                start = self._sample_start_time(profile)

                self.groups[gid] = HouseholdGroup(
                    gid=gid,
                    profile_name=profile.name,
                    size=size,
                    start_time=start,
                )
                gid += 1
                remaining -= size

    def _sample_start_time(self, profile: HouseholdProfile) -> int:
        windows = profile.start_windows if profile.start_windows else [profile.start_window]
        valid = [(lo, hi) for (lo, hi) in windows if hi > lo]
        if not valid:
            valid = [(profile.start_window[0], max(profile.start_window[0] + 1, profile.start_window[1]))]

        lengths = [hi - lo for (lo, hi) in valid]
        total = sum(lengths)
        pick = self.rng.uniform(0, total)
        acc = 0.0
        selected = valid[-1]
        for (w, length) in zip(valid, lengths):
            acc += length
            if pick <= acc:
                selected = w
                break

        s_lo, s_hi = selected
        start = self.rng.randint(s_lo, s_hi - 1)
        if profile.start_interval > 1:
            start -= start % profile.start_interval
        return start

    def _service_zone_transfer_delay_minutes(self, from_idx: int, to_idx: int) -> int:
        z1 = self.service_zones[from_idx]
        z2 = self.service_zones[to_idx]
        distance = ((z1.x - z2.x) ** 2 + (z1.y - z2.y) ** 2) ** 0.5
        return max(1, int(distance / 8.0))

    def _admit_to_service_zone(self, group: HouseholdGroup, service_zone_idx: int):
        service_zone = self.service_zones[service_zone_idx]
        used = sum(self.groups[gid].size for gid in service_zone.inside.keys())
        if used + group.size <= service_zone.capacity:
            profile = self.profile_by_name[group.profile_name]
            low, high = profile.dwell_range
            dwell = self.rng.randint(low, max(low, high))
            group.in_service_zone = service_zone_idx
            group.in_queue = False
            group.remaining_dwell = max(1, dwell)
            service_zone.inside[group.gid] = group.size
        else:
            if group.gid not in service_zone.queue:
                service_zone.queue.append(group.gid)
            group.in_queue = True
            group.in_service_zone = service_zone_idx
            group.remaining_dwell = 0

    def _release_from_service_zone(self, group: HouseholdGroup):
        if group.in_service_zone is None:
            return
        service_zone = self.service_zones[group.in_service_zone]
        service_zone.inside.pop(group.gid, None)
        group.in_service_zone = None

    def _process_service_zone_queue(self, service_zone: ServiceZone):
        while service_zone.queue:
            gid = service_zone.queue[0]
            group = self.groups[gid]
            used = sum(self.groups[x].size for x in service_zone.inside.keys())
            if used + group.size <= service_zone.capacity:
                service_zone.queue.popleft()
                profile = self.profile_by_name[group.profile_name]
                low, high = profile.dwell_range
                group.in_queue = False
                group.remaining_dwell = max(1, self.rng.randint(low, max(low, high)))
                service_zone.inside[gid] = group.size
            else:
                break

    def step(self):
        self.time += 1

        for group in self.groups.values():
            if group.completed or self.time < group.start_time:
                continue

            if group.in_transfer:
                group.transfer_time -= 1
                if group.transfer_time <= 0:
                    group.in_transfer = False
                    if group.service_zone_index >= self.num_service_zones:
                        group.completed = True
                    else:
                        self._admit_to_service_zone(group, group.service_zone_index)
                continue

            if group.in_queue:
                continue

            if group.in_service_zone is None:
                if group.service_zone_index >= self.num_service_zones:
                    group.completed = True
                else:
                    self._admit_to_service_zone(group, group.service_zone_index)
                continue

            if group.remaining_dwell > 0:
                group.remaining_dwell -= 1
                if group.remaining_dwell > 0:
                    continue

                self._release_from_service_zone(group)
                group.service_zone_index += 1
                if group.service_zone_index >= self.num_service_zones:
                    group.completed = True
                else:
                    group.in_transfer = True
                    group.transfer_time = self._service_zone_transfer_delay_minutes(
                        group.service_zone_index - 1, group.service_zone_index
                    )

        for service_zone in self.service_zones:
            self._process_service_zone_queue(service_zone)

        if self.validate_state:
            self._validate_state()

        self._collect_metrics()

    def _validate_state(self):
        for service_zone in self.service_zones:
            inside_households = sum(self.groups[gid].size for gid in service_zone.inside.keys())
            if inside_households > service_zone.capacity:
                raise RuntimeError(
                    f"Service zone {service_zone.service_zone_id} capacity exceeded: "
                    f"{inside_households} > {service_zone.capacity}"
                )

            if len(service_zone.queue) != len(set(service_zone.queue)):
                raise RuntimeError(f"Duplicate group in queue at service zone {service_zone.service_zone_id}")

            for gid in service_zone.inside.keys():
                group = self.groups[gid]
                if group.in_service_zone != service_zone.service_zone_id:
                    raise RuntimeError(
                        f"Group {gid} inside service zone {service_zone.service_zone_id} "
                        f"but in_service_zone={group.in_service_zone}"
                    )
                if group.in_queue:
                    raise RuntimeError(f"Group {gid} marked both inside and queued")

            for gid in service_zone.queue:
                group = self.groups[gid]
                if group.in_service_zone != service_zone.service_zone_id:
                    raise RuntimeError(
                        f"Queued group {gid} at service zone {service_zone.service_zone_id} "
                        f"but in_service_zone={group.in_service_zone}"
                    )
                if not group.in_queue:
                    raise RuntimeError(f"Group {gid} in service zone queue but in_queue=False")

    def _collect_metrics(self):
        active_demand = 0
        queued_households = 0
        consumption_lpm = 0.0
        profile_active_count = {k: 0 for k in self.profile_active.keys()}

        for group in self.groups.values():
            if group.completed or self.time < group.start_time:
                continue
            if group.in_service_zone is not None or group.in_queue:
                active_demand += group.size
                profile_active_count[group.profile_name] += group.size
            if group.in_queue:
                queued_households += group.size
            if group.in_service_zone is not None and not group.in_queue:
                profile = self.profile_by_name[group.profile_name]
                consumption_lpm += group.size * profile.flow_lpm_per_household

        service_zone_loads = []
        service_zone_pressures = []
        for service_zone in self.service_zones:
            inside = sum(self.groups[gid].size for gid in service_zone.inside.keys())
            queued = sum(self.groups[gid].size for gid in service_zone.queue)
            total_demand_units = inside + queued
            service_zone_loads.append(total_demand_units)
            pressure = min(1.0, service_zone.capacity / total_demand_units) if total_demand_units > 0 else 1.0
            service_zone_pressures.append(pressure)

        pressure_index = float(np.mean(service_zone_pressures)) if service_zone_pressures else 1.0
        pressure_deficit = 1.0 - pressure_index
        min_service_zone_pressure = float(np.min(service_zone_pressures)) if service_zone_pressures else 1.0

        self.metrics["active_demand"].append(active_demand)
        self.metrics["queued_households"].append(queued_households)
        self.metrics["consumption_lpm"].append(consumption_lpm)
        self.metrics["max_service_zone_load"].append(max(service_zone_loads) if service_zone_loads else 0)
        self.metrics["pressure_index"].append(pressure_index)
        self.metrics["pressure_deficit"].append(pressure_deficit)
        self.metrics["min_service_zone_pressure"].append(min_service_zone_pressure)

        for name, val in profile_active_count.items():
            self.profile_active[name].append(val)

    def run(self) -> Dict[str, np.ndarray]:
        for _ in range(self.day_minutes):
            self.step()
        out = {k: np.array(v) for k, v in self.metrics.items()}
        for k, v in self.profile_active.items():
            out[f"profile_active::{k}"] = np.array(v)
        return out
