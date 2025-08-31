# poi.py
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

@dataclass
class POI:
    id: int
    node: int             # node id on the OSM network
    name: str
    capacity: int         # max concurrent visitors inside
    inside: int = 0       # current number inside
    queue: Deque[Tuple[int, int, bool]] = field(default_factory=deque)
    # queue holds (group_id, group_size, is_guided)

    def can_enter(self, group_size: int) -> bool:
        return self.inside + group_size <= self.capacity

    def try_enter(self, group_id: int, group_size: int, is_guided: bool) -> bool:
        if self.can_enter(group_size):
            self.inside += group_size
            return True
        self.queue.append((group_id, group_size, is_guided))
        return False

    def release(self, group_size: int):
        self.inside = max(0, self.inside - group_size)
        # admit queued groups while capacity allows
        while self.queue and self.can_enter(self.queue[0][1]):
            _, sz, _ = self.queue.popleft()
            self.inside += sz
