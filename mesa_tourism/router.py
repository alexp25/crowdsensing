# router.py
from typing import Tuple
import os
import networkx as nx

try:
    import osmnx as ox
except Exception:
    ox = None


class Router:
    """Wrapper around a NetworkX MultiDiGraph with OSMnx helpers."""

    def __init__(self, G: nx.MultiDiGraph):
        self.G = G

    @classmethod
    def from_graphml(cls, path: str):
        """Load a router from a saved GraphML file."""
        G = nx.read_graphml(path)

        # GraphML sometimes stores coordinates as strings → convert to float
        for _, data in G.nodes(data=True):
            for key in ("x", "y", "lon", "lat"):
                if key in data:
                    try:
                        data[key] = float(data[key])
                    except Exception:
                        pass
        return cls(G)

    @classmethod
    def build_and_cache(
        cls,
        graphml_path: str,
        *,
        center_latlon: Tuple[float, float] = (52.3728, 4.8936),  # Amsterdam center
        dist_m: int = 1500,
        network_type: str = "walk"
    ):
        """Build a small walk network around a lat/lon point and cache it to GraphML."""
        if ox is None:
            raise RuntimeError("osmnx not installed. Run: pip install osmnx")

        ox.settings.use_cache = True
        ox.settings.log_console = False

        G = ox.graph_from_point(center_latlon, dist=dist_m, network_type=network_type, simplify=True)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)

        os.makedirs(os.path.dirname(graphml_path) or ".", exist_ok=True)
        ox.save_graphml(G, graphml_path)
        return cls(G)

    def node_xy(self, node):
        """Return (x,y) for a node; tolerant to str/int key mismatches."""
        G = self.G
        # normalize key
        if node in G:
            key = node
        else:
            s = str(node)
            if s in G:
                key = s
            else:
                # final attempt: int-like string
                try:
                    i = int(node)
                except Exception:
                    i = None
                if i is not None and i in G:
                    key = i
                else:
                    raise KeyError(f"Node {node!r} not found in graph (tried {node!r}, {s!r}, {i!r}).")

        d = G.nodes[key]
        # OSMnx uses 'x' (lon) and 'y' (lat)
        if "x" in d and "y" in d:
            return float(d["x"]), float(d["y"])
        if "lon" in d and "lat" in d:
            return float(d["lon"]), float(d["lat"])
        if "pos" in d and isinstance(d["pos"], (tuple, list)) and len(d["pos"]) == 2:
            x, y = d["pos"]
            return float(x), float(y)
        raise KeyError(f"Node {key!r} has no coordinate attributes (x/y, lon/lat, or pos).")


def get_router(
    graphml_path: str,
    center_latlon: Tuple[float, float] = (52.3728, 4.8936),
    dist_m: int = 1500
) -> Router:
    """
    Load a cached Router from GraphML if present;
    otherwise build from OSM and cache it.
    """
    if os.path.exists(graphml_path):
        return Router.from_graphml(graphml_path)
    return Router.build_and_cache(graphml_path, center_latlon=center_latlon, dist_m=dist_m)
