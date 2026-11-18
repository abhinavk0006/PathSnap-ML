"""Shared helpers: config loading, raw-file IO, projection and point-to-segment geometry."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the repository root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


# --------------------------------------------------------------------------- projection
_TRANSFORMERS: dict[int, Transformer] = {}


def project(lon, lat, epsg: int = 32632) -> np.ndarray:
    """lon/lat (degrees) -> metres in a projected CRS. Returns an (n, 2) array [x, y]."""
    tr = _TRANSFORMERS.get(epsg)
    if tr is None:
        tr = _TRANSFORMERS[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = tr.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.column_stack([x, y])


def utm_epsg(lon, lat) -> int:
    """EPSG code of the UTM zone containing the centroid of (lon, lat).

    The dataset is global (trips fall in UTM zones 14-53), so one fixed projected CRS would badly distort
    distances for most trips. Each trip is small (< ~1 degree), so its own UTM zone keeps scale error ~0.1%.
    """
    lo, la = float(np.median(lon)), float(np.median(lat))
    zone = int((lo + 180.0) // 6.0) % 60 + 1
    return (32600 if la >= 0 else 32700) + zone


def seg_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Distance from points P to segments A-B (all (n, 2)). Returns (distance, fraction along segment in [0, 1])."""
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    t = np.clip(np.einsum("ij,ij->i", P - A, AB) / np.maximum(L2, 1e-9), 0.0, 1.0)
    C = A + t[:, None] * AB
    return np.linalg.norm(P - C, axis=1), t


# --------------------------------------------------------------------------- raw IO
def _read(path: Path, cols: int, dtype) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype)
    return df.to_numpy()


@dataclass
class Trip:
    trip_id: int
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    nodes_ll: np.ndarray            # (n_nodes, 2) lon, lat
    arcs: np.ndarray                # (n_arcs, 2) from_node, to_node ; row index = arc_id
    route: np.ndarray | None = None  # reference arc ids (only loaded when explicitly requested)
    extra: dict = field(default_factory=dict)


def trip_paths(trip_dir: str | os.PathLike) -> dict[str, Path]:
    d = Path(trip_dir)
    tid = d.name
    return {ext: d / f"{tid}.{ext}" for ext in ("track", "nodes", "arcs", "route")}


def read_trip(trip_dir: str | os.PathLike, with_route: bool = False) -> Trip:
    """Load one trip. The reference route is NOT read unless with_route=True, so the prediction
    pipeline can never touch labels by accident."""
    p = trip_paths(trip_dir)
    track = _read(p["track"], 3, float)
    nodes = _read(p["nodes"], 2, float)
    arcs = _read(p["arcs"], 2, np.int64)
    route = None
    if with_route:
        route = pd.read_csv(p["route"], header=None, dtype=np.int64)[0].to_numpy()
    return Trip(int(Path(trip_dir).name), track[:, 0], track[:, 1], track[:, 2], nodes, arcs, route)


def read_reference(trip_dir: str | os.PathLike) -> np.ndarray:
    """Reference route only. Used by evaluation code, never by the prediction pipeline."""
    p = trip_paths(trip_dir)
    return pd.read_csv(p["route"], header=None, dtype=np.int64)[0].to_numpy()


def list_trips(split_dir: str | os.PathLike) -> list[Path]:
    d = Path(split_dir)
    return sorted(x for x in d.iterdir() if x.is_dir())


# --------------------------------------------------------------------------- road network
