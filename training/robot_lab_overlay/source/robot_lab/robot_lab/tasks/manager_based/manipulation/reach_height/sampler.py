"""T2 target sampler (protocol v0.4 §4.1, §4.3 step 4, §8). CPU reference implementation.

Sampling set (v0.4) = per band, the largest connected (6-neighbour) region of its reachable voxels;
isolated voxels outside it are dropped.
Band -> voxel of that band's sampling set (uniform; voxels have equal volume) -> uniform point inside the
voxel, re-drawn until its azimuth lies inside the ±60° sector -> Mode A canonical orientation (from azimuth and z).
Positions are in the episode ground frame G; band edges coincide with voxel faces, so z stays in band.
"""

import hashlib
import json

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from . import protocol as P
from .wholebody import canonical_rotation


class ReachableSet:
    """Sampling set from a reachability_T2*.npz file (or the equivalent dict of arrays).

    `reachable` / `volume` are restricted to the largest connected region of each band (v0.4);
    `reachable_map` keeps the unrestricted map and `dropped` counts the removed voxels per band."""

    def __init__(self, data):
        self.resolution = float(data["resolution"])
        self.axes = [np.asarray(data[k], dtype=float) for k in ("grid_x", "grid_y", "grid_z")]
        self.volume = np.asarray(data["reachable_volume"], dtype=bool)
        self.centres = np.asarray(data["voxel_centre_G"], dtype=float)
        self.band = np.asarray(data["voxel_band"], dtype=int)
        self.reachable_map = np.asarray(data["reachable"], dtype=bool)
        ijk = np.rint((self.centres - np.array([a[0] for a in self.axes])) / self.resolution).astype(int)
        self.reachable = np.zeros_like(self.reachable_map)
        self.dropped = []
        for b in range(len(P.BANDS)):
            hit = self.reachable_map & (self.band == b)
            vol = np.zeros(self.volume.shape, dtype=bool)
            vol[tuple(ijk[hit].T)] = True
            labels, n = ndimage.label(vol)
            if n:
                largest = 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
                self.reachable |= hit & (labels[tuple(ijk.T)] == largest)
            self.dropped.append(int(hit.sum() - (self.reachable & (self.band == b)).sum()))
        self.volume = np.zeros(self.volume.shape, dtype=bool)
        self.volume[tuple(ijk[self.reachable].T)] = True
        for _, lo, hi in P.BANDS:
            for edge in (lo, hi):
                if abs(edge / self.resolution - round(edge / self.resolution)) > 1e-6:
                    raise ValueError("Band edges must coincide with voxel faces")
        self.by_band = [np.flatnonzero(self.reachable & (self.band == i)) for i in range(len(P.BANDS))]
        if not all(len(v) for v in self.by_band):
            raise ValueError("Every band needs at least one reachable voxel")

    @classmethod
    def load(cls, path, require_full_grid=True):
        """Load a reachability npz. Refuses a reachability version other than REACHABILITY["version"] and,
        if require_full_grid, any grid other than the protocol grid (e.g. a coarse --resolution run)."""
        with np.load(path, allow_pickle=False) as d:
            data = {k: d[k] for k in d.files}
        check_reachability_metadata(data, require_full_grid)
        return cls(data)

    def contains(self, positions):
        """True where a G-frame position lies in a reachable voxel."""
        p = np.atleast_2d(np.asarray(positions, dtype=float))
        ijk = np.floor((p - (np.array([a[0] for a in self.axes]) - self.resolution / 2)) / self.resolution).astype(int)
        inside = np.all((ijk >= 0) & (ijk < np.array(self.volume.shape)), axis=1)
        out = np.zeros(len(p), dtype=bool)
        out[inside] = self.volume[tuple(ijk[inside].T)]
        return out


def check_reachability_metadata(data, require_full_grid=True):
    """Raise ValueError unless the npz carries this protocol's reachability version (and the full grid)."""
    if "metadata" not in data:
        raise ValueError("Reachability file has no metadata; refusing to use it")
    meta = json.loads(str(data["metadata"]))
    found = meta.get("reachability", {}).get("version")
    if found != P.REACHABILITY["version"]:
        raise ValueError(f"Reachability version {found!r} != required {P.REACHABILITY['version']!r}")
    if require_full_grid:
        spec = P.REACHABILITY
        res = spec["grid_resolution_m"]   # same axes as reachability.voxel_grid
        expected = [np.arange(lo + res / 2, hi - 1e-9, res) for lo, hi in (spec["grid_x_m"], spec["grid_y_m"], spec["grid_z_m"])]
        same = (not meta.get("grid_overridden", True)
                and abs(float(data["resolution"]) - spec["grid_resolution_m"]) < 1e-12
                and all(len(e) == len(data[k]) and np.allclose(e, data[k], atol=1e-9)
                        for e, k in zip(expected, ("grid_x", "grid_y", "grid_z"))))
        if not same:
            raise ValueError("Reachability file is not a full protocol-grid map (coarse or overridden grid)")
    return meta


def _points_in_voxels(rs, voxels, rng):
    centres = rs.centres[voxels]
    out = centres + rng.uniform(-0.5, 0.5, size=centres.shape) * rs.resolution
    bad = np.abs(np.arctan2(out[:, 1], out[:, 0])) > P.AZIMUTH_MAX_RAD
    while bad.any():
        out[bad] = centres[bad] + rng.uniform(-0.5, 0.5, size=(bad.sum(), 3)) * rs.resolution
        bad = np.abs(np.arctan2(out[:, 1], out[:, 0])) > P.AZIMUTH_MAX_RAD
    return out


def _targets(rs, bands, rng):
    voxels = np.empty(len(bands), dtype=int)
    for b in range(len(P.BANDS)):
        sel = bands == b
        voxels[sel] = rng.choice(rs.by_band[b], size=int(sel.sum()))
    positions = _points_in_voxels(rs, voxels, rng)
    theta = np.arctan2(positions[:, 1], positions[:, 0])
    quat = Rotation.from_matrix(canonical_rotation(theta, positions[:, 2])).as_quat(canonical=True)[:, [3, 0, 1, 2]]
    return dict(band=bands, voxel=voxels, position_G=positions, quat_G_wxyz=quat, azimuth_rad=theta)


def sample(rs, n, rng, band_weights=(1.0, 1.0, 1.0)):
    """Training-style sampling: band ~ band_weights (curriculum hook, §9), then uniform in its reachable set."""
    w = np.asarray(band_weights, dtype=float)
    return _targets(rs, rng.choice(len(P.BANDS), size=n, p=w / w.sum()), rng)


def statistics(targets, rs, band_weights=(1.0, 1.0, 1.0)):
    """Acceptance test A2: band frequencies vs. configuration and reachable-set membership."""
    w = np.asarray(band_weights, dtype=float) / np.sum(band_weights)
    freq = np.bincount(targets["band"], minlength=len(P.BANDS)) / len(targets["band"])
    z_band = P.band_index(targets["position_G"][:, 2])
    return dict(n=int(len(targets["band"])), frequency=freq.tolist(), expected=w.tolist(),
                max_frequency_deviation=float(np.abs(freq - w).max()),
                all_in_reachable_set=bool(rs.contains(targets["position_G"]).all()),
                all_in_band=bool((z_band == targets["band"]).all()),
                max_abs_azimuth_rad=float(np.abs(targets["azimuth_rad"]).max()))


def digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        h.update(np.ascontiguousarray(arrays[key]).tobytes())
    return h.hexdigest()


def eval_target_set(rs, seed, reachability_sha256, n_per_band=P.N_EVAL_PER_BAND):
    """Fixed evaluation set (§8): exactly n_per_band targets per band, drawn from `seed`."""
    rng = np.random.default_rng(seed)
    bands = np.repeat(np.arange(len(P.BANDS)), n_per_band)
    arrays = _targets(rs, bands, rng)
    arrays["target_id"] = np.arange(len(bands))
    meta = dict(benchmark=P.BENCHMARK, protocol_version=P.VERSION, seed=int(seed), n_per_band=int(n_per_band),
                reachability_sha256=reachability_sha256, frame="G", tcp_parent=P.TCP_PARENT,
                tcp_offset_m=list(P.TCP_OFFSET_M), orientation_mode=P.ORIENTATION_MODE,
                approach_profile=[list(p) for p in P.APPROACH_PROFILE])
    meta["targets_sha256"] = digest(arrays)
    return arrays, meta


def save_eval_set(path, arrays, meta):
    np.savez_compressed(path, metadata=np.array(json.dumps(meta, sort_keys=True)), **arrays)


def load_eval_set(path, rs=None):
    """Load and verify an evaluation target set (digest, version, counts and, if given, membership)."""
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files if k != "metadata"}
        meta = json.loads(str(d["metadata"]))
    if meta.get("protocol_version") != P.VERSION or digest(arrays) != meta.get("targets_sha256"):
        raise ValueError("Evaluation target set changed or belongs to another protocol version")
    if np.bincount(arrays["band"], minlength=len(P.BANDS)).tolist() != [meta["n_per_band"]] * len(P.BANDS):
        raise ValueError("Evaluation target counts differ from the metadata")
    if rs is not None and not rs.contains(arrays["position_G"]).all():
        raise ValueError("Evaluation targets outside the reachable set")
    return arrays, meta
