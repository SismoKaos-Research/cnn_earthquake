"""Cached iasp91 travel times, shared instead of re-implemented per script.

Three scripts each grew their own copy of the same thing: a `TauPyModel`, a dict
keyed on a rounded (distance, depth) grid, and a try/except returning None. They
differed only in grid resolution and in whether the phase list was a parameter,
which is the shape duplication takes when it is copied rather than imported.

**The grid is the point, not an optimisation detail.** A catalogue run asks for
tens of thousands of arrivals, and an uncached `get_travel_times` call is
milliseconds; rounding distance and depth to a few kilometres collapses that to a
few hundred distinct calls. Travel time varies slowly enough over a 5 km cell
that the error is far below the location error already present in the catalogue
(median RMS residual 0.42 s for AFAD), so the approximation is free in practice.

**`grid_km` is explicit and not defaulted quietly**, because changing it changes
every arrival a caller computes. The callers here keep the resolution they were
written with -- 5 km for the detection and station-range work, 10 km for the FDSN
planner -- so consolidating did not move any published number.

    from sismokaos.arrivals import ArrivalTimes, P_PHASES, S_PHASES
    taup = ArrivalTimes(grid_km=5.0)
    tp = taup.travel(dist_km, depth_km)            # P, or None
    ts = taup.travel(dist_km, depth_km, S_PHASES)  # S
"""
from obspy.taup import TauPyModel

# Phase names in the order TauP should try them. Both lists are the local/
# regional set: `p`/`s` are the upgoing branches that matter for shallow events
# at short distance, and omitting them loses arrivals inside ~100 km.
P_PHASES = ("p", "P", "Pn", "Pg")
S_PHASES = ("s", "S", "Sn", "Sg")

DEG_PER_KM = 1.0 / 111.195


class ArrivalTimes:
    """Travel times on a rounded (distance, depth) grid, cached per instance."""

    def __init__(self, model="iasp91", grid_km=5.0, grid_depth_km=None):
        """Builds the velocity model and an empty cache.

        Args:
            model: TauP model name.
            grid_km: Distance rounding, in km. Larger is faster and coarser;
                see the module docstring for why this is not defaulted silently.
            grid_depth_km: Depth rounding; defaults to `grid_km`.
        """
        self.model = TauPyModel(model=model)
        self.grid_km = float(grid_km)
        self.grid_depth_km = float(grid_depth_km if grid_depth_km is not None else grid_km)
        self._cache = {}
        self.calls = 0
        self.hits = 0

    def travel(self, dist_km, depth_km, phases=P_PHASES):
        """First arrival time in seconds, or None if TauP finds no phase.

        None is returned rather than raised: at some (distance, depth) pairs the
        model genuinely has no arrival for the requested phases, and a caller
        looping over a catalogue wants to skip those rather than stop.
        """
        gk, gd = self.grid_km, self.grid_depth_km
        key = (round(dist_km / gk), round(max(depth_km, 0.0) / gd), tuple(phases))
        self.calls += 1
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        try:
            arr = self.model.get_travel_times(
                source_depth_in_km=key[1] * gd,
                distance_in_degree=key[0] * gk * DEG_PER_KM,
                phase_list=list(phases))
            self._cache[key] = arr[0].time if arr else None
        except Exception:
            self._cache[key] = None
        return self._cache[key]

    def stats(self):
        """Cache hit rate, for confirming the grid is actually doing work."""
        return {"calls": self.calls, "hits": self.hits,
                "distinct": len(self._cache),
                "hit_rate": self.hits / self.calls if self.calls else 0.0}
