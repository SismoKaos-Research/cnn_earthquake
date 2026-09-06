"""Reading one archive chunk, and conditioning its samples.

The preprocessing here is not a reimplementation-by-eye of the training
pipeline: it is the same order of operations `seismic_cli.core` applies when it
builds a training window (detrend twice, 5% Hann taper, 4th-order bandpass,
resample, standardize), applied to a whole block of windows at once instead of
one at a time. `verify` checks that claim against real dataset tensors.
"""
import pathlib
import tempfile
import zipfile

import numpy as np
from obspy import read
from scipy import signal

# Component roles, in the order the training encoder stacks them. Taking the
# first three channels alphabetically would grab ['1','2','E'] at a station with
# mixed sensor codes -- two horizontals and no vertical. See core._COMPONENT_ROLES.
COMPONENT_ROLES = (("Z",), ("N", "1"), ("E", "2"))


def add_chunk_args(q):
    """The flags that say which archive to read and how to condition it.

    Shared by `baseline` and `scan` because they must agree: a baseline built
    at one passband is not the standardization a scan at another passband
    needs, and nothing downstream would notice the mismatch.
    """
    q.add_argument("--zips", required=True, help="glob of chunk archives")
    q.add_argument("--fs", type=float, default=100.0)
    q.add_argument("--freqmin", type=float, default=1.0)
    q.add_argument("--freqmax", type=float, default=45.0)


def read_chunk(zpath):
    """Reads one campaign archive into a Stream, merged and gap-split.

    `merge(fill_value=None)` leaves masked arrays where data is missing and
    `split()` turns those back into contiguous unmasked segments. Keeping the
    segmentation is the whole point: a window that would straddle a gap simply
    never gets built, rather than being built across interpolated samples.
    """
    with zipfile.ZipFile(zpath) as zf, tempfile.TemporaryDirectory() as tmp:
        member = next(n for n in zf.namelist() if n.lower().endswith(".mseed"))
        zf.extract(member, tmp)
        st = read(str(pathlib.Path(tmp) / member))
    st.merge(method=1, fill_value=None)
    return st.split()


def pick_components(stream):
    """Returns the three channel codes to use, in Z/N/E role order."""
    have = {tr.stats.channel[-1].upper() for tr in stream}
    out = []
    for role in COMPONENT_ROLES:
        match = next((c for c in role if c in have), None)
        if match is None:
            return None
        out.append(match)
    return out


def component_segments(stream, comp, fs):
    """Contiguous (t0, data) segments for one component, at the nominal rate."""
    segs = []
    for tr in stream:
        if tr.stats.channel[-1].upper() != comp:
            continue
        if abs(tr.stats.sampling_rate - fs) > 1e-6:
            tr = tr.copy()
            tr.resample(fs)
        segs.append((tr.stats.starttime.timestamp, np.asarray(tr.data, dtype=np.float64)))
    segs.sort(key=lambda s: s[0])
    return segs


def make_windows(data, offset, n_windows, win, step):
    """A (n_windows, win) view over `data`, without copying."""
    sub = data[offset:offset + (n_windows - 1) * step + win]
    return np.lib.stride_tricks.as_strided(
        sub, shape=(n_windows, win),
        strides=(sub.strides[0] * step, sub.strides[0]), writeable=False)


def clean_block(x, fs, freqmin, freqmax, taper):
    """`core.clean_and_filter_1d`, applied to a (n, win) block at once.

    Identical operations in identical order; the only difference is that scipy
    is given an axis instead of being called once per window.
    """
    x = signal.detrend(x, type="linear", axis=-1)
    x = signal.detrend(x, type="constant", axis=-1)
    x = x * taper
    nyquist = fs / 2.0
    actual_freqmax = freqmax if nyquist > freqmax else nyquist - 1.0
    if actual_freqmax > freqmin:
        b, a = signal.butter(4, [freqmin, actual_freqmax], btype="bandpass", fs=fs)
        x = signal.filtfilt(b, a, x, axis=-1)
    return x


def taper_vector(n):
    """The 5% Hann cosine taper `clean_and_filter_1d` applies to each end."""
    t = np.ones(n)
    k = int(n * 0.05)
    if k > 0:
        w = signal.windows.hann(k * 2)
        t[:k] = w[:k]
        t[-k:] = w[k:]
    return t
