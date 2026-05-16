# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
# LCM metrics I/O — accelerated binary read/write for training metrics.
#
# Build with: python setup.py build_ext --inplace
# Pure-Python fallback in train/monitor.py

import struct
import os

# ---------------------------------------------------------------------------
# Binary format:
#   Magic   "LCM_M"     5 B
#   Version  uint32      4 B  (currently 1)
#   Window   uint32      4 B
#   N_metrics uint32     4 B
#   For each metric:
#     name_len uint16    2 B
#     name     char[]    name_len B
#     n_points uint32    4 B
#     steps[]  uint64[]  n_points × 8 B
#     values[] float64[] n_points × 8 B
# ---------------------------------------------------------------------------

_MAGIC = b"LCM_M"
_VERSION = 1


def save_metrics_cy(path, window, metrics):
    """Write metrics dict to binary file.

    Args:
        path: Output file path.
        window: Recording interval (stored in header).
        metrics: dict[str, list[(int, float)]] — step-value pairs.
    """
    names = list(metrics.keys())
    n_metrics = len(names)

    # First pass: compute total size
    # Layout: header + table-of-contents then all point data
    toc_entries = []
    offset = 5 + 4 + 4 + 4  # magic + version + window + n_metrics
    for name in names:
        pts = metrics[name]
        n = len(pts)
        name_enc = name.encode("utf-8")
        name_len = len(name_enc)
        entry_size = 2 + name_len + 4 + n * 16  # 8 step + 8 value
        toc_entries.append((name_enc, name_len, n, entry_size))
        offset += entry_size

    buf = bytearray(offset)
    view = memoryview(buf)

    # Write header
    pos = 0
    view[pos:pos + 5] = _MAGIC; pos += 5
    struct.pack_into("<I", buf, pos, _VERSION); pos += 4
    struct.pack_into("<I", buf, pos, window); pos += 4
    struct.pack_into("<I", buf, pos, n_metrics); pos += 4

    for i, name in enumerate(names):
        name_enc, name_len, n, _ = toc_entries[i]
        pts = metrics[name]

        # name length + name
        struct.pack_into("<H", buf, pos, name_len); pos += 2
        view[pos:pos + name_len] = name_enc; pos += name_len
        # num points
        struct.pack_into("<I", buf, pos, n); pos += 4
        # steps and values interleaved as (uint64, float64) pairs
        for j, (step, val) in enumerate(pts):
            struct.pack_into("<Qd", buf, pos, step, val)
            pos += 16

    # Write atomically
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_metrics_cy(path):
    """Read metrics from binary file.

    Returns:
        (window, metrics_dict) or raises on error.
    """
    with open(path, "rb") as f:
        buf = f.read()

    pos = 0
    magic = buf[pos:pos + 5]; pos += 5
    if magic != _MAGIC:
        raise ValueError(f"Bad magic: {magic!r}")

    ver = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    if ver != _VERSION:
        raise ValueError(f"Unsupported version: {ver}")

    window = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    n_metrics = struct.unpack_from("<I", buf, pos)[0]; pos += 4

    result = {}
    for _ in range(n_metrics):
        name_len = struct.unpack_from("<H", buf, pos)[0]; pos += 2
        name = buf[pos:pos + name_len].decode("utf-8"); pos += name_len
        n = struct.unpack_from("<I", buf, pos)[0]; pos += 4
        pts = []
        for _ in range(n):
            step, val = struct.unpack_from("<Qd", buf, pos)
            pos += 16
            pts.append((step, val))
        result[name] = pts

    return window, result
