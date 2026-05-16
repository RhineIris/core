"""Training metrics recorder and HTML chart generator.

Two roles:
  1. Record data during training  (``record`` → binary/JSON)
  2. Generate HTML chart offline  (``python lcm.py chart`` → HTML)

Cython acceleration:
  - ``train/_metrics_cy.pyx`` provides fast binary I/O
  - Pure-Python JSON fallback when .so is not built

Usage — training:
    recorder = MetricsRecorder(save_dir="checkpoints")
    for step in range(steps):
        if step % 50 == 0:
            recorder.record(step, loss=loss_val, lr=lr_val)
    recorder.save()

Usage — chart:
    python lcm.py chart --input checkpoints/metrics.bin --output chart.html
"""

import json
import os

# Cython-accelerated binary I/O (fallback = pure Python)
_HAS_CY = False
try:
    from train._metrics_cy import save_metrics_cy, load_metrics_cy
    _HAS_CY = True
except ImportError:
    pass


# File extension used by save/load.
# Binary when Cython is available, JSON otherwise.
def _ext():
    return ".bin" if _HAS_CY else ".json"


# ── core recorder ──────────────────────────────────────────────────────

class MetricsRecorder:
    """Lightweight metrics recorder.

    Records one data point per call (no buffering / averaging).
    Saves/loads JSON for resume support.
    """

    def __init__(self, save_dir=".", window=50):
        self.save_dir = save_dir
        self.window = window
        self.metrics = {}  # name -> [(step, value), ...]
        os.makedirs(save_dir, exist_ok=True)
        self._load()

    def record(self, step, **kwargs):
        """Record metric values at *step*.

        Caller is responsible for calling only at desired intervals
        (e.g. every 50 steps).  Each call stores one data point.
        """
        for name, value in kwargs.items():
            self.metrics.setdefault(name, []).append((step, float(value)))

    # ── persist ─────────────────────────────────────────────────────────

    def save(self, path=None):
        """Write metrics to binary (Cython) or JSON (fallback)."""
        path = path or os.path.join(self.save_dir, "metrics" + _ext())
        if _HAS_CY:
            save_metrics_cy(path, self.window, self.metrics)
        else:
            data = {"_window": self.window}
            for name, pts in self.metrics.items():
                data[name] = [(int(s), float(v)) for s, v in pts]
            with open(path, "w") as f:
                json.dump(data, f)

    def _load(self, path=None):
        """Restore metrics from disk (supports resume)."""
        # Try binary first, then JSON
        path_bin = path or os.path.join(self.save_dir, "metrics.bin")
        path_json = path or os.path.join(self.save_dir, "metrics.json")

        loaded = False
        if _HAS_CY and os.path.exists(path_bin):
            try:
                self.window, self.metrics = load_metrics_cy(path_bin)
                loaded = True
            except Exception as e:
                print(f"[MONITOR] Binary restore failed: {e}")

        if not loaded and os.path.exists(path_json):
            try:
                with open(path_json) as f:
                    data = json.load(f)
                self.window = data.pop("_window", self.window)
                self.metrics = {
                    name: [(int(s), float(v)) for s, v in pts]
                    for name, pts in data.items()
                }
                loaded = True
            except Exception as e:
                print(f"[MONITOR] JSON restore failed: {e}")

        if loaded:
            n = sum(len(v) for v in self.metrics.values())
            if n:
                print(f"[MONITOR] Restored {n} data points")


# ── HTML chart generation (called from lcm.py) ────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8" /><title>LCM Training Metrics</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script></head>
<body style="font-family:system-ui,sans-serif;margin:32px;">
<h2 style="margin-bottom:4px;">LCM Training Metrics</h2>
<p style="color:#666;margin-top:0;">{npts} data points · window: every {window} steps</p>
<div id="chart"></div>
<script>
var data = {TRACE_DATA};
var layout = {{
    grid: {{rows: {nrows}, columns: 1, pattern: 'independent'}},
    height: {height},
    hovermode: 'x unified',
    template: 'plotly_white',
    margin: {{l:60, r:30, t:30, b:40}},
}};
Plotly.newPlot('chart', data, layout, {{responsive:true}});
</script>
</body>
</html>"""


def _read_metrics(path):
    """Read metrics from binary (.bin) or JSON (.json), auto-detect."""
    if path.endswith(".bin") and _HAS_CY:
        return load_metrics_cy(path)
    # JSON fallback
    with open(path) as f:
        raw = json.load(f)
    window = raw.pop("_window", 50)
    return window, raw


def make_chart_html(metrics_path, output_path="metrics.html"):
    """Read metrics file and write interactive HTML chart (auto-detect format)."""
    window, raw = _read_metrics(metrics_path)
    names = sorted(raw.keys())
    pts_list = [raw[n] for n in names]

    if not names:
        print("[CHART] No metrics found.")
        return

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]

    traces = []
    for i, (name, pts) in enumerate(zip(names, pts_list)):
        steps, vals = zip(*pts) if pts else ([], [])
        traces.append({
            "x": list(steps),
            "y": list(vals),
            "mode": "lines+markers",
            "name": name,
            "type": "scatter",
            "marker": {"size": 3, "color": colors[i % len(colors)]},
            "line": {"width": 1.5, "color": colors[i % len(colors)]},
            "yaxis": f"y{i+1}" if i > 0 else "y",
            "xaxis": f"x{i+1}" if i > 0 else "x",
            "hovertemplate": f"<b>{name}</b><br>step=%{{x}}<br>value=%{{y:.6f}}<extra></extra>",
        })

    npts = sum(len(v) for v in pts_list)
    height = 200 * len(names) + 60

    import json as _json
    trace_json = _json.dumps(traces)
    html = _HTML_TEMPLATE.format(
        TRACE_DATA=trace_json, nrows=len(names),
        npts=npts, window=window, height=height,
    )
    with open(output_path, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[CHART] {npts} points across {len(names)} metrics → {output_path} ({size_kb:.0f} KB)")
