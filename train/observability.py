"""自我观察（黑匣子）— 纯记录，零判断。

职责：每步推理完成后，将原始轨迹数据无损写入环形缓冲区。
不评判重要性，不做筛选，不参与推理。纯粹的数据记录仪。

类比：赛车上的黑匣子。不判断哪些操作重要，只记录所有数据。
赛后分析时，黑匣子提供精确的回放能力。

Usage:
    from train.observability import ObservabilityRecorder

    obs = ObservabilityRecorder(raw_capacity=1000)
    obs.record_step(step=42, soft_mask=..., route_idx=..., ...)
    trace = obs.get_trace(42)
    obs.print_summary()
"""
import time
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


# ─── Record schema ──────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Single inference step record — immutable snapshot.

    All fields are optional; only provided fields are stored.
    Vectors (z_q, z) are stored as numpy arrays if record_vectors=True.
    """
    # Meta
    step: int
    timestamp: float = 0.0
    step_time_ms: float = 0.0
    source: int = 0  # 0=external, 1=internal (因果主体源标签)

    # Routing
    soft_mask: Optional[np.ndarray] = None   # (n_lattices,)
    route_idx: Optional[int] = None

    # Retrieval path
    hrq_top_sim: Optional[float] = None
    hrq_idx: Optional[List[int]] = None
    sparse_idx: Optional[int] = None
    man_idx: Optional[int] = None

    # Self
    self_mode: Optional[int] = None
    world_dev: Optional[float] = None

    # Safety
    safety_margin: Optional[float] = None
    value_signals: Optional[np.ndarray] = None  # (n_lattices,)
    is_safe: bool = True
    violated_law: Optional[int] = None

    # Convergence
    convergence_diff: Optional[float] = None
    convergence_entropy: Optional[float] = None
    n_retrievals: int = 0

    # C engine DAG trace
    dag_nodes: Optional[List[Dict[str, Any]]] = None
    conflict_source: Optional[str] = None
    conflict_detail: Optional[str] = None

    # Vectors
    z_q: Optional[np.ndarray] = None  # (d,)
    z: Optional[np.ndarray] = None    # (d,)


# ─── Ring buffer ────────────────────────────────────────────────────────────

class RingBuffer:
    """Fixed-capacity circular buffer — O(1) append, O(capacity) memory.

    Once full, oldest entries are silently overwritten (no blurring).
    Perfect for short-term trajectory storage.
    """
    def __init__(self, capacity: int):
        assert capacity > 0, "capacity must be > 0"
        self.capacity = capacity
        self._buffer = [None] * capacity
        self._head = 0
        self._count = 0

    def append(self, item: Any):
        self._buffer[self._head] = item
        self._head = (self._head + 1) % self.capacity
        self._count += 1

    def __getitem__(self, idx: int):
        if len(self) == 0:
            raise IndexError("RingBuffer is empty")
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        # Items are stored from (head-count) to (head-1), wrapping as needed
        pos = (self._head - self._count + idx) % self.capacity
        return self._buffer[pos]

    def __len__(self):
        return min(self._count, self.capacity)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_by_step(self, step: int) -> Optional[Any]:
        for record in self:
            if record and record.step == step:
                return record
        return None

    def pop_by_step(self, step: int) -> Optional[Any]:
        """Remove and return a specific step (for promotion)."""
        for i in range(len(self)):
            record = self[i]
            if record and record.step == step:
                self._buffer[(self._head - len(self) + i) % self.capacity] = None
                return record
        return None

    def clear(self):
        self._buffer = [None] * self.capacity
        self._head = 0
        self._count = 0

    @property
    def utilization(self):
        return len(self) / self.capacity

    def to_list(self):
        return [r for r in self if r is not None]


# ─── Observability Recorder (pure recording, no judgment) ──────────────────

class ObservabilityRecorder:
    """自我观察（黑匣子）— 纯记录，零判断。

    唯一职责：记录原始推理轨迹到环形缓冲区。
    不做重要性评估，不做筛选，不干扰推理。
    """
    def __init__(self, raw_capacity: int = 1000,
                 record_vectors: bool = True,
                 record_every_n: int = 1):
        """
        Args:
            raw_capacity: 环形缓冲区容量（步数）。
            record_vectors: 是否存储 z/z_q 向量（增加内存）。
            record_every_n: 采样率（1=每步都记录）。
        """
        self.ring = RingBuffer(raw_capacity)
        self.record_vectors = record_vectors
        self.record_every_n = max(1, record_every_n)
        self._last_step = -1
        self._start_time = time.time()

    def should_record(self, step: int) -> bool:
        return step % self.record_every_n == 0

    def record_step(self, step: int, **kwargs) -> Optional[StepRecord]:
        """Record one step's trajectory data.

        Call this AFTER inference step completes.
        O(1) — fixed memory append, no allocation spikes.

        Args:
            step: Step number.
            **kwargs: Fields matching StepRecord attributes.

        Returns:
            The created StepRecord, or None if skipped (sampling).
        """
        if not self.should_record(step):
            return None

        record_kwargs = {'step': step, 'timestamp': time.time()}
        for key in ['step_time_ms', 'source', 'soft_mask', 'route_idx', 'hrq_top_sim',
                     'hrq_idx', 'sparse_idx', 'man_idx', 'self_mode',
                     'world_dev', 'safety_margin', 'value_signals', 'is_safe',
                     'violated_law', 'convergence_diff', 'convergence_entropy',
                     'n_retrievals', 'dag_nodes', 'conflict_source',
                     'conflict_detail']:
            if key in kwargs:
                record_kwargs[key] = kwargs[key]

        if self.record_vectors:
            for key in ['z_q', 'z']:
                if key in kwargs and kwargs[key] is not None:
                    val = kwargs[key]
                    if hasattr(val, 'numpy'):
                        val = np.asarray(val)
                    if isinstance(val, np.ndarray):
                        record_kwargs[key] = val.copy() if val.ndim > 0 else val

        record = StepRecord(**record_kwargs)
        self.ring.append(record)
        self._last_step = step
        return record

    # ── Query ──

    def get_trace(self, step: int) -> Optional[StepRecord]:
        return self.ring.get_by_step(step)

    def get_recent_traces(self, n: int = 10) -> List[StepRecord]:
        return self.ring.to_list()[-n:]

    def get_traces_by_range(self, start_step: int,
                            end_step: int) -> List[StepRecord]:
        return [r for r in self.ring if r and start_step <= r.step <= end_step]

    def get_traces_by_route(self, route_idx: int) -> List[StepRecord]:
        return [r for r in self.ring if r and r.route_idx == route_idx]

    # ── Summary ──

    def print_summary(self) -> None:
        n = len(self.ring)
        if n == 0:
            print("[OBS] No records yet.")
            return

        unsafe = sum(1 for r in self.ring if r and not r.is_safe)
        steps = [r.step for r in self.ring if r]

        print(f"\n{'=' * 46}")
        print(f"  自我观察 (Observability Recorder)")
        print(f"{'=' * 46}")
        print(f"  Records:          {n}")
        print(f"  Capacity:         {self.ring.capacity}")
        print(f"  Utilization:      {self.ring.utilization:.1%}")
        print(f"  Step range:       {min(steps)} — {max(steps)}")
        print(f"  Unsafe steps:     {unsafe}")
        print(f"  Elapsed:          {time.time() - self._start_time:.1f}s")
        print(f"{'=' * 46}\n")

    # ── Export ──

    def export_json(self, path: str, max_records: int = 0,
                    include_vectors: bool = False) -> None:
        records = self.ring.to_list()
        if max_records > 0:
            records = records[-max_records:]

        data = []
        for r in records:
            if r is None:
                continue
            d = asdict(r)
            for k, v in list(d.items()):
                if isinstance(v, np.ndarray):
                    if include_vectors and k in ('z_q', 'z'):
                        d[k] = v.tolist()
                    elif k in ('soft_mask', 'value_signals') and v is not None:
                        d[k] = v.tolist()
                    else:
                        d[k] = None
                elif isinstance(v, (np.integer,)):
                    d[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    d[k] = float(v)
            data.append(d)

        with open(path, 'w') as f:
            json.dump({'n_records': len(data), 'records': data},
                       f, indent=2, default=str)
        print(f"[OBS] Exported {len(data)} raw records → {path}")

    def clear(self) -> None:
        self.ring.clear()
        self._last_step = -1


__all__ = [
    'StepRecord',
    'RingBuffer',
    'ObservabilityRecorder',
]
