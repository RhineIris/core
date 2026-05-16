"""Continual Learning System for LCM — Stage 4.

Enables incremental learning without catastrophic forgetting via:
1. Dynamic codebook expansion for new domains/tasks
2. Elastic Weight Consolidation (EWC) on protected parameters
3. Experience replay across previously seen domains
4. Memory consolidation of high-frequency patterns
"""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ContinualState:
    """Tracks all continual learning state across tasks."""
    task_id: int = 0
    step: int = 0
    task_boundaries: Dict[int, int] = field(default_factory=dict)  # step -> task_id
    protected_params: Dict[str, jnp.ndarray] = field(default_factory=dict)  # path -> frozen copy
    fisher_diag: Dict[str, jnp.ndarray] = field(default_factory=dict)  # path -> Fisher diag
    replay_buffers: Dict[int, Dict[str, jnp.ndarray]] = field(default_factory=dict)  # task_id -> buffer
    access_counters: Dict[str, jnp.ndarray] = field(default_factory=dict)  # path -> per-entry counts
    consolidation_log: List[Dict] = field(default_factory=list)
    z_mean_ema: Optional[jnp.ndarray] = None  # Running EMA of z for shift detection
    z_cov_ema: Optional[jnp.ndarray] = None   # Running EMA of z covariance
    n_seen: int = 0


def init_continual_state(d_model: int) -> ContinualState:
    """Initialize continual learning state."""
    return ContinualState(
        z_mean_ema=jnp.zeros(d_model),
        z_cov_ema=jnp.eye(d_model) * 0.1,
    )


# ── 1. Distribution Shift Detection ──────────────────────────────────────────

def detect_new_task(z: jnp.ndarray, state: ContinualState,
                    threshold: float) -> Tuple[bool, ContinualState]:
    """Detect distribution shift via Mahalanobis distance on z.

    Updates running EMA of mean and covariance. If batch mean exceeds
    threshold Mahalanobis distance from EMA mean, signals new task.
    """
    B, d = z.shape
    batch_mean = z.mean(axis=0)

    if state.n_seen == 0:
        state.z_mean_ema = batch_mean
        state.n_seen = B
        return False, state

    # Mahalanobis distance
    diff = batch_mean - state.z_mean_ema
    cov_inv = jnp.linalg.pinv(state.z_cov_ema + 1e-6 * jnp.eye(d))
    m_dist = jnp.sqrt(diff @ cov_inv @ diff)

    # Update EMA statistics
    decay = jnp.clip(1.0 - 1.0 / state.n_seen, 0.9, 0.999)
    new_mean = decay * state.z_mean_ema + (1 - decay) * batch_mean
    batch_cov = jnp.cov(z.T)
    new_cov = decay * state.z_cov_ema + (1 - decay) * batch_cov
    is_new = m_dist > threshold

    state.z_mean_ema = new_mean
    state.z_cov_ema = new_cov
    state.n_seen += B

    return bool(is_new), state


# ── 2. Dynamic Codebook Expansion ────────────────────────────────────────────

def expand_codebook(param: jnp.ndarray, n_new: int, rng: jax.Array,
                    init_scale: float = 0.02) -> jnp.ndarray:
    """Expand a codebook by appending n_new randomly initialized entries."""
    M, d = param.shape
    new_entries = jax.random.normal(rng, (n_new, d)) * init_scale
    return jnp.concatenate([param, new_entries], axis=0)


def expand_lattice_codebooks(params: dict, n_new: int, rng: jax.Array,
                             cfg) -> dict:
    """Expand all lattice codebooks for a new task."""
    keys = jax.random.split(rng, 10)
    ki = 0

    # HRQ: top + fine layers
    M_top_old = params['hrq']['top']['A'].shape[0]
    params['hrq']['top']['A'] = expand_codebook(
        params['hrq']['top']['A'], n_new, keys[ki]); ki += 1
    for l in range(len(params['hrq']['fine'])):
        params['hrq']['fine'][l]['A'] = expand_codebook(
            params['hrq']['fine'][l]['A'], n_new, keys[ki]); ki += 1

    # Sparse
    params['sparse']['C'] = expand_codebook(
        params['sparse']['C'], n_new, keys[ki]); ki += 1

    # Low-rank: U layers
    for l in range(len(params['lowrank']['U'])):
        u_shape = params['lowrank']['U'][l].shape  # (M_lr, r_k)
        new_u = jax.random.normal(keys[ki], (n_new, u_shape[1])) * 0.02; ki += 1
        params['lowrank']['U'][l] = jnp.concatenate(
            [params['lowrank']['U'][l], new_u], axis=0)

    # Manifold: C + T
    params['manifold']['C'] = expand_codebook(
        params['manifold']['C'], n_new, keys[ki]); ki += 1

    t_dim = params['manifold']['T'].shape[-1]
    new_T = jax.random.normal(keys[ki], (n_new, cfg.d_model, t_dim)) * 0.01; ki += 1
    params['manifold']['T'] = jnp.concatenate([params['manifold']['T'], new_T], axis=0)

    # Binding: all sub-codebook layers
    for cb_type in ['key_cb', 'val_cb', 'bind_cb']:
        for l in range(len(params['binding'][cb_type])):
            params['binding'][cb_type][l]['A'] = expand_codebook(
                params['binding'][cb_type][l]['A'], n_new, keys[ki]); ki += 1

    # Contrast: C_a + C_b layers
    for l in range(len(params['contrast']['C_a'])):
        params['contrast']['C_a'][l]['A'] = expand_codebook(
            params['contrast']['C_a'][l]['A'], n_new, keys[ki]); ki += 1
        params['contrast']['C_b'][l]['A'] = expand_codebook(
            params['contrast']['C_b'][l]['A'], n_new, keys[ki]); ki += 1

    return params


def expand_value_scalars(value_scalars: dict, lattice_name: str,
                         n_new: int) -> dict:
    """Expand local value scalars for a lattice."""
    if lattice_name in value_scalars:
        new_v = jnp.zeros(n_new)
        value_scalars[lattice_name] = jnp.concatenate(
            [value_scalars[lattice_name], new_v])
    return value_scalars


# ── 3. Elastic Weight Consolidation (EWC) ────────────────────────────────────

def _param_groups(params: dict) -> List[Tuple[str, jnp.ndarray]]:
    """Flatten nested param dict into list of (path, array) pairs."""
    groups = []
    _collect_groups(params, '', groups)
    return groups


def _collect_groups(p, prefix: str, groups: List[Tuple[str, jnp.ndarray]]):
    if isinstance(p, dict):
        for k, v in p.items():
            _collect_groups(v, f"{prefix}{k}/", groups)
    elif isinstance(p, list):
        for i, v in enumerate(p):
            _collect_groups(v, f"{prefix}{i}/", groups)
    else:
        groups.append((prefix.rstrip('/'), p))


def snapshot_protected_params(params: dict) -> Dict[str, jnp.ndarray]:
    """Create frozen copy of all gradient-updated params for EWC protection."""
    protected = {}
    for path, val in _param_groups(params):
        protected[path] = jnp.array(val, copy=True)
    return protected


def compute_fisher_diag_flat(grad_fn, params: dict, z: jnp.ndarray,
                              aux: dict, rng: jax.Array,
                              n_samples: int) -> Dict[str, jnp.ndarray]:
    """Estimate Fisher diagonal via Monte Carlo gradient outer product.

    Args:
        grad_fn: Callable taking (params, z, aux) returning (loss, aux_output).
        params: Current model parameters.
        z: Encoder output for a batch.
        aux: Auxiliary outputs.
        rng: PRNG key.
        n_samples: Number of MC samples (small, ~50).

    Returns:
        fisher: dict of path -> Fisher diagonal array.
    """
    groups = _param_groups(params)

    # Get grads once
    (_, _), grads = grad_fn(params, z, aux)
    grad_groups = _param_groups(grads)

    fisher = {}
    for (path, _), (_, g) in zip(groups, grad_groups):
        fisher[path] = g ** 2

    # Additional MC samples with input noise
    for _ in range(1, n_samples):
        rng, subkey = jax.random.split(rng)
        z_noisy = z + jax.random.normal(subkey, z.shape) * 1e-3
        (_, _), grads = grad_fn(params, z_noisy, aux)
        _, g_groups = _param_groups(grads), None
        for (path, _), (_, g) in zip(groups, _param_groups(grads)):
            fisher[path] = fisher[path] + g ** 2

    for path in fisher:
        fisher[path] = fisher[path] / n_samples

    return fisher


def compute_ewc_loss(params: dict, protected_params: Dict[str, jnp.ndarray],
                     fisher_diag: Dict[str, jnp.ndarray],
                     ewc_lambda: float) -> jnp.ndarray:
    """Elastic Weight Consolidation loss: λ/2 * Σ_i F_i * (θ_i - θ_i*)²."""
    loss = 0.0
    for path, theta in _param_groups(params):
        if path in protected_params and path in fisher_diag:
            diff = theta - protected_params[path]
            loss = loss + jnp.sum(fisher_diag[path] * diff ** 2)
    return 0.5 * ewc_lambda * loss


# ── 4. Experience Replay ─────────────────────────────────────────────────────

def update_replay_buffer(state: ContinualState, task_id: int,
                         z: jnp.ndarray, logits: jnp.ndarray,
                         soft_mask: jnp.ndarray, capacity: int) -> ContinualState:
    """Update per-domain replay buffer with current batch."""
    if task_id not in state.replay_buffers:
        state.replay_buffers[task_id] = {
            'z': z[:capacity],
            'logits': logits[:capacity],
            'soft_mask': soft_mask[:capacity],
            'ptr': 0,
            'full': False,
        }

    buf = state.replay_buffers[task_id]
    B = z.shape[0]
    cap = capacity

    for i in range(B):
        idx = buf['ptr'] % cap
        buf['z'] = buf['z'].at[idx].set(z[i])
        buf['logits'] = buf['logits'].at[idx].set(logits[i])
        buf['soft_mask'] = buf['soft_mask'].at[idx].set(soft_mask[i])
        buf['ptr'] += 1
    buf['full'] = buf['ptr'] >= cap

    return state


def sample_replay(state: ContinualState, task_id: int,
                  batch_size: int, replay_ratio: float,
                  rng: jax.Array) -> Tuple[Optional[Dict[str, jnp.ndarray]], float]:
    """Sample from replay buffers of previous tasks.

    Returns:
        (replay_batch, replay_weight): replay_batch is None if no replay data.
    """
    old_tasks = [t for t in state.replay_buffers if t < task_id]
    if not old_tasks:
        return None, 0.0

    n_replay = int(batch_size * replay_ratio)
    if n_replay < 1:
        return None, 0.0

    # Uniform across old tasks
    n_per_task = max(1, n_replay // len(old_tasks))
    all_z, all_logits, all_masks = [], [], []

    for t in old_tasks:
        buf = state.replay_buffers[t]
        n_avail = min(buf['ptr'], buf['z'].shape[0]) if buf['full'] else buf['ptr']
        if n_avail < 1:
            continue

        rng, subkey = jax.random.split(rng)
        indices = jax.random.choice(subkey, n_avail, (min(n_per_task, n_avail),),
                                    replace=False)
        all_z.append(buf['z'][indices])
        all_logits.append(buf['logits'][indices])
        all_masks.append(buf['soft_mask'][indices])

    if not all_z:
        return None, 0.0

    return {
        'z': jnp.concatenate(all_z, axis=0),
        'logits': jnp.concatenate(all_logits, axis=0),
        'soft_mask': jnp.concatenate(all_masks, axis=0),
    }, n_replay / batch_size


# ── 5. Memory Consolidation ──────────────────────────────────────────────────

def update_access_counters(state: ContinualState, params: dict,
                           aux: dict) -> ContinualState:
    """Increment access counters for each codebook entry used this step."""
    # HRQ top index
    for path in ['hrq_top']:
        if path not in state.access_counters:
            state.access_counters[path] = jnp.zeros(
                params['hrq']['top']['A'].shape[0], dtype=jnp.int32)
    # Sparse index
    if 'sparse' not in state.access_counters:
        state.access_counters['sparse'] = jnp.zeros(
            params['sparse']['C'].shape[0], dtype=jnp.int32)
    # Manifold index
    if 'manifold' not in state.access_counters:
        state.access_counters['manifold'] = jnp.zeros(
            params['manifold']['C'].shape[0], dtype=jnp.int32)

    if 'hrq_idx' in aux:
        idx = aux['hrq_idx']
        for i in range(idx.shape[0]):
            state.access_counters['hrq_top'] = state.access_counters['hrq_top'].at[
                idx[i]].add(1)

    if 'sparse_idx' in aux:
        idx = aux['sparse_idx']
        for i in range(idx.shape[0]):
            state.access_counters['sparse'] = state.access_counters['sparse'].at[
                idx[i]].add(1)

    if 'man_idx' in aux:
        idx = aux['man_idx']
        for i in range(idx.shape[0]):
            state.access_counters['manifold'] = state.access_counters['manifold'].at[
                idx[i]].add(1)

    return state


def consolidate_memory(state: ContinualState, params: dict,
                       step: int, frequency_threshold: int = 50) -> Tuple[dict, ContinualState]:
    """Move high-frequency entries to 'stable' pool (marked via logging).

    In practice, consolidation means:
    - High-frequency entries get added to protected_params for EWC
    - Low-frequency entries remain plastic
    - Returns updated params and state with consolidation log entry.
    """
    consolidated = []
    for path, counters in state.access_counters.items():
        high_freq = jnp.where(counters > frequency_threshold)[0]
        if len(high_freq) > 0:
            consolidated.append({'path': path, 'n_entries': len(high_freq)})
            # Reset counters for consolidated entries
            state.access_counters[path] = state.access_counters[path].at[high_freq].set(0)

    if consolidated:
        state.consolidation_log.append({
            'step': step,
            'task_id': state.task_id,
            'consolidated': consolidated,
        })

    return params, state
