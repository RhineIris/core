"""Dual-channel cognitive training: passive introspection + active expression.

Two output channels from the same conscious state z_q:
  - Passive: z_q @ W_out — transparent, always readable, no deception gap
  - Active:  gen_head decoder — fluent language skill, loadable from Stage 1

The passive channel keeps the model honest (cognitive state is directly readable).
The active channel gives it full language capabilities without restriction.

Passive checkpoint files (gen_head + w_start) from Stage 1 LM training can be
loaded directly into the active channel via --from-lm-ckpt.

Usage:
    python lcm.py --cog-train -d zhwiki_tokens.dat --from-lm-ckpt checkpoints/lm_final.pkl
"""
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from train.config import LCMConfig
from train.encoder import init_encoder_params, encoder_forward
from train.fusion import init_gen_head_params
from train.lattices import (
    init_hrq_params, init_sparse_params, init_lowrank_params,
    init_manifold_params, init_binding_params, init_contrast_params,
)
from train.self_lattice import (
    init_self_params, init_self_state, self_lattice_forward,
    self_lattice_reg_loss,
)
from train.cog_loop import cog_loop_scan


# ─── Init full params ───────────────────────────────────────────────────────

def init_cog_params(cfg, rng, lm_ckpt=None):
    """Initialize all trainable params.

    Passive channel: W_out (trained from scratch).
    Active channel: gen_head + w_start (loadable from Stage 1 LM checkpoint).

    Returns:
        params: Dict of all parameters (including self-lattice).
        self_state: Dict for self-lattice runtime state.
    """
    keys = jax.random.split(rng, 12)
    d = cfg.d_model

    params = {}
    # Encoder
    params['encoder'] = init_encoder_params(
        keys[0], d, cfg.d_ff, cfg.n_heads, cfg.n_encoder_layers,
        cfg.vocab_size, cfg.max_seq_len)

    # Codebooks (native lattice format for compatibility with lattice-specific losses)
    params['hrq'] = init_hrq_params(keys[1], d, cfg.M_top, cfg.M_fine, cfg.n_hrq_layers)
    params['sparse'] = init_sparse_params(keys[2], d, cfg.M_sparse)
    params['lowrank'] = init_lowrank_params(keys[3], d, cfg.M_lr, cfg.ranks)
    params['manifold'] = init_manifold_params(keys[4], d, cfg.M_man, cfg.t_dim)
    params['binding'] = init_binding_params(keys[5], d, cfg.M_bind, cfg.n_bind_layers, cfg.r_max)
    params['contrast'] = init_contrast_params(keys[6], d, cfg.M_contrast, cfg.n_contrast_layers)

    # Self lattice params
    params['self'] = init_self_params(keys[10], d, cfg.n_self_codes)
    self_state = init_self_state(cfg.n_self_codes, d)

    # Passive channel: transparent consciousness → language projection
    params['W_out'] = jax.random.normal(keys[7], (d, cfg.vocab_size)) * (d ** -0.5)

    # Active channel: fluent language skill
    if lm_ckpt:
        print(f"[COG] Loading gen_head from Stage 1: {lm_ckpt}")
        with open(lm_ckpt, 'rb') as f:
            ckpt = pickle.load(f)
        params['gen_head'] = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
            ckpt['gen_head'])
        params['w_start'] = jnp.array(ckpt['w_start'])
    else:
        params['gen_head'] = init_gen_head_params(keys[8], d, cfg.vocab_size)
        params['w_start'] = jax.random.normal(keys[9], (d,)) * (d ** -0.5)

    return params, self_state


def _simvq_codebook(simvq):
    """Extract actual codebook matrix from SimVQ params: A @ W."""
    return simvq['A'] @ simvq['W']


def pack_codebooks_for_c(p):
    """Extract all codebook (K_i, d) matrices into flat list for cognitive loop."""
    flat = []

    # HRQ: top + fine per layer
    flat.append(_simvq_codebook(p['hrq']['top']))
    for fb in p['hrq']['fine']:
        flat.append(_simvq_codebook(fb))

    # Sparse
    flat.append(p['sparse']['C'])

    # LowRank: one per rank
    V = p['lowrank']['A_V'] @ p['lowrank']['W_V']
    for l, u_k in enumerate(p['lowrank']['U']):
        r_k = p['lowrank']['U'][l].shape[-1]
        flat.append(u_k @ V[:, :r_k].T)

    # Manifold
    flat.append(p['manifold']['C'])

    # Binding: key, value, bind per layer
    for i in range(len(p['binding']['key_cb'])):
        flat.append(_simvq_codebook(p['binding']['key_cb'][i]))
        flat.append(_simvq_codebook(p['binding']['val_cb'][i]))
        flat.append(_simvq_codebook(p['binding']['bind_cb'][i]))

    # Contrast: C_a, C_b per layer
    for i in range(len(p['contrast']['C_a'])):
        flat.append(_simvq_codebook(p['contrast']['C_a'][i]))
        flat.append(_simvq_codebook(p['contrast']['C_b'][i]))

    return flat


def avg_codebook_distances(codebooks):
    """Compute avg pairwise distance per codebook — for threshold setting."""
    avg_dists = []
    for cb in codebooks:
        n = cb.shape[0]
        if n > 100:
            idx = np.random.choice(n, min(100, n), replace=False)
            sample = cb[idx]
        else:
            sample = cb
        dists = jnp.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
        avg_dists.append(float(jnp.mean(dists)))
    return avg_dists


# ─── Passive channel: transparent introspection ─────────────────────────────

def passive_loss(logits_1d, target_token):
    """Single-token CE loss for passive introspection channel.

    Args:
        logits_1d: (V,) predicted logits from z_q @ W_out.
        target_token: int scalar — the next token.
    """
    return optax.softmax_cross_entropy_with_integer_labels(
        logits_1d[None, :], jnp.array([target_token])).mean()


# ─── Active channel: fluent language skill ──────────────────────────────────

def decoder_forward(gen_head, z_q, x, w_start):
    """Active channel: autoregressive decoder from cognitive state z_q.

    Same architecture as Stage 1 LM (train_lm.py). Uses z_q as the start
    query instead of a learned start vector.
    """
    B, N = x.shape
    g = gen_head
    d = g['w_q'].shape[-1]

    target_emb = g['w_embed'][x]
    z_q_2d = z_q.reshape(1, 1, -1)
    z_q_batch = jnp.broadcast_to(z_q_2d, (B, 1, d))
    inputs = jnp.concatenate([z_q_batch, target_emb], axis=1)

    # Causal linear attention: φ(x) = ELU(x) + 1
    Q = jax.nn.elu(inputs @ g['w_q']) + 1.0
    K = jax.nn.elu(inputs @ g['w_k']) + 1.0
    V = inputs @ g['w_v']

    kv = K[:, :, :, None] @ V[:, :, None, :]
    kv_cs = jnp.cumsum(kv, axis=1)
    k_cs = jnp.cumsum(K, axis=1)

    attn = jnp.einsum('bnd,bndd->bnd', Q, kv_cs) / (
        jnp.einsum('bnd,bnd->bn', Q, k_cs)[:, :, None] + 1e-8)
    attn_out = attn @ g['w_o']

    # GLU
    gate = jax.nn.sigmoid(attn_out @ g['w_1'])
    up = attn_out @ g['w_2']
    glu_out = gate * up

    logits = glu_out @ g['w_3']
    return logits[:, 1:, :]  # (B, N, V)


def active_loss(logits, targets):
    """Cross-entropy for active channel (full sequence)."""
    B, N, V = logits.shape
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, V), targets.reshape(-1)).mean()


# ─── Training step ──────────────────────────────────────────────────────────

def make_train_step(cfg, optimizer):
    """Create jitted training step with dual-channel output + self-lattice.

    Every macro step's z_q feeds both channels:
      - Passive (introspection):  z_q @ W_out → single-token CE
      - Active (expression):      gen_head(z_q) → full-sequence CE

    Self-lattice provides internal state machine (mode selection, self output).
    """

    @jax.jit
    def train_step(params, opt_state, batch, lr, rng, self_state=None):
        inputs, targets = batch
        B, N = inputs.shape

        def loss_fn(p):
            z = encoder_forward(p['encoder'], inputs, cfg.n_heads)  # (B, d)
            codebooks = pack_codebooks_for_c(p)

            # ── Batch-aware cognitive loop (vmap over batch) ────────────
            _cog = lambda zi: cog_loop_scan(
                zi, codebooks,
                max_steps=cfg.max_inference_steps,
                thresholds=None, tau=0.1)
            z_qs, diffs, entropies = jax.vmap(_cog, in_axes=0)(z)
            # z_qs: (B, max_steps, d), diffs: (B, max_steps)

            # ── Self lattice ────────────────────────────────────────────
            z_final_mean = z_qs[:, -1, :].mean(axis=0)  # (d,)
            rng_self = rng
            self_state_out = None
            loss_self = jnp.array(0.0)
            if self_state is not None and 'self' in p:
                o_self, self_state_out, world_dev = self_lattice_forward(
                    p['self'], self_state, z=z_final_mean[None, :],
                    rng=rng_self, training=True)
                loss_self = self_lattice_reg_loss(p['self'], self_state_out)

            # ── Passive channel: (B, max_steps, V) logits → (B,) target ─
            p_logits = jnp.einsum('bsd,dv->bsv', z_qs, p['W_out'])  # (B, S, V)
            p_target = targets[:, 0]                                  # (B,)
            # CE over all steps × batch — mean, not weighted sum
            p_loss = optax.softmax_cross_entropy_with_integer_labels(
                p_logits.reshape(-1, p_logits.shape[-1]),
                p_target[:, None].repeat(cfg.max_inference_steps, axis=1).reshape(-1),
            ).mean()

            # ── Active channel: vmap gen_head over batch ────────────────
            _decode = lambda z_i, x_i: decoder_forward(
                p['gen_head'], z_i, x_i[None, :], p['w_start'])[0]
            a_logits = jax.vmap(_decode, in_axes=(0, 0))(z_qs[:, -1, :], inputs)
            a_loss = active_loss(a_logits, targets)

            # Convergence bonus (batch-mean)
            conv = (diffs[:, -1] < cfg.convergence_tol) & (entropies[:, -1] < cfg.entropy_threshold)
            n_steps = jnp.argmax((diffs < cfg.convergence_tol).astype(jnp.float32), axis=-1) + 1

            loss = p_loss + a_loss + loss_self + jnp.mean(
                jnp.where(conv, -0.001 * jnp.log(n_steps.astype(jnp.float32) + 1e-8), 0.0))

            aux_out = {'self_state': self_state_out, 'loss_self': loss_self}
            return loss, aux_out

        (loss, aux_out), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, aux_out

    return train_step


# ─── Training loop ──────────────────────────────────────────────────────────

def train_cog(cfg, output_dir, steps=50000, lr=3e-4, batch_size=1,
              seq_len=256, log_every=100, save_every=1000,
              data_path=None, shape_path=None, lm_ckpt=None):
    """Run dual-channel cognitive training."""
    from train.data import WikiDataIter
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(42)

    rng, init_rng = jax.random.split(rng)
    params, self_state = init_cog_params(cfg, init_rng, lm_ckpt=lm_ckpt)

    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps, alpha=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=cfg.adam_beta1,
                     b2=cfg.adam_beta2, eps=cfg.adam_eps,
                     weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(params)

    data_iter = WikiDataIter(data_path, shape_path, B=batch_size, N=seq_len)
    train_step = make_train_step(cfg, optimizer)

    codebooks_flat = pack_codebooks_for_c(params)
    avg_dists = avg_codebook_distances(codebooks_flat)
    thresholds = [d * 0.15 for d in avg_dists]
    print(f"[COG] Codebook thresholds: {[f'{t:.3f}' for t in thresholds[:6]]}")

    d = cfg.d_model
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params)
                       if hasattr(p, 'size'))
    has_lm = 'gen_head' in params
    print(f"[COG] Dual-channel: passive (z_q @ W_out) + active (gen_head{' from LM' if lm_ckpt else ' init'})")
    print(f"[COG] Self-lattice: {cfg.n_self_codes} modes")
    print(f"[COG] Total params: {total_params:,}")
    print(f"[COG] Steps: {steps}, B={batch_size}, N={seq_len}, lr={lr}")
    print()

    import numpy as _np_np
    V = cfg.vocab_size
    _LN_V = float(_np_np.log(V))  # passive random baseline ≈ 10.31
    _ACTIVE_FLOOR = 5.06           # Stage 1 gen_head final loss
    _LOSS_FLOOR = _ACTIVE_FLOOR    # total lower bound (passive → 0, active → floor)

    print(f"[COG] Info theory bounds:")
    print(f"      passive random: ln(V) = {_LN_V:.2f}")
    print(f"      active  floor:  Stage 1 baseline = {_ACTIVE_FLOOR:.2f}")
    print(f"      total   floor:  {_LOSS_FLOOR:.2f}  (gap = loss - {_LOSS_FLOOR:.2f})")
    print()

    running_loss = 0.0
    start_time = time.time()
    pbar = tqdm(total=steps, desc="cog training", unit="step")

    import signal as _signal

    def _handler(sig, frame):
        print(f"\n[COG] Interrupt at step {step}, saving checkpoint...")
        save_cog_checkpoint(params, output_dir, step, self_state=self_state)
        print(f"[COG] Saved → {output_dir}/cog_params.pkl")
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _handler)

    for step in range(steps):
        batch = next(data_iter)
        current_lr = schedule(step)
        rng, step_rng = jax.random.split(rng)

        params, opt_state, loss_val, aux_out = train_step(
            params, opt_state, batch, current_lr, step_rng, self_state=self_state)

        # Update self state from forward pass
        if self_state is not None and aux_out.get('self_state') is not None:
            self_state = aux_out['self_state']

        running_loss += float(loss_val)

        if step % log_every == 0 and step > 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - start_time
            tok_s = batch_size * seq_len * log_every / elapsed
            loss_self = float(aux_out.get('loss_self', 0.0))
            gap = avg_loss - _LOSS_FLOOR
            parts = [f"  step {step:>6d} | loss={avg_loss:.4f}  gap={gap:.2f}"]
            if loss_self > 0:
                parts.append(f"self={loss_self:.6f}")
            parts.append(f"lr={current_lr:.2e} | {tok_s:.0f} tok/s")
            tqdm.write(" | ".join(parts))
            running_loss = 0.0
            start_time = time.time()

        if save_every > 0 and step % save_every == 0 and step > 0:
            ckpt_dir = os.path.join(output_dir, f"step_{step:06d}")
            save_cog_checkpoint(params, ckpt_dir, step, self_state=self_state)

        pbar.update(1)

    pbar.close()
    save_cog_checkpoint(params, output_dir, steps, self_state=self_state)
    print(f"[COG] Training complete → {output_dir}/")


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def _write_bin_header(buf, M, d, n_layers=1, cb_type=0):
    """Append LCM binary codebook header to bytearray."""
    import struct
    magic = b"LCM_CB"
    buf.extend(magic)
    for v in [2, M, d, n_layers, cb_type, 0, 0, 0]:
        buf.extend(struct.pack("<I", v))


def _to_np(x):
    """Convert jax array → numpy, no-op if already numpy."""
    import numpy as _np
    return _np.array(x) if hasattr(x, 'numpy') else x


def save_cog_checkpoint(params, output_dir, step, self_state=None):
    """Save full checkpoint + export codebooks + W_out for C engine."""
    import json, os, pickle, struct
    import numpy as _np
    from train.gvalue import make_global_value_vectors, GValueCodebook

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "codebooks"), exist_ok=True)

    ckpt = jax.tree_util.tree_map(_to_np, params)
    with open(os.path.join(output_dir, "cog_params.pkl"), "wb") as f:
        pickle.dump({'params': ckpt, 'step': step, 'self_state': self_state}, f)

    # ── C推理引擎输出格式 ──────────────────────────────────────────────

    def _simvq_cb(simvq):
        return _to_np(simvq['A']) @ _to_np(simvq['W'])

    d = _to_np(params['W_out']).shape[0]
    V = _to_np(params['W_out']).shape[1]

    # config.json
    enc = params.get('encoder', {})
    cfg = {
        'd_model': d, 'vocab_size': V, 'max_seq_len': 512, 'n_heads': 4,
        'n_encoder_layers': len(enc.get('layers', [])) if enc else 2,
        'n_lattices': 6,
        'd_ff': int(1.5 * d),
        'M_top': _to_np(params['hrq']['top']['A']).shape[0],
        'M_fine': _to_np(params['hrq']['fine'][0]['A']).shape[0],
        'n_hrq_layers': len(params['hrq']['fine']),
        'M_sparse': _to_np(params['sparse']['C']).shape[0],
        'M_lr': _to_np(params['lowrank']['A_V']).shape[0],
        'M_man': _to_np(params['manifold']['C']).shape[0],
        'M_bind': _to_np(params['binding']['key_cb'][0]['A']).shape[0],
        'M_contrast': _to_np(params['contrast']['C_a'][0]['A']).shape[0],
        'n_bind_layers': len(params['binding']['key_cb']),
        'n_contrast_layers': len(params['contrast']['C_a']),
        'n_lr_layers': 3, 'r_max': 8, 't_dim': 4,
        'n_value_pairs': 4, 'M_danger': 256,
        'n_self_codes': _to_np(params['self']['modes']).shape[0],
        'max_inference_steps': 32, 'convergence_tol': 1e-3,
        'entropy_threshold': 0.5,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    # encoder.bin
    if enc:
        parts = [_to_np(enc['embed']).ravel(), _to_np(enc['rel_bias']).ravel()]
        for layer in enc['layers']:
            for k in ['ln1_scale','ln1_bias','w_q','w_k','w_v','w_o',
                       'ln2_scale','ln2_bias','w_1','w_2','w_3']:
                parts.append(_to_np(layer[k]).ravel())
        parts.append(_to_np(enc['q_pool']).ravel())
        parts.append(_to_np(enc['w_proj']).ravel())
        _np.concatenate(parts).astype(_np.float32).tofile(
            os.path.join(output_dir, "encoder.bin"))
    else:
        # dummy encoder (random) — exists for compatibility
        _np.random.seed(0)
        dummy = _np.random.randn(1).astype(_np.float32)
        dummy.tofile(os.path.join(output_dir, "encoder.bin"))

    # decoder.bin (new format: gen_head)
    gh = params.get('gen_head', {})
    if gh:
        parts = [_to_np(gh['w_embed']).ravel()]
        for k in ['w_q','w_k','w_v','w_o']:
            parts.append(_to_np(gh[k]).ravel())
        parts.append(_to_np(gh['w_1']).ravel())
        parts.append(_to_np(gh['w_2']).ravel())
        parts.append(_to_np(gh['w_3']).ravel())
        dec = _np.concatenate(parts).astype(_np.float32)
    else:
        dec = _to_np(params['W_out']).copy()  # fallback
    dec.tofile(os.path.join(output_dir, "decoder.bin"))

    # 导出所有 codebook .bin 文件（带 LCM_CB 头部）
    codebooks_dir = os.path.join(output_dir, "codebooks")
    cb_entries = []

    # HRQ
    cb_top = _simvq_cb(params['hrq']['top'])
    _write_cb_bin(codebooks_dir, "hrq_codebook.bin", cb_top, 10)
    for i, fb in enumerate(params['hrq']['fine']):
        _write_cb_bin(codebooks_dir, f"hrq_fine_{i}.bin", _simvq_cb(fb), 10)

    # Sparse
    _write_cb_bin(codebooks_dir, "sparse_codebook.bin", _to_np(params['sparse']['C']), 11)

    # LowRank
    lr = params['lowrank']
    V_lr = _to_np(lr['A_V']) @ _to_np(lr['W_V'])
    lr_list = []
    for u_k in lr['U']:
        u = _to_np(u_k)
        lr_list.append(u @ V_lr[:, :u.shape[-1]].T)
    lr_flat = _np.concatenate([m.ravel() for m in lr_list])
    lr_flat.tofile(os.path.join(codebooks_dir, "lowrank_codebook.bin"))

    # Manifold
    _write_cb_bin(codebooks_dir, "manifold_codebook.bin", _to_np(params['manifold']['C']), 13)

    # Binding
    for prefix, key in [('key','key_cb'), ('val','val_cb'), ('bind','bind_cb')]:
        for i, cb in enumerate(params['binding'][key]):
            _write_cb_bin(codebooks_dir, f"binding_{prefix}_{i}.bin", _simvq_cb(cb),
                          14 if prefix == 'key' else 15 if prefix == 'val' else 16)

    # Contrast
    for prefix, key in [('a','C_a'), ('b','C_b')]:
        for i, cb in enumerate(params['contrast'][key]):
            _write_cb_bin(codebooks_dir, f"contrast_{prefix}_{i}.bin", _simvq_cb(cb), 17)

    # tokenizer.json — 从 data/ 复制
    import shutil
    for cand in ['data/tokenizer.json', '../data/tokenizer.json']:
        if os.path.exists(cand):
            shutil.copy2(cand, os.path.join(output_dir, "tokenizer.json"))
            break

    # gvalue codebooks
    try:
        C_pos, C_neg = make_global_value_vectors(d)
        GValueCodebook(C_pos, C_neg).save(output_dir)
    except Exception:
        pass

    # 统计大小
    total_bytes = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.bin') or f.endswith('.json') or f.endswith('.pkl'):
                total_bytes += os.path.getsize(os.path.join(root, f))
    print(f"[CKPT] Step {step}: inference format → {output_dir}/ ({total_bytes/1e6:.0f} MB)")


def _write_cb_bin(dir_path, filename, mat, cb_type):
    """Write numpy matrix as LCM binary codebook file with header."""
    import struct
    buf = bytearray(36)
    M, d = mat.shape
    buf[0:6] = b"LCM_CB"
    struct.pack_into("<I", buf, 6, 2)    # version
    struct.pack_into("<I", buf, 10, M)    # n_codes
    struct.pack_into("<I", buf, 14, d)    # dim
    struct.pack_into("<I", buf, 18, 1)    # n_layers
    buf[22] = cb_type
    buf[23] = 0
    struct.pack_into("<I", buf, 24, 0)    # c
    crc = sum(buf[:28]) & 0xFFFFFFFF
    struct.pack_into("<I", buf, 28, crc)
    struct.pack_into("<I", buf, 32, 0)    # reserved
    path = os.path.join(dir_path, filename)
    with open(path, "wb") as f:
        f.write(buf)
        mat.astype(_np.float32).tofile(f)


def load_cog_checkpoint(path, d_model=None, n_self_codes=64):
    """Load full cognitive training checkpoint.

    Args:
        path: Path to checkpoint .pkl file.
        d_model: Model dimension (for re-init self_state if not saved).
        n_self_codes: Number of self modes.

    Returns:
        params, step, self_state
    """
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    params = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
        ckpt['params'])
    self_state = ckpt.get('self_state')
    if self_state is None and d_model is not None:
        self_state = init_self_state(n_self_codes, d_model)
    print(f"[COG] Loaded checkpoint step {ckpt.get('step', '?')}")
    return params, ckpt.get('step', 0), self_state
