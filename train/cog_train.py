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
import time
from functools import partial

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
from train.cog_loop import cog_loop_scan


# ─── Init full params ───────────────────────────────────────────────────────

def init_cog_params(cfg, rng, lm_ckpt=None):
    """Initialize all trainable params.

    Passive channel: W_out (trained from scratch).
    Active channel: gen_head + w_start (loadable from Stage 1 LM checkpoint).
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

    return params


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
    """Create jitted training step with dual-channel output.

    Every macro step's z_q feeds both channels:
      - Passive (introspection):  z_q @ W_out → single-token CE
      - Active (expression):      gen_head(z_q) → full-sequence CE

    The passive loss is weak but transparent; the active loss is strong but
    goes through the skill network. Their combination ensures the cognitive
    state is both linguistically meaningful and fluently expressible.
    """

    @partial(jax.jit, static_argnums=(4,))
    def train_step(params, opt_state, batch, lr, step):
        inputs, targets = batch
        B, N = inputs.shape

        def loss_fn(p):
            z = encoder_forward(p['encoder'], inputs, cfg.n_heads)
            z_conscious = jnp.mean(z.reshape(-1, z.shape[-1]), axis=0)

            codebooks = pack_codebooks_for_c(p)
            z_qs, diffs, entropies = cog_loop_scan(
                z_conscious, codebooks,
                max_steps=cfg.max_inference_steps,
                thresholds=None,
                tau=0.1)
            # z_qs: (max_steps, d), diffs: (max_steps,), entropies: (max_steps,)

            # Passive channel: transparent introspection at every macro step
            passive_losses = []
            for i in range(cfg.max_inference_steps):
                p_logits = z_qs[i] @ p['W_out']  # (V,) — cheap matmul
                passive_losses.append(passive_loss(p_logits, targets[0, 0]))
            step_weights = jnp.arange(
                1, cfg.max_inference_steps + 1, dtype=jnp.float32)
            p_loss = jnp.sum(jnp.array(passive_losses) * step_weights) / jnp.sum(step_weights)

            # Active channel: fluent language skill at final step only (GPU memory)
            a_logits = decoder_forward(
                p['gen_head'], z_qs[-1], inputs, p['w_start'])  # (B, N, V)
            a_loss = active_loss(a_logits, targets)

            # Combined loss
            loss = p_loss + a_loss

            # Convergence bonus: reward efficient thinking
            converged = (diffs[-1] < cfg.convergence_tol) & (entropies[-1] < cfg.entropy_threshold)
            n_steps = jnp.argmax(diffs < cfg.convergence_tol) + 1
            loss = loss + jnp.where(converged,
                                    -0.001 * jnp.log(n_steps + 1e-8), 0.0)

            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss

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
    params = init_cog_params(cfg, init_rng, lm_ckpt=lm_ckpt)

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
    print(f"[COG] Total params: {total_params:,}")
    print(f"[COG] Steps: {steps}, B={batch_size}, N={seq_len}, lr={lr}")
    print()

    running_loss = 0.0
    start_time = time.time()
    pbar = tqdm(total=steps, desc="cog training", unit="step")

    for step in range(steps):
        batch = next(data_iter)
        current_lr = schedule(step)

        params, opt_state, loss_val = train_step(
            params, opt_state, batch, current_lr, step)

        running_loss += float(loss_val)

        if step % log_every == 0 and step > 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - start_time
            tok_s = batch_size * seq_len * log_every / elapsed
            tqdm.write(f"  step {step:>6d} | loss={avg_loss:.4f} | "
                       f"lr={current_lr:.2e} | {tok_s:.0f} tok/s")
            running_loss = 0.0
            start_time = time.time()

        if save_every > 0 and step % save_every == 0 and step > 0:
            ckpt_dir = os.path.join(output_dir, f"step_{step:06d}")
            save_cog_checkpoint(params, ckpt_dir, step)

        pbar.update(1)

    pbar.close()
    save_cog_checkpoint(params, output_dir, steps)
    print(f"[COG] Training complete → {output_dir}/")


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def save_cog_checkpoint(params, output_dir, step):
    """Save full checkpoint + export codebooks + W_out for C engine."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "codebooks"), exist_ok=True)

    ckpt = jax.tree_util.tree_map(lambda x: np.array(x), params)
    with open(os.path.join(output_dir, "cog_params.pkl"), "wb") as f:
        pickle.dump({'params': ckpt, 'step': step}, f)

    # Export flat codebook matrices as .bin
    codebooks_flat = pack_codebooks_for_c(params)
    bin_dir = os.path.join(output_dir, "codebooks")
    for i, cb in enumerate(codebooks_flat):
        np.array(cb, dtype=np.float32).tofile(os.path.join(bin_dir, f"codebook_{i}.bin"))
    np.array(params['W_out'], dtype=np.float32).tofile(
        os.path.join(bin_dir, "W_out.bin"))

    size_mb = sum(os.path.getsize(os.path.join(bin_dir, f))
                  for f in os.listdir(bin_dir)) / 1e6
    print(f"[CKPT] Step {step}: codebooks + W_out ({size_mb:.1f} MB) → {bin_dir}/")


def load_cog_checkpoint(path):
    """Load full cognitive training checkpoint."""
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    params = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
        ckpt['params'])
    print(f"[COG] Loaded checkpoint step {ckpt.get('step', '?')}")
    return params, ckpt.get('step', 0)
