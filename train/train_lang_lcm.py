"""Stage 1: Language LCM standalone training.

Trains a complete LCM instance (encoder + 6 codebooks + fusion + W_out)
as a pure language model. Codebooks learn semantic-syntactic primitives
through next-token prediction CE loss.

Architecture:
  tokens → embed → causal encoder → codebook retrieval+fuse → W_out → logits

Key differences from the old train_lm.py (gen_head):
  - Uses codebook-based retrieval instead of linear attention + GLU decoder
  - Codebooks store language primitives (sentence skeletons, collocations, etc.)
  - Shares the same LCM architecture as the Cognitive LCM for future integration

Usage:
    python -m train.train_lang_lcm --lr 3e-4 --steps 100000 --save-every 5000

Checkpoint format:
    {'lang_params': ..., 'opt_state': ..., 'step': ..., 'cfg': ...}
"""
import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from train.config import LCMConfig
from train.data import WikiDataIter, MMAP_PATH, MMAP_SHAPE_PATH
from train.lang_lcm import init_lang_lcm_params, lang_lcm_forward


# ─── Loss ─────────────────────────────────────────────────────────────────────

def lang_lm_loss(logits, targets):
    """Cross-entropy language modeling loss.

    Args:
        logits: (B, N, V).
        targets: (B, N) integer token IDs.

    Returns:
        Scalar loss.
    """
    B, N, V = logits.shape
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, V), targets.reshape(-1)).mean()


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(params, opt_state, step, path):
    """Save Language LCM checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    ckpt = jax.tree_util.tree_map(lambda x: np.array(x), params)
    if opt_state is not None:
        ckpt_opt = jax.tree_util.tree_map(
            lambda x: np.array(x) if hasattr(x, 'numpy') else x, opt_state)
    else:
        ckpt_opt = None
    data = {
        'lang_params': ckpt,
        'opt_state': ckpt_opt,
        'step': step,
        'd_model': jnp.array(params['W_out'].shape[0]),
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    size_mb = os.path.getsize(path) / 1e6
    n_saved = sum(p.size for p in jax.tree_util.tree_leaves(ckpt) if hasattr(p, 'size'))
    print(f"[CKPT] Step {step}: saved {path} ({size_mb:.1f} MB, {n_saved:,} params)")


def load_checkpoint(path):
    """Load Language LCM checkpoint.

    Args:
        path: Path to .pkl checkpoint.

    Returns:
        params, step
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    params = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
        data['lang_params'])
    step = data.get('step', 0)
    print(f"[CKPT] Loaded {path} (step {step})")
    return params, step


# ─── Training loop ────────────────────────────────────────────────────────────

def train_lang_lcm(cfg, output_dir, steps=100000, lr=3e-4, batch_size=16,
                   seq_len=512, log_every=100, save_every=5000,
                   from_ckpt=None, data_path=None, shape_path=None):
    """Run Language LCM training.

    Args:
        cfg: LCMConfig.
        output_dir: Output directory for checkpoints.
        steps: Total training steps.
        lr: Learning rate.
        batch_size: Batch size.
        seq_len: Sequence length.
        log_every: Logging interval.
        save_every: Checkpoint save interval.
        from_ckpt: Resume from checkpoint path.
        data_path: Path to .dat mmap file.
        shape_path: Path to shape JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(42)

    # Init params
    rng, init_rng = jax.random.split(rng)
    d = cfg.d_model
    step_offset = 0
    if from_ckpt:
        params, step_offset = load_checkpoint(from_ckpt)
        # ── Strip old codebook-related keys (not used in pure-transformer) ──
        params.pop('codebook_entries', None)
        for layer in params.get('decoder', []):
            layer.pop('cb_read', None)
            layer.pop('ln3_scale', None)
            layer.pop('ln3_bias', None)
        # ── Add pos_embed if missing (old checkpoints trained without it) ──
        if 'pos_embed' not in params:
            max_len = getattr(cfg, 'max_seq_len', 512)
            # Initialize with small random values (same scale as init_lang_lcm_params)
            d_ckpt = params.get('W_out', {}).shape[0] if hasattr(params.get('W_out'), 'shape') else d
            rng, pe_rng = jax.random.split(rng)
            params['pos_embed'] = jax.random.normal(pe_rng, (max_len, d_ckpt)) * (d_ckpt ** -0.5)
            print(f"[CKPT]  Added pos_embed ({max_len}, {d_ckpt}) — old checkpoint had none")
    else:
        params = init_lang_lcm_params(init_rng, cfg)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params)
                   if hasattr(p, 'size'))
    rng_init = jax.random.PRNGKey(0)
    _test_x = jnp.zeros((1, 4), dtype=jnp.int32)
    _logits, _z_qs, _aux = lang_lcm_forward(params, _test_x, cfg, rng=rng_init)
    print(f"[LANG] Language LCM — {n_params:,} params")
    print(f"[LANG] Output: {output_dir}")
    print(f"[LANG] Steps: {steps}, B={batch_size}, N={seq_len}, lr={lr}")

    # Open log file for loss history (avoids tqdm write-overwrite issue)
    log_file_path = os.path.join(output_dir, "training_log.txt")
    _log_file = open(log_file_path, "w", buffering=1)
    _log_file.write(f"# Language LCM training log\n")
    _log_file.write(f"# output_dir={output_dir} steps={steps} lr={lr} "
                    f"batch={batch_size} seq={seq_len}\n")
    _log_file.write(f"# step_offset={step_offset}\n")
    _log_file.write(f"# step,loss,ppl,lr\n")
    _log_file.flush()

    # Optimizer
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps, alpha=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=cfg.adam_beta1,
                     b2=cfg.adam_beta2, eps=cfg.adam_eps,
                     weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(params)
    if from_ckpt and step_offset > 0:
        # Try to load optimizer state from checkpoint.
        # IMPORTANT: If the model structure changed (e.g. pos_embed added),
        # the old optimizer state has mismatched keys — we detect this by
        # comparing key sets and fall back to fresh optimizer init.
        with open(from_ckpt, 'rb') as f:
            data = pickle.load(f)
        old_opt = data.get('opt_state')
        if old_opt is not None:
            try:
                old_opt_jax = jax.tree_util.tree_map(
                    lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
                    old_opt)
                # Verify key compatibility by attempting a dummy update
                dummy_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
                _ = optimizer.update(dummy_grads, old_opt_jax, params)
                opt_state = old_opt_jax
                print(f"[CKPT]  Loaded optimizer state from checkpoint")
            except (ValueError, KeyError, TypeError) as e:
                print(f"[CKPT]  Optimizer state incompatible ({e}), re-initialized")
                opt_state = optimizer.init(params)

    # Data iterator
    mp = data_path or MMAP_PATH
    sp = shape_path or (mp.replace('.dat', '_shape.json'))
    data_iter = WikiDataIter(mmap_path=mp, shape_path=sp, B=batch_size, N=seq_len)

    # JIT-compiled training step (captures optimizer from enclosing scope)
    @jax.jit
    def train_step(p, opt, batch, lr_val, rng_key):
        inputs, targets = batch

        def loss_fn(pp):
            logits, _, _ = lang_lcm_forward(pp, inputs, cfg, rng=rng_key, training=True)
            return lang_lm_loss(logits, targets)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_opt = optimizer.update(grads, opt, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt, loss

    # Training loop
    total_steps = steps + step_offset
    running_loss = 0.0
    start_time = time.time()
    pbar = tqdm(total=steps, desc=" lang lcm training", unit="step",
                initial=step_offset)

    for global_step in range(step_offset, total_steps):
        batch = next(data_iter)
        current_lr = schedule(global_step - step_offset)
        rng, step_rng = jax.random.split(rng)

        params, opt_state, loss_val = train_step(
            params, opt_state, batch, current_lr, step_rng)

        loss_f = float(loss_val)
        if np.isnan(loss_f) or np.isinf(loss_f):
            print(f"\n[LANG] NaN at step {global_step}, skipping...")
            continue

        running_loss += loss_f

        # Logging (to both stderr and log file to avoid tqdm \r overwrite)
        steps_this_run = global_step - step_offset
        if steps_this_run % log_every == 0 and steps_this_run > 0:
            n_steps = min(log_every, steps_this_run)  # first log may have <100 steps
            avg_loss = running_loss / n_steps
            elapsed = time.time() - start_time
            tok_s = batch_size * seq_len * log_every / elapsed
            ppl = np.exp(min(avg_loss, 20.0))  # cap ppl to avoid overflow
            msg = (f"  step {global_step:>6d} | loss={avg_loss:.4f} | "
                   f"ppl={ppl:.1f} | lr={current_lr:.2e} | {tok_s:.0f} tok/s")
            print(f"\r{msg}", flush=True)
            _log_file.write(f"{global_step},{avg_loss:.6f},{ppl:.1f},{current_lr:.2e}\n")
            _log_file.flush()
            running_loss = 0.0
            start_time = time.time()

        # Checkpoint
        if save_every > 0 and (global_step + 1) % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"lang_step_{global_step + 1}.pkl")
            save_checkpoint(params, opt_state, global_step + 1, ckpt_path)

        pbar.update(1)

    pbar.close()

    # Final save
    final_path = os.path.join(output_dir, "lang_final.pkl")
    save_checkpoint(params, opt_state, total_steps, final_path)
    print(f"[LANG] Training complete → {final_path}")

    # Also save W_out for C-inference compatibility
    w_out = np.array(params['W_out'])
    w_out.tofile(os.path.join(output_dir, "W_out.bin"))
    print(f"[LANG] W_out exported to {output_dir}/W_out.bin")

    # Close log file
    _log_file.close()
    print(f"[LANG] Training log saved to {log_file_path}")

    return params


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Language LCM Training")
    parser.add_argument("--output-dir", default="checkpoints/lang_lm",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Logging interval")
    parser.add_argument("--save-every", type=int, default=10000,
                        help="Checkpoint save interval")
    parser.add_argument("--from-ckpt", default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--data", default=None,
                        help="Path to .dat mmap file")
    parser.add_argument("--shape", default=None,
                        help="Path to shape JSON")
    args = parser.parse_args()

    cfg = LCMConfig()
    train_lang_lcm(
        cfg=cfg,
        output_dir=args.output_dir,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        log_every=args.log_every,
        save_every=args.save_every,
        from_ckpt=args.from_ckpt,
        data_path=args.data,
        shape_path=args.shape,
    )


if __name__ == "__main__":
    main()
