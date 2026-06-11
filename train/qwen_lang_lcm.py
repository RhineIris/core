"""Qwen-backed Language LCM — frozen embed + LM head, trainable bridge + decoder.

Architecture:
  Qwen embed (frozen 151936x896) → proj_down (896→256, trainable)
    → 8L decoder (d=256, mHC, trainable) → proj_up (256→896, trainable)
    → RMSNorm (frozen) → Qwen LM head (frozen 151936x896)

The bridge (proj_down + proj_up) lets us use Qwen's vocabulary knowledge
with our lightweight d=256 decoder, fitting in 4GB VRAM.
"""
import jax
import jax.numpy as jnp
import numpy as np
import dataclasses as _dc

from train.lang_lcm import (
    decoder_layer_forward, init_hc_params, _n_heads
)
from train.config import LCMConfig


# ─── Load pretrained components ──────────────────────────────────────────────

def load_qwen_embed(path="checkpoints/qwen_model/qwen_params.npz"):
    """Load frozen Qwen embedding and LM head weights."""
    data = np.load(path)
    params = {
        'embed': jnp.array(data['model.embed_tokens.weight']),
        'lm_head': jnp.array(data.get('lm_head.weight',
                            data['model.embed_tokens.weight'])),
        'norm_weight': jnp.array(data['model.norm.weight']),
    }
    print(f"[QWEN] Loaded: embed {params['embed'].shape}, "
          f"vocab_size={params['embed'].shape[0]}")
    return params


# ─── Init ────────────────────────────────────────────────────────────────────

def init_qwen_lang_lcm(rng, cfg, qwen_embed):
    """Initialize with frozen Qwen embed/LM head + trainable bridge + decoder.

    Decoder runs at d=256 (cfg.d_model).  Bridge projectors handle
    the dimension mismatch with Qwen's 896-d space.
    """
    qwen_d = qwen_embed['embed'].shape[1]  # 896
    V = qwen_embed['embed'].shape[0]       # 151936
    d = cfg.d_model  # 256 (our decoder dim)

    keys = jax.random.split(rng, 12)
    n_layers = getattr(cfg, 'n_lang_layers', 8)
    n_hc = getattr(cfg, 'n_hc', 1)

    params = {}

    # Frozen Qwen embed + LM head
    params['qwen_base'] = qwen_embed

    # Bridge: Qwen 896-d ↔ decoder 256-d
    params['proj_down'] = jax.random.normal(keys[0], (d, qwen_d)) * (qwen_d ** -0.5)
    params['proj_up'] = jax.random.normal(keys[1], (qwen_d, d)) * (d ** -0.5)

    # Positional embedding (at decoder dimension d=256)
    max_len = 512
    params['pos_embed'] = jax.random.normal(keys[2], (max_len, d)) * (d ** -0.5)

    # Decoder layers (d=256, same as before)
    params['decoder'] = []
    d_ff = int(d * 1.5)
    for l in range(n_layers):
        kl = jax.random.split(keys[7], 8)
        layer = {
            'w_q': jax.random.normal(kl[0], (d, d)) * (d ** -0.5),
            'w_k': jax.random.normal(kl[1], (d, d)) * (d ** -0.5),
            'w_v': jax.random.normal(kl[2], (d, d)) * (d ** -0.5),
            'w_o': jax.random.normal(kl[3], (d, d)) * (d ** -0.5),
            'ln1_scale': jnp.ones(d), 'ln1_bias': jnp.zeros(d),
            'w_gate': jax.random.normal(kl[4], (d, d_ff)) * (d ** -0.5),
            'w_up': jax.random.normal(kl[5], (d, d_ff)) * (d ** -0.5),
            'w_down': jax.random.normal(kl[6], (d_ff, d)) * ((d_ff) ** -0.5),
            'ln2_scale': jnp.ones(d), 'ln2_bias': jnp.zeros(d),
        }
        params['decoder'].append(layer)

    # mHC
    if n_hc > 1:
        rng, hc_rng = jax.random.split(keys[9])
        params['hc'] = init_hc_params(hc_rng, d, n_hc, n_layers)
    else:
        params['hc'] = None

    # z_q projection: (B, 256) cognitive state → decoder space (B, d)
    params['z_proj'] = jax.random.normal(keys[10], (d, 256)) * (d ** -0.5)

    return params


# ─── Forward ──────────────────────────────────────────────────────────────────

def qwen_lang_lcm_forward(params, x, cfg, rng=None, training=True,
                           dropout_rate=0.2, z_q=None):
    """Forward: Qwen embed → proj_down → decoder → proj_up → Qwen LM head."""
    B, N = x.shape
    d = cfg.d_model  # 256
    n_hc = getattr(cfg, 'n_hc', 1)

    # 1. Qwen embedding (frozen, 896-d)
    h = params['qwen_base']['embed'][x]  # (B, N, 896)

    # 2. Bridge down: 896 → 256
    h = h @ params['proj_down'].T  # (B, N, 256)

    # 3. Inject z_q at position 0 (in decoder space)
    if z_q is not None:
        z_injected = z_q @ params['z_proj'].T  # (B, 256)
        h = h.at[:, 0, :].set(z_injected)

    # 4. Positional embedding (at 256-d)
    h = h + params['pos_embed'][jnp.arange(N)]

    # 5. Expand for mHC
    if n_hc > 1:
        h = jnp.broadcast_to(h[:, :, None, :], (B, N, n_hc, d))

    n_heads = _n_heads(cfg)
    hc_p = params.get('hc', None)
    sinkhorn_iters = getattr(cfg, 'hc_sinkhorn_iters', 5)

    # 6. Decoder layers (256-d, trainable)
    for i, lp in enumerate(params['decoder']):
        if training and dropout_rate > 0.0 and rng is not None:
            rng, dr = jax.random.split(rng)
        else:
            dr = None
        lhc = hc_p[i] if hc_p else None
        h = decoder_layer_forward(
            h, lp, N, n_heads=n_heads,
            training=training, dropout_rng=dr,
            dropout_rate=dropout_rate,
            n_hc=n_hc, hc_params=lhc,
            sinkhorn_iters=sinkhorn_iters,
            cb_entries=None, tau_cb=0.1)

    # 7. Collapse mHC
    if n_hc > 1:
        h = h.mean(axis=2)

    # 8. Bridge up: 256 → 896
    h = h @ params['proj_up'].T  # (B, N, 896)

    # 9. Final RMSNorm (frozen Qwen)
    h = h * params['qwen_base']['norm_weight'] / jnp.sqrt(
        jnp.mean(h**2, axis=-1, keepdims=True) + 1e-6)

    # 10. LM head (frozen Qwen)
    logits = h @ params['qwen_base']['lm_head'].T

    return logits, h, {}


# ─── Generation ──────────────────────────────────────────────────────────────

@jax.jit
def _gen_forward(params, x, cfg, rng):
    return qwen_lang_lcm_forward(params, x, cfg, rng=rng, training=False, dropout_rate=0.0)


def qwen_lang_generate(params, prompt, max_len, bos_id, eos_id, rng, cfg):
    prompt_len = len(prompt)
    total = prompt_len + max_len
    tokens = list(prompt)
    x = jnp.zeros((1, total), dtype=jnp.int32)
    x = x.at[0, :prompt_len].set(jnp.array(prompt))
    _ = _gen_forward(params, x, cfg, rng)
    pos = prompt_len
    for _ in range(max_len):
        logits, _, _ = _gen_forward(params, x, cfg, rng)
        nxt = int(jax.random.categorical(jax.random.split(rng)[0], logits[0, pos-1, :]))
        tokens.append(nxt)
        if nxt in (eos_id, 0): break
        x = x.at[0, pos].set(nxt)
        pos += 1
    return tokens


# ─── Sanity check ───────────────────────────────────────────────────────────

def sanity_check():
    from train.config import LCMConfig
    cfg = LCMConfig()
    cfg = _dc.replace(cfg, n_hc=1, n_mtp_depth=1, n_lang_layers=4)
    qe = load_qwen_embed('checkpoints/qwen_model/qwen_params.npz')
    rng = jax.random.PRNGKey(0)
    params = init_qwen_lang_lcm(rng, cfg, qe)
    x = jnp.zeros((2, 4), dtype=jnp.int32)
    logits, h, aux = qwen_lang_lcm_forward(params, x, cfg, rng=rng)
    print(f'Forward OK: logits {logits.shape}, h {h.shape}')
    print(f'Decoder: {len(params["decoder"])} layers, d={cfg.d_model}')
    # Test z_q injection
    z_q = jnp.ones((2, 256))
    logits2, _, _ = qwen_lang_lcm_forward(params, x, cfg, rng=rng, z_q=z_q)
    print(f'z_q injection OK: logits {logits2.shape}')
    print('Sanity check PASSED!')
    return True


if __name__ == "__main__":
    sanity_check()
