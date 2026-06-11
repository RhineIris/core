"""Qwen-backed Language LCM — frozen embedding + LM head + trainable decoder.

Uses Qwen2.5-0.5B's pretrained embedding and output projection (frozen).
Between them, our lightweight 8-layer transformer decoder processes the sequence.
This gives the model Qwen's Chinese language knowledge without running 24 layers.

Architecture:
  input_ids → frozen Qwen embed (151936, 896)
            → trainable 8-layer decoder (d=896, with mHC + codebook)
            → frozen Qwen LM head (pre-trained, weight-tied)

z_q injection: z_q (256-d) → projection (896, 256) → replaces first position.
"""
import jax
import jax.numpy as jnp
import numpy as np

from train.lang_lcm import (
    decoder_layer_forward, init_hc_params, _n_heads, _causal_mask,
    _codebook_soft_read, layer_norm
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
    print(f"[QWEN-EMBED] Loaded: embed {params['embed'].shape}, "
          f"vocab_size={params['embed'].shape[0]}")
    return params


# ─── Init full Language LCM with Qwen embed ─────────────────────────────────

def init_qwen_lang_lcm(rng, cfg, qwen_embed):
    """Initialize Language LCM with frozen Qwen embed + LM head.

    Only the decoder + codebook params are trainable.
    qwen_embed is stored as 'qwen_base' (frozen via stop_gradient).

    Note: cfg is overridden with Qwen's d_model (896) and vocab_size (151936).
    """
    import dataclasses as _dc
    d = qwen_embed['embed'].shape[1]  # 896
    V = qwen_embed['embed'].shape[0]  # 151936
    cfg = _dc.replace(cfg, d_model=d, vocab_size=V,
                       n_lang_layers=getattr(cfg, 'n_lang_layers', 8))

    keys = jax.random.split(rng, 10)
    n_layers = cfg.n_lang_layers
    n_hc = cfg.n_hc
    H = _n_heads(cfg)

    params = {}

    # Frozen Qwen embed + LM head
    params['qwen_base'] = qwen_embed

    # Positional embedding for 896-d
    max_len = 512
    params['pos_embed'] = jax.random.normal(keys[1], (max_len, d)) * (d ** -0.5)

    # Decoder layers (d=896, deeper FFN)
    params['decoder'] = []
    d_ff = int(d * 1.5)
    for l in range(n_layers):
        kl = jax.random.split(keys[7], 10)
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
            'cb_read': None,  # codebook read disabled for now
        }
        params['decoder'].append(layer)

    # mHC (optional)
    if n_hc > 1:
        rng, hc_rng = jax.random.split(keys[9])
        params['hc'] = init_hc_params(hc_rng, d, n_hc, n_layers)
    else:
        params['hc'] = None

    # z_q projection: (256, 896) — map cognitive state → decoder space
    params['z_proj'] = jax.random.normal(keys[8], (d, 256)) * (0.1)

    return params, cfg


# ─── Forward pass ────────────────────────────────────────────────────────────

def qwen_lang_lcm_forward(params, x, cfg, rng=None, training=True,
                           dropout_rate=0.2, z_q=None):
    """Forward pass: Qwen embed → trainable decoder → Qwen LM head.

    z_q is injected at position 0 via z_proj projection.
    Qwen embed + LM head are frozen (caller applies stop_gradient).
    """
    B, N = x.shape
    d = cfg.d_model
    n_hc = getattr(cfg, 'n_hc', 1)

    # Token embedding (frozen Qwen)
    h = params['qwen_base']['embed'][x]  # (B, N, 896)

    # Inject z_q at position 0
    if z_q is not None:
        z_projected = z_q @ params['z_proj'].T  # (B, 896)
        h = h.at[:, 0, :].set(z_projected)

    # Positional embedding (trainable)
    pos_indices = jnp.arange(N, dtype=jnp.int32)
    h = h + params['pos_embed'][pos_indices]

    # Expand for mHC
    if n_hc > 1:
        h = jnp.broadcast_to(h[:, :, None, :], (B, N, n_hc, d))

    n_heads = _n_heads(cfg)
    hc_params_list = params.get('hc', None)
    sinkhorn_iters = getattr(cfg, 'hc_sinkhorn_iters', 5)

    # Decoder layers (trainable)
    for i, layer_params in enumerate(params['decoder']):
        if training and dropout_rate > 0.0 and rng is not None:
            rng, do_rng = jax.random.split(rng)
        else:
            do_rng = None
        l_hc = hc_params_list[i] if hc_params_list is not None else None
        h = decoder_layer_forward(
            h, layer_params, N, n_heads=n_heads,
            training=training, dropout_rng=do_rng,
            dropout_rate=dropout_rate,
            n_hc=n_hc, hc_params=l_hc,
            sinkhorn_iters=sinkhorn_iters,
            cb_entries=None, tau_cb=0.1)

    # Collapse mHC streams
    if n_hc > 1:
        h = h.mean(axis=2)

    # Final RMSNorm (from Qwen, frozen)
    h = h * params['qwen_base']['norm_weight'] / jnp.sqrt(jnp.mean(h**2, axis=-1, keepdims=True) + 1e-6)

    # LM head (from Qwen, frozen)
    logits = h @ params['qwen_base']['lm_head'].T

    aux = {}
    return logits, h, aux


# ─── Generation ──────────────────────────────────────────────────────────────

@jax.jit
def _gen_forward(params, x, cfg, rng):
    return qwen_lang_lcm_forward(params, x, cfg, rng=rng, training=False, dropout_rate=0.0)


def qwen_lang_generate(params, prompt, max_len, bos_id, eos_id, rng, cfg):
    from tokenizers import Tokenizer
    # Use the original LCM tokenizer (not Qwen's)
    prompt_len = len(prompt)
    total = prompt_len + max_len
    tokens_list = list(prompt)
    x = jnp.zeros((1, total), dtype=jnp.int32)
    x = x.at[0, :prompt_len].set(jnp.array(prompt))
    _ = _gen_forward(params, x, cfg, rng)
    pos = prompt_len
    for _ in range(max_len):
        logits, _, _ = _gen_forward(params, x, cfg, rng)
        nxt = int(jax.random.categorical(jax.random.split(rng)[0], logits[0, pos-1, :]))
        tokens_list.append(nxt)
        if nxt in (eos_id, 0): break
        x = x.at[0, pos].set(nxt)
        pos += 1
    return tokens_list


# ─── Config override ─────────────────────────────────────────────────────────

QWEN_EMBED_CFG = {
    'd_model': 896,
    'vocab_size': 151936,
    'n_heads': 8,
}


# ─── Sanity check ───────────────────────────────────────────────────────────

def sanity_check():
    from train.config import LCMConfig
    import dataclasses as _dc
    cfg = LCMConfig()
    cfg = _dc.replace(cfg, n_hc=1, n_mtp_depth=1)

    qe = load_qwen_embed('checkpoints/qwen_model/qwen_params.npz')
    rng = jax.random.PRNGKey(0)
    params, cfg = init_qwen_lang_lcm(rng, cfg, qe)

    x = jnp.zeros((2, 4), dtype=jnp.int32)
    logits, h, aux = qwen_lang_lcm_forward(params, x, cfg, rng=rng)
    print(f'Forward OK: logits {logits.shape}, h {h.shape}')
    print(f'Decoder: {len(params["decoder"])} layers, d={cfg.d_model}')
    print('Sanity check PASSED!')
    return True


if __name__ == "__main__":
    sanity_check()
