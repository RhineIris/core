"""Language LCM — continuous language model with codebook memory.

Two output channels from the same cognitive state:
  - Passive: z_q @ W_out — honest direct readout, no deception gap
  - Active:  Language LCM — continuous language model that reads from codebook
             memory of semantic-syntactic primitives at each decoder layer

Architecture (active channel):
  tokens → embed → [transformer decoder × N] → LN → W_out → logits
                     each layer:
                       self_attn → +residual → LN
                       FFN       → +residual → LN
                       codebook_soft_read(all 6 codebooks) → +residual → LN

Key design:
  - Codebook read is SOFT attention over entries (weighted sum, no STE, no VQ)
  - Main computation is continuous transformer (residual+LN, gradient flows freely)
  - Codebook entries store language primitives as memory, read via attention
  - Shares token embedding and W_out with the Cognitive LCM

Reference: human language production from memory — thoughts first, then retrieve
words and sentence frames to articulate them. The codebooks are the memory of
language patterns, the decoder is the articulator.
"""
import functools
import jax
import jax.numpy as jnp
from jax import lax

from train.encoder import layer_norm


# ─── Dropout wrapper ──────────────────────────────────────────────────────────

def _dropout(x, rate, rng):
    """Apply dropout during training, identity at eval."""
    if rate <= 0.0 or rng is None:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)


# ─── Decoder layer: causal self-attention + GLU (pure transformer) ───────────

def _softmax_attention(q, k, v, mask=None):
    """Standard causal softmax attention."""
    d_h = q.shape[-1]
    logits = jnp.einsum('bhnd,bhmd->bhnm', q, k) / jnp.sqrt(d_h)
    if mask is not None:
        logits = logits + mask
    attn = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum('bhnm,bhmd->bhnd', attn, v)


def _causal_mask(N):
    return jnp.tril(jnp.full((N, N), -1e9), k=0)


def _glu(x, w_gate, w_up, w_down):
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    hidden = gate * up
    return hidden @ w_down


def decoder_layer_forward(h, params, N, n_heads=4, training=False,
                           dropout_rng=None, dropout_rate=0.0):
    """Pure transformer decoder layer with dropout.

    Args:
        h: (B, N, d) input.
        params: Layer parameters.
        N: Sequence length.
        n_heads: Number of attention heads.
        training: Apply dropout when True.
        dropout_rng: JAX PRNG key for dropout.
        dropout_rate: Dropout probability (default 0.0).

    Returns:
        (B, N, d) output.
    """
    d = h.shape[-1]
    causal_mask = _causal_mask(N)[None, None, :, :]

    # ── Multi-head self-attention (pre-LN) ──────────────────────────────────
    h_norm = layer_norm(h, params['ln1_scale'], params['ln1_bias'])
    H = n_heads
    d_h = d // H

    def _split_heads(x):
        return x.reshape(-1, N, H, d_h).transpose(0, 2, 1, 3)

    q = _split_heads(h_norm @ params['w_q'])
    k = _split_heads(h_norm @ params['w_k'])
    v = _split_heads(h_norm @ params['w_v'])
    attn_out = _softmax_attention(q, k, v, causal_mask)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(-1, N, d)
    attn_out = _dropout(attn_out @ params['w_o'], dropout_rate,
                        dropout_rng)
    h = h + attn_out

    # ── FFN (pre-LN) ────────────────────────────────────────────────────────
    h_norm = layer_norm(h, params['ln2_scale'], params['ln2_bias'])
    ffn_out = _glu(h_norm, params['w_gate'], params['w_up'], params['w_down'])
    ffn_out = _dropout(ffn_out, dropout_rate, dropout_rng)
    h = h + ffn_out

    # Note: codebook soft read is removed for the pure-transformer baseline.
    # It will be re-added in a later stage once the transformer alone can
    # learn language structure.

    return h


# ─── Init ──────────────────────────────────────────────────────────────────────

def _n_heads(cfg):
    return max(1, min(cfg.n_heads, cfg.d_model // 32))


def init_lang_lcm_params(rng, cfg):
    """Initialize Language LCM parameters (pure transformer version).

    Architecture: 4-layer transformer decoder with GLU.
    No codebook entries — they'll be added in Stage 2.

    Structure:
      embed: (V, d) token embedding
      decoder: list of transformer decoder layer params
      ln_final_scale, ln_final_bias: final LayerNorm
      W_out: (d, V) output projection
    """
    keys = jax.random.split(rng, 10)
    d = cfg.d_model
    n_layers = 4
    H = _n_heads(cfg)

    params = {}

    # Token embedding
    params['embed'] = jax.random.normal(keys[0], (cfg.vocab_size, d)) * (d ** -0.5)

    # Positional embedding (learnable) — critical for word order!
    # Without this, self-attention is permutation-invariant and can't
    # distinguish "猫追老鼠" from "老鼠追猫".
    max_len = getattr(cfg, 'max_seq_len', 512)
    params['pos_embed'] = jax.random.normal(keys[1], (max_len, d)) * (d ** -0.5)

    # Decoder layers (pure transformer, no codebook entries)
    params['decoder'] = []
    for l in range(n_layers):
        kl = jax.random.split(keys[7], 8)
        layer = {
            'w_q': jax.random.normal(kl[0], (d, d)) * (d ** -0.5),
            'w_k': jax.random.normal(kl[1], (d, d)) * (d ** -0.5),
            'w_v': jax.random.normal(kl[2], (d, d)) * (d ** -0.5),
            'w_o': jax.random.normal(kl[3], (d, d)) * (d ** -0.5),
            'ln1_scale': jnp.ones(d), 'ln1_bias': jnp.zeros(d),
            'w_gate': jax.random.normal(kl[4], (d, d * 4)) * (d ** -0.5),
            'w_up': jax.random.normal(kl[5], (d, d * 4)) * (d ** -0.5),
            'w_down': jax.random.normal(kl[6], (d * 4, d)) * ((d * 4) ** -0.5),
            'ln2_scale': jnp.ones(d), 'ln2_bias': jnp.zeros(d),
        }
        params['decoder'].append(layer)

    # Final LayerNorm
    params['ln_final_scale'] = jnp.ones(d)
    params['ln_final_bias'] = jnp.zeros(d)

    # Output projection
    params['W_out'] = jax.random.normal(keys[8], (d, cfg.vocab_size)) * (d ** -0.5)

    return params


# ─── Forward pass (training, teacher forcing) ─────────────────────────────────

def lang_lcm_forward(params, x, cfg, rng=None, training=True,
                      dropout_rate=0.2, z_q=None):
    """Language LCM forward pass (pure transformer).

    Teacher-forced training with optional dropout regularization.
    When z_q is provided, it replaces the first position's token embedding,
    allowing the language LCM to be conditioned on a cognitive state.

    Args:
        params: Language LCM parameters.
        x: Input token IDs (B, N).
        cfg: LCMConfig.
        rng: JAX PRNG key for dropout (required when training=True).
        training: Whether in training mode (enables dropout).
        dropout_rate: Dropout probability (default 0.2).
        z_q: Optional (B, d) cognitive state to inject as start token.

    Returns:
        logits: (B, N, V) next-token predictions.
        h: (B, N, d) final hidden state.
        aux: Dict of auxiliary outputs.
    """
    B, N = x.shape
    h = params['embed'][x]  # (B, N, d)

    # Inject cognitive state as first position's hidden state
    # z_q replaces embed[x[:, 0]] so the model is conditioned on
    # cognitive state rather than a fixed start token.
    if z_q is not None:
        h = h.at[:, 0, :].set(z_q)

    # Add positional embedding (learnable, key for word order)
    pos_indices = jnp.arange(N, dtype=jnp.int32)  # (N,)
    h = h + params['pos_embed'][pos_indices]  # broadcast over batch

    n_heads = _n_heads(cfg)
    n_layers = len(params['decoder'])

    for layer_params in params['decoder']:
        if training and dropout_rate > 0.0 and rng is not None:
            rng, do_rng = jax.random.split(rng)
        else:
            do_rng = None
        h = decoder_layer_forward(
            h, layer_params, N, n_heads=n_heads,
            training=training, dropout_rng=do_rng,
            dropout_rate=dropout_rate)

    h = layer_norm(h, params['ln_final_scale'], params['ln_final_bias'])
    logits = h @ params['W_out']

    aux = {}
    return logits, h, aux


# ─── Autoregressive generation (with JIT-compiled forward) ────────────────────

@functools.partial(jax.jit, static_argnames=('cfg',))
def _gen_forward(params, x, cfg, rng):
    """JIT-compiled forward pass for generation (no dropout)."""
    return lang_lcm_forward(params, x, cfg, rng=rng, training=False, dropout_rate=0.0)


def lang_lcm_generate(params, prompt, max_len, bos_id, eos_id, rng, cfg):
    """Autoregressive generation with Language LCM (JIT-compiled).

    Uses fixed-size input (1, total_len) to avoid JAX recompilation
    on every step.

    Args:
        params: Language LCM parameters.
        prompt: (seq_len,) initial token IDs.
        max_len: Maximum tokens to generate.
        bos_id: BOS token ID.
        eos_id: EOS token ID.
        rng: JAX PRNG key.
        cfg: LCMConfig.

    Returns:
        tokens: List of generated token IDs (including prompt).
    """
    prompt_len = len(prompt)
    total = prompt_len + max_len

    tokens_list = list(prompt)
    x = jnp.zeros((1, total), dtype=jnp.int32)
    x = x.at[0, :prompt_len].set(jnp.array(prompt))

    # Pre-compile: run one dummy step to trigger JIT
    _ = _gen_forward(params, x, cfg, rng)

    pos = prompt_len
    for _ in range(max_len):
        logits, _, _ = _gen_forward(params, x, cfg, rng)
        next_logits = logits[0, pos - 1, :]  # predict token at position pos

        rng, sample_rng = jax.random.split(rng)
        next_id = int(jax.random.categorical(sample_rng, next_logits))
        tokens_list.append(next_id)

        if next_id == eos_id or next_id == 0:
            break

        x = x.at[0, pos].set(next_id)
        pos += 1

    return tokens_list


# ─── Sanity check ─────────────────────────────────────────────────────────────

def sanity_check():
    """Verify forward pass shape and gradient flow."""
    from train.config import LCMConfig
    cfg = LCMConfig()
    rng = jax.random.PRNGKey(0)
    params = init_lang_lcm_params(rng, cfg)

    x = jnp.zeros((2, 8), dtype=jnp.int32)
    logits, h, aux = lang_lcm_forward(params, x, cfg, rng=rng)
    print(f"Input:  (2, 8)")
    print(f"Logits: {logits.shape}  (expected (2, 8, {cfg.vocab_size}))")
    print(f"H:      {h.shape}  (expected (2, 8, {cfg.d_model}))")

    # Gradient check: one step
    import optax
    targets = jnp.ones((2, 8), dtype=jnp.int32)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)).mean()
    grads = jax.grad(lambda p: optax.softmax_cross_entropy_with_integer_labels(
        lang_lcm_forward(p, x, cfg)[0].reshape(-1, cfg.vocab_size),
        targets.reshape(-1)).mean())(params)

    # Check grad norms per module
    for name, g in grads.items():
        if hasattr(g, 'items'):
            gnorm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree_util.tree_leaves(g)))
            print(f"  grad {name}: {float(gnorm):.2f}")
        elif g is not None:
            print(f"  grad {name}: {float(jnp.sqrt(jnp.sum(g**2))):.2f}")

    print("Sanity check OK!")
    return True


if __name__ == "__main__":
    sanity_check()
