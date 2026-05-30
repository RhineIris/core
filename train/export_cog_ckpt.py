"""Export cog_train checkpoint → C inference engine format (encoder.bin, decoder.bin, codebooks)."""

import json
import os
import pickle
import struct
import sys

import numpy as np

LCM_MAGIC = b"LCM_CB"
LCM_VERSION = 2


def _write_bin_header(path, M, d, n_layers, cb_type, c=0):
    """Write binary header for codebook .bin files."""
    buf = bytearray(36)
    buf[0:6] = LCM_MAGIC
    struct.pack_into("<I", buf, 6, LCM_VERSION)
    struct.pack_into("<I", buf, 10, M)
    struct.pack_into("<I", buf, 14, d)
    struct.pack_into("<I", buf, 18, n_layers)
    buf[22] = cb_type
    buf[23] = 0
    struct.pack_into("<I", buf, 24, c)
    crc = sum(buf[:28]) & 0xFFFFFFFF
    struct.pack_into("<I", buf, 28, crc)
    struct.pack_into("<I", buf, 32, 0)  # reserved
    with open(path, "wb") as f:
        f.write(buf)


def export(ckpt_dir: str, out_dir: str, data_dir: str = "data"):
    """Export cog_train checkpoint to C inference format.

    Args:
        ckpt_dir: Directory containing cog_params.pkl.
        out_dir: Output directory for inference checkpoint.
        data_dir: Directory containing tokenizer.json.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Load params ────────────────────────────────────────────────────
    ckpt_path = os.path.join(ckpt_dir, "cog_params.pkl")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    params = ckpt["params"] if "params" in ckpt else ckpt
    step = ckpt.get("step", 0)
    print(f"[EXPORT] Loading params from {ckpt_path} (step {step})")

    d_model = params["W_out"].shape[0]
    vocab_size = params["W_out"].shape[1]

    # ── config.json ────────────────────────────────────────────────────
    cfg = {
        "d_model": d_model,
        "vocab_size": vocab_size,
        "max_seq_len": 512,
        "n_heads": 4,
        "n_encoder_layers": 2,
        "n_lattices": 6,
        "d_ff": int(1.5 * d_model),
        # codebook sizes (hardcoded defaults — override if needed)
        "M_top": params.get("hrq", {}).get("top", {}).get("A", np.zeros((512, 1))).shape[0],
        "M_fine": 256,
        "n_hrq_layers": len(params.get("hrq", {}).get("fine", [])),
        "M_sparse": params.get("sparse", {}).get("C", np.zeros((512, 1))).shape[0],
        "M_lr": params.get("lowrank", {}).get("A_V", np.zeros((256, 1))).shape[0],
        "n_lr_layers": 3,
        "M_man": params.get("manifold", {}).get("C", np.zeros((512, 1))).shape[0],
        "t_dim": 4,
        "M_bind": 512,
        "n_bind_layers": 3,
        "M_contrast": 512,
        "n_contrast_layers": 3,
        "r_max": 8,
        "n_value_pairs": 4,
        "M_danger": 256,
        "n_self_codes": params.get("self", {}).get("modes", np.zeros((64, 1))).shape[0],
        "max_inference_steps": 32,
        "convergence_tol": 1e-3,
        "entropy_threshold": 0.5,
        "tau_route": 0.5,
        "beta_vq": 0.25,
        "gamma_sparse": 0.99,
        "gamma_man": 0.99,
        "gamma_bind": 0.99,
    }
    # Fill actual sizes from params
    hrq_top = params.get("hrq", {}).get("top", {})
    if hrq_top:
        cfg["M_top"] = hrq_top.get("A", np.zeros((1, 1))).shape[0] if "A" in hrq_top else 512
    hrq_fine = params.get("hrq", {}).get("fine", [])
    if hrq_fine:
        cfg["n_hrq_layers"] = len(hrq_fine)
        cfg["M_fine"] = hrq_fine[0].get("A", np.zeros((1, 1))).shape[0] if "A" in hrq_fine[0] else 256
    sparse = params.get("sparse", {})
    if sparse and "C" in sparse:
        cfg["M_sparse"] = sparse["C"].shape[0]
    lowrank = params.get("lowrank", {})
    if lowrank and "A_V" in lowrank:
        cfg["M_lr"] = lowrank["A_V"].shape[0]
    manifold = params.get("manifold", {})
    if manifold and "C" in manifold:
        cfg["M_man"] = manifold["C"].shape[0]
    binding = params.get("binding", {})
    if binding and "key_cb" in binding:
        cfg["n_bind_layers"] = len(binding["key_cb"])
    contrast = params.get("contrast", {})
    if contrast and "C_a" in contrast:
        cfg["n_contrast_layers"] = len(contrast["C_a"])
    self_p = params.get("self", {})
    if self_p and "modes" in self_p:
        cfg["n_self_codes"] = self_p["modes"].shape[0]

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[EXPORT] config.json → {out_dir}/")

    # ── encoder.bin ────────────────────────────────────────────────────
    enc = params.get("encoder", {})
    if not enc:
        print("[EXPORT] WARN: no encoder params found.  Generating dummy.")
        n_layers = cfg["n_encoder_layers"]
        d = cfg["d_model"]
        d_ff = cfg["d_ff"]
        V = cfg["vocab_size"]
        enc = {
            "embed": np.random.randn(V, d).astype(np.float32) * 0.02,
            "rel_bias": np.random.randn(2 * cfg["max_seq_len"] - 1).astype(np.float32) * 0.01,
            "layers": [
                {
                    "ln1_scale": np.ones(d, dtype=np.float32),
                    "ln1_bias": np.zeros(d, dtype=np.float32),
                    "w_q": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
                    "w_k": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
                    "w_v": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
                    "w_o": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
                    "ln2_scale": np.ones(d, dtype=np.float32),
                    "ln2_bias": np.zeros(d, dtype=np.float32),
                    "w_1": np.random.randn(d, d_ff).astype(np.float32) * (d ** -0.5),
                    "w_2": np.random.randn(d, d_ff).astype(np.float32) * (d ** -0.5),
                    "w_3": np.random.randn(d_ff, d).astype(np.float32) * (d_ff ** -0.5),
                }
                for _ in range(n_layers)
            ],
            "q_pool": np.random.randn(d).astype(np.float32) * 0.01,
            "w_proj": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
        }

    def _to_np(x):
        return np.array(x) if hasattr(x, "numpy") else x

    enc_np = {k: _to_np(v) for k, v in enc.items()}
    parts = [enc_np["embed"].ravel()]
    parts.append(enc_np["rel_bias"].ravel())
    for layer in enc_np["layers"]:
        for key in ["ln1_scale", "ln1_bias", "w_q", "w_k", "w_v", "w_o",
                     "ln2_scale", "ln2_bias", "w_1", "w_2", "w_3"]:
            parts.append(_to_np(layer[key]).ravel())
    parts.append(enc_np["q_pool"].ravel())
    parts.append(enc_np["w_proj"].ravel())

    encoder_flat = np.concatenate(parts).astype(np.float32)
    encoder_flat.tofile(os.path.join(out_dir, "encoder.bin"))
    print(f"[EXPORT] encoder.bin ({encoder_flat.nbytes / 1e6:.1f} MB) → {out_dir}/")

    # ── decoder.bin ────────────────────────────────────────────────────
    gen_head = params.get("gen_head", {})
    if gen_head and "w_embed" in gen_head:
        # new format decoder
        d = cfg["d_model"]
        V = cfg["vocab_size"]
        parts = []
        parts.append(_to_np(gen_head["w_embed"]).ravel())
        for key in ["w_q", "w_k", "w_v", "w_o"]:
            parts.append(_to_np(gen_head[key]).ravel())
        for key in ["w_1", "w_2"]:
            parts.append(_to_np(gen_head[key]).ravel())
        parts.append(_to_np(gen_head["w_3"]).ravel())
        decoder_flat = np.concatenate(parts).astype(np.float32)
    elif "W_out" in params:
        # old format: W_out as decoder
        W_out = _to_np(params["W_out"])
        d = W_out.shape[0]
        V = W_out.shape[1]
        W_proj = np.eye(d, dtype=np.float32)
        decoder_flat = np.concatenate([W_proj.ravel(), W_out.ravel()]).astype(np.float32)
    else:
        print("[EXPORT] WARN: no decoder/W_out found.  Writing dummy.")
        decoder_flat = np.random.randn(d_model * d_model + d_model * vocab_size).astype(np.float32)

    decoder_flat.tofile(os.path.join(out_dir, "decoder.bin"))
    print(f"[EXPORT] decoder.bin ({decoder_flat.nbytes / 1e6:.1f} MB) → {out_dir}/")

    # ── codebooks ──────────────────────────────────────────────────────
    codebooks_dir = os.path.join(out_dir, "codebooks")
    os.makedirs(codebooks_dir, exist_ok=True)

    # Build list of (filename_prefix, cb_type, matrix) entries
    cb_entries = []

    def _simvq_cb(simvq):
        """Extract actual codebook: A @ W."""
        A = _to_np(simvq["A"])
        W = _to_np(simvq["W"])
        return A @ W

    # HRQ: top + fine per layer
    hrq = params.get("hrq", {})
    if "top" in hrq:
        cb_top = _simvq_cb(hrq["top"])
        cb_entries.append((f"hrq_codebook", 10, cb_top))  # cb_type 10 = HRQ

    # Sparse
    sparse = params.get("sparse", {})
    if "C" in sparse:
        cb_entries.append(("sparse_codebook", 11, _to_np(sparse["C"])))

    # LowRank
    lr = params.get("lowrank", {})
    if "A_V" in lr and "W_V" in lr and "U" in lr:
        V = _to_np(lr["A_V"]) @ _to_np(lr["W_V"])
        lr_matrices = [_to_np(u) @ V[:, :_to_np(u).shape[-1]].T for u in lr["U"]]
        # If multiple ranks, concatenate
        lr_flat = np.concatenate([m.ravel() for m in lr_matrices])
        # Write as single flat file for now
        lr_flat.tofile(os.path.join(codebooks_dir, "lowrank_codebook.bin"))
        # Write header manually
        Mc = lr_matrices[0].shape[0]
        d = lr_matrices[0].shape[1]
        _write_bin_header(os.path.join(codebooks_dir, "lowrank_codebook.bin"), Mc, d, len(lr_matrices), 12)

    # Manifold
    manifold = params.get("manifold", {})
    if "C" in manifold:
        cb_entries.append(("manifold_codebook", 13, _to_np(manifold["C"])))

    # Binding: key, value, bind per layer
    binding = params.get("binding", {})
    for key_list, cb_type in [("key_cb", 14), ("val_cb", 15), ("bind_cb", 16)]:
        if key_list in binding:
            for i, cb in enumerate(binding[key_list]):
                name = f"binding_{key_list.split('_')[0]}_{i}.bin"
                mat = _simvq_cb(cb)
                mat.astype(np.float32).tofile(os.path.join(codebooks_dir, name))
                _write_bin_header(os.path.join(codebooks_dir, name), mat.shape[0], mat.shape[1], 1, cb_type)

    # Write codebooks with headers
    for prefix, cb_type, mat in cb_entries:
        path = os.path.join(codebooks_dir, f"{prefix}.bin")
        mat.astype(np.float32).tofile(path)
        _write_bin_header(path, mat.shape[0], mat.shape[1], 1, cb_type)

    print(f"[EXPORT] Codebooks → {codebooks_dir}/")

    # ── tokenizer.json (copy) ──────────────────────────────────────────
    import shutil
    tok_src = os.path.join(data_dir, "tokenizer.json")
    tok_dst = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_src):
        shutil.copy2(tok_src, tok_dst)
        print(f"[EXPORT] tokenizer.json → {out_dir}/")
    else:
        print(f"[EXPORT] WARN: tokenizer.json not found at {tok_src}")

    # ── gvalue codebooks ───────────────────────────────────────────────
    from train.gvalue import make_global_value_vectors, GValueCodebook
    C_pos, C_neg = make_global_value_vectors(d_model)
    gv = GValueCodebook(C_pos, C_neg)
    gv.save(out_dir)

    print(f"[EXPORT] Done → {out_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export cog_train checkpoint → C inference format")
    p.add_argument("ckpt_dir", help="Path to cog_train checkpoint directory")
    p.add_argument("-o", "--out", default=None, help="Output directory (default: ckpt_dir + _infer)")
    p.add_argument("--data-dir", default="data", help="Data directory (for tokenizer.json)")
    args = p.parse_args()
    out = args.out or args.ckpt_dir.rstrip("/") + "_infer"
    export(args.ckpt_dir, out, args.data_dir)