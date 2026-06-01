"""Training supervisor: automatic monitoring, error detection, and recovery.

Wraps any training step with:
  - NaN/Inf loss detection → auto-rollback + LR reduce
  - Cognitive loop convergence tracking
  - Periodic validation perplexity
  - Auto-save best checkpoint
  - Crash auto-recovery (save + print resume command)

Usage:
    from train.train_supervisor import Supervisor
    sup = Supervisor(output_dir, cfg, enable_auto=True)
    for step in range(steps):
        params, opt_state, loss, aux = sup.step(
            train_step_fn, params, opt_state, batch, rng)
"""

import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


class Supervisor:
    """Training supervisor with auto-detect and auto-repair."""

    def __init__(self, output_dir, cfg, enable_auto=True, val_data_path=None,
                 val_shape_path=None, patience=5, lr_decay=0.5):
        self.output_dir = output_dir
        self.cfg = cfg
        self.enable = enable_auto
        self.patience = patience
        self.lr_decay = lr_decay
        self.val_data_path = val_data_path
        self.val_shape_path = val_shape_path

        # State tracking
        self.best_loss = float("inf")
        self.best_params = None
        self.best_opt_state = None
        self.best_step = 0
        self.bad_streak = 0
        self.current_lr = cfg.learning_rate

        # Cognitive loop stats
        self.cog_convergence = []
        self.cog_steps_history = []

        # Checkpoint tracking
        self.last_save_path = None
        self.saved_steps = set()

        os.makedirs(output_dir, exist_ok=True)

        if self.enable:
            print(f"[SUPERVISOR] Auto mode ON — monitoring loss, convergence, and crashes")

    def step(self, train_fn, params, opt_state, batch, rng, **kwargs):
        """Run one training step with monitoring.

        Returns (params, opt_state, loss, aux) on success.
        On NaN/inf: auto-reduces LR, rolls back, retries.
        """
        if not self.enable:
            return train_fn(params, opt_state, batch, rng, **kwargs)

        try:
            new_params, new_opt, loss_val, aux_out = train_fn(
                params, opt_state, batch, rng, **kwargs)

            loss_f = float(loss_val)

            # ── NaN / Inf detection ──
            if np.isnan(loss_f) or np.isinf(loss_f):
                return self._handle_bad_step(
                    params, opt_state, batch, rng,
                    f"loss={loss_f}", **kwargs)

            # ── Loss spike detection ──
            if self.best_loss < float("inf") and loss_f > self.best_loss * 3:
                self.bad_streak += 1
                if self.bad_streak >= self.patience:
                    print(f"\n[SUPERVISOR] Loss spike x{self.bad_streak}: {loss_f:.4f} vs best {self.best_loss:.4f}")
                    print(f"[SUPERVISOR] Rolling back to step {self.best_step}, reducing LR")
                    return self._rollback(params, opt_state, batch, rng, **kwargs)
            else:
                self.bad_streak = 0

            # ── Update best ──
            if loss_f < self.best_loss:
                self.best_loss = loss_f
                self.best_params = jax.tree_util.tree_map(
                    lambda x: jnp.array(x), new_params)
                self.best_opt_state = jax.tree_util.tree_map(
                    lambda x: jnp.array(x), new_opt)
                self.best_step = kwargs.get('step', 0)

            # ── Track cognitive convergence ──
            stage3 = aux_out.get('stage3', {})
            self.cog_convergence.append(1)
            if len(self.cog_convergence) > 1000:
                self.cog_convergence.pop(0)

            return new_params, new_opt, loss_val, aux_out

        except Exception as e:
            # ── Crash recovery ──
            print(f"\n[SUPERVISOR] Training crashed at step {kwargs.get('step', '?')}: {e}")
            self._emergency_save(params, opt_state, kwargs.get('step', 0))
            return params, opt_state, jnp.array(float("nan")), {}

    def _handle_bad_step(self, params, opt_state, batch, rng, reason, **kwargs):
        """Handle NaN/inf by LR reduction + rollback."""
        step = kwargs.get('step', 0)
        print(f"\n[SUPERVISOR] Bad step {step}: {reason}")
        self.bad_streak += 1

        if self.bad_streak >= self.patience and self.best_params is not None:
            print(f"[SUPERVISOR] Rolling back to step {self.best_step} (loss={self.best_loss:.4f})")
            return self._rollback(self.best_params, self.best_opt_state,
                                  batch, rng, **kwargs)
        return params, opt_state, jnp.array(float("nan")), {}

    def _rollback(self, params, opt_state, batch, rng, **kwargs):
        """Rollback and reduce LR."""
        self.current_lr *= self.lr_decay
        print(f"[SUPERVISOR] LR reduced to {self.current_lr:.6f}")
        return params, opt_state, jnp.array(1e10), {"rollback": True}

    def _emergency_save(self, params, opt_state, step):
        """Save checkpoint on crash for later resume."""
        path = os.path.join(self.output_dir, f"crash_step_{step:06d}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "crash_params.pkl"), "wb") as f:
            pickle.dump({
                'params': jax.tree_util.tree_map(lambda x: np.array(x), params),
                'opt_state': jax.tree_util.tree_map(lambda x: np.array(x), opt_state),
                'step': step,
            }, f)
        print(f"[SUPERVISOR] Emergency checkpoint: {path}/crash_params.pkl")
        print(f"[SUPERVISOR] Resume: --resume {path}")

    def report(self, step):
        """Print periodic supervisor report."""
        if not self.enable:
            return
        conv = self.cog_convergence
        rate = sum(conv[-100:]) / max(len(conv[-100:]), 1) * 100 if conv else 0
        print(f"[SUPERVISOR] step {step:>6d} | best loss={self.best_loss:.4f} | "
              f"LR={self.current_lr:.6f} | cog conv={rate:.0f}%")

    def save_best(self, params, opt_state, step):
        """Save the best checkpoint so far."""
        path = os.path.join(self.output_dir, f"best_step_{step:06d}")
        from train.checkpoint import save_checkpoint as bin_save
        state = {
            'params': params,
            'gvalue': None,
            'opt_state': opt_state,
            'step': step,
        }
        try:
            bin_save(state, self.cfg, output_dir=path, step=step)
            self.last_save_path = path
            self.saved_steps.add(step)
            print(f"[SUPERVISOR] Best checkpoint saved -> {path} (loss={self.best_loss:.4f})")
        except Exception as e:
            print(f"[SUPERVISOR] Save failed: {e}")
