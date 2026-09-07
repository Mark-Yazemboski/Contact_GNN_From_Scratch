"""
run_diagnostics.py

NEW FILE - reads the two checkpoints train_force_gns.py already writes and
turns them into (a) extra columns for the master CSV and (b) one figure per
run. Nothing here trains or evaluates anything; it is pure post-processing, so
it is safe to call on a finished run at any time.

WHAT IT PULLS

  <model>_physics.pt        recovered_mu, recovered_k_over_m, mu_mode, k_mode
  <model>_loss_history.pt   train/val curves, mu_trace, k_trace, best val,
                            total optimizer steps

WHY THIS EXISTS

  recovered_k_over_m was only ever written to _physics.pt and printed to the
  log, so comparing k across runs meant grepping four log files. The converged
  prediction loss had the same problem, and it is the denominator of the
  physics-weight calibration (gamma = 0.03 * L_pred / raw), so it is needed
  often enough to belong in the CSV.

  final_train_loss is a MEAN OVER THE LAST N EPOCHS, not the final value. The
  per-epoch loss bounces by ~10% (0.146 to 0.193 across the last five epochs of
  one run), and reading a single epoch off the end of a log is how the first
  weight calibration ended up 75x wrong.

USAGE

  From run_force_multi_step.py (see the patch at the bottom of this docstring),
  or standalone to backfill a finished run:

      python run_diagnostics.py models/MAG_1e1_1/256_force_gns_model.pt
"""

import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")            # compute node, no display
import matplotlib.pyplot as plt


# ======================================================================
# Loading
# ======================================================================

def _load(path):
    """torch.load that works across versions. The checkpoints hold plain
    dicts of floats, lists and tensors, so weights_only=True is fine on new
    torch; older torch has no such argument."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        pass
    # torch >= 2.6 defaults weights_only=True, so the fallback must pass False
    # EXPLICITLY - omitting it retries the identical call. These are our own
    # checkpoints, written by train_force_gns.py, so unpickling them is safe.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                      # torch < 1.13 has no such argument
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"  [run_diagnostics] could not read {os.path.basename(path)}: {e}")
            return None
    except Exception as e:
        print(f"  [run_diagnostics] could not read {os.path.basename(path)}: {e}")
        return None


def _paths(save_model_path):
    stem = os.path.splitext(save_model_path)[0]
    return stem + "_physics.pt", stem + "_loss_history.pt", stem


def _trace_arrays(trace):
    """[(epoch, value), ...] -> (epochs, values) as float arrays, or None."""
    if not trace:
        return None
    a = np.asarray(trace, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
        return None
    return a[:, 0], a[:, 1]


# ======================================================================
# CSV columns
# ======================================================================

def collect_run_diagnostics(save_model_path, last_n=20):
    """-> flat dict of floats/strings to merge into the run-report settings.

    Every key is optional: a run with learn_k=False has no k_trace, a run
    with Train_model=False has no checkpoints at all, and in both cases the
    corresponding keys are simply absent rather than NaN-filled.
    """
    phys_path, hist_path, _ = _paths(save_model_path)
    out = {}

    ph = _load(phys_path)
    if ph:
        for src, dst in (("recovered_mu", "recovered_mu_ckpt"),
                         ("recovered_k_over_m", "recovered_k_over_m"),
                         ("mu_mode", "mu_mode"), ("k_mode", "k_mode")):
            if src in ph:
                v = ph[src]
                out[dst] = float(v) if isinstance(v, (int, float)) else str(v)

    hi = _load(hist_path)
    if hi:
        tv = hi.get("train_loss_values") or []
        if tv:
            tail = np.asarray(tv[-last_n:], dtype=float)
            tail = tail[np.isfinite(tail)]
            if tail.size:
                # The denominator of gamma = 0.03 * L_pred / raw. Averaged,
                # because the per-epoch value bounces by ~10%.
                out["final_train_loss"] = float(tail.mean())
                out["final_train_loss_std"] = (float(tail.std(ddof=1))
                                               if tail.size > 1 else 0.0)
                out["final_train_loss_n"] = int(tail.size)
        for src, dst in (("best_val_loss", "best_val_loss"),
                         ("best_val_epoch", "best_val_epoch"),
                         ("global_step", "total_optimizer_steps")):
            if hi.get(src) is not None:
                out[dst] = float(hi[src])

        # Endpoint AND drift of each recovered parameter. The drift is what
        # distinguishes "identified" from "never moved" - a monotone descent
        # that has not arrived is a different result from a value that parked.
        for name, key in (("mu", "mu_trace"), ("k", "k_trace")):
            tr = _trace_arrays(hi.get(key))
            if tr is None:
                continue
            ep, val = tr
            out[f"{name}_final"] = float(val[-1])
            out[f"{name}_init"] = float(val[0])
            out[f"{name}_drift"] = float(val[-1] - val[0])
            # movement over the last 10% of training: ~0 means converged
            n_tail = max(2, len(val) // 10)
            out[f"{name}_tail_slope_per_1k_ep"] = float(
                np.polyfit(ep[-n_tail:], val[-n_tail:], 1)[0] * 1000.0)
    return out


# ======================================================================
# Figure
# ======================================================================

def plot_run_diagnostics(save_model_path, out_path=None,
                         mu_true=0.198, k_calibrated=0.0285, last_n=20):
    """One PNG per run: loss curves, mu trace, k trace. Panels with no data
    are dropped rather than drawn empty. Returns the path, or None."""
    phys_path, hist_path, stem = _paths(save_model_path)
    hi = _load(hist_path)
    if not hi:
        print("  [run_diagnostics] no loss history; skipping plot")
        return None

    mu_tr = _trace_arrays(hi.get("mu_trace"))
    k_tr = _trace_arrays(hi.get("k_trace"))
    panels = ["loss"] + (["mu"] if mu_tr else []) + (["k"] if k_tr else [])

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8))
    axes = np.atleast_1d(axes)
    run = os.path.basename(os.path.dirname(save_model_path)) or "run"
    fig.suptitle(run, fontsize=11)

    for ax, which in zip(axes, panels):
        if which == "loss":
            te, tv = hi.get("train_loss_epochs"), hi.get("train_loss_values")
            ve, vv = hi.get("val_loss_epochs"), hi.get("val_loss_values")
            if te and tv:
                ax.plot(te, tv, lw=0.9, label="train")
                tail = np.asarray(tv[-last_n:], dtype=float)
                if tail.size:
                    ax.axhline(tail.mean(), ls="--", lw=1.0, color="k",
                               label=f"last {tail.size} mean = {tail.mean():.4f}")
            if ve and vv:
                ax.plot(ve, vv, lw=1.2, label="val (rollout center)")
            ax.set_yscale("log")
            ax.set_xlabel("epoch"); ax.set_ylabel("loss")
            ax.set_title("training"); ax.legend(fontsize=8)

        elif which == "mu":
            ep, val = mu_tr
            ax.plot(ep, val, lw=1.2)
            ax.axhline(mu_true, ls="--", lw=1.0, color="k",
                       label=f"generator mu = {mu_true}")
            ax.annotate(f"{val[-1]:.4f}", xy=(ep[-1], val[-1]),
                        xytext=(-46, 6), textcoords="offset points", fontsize=9)
            ax.set_xlabel("epoch"); ax.set_ylabel("mu")
            ax.set_title("recovered friction coefficient"); ax.legend(fontsize=8)

        elif which == "k":
            ep, val = k_tr
            ax.plot(ep, val, lw=1.2)
            ax.axhline(k_calibrated, ls="--", lw=1.0, color="k",
                       label=f"calibrated k/m = {k_calibrated}")
            ax.annotate(f"{val[-1]:.5f}", xy=(ep[-1], val[-1]),
                        xytext=(-52, 6), textcoords="offset points", fontsize=9)
            # log scale: k is learned in log space, and a run started at a
            # deliberately wrong init spans more than a decade.
            if val.min() > 0 and val.max() / val.min() > 3:
                ax.set_yscale("log")
            ax.set_xlabel("epoch"); ax.set_ylabel("k/m")
            ax.set_title("recovered drag coefficient"); ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = out_path or (stem + "_diagnostics.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Diagnostics figure saved to {out_path}")
    return out_path


# ======================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        print(f"\n=== {p} ===")
        d = collect_run_diagnostics(p)
        for k, v in sorted(d.items()):
            print(f"  {k:<28} {v}")
        plot_run_diagnostics(p)
