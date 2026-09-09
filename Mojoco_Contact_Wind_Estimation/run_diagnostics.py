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

  Live, from run_force_multi_step.py - the values are added to the run report
  automatically, no action needed.

  Standalone, to backfill runs that finished before this file existed. The
  run_name is taken from the parent directory, which run_force_multi_step.py
  builds from extra_name, so the match against the CSV is exact:

  PASS RUN DIRECTORIES. The bare <stem>.pt is never written by the trainer, so
  a glob on it matches nothing and the shell hands the pattern through
  unexpanded. Directories always exist, so they always glob.

      # print + figure only
      python run_diagnostics.py models/MAG_1e1_1/

      # ALSO write the values into the master CSV
      python run_diagnostics.py --csv models/all_force_runs_master.csv models/*/

      # CSV only, skip the figures
      python run_diagnostics.py --csv models/all_force_runs_master.csv --no-plot models/K_LearnOff_*/

  Individual checkpoint files (_physics.pt, _final.pt, ...) are also accepted.
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


_CKPT_SUFFIXES = ("_physics.pt", "_loss_history.pt", "_best_model.pt",
                  "_final.pt", "_norms.pt", ".pt")


def _resolve_stem(path):
    """Accept anything that points at a run and return the checkpoint stem.

    A RUN DIRECTORY is the friendliest input and the one to prefer:
    train_force_gns.py never writes the bare <stem>.pt - only _final.pt,
    _best_model.pt, _norms.pt, _physics.pt and _loss_history.pt - so a shell
    glob on '<name>/256_force_gns_model.pt' matches nothing and bash passes the
    unexpanded pattern straight through. Globbing on directories always works.

    Also accepts any individual checkpoint file, or the stem itself.
    """
    p = os.path.abspath(str(path).rstrip("/\\"))
    if os.path.isdir(p):
        for suffix in ("_physics.pt", "_loss_history.pt"):
            hits = sorted(f for f in os.listdir(p) if f.endswith(suffix))
            if hits:
                return os.path.join(p, hits[0][: -len(suffix)])
        return os.path.join(p, "")          # directory exists but is empty
    for suffix in _CKPT_SUFFIXES:
        if p.endswith(suffix):
            return p[: -len(suffix)]
    return p


def _paths(save_model_path):
    stem = _resolve_stem(save_model_path)
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
    if ph is None:
        print(f"  [run_diagnostics] MISSING {os.path.basename(phys_path)}"
              " -> no recovered_mu / recovered_k_over_m")
    if ph:
        for src, dst in (("recovered_mu", "recovered_mu_ckpt"),
                         ("recovered_k_over_m", "recovered_k_over_m"),
                         ("mu_mode", "mu_mode"), ("k_mode", "k_mode")):
            if src in ph:
                v = ph[src]
                out[dst] = float(v) if isinstance(v, (int, float)) else str(v)

    hi = _load(hist_path)
    if hi is None:
        print(f"  [run_diagnostics] MISSING {os.path.basename(hist_path)}"
              " -> no loss curves, no mu_trace, no k_trace")
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
                # Say WHY rather than leaving a silent NaN in the CSV: an
                # absent key and an empty list mean different things.
                print(f"  [run_diagnostics] no usable '{key}' "
                      + ("(key absent - trainer did not save it)"
                         if key not in hi else "(present but empty - "
                         "was the parameter learnable?)")
                      + f" -> {name}_final will be NaN")
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
# CSV backfill
# ======================================================================

def run_name_from_path(path):
    """models/<extra_name>/ or models/<extra_name>/<any checkpoint> -> <extra_name>.

    run_force_multi_step.py builds model_folder_path from extra_name and
    passes extra_name straight to save_run_report as run_name, so the run
    directory IS the CSV's run_name. Exact match, no guessing.
    """
    p = os.path.abspath(str(path).rstrip("/\\"))
    return os.path.basename(p if os.path.isdir(p) else os.path.dirname(p))


def backfill_csv(csv_path, model_paths, last_n=20):
    """Write collect_run_diagnostics() output into the matching CSV rows as
    settings.<key> columns. Backs up once, matches exactly on run_name, and
    reports anything it could not place instead of guessing."""
    import warnings
    import pandas as pd
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    df = pd.read_csv(csv_path).copy()   # de-fragment; we add columns one by one
    if "run_name" not in df.columns:
        print(f"  [run_diagnostics] {csv_path} has no run_name column")
        return
    known = set(df["run_name"].astype(str))

    matched, skipped = 0, []
    for p in model_paths:
        name = run_name_from_path(p)
        vals = collect_run_diagnostics(p, last_n=last_n)
        if not vals:
            # Distinguish "run has no checkpoints" from "this path does not
            # exist at all", which is almost always an unexpanded glob - e.g.
            # a stray backslash-space, which bash reads as an escaped space
            # rather than a line continuation.
            probe = os.path.abspath(str(p).rstrip("/\\"))
            folder = probe if os.path.isdir(probe) else os.path.dirname(probe)
            why = ("path does not exist - unexpanded glob? "
                   "(pass the run DIRECTORY, e.g. models/K_LearnOff_*/)"
                   if not os.path.isdir(folder)
                   else "no _physics.pt / _loss_history.pt in that folder")
            skipped.append((name, why))
            continue
        if name not in known:
            skipped.append((name, "run_name not in CSV"))
            continue
        rows = df["run_name"].astype(str) == name
        for col, v in vals.items():
            key = f"settings.{col}"
            # mu_mode / k_mode are strings; a float64 column refuses them, and
            # a pre-existing all-NaN column is float64 by default.
            if key not in df.columns:
                df[key] = pd.Series([np.nan] * len(df),
                                    dtype=object if isinstance(v, str) else float)
            elif isinstance(v, str) and df[key].dtype != object:
                df[key] = df[key].astype(object)
            df.loc[rows, key] = v
        matched += 1
        bits = [f"{k}={vals[k]:.5g}" for k in
                ("recovered_k_over_m", "mu_final", "final_train_loss")
                if k in vals]
        print(f"  {name:<22} " + "  ".join(bits))

    if skipped:
        print("\n  Skipped:")
        for n, why in skipped:
            print(f"    {n:<22} {why}")

    if not matched:
        print("\n  Nothing matched - CSV left untouched.")
        return
    backup = csv_path + ".bak"
    if not os.path.exists(backup):
        pd.read_csv(csv_path).to_csv(backup, index=False)
        print(f"\n  Backed up original to {backup}")
    df.to_csv(csv_path, index=False)
    print(f"  Updated {matched} row(s) in {csv_path}")


# ======================================================================
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)

    csv_path, do_plot, paths = None, True, []
    i = 0
    while i < len(args):
        if args[i] == "--csv":
            csv_path = args[i + 1]; i += 2
        elif args[i] == "--no-plot":
            do_plot = False; i += 1
        else:
            paths.append(args[i]); i += 1

    for p in paths:
        print(f"\n=== {run_name_from_path(p)} ===")
        for k, v in sorted(collect_run_diagnostics(p).items()):
            print(f"  {k:<28} {v}")
        if do_plot:
            plot_run_diagnostics(p)

    if csv_path:
        print(f"\n=== writing to {csv_path} ===")
        backfill_csv(csv_path, paths)
