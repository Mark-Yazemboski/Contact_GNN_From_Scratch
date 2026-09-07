"""
rescue_runs.py

Writes master-CSV rows for runs that TRAINED and EVALUATED successfully but
crashed in run_force_multi_step.py before save_run_report was reached (the
`{v:.6g}` on a string bug). Training is not repeated - only evaluation, which
takes about a minute per run against the saved checkpoints.

WHAT IT READS
  models/<run>/*_norms.pt          force_cfg: the physics knobs for THAT run
  models/<run>/*_physics.pt        recovered mu / k
  models/<run>/*_loss_history.pt   loss curves, mu_trace, k_trace

WHAT IT CANNOT READ
  The architecture settings (n_train, multistep, batch size, ...) are not in
  any checkpoint. They are identical across every run in this sweep, so they
  live in SHARED below. CHECK THEM against your run_force_multi_step.py before
  running - a wrong value here writes a wrong CSV row, which is worse than a
  missing one.

  The per-epoch diagnostics (diag_align, diag_mu_implied, raw_fric_*) come from
  physics_losses.DIAG_HISTORY, which only exists inside a training process.
  Recover those separately from the logs:

      python backfill_diagnostics.py models/all_force_runs_master.csv logs/

USAGE
  python rescue_runs.py MAG_1e4_1 MAG_1e4_2 K_LearnBad_1
  python rescue_runs.py --dry-run MAG_1e4_1          # print, write nothing
"""

import os
import sys

import torch

from evaluate_force_model import evaluate_force_model
from run_report import save_run_report
from run_diagnostics import collect_run_diagnostics
from generate_node_states import BLOCK_HALF_WIDTH

script_dir = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# SHARED - identical across every run in this sweep. VERIFY against
# run_force_multi_step.py before use.
# ======================================================================
DATA = os.path.join(script_dir, "data/mojoco_paper_replica_20_wind")
MASTER_CSV = os.path.join(script_dir, "models", "all_force_runs_master.csv")
MODELS_DIR = os.path.join(script_dir, "models")

Num_total_trajectories = 569
Num_train = int(0.5 * Num_total_trajectories)
Num_val = int(0.3 * Num_total_trajectories)
Used_Num_train_trajectories = 256
train_range = range(0, Used_Num_train_trajectories)
val_range = range(Num_train, Num_train + Num_val)
test_range = range(Num_train + Num_val, Num_total_trajectories - 1)

SHARED = dict(
    architecture="force",
    dataset=DATA,
    n_train=Used_Num_train_trajectories,
    train_range=f"{train_range.start}-{train_range.stop}",
    val_range=f"{val_range.start}-{val_range.stop}",
    test_range=f"{test_range.start}-{test_range.stop}",
    nodes_per_edge=2,
    nearest_neighbors=3,
    message_passing_layers=5,
    repeat_blocks=1,
    latent_dim=128,
    pos_history=3,
    batch_size=512,
    learning_rate=1e-4,
    epochs=10000,
    noise_scale=3e-4 * BLOCK_HALF_WIDTH,
    rot_noise_scale=None,
    multistep=4,
    curriculum_epochs=50,
    scheduler=None,
    use_wind=True,
)

# Physics knobs taken from the run's own _norms.pt, never from SHARED.
FROM_CFG = ["dt", "gravity", "mass", "loss_mode",
            "use_drag_baseline", "k_over_m", "learn_k", "fix_k",
            "contact_d0", "contact_tau",
            "w_diss", "w_sparse", "w_fric_dir", "w_fric_mag", "w_fric_cone",
            "w_fluid_anchor", "w_fluid_smooth",
            "mu_init", "learn_mu", "fix_mu"]


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        pass
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"    could not read {os.path.basename(path)}: {e}")
        return None


def _find(folder, suffix):
    hits = sorted(f for f in os.listdir(folder) if f.endswith(suffix))
    return os.path.join(folder, hits[0]) if hits else None


def rescue(run_name, dry_run=False):
    folder = os.path.join(MODELS_DIR, run_name)
    if not os.path.isdir(folder):
        print(f"  {run_name}: no such folder"); return False

    norms = _find(folder, "_norms.pt")
    if norms is None:
        print(f"  {run_name}: no _norms.pt - did training finish?"); return False
    nd = _load(norms) or {}
    cfg = nd.get("force_cfg", nd)          # some versions nest it, some do not
    have = [k for k in FROM_CFG if k in cfg]
    missing = [k for k in FROM_CFG if k not in cfg]
    print(f"  {run_name}: force_cfg supplied {len(have)}/{len(FROM_CFG)} physics keys")
    if missing:
        print(f"    NOT in _norms.pt (omitted from the row): {missing}")

    stem = None
    for suf in ("_physics.pt", "_loss_history.pt", "_norms.pt"):
        p = _find(folder, suf)
        if p:
            stem = p[: -len(suf)] + ".pt"; break

    settings = dict(SHARED)
    settings.update({k: cfg[k] for k in have})
    settings.update(collect_run_diagnostics(stem))

    key = ("w_fric_dir", "w_fric_mag", "w_fric_cone", "k_over_m", "learn_k")
    print("    " + "  ".join(f"{k}={settings.get(k)}" for k in key if k in settings))

    if dry_run:
        print("    (dry run - nothing written)"); return True

    metrics = evaluate_force_model(
        model_folder=folder, data_folder=DATA, test_indices=test_range,
        weights_only=False, unscale=False)
    save_run_report(folder, settings, metrics, slopes=[],
                    run_name=run_name, master_csv=MASTER_CSV)
    print(f"    wrote row: center={metrics.get('center_error'):.4f} "
          f"contact_err={metrics.get('force_contact_err_contact'):.4f} "
          f"mu={metrics.get('recovered_mu'):.4f}")
    return True


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    if not args:
        sys.exit(__doc__)
    ok = sum(rescue(r, dry) for r in args)
    print(f"\n{ok}/{len(args)} run(s) processed"
          + ("" if dry else f" -> {MASTER_CSV}"))
    if not dry:
        print("\nNow recover the per-epoch diagnostics from the logs:")
        print(f"  python backfill_diagnostics.py {MASTER_CSV} <log dir>")
