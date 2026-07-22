"""
cross_eval_2x2.py

Sim2real decomposition: evaluate BOTH checkpoints on BOTH test sets.

            eval: real test        eval: twin test
train real  [A] known ~0.10        [B] eval-side noise effect
train twin  [C] sim2real transfer  [D] known ~0.056-0.075

Because twin i shares initial conditions with real i, comparisons across the
same test set (A vs C, B vs D) and across test sets for the same model
(A vs B, C vs D) are PAIRED per trajectory - the script reports paired deltas
with SEM, which is far more powerful than comparing means.

Reads each model with ITS OWN training norms. No training, eval only.
"""

import os
import csv
import numpy as np
import torch

import wall
from train_gnn_multi_step import GNSModel
from generate_node_states import mesh_cube_surface, BLOCK_HALF_WIDTH
from evaluate_metrics import compute_metrics, BLOCK_WIDTH
from display_results import rollout_trajectory_feedback_shape_match

# ======================================================================
# CONFIG - fill in the four paths
# ======================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "real_trained": dict(
        model_path=os.path.join(script_dir, "models/CHANGE_ME_real/256_train_gns_model_best_model.pt"),
        norms_path=os.path.join(script_dir, "models/CHANGE_ME_real/256_train_gns_model_norms.pt"),
    ),
    "twin_trained": dict(
        model_path=os.path.join(script_dir, "models/CHANGE_ME_twin/256_train_gns_model_best_model.pt"),
        norms_path=os.path.join(script_dir, "models/CHANGE_ME_twin/256_train_gns_model_norms.pt"),
    ),
}

DATASETS = {
    "real_test": dict(folder=os.path.join(script_dir, "data/tosses_processed"),
                      weights_only=True, unscale=True),
    "twin_test": dict(folder=os.path.join(script_dir, "data/mojoco_paper_replica"),
                      weights_only=False, unscale=False),
}

TEST_INDICES = range(454, 568)      # identical indices exist in both folders

# architecture / features - match the training runs
H = 3
NODES_PER_EDGE = 2
K_NN = 3
LATENT, L_MP, K_REP = 128, 5, 1
USE_WIND = False

OUT_CSV = os.path.join(script_dir, "cross_eval_2x2.csv")

# ======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))
nodes_body = torch.tensor(mesh_cube_surface(BLOCK_HALF_WIDTH * 2, NODES_PER_EDGE),
                          dtype=torch.float32)


def load_model(entry):
    norms = torch.load(entry["norms_path"], weights_only=False)
    model = GNSModel(norms["x_mean"].shape[0], norms["e_mean"].shape[0],
                     latent_dim=LATENT, L=L_MP, K=K_REP)
    sd = torch.load(entry["model_path"], map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, norms


def eval_cell(model, norms, ds):
    """Returns {idx: (center, angle)} for one model on one dataset."""
    out = {}
    for i, idx in enumerate(TEST_INDICES, 1):
        try:
            pred, true, _ = rollout_trajectory_feedback_shape_match(
                ds["folder"], model, Floor, throw_number=idx,
                nodes_per_edge=NODES_PER_EDGE, nearest_neighbors=K_NN,
                rest_positions=nodes_body,
                accel_std=norms["acc_std"], accel_mean=norms["acc_mean"],
                x_mean=norms["x_mean"], x_std=norms["x_std"],
                e_mean=norms["e_mean"], e_std=norms["e_std"],
                do_shape_match=True, shape_alpha=1.0, return_edge_info=True,
                weights_only_load=ds["weights_only"],
                unscale_trajectory_data=ds["unscale"],
                h=H, use_wind=USE_WIND)
        except FileNotFoundError:
            continue
        m = compute_metrics(pred, true, nodes_body)
        out[idx] = (float(m["center_error"]), float(m["angle_error_deg"]))
        if i % 25 == 0:
            print(f"    {i}/{len(TEST_INDICES)}", flush=True)
    return out


def cell_stats(res):
    c = np.array([v[0] for v in res.values()])
    a = np.array([v[1] for v in res.values()])
    return (f"{c.mean():.4f} \u00b1 {c.std():.4f}",
            f"{a.mean():.2f} \u00b1 {a.std():.2f}", len(c))


def paired_delta(res_a, res_b, field):
    """mean +/- SEM of (b - a) over shared indices; field 0=center, 1=angle."""
    shared = sorted(set(res_a) & set(res_b))
    d = np.array([res_b[i][field] - res_a[i][field] for i in shared])
    return d.mean(), d.std() / max(1, np.sqrt(len(d))), len(d)


# ======================================================================
results = {}
for mname, mentry in MODELS.items():
    print(f"\nLoading {mname} ...")
    model, norms = load_model(mentry)
    for dname, ds in DATASETS.items():
        print(f"  evaluating {mname} on {dname} ({len(TEST_INDICES)} rollouts)")
        results[(mname, dname)] = eval_cell(model, norms, ds)

# ---- 2x2 matrix ----
print("\n" + "=" * 74)
print("2x2 CROSS-EVALUATION  (center /width | angle deg | n)")
print("=" * 74)
header = f"{'':<16}" + "".join(f"{d:>28}" for d in DATASETS)
print(header)
for mname in MODELS:
    row = f"{mname:<16}"
    for dname in DATASETS:
        c, a, n = cell_stats(results[(mname, dname)])
        row += f"{c + ' | ' + a:>28}"
    print(row)

# ---- paired decompositions ----
print("\n" + "-" * 74)
print("PAIRED DELTAS (mean \u00b1 SEM over shared trajectory indices)")
print("-" * 74)
for mname in MODELS:
    dc, se, n = paired_delta(results[(mname, "real_test")],
                             results[(mname, "twin_test")], 0)
    da, sa, _ = paired_delta(results[(mname, "real_test")],
                             results[(mname, "twin_test")], 1)
    print(f"{mname}: twin_test - real_test   center {dc:+.4f} \u00b1 {se:.4f}   "
          f"angle {da:+.2f} \u00b1 {sa:.2f}   (n={n})")
    print(f"   -> negative = scores better on clean twins = eval-side noise cost"
          if mname == "real_trained" else
          f"   -> positive = degrades on noisy real inputs = sim2real gap")
for dname in DATASETS:
    dc, se, n = paired_delta(results[("real_trained", dname)],
                             results[("twin_trained", dname)], 0)
    print(f"on {dname}: twin_trained - real_trained   center {dc:+.4f} \u00b1 {se:.4f} (n={n})")

# ---- long CSV ----
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "dataset", "traj_idx", "center_error", "angle_error_deg"])
    for (mname, dname), res in results.items():
        for idx, (c, a) in sorted(res.items()):
            w.writerow([mname, dname, idx, f"{c:.6f}", f"{a:.4f}"])
print(f"\nSaved per-trajectory results to {OUT_CSV}")
