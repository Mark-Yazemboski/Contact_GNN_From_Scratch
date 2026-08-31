"""
calibrate_physics_weights.py

NEW FILE - does not modify anything. Answers one question:

    "What should w_diss / w_fluid_anchor / w_fluid_smooth / w_sparse be?"

WHY THIS EXISTS
---------------
physics_losses.py states the calibration rule in its NORMALIZATION section:
set each gamma so that (gamma * raw) is 1-10% of the prediction loss. The raw
magnitudes are printed every epoch, but they differ by ORDERS OF MAGNITUDE
between terms - h_contact_sparsity is a linear L1 while the others are
quadratic, so a gamma that is sane for the anchor is meaningless for sparsity.
Guessing a decade wrong wastes a full 10k-epoch sweep.

This script runs a few epochs with all four terms switched on at a NEGLIGIBLE
probe weight, reads the raws the trainer already prints, and solves

    gamma_j  =  TARGET_FRAC * L_pred / raw_j

for every term at once. Cost: a few epochs, versus a sweep per term.

WHY A PROBE WEIGHT OF 1e-8 AND NOT 1.0
--------------------------------------
Two reasons, both of which would silently corrupt the answer at gamma = 1:

  1. compute_step_terms() computes a term if and only if its weight is > 0, so
     any positive number switches the raw printing on. 1e-8 gets the numbers
     without letting the physics move the model - the raws stay a measurement
     of the CURRENT model rather than of a model the probe already distorted.
  2. The printed "Train Loss" is L_pred + sum_j gamma_j h_j. At gamma = 1 the
     physics terms would dominate that number and the calibration would be
     solving against its own contamination. At 1e-8 the printed train loss IS
     L_pred to seven digits.

WHY curriculum_epochs IS FORCED TO 0
------------------------------------
run_force_multi_step.py uses curriculum_epochs=50, so epochs 0-49 run at K=1.
h_fluid_temporal_smooth returns exactly 0 at K=1 - calibrating on the first
epochs of a curriculum run would divide by zero and hand you a garbage weight
for the one term you most wanted to size. This script pins K to its final
value from epoch 0.

USAGE
-----
    python calibrate_physics_weights.py

Everything below the CONFIG block is mechanical. Keep CONFIG in sync with
run_force_multi_step.py - anything that changes the force scale (dataset, dt,
mass, use_drag_baseline, multistep) changes the raws, so calibrate with the
same settings you intend to sweep under.
"""

import contextlib
import io
import json
import os
import re
import sys

import torch

import wall
from train_force_gns import train_force_gnn
from generate_node_states import BLOCK_HALF_WIDTH


torch.set_float32_matmul_precision('high')
script_dir = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# CONFIG - mirror of run_force_multi_step.py
# ======================================================================
CAL_EPOCHS = 3          # >1 so you can see whether the raws are stable
PROBE_WEIGHT = 1e-8     # switches the terms on without perturbing training
TARGET_FRAC = 0.03      # aim each weighted term at 3% of the prediction loss
                        # (physics_losses.py recommends the 1-10% band)
INCLUDE_SPARSE = False  # sparsity is off by design - it shrinks the legitimate
                        # resting normal forces. Flip on only to size it for a
                        # deliberate ablation.

trajectory_folder = os.path.join(script_dir, "data/mojoco_paper_replica_0_wind")

Num_total_trajectories = 569
training_percentage = 0.5
validation_percentage = 0.3
Num_train = int(training_percentage * Num_total_trajectories)
Num_val = int(validation_percentage * Num_total_trajectories)
Used_Num_train_trajectories = 256

train_range = range(0, Used_Num_train_trajectories)
val_range = range(Num_train, Num_train + Num_val)

# --- architecture / recipe ---
nodes_per_edge = 2
K_nearest_neighbors = 3
message_passing_layers = 5
repeat_blocks = 1
Latent_dimension = 128
pos_history = 3
batch_size = 512
learning_rate = 1e-4
noise_scale = 3e-4 * BLOCK_HALF_WIDTH
rot_noise_scale = None
multistep = 4                    # MUST be >= 2 or w_fluid_smooth cannot be sized
accumulation_steps = 1
weights_only_load = False
unscale_trajectory_data = False

# --- force-model physics ---
DT = 1.0 / 148.0
GRAVITY = None
MASS = 0.37
use_wind_feature = True
use_drag_baseline = True         # CHANGES THE ANCHOR'S MEANING - see notes below
K_OVER_M = 0.0285
contact_d0 = 0.02
contact_tau = 0.005
loss_mode = "accel"
MU_INIT = 0.3
LEARN_MU = True
FIX_MU = None

# Scratch output - never touches your real model folders or the master CSV.
CAL_FOLDER = os.path.join(script_dir, "models", "_calibration")
os.makedirs(CAL_FOLDER, exist_ok=True)
save_model_path = os.path.join(CAL_FOLDER, "calibration_probe.pt")

# ======================================================================
# Trainer output parsing
# ----------------------------------------------------------------------
# The trainer prints, once per epoch, a line like
#   Physics terms (raw) | diss: 1.23e-02 | fluid_anchor: 4.5e-01 | ...
# and an epoch line containing "Train Loss: <number>". Both are matched
# loosely so a reformat upstream does not break this file; if parsing fails
# the script dumps the candidate lines it saw so you can fix these two
# constants in one place.
# ======================================================================
RAW_MARKER = "Physics terms"
LOSS_MARKER = "Train Loss"
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_SKIP_KEYS = {"terms", "raw", "physics"}


class _Tee(io.TextIOBase):
    """Forward the trainer's output to the console AND to a buffer, so a long
    run still streams normally on ROAR while we keep a copy to parse."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def parse_raw_rows(text):
    """-> list of {term_name: raw_value}, one dict per epoch that printed."""
    rows = []
    for line in text.splitlines():
        if RAW_MARKER not in line:
            continue
        pairs = re.findall(rf"([A-Za-z_][A-Za-z_0-9]*)\s*[:=]\s*({_NUM})", line)
        row = {k: float(v) for k, v in pairs if k.lower() not in _SKIP_KEYS}
        if row:
            rows.append(row)
    return rows


def parse_losses(text):
    return [float(m.group(1)) for m in
            re.finditer(rf"{LOSS_MARKER}\s*[:=]\s*({_NUM})", text)]


def dump_candidates(text):
    print("\n  Lines containing 'Physics' or 'Loss' (for fixing the markers):")
    seen = 0
    for line in text.splitlines():
        if "Physics" in line or "Loss" in line:
            print("   |", line.strip()[:150])
            seen += 1
            if seen >= 12:
                break
    if seen == 0:
        print("   (none - the trainer printed no physics or loss lines at all)")


# ======================================================================
# Run the probe
# ======================================================================
print("=" * 74)
print("PHYSICS-WEIGHT CALIBRATION PROBE")
print("=" * 74)
print(f"  epochs             = {CAL_EPOCHS}")
print(f"  probe weight       = {PROBE_WEIGHT:g}  (all terms on, none influential)")
print(f"  target fraction    = {TARGET_FRAC:.0%} of the prediction loss")
print(f"  multistep K        = {multistep}   curriculum FORCED OFF")
print(f"  use_drag_baseline  = {use_drag_baseline}")
print(f"  dataset            = {trajectory_folder}")
print(f"  scratch folder     = {CAL_FOLDER}")
print("=" * 74 + "\n")

if multistep < 2:
    print("WARNING: multistep < 2, so h_fluid_temporal_smooth is identically 0")
    print("         and w_fluid_smooth CANNOT be calibrated from this run.\n")

buf = io.StringIO()
tee = _Tee(sys.stdout, buf)

with contextlib.redirect_stdout(tee):
    train_force_gnn(
        Wall=wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1)),
        train_range=train_range,
        val_range=val_range,
        save_model_path=save_model_path,
        trajectory_folder=trajectory_folder,
        epochs=CAL_EPOCHS,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        lr=learning_rate,
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=K_nearest_neighbors,
        h=pos_history,
        message_passing_layers=message_passing_layers,
        repeat_blocks=repeat_blocks,
        latent_dim=Latent_dimension,
        weights_only=weights_only_load,
        unscale_data=unscale_trajectory_data,
        noise_scale=noise_scale,
        rot_noise_scale=rot_noise_scale,
        multistep=multistep,
        curriculum_epochs=0,            # pin K from epoch 0 - see module docstring
        curriculum_schedule=None,
        Learning_Rate_Scheduler=None,
        use_wind=use_wind_feature,
        dt=DT,
        gravity=GRAVITY,
        mass=MASS,
        use_drag_baseline=use_drag_baseline,
        k_over_m=K_OVER_M,
        contact_d0=contact_d0,
        contact_tau=contact_tau,
        loss_mode=loss_mode,
        w_diss=PROBE_WEIGHT,
        w_sparse=PROBE_WEIGHT if INCLUDE_SPARSE else 0.0,
        w_fluid_anchor=PROBE_WEIGHT,
        w_fluid_smooth=PROBE_WEIGHT,
        mu_init=MU_INIT, learn_mu=LEARN_MU, fix_mu=FIX_MU,
        validation_check_interval=10 ** 9,   # skip validation, it costs minutes
        epoch_checkpoint_interval=10 ** 9,
        keep_last_n_checkpoints=0,
    )

text = buf.getvalue()

# ======================================================================
# Report
# ======================================================================
print("\n" + "=" * 74)
print("CALIBRATION RESULT")
print("=" * 74)

rows = parse_raw_rows(text)
losses = parse_losses(text)

if not rows:
    print("  Could not find any physics-raw line.")
    print(f"  Looked for lines containing: {RAW_MARKER!r}")
    print("  Most likely cause: every weight reached compute_step_terms as 0,")
    print("  or the print marker changed. Check that PROBE_WEIGHT is reaching")
    print("  the trainer and that the header said 'physics losses ON'.")
    dump_candidates(text)
    sys.exit(1)

if not losses:
    print("  Found physics raws but no prediction loss.")
    print(f"  Looked for lines containing: {LOSS_MARKER!r}")
    dump_candidates(text)
    sys.exit(1)

last = rows[-1]
L_pred = losses[-1]     # == L_pred to 7 digits, because PROBE_WEIGHT is 1e-8

print(f"  prediction loss (last epoch) = {L_pred:.6e}")
print(f"  epochs of raws captured      = {len(rows)}\n")

hdr = f"  {'term':<16}{'raw':>13}{'drift':>9}{'gamma':>13}{'gamma*raw':>13}{'% of Lpred':>12}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

recommended, problems = {}, []
for term in sorted(last):
    raw = last[term]
    series = [r[term] for r in rows if term in r]
    drift = (max(series) / min(series)) if (len(series) > 1 and min(series) > 0) else float('nan')

    if not (raw > 0) or raw != raw:
        print(f"  {term:<16}{raw:>13.3e}{'--':>9}{'--':>13}{'--':>13}{'--':>12}")
        problems.append(term)
        continue

    gamma = TARGET_FRAC * L_pred / raw
    recommended[term] = gamma
    print(f"  {term:<16}{raw:>13.3e}{drift:>9.2f}{gamma:>13.2e}"
          f"{gamma * raw:>13.2e}{100 * gamma * raw / L_pred:>11.1f}%")

print("\n  drift = max/min of the raw across the probe epochs. Near 1.0 means the")
print("  scale is settled; above ~3 means the term is still moving and the")
print("  calibrated weight is provisional - re-check it after the first real run.")

if problems:
    print("\n  ZERO OR NON-FINITE RAWS: " + ", ".join(problems))
    print("    fluid_smooth = 0        -> K collapsed to 1, or the term is not")
    print("                              wired into the trainer at all. Check:")
    print("                              grep -n 'h_fluid_temporal_smooth' train_force_gns.py")
    print("    anything else = 0       -> the term's gate never fired. For diss")
    print("                              that means slip_v0 is above every slip")
    print("                              speed in the data; sweep slip_v0, not w_diss.")

missing = {"diss", "fluid_anchor", "fluid_smooth"} - set(last)
if missing:
    print(f"\n  TERMS THAT NEVER PRINTED: {', '.join(sorted(missing))}")
    print("    These are not reaching compute_step_terms / weighted_total at all.")
    print("    A term that never prints also never contributes, whatever weight")
    print("    you give it in run_force_multi_step.py.")

# ---- paste-ready block ----
if recommended:
    print("\n" + "=" * 74)
    print("PASTE INTO run_force_multi_step.py")
    print("=" * 74)
    for term, gamma in sorted(recommended.items()):
        print(f"w_{term:<14} = {gamma:.2e}")
    if not INCLUDE_SPARSE:
        print("w_sparse        = 0.0    # off by design (shrinks resting normal forces)")

    print("\n" + "=" * 74)
    print("SWEEP LADDER  (one term at a time; hold the others at calibrated)")
    print("=" * 74)
    print(f"  {'term':<16}{'low (/3)':>13}{'mid':>13}{'high (x3)':>13}")
    print("  " + "-" * 53)
    for term, gamma in sorted(recommended.items()):
        print(f"  {term:<16}{gamma / 3:>13.2e}{gamma:>13.2e}{gamma * 3:>13.2e}")
    print("\n  Rank cells by the WRENCH-DECOMPOSITION metric, not center error -")
    print("  these terms constrain a direction the position loss cannot see, so")
    print("  ranking on fit alone always picks weight ~ 0. Use center error as a")
    print("  guardrail instead: accept the largest weight whose center error is")
    print("  still within ~2 sigma (about 0.008) of the no-physics baseline.")

out_path = os.path.join(CAL_FOLDER, "calibrated_weights.json")
with open(out_path, "w") as f:
    json.dump(dict(prediction_loss=L_pred,
                   raws_per_epoch=rows,
                   recommended={f"w_{k}": v for k, v in recommended.items()},
                   target_frac=TARGET_FRAC,
                   probe_weight=PROBE_WEIGHT,
                   cal_epochs=CAL_EPOCHS,
                   multistep=multistep,
                   use_drag_baseline=use_drag_baseline,
                   dataset=trajectory_folder), f, indent=2)
print(f"\n  Saved to {out_path}")

print("\n  NOTE ON use_drag_baseline: with it TRUE, a_fluid_total already contains")
print("  the analytic drag and the anchor only keeps the learned RESIDUAL small.")
print("  With it FALSE the anchor pulls the whole fluid force onto the law. Those")
print("  are different constraints with different natural weights - recalibrate if")
print("  you change it, and pin it for the whole sweep.")
print("\nDone.")
