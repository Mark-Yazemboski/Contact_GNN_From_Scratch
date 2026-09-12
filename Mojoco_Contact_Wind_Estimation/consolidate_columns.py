"""
consolidate_columns.py

ONE-TIME cleanup. Diagnostic values ended up under two prefixes:

  settings.diag_align, settings.raw_fric_dir, ...   written by the run script
  metrics.recovered_k_over_m, metrics.mu_final, ... written by the evaluator
  settings.recovered_k_over_m, ...                 written by the old backfill
                                                   scripts, now dead columns

Everything is a RESULT, not configuration, so everything belongs under
metrics.*. This folds settings.<key> into metrics.<key> for the diagnostic keys
only - real configuration (w_fric_dir, k_over_m, use_drag_baseline, ...) is
left exactly where it is - and drops the emptied duplicates.

Non-destructive: writes a .bak first, and never overwrites a metrics.* value
that is already populated.

USAGE
    python consolidate_columns.py models/all_force_runs_master.csv
    python consolidate_columns.py --dry-run models/all_force_runs_master.csv
"""

import os
import sys

import numpy as np
import pandas as pd

# Prefixes and exact names that identify a DIAGNOSTIC rather than a setting.
DIAG_PREFIXES = ("diag_", "raw_", "impulse_")
DIAG_EXACT = {
    "recovered_mu_ckpt", "recovered_k_over_m", "mu_mode", "k_mode",
    "final_train_loss", "final_train_loss_std", "final_train_loss_n",
    "best_val_loss", "best_val_epoch", "total_optimizer_steps",
    "mu_final", "mu_init", "mu_drift", "mu_tail_slope_per_1k_ep",
    "k_final", "k_init", "k_drift", "k_tail_slope_per_1k_ep",
}


def is_diag(key):
    return key.startswith(DIAG_PREFIXES) or key in DIAG_EXACT


def main(csv_path, dry_run=False):
    import warnings
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    df = pd.read_csv(csv_path).copy()
    moved, merged, dropped = [], [], []

    for col in list(df.columns):
        if not col.startswith("settings."):
            continue
        key = col[len("settings."):]
        if not is_diag(key):
            continue                       # a real setting - leave it alone

        target = f"metrics.{key}"
        src = df[col]
        if target not in df.columns:
            df[target] = src               # straight rename
            moved.append(key)
        else:
            # fill only where the metrics column has no value
            gap = df[target].isna() & src.notna()
            n = int(gap.sum())
            if n:
                if df[target].dtype != src.dtype:
                    df[target] = df[target].astype(object)
                df.loc[gap, target] = src[gap]
            merged.append((key, n))
        df.drop(columns=[col], inplace=True)
        dropped.append(col)

    if moved:
        print(f"  renamed to metrics.* ({len(moved)}): " + ", ".join(sorted(moved)))
    if merged:
        print("  merged into an existing metrics.* column:")
        for k, n in sorted(merged):
            print(f"    {k:<28} filled {n} empty row(s)")
    if not dropped:
        print("  nothing to consolidate - already clean.")
        return

    if dry_run:
        print(f"\n  dry run: would drop {len(dropped)} settings.* column(s), "
              "CSV untouched")
        return

    backup = csv_path + ".precons.bak"
    if not os.path.exists(backup):
        pd.read_csv(csv_path).to_csv(backup, index=False)
        print(f"\n  backed up to {backup}")
    df.to_csv(csv_path, index=False)
    print(f"  dropped {len(dropped)} duplicate settings.* column(s)")
    print(f"  wrote {csv_path}  ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if not args:
        sys.exit(__doc__)
    main(args[0], "--dry-run" in sys.argv)
