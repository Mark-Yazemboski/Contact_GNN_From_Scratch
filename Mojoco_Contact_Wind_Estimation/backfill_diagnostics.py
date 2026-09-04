"""
backfill_diagnostics.py

One-off. The runs finished before physics_losses.py started recording
diagnostics (Split_R_Probe_*, Split_DIR_*, and anything else already in the
master CSV) have their alignment / mu_implied / gate-occupancy / raw-term
numbers only in the training logs. This parses those logs and writes the same
columns the live path now produces, so old and new rows are comparable in one
dataframe.

USAGE
    python backfill_diagnostics.py <master_csv> <log_file_or_dir> [...]

The run name is taken from the log file's stem unless the log contains a
"models/<name>/" path, which is preferred when present. Matching against the
CSV is exact on run_name; unmatched logs are reported and skipped rather than
guessed at.

Averages the LAST 20 recorded epochs, matching summarize_diagnostics(). The
per-epoch numbers come from one batch and are noisy - on the R arm, alignment
was 0.726 +/- 0.094 with a 0.503 outlier, so the final value can differ from
the tail mean by ~6 degrees.
"""

import os
import re
import sys

import numpy as np
import pandas as pd

LAST_N = 20
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

RE_ALIGN = re.compile(rf"friction alignment\s*=\s*({NUM})")
RE_MU_IMP = re.compile(rf"mu implied by predicted forces\s*=\s*({NUM})")
RE_GATE = re.compile(rf"Slip gate\s*\|\s*OPEN\s+({NUM})\s*%")
RE_RAWLINE = re.compile(r"Physics terms \(raw\)\s*\|(.*)")
RE_RAWPAIR = re.compile(rf"([A-Za-z_][A-Za-z_0-9]*)\s*:\s*({NUM})")
RE_MODELDIR = re.compile(r"models[/\\]([A-Za-z0-9_.\-]+)[/\\]")


def parse_log(path):
    """-> (run_name_hint, {column: value}) or (hint, {}) if nothing found."""
    with open(path, "r", errors="replace") as f:
        text = f.read()

    hits = RE_MODELDIR.findall(text)
    hint = hits[-1] if hits else os.path.splitext(os.path.basename(path))[0]

    def tail_mean(values):
        v = np.array(values[-LAST_N:], dtype=float)
        v = v[np.isfinite(v)]
        return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0,
                v.size) if v.size else (np.nan, np.nan, 0)

    out = {}
    al = [float(m) for m in RE_ALIGN.findall(text)]
    mi = [float(m) for m in RE_MU_IMP.findall(text)]
    gf = [float(m) / 100.0 for m in RE_GATE.findall(text)]

    if al:
        m, sd, n = tail_mean(al)
        out["diag_align"] = m
        out["diag_align_std"] = sd
        out["diag_misalign_deg"] = float(np.degrees(np.arccos(min(1.0, max(-1.0, m)))))
        out["diag_n_epochs"] = n
    if mi:
        m, sd, _ = tail_mean(mi)
        out["diag_mu_implied"] = m
        out["diag_mu_implied_std"] = sd
    if gf:
        out["diag_gate_frac"] = tail_mean(gf)[0]

    raws = {}
    for line in RE_RAWLINE.findall(text):
        for k, v in RE_RAWPAIR.findall(line):
            raws.setdefault(k, []).append(float(v))
    for k, vals in raws.items():
        out[f"raw_{k}"] = tail_mean(vals)[0]

    return hint, out


def collect_logs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names
                          if n.endswith((".log", ".out", ".txt"))]
        else:
            files.append(p)
    return sorted(set(files))


def main(csv_path, log_paths):
    df = pd.read_csv(csv_path)
    if "run_name" not in df.columns:
        sys.exit("ERROR: no run_name column - is this the right master CSV?")
    known = set(df["run_name"].astype(str))

    matched, skipped = 0, []
    for path in collect_logs(log_paths):
        hint, vals = parse_log(path)
        if not vals:
            skipped.append((path, "no diagnostic lines found"))
            continue
        if hint not in known:
            skipped.append((path, f"run_name {hint!r} not in CSV"))
            continue
        rows = df["run_name"].astype(str) == hint
        for col, v in vals.items():
            key = f"settings.{col}"
            if key not in df.columns:
                df[key] = np.nan
            df.loc[rows, key] = v
        matched += 1
        print(f"  {hint:<26} "
              + "  ".join(f"{k.replace('diag_', '').replace('raw_', ''):>12}="
                          f"{v:.4g}" for k, v in sorted(vals.items())
                          if k in ("diag_align", "diag_misalign_deg",
                                   "diag_mu_implied", "raw_fric_dir")))

    if skipped:
        print("\nSkipped:")
        for p, why in skipped:
            print(f"  {os.path.basename(p):<40} {why}")

    if matched:
        backup = csv_path + ".bak"
        if not os.path.exists(backup):
            pd.read_csv(csv_path).to_csv(backup, index=False)
            print(f"\nBacked up original to {backup}")
        df.to_csv(csv_path, index=False)
        print(f"Updated {matched} row(s) in {csv_path}")
    else:
        print("\nNothing matched - CSV left untouched.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
