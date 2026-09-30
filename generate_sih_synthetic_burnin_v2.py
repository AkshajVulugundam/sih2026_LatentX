#!/usr/bin/env python3
"""
SIH26170 — SIH-specific synthetic burn-in dataset generator (v2)

PURPOSE
-------
Controlled, explainable synthetic burn-in data that mirrors the SIH problem
structure. It is NOT ISRO data and NOT NASA data. All limits and
distributions are SIMULATION ASSUMPTIONS, not ISRO specifications.

WHAT CHANGED FROM v1 (found by the validation report)
-----------------------------------------------------
Bug fix (needed for everything below)
  * v1 re-drew slope_mult / step_mult separately at every time point. A
    component's defect "personality" is now drawn ONCE per component.

Fix 1 - LATENT_DEFECT was not latent
  * v1 overwrote the 96h/168h values with a straight line from the 24h value
    to ~90% of the limit, so 96h was already halfway there and obvious.
  * v2 uses a convex (slow-then-fast) acceleration: 96h is only slightly
    elevated, most of the rise happens between 96h and 168h. The 168h target
    is still 75-95% of the limit, so latents stay below the limit.

Fix 2 - GRADUAL_DRIFT was just NORMAL scaled up ~4x
  * v2 has two severities: 12 MILD (1.4-2.2x normal slope, overlaps the
    normal spread early) and 18 SEVERE (3.0-4.6x).

Fix 3 - SUDDEN_ANOMALY step was always in 24->96h
  * v2: exactly half (10) have the step by 96h (SUDDEN_BY_96H), and half (10)
    only after 96h (SUDDEN_AFTER_96H, first visible at 168h).

Fix 4 - batch-relative detection barely mattered
  * Batch-to-batch spread is x1.5 larger than v1 (lot-to-lot variation
    bigger than within-lot), and batch-anomaly shifts now range from
    subtle (1.0x) to clear (2.5x) instead of always clear.

GROUND-TRUTH COLUMNS (evaluation only - NEVER model inputs)
  Defect_Type, Defect_Subtype

OUTPUTS (written to ./sih_synthetic_v2/)
  sih_synthetic_burnin_long.csv, sih_synthetic_burnin_wide.csv,
  sih_synthetic_summary.csv, sih_synthetic_generator_report.txt,
  three trajectory PNGs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 26170
BATCHES = 10
COMPONENTS_PER_BATCH = 50
TIME_POINTS = [0, 24, 96, 168]
OUT_DIR = Path("sih_synthetic_v2")

# Simulation-only absolute limits.
LIMITS = {"Iddq_mA": 5.0, "Leakage_uA": 200.0, "Propagation_Delay_ns": 15.0}
BASE = {"Iddq_mA": 2.0, "Leakage_uA": 60.0, "Propagation_Delay_ns": 8.0}
COMPONENT_SD = {"Iddq_mA": 0.10, "Leakage_uA": 3.0, "Propagation_Delay_ns": 0.12}

BATCH_SD_V1 = {"Iddq_mA": 0.08, "Leakage_uA": 2.5, "Propagation_Delay_ns": 0.10}
BATCH_SD_SCALE = 1.5  # Fix 4
BATCH_SD = {k: v * BATCH_SD_SCALE for k, v in BATCH_SD_V1.items()}

NORMAL_SLOPE = {  # per hour
    "Iddq_mA": 0.0040,
    "Leakage_uA": 0.090,
    "Propagation_Delay_ns": 0.0045,
}
MEASUREMENT_SD = {"Iddq_mA": 0.025, "Leakage_uA": 1.2, "Propagation_Delay_ns": 0.035}

# ---- Defect configuration (all simulation assumptions) --------------------
DEFECT_COUNTS = {
    "NORMAL": 390,
    "BATCH_ANOMALY": 40,
    "GRADUAL_DRIFT": 30,
    "SUDDEN_ANOMALY": 20,
    "LATENT_DEFECT": 20,
}
GRADUAL_MILD_COUNT = 12          # Fix 2
GRADUAL_MILD_SLOPE = (1.4, 2.2)
GRADUAL_SEVERE_SLOPE = (3.0, 4.6)
SUDDEN_BY_96H_COUNT = 10         # Fix 3 (rest are AFTER_96H)
SUDDEN_STEP_FRAC = (0.18, 0.35)  # step size as fraction of BASE value
BATCH_OFFSET_MULT = (1.0, 2.5)   # Fix 4 (v1: 1.7-2.5)
LATENT_SLOPE = (1.6, 2.4)        # Fix 1
LATENT_EXPONENT = (3.0, 5.0)     # bigger = later, sharper acceleration
LATENT_TARGET_FRAC = (0.75, 0.95)  # 168h value as fraction of limit


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def draw_component_profile(defect: str, subtype: str, rng: np.random.Generator) -> dict:
    """Draw a component's defect parameters ONCE (not per time point)."""
    p = {
        "slope_mult": 1.0,
        "offset_mult": 0.0,
        "step_mult": 0.0,
        "step_at": None,
        "latent_exponent": None,
        "latent_target_frac": None,
    }
    if defect == "NORMAL":
        p["slope_mult"] = rng.normal(1.0, 0.05)
    elif defect == "BATCH_ANOMALY":
        p["slope_mult"] = rng.normal(1.0, 0.05)
        p["offset_mult"] = rng.uniform(*BATCH_OFFSET_MULT)
    elif defect == "GRADUAL_DRIFT":
        lo, hi = GRADUAL_MILD_SLOPE if subtype == "GRADUAL_MILD" else GRADUAL_SEVERE_SLOPE
        p["slope_mult"] = rng.uniform(lo, hi)
    elif defect == "SUDDEN_ANOMALY":
        p["slope_mult"] = rng.normal(1.0, 0.08)
        p["step_mult"] = rng.uniform(*SUDDEN_STEP_FRAC)
        p["step_at"] = 96 if subtype == "SUDDEN_BY_96H" else 168
    elif defect == "LATENT_DEFECT":
        p["slope_mult"] = rng.uniform(*LATENT_SLOPE)
        p["latent_exponent"] = rng.uniform(*LATENT_EXPONENT)
        p["latent_target_frac"] = {f: rng.uniform(*LATENT_TARGET_FRAC) for f in BASE}
    else:
        raise ValueError(defect)
    return p


def build_dataset():
    rng = np.random.default_rng(SEED)

    defect_plan = []
    for d, n in DEFECT_COUNTS.items():
        defect_plan += [d] * n
    rng.shuffle(defect_plan)

    # Subtype pools (shuffled) so counts are exact.
    grad_pool = ["GRADUAL_MILD"] * GRADUAL_MILD_COUNT + ["GRADUAL_SEVERE"] * (
        DEFECT_COUNTS["GRADUAL_DRIFT"] - GRADUAL_MILD_COUNT
    )
    sud_pool = ["SUDDEN_BY_96H"] * SUDDEN_BY_96H_COUNT + ["SUDDEN_AFTER_96H"] * (
        DEFECT_COUNTS["SUDDEN_ANOMALY"] - SUDDEN_BY_96H_COUNT
    )
    rng.shuffle(grad_pool)
    rng.shuffle(sud_pool)
    subtype_plan = []
    for d in defect_plan:
        if d == "GRADUAL_DRIFT":
            subtype_plan.append(grad_pool.pop())
        elif d == "SUDDEN_ANOMALY":
            subtype_plan.append(sud_pool.pop())
        else:
            subtype_plan.append(d)

    batch_effects = {
        b: {f: rng.normal(0, BATCH_SD[f]) for f in BASE}
        for b in range(1, BATCHES + 1)
    }

    rows = []
    idx = 0
    for b in range(1, BATCHES + 1):
        for _ in range(COMPONENTS_PER_BATCH):
            idx += 1
            component_id = f"C{idx:04d}"
            defect = defect_plan[idx - 1]
            subtype = subtype_plan[idx - 1]

            component_offsets = {f: rng.normal(0, COMPONENT_SD[f]) for f in BASE}
            batch_anomaly_sign = rng.choice([-1.0, 1.0])
            sensitivity = {
                "Iddq_mA": rng.uniform(0.85, 1.20),
                "Leakage_uA": rng.uniform(0.80, 1.25),
                "Propagation_Delay_ns": rng.uniform(0.85, 1.15),
            }
            prof = draw_component_profile(defect, subtype, rng)

            for t in TIME_POINTS:
                row = {
                    "Component_ID": component_id,
                    "Batch_ID": f"B{b:02d}",
                    "Time_h": t,
                    "Defect_Type": defect,
                    "Defect_Subtype": subtype,
                }
                for f in BASE:
                    base_value = BASE[f] + batch_effects[b][f] + component_offsets[f]
                    aging = NORMAL_SLOPE[f] * t * prof["slope_mult"] * sensitivity[f]

                    if defect == "BATCH_ANOMALY":
                        anomaly_offset = (
                            prof["offset_mult"]
                            * batch_anomaly_sign
                            * max(COMPONENT_SD[f] * 2.2, 0.04 * BASE[f])
                        )
                    else:
                        anomaly_offset = 0.0

                    if prof["step_at"] is not None and t >= prof["step_at"]:
                        step_effect = prof["step_mult"] * BASE[f] * sensitivity[f]
                    else:
                        step_effect = 0.0

                    if defect == "LATENT_DEFECT":
                        aging_168 = NORMAL_SLOPE[f] * 168 * prof["slope_mult"] * sensitivity[f]
                        target = LIMITS[f] * prof["latent_target_frac"][f]
                        amp = max(target - (base_value + aging_168), 0.0)
                        late = amp * (max(t - 24, 0) / 144.0) ** prof["latent_exponent"]
                    else:
                        late = 0.0

                    value = (
                        base_value + aging + anomaly_offset + step_effect + late
                        + rng.normal(0, MEASUREMENT_SD[f])
                    )
                    value = max(value, 0.001)
                    row[f] = value
                    row[f"{f}_Absolute_Limit"] = LIMITS[f]
                    row[f"{f}_Absolute_Fail"] = bool(value > LIMITS[f])
                rows.append(row)

    long_df = pd.DataFrame(rows)

    # ---- wide format --------------------------------------------------
    id_cols = ["Component_ID", "Batch_ID", "Defect_Type", "Defect_Subtype"]
    wide_parts = []
    for f in BASE:
        p = long_df.pivot(index=id_cols, columns="Time_h", values=f).reset_index()
        p.columns = [c if c in id_cols else f"{f}_{int(c)}h" for c in p.columns]
        wide_parts.append(p)
    wide = wide_parts[0]
    for p in wide_parts[1:]:
        wide = wide.merge(p, on=id_cols, how="inner")

    for f in BASE:
        wide[f"{f}_Delta_0_24"] = wide[f"{f}_24h"] - wide[f"{f}_0h"]
        wide[f"{f}_Delta_24_96"] = wide[f"{f}_96h"] - wide[f"{f}_24h"]
        wide[f"{f}_Delta_96_168"] = wide[f"{f}_168h"] - wide[f"{f}_96h"]
        wide[f"{f}_Delta_24_168"] = wide[f"{f}_168h"] - wide[f"{f}_24h"]
        wide[f"{f}_168h_Absolute_Fail"] = wide[f"{f}_168h"] > LIMITS[f]
    wide["Any_168h_Absolute_Fail"] = wide[[f"{f}_168h_Absolute_Fail" for f in BASE]].any(axis=1)

    # ---- summary ------------------------------------------------------
    late = long_df[long_df["Time_h"] == 168]
    fail_cols = [f"{f}_Absolute_Fail" for f in BASE]
    summary_rows = []
    for (defect, sub), g in late.groupby(["Defect_Type", "Defect_Subtype"]):
        summary_rows.append(
            {
                "Defect_Type": defect,
                "Defect_Subtype": sub,
                "Components": g["Component_ID"].nunique(),
                "Any_Parameter_Fail_168h_pct": 100.0 * g[fail_cols].any(axis=1).mean(),
                "Mean_Iddq_168h_mA": g["Iddq_mA"].mean(),
                "Mean_Leakage_168h_uA": g["Leakage_uA"].mean(),
                "Mean_Propagation_Delay_168h_ns": g["Propagation_Delay_ns"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    long_df.to_csv(OUT_DIR / "sih_synthetic_burnin_long.csv", index=False)
    wide.to_csv(OUT_DIR / "sih_synthetic_burnin_wide.csv", index=False)
    summary.to_csv(OUT_DIR / "sih_synthetic_summary.csv", index=False)

    # ---- plots ----------------------------------------------------------
    sample_ids = (
        long_df["Component_ID"].drop_duplicates()
        .sample(n=80, random_state=SEED)
    )
    for f, fn in [
        ("Iddq_mA", "sih_synthetic_iddq_trajectories.png"),
        ("Leakage_uA", "sih_synthetic_leakage_trajectories.png"),
        ("Propagation_Delay_ns", "sih_synthetic_delay_trajectories.png"),
    ]:
        plt.figure(figsize=(10, 6))
        for _, g in long_df[long_df["Component_ID"].isin(sample_ids)].groupby("Component_ID"):
            g = g.sort_values("Time_h")
            plt.plot(g["Time_h"], g[f], linewidth=0.8, alpha=0.35)
        med = long_df[long_df["Defect_Type"] == "NORMAL"].groupby("Time_h")[f].median()
        plt.plot(med.index, med.values, linewidth=2.5, label="NORMAL median")
        plt.axhline(LIMITS[f], linewidth=2.0, linestyle="--", label=f"Absolute limit = {LIMITS[f]}")
        plt.xlabel("Burn-in time (hours)")
        plt.ylabel(f)
        plt.title(f"SIH synthetic burn-in trajectories (v2) - {f}")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUT_DIR / fn, dpi=160)
        plt.close()

    return long_df, wide, summary


def main():
    print("=" * 78)
    print("SIH26170 — SYNTHETIC BURN-IN DATA GENERATOR (v2)")
    print("=" * 78)
    print(f"Random seed: {SEED}")
    print(f"Population: {BATCHES} batches × {COMPONENTS_PER_BATCH} components = {BATCHES * COMPONENTS_PER_BATCH}")
    print(f"Time points: {TIME_POINTS}")
    print("\nIMPORTANT: All numerical limits and distributions are simulation")
    print("assumptions for a prototype. They are NOT ISRO specifications.")

    long_df, wide, summary = build_dataset()

    print("\n" + "=" * 78 + "\nDATASET SHAPE\n" + "=" * 78)
    print(f"Long rows:  {len(long_df):,}")
    print(f"Wide rows:  {len(wide):,}")
    print(f"Components: {long_df['Component_ID'].nunique():,}")
    print(f"Batches:    {long_df['Batch_ID'].nunique():,}")

    print("\n" + "=" * 78 + "\nDEFECT DISTRIBUTION (Type / Subtype)\n" + "=" * 78)
    print(wide.groupby(["Defect_Type", "Defect_Subtype"]).size().to_string())

    print("\n" + "=" * 78 + "\n168h ABSOLUTE-THRESHOLD CHECK\n" + "=" * 78)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 78 + "\nNOTES\n" + "=" * 78)
    print("Defect_Type and Defect_Subtype are hidden ground truth (evaluation only).")
    print("Do NOT use *_168h, *_Delta_96_168, *_Delta_24_168 or *_168h_Absolute_Fail")
    print("columns as inputs when predicting 168h behaviour.")
    print(f"Outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    report = OUT_DIR / "sih_synthetic_generator_report.txt"
    so, se = sys.stdout, sys.stderr
    fh = report.open("w", encoding="utf-8")
    try:
        sys.stdout = Tee(sys.stdout, fh)
        sys.stderr = Tee(sys.stderr, fh)
        main()
    finally:
        sys.stdout, sys.stderr = so, se
        fh.close()