#!/usr/bin/env python3
"""
Compare Polaris HLM projections for a simple MLP:
TTSIM workload vs ONNX workload, same model, same shapes, same hardware.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

def pick_first_existing_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of the candidate columns for {label} exist in DataFrame. "
        f"Candidates: {candidates}\nColumns: {df.columns.tolist()}"
    )

def load_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_opstats(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ttsim_root",
        type=Path,
        default=Path("out_mlp_ttsim") / "mlp_ttsim",
        help="Root dir for TTSIM run (contains SUMMARY/ and STATS/)",
    )
    p.add_argument(
        "--onnx_root",
        type=Path,
        default=Path("out_mlp_onnx") / "mlp_onnx",
        help="Root dir for ONNX run (contains SUMMARY/ and STATS/)",
    )
    args = p.parse_args()

    # ----- 1) Workload-level summary -----
    ttsim_sum_path = args.ttsim_root / "SUMMARY" / "study-summary.csv"
    onnx_sum_path  = args.onnx_root  / "SUMMARY" / "study-summary.csv"

    ttsim_sum = load_summary(ttsim_sum_path)
    onnx_sum  = load_summary(onnx_sum_path)

    print("TTSIM summary columns:", ttsim_sum.columns.tolist())
    print("ONNX  summary columns:", onnx_sum.columns.tolist())

    keys = ["wlname", "wlinstance", "bs", "archname", "devname", "freq_Mhz"]

    cycle_col = pick_first_existing_column(
        ttsim_sum, ["tot_cycles", "total_cycles", "cycles"], "workload cycles"
    )
    time_col = pick_first_existing_column(
        ttsim_sum, ["tot_msecs", "total_msecs", "msecs", "time_ms"], "workload time"
    )

    merged_sum = onnx_sum.merge(ttsim_sum, on=keys, suffixes=("_onnx", "_ttsim"))

    merged_sum["cycle_ratio"] = (
        merged_sum[f"{cycle_col}_ttsim"] / merged_sum[f"{cycle_col}_onnx"]
    )
    merged_sum["time_ratio"] = (
        merged_sum[f"{time_col}_ttsim"] / merged_sum[f"{time_col}_onnx"]
    )

    print("\n=== Workload-level correlation (TTSIM / ONNX) ===")
    cols_to_show = (
        keys
        + [
            f"{cycle_col}_ttsim",
            f"{cycle_col}_onnx",
            "cycle_ratio",
            f"{time_col}_ttsim",
            f"{time_col}_onnx",
            "time_ratio",
        ]
    )
    print(merged_sum[cols_to_show].to_string(index=False))

    # ----- 2) Per-optype aggregation -----
    ttsim_opstats_files = list((args.ttsim_root / "STATS").glob("*-opstats.csv"))
    onnx_opstats_files  = list((args.onnx_root  / "STATS").glob("*-opstats.csv"))

    if not ttsim_opstats_files or not onnx_opstats_files:
        print("\nNo opstats CSV found; make sure both runs used --dump_stats_csv.")
        return

    ttsim_ops = load_opstats(ttsim_opstats_files[0])
    onnx_ops  = load_opstats(onnx_opstats_files[0])

    print("TTSIM opstats columns:", ttsim_ops.columns.tolist())
    print("ONNX  opstats columns:", onnx_ops.columns.tolist())

    group_cols = ["optype"]

    op_cycle_col = pick_first_existing_column(
        ttsim_ops, ["compute_cycles", "cycles"], "op-level cycles"
    )
    op_time_col = pick_first_existing_column(
        ttsim_ops, ["msecs", "time_ms", "exec_msecs"], "op-level time"
    )

    ttsim_agg = (
        ttsim_ops.groupby(group_cols)[[op_cycle_col, op_time_col]]
        .sum()
        .reset_index()
        .rename(
            columns={
                op_cycle_col: f"{op_cycle_col}_ttsim",
                op_time_col:  f"{op_time_col}_ttsim",
            }
        )
    )

    onnx_agg = (
        onnx_ops.groupby(group_cols)[[op_cycle_col, op_time_col]]
        .sum()
        .reset_index()
        .rename(
            columns={
                op_cycle_col: f"{op_cycle_col}_onnx",
                op_time_col:  f"{op_time_col}_onnx",
            }
        )
    )

    merged_ops = onnx_agg.merge(ttsim_agg, on=group_cols, how="inner")
    if merged_ops.empty:
        print("\nPer-optype aggregation produced no common optypes.")
        return

    merged_ops["cycle_ratio"] = (
        merged_ops[f"{op_cycle_col}_ttsim"] / merged_ops[f"{op_cycle_col}_onnx"]
    )
    merged_ops["time_ratio"] = (
        merged_ops[f"{op_time_col}_ttsim"] / merged_ops[f"{op_time_col}_onnx"]
    )

    print("\n=== Per-optype correlation (TTSIM / ONNX) ===")
    op_cols_to_show = (
        group_cols
        + [
            f"{op_cycle_col}_ttsim",
            f"{op_cycle_col}_onnx",
            "cycle_ratio",
            f"{op_time_col}_ttsim",
            f"{op_time_col}_onnx",
            "time_ratio",
        ]
    )
    print(
        merged_ops[op_cols_to_show]
        .sort_values("cycle_ratio")
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()