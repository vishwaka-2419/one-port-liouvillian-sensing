#!/usr/bin/env python3
"""Export the Fig. 2a pump-probe trace directly from the archived raw dataset.

This creates the repository fallback file data_processed/fig2a_trace.csv from
SM/S16/dI(Delay).csv without digitizing a figure.
"""
from pathlib import Path
import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the extracted external dataset directory containing SM/ and F4/.",
    )
    parser.add_argument(
        "--out",
        default="data_processed/fig2a_trace.csv",
        help="Output CSV path (default: data_processed/fig2a_trace.csv).",
    )
    args = parser.parse_args()

    src = Path(args.data_root) / "SM" / "S16" / "dI(Delay).csv"
    if not src.is_file():
        raise FileNotFoundError(f"Expected raw-data file not found: {src}")

    raw = pd.read_csv(src)
    required = ["Delay (ns)", "dI (sig) (fA)", "dI_err (sig) (fA)"]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise KeyError(f"Missing expected columns in {src}: {missing}")

    out = pd.DataFrame(
        {
            "tau_ns": raw["Delay (ns)"].astype(float),
            "dI_fA": raw["dI (sig) (fA)"].astype(float),
            "dI_err_fA": raw["dI_err (sig) (fA)"].astype(float),
        }
    )

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"Wrote {len(out)} rows to {dst}")
    print(f"Source: {src}")


if __name__ == "__main__":
    main()
