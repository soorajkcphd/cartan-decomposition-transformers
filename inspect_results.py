#!/usr/bin/env python3
"""
inspect_results.py  --  P1 results inspector
Reads E1 and E3 JSONs and prints per-layer breakdowns.

Usage:
    python inspect_results.py                    # full summary all models
    python inspect_results.py --model opt_350m   # per-layer for one model
    python inspect_results.py --model all        # per-layer for every model
"""

import json
import argparse
import math
from pathlib import Path

RESULTS_DIR = Path("results/p1")

MODEL_ORDER = [
    "gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl",
    "bert_base",
    "openllama_3b", "mistral_7b",
    "opt_125m", "opt_350m",
]


def load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def bar(ratio, width=20):
    """ASCII bar centred on 1.0."""
    centre = width // 2
    pos = int(centre * (1.0 - ratio) + centre)
    pos = max(0, min(width - 1, pos))
    b = ["-"] * width
    b[centre] = "|"   # baseline
    b[pos]    = "#"
    return "".join(b)


def fmt(v):
    return f"{v:.4f}" if not math.isnan(v) else "  nan "


def print_summary(e1, e3):
    print("\n" + "=" * 72)
    print(f"  {'Model':<18}  {'mean_r':>7}  {'mean_rho':>7}  "
          f"{'E3 sig':>8}  {'E1 type':<10}  bar (1.0 = random)")
    print("=" * 72)
    for mname in MODEL_ORDER:
        m = e1.get(mname)
        if not m:
            continue
        s     = m.get("summary", {})
        mr    = s.get("mean_r_WqWk",    float("nan"))
        ratio = s.get("mean_ratio_WqWk", float("nan"))
        rtype = s.get("result_type",    "?")
        marker = "[OK]" if rtype == "POSITIVE" else "[NO]"
        e3m   = e3.get(mname, {})
        n_sig = e3m.get("n_layers_sig_WqWk")
        n_tot = e3m.get("n_layers_total")
        e3_str = f"{n_sig:2d}/{n_tot}" if n_sig is not None else "  --"
        b = bar(ratio) if not math.isnan(ratio) else " " * 20
        print(f"  {marker} {mname:<16}  {fmt(mr)}  {fmt(ratio)}  "
              f"{e3_str:>8}  {rtype:<10}  {b}")
    print("=" * 72)
    print("  Bar: # = mean structure ratio  | = random baseline (rho=1.0)")
    print("  rho < 1 -> symmetric excess (APE structure)  "
          "rho > 1 -> skew excess (above baseline)")


def print_per_layer(mname, e1, e3):
    e1m = e1.get(mname, {})
    e3m = e3.get(mname, {})
    if not e1m:
        print(f"\n  [!] '{mname}' not found in E1 results.")
        return

    s     = e1m.get("summary", {})
    ratio = s.get("mean_ratio_WqWk", float("nan"))
    rtype = s.get("result_type", "?")
    n_sig = e3m.get("n_layers_sig_WqWk", "?")
    n_tot = e3m.get("n_layers_total",    "?")

    print(f"\n{'-'*68}")
    print(f"  {mname}  |  mean_ratio={fmt(ratio)}  [{rtype}]  "
          f"E3: {n_sig}/{n_tot} significant")
    print(f"{'-'*68}")
    print(f"  {'Layer':>5}  {'r_WqWk':>7}  {'rand':>7}  {'ratio':>7}  "
          f"{'CI_lo':>7}  {'CI_hi':>7}  sig  bar")
    print(f"  {'':->5}  {'':->7}  {'':->7}  {'':->7}  "
          f"{'':->7}  {'':->7}  ---  {'-'*20}")

    e3_layers = e3m.get("per_layer", {})
    e1_layers = e1m.get("per_layer", {})

    # union of layer keys from both
    all_keys = sorted(
        set(e3_layers.keys()) | set(e1_layers.keys()),
        key=lambda x: int(x)
    )

    for li in all_keys:
        e3l  = e3_layers.get(li, {})
        e1l  = e1_layers.get(li, {})

        r      = e3l.get("r_WqWk",              e1l.get("r_WqWk",    float("nan")))
        rnd    = e1l.get("r_WqWk_rand",                               float("nan"))
        ratio_l= e3l.get("ratio_WqWk",          e1l.get("ratio_WqWk", float("nan")))
        lo     = e3l.get("ratio_WqWk_ci95_lo",                        float("nan"))
        hi     = e3l.get("ratio_WqWk_ci95_hi",                        float("nan"))
        sig    = e3l.get("significant_WqWk",    False)

        sig_str = "v [OK]" if sig else "^  "
        b = bar(ratio_l) if not math.isnan(ratio_l) else " " * 20
        print(f"  L{int(li):4d}  {fmt(r)}  {fmt(rnd)}  {fmt(ratio_l)}  "
              f"{fmt(lo)}  {fmt(hi)}  {sig_str}  {b}")

    print(f"{'-'*68}")
    print(f"  v [OK] = 95% CI entirely below 1.0 (significant APE structure)")
    print(f"  ^   = CI includes or exceeds 1.0 (not significant / above baseline)")


def main():
    ap = argparse.ArgumentParser(description="Inspect P1 E1/E3 results")
    ap.add_argument("--model", default=None,
                    help="Model to inspect per-layer (e.g. opt_350m, all)")
    ap.add_argument("--results_dir", default="results/p1",
                    help="Path to results directory")
    args = ap.parse_args()

    global RESULTS_DIR
    RESULTS_DIR = Path(args.results_dir)

    e1 = load("E1_layer_breakdown.json")
    e3 = load("E3_bootstrap_ci.json")

    if not e1:
        print(f"[!] E1 results not found in {RESULTS_DIR}. Run E1 first.")
        return
    if not e3:
        print(f"[!] E3 results not found in {RESULTS_DIR}. Run E3 first.")

    # Always print summary
    print_summary(e1, e3)

    # Per-layer if requested
    if args.model:
        if args.model.lower() == "all":
            for mname in MODEL_ORDER:
                if mname in e1:
                    print_per_layer(mname, e1, e3)
        else:
            print_per_layer(args.model, e1, e3)


if __name__ == "__main__":
    main()
