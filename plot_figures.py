#!/usr/bin/env python3
"""
P1 Figure Generator
===================
Generates all paper figures from experiment JSON output files.

Input:   results/p1/E1_layer_breakdown.json
         results/p1/E2_spg_correlation.json
         results/p1/E3_bootstrap_ci.json
         results/p1/E4_bert.json
         results/p1/E5_null_task.json

Output:  figs/depth_profile.pdf
         figs/e2_scatter.pdf
         figs/e3_bootstrap.pdf
         figs/e4_bert_comparison.pdf
         figs/e5_null_task.pdf

Usage:
    python p1_plot_figures.py
    python p1_plot_figures.py --results_dir results/p1 --figs_dir figs
    python p1_plot_figures.py --only e1 e2   # specific figures only

Requirements:
    pip install matplotlib scipy numpy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# -- Style ----------------------------------------------------------------------
COLORS = {
    "gpt2_medium": "#1f77b4",
    "gpt2_small":  "#ff7f0e",
    "gpt2_large":  "#9467bd",
    "gpt2_xl":     "#8c564b",
    "bert":        "#2ca02c",
    "mistral":     "#d62728",
}
MARKERS = {
    "gpt2_medium": "o",
    "gpt2_small":  "s",
    "gpt2_large":  "P",
    "gpt2_xl":     "D",
    "bert":        "^",
    "mistral":     "X",
}


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def extract_layers(model_data):
    """Extract (layer_indices, ratios, rand_means, rand_stds) from per_layer dict."""
    layers = model_data["per_layer"]
    idxs, ratios, rmeans, rstds = [], [], [], []
    for k in sorted(layers, key=int):
        v = layers[k]
        idxs.append(int(k))
        ratios.append(v["ratio_WqWk"])
        rmeans.append(v["r_WqWk_rand"])
        rstds.append(v["r_WqWk_rand_std"])
    return np.array(idxs), np.array(ratios), np.array(rmeans), np.array(rstds)


# -- Figure 1: Depth profile (E1) -----------------------------------------------

def fig_depth_profile(results_dir: Path, figs_dir: Path):
    E1 = load(results_dir / "E1_layer_breakdown.json")
    E4 = load(results_dir / "E4_bert.json")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Baseline band -- use first available GPT-2 model for band estimate
    _baseline_model = None
    for _key in ["gpt2_medium", "gpt2_small", "gpt2_large", "gpt2_xl"]:
        if _key in E1:
            _baseline_model = E1[_key]
            break
    if _baseline_model:
        _, _, gm_rm, gm_rs = extract_layers(_baseline_model)
        rand_mean = np.mean(gm_rm)
        rand_std  = np.mean(gm_rs)
        band_lo = 1.0 - rand_std / rand_mean
        band_hi = 1.0 + rand_std / rand_mean
        ax.axhspan(band_lo, band_hi, color="#888888", alpha=0.12,
                   label=r"$\pm1\sigma$ random baseline")
    ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--",
               alpha=0.7, label="Random baseline ($\\rho=1$)")

    # Models to plot (use what's available in E1 JSON)
    # All 4 GPT-2 variants + BERT + Mistral; each silently skipped if absent
    available = {
        "gpt2_small":  ("GPT-2 small",       E1.get("gpt2_small")),
        "gpt2_medium": ("GPT-2 medium",       E1.get("gpt2_medium")),
        "gpt2_large":  ("GPT-2 large",        E1.get("gpt2_large")),
        "gpt2_xl":     ("GPT-2 XL",           E1.get("gpt2_xl")),
        "bert":        ("BERT-base",           E4),
        "mistral":     ("Mistral-7B / RoPE",  E1.get("mistral_7b")),
    }

    for key, (label, data) in available.items():
        if data is None:
            continue
        idx, rho, rm, rs = extract_layers(data)
        ax.plot(idx, rho, color=COLORS[key], marker=MARKERS[key],
                markersize=5, linewidth=1.4, label=label)

    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel(
        r"Structure ratio $\rho^{(l)} = r_{\mathrm{sk}}(\mathbf{W}_{\mathrm{qk}}^{(l)}) / r_{\mathrm{rand}}^{(l)}$",
        fontsize=10)
    ax.set_title("Layer-wise structure ratio across architectures (E1)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.set_ylim(0.58, 1.12)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = figs_dir / "depth_profile.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(figs_dir / "depth_profile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# -- Figure 2: E2 scatter --------------------------------------------------------

def fig_e2_scatter(results_dir: Path, figs_dir: Path):
    E2 = load(results_dir / "E2_spg_correlation.json")

    layers = E2["per_layer"]
    r_wqwk = np.array([layers[k]["r_WqWk"]            for k in sorted(layers, key=int)])
    g_sym  = np.array([layers[k]["mean_grad_sym_ratio"] for k in sorted(layers, key=int)])
    l_idx  = np.array([int(k)                           for k in sorted(layers, key=int)])

    sp_r = E2["corr_r_WqWk_vs_grad_sym"]["spearman_r"]
    sp_p = E2["corr_r_WqWk_vs_grad_sym"]["spearman_p"]
    pe_r = E2["corr_r_WqWk_vs_grad_sym"]["pearson_r"]
    pe_p = E2["corr_r_WqWk_vs_grad_sym"]["pearson_p"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(r_wqwk, g_sym, c=l_idx, cmap="viridis",
                    norm=plt.Normalize(l_idx.min(), l_idx.max()),
                    s=55, zorder=3, edgecolors="white", linewidths=0.4)

    slope, intercept, *_ = stats.linregress(r_wqwk, g_sym)
    x_line = np.linspace(r_wqwk.min() - 0.003, r_wqwk.max() + 0.003, 100)
    ax.plot(x_line, slope * x_line + intercept,
            color="#d62728", linewidth=1.2, linestyle="--", alpha=0.7)

    for x, y, l in zip(r_wqwk, g_sym, l_idx):
        if l in [0, 1, 2]:
            ax.annotate(f"L{l}", (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color="#555555")

    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Layer index", fontsize=9)

    textstr = (f"Spearman $\\rho_s = {sp_r:.3f}$, $p = {sp_p:.3f}$\n"
               f"Pearson $r = {pe_r:.3f}$, $p = {pe_p:.3f}$\n"
               f"$n = {len(r_wqwk)}$ layers")
    ax.text(0.04, 0.96, textstr, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor="#bbbbbb", alpha=0.9))

    ax.set_xlabel(r"$r_{\mathrm{sk}}(\mathbf{W}_{\mathrm{qk}}^{(l)})$",
                  fontsize=10)
    ax.set_ylabel(r"Gradient symmetry ratio $g^{(l)}$", fontsize=10)
    ax.set_title("E2: Weight structure vs gradient geometry\n"
                 "(GPT-2 medium, 24 layers)", fontsize=10, fontweight="bold")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = figs_dir / "e2_scatter.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(figs_dir / "e2_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# -- Figure 3: E3 bootstrap bar chart -------------------------------------------

def fig_e3_bootstrap(results_dir: Path, figs_dir: Path):
    E3 = load(results_dir / "E3_bootstrap_ci.json")

    model_keys = ["gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl", "mistral_7b"]
    labels     = ["GPT-2\nsmall", "GPT-2\nmedium", "GPT-2\nlarge",
                  "GPT-2\nXL", "Mistral-7B\n(RoPE)"]
    colors     = [COLORS["gpt2_small"], COLORS["gpt2_medium"],
                  COLORS["gpt2_large"], COLORS["gpt2_xl"], COLORS["mistral"]]

    sig   = []
    total = []
    for k in model_keys:
        if k in E3:
            sig.append(E3[k]["n_layers_sig_WqWk"])
            total.append(E3[k]["n_layers_total"])
        else:
            sig.append(0)
            total.append(0)
    frac = [s / t if t > 0 else 0 for s, t in zip(sig, total)]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(labels, frac, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.6)
    for bar, s, t in zip(bars, sig, total):
        if t > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{s}/{t}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.axhline(1.0, color="#444", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Fraction of layers with\n95% CI entirely below $\\rho=1.0$",
                  fontsize=10)
    ax.set_title("E3: Bootstrap significance -- layers showing algebraic structure",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ape_patch  = mpatches.Patch(color=COLORS["gpt2_medium"], alpha=0.85, label="APE models")
    rope_patch = mpatches.Patch(color=COLORS["mistral"], alpha=0.85, label="RoPE model")
    ax.legend(handles=[ape_patch, rope_patch], fontsize=9, loc="upper right")

    plt.tight_layout()
    out = figs_dir / "e3_bootstrap.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(figs_dir / "e3_bootstrap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# -- Figure 4: E4 BERT comparison -----------------------------------------------

def fig_e4_bert(results_dir: Path, figs_dir: Path):
    E1 = load(results_dir / "E1_layer_breakdown.json")
    E4 = load(results_dir / "E4_bert.json")

    bert_idx, bert_rho, bert_rm, bert_rs = extract_layers(E4)

    # Use gpt2_medium for comparison; fall back to any available GPT-2 variant
    _gpt2_key = next((k for k in ["gpt2_medium", "gpt2_small", "gpt2_large", "gpt2_xl"]
                      if k in E1), None)
    if _gpt2_key is None:
        print("  [E4] No GPT-2 data in E1 -- skipping GPT-2 comparison line")
        return
    _gpt2_label = {"gpt2_medium": "GPT-2 medium", "gpt2_small": "GPT-2 small",
                   "gpt2_large": "GPT-2 large", "gpt2_xl": "GPT-2 XL"}[_gpt2_key]
    gpt2m_idx, gpt2m_rho, gpt2m_rm, gpt2m_rs = extract_layers(E1[_gpt2_key])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for idx, rho, rm, rs, color, lbl, mk in [
        (bert_idx,  bert_rho,  bert_rm,  bert_rs,  COLORS["bert"],      "BERT-base",   "o"),
        (gpt2m_idx, gpt2m_rho, gpt2m_rm, gpt2m_rs, COLORS[_gpt2_key],  _gpt2_label,   "s"),
    ]:
        band = rs / rm
        ax.fill_between(idx, 1 - band * 2, 1 + band * 2, alpha=0.07, color=color)
        ax.plot(idx, rho, color=color, marker=mk, markersize=5.5,
                linewidth=1.5, label=lbl)

    ax.axhline(1.0, color="#444", linewidth=1.0, linestyle="--", alpha=0.7,
               label="Random baseline")

    # Annotate BERT layer 2 inversion
    bert_l2 = bert_rho[2] if len(bert_rho) > 2 else 1.046
    ax.annotate(f"L2 inversion\n($\\rho={bert_l2:.3f}$)",
                xy=(2, bert_l2), xytext=(4.2, bert_l2 + 0.025),
                fontsize=8, color=COLORS["bert"],
                arrowprops=dict(arrowstyle="->", color=COLORS["bert"], lw=1.0))

    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel(r"Structure ratio $\rho^{(l)}$", fontsize=11)
    ax.set_title(f"E4: BERT-base vs {_gpt2_label} -- both APE, bidirectional vs causal",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.set_ylim(0.58, 1.13)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = figs_dir / "e4_bert_comparison.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(figs_dir / "e4_bert_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# -- Figure 5: E5 null fine-tuning ----------------------------------------------

def fig_e5_null(results_dir: Path, figs_dir: Path):
    E5 = load(results_dir / "E5_null_task.json")

    loss  = E5["loss_history"]
    steps = list(range(len(loss)))
    pre   = E5["comparison"]["pretrained_mean_ratio"]
    post  = E5["comparison"]["null_task_mean_ratio"]
    delta = E5["comparison"]["delta_ratio"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    ax1.plot(steps, loss, color="#555555", linewidth=0.8, alpha=0.7)
    ax1.axhline(np.log(2), color="#d62728", linewidth=1.2, linestyle="--",
                label=f"$\\ln(2)\\approx{np.log(2):.3f}$")
    ax1.set_xlabel("Training step", fontsize=10)
    ax1.set_ylabel("Cross-entropy loss", fontsize=10)
    ax1.set_title("E5: Null fine-tuning loss\n(random labels, GPT-2 small)",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linewidth=0.3, alpha=0.5)

    cats  = ["Pretrained", "After 200\nnull steps"]
    vals  = [pre, post]
    bars2 = ax2.bar(cats, vals, color=[COLORS["gpt2_medium"], "#ff7f0e"],
                    alpha=0.85, edgecolor="white", width=0.4)
    ax2.axhline(1.0, color="#444", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, v in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0002,
                 f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0.90, 0.935)
    ax2.set_ylabel(r"Mean structure ratio $\bar{\rho}$", fontsize=10)
    ax2.set_title(f"$\\Delta\\rho = {delta:.2e}$\n(structure unchanged)",
                  fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    out = figs_dir / "e5_null_task.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(figs_dir / "e5_null_task.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# -- Main -----------------------------------------------------------------------

FIGURE_MAP = {
    "e1": fig_depth_profile,
    "e2": fig_e2_scatter,
    "e3": fig_e3_bootstrap,
    "e4": fig_e4_bert,
    "e5": fig_e5_null,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate P1 paper figures from experiment JSON results.")
    parser.add_argument("--results_dir", default="results/p1",
                        help="Directory containing E1-E5 JSON files.")
    parser.add_argument("--figs_dir", default="figs",
                        help="Output directory for PDF/PNG figures.")
    parser.add_argument("--only", nargs="+", choices=list(FIGURE_MAP.keys()),
                        help="Generate only specified figures (e.g. --only e1 e2).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figs_dir    = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    to_run = args.only if args.only else list(FIGURE_MAP.keys())
    print(f"Generating figures: {to_run}")
    print(f"  Results dir : {results_dir}")
    print(f"  Figures dir : {figs_dir}")
    print()

    for key in to_run:
        json_name = f"E{key[1]}_" if key.startswith("e") else key
        try:
            FIGURE_MAP[key](results_dir, figs_dir)
        except FileNotFoundError as e:
            print(f"  SKIP {key}: {e}")
        except Exception as e:
            print(f"  ERROR {key}: {e}")
            raise

    print("\nDone.")


if __name__ == "__main__":
    main()
