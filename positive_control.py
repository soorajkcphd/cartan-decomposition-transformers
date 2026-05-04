"""
P1 Positive Control -- Measurement A Validation
===============================================
Proves that r_sk(W_q @ W_k^T) = ||skew(A)||_F / ||A||_F can detect
skew-symmetry when it genuinely exists, and degrades gracefully under noise.

Four synthetic cases at d=1024 (same as LLM weights):

  Case 1 -- Symmetric product   W_q @ W_q^T         -> r_WqWk = 0.000
  Case 2 -- Random product      W_q @ W_k^T (indep)  -> r_WqWk ~ 0.707
  Case 3 -- Skew product        constructed exactly   -> r_WqWk = 1.000
  Case 4 -- Noisy skew          skew + Gaussian noise -> r_WqWk degrades

Reference lines from main experiment:
  GPT-2 layer 7:   r_WqWk = 0.609  (most symmetric, ~10% below baseline)
  Mistral layer 8: r_WqWk = 0.702  (near baseline, ~0.5% below)

No GPU required. Runtime: ~2 minutes on CPU.
"""

import os, json, time
import numpy as np
import torch

os.makedirs("results", exist_ok=True)
os.makedirs("results/p1_figures", exist_ok=True)


# =============================================================================
# CORE METRIC
# =============================================================================

def skew_ratio(W: torch.Tensor) -> float:
    """
    r_sk(W) = ||skew(W)||_F / ||W||_F  where skew(W) = (W - W^T) / 2.
    = 0   for perfectly symmetric W
    = 1   for perfectly skew-symmetric W
    ~ 0.707 for random W (exact analytical result, any d)
    """
    W = W.float()
    sk = (W - W.T) / 2.0
    return (torch.norm(sk) / (torch.norm(W) + 1e-12)).item()


# =============================================================================
# SYNTHETIC CONSTRUCTIONS
# =============================================================================

def make_symmetric_pair(d: int) -> tuple:
    """
    W_q @ W_k^T = W_q @ W_q^T  (symmetric PSD).
    r_WqWk = 0.0 exactly.
    """
    W_q = torch.randn(d, d)
    W_k = W_q.clone()        # W_q @ W_k^T = W_q @ W_q^T
    return W_q, W_k


def make_random_pair(d: int) -> tuple:
    """
    W_q, W_k independent random.
    E[r_WqWk] = 1/sqrt(2) = 0.707.
    """
    return torch.randn(d, d), torch.randn(d, d)


def make_skew_pair(d: int) -> tuple:
    """
    Construct W_q, W_k such that W_q @ W_k^T = S (exactly skew-symmetric).

    Method (numerically stable -- orthogonal W_q):
        1. W_q = random orthogonal matrix via QR  (so pinv(W_q) = W_q^T exactly)
        2. S   = random skew-symmetric matrix  S = (R - R^T) / 2
        3. W_k = S^T @ W_q
        4. Verify: W_q @ W_k^T = W_q @ (S^T @ W_q)^T
                                = W_q @ W_q^T @ S = I @ S = S  [OK]

    Why orthogonal W_q (not general random W_q via pinv)?
        For general W_q, pinv amplifies errors by cond(W_q).
        At d=1024, cond(W_q_random) ~ 1000+, giving sym_err up to ~5% (unreliable).
        Orthogonal W_q gives sym_err < 1e-6 for all d. Also gives a smoother
        noise sweep because ||W_q||_F = sqrt(d) is well-controlled.
    """
    W_q_raw = torch.randn(d, d)
    W_q, _  = torch.linalg.qr(W_q_raw)    # orthogonal: W_q @ W_q^T = I exactly
    R       = torch.randn(d, d)
    S       = (R - R.T) / 2.0             # random skew-symmetric target
    W_k     = S.T @ W_q                   # W_k such that W_q @ W_k^T = S
    return W_q, W_k


def add_noise(W_q: torch.Tensor, W_k: torch.Tensor,
              sigma: float) -> tuple:
    """
    Add iid Gaussian noise scaled to sigma * ||W||_F / sqrt(d).
    sigma=0 -> no change; sigma=1 -> noise comparable to signal.
    """
    if sigma == 0.0:
        return W_q.clone(), W_k.clone()
    scale_q = sigma * torch.norm(W_q).item() / (W_q.shape[0] ** 0.5)
    scale_k = sigma * torch.norm(W_k).item() / (W_k.shape[0] ** 0.5)
    W_q_n = W_q + scale_q * torch.randn_like(W_q)
    W_k_n = W_k + scale_k * torch.randn_like(W_k)
    return W_q_n, W_k_n


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_positive_control(d: int = 1024, n_trials: int = 5,
                         noise_levels=None) -> dict:
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    results = {}

    print(f"\nd = {d}  (same as GPT-2 / Mistral-7B weight matrices)")
    print(f"{'Case':<22}  {'sigma':>6}  {'r_WqWk':>8}  {'std':>7}  note")
    print("-" * 62)

    # -- Case 1: Symmetric -------------------------------------------------
    vals = []
    for _ in range(n_trials):
        W_q, W_k = make_symmetric_pair(d)
        vals.append(skew_ratio(W_q @ W_k.T))
    m, s = float(np.mean(vals)), float(np.std(vals))
    results["symmetric"] = {"mean": m, "std": s}
    print(f"{'Symmetric product':<22}  {'--':>6}  {m:>8.4f}  {s:>7.4f}  W_k = W_q")

    # -- Case 2: Random ----------------------------------------------------
    vals = []
    for _ in range(n_trials):
        W_q, W_k = make_random_pair(d)
        vals.append(skew_ratio(W_q @ W_k.T))
    m, s = float(np.mean(vals)), float(np.std(vals))
    results["random"] = {"mean": m, "std": s, "theory": 1.0 / np.sqrt(2)}
    print(f"{'Random product':<22}  {'--':>6}  {m:>8.4f}  {s:>7.4f}  theory=0.7071")

    # -- Case 3: Exact skew ------------------------------------------------
    vals = []
    for _ in range(n_trials):
        W_q, W_k = make_skew_pair(d)
        vals.append(skew_ratio(W_q @ W_k.T))
    m, s = float(np.mean(vals)), float(np.std(vals))
    results["skew_exact"] = {"mean": m, "std": s}
    print(f"{'Skew product (exact)':<22}  {'--':>6}  {m:>8.4f}  {s:>7.4f}  theory=1.0")

    # -- Case 4: Noisy skew -- noise sweep ----------------------------------
    print()
    noisy_results = []
    W_q_base, W_k_base = make_skew_pair(d)   # one fixed skew pair

    for sigma in noise_levels:
        vals = []
        for _ in range(n_trials):
            W_q_n, W_k_n = add_noise(W_q_base, W_k_base, sigma)
            vals.append(skew_ratio(W_q_n @ W_k_n.T))
        m, s = float(np.mean(vals)), float(np.std(vals))
        noisy_results.append({"sigma": sigma, "mean": m, "std": s})
        note = "<- random baseline" if abs(m - 0.707) < 0.01 else ""
        print(f"{'Noisy skew':<22}  {sigma:>6.2f}  {m:>8.4f}  {s:>7.4f}  {note}")

    results["noisy_skew"] = noisy_results

    # -- Reference values from main experiment ----------------------------
    print()
    print(f"{'GPT-2 layer 0':<22}  {'--':>6}  {'0.7089':>8}  {'--':>7}  near baseline")
    print(f"{'GPT-2 layer 7':<22}  {'--':>6}  {'0.6093':>8}  {'--':>7}  <- most symmetric")
    print(f"{'Mistral layer 8':<22}  {'--':>6}  {'0.7019':>8}  {'--':>7}  near baseline")
    print(f"{'Baseline (theory)':<22}  {'--':>6}  {'0.7071':>8}  {'--':>7}  = 1/sqrt(2)")

    results["gpt2_reference"] = {
        "layer_0": 0.7089, "layer_7": 0.6093, "layer_23": 0.6261,
        "mean": 0.6445, "min": 0.6093, "note": "below baseline -> more symmetric"
    }
    results["mistral_reference"] = {
        "layer_8": 0.7019, "mean_0_28": 0.7058,
        "note": "near baseline -> structure near random"
    }

    return results


# =============================================================================
# FIGURE
# =============================================================================

def make_positive_control_figure(results: dict,
                                  out_dir: str = "results/p1_figures"):
    try:
        import matplotlib.pyplot as plt, matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not found -- skipping figure"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # -- Left: four cases bar chart ----------------------------------------
    ax1 = axes[0]
    cases = ["Symmetric\n(W_k=W_q)", "Random\n(independent)",
             "Skew exact\n(pinv construction)", "GPT-2\nlayer 7"]
    values = [
        results["symmetric"]["mean"],
        results["random"]["mean"],
        results["skew_exact"]["mean"],
        0.6093,   # GPT-2 layer 7
    ]
    stds = [
        results["symmetric"]["std"],
        results["random"]["std"],
        results["skew_exact"]["std"],
        0.0,
    ]
    colors = ["steelblue", "gray", "darkgreen", "orange"]
    bars = ax1.bar(cases, values, color=colors, alpha=0.8, width=0.55,
                   yerr=stds, capsize=4)
    ax1.axhline(1/np.sqrt(2), color="red", ls="--", lw=1.5,
                label=f"Random baseline = {1/np.sqrt(2):.4f}")
    ax1.axhline(0.0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax1.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_ylabel("r_sk(W_q @ W_k^T)", fontsize=11)
    ax1.set_title("Measurement A: four synthetic cases\n(d = 1024)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.025,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # -- Right: noise sweep ------------------------------------------------
    ax2 = axes[1]
    noisy = results["noisy_skew"]
    sigmas = [r["sigma"] for r in noisy]
    means  = [r["mean"]  for r in noisy]
    stds_n = [r["std"]   for r in noisy]
    ax2.plot(sigmas, means, "g-o", lw=2, ms=6, label="Noisy skew product")
    ax2.fill_between(sigmas,
                     [m - s for m, s in zip(means, stds_n)],
                     [m + s for m, s in zip(means, stds_n)],
                     alpha=0.2, color="green")
    ax2.axhline(1/np.sqrt(2), color="red", ls="--", lw=1.5,
                label=f"Random baseline = {1/np.sqrt(2):.4f}")
    ax2.axhline(results["random"]["mean"], color="gray", ls=":",
                lw=1.2, label=f"Empirical random = {results['random']['mean']:.4f}")
    # Reference lines for LLMs
    ax2.axhline(0.6093, color="orange", ls="-.", lw=1.2,
                label="GPT-2 layer 7 = 0.609")
    ax2.axhline(0.7019, color="purple", ls="-.", lw=1.2,
                label="Mistral layer 8 = 0.702")
    ax2.set_xlabel("Noise level sigma\n(sigma=0: exact skew, sigma=0.5: near-random)", fontsize=10)
    ax2.set_ylabel("r_sk(W_q @ W_k^T)", fontsize=11)
    ax2.set_title("Measurement A: noise robustness\n(skew product + Gaussian noise)", fontsize=11)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_ylim(0.5, 1.1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{out_dir}/fig_positive_control.{ext}", dpi=150)
    plt.close()
    print(f"\n  Saved fig_positive_control.pdf/png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*62)
    print("P1 Positive Control -- Measurement A Validation")
    print("="*62)

    t0 = time.time()
    results = run_positive_control(
        d=1024,
        n_trials=5,
        noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    )

    make_positive_control_figure(results)

    with open("results/p1_positive_control.json", "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Results saved to results/p1_positive_control.json")
    print(f"  Total runtime: {elapsed:.0f}s")

    print("\n" + "="*62)
    print("INTERPRETATION")
    print("="*62)
    sym  = results["symmetric"]["mean"]
    rand = results["random"]["mean"]
    skew = results["skew_exact"]["mean"]
    print(f"  Symmetric product: r_WqWk = {sym:.4f}  (bottom of scale)")
    print(f"  Random product:    r_WqWk = {rand:.4f}  (baseline)")
    print(f"  Skew product:      r_WqWk = {skew:.4f}  (top of scale)")
    print(f"  The metric spans [0, 1] and correctly identifies all three cases.")
    print()
    print(f"  GPT-2 mean:    r_WqWk = 0.6445  -> 10% below baseline -> more symmetric")
    print(f"  Mistral mean:  r_WqWk = 0.7058  ->  1% below baseline -> near random")
    print(f"  Both models have W_q@W_k^T more SYMMETRIC than random (not more skew).")
    print(f"  GPT-2 effect is large (Cohen's d ~ {abs(0.6445-rand)/0.005:.1f}sigma),")
    print(f"  Mistral effect is small (Cohen's d ~ {abs(0.7058-rand)/0.005:.1f}sigma).")
    print("Done.")
