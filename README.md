# Cartan Decomposition Structure in Transformer Attention Weights

Reproducibility code for the paper:

> **Symmetric Excess in Frozen Transformer Attention Weights:
> Cartan Decomposition Structure, Positional Encoding, and a
> Benchmark Protocol for Weight-Space Geometry**
>
> Sooraj K.C and Vivek Mishra (2026).
> Submitted to *Journal of Computational Mathematics and Data Science* (Elsevier).

This repository provides the full benchmark protocol and reproduces every
numerical result in the paper.

---

## Quick Start

```bash
git clone https://github.com/soorajkcphd/cartan-decomposition-transformers.git
cd cartan-decomposition-transformers
pip install -r requirements.txt

# Run the validation control (no GPU, ~2 min)
python positive_control.py

# Run all main experiments (E0-E6) -- needs GPU for E1-E6
python run_experiments.py --run all

# Generate paper figures from saved results
python plot_figures.py

# Browse per-layer results
python inspect_results.py --model all
```

---

## Repository Contents

| File | Purpose |
|------|---------|
| `run_experiments.py` | **Master driver.** Runs all six experiments E0-E6: protocol validation on synthetic Lie algebras, layer-wise breakdown across nine pretrained transformers, gradient-symmetry correlation, bootstrap confidence intervals, BERT contrast, and null fine-tuning robustness. |
| `positive_control.py` | Stand-alone synthetic validation of the skewness ratio metric (Cases 1-4 from Sec.3 of the paper). Runs in ~2 minutes on CPU. |
| `plot_figures.py` | Generates the five paper figures from the JSON outputs of `run_experiments.py`. |
| `inspect_results.py` | Pretty-prints per-layer skewness ratios and bracket closure defects from the result JSONs. |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | MIT license. |
| `results/` | (Created at runtime.) JSON outputs of every experiment. |
| `figs/` | (Created at runtime.) Generated PDF figures. |

---

## Experiments

Each experiment writes a JSON file to `results/p1/`:

| ID | Experiment | Output | Runtime (RTX 5060 8 GB) |
|----|-----------|--------|------|
| E0 | Synthetic Lie algebras (so(3), sl(2), Heisenberg, Nilpotent n=4) | `E0_synthetic.json` | < 1 min CPU |
| E1 | Layer-wise breakdown across 9 pretrained models | `E1_layer_breakdown.json` | ~6 hr |
| E2 | Gradient-symmetry vs. weight-symmetry correlation (GPT-2 medium) | `E2_spg_correlation.json` | ~30 min |
| E3 | 95% bootstrap CIs (parametric, 5000 resamples) | `E3_bootstrap_ci.json` | ~10 min |
| E4 | BERT-base bidirectional contrast | `E4_bert.json` | ~20 min |
| E5 | Null fine-tuning stability (modest perturbation, 200 steps) | `E5_null_task.json` | ~30 min |
| E6 | Null fine-tuning stability (random-label memorisation, 1000 steps) | `E6_null_aggressive.json` | ~1 hr |

To run a subset:

```bash
python run_experiments.py --run E0 E1 E3
```

---

## Models Covered (E1)

The benchmark protocol is applied to:

| Model | Positional Encoding | Layers | Hidden d |
|-------|---------------------|--------|----------|
| GPT-2 small | APE (learned) | 12 | 768 |
| GPT-2 medium | APE (learned) | 24 | 1024 |
| GPT-2 large | APE (learned) | 36 | 1280 |
| GPT-2 XL | APE (learned) | 48 | 1600 |
| BERT-base-uncased | APE (learned) | 12 | 768 |
| OPT-125M | APE (learned) | 12 | 768 |
| OPT-350M | APE (learned) | 24 | 1024 |
| Mistral-7B-v0.1 | RoPE | 32 | 4096 |
| OpenLLaMA-3B | RoPE | 26 | 3200 |

Mistral-7B is loaded in 4-bit via `bitsandbytes` to fit on 8 GB GPUs. The
script auto-dequantises on the fly per-layer to recover float16 weights for
analysis.

---

## Mathematical Background

For any square matrix $W \in \mathbb{R}^{d \times d}$, the Cartan
decomposition

$$\mathfrak{gl}(d) = \mathfrak{so}(d) \oplus \mathrm{Sym}(d)$$

splits $W$ into a skew-symmetric and a symmetric part. We measure the
**skewness ratio**

$$r_{\text{sk}}(W) = \frac{\| (W - W^\top) / 2 \|_F}{\| W \|_F}$$

which equals $1/\sqrt{2} \approx 0.707$ for Gaussian random matrices (proved
in Sec.2 of the paper) and ranges from 0 (purely symmetric) to 1 (purely skew).

For attention weights, we apply this to the **query-key product**
$W_q W_k^\top$ at each layer of each model. The complementary metric is the
**bracket closure defect** $\delta$ on a chosen matrix family $\{A_i\}$:

$$\delta = \frac{\|[A_i, A_j] - \Pi_S [A_i, A_j]\|_F}{\|[A_i, A_j]\|_F}$$

where $\Pi_S$ is the orthogonal projection onto the span of the family.
$\delta = 0$ iff the family is bracket-closed.

Both metrics ship with proved Gaussian baselines and parametric bootstrap CIs.

---

## Key Reproducible Findings

| Model family | $\bar{\rho} = r_{\text{sk}}(W_qW_k^\top) / r_{\text{rand}}$ | Layers significantly below baseline |
|--------------|---|---|
| GPT-2 (4 sizes) | 0.893-0.946 | 87%-100% |
| BERT-base | 0.727 | strongest signal in study |
| OPT-125M / 350M | 0.879 / 1.013 | OPT-350M shows layer-specific inversions |
| OpenLLaMA-3B | 0.970 | weak structure |
| Mistral-7B | 0.998 | indistinguishable from baseline |

The bracket closure defect is **null** across all nine models -- structure
lives in the Cartan decomposition balance, not in Lie algebra membership.

---

## Hardware Requirements

- **Minimum:** 16 GB system RAM, modern CPU (E0 and positive_control only).
- **Recommended:** NVIDIA GPU with >=8 GB VRAM (RTX 3070, RTX 4060, RTX 5060, A4000+).
- **For Mistral-7B and OpenLLaMA-3B:** GPU required (4-bit quantisation used).

Total runtime end-to-end: ~10-14 hours on an RTX 5060 (8 GB).

---

## Citation

If you use this code or benchmark protocol, please cite:

```bibtex
@article{kc2026symmetric,
  author    = {Sooraj K.C and Vivek Mishra},
  title     = {Symmetric Excess in Frozen Transformer Attention Weights:
               Cartan Decomposition Structure, Positional Encoding, and a
               Benchmark Protocol for Weight-Space Geometry},
  journal   = {Journal of Computational Mathematics and Data Science},
  year      = {2026},
  publisher = {Elsevier},
  note      = {Under review}
}
```

---

## License

MIT License -- see [`LICENSE`](LICENSE).

---

## Contact

**Sooraj K.C** (Corresponding author)
PhD Candidate, Department of Pure and Applied Mathematics
Alliance University, Bengaluru 562 106, India
Email: `ksoorajPHD23@sam.alliance.edu.in`

For questions about the protocol, model-specific extraction logic, or
reproducing specific numerical results, please open a GitHub issue.
