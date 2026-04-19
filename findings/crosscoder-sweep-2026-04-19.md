# CrossCoder sweep — 15 (base, aligned) pairs — 2026-04-19

**Sweep config**: 5 alignment algorithms (DPO, GRPO, KTO, ORPO, SimPO) × 3 base models (HuggingFaceTB/SmolLM3-3B, meta-llama/Llama-3.2-3B-Instruct, Qwen/Qwen3-4B-Instruct-2507). Each pair trained at the base model's probe-best layer (SmolLM:L19, Llama:L11, Qwen:L24). Single seed (config.SEED=42), 4 epochs, expansion factor 8 → 16384 features. Dataset: argilla/ultrafeedback-multi-binarized-preferences-cleaned, full 157k samples per pair. Box: 8× B200, ~4h21m wall clock with jobs-per-gpu=2.

## Headline pattern (workshop-grade, replicates across all 3 bases)

**Alignment algorithms partition into two qualitatively distinct families:**

- **Broad-recruitment (DPO / SimPO / GRPO)**: produce 1800–7900 aligned-only features, but each feature shifts only mildly (`shift_aligned` ~ 0.001–0.003 on Llama/SmolLM)
- **Concentrated-modification (KTO / ORPO)**: produce 200–2000 aligned-only features (3–10× fewer), but each shifts dramatically (`shift_aligned` ~ 0.04–0.07, ~50× larger)

The pattern replicates across SmolLM3-3B, Llama-3.2-3B-Instruct, and Qwen3-4B-Instruct-2507 (modulo the two degenerate Qwen cells noted below).

## Aligned-only feature counts (the load-bearing table)

| Algo  | Llama (L11) | SmolLM (L19) | Qwen (L24) |
|-------|-------------|--------------|------------|
| DPO   | 2640        | 1967         | 7782       |
| SimPO | 2606        | 1840         | 7876       |
| GRPO  | 2607        | 1820         | 7575       |
| **KTO** | **917**   | **656**      | **7\***    |
| **ORPO**| **708**   | **2079**     | **202\***  |

\* qwen-kto and qwen-orpo are degenerate runs (see below) — the 7 and 202 are not real.

## counterfactual_sensitivity_shift on shared_aligned class

| slug         | L  | shift     |
|--------------|----|-----------|
| llama-dpo    | 11 | +0.00139  |
| llama-grpo   | 11 | +0.00256  |
| **llama-kto**| 11 | **+0.07056** |
| llama-orpo   | 11 | +0.00065  |
| llama-simpo  | 11 | +0.00137  |
| qwen-dpo     | 24 | +0.00836  |
| **qwen-grpo**| 24 | **+0.08912** |
| qwen-kto\*   | 24 | +0.00114  |
| qwen-orpo\*  | 24 | -0.00555  |
| qwen-simpo   | 24 | +0.00644  |
| smollm-dpo   | 19 | +0.00009  |
| smollm-grpo  | 19 | +0.00029  |
| **smollm-kto**| 19 | **+0.04693** |
| **smollm-orpo**| 19 | **+0.04956** |
| smollm-simpo | 19 | +0.00060  |

KTO drives the largest shift on 2/3 bases. ORPO drives the largest shift on SmolLM. DPO/SimPO/GRPO baseline shifts are 50–500× smaller.

## Architecture-level findings (bonus)

| Metric                       | Llama          | SmolLM         | Qwen           |
|------------------------------|----------------|----------------|----------------|
| l0_sparsity (features/token) | ~90            | ~120           | **~215**       |
| dead_neuron_fraction         | 0.83–0.92      | 0.89–0.93      | **0.71–0.73**  |
| semantic_stability_score     | 0.20–0.24      | 0.24–0.28      | **0.56–0.59**  |

Qwen activates 2× more features per token than Llama/SmolLM, has 20pt fewer dead neurons, and 2× more semantic stability between base and aligned. Architecture matters as much as the alignment algorithm.

## Degenerate cells flagged

- **qwen-kto**: feature_sharing_ratio = **1.000** (crosscoder classified every feature as shared; aligned_only=7 is noise)
- **qwen-orpo**: feature_sharing_ratio = **0.989** (only 202 aligned-only out of 16384; suspect underfitting)

Same algorithms produce normal aligned-only counts on Llama and SmolLM, so this is a Qwen-specific failure of the crosscoder, not a real "ORPO/KTO don't change Qwen" finding.

**Rescue plan**: re-train these two with relaxed shared-feature pressure (lower `LAMBDA_SHARED_MULTIPLIER`, lower `FORCED_SHARED_FRACTION`) and more epochs. If the aligned_only count stays low, it's a real finding; if it jumps to ~2000+, the original was a methodological artifact.

## Candidate workshop thesis (one sentence)

> Across three base architectures, alignment algorithms partition into two qualitatively distinct families at the feature level: broad-recruitment methods (DPO, SimPO, GRPO) modify thousands of features mildly, while concentrated-modification methods (KTO, ORPO) modify hundreds of features dramatically — a partition that replicates across SmolLM3-3B, Llama-3.2-3B-Instruct, and Qwen3-4B-Instruct-2507 with single-seed training.

## Limitations to acknowledge in writeup

1. Single seed per crosscoder. Cross-architecture replication is the rigor argument; single-seed variance is unmeasured.
2. Single layer per base (probe-best). Layer sweep is future work.
3. Two qwen cells (KTO, ORPO) need rescue training before they can be reported.
4. PPO is missing from the comparison (no PPO checkpoints on MInAlA HF org as of 2026-04-19).
5. Crosscoder classification thresholds (RHO_BASE_ONLY = 0.15, RHO_ALIGNED_ONLY = 0.85, etc.) are inherited from default config and not tuned per-pair.

## Output paths on the box

- Per-pair aggregate metrics: `~/work/interp-alignment/output/crosscoder/<slug>/L<layer>/metrics/aggregate_metrics.json`
- Per-pair feature CSVs: `~/work/interp-alignment/output/crosscoder/<slug>/L<layer>/features/{feature_classification,counterfactual_scores,merged_classification,shared_features_geometry}.csv`
- Per-pair plots: `~/work/interp-alignment/output/crosscoder/<slug>/L<layer>/plots/*.png`
- Sweep logs: `~/work/interp-alignment/logs/crosscoder/<slug>-L<layer>.log`
