# Summary-Estimate Parity vs `metafor::rma.mv`

Cross-checks the Rust [`LinearMixedModel::from_summary_estimates`] front
door against R's `metafor::rma.mv` on the Berkey et al. (1995) BCG vaccine
dataset.

## Workflow

```bash
# 1. Generate the shared input fixture (yi, vi). Also runs rma.mv if
#    metafor is installed and writes metafor_results.json.
Rscript scripts/compare_metafor.R

# 2. Fit on the Rust side and dump rust_results.json.
cargo run --release --example compare_metafor

# 3. Diff the two JSON files. (A formal integration test lives in
#    tests/summary_estimate_parity.rs; see below.)
```

## Files

| File | Owner | Description |
|---|---|---|
| `bcg_yi_vi.csv` | R script | Shared input: `trial`, log-RR `yi`, sampling variance `vi` |
| `metafor_results.json` | R script | `rma.mv` REML fit (β̂, SE, vcov, τ², log-likelihood) |
| `rust_results.json` | Rust example | Same fields from `from_summary_estimates` + `fit(REML)` |

## Reference values (measured against a fresh `metafor::rma.mv` REML fit)

| field    | Rust                | metafor              | \|Δ\|       | tol  | status |
|----------|---------------------|----------------------|-------------|------|--------|
| β̂       | -0.7145314134       | -0.7145323673        | 9.5e-7      | 1e-6 | OK     |
| SE(β̂)   |  0.1797791706       |  0.1797815796        | 2.4e-6      | 1e-5 | OK     |
| τ²       |  0.3132331064       |  0.3132435332        | 1.0e-5      | 1e-4 | OK     |
| logLik   | -13.4848460961      | -12.2023714155       | 1.28e0      | —    | info   |

The β̂ / SE / τ² agreement is the inferential parity that matters: any
Wald or LRT-style summary computed off these matches across packages.

The log-likelihood difference is a **REML normalization convention**
difference between `metafor::rma.mv` and the lme4-style PLS REML log-lik
this crate uses (also seen between `lme4::lmer` and `nlme::lme`). The
offset is study-set-dependent but model-independent on a given dataset,
so likelihood-ratio statistics (deviance differences) on the same data
still match. See
[`metafor` docs / `tips:rma_vs_lm_lme_lmer`](https://www.metafor-project.org/doku.php/tips:rma_vs_lm_lme_lmer).

## Installing metafor

```bash
Rscript -e 'install.packages("metafor", repos = "https://cloud.r-project.org")'
```

The R script writes the CSV fixture even when `metafor` is missing, so
the Rust side can run independently.
