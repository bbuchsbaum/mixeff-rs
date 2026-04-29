# Cross-Engine Parity Scoreboard

This scoreboard records cross-engine outcomes as comparison data, not as
ground truth. The Rust pathology certificate remains the contract oracle.
Fully identified fixtures gate coefficient checks on log-likelihood parity
first.

| fixture | certificate stratum | lme4 | MixedModels.jl | rust | verdict |
| --- | --- | --- | --- | --- | --- |
| easy_full_rank | easy | ok | ok | ConvergedInterior | parity |
| reduced_rank_unit_correlation | reduced_rank | ok | ok | ConvergedReducedRank | documented_divergence |

The reduced-rank fixture is an intentional divergence example: both external
engines can return an ordinary `ok` fit on the deterministic draw, while the
Rust certificate classifies the truth as `ConvergedReducedRank`. This is
retained as signal about engine behavior, not a regression.
