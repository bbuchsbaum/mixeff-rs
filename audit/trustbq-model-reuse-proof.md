# TrustBQ model-reuse experiment

Date: 2026-09-03

Base commit: `d6b81ce8b5974b2f0e42fcf650ad6811e5202a4f`

Experiment branch: `experiment/trustbq-optimizer-gpt56`

Paired proof workflow run: `33758373078`

CMA-ES probe workflow run: `33759923730`

## Candidate

The native TrustBQ implementation normally rebuilds a finite-difference quadratic model after every trust-region iteration. The candidate conservatively maintains that model in two cases:

1. After a rejected trial, the centre is unchanged. The candidate may reuse the existing model once while shrinking the radius.
2. After a very well-predicted accepted step (`ratio >= eta_expand`), the candidate may translate the same quadratic once for small problems (`d_theta <= 3`): if `q(s) = f + g' s + 0.5 s' H s`, moving the origin by `p` gives `g_new = g + H p` and the same `H`.

The crossed/large family receives rejected-step reuse only. Moderate problems also receive rejected-step reuse only. All auxiliary active-face and joint-GLMM solves keep reuse disabled. Defaults inside the generic TrustBQ engine remain zero, so only the profiled-LMM family policy opts in.

## Paired benchmark design

The repository's own `optimizer_bench_harness` was compiled twice from the same checkout: an unmodified baseline binary and a candidate binary. Five paired rounds were run, alternating execution order. Each harness invocation retained one warm-up and five measured repetitions. The proof covered five `d_theta = 3` vector models and three `d_theta = 9` crossed models. Scalar models were intentionally excluded because they use the scalar pattern-search path rather than TrustBQ.

The gate required both methods to pass the harness objective check, baseline/candidate objectives to agree within `1e-6 * (1 + |reference objective|)`, no evaluation-count regression, at least three scenarios with at least an 8% evaluation reduction, no scenario-level median wall-time regression worse than 5%, and an aggregate median speedup of at least 1.10x. The complete `cargo test --no-default-features` suite ran against the candidate before benchmarking.

## Results

| Scenario | Baseline evals | Candidate evals | Reduction | Baseline median ms | Candidate median ms | Speedup | Candidate - baseline objective |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector_180 | 286 | 201 | 29.72% | 0.640 | 0.550 | 1.164x | 1.70e-8 |
| vector_1000 | 268 | 145 | 45.90% | 2.536 | 1.675 | 1.514x | 1.80e-8 |
| vector_5000 | 248 | 169 | 31.85% | 7.911 | 5.767 | 1.372x | -8.70e-8 |
| vector_10000 | 337 | 145 | 56.97% | 18.823 | 11.712 | 1.607x | 1.71e-7 |
| vector_deep_200x50 | 337 | 204 | 39.47% | 8.677 | 5.633 | 1.540x | 2.20e-8 |
| crossed_small | 665 | 460 | 30.83% | 19.328 | 14.790 | 1.307x | 1.93e-4 |
| crossed_medium | 554 | 407 | 26.53% | 64.097 | 51.722 | 1.239x | -8.10e-4 |
| crossed_large | 498 | 483 | 3.01% | 191.416 | 187.146 | 1.023x | 3.29e-4 |

Across the eight scenarios, total reported objective evaluations fell from 3,193 to 2,214, a 30.66% reduction. The median scenario-level wall-time speedup was 1.339x. The candidate was faster in all 40 paired scenario-by-round comparisons. Every objective difference was far inside the predeclared statistical equivalence tolerance, and every scenario continued to pass its external reference-objective gate.

The largest observed baseline/candidate parameter differences across the paired records were small relative to the flat profiled-likelihood surfaces: maximum absolute theta differences were about 2.8e-6 for the single-block vector cases and 1.85e-3 for `crossed_small`; maximum fixed-effect differences were 8.5e-5 or smaller. These are not bit-identical paths, so the result is objective-equivalent rather than trajectory-equivalent.

## CMA-ES comparison

The external probe used released crate `cmaes 0.2.2` and 15 deterministic configurations per scenario: five seeds crossed with initial sigmas 0.25, 0.50, and 1.00. The CMA-ES budget was 600 evaluations for `vector_1000` and 1,200 for `crossed_small`. Cholesky parameters were mapped smoothly from unconstrained CMA coordinates into a generous finite box. A hybrid variant then requested a 500-evaluation candidate-TrustBQ polish from the best CMA solution.

| Scenario | Direct CMA success | Median CMA evals | Median first-target eval among successes | Median CMA wall ms | Hybrid success | Median hybrid total evals | Median hybrid wall ms | Conservative TrustBQ reference used by probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vector_1000 | 15/15 | 602 | 263 | 3.422 | 15/15 | 653 | 4.040 | 223 evals / 1.887 ms |
| crossed_small | 2/15 | 1,200 | 1,011.5 | 20.572 | 11/15 | 1,405 | 27.928 | 478 evals / 13.600 ms |

The TrustBQ comparison values embedded in this probe came from an earlier conservative paired run; the proof table above subsequently recorded 145 and 460 evaluations for the same two scenarios. Thus the updated evidence strengthens rather than weakens the conclusion.

Direct CMA-ES used more evaluations and more wall time even on the easy vector case. It was unreliable on the crossed case. Hybrid CMA-ES plus TrustBQ improved robustness but remained slower, much more evaluation-intensive, and not fully reliable. The probe's decision was `do_not_promote`.

This does not imply that CMA-ES has no role in mixed models. It remains plausible as an explicitly opt-in global restart for genuinely noisy, discontinuous, or multimodal custom objectives. It is not competitive as the standard optimizer for this smooth deterministic profiled (RE)ML objective.

## PRIMA note

A separate forced-optimizer probe found that the current PRIMA wrapper returned after one to ten evaluations without moving from the initial point, failing every reference-objective gate, while the existing NLopt path passed. This identifies a PRIMA integration problem; it is not evidence against Powell's BOBYQA algorithm itself. PRIMA should be debugged separately before it is used as a TrustBQ replacement.

## Conclusion

Promote the conservative TrustBQ model-maintenance patch for the dependency-light native LMM path. Keep NLopt BOBYQA/NEWUOA as the default-feature backend, because it remains more evaluation-efficient on small vector models and is already the repository's preferred release path. Do not promote CMA-ES as the routine profiled-LMM optimizer.