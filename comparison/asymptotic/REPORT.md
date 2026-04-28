# Asymptotic Benchmark — Rust vs lme4

Synthetic sleepstudy-shaped data, formula `reaction ~ 1 + days + (1 + days | subj)`.

Sources: **mixedmodels (rust)** vs **lme4 2.0.1**.

Each side runs 3 warmup + 5 measured fits; medians reported.

| label | n | t_R median (ms) | t_Rust median (ms) | speedup (median) | t_Rust min | t_R min | speedup (min) | Rust fevals | R fevals | Δ obj | Δ σ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s | 1000 | 11.0 | 0.34 | **32.1×** | 0.34 | 11.0 | 32.8× | 68 | 62 | 0.0000 | 0.0001 |
| m | 5000 | 25.0 | 0.85 | **29.6×** | 0.83 | 23.0 | 27.6× | 40 | 42 | 0.0000 | 0.0000 |
| l | 20000 | 82.0 | 3.25 | **25.2×** | 3.17 | 78.0 | 24.6× | 42 | 40 | 0.0000 | 0.0000 |
| xl | 100000 | 508.0 | 17.20 | **29.5×** | 16.66 | 446.0 | 26.8× | 40 | 43 | 0.0002 | 0.0002 |

## Rust phase breakdown (median)

| label | n | parse + build | fit (optimizer) | total | optimizer |
|---|---:|---:|---:|---:|---|
| s | 1000 | 0.08 ms | 0.27 ms | 0.34 ms | bobyqa |
| m | 5000 | 0.36 ms | 0.48 ms | 0.85 ms | bobyqa |
| l | 20000 | 1.32 ms | 1.93 ms | 3.25 ms | bobyqa |
| xl | 100000 | 7.74 ms | 9.41 ms | 17.20 ms | bobyqa |
