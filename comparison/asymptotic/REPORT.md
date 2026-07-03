# Asymptotic Benchmark — Rust vs lme4

Synthetic sleepstudy-shaped data, formula `reaction ~ 1 + days + (1 + days | subj)`.

Sources: **mixeff-rs** vs **lme4 2.0.1**.

Each side runs 3 warmup + 5 measured fits; medians reported.

| label | n | t_R median (ms) | t_Rust median (ms) | speedup (median) | t_Rust min | t_R min | speedup (min) | Rust fevals | R fevals | Δ obj | Δ σ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s | 1000 | 11.0 | 0.43 | **25.8×** | 0.42 | 10.0 | 23.6× | 58 | 62 | 0.0000 | 0.0000 |
| m | 5000 | 25.0 | 1.22 | **20.4×** | 1.18 | 24.0 | 20.3× | 40 | 42 | 0.0000 | 0.0000 |
| l | 20000 | 81.0 | 5.13 | **15.8×** | 5.02 | 78.0 | 15.5× | 46 | 40 | 0.0000 | 0.0000 |
| xl | 100000 | 495.0 | 26.07 | **19.0×** | 25.70 | 434.0 | 16.9× | 40 | 43 | 0.0002 | 0.0002 |

## Rust phase breakdown (median)

| label | n | parse + build | fit (optimizer) | total | optimizer |
|---|---:|---:|---:|---:|---|
| s | 1000 | 0.10 ms | 0.33 ms | 0.43 ms | bobyqa |
| m | 5000 | 0.46 ms | 0.76 ms | 1.22 ms | bobyqa |
| l | 20000 | 1.95 ms | 3.18 ms | 5.13 ms | bobyqa |
| xl | 100000 | 10.74 ms | 15.39 ms | 26.07 ms | bobyqa |
