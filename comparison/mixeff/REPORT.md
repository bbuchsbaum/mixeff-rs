# mixeff Fixture Speed Parity

Rust source: `cargo run --release --features nlopt --example bench_mixeff_parity`
lme4 source: `lme4 2.0.1`

| case | n | Rust min ms | lme4 min ms | lme4/Rust | Rust feval | lme4 feval | status |
|---|---:|---:|---:|---:|---:|---:|---|
| `brown_rt_full` | 21679 | 270.2 | 2618.0 | 9.69x | 243 | 531 | pass |
| `iamciera_max_model` | 727 | 16.9 | 40.0 | 2.36x | 58 | 41 | pass |
| `sdamr_speeddate_maximal_crossed` | 1509 | 1118.2 | 1160.0 | 1.04x | 507 | 613 | pass |
| `sdamr_speeddate_uncorrelated_crossed` | 1509 | 101.4 | 128.0 | 1.26x | 116 | 115 | pass |
