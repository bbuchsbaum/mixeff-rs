# mixeff Fixture Speed Parity

Targeted speed and numeric parity harness for the large fixtures that feed the
mixeff R bridge tests.

Run Rust first:

```sh
MIXEFF_BENCH_REPEATS=3 MIXEFF_BENCH_WARMUPS=1 cargo run --release --features nlopt --example bench_mixeff_parity
```

Then run lme4 and generate the joined report:

```sh
MIXEFF_BENCH_REPEATS=3 MIXEFF_BENCH_WARMUPS=1 Rscript scripts/bench_mixeff_lme4.R
```

Outputs:

- `rust_results.json`
- `lme4_results.json`
- `REPORT.md`

Set `MIXEFF_BENCH_ENFORCE=true` on the lme4 stage to fail when any matched
Rust fixture is slower than lme4 by minimum fit time. Keep that opt-in until the
engine work has made all targeted fixtures pass consistently on stable hardware.

Fixture paths default to `/Users/bbuchsbaum/code/mixeff/tests/fixtures`. Override
with:

- `MIXEFF_REPO`
- `BROWN_RT_CSV`
- `IAMCIERA_STOMATA_TSV`
- `SDAMR_SPEEDDATE_CSV`
