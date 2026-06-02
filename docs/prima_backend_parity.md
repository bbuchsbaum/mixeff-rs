# PRIMA Backend Parity

This page records the current Rust parity state for the
`backend = :prima` optimizer matrix exercised by MixedModels.jl.
It is an availability manifest, not a numeric fit oracle.

PRIMA is intentionally not a default dependency. Numeric PRIMA fits in Rust
require `--features prima` and a system `libprimac`; default builds continue to
use the existing native/NLopt paths.

When `libprimac` is installed outside the platform linker defaults, set
`PRIMA_DIR` to the PRIMA install prefix. `build.rs` adds
`$PRIMA_DIR/lib` to the native link search path for `--features prima`.

| Julia backend | Julia optimizer | Rust optimizer | Rust status |
| --- | --- | --- | --- |
| `prima` | `bobyqa` | `PrimaBobyqa` | `feature_gated_system_lib` |
| `prima` | `cobyla` | `PrimaCobyla` | `reserved_unavailable` |
| `prima` | `lincoa` | `PrimaLincoa` | `reserved_unavailable` |
| `prima` | `newuoa` | `PrimaNewuoa` | `reserved_unavailable` |

All PRIMA variants use the PRIMA backend display label and the PRIMA option
surface `rhobeg`, `rhoend`, and `maxfeval`. `PrimaBobyqa` is the only variant
wired to the LMM optimizer path today, and only behind the non-default feature.
COBYLA, LINCOA, and NEWUOA remain explicit unavailable states until the C API
coverage and parity tests are expanded.
