// Node driver for the wasm32 build of the harness (see run.sh).
// Usage: node run_node.mjs <pkg-dir> <reps> [design...]
// Times each exported call with performance.now(); wasm32-unknown-unknown has
// no std clock, so timing must come from the host. Same JSON-lines schema as
// the native driver.
import { createRequire } from "node:module";
import path from "node:path";
import { performance } from "node:perf_hooks";

const [pkgDir, repsArg, ...designArgs] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const bench = require(path.resolve(pkgDir, "mixeff_wasm_crossed_bench.js"));
const reps = Number(repsArg ?? 30);
const designs = designArgs.length
  ? designArgs.map((n) => [Number(n), Number(n)])
  : [
      [16, 16],
      [24, 24],
      [32, 32],
      [40, 40],
      [64, 64],
    ];
const platform = `wasm32-node-${process.versions.node}`;

function emit(mode, np, ni, seed, fitMs, sattMs, s, sat, method = null) {
  console.log(
    JSON.stringify({
      platform,
      mode,
      n_participants: np,
      n_items: ni,
      seed,
      fit_ms: fitMs,
      satterthwaite_ms: sattMs,
      beta_condition: s[0],
      se_condition: s[1],
      singular: s[2] !== 0,
      converged: s[3] !== 0,
      feval: s[4],
      objective: s[5],
      satterthwaite_df: sat ? sat[0] : null,
      satterthwaite_p: sat ? sat[1] : null,
      satterthwaite_method: method,
    }),
  );
}

for (const [np, ni] of designs) {
  // Untimed warm-up so the first timed fit is not paying for JIT tier-up.
  bench.simulate(np, ni, 999999);
  bench.fit();
  bench.satterthwaite();
  for (let seed = 1; seed <= reps; seed++) {
    bench.simulate(np, ni, seed);
    let t = performance.now();
    const s = bench.fit();
    const fitMs = performance.now() - t;
    t = performance.now();
    const sat = bench.satterthwaite();
    const sattMs = performance.now() - t;
    emit("fit", np, ni, seed, fitMs, sattMs, s, sat, bench.satterthwaite_reason());
  }
  bench.prepare_template(np, ni, 0);
  for (let seed = 1; seed <= reps; seed++) {
    bench.simulate(np, ni, 1000 + seed);
    const t = performance.now();
    const s = bench.refit();
    emit("refit", np, ni, seed, performance.now() - t, null, s, null);
  }
}
