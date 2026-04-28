#!/usr/bin/env Rscript
# Companion to examples/bench_asymptotic.rs. Reads each scenario's data.csv
# (written by the Rust side) and times lmer() on the same formula. Writes
# comparison/asymptotic/lme4_results.json with matching schema.

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

WARMUP_RUNS  <- 3L
MEASURED_RUNS <- 5L
FORMULA <- reaction ~ 1 + days + (1 + days | subj)

find_repo_root <- function(start = getwd()) {
  d <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(d, "Cargo.toml"))) return(d)
    parent <- dirname(d)
    if (parent == d) stop("could not find Cargo.toml ancestor")
    d <- parent
  }
}

repo_root <- find_repo_root()
asymp_dir <- file.path(repo_root, "comparison", "asymptotic")
if (!dir.exists(asymp_dir)) {
  stop("comparison/asymptotic not populated — run `cargo run --release --example bench_asymptotic` first.")
}

# Discover scenarios = subdirectories with a data.csv.
scenarios <- list.dirs(asymp_dir, recursive = FALSE)
scenarios <- scenarios[file.exists(file.path(scenarios, "data.csv"))]
labels <- basename(scenarios)
order_lookup <- c(s = 1, m = 2, l = 3, xl = 4)
ord <- order(order_lookup[labels])
scenarios <- scenarios[ord]; labels <- labels[ord]

run_one <- function(label, dir) {
  cat(sprintf("\n=== scenario %s ===\n", label))
  df <- utils::read.csv(file.path(dir, "data.csv"), stringsAsFactors = FALSE)
  df$subj <- factor(df$subj)
  cat(sprintf("  loaded %d rows, %d subjects\n", nrow(df), nlevels(df$subj)))

  fit_once <- function() {
    lme4::lmer(FORMULA, data = df, REML = TRUE,
               control = lme4::lmerControl(calc.derivs = FALSE))
  }

  for (i in seq_len(WARMUP_RUNS)) invisible(fit_once())

  times <- numeric(MEASURED_RUNS)
  last_obj <- NA_real_; last_sigma <- NA_real_; last_fevals <- NA_integer_
  for (k in seq_len(MEASURED_RUNS)) {
    t0 <- proc.time()[["elapsed"]]
    m <- fit_once()
    times[k] <- (proc.time()[["elapsed"]] - t0) * 1000
    last_obj <- as.numeric(REMLcrit(m))
    last_sigma <- attr(lme4::VarCorr(m), "sc")
    last_fevals <- as.integer(m@optinfo$feval %||% NA_integer_)
    cat(sprintf("  run %d: total=%.1f ms (fevals=%s)\n", k, times[k], last_fevals))
  }

  list(
    label = label,
    n_subjects = nlevels(df$subj),
    n_obs_per_subject = nrow(df) %/% nlevels(df$subj),
    n_obs = nrow(df),
    formula = "reaction ~ 1 + days + (1 + days | subj)",
    fit_time_ms_min    = min(times),
    fit_time_ms_median = stats::median(times),
    fevals = last_fevals,
    objective = last_obj,
    sigma = last_sigma
  )
}

results <- mapply(run_one, labels, scenarios, SIMPLIFY = FALSE, USE.NAMES = FALSE)

out <- list(
  tool = paste0("lme4 ", as.character(packageVersion("lme4"))),
  R_version = paste(R.version$major, R.version$minor, sep = "."),
  results = results
)
out_path <- file.path(asymp_dir, "lme4_results.json")
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null"),
           con = out_path)
cat(sprintf("\nwrote %s\n", out_path))
