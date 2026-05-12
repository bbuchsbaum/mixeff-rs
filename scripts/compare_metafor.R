#!/usr/bin/env Rscript
# Cross-check the Rust summary-estimate (meta-analysis) LMM front door
# against metafor::rma.mv on the Berkey et al. (1995) BCG vaccine dataset.
#
# Usage:
#   Rscript scripts/compare_metafor.R
#
# This script:
#   1. Embeds the BCG raw counts so it runs without external data files.
#   2. Computes log relative risk (yi) and its sampling variance (vi) from
#      the raw counts using the standard formula (no metafor::escalc
#      dependency — keeps yi/vi byte-identical between R and Rust).
#   3. Writes comparison/metafor/bcg_yi_vi.csv (the shared input fixture).
#   4. Fits a random-effect intercept model with metafor::rma.mv (REML).
#   5. Writes comparison/metafor/metafor_results.json with beta, SE, vcov,
#      tau^2, and the log-likelihood.
#
# The Rust counterpart (examples/compare_metafor.rs) reads the same CSV,
# fits via LinearMixedModel::from_summary_estimates, and writes
# comparison/metafor/rust_results.json. Pair them up by hand or with a
# downstream comparison harness.

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required: install.packages('jsonlite').")
  }
  library(jsonlite)
})

have_metafor <- requireNamespace("metafor", quietly = TRUE)

find_repo_root <- function(start = getwd()) {
  d <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(d, "Cargo.toml"))) return(d)
    parent <- dirname(d)
    if (parent == d) {
      stop("could not find Cargo.toml ancestor; run from repo or a subdir.")
    }
    d <- parent
  }
}

repo_root <- find_repo_root()
out_dir <- file.path(repo_root, "comparison", "metafor")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Berkey et al. (1995) BCG vaccine dataset. Public domain. Same as
# metafor::dat.bcg but inlined so this script is self-contained.
bcg <- data.frame(
  trial = 1:13,
  tpos  = c(  4,   6,   3,  62, 33, 180,  8,505,29,17,186,  5, 27),
  tneg  = c(119, 300, 228, 13536,5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886),
  cpos  = c( 11,  29,  11,248,47, 372, 10,499, 45,65, 141,  3, 29),
  cneg  = c(128, 274, 209, 12619, 5761,1079, 619, 87892, 7232, 1600, 27197, 2338, 17825)
)

# Standard formula for log RR + sampling variance, matches metafor::escalc
# with measure="RR".
bcg$yi <- with(bcg, log((tpos * (cpos + cneg)) / ((tpos + tneg) * cpos)))
bcg$vi <- with(bcg, 1 / tpos - 1 / (tpos + tneg) + 1 / cpos - 1 / (cpos + cneg))

write.csv(
  bcg[, c("trial", "yi", "vi")],
  file = file.path(out_dir, "bcg_yi_vi.csv"),
  row.names = FALSE
)

cat("wrote ", file.path("comparison", "metafor", "bcg_yi_vi.csv"), "\n", sep = "")

if (!have_metafor) {
  message(
    "metafor is not installed; CSV fixture written, skipping rma.mv fit.\n",
    "Install with: install.packages('metafor', repos = 'https://cloud.r-project.org')"
  )
  quit(status = 0)
}

# Random-effect intercept model: yi ~ 1 with random study-level intercept.
# Equivalent to rma() with method="REML" but written via rma.mv to keep the
# random-structure explicit and parallel to the Rust formula
# `yi ~ 1 + (1 | trial)`.
fit <- metafor::rma.mv(
  yi   = yi,
  V    = vi,
  random = ~ 1 | trial,
  data = bcg,
  method = "REML"
)

# Random-effect SD from the rma.mv sigma2 vector. For this single-component
# model sigma2 == tau^2.
tau2 <- as.numeric(fit$sigma2)

log_lik <- as.numeric(stats::logLik(fit))

results <- list(
  schema_version = 1L,
  source = "metafor::rma.mv",
  method = "REML",
  fixture = "Berkey 1995 BCG vaccine (n=13)",
  formula = "yi ~ 1 + (1 | trial)",
  n_studies = nrow(bcg),
  beta = as.numeric(fit$beta),
  se   = as.numeric(fit$se),
  zval = as.numeric(fit$zval),
  pval = as.numeric(fit$pval),
  ci_lb = as.numeric(fit$ci.lb),
  ci_ub = as.numeric(fit$ci.ub),
  vcov_beta = as.matrix(fit$vb),
  tau_sq = tau2,
  tau_sd = sqrt(tau2),
  log_likelihood = log_lik
)

writeLines(
  jsonlite::toJSON(results, pretty = TRUE, digits = 17, auto_unbox = TRUE),
  con = file.path(out_dir, "metafor_results.json")
)

cat("wrote ", file.path("comparison", "metafor", "metafor_results.json"), "\n", sep = "")
cat("beta = ", results$beta, ", tau^2 = ", tau2, ", logLik = ", results$log_likelihood, "\n", sep = "")
