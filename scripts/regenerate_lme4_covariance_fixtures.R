#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) {
  args[[1]]
} else {
  file.path("tests", "fixtures", "parity", "lme4_covariance_families.json")
}

extract_case <- function(id, rust_formula, lme4_formula, covariance_family, fit) {
  list(
    id = id,
    rust_formula = rust_formula,
    lme4_formula = lme4_formula,
    reml = isREML(fit),
    covariance_family = covariance_family,
    beta = unname(fixef(fit)),
    sigma = unname(sigma(fit)),
    theta = unname(getME(fit, "theta")),
    objective = unname(-2 * as.numeric(logLik(fit))),
    loglik = unname(as.numeric(logLik(fit))),
    fitted_head = unname(head(fitted(fit), 10)),
    varcorr = lapply(seq_len(nrow(as.data.frame(VarCorr(fit)))), function(i) {
      row <- as.data.frame(VarCorr(fit))[i, ]
      list(
        group = as.character(row$grp),
        var1 = as.character(row$var1),
        var2 = if (is.na(row$var2)) NULL else as.character(row$var2),
        vcov = unname(row$vcov),
        sdcor = unname(row$sdcor)
      )
    })
  )
}

full_fit <- lmer(
  Reaction ~ 1 + Days + (1 + Days | Subject),
  data = sleepstudy,
  REML = FALSE
)

diagonal_fit <- lmer(
  Reaction ~ 1 + Days + (1 + Days || Subject),
  data = sleepstudy,
  REML = FALSE
)

fixture <- list(
  source = "Local R/lme4 covariance-family fixture generated from lme4::sleepstudy.",
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  lme4_version = as.character(packageVersion("lme4")),
  dataset = "lme4::sleepstudy",
  tolerances = list(
    beta_abs = 5e-4,
    sigma_abs = 5e-4,
    theta_abs = 5e-4,
    objective_abs = 5e-3,
    loglik_abs = 5e-3,
    fitted_abs = 5e-4,
    varcorr_sd_abs = 1e-3,
    varcorr_corr_abs = 5e-4
  ),
  cases = list(
    extract_case(
      "sleepstudy_full_ml",
      "Reaction ~ 1 + Days + (1 + Days | Subject)",
      "Reaction ~ 1 + Days + (1 + Days | Subject)",
      "full_cholesky",
      full_fit
    ),
    extract_case(
      "sleepstudy_diagonal_ml",
      "Reaction ~ 1 + Days + diag(1 + Days | Subject)",
      "Reaction ~ 1 + Days + (1 + Days || Subject)",
      "diagonal",
      diagonal_fit
    )
  )
)

dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, out, pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null")

repo_root <- tryCatch(
  system2("git", c("rev-parse", "--show-toplevel"), stdout = TRUE, stderr = FALSE)[1],
  error = function(e) getwd()
)
commit <- tryCatch(
  system2("git", c("-C", repo_root, "rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE)[1],
  error = function(e) NA_character_
)
provenance <- list(
  schema_version = "1.0",
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  crate_commit = if (is.na(commit)) NULL else commit,
  regenerator = "scripts/regenerate_lme4_covariance_fixtures.R",
  source_case = "lme4::sleepstudy",
  reference_engine = sprintf("lme4 %s", as.character(packageVersion("lme4"))),
  notes = sprintf(
    "R %s; ML full and zero-correlation sleepstudy covariance-family fixtures",
    getRversion()
  )
)
write_json(
  provenance,
  sub("\\.json$", ".provenance.json", out),
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null"
)
