#!/usr/bin/env Rscript
# Targeted lme4 timing companion for examples/bench_mixeff_parity.rs.

suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

find_repo_root <- function(start = getwd()) {
  d <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(d, "Cargo.toml"))) return(d)
    parent <- dirname(d)
    if (parent == d) stop("could not find Cargo.toml ancestor")
    d <- parent
  }
}

fixture_path <- function(env_name, default) {
  value <- Sys.getenv(env_name, unset = NA_character_)
  if (!is.na(value) && nzchar(value)) value else default
}

mixeff_root <- Sys.getenv("MIXEFF_REPO", unset = "/Users/bbuchsbaum/code/mixeff")
fixture_dir <- file.path(mixeff_root, "tests", "fixtures")
repo_root <- find_repo_root()
out_dir <- file.path(repo_root, "comparison", "mixeff")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

warmups <- as.integer(Sys.getenv("MIXEFF_BENCH_WARMUPS", unset = "0"))
repeats <- as.integer(Sys.getenv("MIXEFF_BENCH_REPEATS", unset = "1"))
enforce <- identical(tolower(Sys.getenv("MIXEFF_BENCH_ENFORCE", unset = "false")), "true")

load_brown_rt <- function() {
  path <- fixture_path("BROWN_RT_CSV", file.path(fixture_dir, "brown_rt_dummy_data.csv"))
  dat <- utils::read.csv(path, stringsAsFactors = FALSE)
  dat$PID <- factor(dat$PID)
  dat$stim <- factor(dat$stim)
  dat$modality <- ifelse(dat$modality == "Audio-only", 0, 1)
  dat
}

load_iamciera_stomata <- function() {
  path <- fixture_path("IAMCIERA_STOMATA_TSV", file.path(fixture_dir, "iamciera_modeling_example.txt"))
  dat <- utils::read.delim(path, stringsAsFactors = TRUE)
  dat$trans_abs_stom <- sqrt(dat$abs_stom)
  dat
}

load_sdamr_speeddate <- function() {
  path <- fixture_path("SDAMR_SPEEDDATE_CSV", file.path(fixture_dir, "sdamr_speeddate_lmm.csv"))
  dat <- utils::read.csv(path, stringsAsFactors = TRUE)
  dat$iid <- factor(dat$iid)
  dat$pid <- factor(dat$pid)
  dat <- stats::na.omit(dat)
  dat$attr_by_intel <- dat$other_attr_c * dat$other_intel_c
  dat
}

cases <- list(
  brown_rt_full = list(
    fixture = "brown_rt_dummy_data",
    data = load_brown_rt,
    formula = RT ~ 1 + modality + (1 + modality | PID) + (1 + modality | stim),
    formula_text = "RT ~ 1 + modality + (1 + modality | PID) + (1 + modality | stim)",
    control = lme4::lmerControl(calc.derivs = FALSE, optimizer = "bobyqa")
  ),
  iamciera_max_model = list(
    fixture = "iamciera_modeling_example",
    data = load_iamciera_stomata,
    formula = trans_abs_stom ~ il + (1 | tray) + (1 | row) + (1 | col),
    formula_text = "trans_abs_stom ~ il + (1 | tray) + (1 | row) + (1 | col)",
    control = lme4::lmerControl(calc.derivs = FALSE)
  ),
  sdamr_speeddate_maximal_crossed = list(
    fixture = "sdamr_speeddate_lmm",
    data = load_sdamr_speeddate,
    formula = other_like ~ other_attr_c + other_intel_c + attr_by_intel +
      (1 + other_attr_c + other_intel_c + attr_by_intel | iid) +
      (1 + other_attr_c + other_intel_c + attr_by_intel | pid),
    formula_text = "other_like ~ other_attr_c + other_intel_c + attr_by_intel + (1 + other_attr_c + other_intel_c + attr_by_intel | iid) + (1 + other_attr_c + other_intel_c + attr_by_intel | pid)",
    control = lme4::lmerControl(calc.derivs = FALSE)
  ),
  sdamr_speeddate_uncorrelated_crossed = list(
    fixture = "sdamr_speeddate_lmm",
    data = load_sdamr_speeddate,
    formula = other_like ~ other_attr_c + other_intel_c + attr_by_intel +
      (1 + other_attr_c + other_intel_c || iid) +
      (1 + other_attr_c + other_intel_c || pid),
    formula_text = "other_like ~ other_attr_c + other_intel_c + attr_by_intel + (1 + other_attr_c + other_intel_c || iid) + (1 + other_attr_c + other_intel_c || pid)",
    control = lme4::lmerControl(calc.derivs = FALSE)
  )
)

args <- commandArgs(trailingOnly = TRUE)
selected <- if (length(args) == 0L || identical(args[[1]], "all")) names(cases) else args[[1]]
missing <- setdiff(selected, names(cases))
if (length(missing) > 0L) {
  stop(sprintf("unknown benchmark case(s): %s", paste(missing, collapse = ", ")))
}

median_or_null <- function(x) if (length(x) == 0L) NULL else stats::median(x)
min_or_null <- function(x) if (length(x) == 0L) NULL else min(x)

fit_once <- function(case, dat) {
  suppressMessages(suppressWarnings(
    lme4::lmer(case$formula, data = dat, REML = TRUE, control = case$control)
  ))
}

case_result <- function(id, case) {
  cat(sprintf("case: %s\n", id))
  dat <- case$data()
  for (i in seq_len(warmups)) invisible(fit_once(case, dat))

  times <- numeric()
  last_fit <- NULL
  last_error <- NULL
  for (i in seq_len(repeats)) {
    t0 <- proc.time()[["elapsed"]]
    fit <- tryCatch(fit_once(case, dat), error = identity)
    elapsed_ms <- (proc.time()[["elapsed"]] - t0) * 1000
    if (inherits(fit, "error")) {
      last_error <- conditionMessage(fit)
      cat(sprintf("  run %d failed: %s\n", i, last_error))
    } else {
      times <- c(times, elapsed_ms)
      last_fit <- fit
      cat(sprintf("  run %d: fit=%.1f ms feval=%s\n",
                  i, elapsed_ms, as.character(fit@optinfo$feval %||% NA_integer_)))
    }
  }

  if (is.null(last_fit)) {
    return(list(
      case_id = id, fixture = case$fixture, formula = case$formula_text,
      estimator = "REML", n_obs = nrow(dat), status = "error", error = last_error,
      fit_time_ms_min = NULL, fit_time_ms_median = NULL, fit_time_ms_repeats = length(times)
    ))
  }

  list(
    case_id = id,
    fixture = case$fixture,
    formula = case$formula_text,
    estimator = "REML",
    n_obs = nrow(dat),
    fit_time_ms_min = min_or_null(times),
    fit_time_ms_median = median_or_null(times),
    fit_time_ms_repeats = length(times),
    fevals = as.integer(last_fit@optinfo$feval %||% NA_integer_),
    objective = as.numeric(REMLcrit(last_fit)),
    sigma = as.numeric(attr(lme4::VarCorr(last_fit), "sc")),
    beta = as.numeric(lme4::fixef(last_fit)),
    coef_names = names(lme4::fixef(last_fit)),
    is_singular = isTRUE(lme4::isSingular(last_fit, tol = 1e-4)),
    status = "ok",
    error = NULL
  )
}

results <- lapply(selected, function(id) case_result(id, cases[[id]]))

out <- list(
  schema_name = "mixedmodels.mixeff_speed_parity",
  schema_version = "1.0.0",
  engine = "lme4",
  tool = paste0("lme4 ", as.character(utils::packageVersion("lme4"))),
  R_version = paste(R.version$major, R.version$minor, sep = "."),
  warmups = warmups,
  repeats = repeats,
  results = results
)

lme4_path <- file.path(out_dir, "lme4_results.json")
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null"), con = lme4_path)
cat(sprintf("wrote %s\n", lme4_path))

rust_path <- file.path(out_dir, "rust_results.json")
if (file.exists(rust_path)) {
  rust <- jsonlite::fromJSON(rust_path, simplifyVector = FALSE)
  rust_by_id <- setNames(rust$results, vapply(rust$results, `[[`, character(1), "case_id"))
  lme4_by_id <- setNames(results, vapply(results, `[[`, character(1), "case_id"))

  rows <- c(
    "# mixeff Fixture Speed Parity",
    "",
    sprintf("Rust source: `%s`", rust$tool %||% "unknown"),
    sprintf("lme4 source: `%s`", out$tool),
    "",
    "| case | n | Rust min ms | lme4 min ms | lme4/Rust | Rust feval | lme4 feval | status |",
    "|---|---:|---:|---:|---:|---:|---:|---|"
  )

  failures <- character()
  for (id in intersect(names(rust_by_id), names(lme4_by_id))) {
    rr <- rust_by_id[[id]]
    lr <- lme4_by_id[[id]]
    rust_ms <- rr$fit_time_ms_min
    lme4_ms <- lr$fit_time_ms_min
    speedup <- if (!is.null(rust_ms) && !is.null(lme4_ms) && rust_ms > 0) lme4_ms / rust_ms else NA_real_
    status <- if (is.finite(speedup) && speedup >= 1.0) "pass" else "rust_slower"
    if (identical(status, "rust_slower")) failures <- c(failures, id)
    rows <- c(rows, sprintf(
      "| `%s` | %s | %.1f | %.1f | %.2fx | %s | %s | %s |",
      id,
      as.character(rr$n_obs %||% lr$n_obs %||% NA_integer_),
      rust_ms %||% NA_real_,
      lme4_ms %||% NA_real_,
      speedup,
      as.character(rr$fevals %||% NA_integer_),
      as.character(lr$fevals %||% NA_integer_),
      status
    ))
  }
  report_path <- file.path(out_dir, "REPORT.md")
  writeLines(rows, con = report_path)
  cat(sprintf("wrote %s\n", report_path))
  if (enforce && length(failures) > 0L) {
    stop(sprintf("Rust slower than lme4 for: %s", paste(failures, collapse = ", ")))
  }
}
