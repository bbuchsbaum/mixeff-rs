#!/usr/bin/env Rscript
# Fit each entry in comparison/manifest.json (produced by `cargo run --example compare_rust`)
# with lme4 / nlme equivalents and emit comparison/lme4_results.json with
# matching schema. Shapes pair up downstream in `examples/compare_report.rs`.
#
# Usage:  Rscript scripts/compare_lme4.R
#
# Timing repeats: each fit is run TIMING_REPEATS times and the best (min)
# elapsed time is recorded alongside the cold start time, so transient OS
# noise doesn't dominate the comparison.

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

TIMING_REPEATS <- 3L
include_stress <- nzchar(Sys.getenv("MIXEDMODELS_INCLUDE_STRESS", unset = ""))

find_repo_root <- function(start = getwd()) {
  d <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(d, "Cargo.toml"))) return(d)
    parent <- dirname(d)
    if (parent == d) stop("could not find Cargo.toml ancestor; run from repo or a subdir.")
    d <- parent
  }
}

repo_root <- find_repo_root()
manifest_path <- file.path(repo_root, "comparison", "manifest.json")
if (!file.exists(manifest_path)) {
  stop("manifest.json not found — run `cargo run --release --example compare_rust` first.")
}
manifest <- jsonlite::read_json(manifest_path, simplifyVector = TRUE)$fits
if (is.data.frame(manifest)) manifest <- as.data.frame(manifest, stringsAsFactors = FALSE)

# Replace lme4-style proportion notation with the real cbind() form so glmer
# accepts it: "incidence / size ~ ..."  ->  "cbind(incidence, size - incidence) ~ ..."
adapt_formula_for_glmer <- function(f, weights_col) {
  if (!is.null(weights_col) && grepl(" / ", f, fixed = TRUE)) {
    f <- sub(
      "^([^ ]+) / ([^ ]+) ~",
      paste0("cbind(\\1, ", weights_col, " - \\1) ~"),
      f
    )
  }
  f
}

read_dataset <- function(name) {
  csv <- file.path(repo_root, "datasets", name, "data.csv")
  utils::read.csv(csv, stringsAsFactors = FALSE)
}

dataset_difficulty <- function(name) {
  meta_path <- file.path(repo_root, "datasets", name, "meta.toml")
  if (!file.exists(meta_path)) return(NA_character_)
  text <- readLines(meta_path, warn = FALSE)
  line <- grep('^difficulty *=', text, value = TRUE)
  if (!length(line)) return(NA_character_)
  sub('^difficulty *= *"([^"]+)".*', '\\1', line[1])
}

apply_canonical_factor_levels <- function(df, name) {
  meta_path <- file.path(repo_root, "datasets", name, "meta.toml")
  if (!file.exists(meta_path)) return(df)
  text <- readLines(meta_path)
  i <- 1
  while (i <= length(text)) {
    if (grepl("^\\[\\[columns\\]\\]", text[i])) {
      blk <- character()
      i <- i + 1
      # Slurp until the next top-level [...] / [[...]] heading.
      while (i <= length(text) && !grepl("^\\[", text[i])) {
        blk <- c(blk, text[i]); i <- i + 1
      }
      cn <- sub('^name *= *"([^"]+)".*', '\\1', grep('^name *=', blk, value = TRUE)[1])
      ct <- sub('^type *= *"([^"]+)".*', '\\1', grep('^type *=', blk, value = TRUE)[1])

      # Handle multi-line `levels = [...]`: stitch together all lines from the
      # one starting with `levels =` until the closing `]`.
      lvls <- NULL
      start <- which(grepl('^levels *=', blk))
      if (length(start)) {
        s <- start[1]
        joined <- blk[s]
        while (!grepl("\\]", joined) && s < length(blk)) {
          s <- s + 1
          joined <- paste(joined, blk[s])
        }
        body <- sub("^levels *= *\\[", "", joined)
        body <- sub("\\].*$", "", body)
        # Drop comments inside the array.
        body <- gsub("#[^,]*", "", body)
        toks <- strsplit(body, " *, *")[[1]]
        toks <- gsub('^[ \t"]+|[ \t"]+$', "", toks)
        toks <- toks[nzchar(toks)]
        lvls <- toks
      }

      if (identical(ct, "categorical") && cn %in% names(df)) {
        if (!is.null(lvls) && length(lvls) >= 1) {
          observed <- unique(as.character(df[[cn]]))
          missing_lvls <- setdiff(observed, lvls)
          if (length(missing_lvls)) {
            warning(sprintf(
              "%s$%s: %d observed values not in canonical levels (e.g. %s) — falling back to first-appearance order",
              name, cn, length(missing_lvls), paste(head(missing_lvls, 3), collapse = ", ")
            ))
            df[[cn]] <- factor(as.character(df[[cn]]), levels = observed)
          } else {
            df[[cn]] <- factor(as.character(df[[cn]]), levels = lvls)
          }
        } else {
          df[[cn]] <- factor(as.character(df[[cn]]),
                             levels = unique(as.character(df[[cn]])))
        }
      }
    } else {
      i <- i + 1
    }
  }
  df
}

fit_one <- function(entry) {
  ds_name <- entry$dataset
  est <- entry$estimator
  family <- entry$family
  link <- entry$link
  weights_col <- entry$weights
  formula_str <- entry$formula

  df <- read_dataset(ds_name)
  df <- apply_canonical_factor_levels(df, ds_name)

  is_gaussian <- identical(family, "Gaussian") && identical(link, "Identity")
  is_lmm <- is_gaussian && est %in% c("REML", "ML")

  result <- list(
    dataset = ds_name, formula = formula_str, family = family, link = link,
    estimator = est, n_obs = nrow(df),
    status = "skipped", error = NA_character_,
    beta = NULL, coef_names = NULL, sigma = NA_real_,
    theta = NULL, objective = NA_real_, loglik = NA_real_,
    aic = NA_real_, bic = NA_real_,
    objective_definition = NA_character_, response_constants = NA_character_,
    optimizer = NA_character_, optimizer_backend = NA_character_,
    optimizer_return_code = NA_character_, optimizer_fevals = NA_integer_,
    optimizer_fmin = NA_real_, optimizer_max_fevals = NA_integer_,
    is_singular = NA, fit_time_ms = NA_real_,
    fit_time_ms_min = NA_real_, fit_time_ms_repeats = NA_integer_,
    warnings = I(character())
  )

  if (identical(dataset_difficulty(ds_name), "stress") && !include_stress) {
    result$status <- "skipped_stress"
    result$error <- "stress fixture; set MIXEDMODELS_INCLUDE_STRESS=1 to fit"
    return(result)
  }

  fit_call <- if (is_lmm) {
    quote(lme4::lmer(stats::as.formula(formula_str), data = df,
                    REML = identical(est, "REML"),
                    control = lme4::lmerControl(calc.derivs = FALSE)))
  } else if (is_gaussian) {
    NULL  # unknown estimator for Gaussian
  } else {
    fam_fn <- switch(tolower(family),
                     bernoulli = stats::binomial(link = tolower(link)),
                     binomial = stats::binomial(link = tolower(link)),
                     poisson  = stats::poisson(link = tolower(link)),
                     gamma    = stats::Gamma(link = tolower(link)),
                     NULL)
    if (is.null(fam_fn)) NULL else {
      formula_str <- adapt_formula_for_glmer(formula_str, weights_col)
      n_agq <- if (identical(est, "AGQ")) 7L else 1L
      quote(lme4::glmer(stats::as.formula(formula_str), data = df, family = fam_fn,
                       nAGQ = n_agq,
                       control = lme4::glmerControl(calc.derivs = FALSE)))
    }
  }

  if (is.null(fit_call)) {
    result$status <- "not_implemented"
    result$error <- sprintf("no R driver for family=%s link=%s estimator=%s", family, link, est)
    return(result)
  }

  warnings_seen <- character()
  capture_warning <- function(w) {
    warnings_seen <<- c(warnings_seen, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
  capture_message <- function(m) {
    msg <- sub("\\n$", "", conditionMessage(m))
    if (nzchar(msg)) warnings_seen <<- c(warnings_seen, msg)
    invokeRestart("muffleMessage")
  }
  fit <- tryCatch(
    withCallingHandlers({
      t0 <- proc.time()[["elapsed"]]
      f <- eval(fit_call)
      cold <- (proc.time()[["elapsed"]] - t0) * 1000
      times <- numeric(TIMING_REPEATS); times[1] <- cold
      for (k in seq_len(TIMING_REPEATS - 1L)) {
        tk <- proc.time()[["elapsed"]]
        f <- eval(fit_call)
        times[k + 1L] <- (proc.time()[["elapsed"]] - tk) * 1000
      }
      list(model = f, cold_ms = cold, min_ms = min(times))
    },
    warning = capture_warning,
    message = capture_message),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    result$status <- "error"
    result$error <- conditionMessage(fit)
    return(result)
  }

  m <- fit$model
  vc <- as.data.frame(lme4::VarCorr(m))
  result$status <- "ok"
  # Wrap vectors in I() so jsonlite::toJSON(auto_unbox = TRUE) keeps length-1
  # arrays as arrays — the Rust side expects Vec<f64> / Vec<String>, not scalars.
  result$beta <- I(as.numeric(lme4::fixef(m)))
  result$coef_names <- I(names(lme4::fixef(m)))
  if (is_lmm) {
    result$sigma <- attr(lme4::VarCorr(m), "sc")
    result$objective <- if (identical(est, "REML")) {
      as.numeric(REMLcrit(m))
    } else {
      as.numeric(deviance(m, REML = FALSE))
    }
    result$objective_definition <- if (identical(est, "REML")) "restricted_deviance" else "deviance"
    result$response_constants <- "not_applicable"
  } else {
    # GLMM: dispersion is fixed at 1; objective is -2 * logLik (Laplace).
    result$sigma <- attr(lme4::VarCorr(m), "sc")
    result$objective <- as.numeric(-2 * stats::logLik(m))
    result$objective_definition <- "minus_two_loglik"
    result$response_constants <- "included"
  }
  result$theta <- I(as.numeric(getME(m, "theta")))
  result$loglik <- as.numeric(stats::logLik(m))
  result$aic <- as.numeric(stats::AIC(m))
  result$bic <- as.numeric(stats::BIC(m))
  optinfo <- m@optinfo
  result$optimizer_backend <- "lme4"
  if (!is.null(optinfo$optimizer)) {
    result$optimizer <- paste(as.character(optinfo$optimizer), collapse = ",")
  }
  if (!is.null(optinfo$conv$opt)) {
    result$optimizer_return_code <- as.character(optinfo$conv$opt)
  } else if (!is.null(optinfo$conv$lme4$code)) {
    result$optimizer_return_code <- as.character(optinfo$conv$lme4$code)
  }
  if (!is.null(optinfo$feval)) {
    result$optimizer_fevals <- as.integer(optinfo$feval)
  }
  result$is_singular <- isTRUE(lme4::isSingular(m, tol = 1e-4))
  result$fit_time_ms <- fit$cold_ms
  result$fit_time_ms_min <- fit$min_ms
  result$fit_time_ms_repeats <- TIMING_REPEATS
  result$warnings <- I(unique(warnings_seen))
  result
}

cat(sprintf("running %d fits...\n", length(manifest$dataset) %||% nrow(manifest)))
n <- if (is.data.frame(manifest)) nrow(manifest) else length(manifest)
results <- vector("list", n)
for (i in seq_len(n)) {
  entry <- if (is.data.frame(manifest)) as.list(manifest[i, , drop = FALSE]) else manifest[[i]]
  cat(sprintf("[%2d/%d] %s :: %s [%s] ... ", i, n,
              entry$dataset, entry$formula, entry$estimator))
  r <- fit_one(entry)
  if (identical(r$status, "ok")) {
    cat(sprintf("obj=%.4f  σ=%.4f  cold=%.1fms  min=%.1fms\n",
                r$objective, r$sigma, r$fit_time_ms, r$fit_time_ms_min))
  } else {
    cat(sprintf("%s%s\n", r$status,
                if (is.na(r$error) || identical(r$error, "")) "" else paste0(": ", r$error)))
  }
  results[[i]] <- r
}

out <- list(
  tool = paste0("lme4 ", as.character(packageVersion("lme4"))),
  R_version = paste(R.version$major, R.version$minor, sep = "."),
  results = results
)
out_path <- file.path(repo_root, "comparison", "lme4_results.json")
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null", null = "null"),
           con = out_path)
cat(sprintf("\nwrote %d results to %s\n", n, out_path))
