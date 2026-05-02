# Shared helpers for the dataset-pinning scripts.
# Sourced by:
#   scripts/dump_datasets.R               (Tier 1+2: lme4/nlme upstream)
#   scripts/dump_synthesized_datasets.R   (Tier 3 vendored: tungara, singular,
#                                          station_season_duration, nested_constant_response)
#
# Responsibilities:
#   - locate the repo root,
#   - fit a single recommended formula via lme4,
#   - extract a [fits.expected]-shaped record,
#   - emit auto-managed sibling expected.toml + provenance.toml.

suppressPackageStartupMessages({
  library(lme4)
  library(RcppTOML)
})

# ---- Repo location ------------------------------------------------------

find_repo_root <- function(start = getwd()) {
  d <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(d, "Cargo.toml"))) return(d)
    parent <- dirname(d)
    if (parent == d) stop("could not find Cargo.toml ancestor; run from repo or a subdir.")
    d <- parent
  }
}

# Read a vendored dataset back into an R data.frame, restoring categorical
# columns with the canonical level order from meta.toml.
read_dataset_dataframe <- function(repo_root, name) {
  meta_path <- file.path(repo_root, "datasets", name, "meta.toml")
  csv_path  <- file.path(repo_root, "datasets", name, "data.csv")
  meta <- RcppTOML::parseTOML(meta_path)
  df <- utils::read.csv(csv_path, stringsAsFactors = FALSE, na.strings = "")
  for (col in meta$columns) {
    if (identical(col$type, "categorical")) {
      lvls <- if (!is.null(col$levels)) col$levels else unique(df[[col$name]])
      df[[col$name]] <- factor(df[[col$name]], levels = lvls)
    }
  }
  df
}

# ---- Fitting ------------------------------------------------------------

# Fit one recommended formula. Returns the lme4 model or NULL with a
# warning. Supports lmer (Gaussian/Identity) and glmer (Binomial/Logit,
# Poisson/Log, Gamma/Log).
fit_one <- function(formula_text, family_text, link_text, estimator_text,
                    weights_text, data) {
  form <- stats::as.formula(formula_text)
  fam_lower <- tolower(family_text)
  link_lower <- tolower(link_text)
  est_lower <- tolower(estimator_text)
  weights_vec <- if (!is.null(weights_text) && nzchar(weights_text)) {
    data[[weights_text]]
  } else NULL

  if (fam_lower == "gaussian" && link_lower == "identity") {
    reml <- est_lower == "reml"
    fit <- tryCatch(
      lme4::lmer(form, data = data, REML = reml,
                 weights = weights_vec,
                 control = lme4::lmerControl(check.conv.singular = "ignore")),
      error = function(e) {
        warning(sprintf("lmer failed for `%s`: %s", formula_text, conditionMessage(e)))
        NULL
      }
    )
    return(fit)
  }

  fam_obj <- switch(fam_lower,
    binomial = stats::binomial(link = link_lower),
    poisson  = stats::poisson(link = link_lower),
    gamma    = stats::Gamma(link = link_lower),
    NULL
  )
  if (is.null(fam_obj)) {
    warning(sprintf("unsupported family `%s` for `%s` — skipping",
                    family_text, formula_text))
    return(NULL)
  }
  nAGQ <- if (est_lower == "agq") 9L else 1L
  fit <- tryCatch(
    lme4::glmer(form, data = data, family = fam_obj, nAGQ = nAGQ,
                weights = weights_vec,
                control = lme4::glmerControl(check.conv.singular = "ignore")),
    error = function(e) {
      warning(sprintf("glmer failed for `%s`: %s", formula_text, conditionMessage(e)))
      NULL
    }
  )
  fit
}

extract_expected <- function(fit, family_text) {
  if (is.null(fit)) return(NULL)
  beta <- as.numeric(unname(lme4::fixef(fit)))
  vc <- as.data.frame(lme4::VarCorr(fit))
  re_rows  <- vc[is.na(vc$var2), , drop = FALSE]
  re_sigmas <- as.numeric(re_rows$sdcor)
  cor_rows <- vc[!is.na(vc$var2), , drop = FALSE]
  re_corr  <- if (nrow(cor_rows) == 1) as.numeric(cor_rows$sdcor) else NULL
  theta    <- as.numeric(lme4::getME(fit, "theta"))
  is_gauss <- tolower(family_text) == "gaussian"
  sigma    <- if (is_gauss) as.numeric(stats::sigma(fit)) else NULL
  ll <- as.numeric(stats::logLik(fit))
  is_singular <- tryCatch(isTRUE(lme4::isSingular(fit, tol = 1e-4)),
                          error = function(e) FALSE)
  list(
    beta = beta,
    sigma = sigma,
    re_sigmas = re_sigmas,
    re_corr = re_corr,
    theta = theta,
    objective = -2 * ll,
    is_singular = is_singular
  )
}

# ---- TOML emission ------------------------------------------------------

toml_str <- function(s) {
  s <- gsub("\\\\", "\\\\\\\\", s)
  s <- gsub("\"", "\\\\\"", s)
  paste0("\"", s, "\"")
}

toml_num_array <- function(v) {
  if (length(v) == 0) return("[]")
  paste0("[", paste(sprintf("%.17g", v), collapse = ", "), "]")
}

format_expected_block <- function(exp, formula_text, estimator_text) {
  out <- c(
    "[[expected]]",
    sprintf("formula = %s", toml_str(formula_text)),
    sprintf("estimator = %s", toml_str(estimator_text)),
    sprintf("beta = %s", toml_num_array(exp$beta))
  )
  if (!is.null(exp$sigma))     out <- c(out, sprintf("sigma = %.17g", exp$sigma))
  if (length(exp$re_sigmas) > 0) out <- c(out, sprintf("re_sigmas = %s", toml_num_array(exp$re_sigmas)))
  if (!is.null(exp$re_corr))   out <- c(out, sprintf("re_corr = %.17g", exp$re_corr))
  if (length(exp$theta) > 0)   out <- c(out, sprintf("theta = %s", toml_num_array(exp$theta)))
  out <- c(out,
    sprintf("objective = %.17g", exp$objective),
    sprintf("is_singular = %s", tolower(as.character(exp$is_singular)))
  )
  paste(out, collapse = "\n")
}

write_expected_toml <- function(repo_root, name, entries, regenerator) {
  if (length(entries) == 0) return(invisible(NULL))
  path <- file.path(repo_root, "datasets", name, "expected.toml")
  header <- paste(
    sprintf("# Auto-generated by %s — do not hand-edit.", regenerator),
    "# Pinned reference fits for entries that meta.toml leaves empty.",
    "# Loader: src/datasets/mod.rs::load_meta merges these into Meta.fits[i].expected.",
    sep = "\n"
  )
  body <- paste(entries, collapse = "\n\n")
  writeLines(c(header, "", body), path)
  cat(sprintf("wrote %s (%d entries)\n", path, length(entries)))
}

write_provenance_toml <- function(repo_root, name, regenerator,
                                  optimizer = "bobyqa", notes = NULL) {
  path <- file.path(repo_root, "datasets", name, "provenance.toml")
  lme4_ver <- as.character(utils::packageVersion("lme4"))
  r_ver <- paste(R.Version()[c("major", "minor")], collapse = ".")
  host <- paste(Sys.info()[c("sysname", "machine")], collapse = "/")
  date <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  lines <- c(
    sprintf("# Auto-generated by %s — do not hand-edit.", regenerator),
    "# Regeneration provenance for the auto-managed sibling expected.toml.",
    "",
    sprintf("tool = %s", toml_str(sprintf("lme4 %s", lme4_ver))),
    sprintf("tool_name = \"lme4\""),
    sprintf("tool_version = %s", toml_str(lme4_ver)),
    sprintf("r_version = %s", toml_str(r_ver)),
    sprintf("date = %s", toml_str(date)),
    sprintf("host = %s", toml_str(host)),
    sprintf("regenerator = %s", toml_str(regenerator)),
    sprintf("optimizer = %s", toml_str(optimizer))
  )
  if (!is.null(notes) && nzchar(notes)) {
    lines <- c(lines, sprintf("notes = %s", toml_str(notes)))
  }
  writeLines(lines, path)
  cat(sprintf("wrote %s\n", path))
}

# ---- Driver -------------------------------------------------------------

fit_inline_already_set <- function(fit_entry) {
  !is.null(fit_entry$expected) && length(fit_entry$expected) > 0
}

# Refit each [[fits]] entry in meta.toml whose inline [fits.expected] is
# absent, and write expected.toml + provenance.toml.
pin_dataset <- function(repo_root, df, name, regenerator, notes = NULL) {
  meta_path <- file.path(repo_root, "datasets", name, "meta.toml")
  if (!file.exists(meta_path)) {
    cat(sprintf("[skip pin] %s: no meta.toml\n", name))
    return(invisible(NULL))
  }
  meta <- RcppTOML::parseTOML(meta_path)
  fits <- meta$fits
  if (is.null(fits) || length(fits) == 0) {
    cat(sprintf("[skip pin] %s: no [[fits]] entries\n", name))
    write_provenance_toml(repo_root, name, regenerator, notes = notes)
    return(invisible(NULL))
  }
  entries <- character(0)
  for (fit_entry in fits) {
    if (fit_inline_already_set(fit_entry)) {
      cat(sprintf("[keep inline] %s :: %s (%s)\n",
                  name, fit_entry$formula, fit_entry$estimator))
      next
    }
    fit <- fit_one(
      formula_text   = fit_entry$formula,
      family_text    = fit_entry$family,
      link_text      = fit_entry$link,
      estimator_text = fit_entry$estimator,
      weights_text   = fit_entry$weights,
      data           = df
    )
    if (is.null(fit)) next
    exp <- extract_expected(fit, fit_entry$family)
    if (is.null(exp)) next
    entries <- c(entries, format_expected_block(exp, fit_entry$formula, fit_entry$estimator))
    cat(sprintf("[pin] %s :: %s (%s)\n",
                name, fit_entry$formula, fit_entry$estimator))
  }
  write_expected_toml(repo_root, name, entries, regenerator)
  write_provenance_toml(repo_root, name, regenerator, notes = notes)
}
