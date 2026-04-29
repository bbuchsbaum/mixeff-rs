#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
})

find_repo_root <- function(start = getwd()) {
  d <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(d, "Cargo.toml"))) return(d)
    parent <- dirname(d)
    if (identical(parent, d)) stop("could not find Cargo.toml ancestor")
    d <- parent
  }
}

flag <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  hit <- grep(paste0("^", prefix), commandArgs(trailingOnly = TRUE), value = TRUE)
  if (!length(hit)) return(default)
  sub(prefix, "", hit[[1]], fixed = TRUE)
}

parse_scalar <- function(lines, key) {
  hit <- grep(paste0("^", key, " *="), lines, value = TRUE)
  if (!length(hit)) stop("missing key ", key)
  value <- sub("^[^=]+=", "", hit[[1]])
  value <- trimws(sub("#.*$", "", value))
  if (grepl('^"', value)) return(sub('"$', "", sub('^"', "", value)))
  if (grepl("^[0-9.]+$", value)) return(as.numeric(value))
  value
}

parse_json_value <- function(lines, key) {
  hit <- grep(paste0("^", key, " *="), lines, value = TRUE)
  if (!length(hit)) stop("missing key ", key)
  value <- trimws(sub("^[^=]+=", "", hit[[1]]))
  jsonlite::fromJSON(value)
}

read_spec <- function(path) {
  lines <- readLines(path, warn = FALSE)
  list(
    path = path,
    contract_version = parse_scalar(lines, "contract_version"),
    name = parse_scalar(lines, "name"),
    stratum = parse_scalar(lines, "stratum"),
    group_sizes = as.integer(parse_json_value(lines, "group_sizes")),
    fe_truth = as.numeric(parse_json_value(lines, "fe_truth")),
    re_cov_truth = as.matrix(parse_json_value(lines, "re_cov_truth")),
    family = parse_scalar(lines, "family"),
    link = parse_scalar(lines, "link"),
    seed = as.integer(parse_scalar(lines, "seed")),
    residual_sd = as.numeric(parse_scalar(lines, "residual_sd"))
  )
}

sqrt_psd <- function(sigma) {
  eig <- eigen(sigma, symmetric = TRUE)
  eig$vectors %*% diag(sqrt(pmax(eig$values, 0)), nrow = nrow(sigma)) %*% t(eig$vectors)
}

deterministic_data <- function(spec) {
  n_pred <- length(spec$fe_truth) - 1L
  q <- nrow(spec$re_cov_truth)
  n_slopes <- max(0L, q - 1L)
  sigma_sqrt <- sqrt_psd(spec$re_cov_truth)

  y <- numeric()
  g <- character()
  predictors <- replicate(n_pred, numeric(), simplify = FALSE)
  row_index <- 0L

  for (group_index in seq_along(spec$group_sizes)) {
    z <- c(
      sin(spec$seed + group_index * 1.7),
      cos(spec$seed * 0.5 + group_index * 2.3),
      sin(spec$seed * 0.25 + group_index * 3.1)
    )[seq_len(q)]
    u <- as.numeric(sigma_sqrt %*% z)
    n_g <- spec$group_sizes[[group_index]]
    for (within in seq_len(n_g)) {
      row_index <- row_index + 1L
      centered <- if (n_g <= 1L) 0 else ((within - 1L) - (n_g - 1L) / 2) / (n_g - 1L)
      x <- if (n_pred == 0L) numeric() else vapply(seq_len(n_pred), function(j) {
        centered ^ j + 0.07 * sin((row_index + j) * 0.61)
      }, numeric(1))
      eta <- spec$fe_truth[[1L]]
      if (n_pred > 0L) eta <- eta + sum(spec$fe_truth[-1L] * x)
      if (q >= 1L) eta <- eta + u[[1L]]
      if (n_slopes > 0L) eta <- eta + sum(u[seq_len(n_slopes) + 1L] * x[seq_len(n_slopes)])
      noise_scale <- if (identical(spec$stratum, "reduced-rank")) 0 else 0.1
      y <- c(y, eta + spec$residual_sd * noise_scale * sin(spec$seed + row_index * 0.73))
      g <- c(g, sprintf("g%03d", group_index))
      for (j in seq_len(n_pred)) predictors[[j]] <- c(predictors[[j]], x[[j]])
    }
  }

  out <- data.frame(y = y, g = factor(g))
  for (j in seq_len(n_pred)) out[[paste0("x", j)]] <- predictors[[j]]
  out
}

fit_lme4 <- function(spec) {
  df <- deterministic_data(spec)
  n_pred <- length(spec$fe_truth) - 1L
  q <- nrow(spec$re_cov_truth)
  fe <- if (n_pred == 0L) "1" else paste(c("1", paste0("x", seq_len(n_pred))), collapse = " + ")
  re <- if (q <= 1L) "(1 | g)" else paste0("(1 + ", paste0("x", seq_len(q - 1L), collapse = " + "), " | g)")
  formula_str <- paste("y ~", fe, "+", re)

  warnings_seen <- character()
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(
    withCallingHandlers(
      lme4::lmer(
        stats::as.formula(formula_str),
        data = df,
        REML = TRUE,
        control = lme4::lmerControl(calc.derivs = FALSE)
      ),
      warning = function(w) {
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) {
        msg <- sub("\\n$", "", conditionMessage(m))
        if (nzchar(msg)) warnings_seen <<- c(warnings_seen, msg)
        invokeRestart("muffleMessage")
      }
    ),
    error = function(e) e
  )
  runtime_ms <- (proc.time()[["elapsed"]] - t0) * 1000

  if (inherits(fit, "error")) {
    return(list(
      schema_version = "1.0.0", fixture = spec$name, stratum = spec$stratum,
      engine = "lme4::lmer", source = "scripts/parity_pathologies.R",
      status = "error", error = conditionMessage(fit), warnings = I(warnings_seen),
      converged = FALSE, objective = NA_real_, theta = I(numeric()),
      beta = I(numeric()), sigma = NA_real_, loglik = NA_real_, runtime_ms = runtime_ms
    ))
  }

  list(
    schema_version = "1.0.0",
    fixture = spec$name,
    stratum = spec$stratum,
    engine = "lme4::lmer",
    version = paste0("lme4 ", as.character(utils::packageVersion("lme4")), "; R ", getRversion()),
    source = "scripts/parity_pathologies.R",
    status = "ok",
    warnings = I(warnings_seen),
    converged = is.null(fit@optinfo$conv$lme4$messages),
    objective = as.numeric(lme4::REMLcrit(fit)),
    theta = I(as.numeric(lme4::getME(fit, "theta"))),
    beta = I(as.numeric(lme4::fixef(fit))),
    sigma = as.numeric(attr(lme4::VarCorr(fit), "sc")),
    loglik = as.numeric(stats::logLik(fit)),
    runtime_ms = runtime_ms,
    singular = lme4::isSingular(fit)
  )
}

main <- function() {
  repo <- find_repo_root()
  fixture <- flag("fixture", file.path(repo, "tests/fixtures/pathology_corpus/easy.toml"))
  spec <- read_spec(if (grepl("^/", fixture)) fixture else file.path(repo, fixture))
  result <- fit_lme4(spec)
  json <- jsonlite::toJSON(result, auto_unbox = TRUE, pretty = TRUE, na = "null", digits = 17)
  out <- flag("out", NULL)
  if (is.null(out)) {
    cat(json, "\n")
  } else {
    writeLines(json, if (grepl("^/", out)) out else file.path(repo, out))
  }
}

main()
