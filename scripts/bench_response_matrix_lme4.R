#!/usr/bin/env Rscript
# lme4 companion benchmark for examples/bench_response_matrix_batch.rs.
#
# Runs repeated lmer() fits over affine response-column variants.  The
# generated columns preserve the same design and relative variance structure, so
# they measure how much work lme4 repeats when there is no response-matrix API.

suppressPackageStartupMessages({
  library(lme4)
})

parse_qs <- function() {
  raw <- Sys.getenv("MIXEDMODELS_RESPONSE_BATCH_QS", unset = "1,4,16,64")
  values <- suppressWarnings(as.integer(strsplit(raw, ",", fixed = TRUE)[[1]]))
  values <- values[is.finite(values) & values > 0L]
  if (length(values) == 0L) c(1L, 4L, 16L, 64L) else values
}

response_matrix <- function(y, q) {
  y <- as.numeric(y)
  sd_y <- max(stats::sd(y), 1.0)
  out <- vector("list", q)
  for (j in seq_len(q)) {
    col <- j - 1L
    scale <- 0.75 + 0.5 * ((col %% 17L) / 16.0)
    offset <- ((col %% 5L) - 2.0) * 0.05 * sd_y
    out[[j]] <- scale * y + offset
  }
  out
}

control <- lme4::lmerControl(calc.derivs = FALSE)

cases <- list(
  dyestuff_scalar_re = list(
    data = function() {
      data("Dyestuff", package = "lme4", envir = environment())
      Dyestuff
    },
    formula = Yield ~ 1 + (1 | Batch),
    response = "Yield"
  ),
  sleepstudy_slope = list(
    data = function() {
      data("sleepstudy", package = "lme4", envir = environment())
      sleepstudy
    },
    formula = Reaction ~ 1 + Days + (1 + Days | Subject),
    response = "Reaction"
  ),
  penicillin_crossed = list(
    data = function() {
      data("Penicillin", package = "lme4", envir = environment())
      Penicillin
    },
    formula = diameter ~ 1 + (1 | plate) + (1 | sample),
    response = "diameter"
  )
)

fit_loop <- function(dat, formula, response, responses) {
  successes <- 0L
  for (j in seq_along(responses)) {
    dat[[".response_batch_y"]] <- responses[[j]]
    f <- stats::update(formula, .response_batch_y ~ .)
    suppressMessages(suppressWarnings(
      lme4::lmer(f, data = dat, REML = TRUE, control = control)
    ))
    successes <- successes + 1L
  }
  successes
}

cat("engine,case,n,q,mode,total_ms,per_response_ms,success_count,theta_dim\n")
for (case_id in names(cases)) {
  case <- cases[[case_id]]
  dat <- case$data()
  y <- dat[[case$response]]
  theta_dim <- length(lme4::getME(suppressMessages(suppressWarnings(
    lme4::lmer(case$formula, data = dat, REML = TRUE, control = control)
  )), "theta"))
  for (q in parse_qs()) {
    responses <- response_matrix(y, q)
    start <- proc.time()[["elapsed"]]
    successes <- fit_loop(dat, case$formula, case$response, responses)
    total_ms <- (proc.time()[["elapsed"]] - start) * 1000.0
    cat(sprintf(
      "lme4,%s,%d,%d,lmer_loop,%.3f,%.6f,%d,%d\n",
      case_id,
      nrow(dat),
      q,
      total_ms,
      total_ms / q,
      successes,
      theta_dim
    ))
  }
}
