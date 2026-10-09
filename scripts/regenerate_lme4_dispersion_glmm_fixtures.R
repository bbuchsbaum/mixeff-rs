# Regenerate the lme4 2.1-0 references for estimated-dispersion GLMMs
# (Gamma, inverse Gaussian, Gaussian with a non-identity link).
#
# lme4 2.1-0 profiles the dispersion phi inside PIRLS
# (glmerControl(disp_method = "moment", disp_dof_correction = TRUE)), fixes
# the "+2" logLik constant and reports sigma() = sqrt(phi) with the
# (n - rank([X, Z])) correction. These fixtures pin the engine to that
# behaviour, and to disp_method = "old/buggy" for the Legacy control.
#
# Usage (from the repository root):
#   R_LIBS=/path/to/lme4-2.1 Rscript scripts/regenerate_lme4_dispersion_glmm_fixtures.R
# Writes tests/fixtures/parity/dispersion_glmm_lme4_2_1.json and its
# provenance file, and prints the lme4 2.1 rows used by
# gamma_glmm_engines.json and gamma_glmm_lme4_643.json.

lib <- Sys.getenv("LME4_2_1_LIB", "/home/user/rlib-new")
if (dir.exists(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})
stopifnot(packageVersion("lme4") >= "2.1.0")

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
if (is.na(script_path) || !nzchar(script_path)) {
  script_path <- file.path("scripts", "regenerate_lme4_dispersion_glmm_fixtures.R")
}
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
if (!dir.exists(file.path(repo_root, "tests"))) repo_root <- normalizePath(getwd(), mustWork = TRUE)
fixture_dir <- file.path(repo_root, "tests", "fixtures", "parity")

mat_rows <- function(m) unname(lapply(seq_len(nrow(m)), function(i) unname(as.numeric(m[i, ]))))

reference <- function(fit) {
  dc <- fit@devcomp
  vc <- as.data.frame(VarCorr(fit))
  list(
    fixef = unname(fixef(fit)),
    theta = unname(getME(fit, "theta")),
    sigma = sigma(fit),
    phi = fit@resp$phi(),
    q_eff = unname(dc$dims[["qEff"]]),
    logLik = as.numeric(logLik(fit)),
    deviance = deviance(fit),
    AIC = AIC(fit),
    df = attr(logLik(fit), "df"),
    varcorr_sd = vc$sdcor[is.na(vc$var2) & vc$grp != "Residual"],
    vcov = mat_rows(as.matrix(vcov(fit))),
    vcov_rx = mat_rows(as.matrix(suppressWarnings(vcov(fit, use.hessian = FALSE)))),
    unscaled_rx = mat_rows(as.matrix(fit@pp$unsc())),
    optimizer_messages = unname(as.character(unlist(fit@optinfo$conv$lme4$messages)))
  )
}

# Shared positive-response design (tests/fixtures/parity/dispersion_glmm_lme4_scale.csv).
d <- read.csv(file.path(fixture_dir, "dispersion_glmm_lme4_scale.csv"))
d$g <- factor(d$g)

families <- list(
  gamma = Gamma(link = "log"),
  inverse_gaussian = inverse.gaussian(link = "log"),
  gaussian_log = gaussian(link = "log")
)
cases <- list()
for (name in names(families)) {
  for (agq in c(0L, 1L, 5L)) {
    fit <- glmer(y ~ x + (1 | g), data = d, family = families[[name]], nAGQ = agq)
    cases[[length(cases) + 1L]] <- c(
      list(family = name, n_agq = agq, disp_method = "moment", disp_dof_correction = TRUE),
      reference(fit)
    )
  }
}
for (agq in c(0L, 1L)) {
  fit <- glmer(y ~ x + (1 | g), data = d, family = Gamma(link = "log"), nAGQ = agq,
               control = glmerControl(disp_method = "old/buggy", disp_dof_correction = FALSE))
  cases[[length(cases) + 1L]] <- c(
    list(family = "gamma", n_agq = agq, disp_method = "old/buggy", disp_dof_correction = FALSE),
    reference(fit)
  )
  fit <- glmer(y ~ x + (1 | g), data = d, family = Gamma(link = "log"), nAGQ = agq,
               control = glmerControl(disp_dof_correction = FALSE))
  cases[[length(cases) + 1L]] <- c(
    list(family = "gamma", n_agq = agq, disp_method = "moment", disp_dof_correction = FALSE),
    reference(fit)
  )
}

script_text <- paste(readLines(script_path), collapse = "\n")
out <- list(
  schema_version = "1.0.0",
  source = "lme4::glmer y ~ x + (1 | g) on dispersion_glmm_lme4_scale.csv, default glmerControl() unless noted",
  formula = "y ~ x + (1 | g)",
  data = "dispersion_glmm_lme4_scale.csv",
  lme4_version = as.character(packageVersion("lme4")),
  cases = cases
)
write_json(out, file.path(fixture_dir, "dispersion_glmm_lme4_2_1.json"),
           auto_unbox = TRUE, digits = NA, pretty = TRUE)
provenance <- list(
  schema_version = "1.0",
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  regenerator = "scripts/regenerate_lme4_dispersion_glmm_fixtures.R",
  reference_engine = paste("lme4", packageVersion("lme4")),
  r_version = R.version.string,
  package_versions = list(
    lme4 = as.character(packageVersion("lme4")),
    Matrix = as.character(packageVersion("Matrix")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  notes = paste(
    "lme4 2.1-0 estimated-dispersion GLMM references (disp_method = 'moment',",
    "disp_dof_correction = TRUE by default; 'old/buggy' rows pin the Legacy control).",
    "vcov is lme4's default vcov(); for nAGQ = 0 that is sigma^2 * unsc(), which double-counts",
    "phi (unsc() is already phi-weighted); unscaled_rx is unsc() itself."
  ),
  script = script_text
)
write_json(provenance, file.path(fixture_dir, "dispersion_glmm_lme4_2_1.provenance.json"),
           auto_unbox = TRUE, digits = NA, pretty = TRUE)

# lme4 2.1 rows for the other Gamma fixtures (pasted into their JSON).
toy <- local({
  group_effects <- c(-0.25, 0.1, 0.3, -0.15)
  rows <- list()
  for (g in 0:3) for (obs in 0:4) {
    xv <- obs - 2
    eta <- 1.2 + 0.25 * xv + group_effects[g + 1]
    wiggle <- 1 + 0.06 * ((g + obs) %% 3)
    rows[[length(rows) + 1]] <- data.frame(y = exp(eta) * wiggle, x = xv, group = sprintf("g%d", g + 1))
  }
  do.call(rbind, rows)
})
toy_fit <- glmer(y ~ 1 + x + (1 | group), data = toy, family = Gamma(link = "log"), nAGQ = 0)
cat("gamma_glmm_engines (nAGQ = 0):\n")
print(toJSON(reference(toy_fit)[c("fixef", "theta", "sigma", "phi", "logLik")], auto_unbox = TRUE, digits = NA))

set.seed(123)
N <- 1000
y643 <- rgamma(N, 3, 30)
grp643 <- sample(1:20, N, replace = TRUE)
for (agq in 0:1) {
  fit643 <- glmer(y ~ 1 + (1 | grp), data = data.frame(y = y643, grp = factor(grp643)),
                  family = Gamma(link = "log"), nAGQ = agq)
  cat(sprintf("gamma_glmm_lme4_643 (nAGQ = %d):\n", agq))
  print(toJSON(reference(fit643)[c("fixef", "theta", "sigma", "phi", "logLik")], auto_unbox = TRUE, digits = NA))
}
