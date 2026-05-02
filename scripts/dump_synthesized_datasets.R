#!/usr/bin/env Rscript
# Pin reference fits for the four "non-package-upstream" datasets:
#   - tungara_single_caller   (Dryad doi:10.5061/dryad.3n5tb2rrz)
#   - singular                (Cross Validated / GitHub mirror)
#   - station_season_duration (Cross Validated forum example)
#   - nested_constant_response (Synthetic lme4 issue 489 repro)
#
# These do not have an upstream R package as their canonical source, so
# the CSVs in datasets/<name>/data.csv are themselves the canonical data
# and are NOT regenerated here. This script only refits each recommended
# formula and writes auto-managed expected.toml + provenance.toml.
#
# Usage:
#   Rscript scripts/dump_synthesized_datasets.R
#
# To regenerate the underlying CSVs themselves, follow the source notes
# in datasets/REGISTRY.md per dataset. Doing so is intentionally a manual
# step because the upstream sources are forum posts / Dryad archives /
# issue-tracker repros without a canonical R package.

# Locate this script's directory regardless of caller cwd.
script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- regmatches(args, regexpr("(?<=^--file=).+", args, perl = TRUE))
  if (length(m) > 0) return(dirname(normalizePath(m[1], mustWork = TRUE)))
  normalizePath(".", mustWork = TRUE)
})()
source(file.path(script_dir, "_pin_helpers.R"))

repo_root <- find_repo_root()
REGENERATOR <- "scripts/dump_synthesized_datasets.R"

# Each dataset: a free-form note recording the upstream source + any
# relevant generation hint (seed, transform). Echoed into provenance.toml
# so a future contributor can trace the data back to its origin without
# spelunking REGISTRY.md.
DATASETS <- list(
  list(
    name  = "tungara_single_caller",
    notes = "Dryad doi:10.5061/dryad.3n5tb2rrz; binomial GLMM cell-level random-slope fixture (lme4 GH#720 fallback)."
  ),
  list(
    name  = "singular",
    notes = "Cross Validated / GitHub mirror; 8-D random coefficient covariance, maximal model is structurally singular."
  ),
  list(
    name  = "station_season_duration",
    notes = "Cross Validated forum example; balanced site x season x duration cells, weakly-identified RE."
  ),
  list(
    name  = "nested_constant_response",
    notes = "Synthetic lme4 issue 489 repro; nested lower-level random intercept with constant-response duplicates."
  ),
  list(
    name  = "gopherdat2",
    notes = "Bolker mixedmodels-misc/data/gopherdat2.RData; canonical Poisson-GLMM-with-offset fixture (Ozgul et al. 2009)."
  ),
  list(
    name  = "culcitalogreg",
    notes = "Bolker mixedmodels-misc/data/culcita.RData; small-N binomial GLMM block design (Schmitt et al. 2009)."
  ),
  list(
    name  = "contraception",
    notes = "MixedModels.jl :contra; 1934 obs Bangladesh BDHS contraceptive use; CSV authored from Julia with use Y/N → 0/1."
  ),
  list(
    name  = "insteval",
    notes = "lme4::InstEval; 73421 obs course evaluations (2972 students × 1128 lecturers); large-N crossed-RE benchmark fixture."
  ),
  list(
    name  = "arabidopsis",
    notes = "lme4::Arabidopsis (Banta et al. 2010); 625 obs Poisson GLMM with severe overdispersion (var/mean ≈ 56)."
  )
)

for (entry in DATASETS) {
  name <- entry$name
  meta_path <- file.path(repo_root, "datasets", name, "meta.toml")
  csv_path  <- file.path(repo_root, "datasets", name, "data.csv")
  if (!file.exists(meta_path) || !file.exists(csv_path)) {
    cat(sprintf("[skip] %s: missing meta.toml or data.csv\n", name))
    next
  }
  df <- read_dataset_dataframe(repo_root, name)
  pin_dataset(repo_root, df, name, REGENERATOR, notes = entry$notes)
}

cat("done.\n")
