#!/usr/bin/env Rscript
# Dump reference mixed-models datasets to CSV under datasets/<name>/data.csv,
# fit each recommended formula with lme4/nlme, and emit auto-managed
# `expected.toml` + `provenance.toml` siblings.
#
# Tier-1 (always): lme4 — sleepstudy, Dyestuff, Dyestuff2, Pastes, Penicillin, cbpp
# Tier-2 (if --tier2): lme4 — cake, VerbAgg, grouseticks; nlme — ergoStool, Machines, Orthodont, Oats, Rail
#
# Usage:
#   Rscript scripts/dump_datasets.R                     # tier 1 only
#   Rscript scripts/dump_datasets.R --tier2             # tier 1 + tier 2
#   Rscript scripts/dump_datasets.R --pin-only          # skip CSV dump; just refit + repin
#   Rscript scripts/dump_datasets.R --tier2 --pin-only  # tier 1+2, pin-only
#
# Idempotent: re-running on a clean tree produces a no-op git diff modulo
# `[provenance].date` (and float noise at numeric tolerance). Hand-authored
# `[fits.expected]` blocks in meta.toml always win — sibling expected.toml
# only fills slots that meta.toml leaves empty.
#
# Pinning helpers live in scripts/_pin_helpers.R and are shared with
# scripts/dump_synthesized_datasets.R.

# Locate this script's directory regardless of caller cwd.
script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- regmatches(args, regexpr("(?<=^--file=).+", args, perl = TRUE))
  if (length(m) > 0) return(dirname(normalizePath(m[1], mustWork = TRUE)))
  normalizePath(".", mustWork = TRUE)
})()
source(file.path(script_dir, "_pin_helpers.R"))

args <- commandArgs(trailingOnly = TRUE)
include_tier2 <- "--tier2" %in% args
pin_only <- "--pin-only" %in% args

repo_root <- find_repo_root()
REGENERATOR <- "scripts/dump_datasets.R"

# ---- CSV dump ----------------------------------------------------------

dump_csv <- function(df, name) {
  outdir <- file.path(repo_root, "datasets", name)
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  out <- as.data.frame(df)
  for (nm in names(out)) {
    if (is.factor(out[[nm]])) out[[nm]] <- as.character(out[[nm]])
  }
  csv_path <- file.path(outdir, "data.csv")
  utils::write.table(
    out,
    csv_path,
    sep = ",",
    quote = TRUE,
    row.names = FALSE,
    col.names = TRUE,
    na = "",
    fileEncoding = "UTF-8",
    eol = "\n"
  )
  lvl_path <- file.path(outdir, "_levels.txt")
  con <- file(lvl_path, "w")
  on.exit(close(con), add = TRUE)
  for (nm in names(df)) {
    if (is.factor(df[[nm]])) {
      writeLines(paste0(nm, ": ", paste(levels(df[[nm]]), collapse = ",")), con)
    }
  }
  cat(sprintf("wrote %s (%d rows, %d cols)\n", csv_path, nrow(out), ncol(out)))
}

dump_one <- function(df, name) {
  if (!pin_only) dump_csv(df, name)
  pin_dataset(repo_root, df, name, REGENERATOR)
}

# ---- Tier 1 -----------------------------------------------------------
data("sleepstudy", "Dyestuff", "Dyestuff2", "Pastes", "Penicillin", "cbpp", package = "lme4")
dump_one(sleepstudy, "sleepstudy")
dump_one(Dyestuff,   "dyestuff")
dump_one(Dyestuff2,  "dyestuff2")
dump_one(Pastes,     "pastes")
dump_one(Penicillin, "penicillin")
dump_one(cbpp,       "cbpp")

# ---- Tier 2 -----------------------------------------------------------
if (include_tier2) {
  suppressPackageStartupMessages(library(nlme))
  data("cake", "VerbAgg", "grouseticks", package = "lme4")
  data("ergoStool", "Machines", "Orthodont", "Oats", "Rail", package = "nlme")
  dump_one(cake,        "cake")
  dump_one(VerbAgg,     "verbagg")
  dump_one(grouseticks, "grouseticks")
  dump_one(ergoStool,   "ergostool")
  dump_one(Machines,    "machines")
  dump_one(Orthodont,   "orthodont")
  dump_one(Oats,        "oats")
  dump_one(Rail,        "rail")
}

cat("done.\n")
