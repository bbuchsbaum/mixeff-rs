#!/usr/bin/env Rscript
# Dump reference mixed-models datasets to CSV under datasets/<name>/data.csv.
#
# Tier-1 (always): lme4 — sleepstudy, Dyestuff, Dyestuff2, Pastes, Penicillin, cbpp
# Tier-2 (if --tier2): lme4 — cake, VerbAgg, grouseticks; nlme — ergoStool, Machines, Orthodont, Oats
#
# Usage:
#   Rscript scripts/dump_datasets.R           # tier 1 only
#   Rscript scripts/dump_datasets.R --tier2   # tier 1 + tier 2

suppressPackageStartupMessages({
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
include_tier2 <- "--tier2" %in% args

# Locate repo root by walking up until we find Cargo.toml.
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

dump_one <- function(df, name) {
  outdir <- file.path(repo_root, "datasets", name)
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  # canonicalize: write factors as their character labels; row order preserved.
  out <- as.data.frame(df)
  # Factors -> character (so CSV round-trips identically; meta.toml records the canonical level order).
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
  # also dump factor levels for cross-checking with meta.toml
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

# ---- Tier 1 ----
data("sleepstudy", "Dyestuff", "Dyestuff2", "Pastes", "Penicillin", "cbpp", package = "lme4")
dump_one(sleepstudy, "sleepstudy")
dump_one(Dyestuff,   "dyestuff")
dump_one(Dyestuff2,  "dyestuff2")
dump_one(Pastes,     "pastes")
dump_one(Penicillin, "penicillin")
dump_one(cbpp,       "cbpp")

# ---- Tier 2 ----
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
