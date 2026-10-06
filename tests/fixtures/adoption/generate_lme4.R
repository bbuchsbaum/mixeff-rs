# Run from the repository root; requires lme4 and jsonlite.
root <- "tests/fixtures/adoption"
frame <- function(name) {
  j <- jsonlite::read_json(file.path(root, name), simplifyVector = TRUE)
  d <- as.data.frame(j$numeric_columns)
  for (key in names(j$categorical_values)) {
    d[[key]] <- factor(j$categorical_values[[key]], levels = j$categorical_levels[[key]])
  }
  list(data = d[j$column_order], formula = as.formula(j$formula))
}
l <- frame("pw2_nested.json")
g <- frame("osf_centered.json")
lmm <- lme4::lmer(l$formula, l$data, REML = TRUE)
glmm <- lme4::glmer(g$formula, g$data, family = binomial("logit"), nAGQ = 1)
vc <- as.data.frame(lme4::VarCorr(lmm))
reference <- list(
  R_version = R.version.string,
  lme4_version = as.character(packageVersion("lme4")),
  lmm = list(beta = unname(lme4::fixef(lmm)), sigma = sigma(lmm),
             loglik = as.numeric(logLik(lmm)), aic = AIC(lmm),
             fitted = unname(fitted(lmm)),
             std_dev = setNames(as.list(vc$sdcor), vc$grp)),
  glmm = list(beta = unname(lme4::fixef(glmm)), loglik = as.numeric(logLik(glmm)))
)
jsonlite::write_json(reference, file.path(root, "lme4_reference.json"),
                     auto_unbox = TRUE, digits = NA, pretty = TRUE)
print(reference[c("R_version", "lme4_version")])
print(reference$lmm[c("beta", "sigma", "loglik", "std_dev")])
print(reference$glmm)
