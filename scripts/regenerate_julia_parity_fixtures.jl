#!/usr/bin/env julia

using CategoricalArrays
using DataFrames
using Dates
using MixedModels
using Printf
using Statistics

struct JObj
    fields::Vector{Pair{String, Any}}
end

function flag(name, default=nothing)
    prefix = "--" * name * "="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return split(arg, "=", limit=2)[2]
    end
    return default
end

function json_string(value)
    escaped = replace(String(value), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")
    return "\"" * escaped * "\""
end

json_number(value::Integer) = string(value)
json_number(value::AbstractFloat) = isfinite(Float64(value)) ? @sprintf("%.17g", Float64(value)) : "null"

function render_json(value, indent=0)
    pad = repeat("  ", indent)
    nextpad = repeat("  ", indent + 1)
    if value isa JObj
        isempty(value.fields) && return "{}"
        lines = String["{"]
        for (index, pair) in enumerate(value.fields)
            suffix = index == length(value.fields) ? "" : ","
            push!(lines, nextpad * json_string(pair.first) * ": " * render_json(pair.second, indent + 1) * suffix)
        end
        push!(lines, pad * "}")
        return join(lines, "\n")
    elseif value isa AbstractVector
        isempty(value) && return "[]"
        if all(item -> item isa Number || item isa AbstractString || item === nothing || item isa Bool, value)
            return "[" * join(render_json.(value, indent), ", ") * "]"
        end
        lines = String["["]
        for (index, item) in enumerate(value)
            suffix = index == length(value) ? "" : ","
            push!(lines, nextpad * render_json(item, indent + 1) * suffix)
        end
        push!(lines, pad * "]")
        return join(lines, "\n")
    elseif value isa AbstractString
        return json_string(value)
    elseif value isa Integer
        return json_number(value)
    elseif value isa AbstractFloat
        return json_number(value)
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value === nothing
        return "null"
    else
        error("unsupported JSON value $(typeof(value))")
    end
end

function write_json(root, relative_path, value)
    path = joinpath(root, split(relative_path, '/')...)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, render_json(value))
        write(io, "\n")
    end
    write_provenance_sibling(path)
end

# Write a sibling `<stem>.provenance.json` next to a freshly-emitted
# parity golden, matching the schema enforced by
# `tests/fixture_hygiene::every_golden_has_provenance_sibling`.
function write_provenance_sibling(json_path)
    stem = first(splitext(basename(json_path)))
    prov_path = joinpath(dirname(json_path), stem * ".provenance.json")
    mm_ver = module_version("MixedModels")
    julia_ver = string(VERSION)
    timestamp = string(Dates.now(Dates.UTC)) * "Z"
    commit = try
        chomp(read(`git rev-parse HEAD`, String))
    catch
        "unknown"
    end
    body = """
    {
      "schema_version": "1.0",
      "generated_at": "$timestamp",
      "crate_commit": "$commit",
      "regenerator": "scripts/regenerate_julia_parity_fixtures.jl",
      "source_case": null,
      "reference_engine": "MixedModels.jl $mm_ver",
      "notes": "Julia $julia_ver"
    }
    """
    open(prov_path, "w") do io
        write(io, body)
    end
end

function rows(matrix)
    [collect(matrix[row_index, :]) for row_index in axes(matrix, 1)]
end

function matrix_terms(mats)
    [rows(matrix) for matrix in mats]
end

function module_version(name)
    for (pkgid, mod) in Base.loaded_modules
        String(pkgid.name) == name && return string(pkgversion(mod))
    end
    return "unknown"
end

function source_version(; suffix="")
    return "MixedModels.jl $(pkgversion(MixedModels))" * suffix
end

function cbpp_agq5_fixture()
    df = DataFrame(MixedModels.dataset(:cbpp))
    df.proportion = Float64.(df.incid) ./ Float64.(df.hsz)
    model = fit(
        MixedModel,
        @formula(proportion ~ 1 + period + (1 | herd)),
        df,
        Binomial();
        wts=Float64.(df.hsz),
        fast=true,
        nAGQ=5,
        progress=false,
    )
    objective_value = objective(model)
    theta = collect(getproperty(model, :theta))
    beta = collect(coef(model))
    return JObj([
        "schema_version" => "1.0.0",
        "source" => source_version(suffix=" fast=true"),
        "id" => "cbpp_agq5",
        "formula" => "proportion ~ 1 + period + (1 | herd)",
        "family" => "binomial",
        "link" => "logit",
        "n_agq" => 5,
        "nobs" => nrow(df),
        "dof" => length(theta) + length(beta),
        "theta" => theta,
        "beta" => beta,
        "objective" => objective_value,
        "deviance_laplace" => deviance(model, 1),
        "deviance_agq" => deviance(model, 5),
        "loglik" => -objective_value / 2,
    ])
end

function kb07_style_data()
    subj_effects = [-1.0, 0.5, 1.2, -0.4, -0.3]
    subj_slopes = [-0.3, 0.2, 0.1, -0.2, 0.4]
    item_effects = [-0.2, 0.4, -0.1, 0.3]
    y = Float64[]
    x = Float64[]
    subj = String[]
    item = String[]
    for s in 0:4
        for i in 0:3
            xi = Float64(i)
            row = s * 4 + i + 1
            push!(y, 20.0 + 2.0 * xi + subj_effects[s + 1] + item_effects[i + 1] +
                     subj_slopes[s + 1] * xi + ((row % 7) - 3.0) * 0.03)
            push!(x, xi)
            push!(subj, "S$(s + 1)")
            push!(item, "I$(i + 1)")
        end
    end
    return DataFrame(y=y, x=x, subj=categorical(subj), item=categorical(item))
end

function kb07_ranef_fixture()
    df = kb07_style_data()
    model = fit(
        MixedModel,
        @formula(y ~ 1 + x + (1 + x | subj) + (1 | item)),
        df;
        REML=true,
        progress=false,
    )
    return JObj([
        "schema_version" => "1.0.0",
        "source" => source_version(),
        "id" => "kb07_style_ranef",
        "formula" => "y ~ 1 + x + (1 + x | subj) + (1 | item)",
        "nobs" => nrow(df),
        "theta" => collect(getproperty(model, :theta)),
        "beta" => collect(coef(model)),
        "ranef_u" => matrix_terms(ranef(model; uscale=true)),
        "ranef_b" => matrix_terms(ranef(model)),
    ])
end

function parmap_vsize3_data()
    subj_effects = [-0.8, 0.35, 0.6, -0.15]
    y = Float64[]
    x = Float64[]
    z = Float64[]
    subj = String[]
    for subject in 0:3
        for obs in 0:4
            xv = Float64(obs) - 2.0
            zv = Float64(obs % 3) - 1.0 + Float64(subject) * 0.1
            push!(y, 3.0 + 0.5 * xv - 0.2 * zv + subj_effects[subject + 1])
            push!(x, xv)
            push!(z, zv)
            push!(subj, "S$(subject + 1)")
        end
    end
    return DataFrame(y=y, x=x, z=z, subj=categorical(subj))
end

function parmap_vsize3_fixture()
    df = parmap_vsize3_data()
    model = LinearMixedModel(@formula(y ~ 1 + x + z + (1 + x + z | subj)), df)
    julia_parmap = [(term, row, col) for (term, row, col) in model.parmap]
    rust_parmap = [
        JObj(["term" => term - 1, "row" => row - 1, "col" => col - 1])
        for (term, row, col) in julia_parmap
    ]
    return JObj([
        "schema_version" => "1.0.0",
        "source" => source_version(),
        "id" => "parmap_vsize3",
        "formula" => "y ~ 1 + x + z + (1 + x + z | subj)",
        "nobs" => nrow(df),
        "grouping" => "subj",
        "cnames" => collect(model.reterms[1].cnames),
        "linear_indices_column_major" => collect(model.reterms[1].inds) .- 1,
        "parmap_zero_based" => rust_parmap,
        "julia_parmap_one_based" => [collect(item) for item in julia_parmap],
    ])
end

function rank_deficient_data()
    n = 24
    x = [Float64(i % 4) for i in 0:(n - 1)]
    x2 = 2.0 .* x
    group_effects = [-1.2, 0.8, 0.3, -0.4, 1.1, -0.6]
    y = Float64[]
    g = String[]
    for i in 0:(n - 1)
        group = div(i, 4)
        push!(y, 10.0 + 1.5 * x[i + 1] + group_effects[group + 1] + ((i + 1) % 5) * 0.07 - 0.14)
        push!(g, "g$(group + 1)")
    end
    return DataFrame(y=y, x=x, x2=x2, g=categorical(g))
end

function rank_deficient_metrics_fixture()
    df = rank_deficient_data()
    ml = fit(MixedModel, @formula(y ~ 1 + x + x2 + (1 | g)), df; REML=false, progress=false)
    reml = fit(MixedModel, @formula(y ~ 1 + x + x2 + (1 | g)), df; REML=true, progress=false)
    return JObj([
        "schema_version" => "1.0.0",
        "source" => source_version(),
        "id" => "rank_deficient_metrics",
        "formula" => "y ~ 1 + x + x2 + (1 | g)",
        "nobs" => nrow(df),
        "fixed_effect_rank" => ml.feterm.rank,
        "dof" => length(getproperty(ml, :theta)) + ml.feterm.rank + 1,
        "ml" => JObj([
            "objective" => objective(ml),
            "aic" => aic(ml),
            "bic" => bic(ml),
            "sigma" => getproperty(ml, :sigma),
        ]),
        "reml" => JObj([
            "objective" => objective(reml),
            "sigma" => getproperty(reml, :sigma),
            "varest" => getproperty(reml, :sigma)^2,
        ]),
    ])
end

function gamma_log_data()
    group_effects = [-0.25, 0.1, 0.3, -0.15]
    y = Float64[]
    x = Float64[]
    group = String[]
    for g in 0:3
        for obs in 0:4
            xv = Float64(obs) - 2.0
            eta = 1.2 + 0.25 * xv + group_effects[g + 1]
            wiggle = 1.0 + 0.06 * Float64((g + obs) % 3)
            push!(y, exp(eta) * wiggle)
            push!(x, xv)
            push!(group, "g$(g + 1)")
        end
    end
    return DataFrame(y=y, x=x, group=categorical(group))
end

function gamma_glmm_engines_fixture()
    df = gamma_log_data()
    model = fit(
        MixedModel,
        @formula(y ~ 1 + x + (1 | group)),
        df,
        Gamma(),
        LogLink();
        fast=false,
        nAGQ=1,
        progress=false,
    )
    return JObj([
        "schema_version" => "1.0.0",
        "source" => "Deterministic Gamma-log GLMM fixture cross-checked against MixedModels.jl 5.3.0 and lme4 2.0-1 on 2026-04-29. rust_reference.loglik updated 2026-05-18 (mote bd-01KRXCQ85SAMGEAH3HBZESJ16H, B1): MixedModelFit::loglikelihood now reports the full normalized -2logLik scale (response/dispersion constants retained) instead of -objective/2; the corrected value -23.9751 sits ~0.3% from the independent MixedModels.jl engine reference (-23.8936), the residual being the documented dispersion-family scale divergence — the prior -0.5151 was off the Julia oracle by ~46x. beta/theta/objective are unchanged by B1.",
        "formula" => "y ~ 1 + x + (1 | group)",
        "family" => "gamma",
        "link" => "log",
        "n_agq" => 1,
        "nobs" => nrow(df),
        "dof" => 4,
        "data_recipe" => JObj([
            "groups" => 4,
            "observations_per_group" => 5,
            "intercept" => 1.2,
            "slope" => 0.25,
            "group_effects" => [-0.25, 0.1, 0.3, -0.15],
            "wiggle_base" => 1.0,
            "wiggle_step" => 0.06,
            "wiggle_modulus" => 3,
        ]),
        "rust_reference" => JObj([
            "beta" => [1.2801500995982815, 0.2500623869349765],
            "theta" => [0.000000007500000000000006],
            "dispersion_sigma" => 0.24113736806697922,
            "dispersion_phi" => 0.05814723027826981,
            "objective" => 1.0302950465115652,
            "loglik" => -23.975120721071598,
            "fitted_mu_head" => [
                2.181527513571749,
                2.801311534418864,
                3.5971796202652278,
                4.6191582269466664,
                5.931486602827991,
                2.181527513571749,
            ],
        ]),
        "engines" => [
            JObj([
                "engine" => "MixedModels.jl",
                "status" => "fit",
                "version" => "MixedModels $(pkgversion(MixedModels)); Julia $(VERSION); DataFrames $(pkgversion(DataFrames)); GLM $(module_version("GLM"))",
                "beta" => collect(coef(model)),
                "theta" => collect(getproperty(model, :theta)),
                "dispersion" => getproperty(model, :sigma),
                "objective" => objective(model),
                "loglik" => loglikelihood(model),
                "verdict" => "parity_reference",
                "note" => "Matches the Rust profiled objective and fixed effects at fixture tolerance; MixedModels.jl warns that dispersion-family GLMM results are not yet reliable.",
            ]),
            JObj([
                "engine" => "lme4::glmer",
                "status" => "fit",
                "version" => "lme4 2.0-1; Matrix 1.7-3; R 4.5.1",
                "beta" => [1.2525348960238136, 0.2514329570450088],
                "theta" => [1.673899552748074],
                "dispersion" => 0.005280706050124609,
                "objective" => nothing,
                "loglik" => 1.231780136165594,
                "verdict" => "documented_divergence",
                "note" => "glmer profiles Gamma dispersion differently and is recorded as a comparison point, not as the sole oracle for this fixture.",
            ]),
            JObj([
                "engine" => "glmmTMB",
                "status" => "unavailable",
                "version" => nothing,
                "beta" => nothing,
                "theta" => nothing,
                "dispersion" => nothing,
                "objective" => nothing,
                "loglik" => nothing,
                "verdict" => "not_run",
                "note" => "R package glmmTMB was not installed in the local validation environment.",
            ]),
        ],
        "notes" => [
            "This compact fixture intentionally uses deterministic positive responses so Rust, R, and Julia can rebuild the same data without relying on language-specific Gamma RNG streams.",
            "glmer is preserved as a drift sentinel because its Gamma dispersion profiling can disagree with the MixedModels.jl-style PIRLS objective; it must not be promoted to the only oracle for this path.",
        ],
    ])
end

# Regenerate tests/fixtures/parity/glmm_fast_oracles.json: MixedModels.jl
# fast=true oracle rows for large GLMMs where Rust intentionally uses the
# fast-PIRLS comparison path (see comparison/parity_scorecard.toml). Oracle
# numerics (objective, beta, theta, feval counts) are refreshed from live
# fits; tolerances, comparability policy, and classification strings are
# versioned here because they encode Rust-side gating decisions, not Julia
# outputs.
function glmm_fast_oracles_fixture()
    rows = Any[]

    contra = DataFrame(MixedModels.dataset(:contra))
    contra.usenum = Float64.(contra.use .== "Y")
    contra_ri = fit(
        MixedModel,
        @formula(usenum ~ 1 + age + livch + urban + (1 | dist)),
        contra,
        Bernoulli();
        fast=true,
        progress=false,
    )
    push!(rows, fast_oracle_row(
        contra_ri;
        dataset="contraception",
        formula="use ~ 1 + age + livch + urban + (1 | dist)",
        family="Binomial",
        link="Logit",
        compare_beta=false,
        beta_abs_tol=nothing,
        compare_theta=true,
        theta_abs_tol=0.001,
        classification="fast-PIRLS oracle: Rust and MixedModels.jl fast=true agree on the profiled objective; beta is not directly compared because the local fixture uses different factor coding than the lme4 comparison artifact.",
    ))

    contra_rs = fit(
        MixedModel,
        @formula(usenum ~ 1 + age + livch + urban + (1 + urban | dist)),
        contra,
        Bernoulli();
        fast=true,
        progress=false,
    )
    push!(rows, fast_oracle_row(
        contra_rs;
        dataset="contraception",
        formula="use ~ 1 + age + livch + urban + (1 + urban | dist)",
        family="Binomial",
        link="Logit",
        compare_beta=false,
        beta_abs_tol=nothing,
        compare_theta=false,
        theta_abs_tol=nothing,
        classification="fast-PIRLS oracle: Rust and MixedModels.jl fast=true agree on the profiled objective; beta and Cholesky parameters are not directly compared because factor coding changes the random-slope basis.",
    ))

    grouseticks = DataFrame(MixedModels.dataset(:grouseticks))
    # The centered-height covariate is named `ch` so the emitted coefficient
    # name matches the original oracle run.
    grouseticks.ch = Float64.(grouseticks.height) .- mean(Float64.(grouseticks.height))
    grouseticks_fit = fit(
        MixedModel,
        @formula(ticks ~ 1 + year + ch + (1 | brood) + (1 | index) + (1 | location)),
        grouseticks,
        Poisson();
        fast=true,
        progress=false,
    )
    push!(rows, fast_oracle_row(
        grouseticks_fit;
        dataset="grouseticks",
        formula="TICKS ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | INDEX) + (1 | LOCATION)",
        family="Poisson",
        link="Log",
        compare_beta=true,
        beta_abs_tol=0.0005,
        compare_theta=true,
        theta_abs_tol=0.002,
        classification="fast-PIRLS oracle: Rust and MixedModels.jl fast=true agree on the profiled objective, fixed effects, and random-intercept scales; lme4 reports a different joint optimum.",
    ))

    verbagg = DataFrame(MixedModels.dataset(:verbagg))
    verbagg.r2num = Float64.(verbagg.r2 .== "Y")
    verbagg_fit = fit(
        MixedModel,
        @formula(r2num ~ 1 + anger + gender + btype + situ + mode + (1 | subj) + (1 | item)),
        verbagg,
        Bernoulli();
        fast=true,
        progress=false,
    )
    push!(rows, fast_oracle_row(
        verbagg_fit;
        dataset="verbagg",
        formula="r2 ~ 1 + Anger + Gender + btype + situ + mode + (1 | id) + (1 | item)",
        family="Binomial",
        link="Logit",
        compare_beta=false,
        beta_abs_tol=nothing,
        compare_theta=true,
        theta_abs_tol=0.001,
        classification="fast-PIRLS oracle: Rust and MixedModels.jl fast=true agree on the profiled objective and random-intercept scales; beta is not directly compared because the local fixture uses different factor coding than the lme4 comparison artifact.",
    ))

    return JObj([
        "schema_version" => "1.0.0",
        "reference_engine" => source_version(),
        "fit_mode" => "fast=true",
        "generated_at" => Dates.format(Dates.today(), "yyyy-mm-dd"),
        "source" => "Local MixedModels.jl project oracle run for large GLMM rows where Rust intentionally uses the fast PIRLS comparison path.",
        "rows" => rows,
        "notes" => [
            "This fixture does not promote fast=true semantics as the final GLMM target; it prevents the current large-row allowlist from being unexplained.",
            "Rows that can use the joint beta+theta path in routine time are still gated directly against lme4 in comparison/rust_results.json.",
        ],
    ])
end

function fast_oracle_row(
    model;
    dataset,
    formula,
    family,
    link,
    compare_beta,
    beta_abs_tol,
    compare_theta,
    theta_abs_tol,
    classification,
)
    beta = collect(coef(model))
    theta = collect(getproperty(model, :theta))
    return JObj([
        "dataset" => dataset,
        "formula" => formula,
        "family" => family,
        "link" => link,
        "estimator" => "Laplace",
        "objective" => objective(model),
        "objective_abs_tol" => 0.001,
        "optimizer_fevals" => model.optsum.feval,
        "coef_names" => collect(String.(coefnames(model))),
        "mixedmodels_jl_beta" => beta,
        "mixedmodels_jl_theta" => theta,
        "rust_comparable_beta" => compare_beta ? beta : nothing,
        "beta_abs_tol" => beta_abs_tol,
        "rust_comparable_theta_abs" => compare_theta ? abs.(theta) : nothing,
        "theta_abs_tol" => theta_abs_tol,
        "classification" => classification,
    ])
end

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("usage: regenerate_julia_parity_fixtures.jl --out-dir=<dir>")
        return
    end
    out_dir = flag("out-dir", nothing)
    isnothing(out_dir) && error("usage: regenerate_julia_parity_fixtures.jl --out-dir=<dir>")
    write_json(out_dir, "tests/fixtures/parity/cbpp_agq5.json", cbpp_agq5_fixture())
    write_json(out_dir, "tests/fixtures/parity/kb07_ranef.json", kb07_ranef_fixture())
    write_json(out_dir, "tests/fixtures/parity/parmap_vsize3.json", parmap_vsize3_fixture())
    write_json(out_dir, "tests/fixtures/parity/rank_deficient_metrics.json", rank_deficient_metrics_fixture())
    write_json(out_dir, "tests/fixtures/parity/gamma_glmm_engines.json", gamma_glmm_engines_fixture())
    write_json(out_dir, "tests/fixtures/parity/glmm_fast_oracles.json", glmm_fast_oracles_fixture())
end

main()
