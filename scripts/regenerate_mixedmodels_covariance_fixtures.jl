#!/usr/bin/env julia

using DataFrames
using Dates
using MixedModels
using Printf

struct JObj
    fields::Vector{Pair{String, Any}}
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
    elseif value isa AbstractVector || value isa Tuple
        isempty(value) && return "[]"
        if all(item -> item isa Number || item isa AbstractString || item === nothing || item isa Bool, value)
            return "[" * join(render_json.(collect(value), indent), ", ") * "]"
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

function module_version(name)
    for (pkgid, mod) in Base.loaded_modules
        String(pkgid.name) == name && return string(pkgversion(mod))
    end
    return "unknown"
end

function write_json(path, value)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, render_json(value))
        write(io, "\n")
    end
end

function git_commit(repo_root)
    try
        return chomp(read(`git -C $repo_root rev-parse HEAD`, String))
    catch
        return nothing
    end
end

function covariance_summary(model)
    block = first(values(model.sigmarhos))
    std_dev = collect(values(getproperty(block, Symbol("σ"))))
    correlations = collect(getproperty(block, Symbol("ρ")))
    return std_dev, correlations
end

function extract_case(id, rust_formula, mixedmodels_formula, covariance_family, model)
    std_dev, correlations = covariance_summary(model)
    objective_value = objective(model)
    return JObj([
        "id" => id,
        "rust_formula" => rust_formula,
        "mixedmodels_formula" => mixedmodels_formula,
        "reml" => false,
        "covariance_family" => covariance_family,
        "oracle_role" => "supported full/diagonal baseline only; MixedModels.jl does not provide CS or random-effect AR(1) covariance-family syntax",
        "beta" => collect(coef(model)),
        "sigma" => Float64(model.sigma),
        "theta" => collect(model.theta),
        "objective" => objective_value,
        "loglik" => loglikelihood(model),
        "fitted_head" => collect(first(fitted(model), 10)),
        "std_dev" => std_dev,
        "correlations" => correlations,
    ])
end

repo_root = abspath(joinpath(@__DIR__, ".."))
out = length(ARGS) >= 1 ? ARGS[1] : joinpath(repo_root, "tests", "fixtures", "parity", "mixedmodels_covariance_families.json")

sleepstudy = DataFrame(MixedModels.dataset(:sleepstudy))
full_model = fit(
    MixedModel,
    @formula(reaction ~ 1 + days + (1 + days | subj)),
    sleepstudy;
    REML=false,
    progress=false,
)
diagonal_model = fit(
    MixedModel,
    @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
    sleepstudy;
    REML=false,
    progress=false,
)

payload = JObj([
    "source" => "Local Julia/MixedModels.jl covariance-family baseline fixture generated from MixedModels.dataset(:sleepstudy).",
    "generated_at" => string(Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "Z"),
    "julia_version" => string(VERSION),
    "mixedmodels_version" => module_version("MixedModels"),
    "dataset" => "MixedModels.dataset(:sleepstudy)",
    "tolerances" => JObj([
        "beta_abs" => 5e-4,
        "sigma_abs" => 5e-4,
        "theta_abs" => 5e-4,
        "objective_abs" => 5e-3,
        "loglik_abs" => 5e-3,
        "fitted_abs" => 5e-4,
        "std_dev_abs" => 1e-3,
        "correlation_abs" => 5e-4,
    ]),
    "cases" => [
        extract_case(
            "sleepstudy_full_ml",
            "Reaction ~ 1 + Days + (1 + Days | Subject)",
            "reaction ~ 1 + days + (1 + days | subj)",
            "full_cholesky",
            full_model,
        ),
        extract_case(
            "sleepstudy_diagonal_ml",
            "Reaction ~ 1 + Days + diag(1 + Days | Subject)",
            "reaction ~ 1 + days + zerocorr(1 + days | subj)",
            "diagonal",
            diagonal_model,
        ),
    ],
    "notes" => [
        "MixedModels.jl uses zerocorr(...) for diagonal random-effect covariance; it does not parse lme4's || syntax in @formula.",
        "This fixture is a supported-family baseline, not a direct oracle for compound symmetry or random-effect AR(1).",
    ],
])

write_json(out, payload)

provenance = JObj([
    "schema_version" => "1.0",
    "generated_at" => string(Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "Z"),
    "crate_commit" => git_commit(repo_root),
    "regenerator" => "scripts/regenerate_mixedmodels_covariance_fixtures.jl",
    "source_case" => "MixedModels.dataset(:sleepstudy)",
    "reference_engine" => "MixedModels.jl $(module_version("MixedModels"))",
    "notes" => "Julia $(VERSION); ML full and zerocorr sleepstudy covariance-family baselines",
])
write_json(replace(out, r"\.json$" => ".provenance.json"), provenance)
