#!/usr/bin/env julia
# Dump MixedModels.jl-only datasets to CSV under datasets/<name>/, fit each
# recommended formula, and emit auto-managed expected.toml + provenance.toml.
#
# Currently covers kb07 (Kliegl, Kuschela & Laubrock 2015 — the optimizer
# stress test). Future additions: :insteval, :contra, :oxide, :mrk17_exp1.
#
# Usage:
#   julia --project=. scripts/dump_julia_datasets.jl              # full: csv + pin
#   julia --project=. scripts/dump_julia_datasets.jl --pin-only   # skip csv, just refit
#
# The CSV format mirrors scripts/dump_datasets.R. Hand-authored
# `[fits.expected]` blocks in meta.toml always win — sibling expected.toml
# only fills slots that meta.toml leaves empty.

using MixedModels, DataFrames, StatsModels, Dates, Printf
using TOML

# ---- Repo location ------------------------------------------------------

function repo_root()
    d = pwd()
    while !isfile(joinpath(d, "Cargo.toml"))
        parent = dirname(d)
        parent == d && error("could not find Cargo.toml ancestor; run from repo or a subdir.")
        d = parent
    end
    d
end

# ---- CSV dump (unchanged) -----------------------------------------------

function write_csv(path::String, df::DataFrame)
    qs(s) = "\"" * replace(String(s), "\"" => "\"\"") * "\""
    fmt(v) = v isa AbstractString ? qs(v) :
             v isa Number          ? string(v) :
             qs(string(v))
    open(path, "w") do io
        println(io, join((qs(string(n)) for n in propertynames(df)), ","))
        for row in eachrow(df)
            println(io, join((fmt(row[c]) for c in propertynames(df)), ","))
        end
    end
end

function dump_csv(df::DataFrame, name::String, root::String)
    outdir = joinpath(root, "datasets", name)
    mkpath(outdir)
    out = DataFrame()
    factor_levels = Pair{String,Vector{String}}[]
    for col in propertynames(df)
        vals = df[!, col]
        nm = String(col)
        if eltype(vals) <: Number
            out[!, nm] = vals
        else
            ss = String.(vals)
            out[!, nm] = ss
            push!(factor_levels, nm => unique(ss))
        end
    end
    csv_path = joinpath(outdir, "data.csv")
    write_csv(csv_path, out)
    open(joinpath(outdir, "_levels.txt"), "w") do io
        for (nm, lvls) in factor_levels
            println(io, "$nm: ", join(lvls, ","))
        end
    end
    @info "wrote csv" file=csv_path rows=nrow(out) cols=ncol(out)
end

# ---- Reference fitting --------------------------------------------------

# StatsModels' formula parser does not desugar lme4's `||` (zerocorr)
# operator when fed through `@formula(eval(parse(text)))`. Rewrite each
# `(lhs || rhs)` into `(zerocorr(lhs | rhs))` before handing to the macro.
function lme4_to_mm(text::AbstractString)
    s = String(text)
    while true
        idx = findfirst("||", s)
        idx === nothing && return s
        l = first(idx)
        # Walk back to the enclosing `(` at the same paren depth.
        depth = 0
        open_pos = 0
        i = l - 1
        while i >= 1
            c = s[i]
            if c == ')'
                depth += 1
            elseif c == '('
                if depth == 0
                    open_pos = i
                    break
                end
                depth -= 1
            end
            i -= 1
        end
        open_pos == 0 && error("unmatched `||` outside parens in `$s`")
        # Walk forward to the enclosing `)`.
        depth = 0
        close_pos = 0
        i = last(idx) + 1
        while i <= lastindex(s)
            c = s[i]
            if c == '('
                depth += 1
            elseif c == ')'
                if depth == 0
                    close_pos = i
                    break
                end
                depth -= 1
            end
            i = nextind(s, i)
        end
        close_pos == 0 && error("unmatched `||` inside parens in `$s`")
        inner = s[(open_pos + 1):(l - 1)] * "|" * s[(last(idx) + 1):(close_pos - 1)]
        s = s[1:(open_pos - 1)] * "(zerocorr(" * inner * "))" * s[(close_pos + 1):end]
    end
end

# Convert a meta.toml formula string into a runtime formula. Wraps the
# preprocessed string in `@formula(...)` and evals.
function parse_formula(text::AbstractString)
    rewritten = lme4_to_mm(text)
    expr = Meta.parse("@formula($(rewritten))")
    Core.eval(@__MODULE__, expr)
end

function fit_one(formula_text::String, family_text::String, link_text::String,
                 estimator_text::String, weights_text::Union{Nothing,String},
                 df::DataFrame)
    fam = lowercase(family_text)
    est = lowercase(estimator_text)
    if fam == "gaussian" && lowercase(link_text) == "identity"
        reml = est == "reml"
        try
            form = parse_formula(formula_text)
            return fit(MixedModel, form, df; REML = reml, progress = false)
        catch e
            @warn "lmm fit failed" formula=formula_text exception=e
            return nothing
        end
    end
    @warn "unsupported family/link in Julia pinner; skipping" family=family_text link=link_text formula=formula_text
    nothing
end

function extract_expected(m, family_text::String)
    m === nothing && return nothing
    β = collect(coef(m))

    # Random-effects σ and (when 2-D) correlation.
    re_sigmas = Float64[]
    re_corr = nothing
    cors_seen = Tuple{Int,Int,Float64}[]
    for vc in MixedModels.VarCorr(m).σρ
        for σ in vc.σ
            push!(re_sigmas, σ)
        end
        for ρ in vc.ρ
            push!(cors_seen, (length(re_sigmas), length(re_sigmas), ρ))
        end
    end
    if length(cors_seen) == 1
        re_corr = cors_seen[1][3]
    end

    θ = collect(m.θ)
    is_singular = issingular(m)
    obj = objective(m)

    σ = nothing
    if lowercase(family_text) == "gaussian"
        σ = m.σ
    end

    return (
        beta = β,
        sigma = σ,
        re_sigmas = re_sigmas,
        re_corr = re_corr,
        theta = θ,
        objective = obj,
        is_singular = is_singular,
    )
end

# ---- TOML emission ------------------------------------------------------

toml_str(s::AbstractString) = "\"" * replace(replace(s, "\\" => "\\\\"), "\"" => "\\\"") * "\""

toml_num_array(v) = isempty(v) ? "[]" : "[" * join((@sprintf("%.17g", x) for x in v), ", ") * "]"

function format_expected_block(exp, formula_text::String, estimator_text::String)
    out = String[]
    push!(out, "[[expected]]")
    push!(out, "formula = " * toml_str(formula_text))
    push!(out, "estimator = " * toml_str(estimator_text))
    push!(out, "beta = " * toml_num_array(exp.beta))
    if exp.sigma !== nothing
        push!(out, @sprintf("sigma = %.17g", exp.sigma))
    end
    if !isempty(exp.re_sigmas)
        push!(out, "re_sigmas = " * toml_num_array(exp.re_sigmas))
    end
    if exp.re_corr !== nothing
        push!(out, @sprintf("re_corr = %.17g", exp.re_corr))
    end
    if !isempty(exp.theta)
        push!(out, "theta = " * toml_num_array(exp.theta))
    end
    push!(out, @sprintf("objective = %.17g", exp.objective))
    push!(out, "is_singular = " * (exp.is_singular ? "true" : "false"))
    join(out, "\n")
end

function write_expected_toml(name::String, entries::Vector{String}, root::String)
    isempty(entries) && return
    outdir = joinpath(root, "datasets", name)
    path = joinpath(outdir, "expected.toml")
    header = """
    # Auto-generated by scripts/dump_julia_datasets.jl — do not hand-edit.
    # Pinned reference fits for entries that meta.toml leaves empty.
    # Loader: src/datasets/mod.rs::load_meta merges these into Meta.fits[i].expected.
    """
    open(path, "w") do io
        write(io, header)
        write(io, "\n")
        write(io, join(entries, "\n\n"))
        write(io, "\n")
    end
    @info "wrote expected.toml" name=name entries=length(entries)
end

function write_provenance_toml(name::String, root::String; optimizer::String = "default")
    outdir = joinpath(root, "datasets", name)
    path = joinpath(outdir, "provenance.toml")
    mm_ver = string(pkgversion(MixedModels))
    julia_ver = string(VERSION)
    host = string(Sys.KERNEL, "/", Sys.MACHINE)
    date = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
    open(path, "w") do io
        println(io, "# Auto-generated by scripts/dump_julia_datasets.jl — do not hand-edit.")
        println(io, "# Regeneration provenance for the auto-managed sibling expected.toml.")
        println(io)
        println(io, "tool = ", toml_str("MixedModels.jl $mm_ver"))
        println(io, "tool_name = \"MixedModels.jl\"")
        println(io, "tool_version = ", toml_str(mm_ver))
        println(io, "julia_version = ", toml_str(julia_ver))
        println(io, "date = ", toml_str(date))
        println(io, "host = ", toml_str(host))
        println(io, "regenerator = \"scripts/dump_julia_datasets.jl\"")
        println(io, "optimizer = ", toml_str(optimizer))
    end
    @info "wrote provenance.toml" name=name
end

# ---- Driver ------------------------------------------------------------

# Inline expected blocks already pinned in meta.toml win — only fill
# slots that meta.toml leaves empty so we never silently overwrite.
function fit_inline_already_set(fit_entry)
    haskey(fit_entry, "expected") && fit_entry["expected"] !== nothing && !isempty(fit_entry["expected"])
end

function pin_dataset(df::DataFrame, name::String, root::String)
    meta_path = joinpath(root, "datasets", name, "meta.toml")
    if !isfile(meta_path)
        @info "skip pin: no meta.toml" name=name
        return
    end
    meta = TOML.parsefile(meta_path)
    fits = get(meta, "fits", Vector{Any}())
    if isempty(fits)
        @info "skip pin: no [[fits]]" name=name
        write_provenance_toml(name, root)
        return
    end
    entries = String[]
    for fit_entry in fits
        if fit_inline_already_set(fit_entry)
            @info "keep inline" name=name formula=fit_entry["formula"] estimator=fit_entry["estimator"]
            continue
        end
        m = fit_one(
            fit_entry["formula"],
            fit_entry["family"],
            fit_entry["link"],
            fit_entry["estimator"],
            get(fit_entry, "weights", nothing),
            df,
        )
        m === nothing && continue
        exp = extract_expected(m, fit_entry["family"])
        exp === nothing && continue
        push!(entries, format_expected_block(exp, fit_entry["formula"], fit_entry["estimator"]))
        @info "pin" name=name formula=fit_entry["formula"] estimator=fit_entry["estimator"]
    end
    write_expected_toml(name, entries, root)
    write_provenance_toml(name, root)
end

function dump_one(df::DataFrame, name::String, root::String; pin_only::Bool)
    pin_only || dump_csv(df, name, root)
    pin_dataset(df, name, root)
end

# ---- Main --------------------------------------------------------------

pin_only = "--pin-only" in ARGS
root = repo_root()

# kb07 — Kliegl, Kuschela, Laubrock (2015); distributed in MixedModels.jl test data.
# 1789 rows, 6 fixed-effects covariates, 56 subjects × 32 items crossed.
# Frequently produces a singular RE covariance with the maximal model.
kb07 = DataFrame(MixedModels.dataset(:kb07))
dump_one(kb07, "kb07", root; pin_only=pin_only)

@info "done."
