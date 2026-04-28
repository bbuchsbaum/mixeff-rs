#!/usr/bin/env julia
# Dump MixedModels.jl-only datasets (not in lme4) to CSV under datasets/<name>/.
#
# Currently: kb07 (Kliegl & Bates 2007 — psycholinguistic stress test for the optimizer).
#
# Usage:  julia --project=. scripts/dump_julia_datasets.jl
#         (or run from any other Julia project that has MixedModels + CSV available)
#
# The CSV format mirrors scripts/dump_datasets.R: factors are emitted as
# their character labels, with canonical level order recorded separately
# in `_levels.txt` for cross-checking against meta.toml.

using MixedModels, DataFrames

# Tiny CSV writer — avoids the CSV.jl dep so this script runs in the default
# Julia env (which already has MixedModels + DataFrames). String values are
# always quoted; embedded `"` is doubled per RFC 4180.
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

function repo_root()
    d = pwd()
    while !isfile(joinpath(d, "Cargo.toml"))
        parent = dirname(d)
        parent == d && error("could not find Cargo.toml ancestor; run from repo or a subdir.")
        d = parent
    end
    d
end

function dump_one(df::DataFrame, name::String, root::String)
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
            # Treat anything non-numeric as a factor; its observed first-appearance
            # order is recorded for cross-checking against meta.toml.
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
    @info "wrote" file=csv_path rows=nrow(out) cols=ncol(out)
end

root = repo_root()

# kb07 — Kliegl, Kuschela, Laubrock (2015); also distributed in MixedModels.jl test data.
# 1789 rows, 6 fixed-effects covariates, 32 subjects × 32 items crossed.
# Frequently produces a singular RE covariance with the maximal model.
kb07 = DataFrame(MixedModels.dataset(:kb07))
dump_one(kb07, "kb07", root)

@info "done."
