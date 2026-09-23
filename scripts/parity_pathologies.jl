#!/usr/bin/env julia

using CategoricalArrays
using DataFrames
using Dates
using LinearAlgebra
using MixedModels
using Printf
using TOML

function repo_root(start=pwd())
    d = abspath(start)
    while true
        isfile(joinpath(d, "Cargo.toml")) && return d
        parent = dirname(d)
        parent == d && error("could not find Cargo.toml ancestor")
        d = parent
    end
end

function flag(name, default=nothing)
    prefix = "--" * name * "="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return split(arg, "=", limit=2)[2]
    end
    return default
end

function sqrt_psd(sigma)
    ev = eigen(Symmetric(Matrix{Float64}(sigma)))
    ev.vectors * Diagonal(sqrt.(max.(ev.values, 0.0))) * ev.vectors'
end

function deterministic_data(spec)
    fe_truth = Float64.(spec["fe_truth"])
    group_sizes = Int.(spec["group_sizes"])
    re_cov_truth = Matrix{Float64}(hcat(spec["re_cov_truth"]...)')
    residual_sd = Float64(spec["residual_sd"])
    seed = Int(spec["seed"])

    n_pred = length(fe_truth) - 1
    q = size(re_cov_truth, 1)
    n_slopes = max(0, q - 1)
    sigma_sqrt = sqrt_psd(re_cov_truth)

    y = Float64[]
    g = String[]
    predictors = [Float64[] for _ in 1:n_pred]
    row_index = 0

    for group_index in eachindex(group_sizes)
        z_pool = [
            sin(seed + group_index * 1.7),
            cos(seed * 0.5 + group_index * 2.3),
            sin(seed * 0.25 + group_index * 3.1),
        ]
        u = sigma_sqrt * z_pool[1:q]
        n_g = group_sizes[group_index]
        for within in 1:n_g
            row_index += 1
            centered = n_g <= 1 ? 0.0 : ((within - 1) - (n_g - 1) / 2) / (n_g - 1)
            x = [centered^j + 0.07 * sin((row_index + j) * 0.61) for j in 1:n_pred]
            eta = fe_truth[1]
            for j in 1:n_pred
                eta += fe_truth[j + 1] * x[j]
            end
            q >= 1 && (eta += u[1])
            for j in 1:n_slopes
                eta += u[j + 1] * x[j]
            end
            noise_scale = spec["stratum"] == "reduced-rank" ? 0.0 : 0.1
            push!(y, eta + residual_sd * noise_scale * sin(seed + row_index * 0.73))
            push!(g, @sprintf("g%03d", group_index))
            for j in 1:n_pred
                push!(predictors[j], x[j])
            end
        end
    end

    df = DataFrame(y=y, g=categorical(g))
    for j in 1:n_pred
        df[!, Symbol("x$j")] = predictors[j]
    end
    return df
end

# Dense REML profiled objective of a fitted LinearMixedModel at θ, evaluated in
# the element type of θ. Mathematically identical to MixedModels' `objective`:
#   logdet(I + ZΛΛ'Z') + logdet(X'M⁻¹X) + (n-p)(1 + log(2π pwrss/(n-p))).
function dense_reml_objective(model, θ::AbstractVector{T}) where {T}
    X = T.(Matrix(model.X))
    y = T.(model.y)
    n, p = size(X)
    blocks = Matrix{T}[]
    for (k, re) in enumerate(model.reterms)
        s = size(re.λ, 1)
        lambda = zeros(T, s, s)
        for (index, (term, row, col)) in enumerate(model.parmap)
            term == k && (lambda[row, col] = θ[index])
        end
        push!(blocks, T.(Matrix(re)) * kron(Matrix{T}(I, length(re.levels), length(re.levels)), lambda))
    end
    ZL = hcat(blocks...)
    C = cholesky(Symmetric(Matrix{T}(I, n, n) + ZL * ZL'))
    XtMiX = X' * (C \ X)
    β = XtMiX \ (X' * (C \ y))
    r = y - X * β
    pwrss = dot(r, C \ r)
    return logdet(C) + logdet(cholesky(Symmetric(XtMiX))) +
           (n - p) * (1 + log(2 * T(pi) * pwrss / (n - p)))
end

# The easy-stratum data are nearly noiseless (σ ≈ 0.02), so the Float64
# profiled objective carries ~1e-9 of cancellation noise against a smallest
# θ-Hessian eigenvalue of ~0.06: any Float64 optimizer stop is determined only
# to ~2e-4 in θ, and Linux and macOS stop at different points. Polish the
# MixedModels optimum by Newton steps on the same objective in 256-bit BigFloat
# (MPFR is correctly rounded, hence platform-independent) and re-evaluate the
# model at the polished θ.
function polish_theta!(model)
    fitted_objective = objective(model)
    theta0 = collect(model.θ)
    theta = setprecision(BigFloat, 256) do
        f(t) = dense_reml_objective(model, t)
        θ = big.(theta0)
        n = length(θ)
        h = big"1e-25"
        unit(k) = (v = zeros(BigFloat, n); v[k] = one(BigFloat); v)
        converged = false
        for _ in 1:50
            g = [(f(θ + h * unit(i)) - f(θ - h * unit(i))) / (2h) for i in 1:n]
            H = [
                (f(θ + h * unit(i) + h * unit(j)) - f(θ + h * unit(i) - h * unit(j)) -
                 f(θ - h * unit(i) + h * unit(j)) + f(θ - h * unit(i) - h * unit(j))) / (4h^2)
                for i in 1:n, j in 1:n
            ]
            isposdef(Symmetric(H)) || error("θ polish: Hessian is not positive definite")
            step = H \ g
            θ -= step
            if maximum(abs, step) < big"1e-40"
                converged = true
                break
            end
        end
        converged || error("θ polish: Newton iteration did not converge")
        Float64.(θ)
    end
    maximum(abs, theta .- theta0) < 1e-2 ||
        error("θ polish moved θ implausibly far from the MixedModels optimum")
    updateL!(MixedModels.setθ!(model, theta))
    objective(model) <= fitted_objective + 1e-8 ||
        error("θ polish: MixedModels objective worsened at the polished θ")
    return model
end

json_string(x) ="\"" * replace(String(x), "\\" => "\\\\", "\"" => "\\\"") * "\""
json_num(x) = isfinite(Float64(x)) ? @sprintf("%.17g", Float64(x)) : "null"
json_array(xs) = "[" * join(json_num.(collect(xs)), ", ") * "]"
json_string_array(xs) = "[" * join(json_string.(collect(xs)), ", ") * "]"

function fit_mmjl(spec)
    df = deterministic_data(spec)
    re_cov_truth = Matrix{Float64}(hcat(spec["re_cov_truth"]...)')
    q = size(re_cov_truth, 1)
    formula = q <= 1 ? @formula(y ~ 1 + x1 + (1 | g)) : @formula(y ~ 1 + x1 + (1 + x1 | g))

    t0 = time()
    status = "ok"
    warning_text = String[]
    model = nothing
    err = nothing
    try
        model = fit(MixedModel, formula, df; REML=true, progress=false)
    catch e
        status = "error"
        err = sprint(showerror, e)
    end
    runtime_ms = (time() - t0) * 1000
    # Only the well-posed easy stratum has an interior optimum to polish; a
    # failed polish is an error, not a silent fallback to the raw fit.
    status == "ok" && spec["stratum"] == "easy" && polish_theta!(model)

    println("{")
    println("  \"schema_version\": \"1.0.0\",")
    println("  \"fixture\": ", json_string(spec["name"]), ",")
    println("  \"stratum\": ", json_string(spec["stratum"]), ",")
    println("  \"engine\": \"MixedModels.jl\",")
    println("  \"version\": ", json_string("MixedModels " * string(pkgversion(MixedModels)) * "; Julia " * string(VERSION)), ",")
    println("  \"source\": \"scripts/parity_pathologies.jl\",")
    println("  \"status\": ", json_string(status), ",")
    println("  \"warnings\": ", json_string_array(warning_text), ",")
    println("  \"converged\": ", status == "ok" ? "true" : "false", ",")
    if status == "ok"
        println("  \"objective\": ", json_num(objective(model)), ",")
        println("  \"theta\": ", json_array(getproperty(model, :theta)), ",")
        println("  \"beta\": ", json_array(coef(model)), ",")
        println("  \"sigma\": ", json_num(getproperty(model, :sigma)), ",")
        println("  \"loglik\": ", json_num(-objective(model) / 2), ",")
    else
        println("  \"error\": ", json_string(err), ",")
        println("  \"objective\": null,")
        println("  \"theta\": [],")
        println("  \"beta\": [],")
        println("  \"sigma\": null,")
        println("  \"loglik\": null,")
    end
    println("  \"runtime_ms\": ", json_num(runtime_ms))
    println("}")
end

function write_provenance_sibling(out_path)
    stem, _ = splitext(basename(out_path))
    prov_path = joinpath(dirname(out_path), stem * ".provenance.json")
    timestamp = string(Dates.now(Dates.UTC)) * "Z"
    mm_ver = string(pkgversion(MixedModels))
    julia_ver = string(VERSION)
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
      "regenerator": "scripts/parity_pathologies.jl",
      "source_case": null,
      "reference_engine": "MixedModels.jl $mm_ver",
      "notes": "Julia $julia_ver"
    }
    """
    open(prov_path, "w") do io
        write(io, body)
    end
end

function main()
    root = repo_root()
    fixture = flag("fixture", joinpath(root, "tests/fixtures/pathology_corpus/easy.toml"))
    fixture_path = startswith(fixture, "/") ? fixture : joinpath(root, fixture)
    spec = TOML.parsefile(fixture_path)
    out = flag("out", nothing)
    if isnothing(out)
        fit_mmjl(spec)
    else
        out_path = startswith(out, "/") ? out : joinpath(root, out)
        open(out_path, "w") do io
            redirect_stdout(io) do
                fit_mmjl(spec)
            end
        end
        # Pair JSON output with a sibling provenance.json so the
        # fixture_hygiene every_golden_has_provenance_sibling test stays
        # green when this script writes a fresh fixture.
        write_provenance_sibling(out_path)
    end
end

main()
