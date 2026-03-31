#!/usr/bin/env julia
#
# Benchmark MixedModels.jl on simulated datasets of varying size.
# Outputs CSV-formatted timing results for comparison with Rust.

using MixedModels
using DataFrames
using CategoricalArrays
using Random
using Statistics
using Printf

# ── Simulate sleepstudy-like data ──────────────────────────────────────
# y = β₀ + β₁*x + b₀ᵢ + b₁ᵢ*x + ε
# with (b₀, b₁) ~ N(0, Σ), ε ~ N(0, σ²)

function simulate_data(; n_subjects, n_obs_per_subject, seed=42)
    rng = MersenneTwister(seed)
    β = [250.0, 10.0]          # fixed effects
    σ  = 25.0                  # residual SD
    # RE covariance: SD_intercept=24, SD_slope=5.5, corr=0.07
    Λ = [24.0 0.0; 1.68 5.23]  # lower Cholesky of RE cov

    N = n_subjects * n_obs_per_subject
    subj  = repeat(string.("S", lpad.(1:n_subjects, 4, '0')), inner=n_obs_per_subject)
    days  = repeat(0:(n_obs_per_subject-1), outer=n_subjects)

    # Random effects
    U = randn(rng, 2, n_subjects)
    B = Λ * U   # 2 × n_subjects

    y = Float64[]
    for i in 1:n_subjects
        for d in 0:(n_obs_per_subject-1)
            μ = β[1] + β[2]*d + B[1,i] + B[2,i]*d
            push!(y, μ + σ * randn(rng))
        end
    end

    DataFrame(reaction=y, days=Float64.(days), subj=categorical(subj))
end

# ── Benchmark function ─────────────────────────────────────────────────

function bench_fit(df, f; n_warmup=1, n_reps=5)

    # Warmup
    for _ in 1:n_warmup
        fit(MixedModel, f, df; progress=false)
    end

    # Timed runs
    times = Float64[]
    objectives = Float64[]
    for _ in 1:n_reps
        GC.gc()
        t0 = time_ns()
        m = fit(MixedModel, f, df; progress=false)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)  # ms
        push!(objectives, objective(m))
    end

    return times, objectives
end

# ── Main ───────────────────────────────────────────────────────────────

println("scenario,n_subjects,n_obs,total_n,median_ms,mean_ms,min_ms,objective")

scenarios = [
    # (n_subjects, n_obs_per_subject, label)
    (18,   10,  "sleepstudy_like"),
    (50,   10,  "medium_50subj"),
    (100,  10,  "medium_100subj"),
    (200,  10,  "large_200subj"),
    (500,  10,  "large_500subj"),
    (1000, 10,  "xlarge_1000subj"),
    (50,   50,  "deep_50x50"),
    (100,  50,  "deep_100x50"),
    (200,  50,  "deep_200x50"),
]

f_vector = @formula(reaction ~ 1 + days + (1 + days | subj))

for (n_subj, n_obs, label) in scenarios
    df = simulate_data(n_subjects=n_subj, n_obs_per_subject=n_obs)
    total_n = nrow(df)

    times, objs = bench_fit(df, f_vector; n_warmup=2, n_reps=7)

    med = median(times)
    mn  = mean(times)
    mi  = minimum(times)
    obj = mean(objs)

    @printf("%s,%d,%d,%d,%.3f,%.3f,%.3f,%.6f\n",
            label, n_subj, n_obs, total_n, med, mn, mi, obj)
end

# Also test scalar RE (random intercept only) — simpler model
println("\n# Scalar RE: reaction ~ 1 + days + (1 | subj)")
println("scenario,n_subjects,n_obs,total_n,median_ms,mean_ms,min_ms,objective")

f_scalar = @formula(reaction ~ 1 + days + (1 | subj))

for (n_subj, n_obs, label) in scenarios
    df = simulate_data(n_subjects=n_subj, n_obs_per_subject=n_obs)
    total_n = nrow(df)

    times, objs = bench_fit(df, f_scalar; n_warmup=2, n_reps=7)

    med = median(times)
    mn  = mean(times)
    mi  = minimum(times)
    obj = mean(objs)

    @printf("scalar_%s,%d,%d,%d,%.3f,%.3f,%.3f,%.6f\n",
            label, n_subj, n_obs, total_n, med, mn, mi, obj)
end
