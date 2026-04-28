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

centered_mod(value, modulus, center, scale) = (mod(value, modulus) - center) * scale

function simulate_large_theta_data(; n_subjects, n_items, n_sites, n_rep)
    β = [250.0, 9.5]

    reaction = Float64[]
    days = Float64[]
    subj = String[]
    item = String[]
    site = String[]
    sizehint!(reaction, n_subjects * n_items * n_rep)
    sizehint!(days, n_subjects * n_items * n_rep)
    sizehint!(subj, n_subjects * n_items * n_rep)
    sizehint!(item, n_subjects * n_items * n_rep)
    sizehint!(site, n_subjects * n_items * n_rep)

    for s in 0:(n_subjects - 1)
        subj_b0 = centered_mod(7 * s + 3, 19, 9.0, 2.4)
        subj_b1 = centered_mod(11 * s + 5, 17, 8.0, 0.38) + 0.05 * subj_b0
        subj_label = string("S", lpad(s + 1, 3, '0'))

        for i in 0:(n_items - 1)
            item_b0 = centered_mod(13 * i + 2, 23, 11.0, 1.6)
            item_b1 = centered_mod(5 * i + 7, 19, 9.0, 0.27) - 0.04 * item_b0
            item_label = string("I", lpad(i + 1, 3, '0'))

            for r in 0:(n_rep - 1)
                k = mod(5 * s + 3 * i + r, n_sites)
                site_b0 = centered_mod(3 * k + 1, 13, 6.0, 1.2)
                site_b1 = centered_mod(7 * k + 4, 11, 5.0, 0.18) + 0.03 * site_b0
                ϵ = centered_mod(13 * s + 7 * i + 3 * r + 2 * k, 29, 14.0, 0.9)
                x = Float64(r) + Float64(mod(i, 4)) * 0.35 + Float64(mod(s, 3)) * 0.1

                μ = β[1] + β[2] * x +
                    subj_b0 + subj_b1 * x +
                    item_b0 + item_b1 * x +
                    site_b0 + site_b1 * x

                push!(reaction, μ + ϵ)
                push!(days, x)
                push!(subj, subj_label)
                push!(item, item_label)
                push!(site, string("K", lpad(k + 1, 3, '0')))
            end
        end
    end

    DataFrame(
        reaction=reaction,
        days=days,
        subj=categorical(subj),
        item=categorical(item),
        site=categorical(site),
    )
end

# ── Benchmark function ─────────────────────────────────────────────────

function bench_fit(df, f; n_warmup=1, n_reps=5, reml=true)

    # Warmup
    for _ in 1:n_warmup
        fit(MixedModel, f, df; progress=false, REML=reml)
    end

    # Timed runs
    times = Float64[]
    objectives = Float64[]
    for _ in 1:n_reps
        GC.gc()
        t0 = time_ns()
        m = fit(MixedModel, f, df; progress=false, REML=reml)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)  # ms
        push!(objectives, objective(m))
    end

    return times, objectives
end

function bench_fit_with_feval(df, f; n_warmup=1, n_reps=5, reml=true)
    for _ in 1:n_warmup
        fit(MixedModel, f, df; progress=false, REML=reml)
    end

    times = Float64[]
    objectives = Float64[]
    fevals = Float64[]
    for _ in 1:n_reps
        GC.gc()
        t0 = time_ns()
        m = fit(MixedModel, f, df; progress=false, REML=reml)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
        push!(objectives, objective(m))
        push!(fevals, Float64(m.optsum.feval))
    end

    return times, objectives, fevals
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

println("\n# Large-theta RE: reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)")
println("scenario,n_subjects,n_items,n_sites,n_rep,total_n,median_ms,mean_ms,min_ms,objective,median_feval,mean_feval")

large_theta_scenarios = [
    (18, 12, 6, 4, "crossed_small"),
    (36, 24, 8, 4, "crossed_medium"),
    (72, 36, 12, 4, "crossed_large"),
]

f_large_theta = @formula(reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site))

for (n_subj, n_items, n_sites, n_rep, label) in large_theta_scenarios
    df = simulate_large_theta_data(
        n_subjects=n_subj,
        n_items=n_items,
        n_sites=n_sites,
        n_rep=n_rep,
    )
    total_n = nrow(df)

    times, objs, fevals = bench_fit_with_feval(df, f_large_theta; n_warmup=1, n_reps=5)

    med = median(times)
    mn = mean(times)
    mi = minimum(times)
    obj = mean(objs)
    fe_med = median(fevals)
    fe_mean = mean(fevals)

    @printf("%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.6f,%.1f,%.1f\n",
            label, n_subj, n_items, n_sites, n_rep, total_n, med, mn, mi, obj, fe_med, fe_mean)
end
