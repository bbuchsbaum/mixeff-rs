#!/usr/bin/env julia

using MixedModels
using DataFrames
using CategoricalArrays
using Random
using Printf

function simulate_data(; n_subjects, n_obs_per_subject, seed=42)
    rng = MersenneTwister(seed)
    beta = [250.0, 10.0]
    sigma = 25.0
    lambda = [24.0 0.0; 1.68 5.23]

    total_n = n_subjects * n_obs_per_subject
    reaction = Float64[]
    days = Float64[]
    subj = String[]
    sizehint!(reaction, total_n)
    sizehint!(days, total_n)
    sizehint!(subj, total_n)

    u = randn(rng, 2, n_subjects)
    b = lambda * u

    for i in 1:n_subjects
        label = "S" * lpad(string(i), 4, '0')
        for d in 0:(n_obs_per_subject - 1)
            mu = beta[1] + beta[2] * d + b[1, i] + b[2, i] * d
            push!(reaction, mu + sigma * randn(rng))
            push!(days, Float64(d))
            push!(subj, label)
        end
    end

    return DataFrame(reaction=reaction, days=days, subj=categorical(subj))
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

    return DataFrame(
        reaction=reaction,
        days=days,
        subj=categorical(subj),
        item=categorical(item),
        site=categorical(site),
    )
end

function get_flag(name, default)
    prefix = "--" * name * "="
    for arg in ARGS
        startswith(arg, prefix) || continue
        value = split(arg, "=", limit=2)[2]
        if default isa Bool
            return lowercase(value) == "true"
        elseif default isa Int
            return parse(Int, value)
        elseif default isa String
            return value
        else
            error("unsupported flag type")
        end
    end
    return default
end

function get_theta_flag()
    prefix = "--theta="
    for arg in ARGS
        startswith(arg, prefix) || continue
        value = split(arg, "=", limit=2)[2]
        isempty(value) && return Float64[]
        return parse.(Float64, split(value, ","))
    end
    return nothing
end

json_num(x) = @sprintf("%.17g", Float64(x))
json_array(xs) = "[" * join(json_num.(collect(xs)), ",") * "]"
json_string(x) = "\"" * replace(String(x), "\"" => "\\\"") * "\""
json_null_or_array(xs) = isnothing(xs) ? "null" : json_array(xs)
json_null_or_num(x) = isnothing(x) ? "null" : json_num(x)

function objective_components(model)
    L = model.L
    nre = length(model.reterms)
    logdet_re_half = sum(j -> MixedModels.LD(L[MixedModels.kp1choose2(j)]), 1:nre)
    lastL = last(L)
    logdet_xx_half = model.optsum.REML ? MixedModels.LD(lastL) - log(last(lastL)) : zero(eltype(lastL))
    return 2 * logdet_re_half, 2 * logdet_xx_half, 2 * (logdet_re_half + logdet_xx_half)
end

function dump_contra_glmm(n_agq)
    # `MixedModels.dataset(:contra)` ships with the package via
    # MixedModelsDatasets. Formula matches the Rust example for a row-by-row
    # diff: use ~ 1 + age + age² + urban + livch + (1 | urban×dist).
    contra = MixedModels.dataset(:contra)
    formula_str = "use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)"

    # Build the same pre-computed columns the Rust side uses.
    df = DataFrame(contra)
    df.use_num = Float64.(df.use .== "Y")
    df.age2 = df.age .^ 2
    df.urban_dist = string.(df.urban) .* "_" .* string.(df.dist)

    fm = @formula(use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist))
    model = fit(MixedModel, fm, df, Bernoulli(); fast=false, nAGQ=n_agq, progress=false)

    fit_theta = collect(getproperty(model, :theta))
    fit_beta = collect(coef(model))
    dev_agq = deviance(model, n_agq)
    dev_lap = deviance(model, 1)

    println("{")
    println("  \"model\": ", json_string("contra-glmm"), ",")
    println("  \"formula\": ", json_string(formula_str), ",")
    println("  \"family\": ", json_string("bernoulli"), ",")
    println("  \"link\": ", json_string("logit"), ",")
    println("  \"n_rows\": ", nrow(df), ",")
    println("  \"n_groups\": ", length(unique(df.urban_dist)), ",")
    println("  \"fit_n_agq\": ", n_agq, ",")
    println("  \"fit_theta\": ", json_array(fit_theta), ",")
    println("  \"fit_beta\": ", json_array(fit_beta), ",")
    println("  \"fit_objective\": ", json_num(objective(model)), ",")
    println("  \"fit_deviance_laplace\": ", json_num(dev_lap), ",")
    println("  \"fit_deviance_agq\": ", json_num(dev_agq), ",")
    println("  \"fit_feval\": ", model.optsum.feval)
    println("}")
end

function main()
    model_name = get_flag("model", "scalar")
    if model_name == "contra-glmm"
        # GLMM parity dump uses a fixed real dataset; only --n-agq is honoured.
        n_agq = get_flag("n-agq", 7)
        dump_contra_glmm(n_agq)
        return
    end
    n_subjects = get_flag("n-subj", 18)
    n_obs_per_subject = get_flag("n-obs", 10)
    n_items = get_flag("n-items", 12)
    n_sites = get_flag("n-sites", 6)
    n_rep = get_flag("n-rep", 4)
    seed = get_flag("seed", 42)
    reml = get_flag("reml", true)
    input_theta = get_theta_flag()

    formula = model_name == "scalar" ?
        @formula(reaction ~ 1 + days + (1 | subj)) :
        model_name == "vector" ?
            @formula(reaction ~ 1 + days + (1 + days | subj)) :
            model_name == "crossed" ?
                @formula(reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)) :
            error("unknown model $(model_name)")

    df = model_name == "crossed" ?
        simulate_large_theta_data(
            n_subjects=n_subjects,
            n_items=n_items,
            n_sites=n_sites,
            n_rep=n_rep,
        ) :
        simulate_data(n_subjects=n_subjects, n_obs_per_subject=n_obs_per_subject, seed=seed)
    model = fit(MixedModel, formula, df; progress=false, REML=reml)
    data_reaction_sum = sum(df.reaction)
    data_days_sum = sum(df.days)

    fit_theta = copy(getproperty(model, :theta))
    fit_beta = collect(coef(model))
    fit_sigma = getproperty(model, :sigma)
    fit_objective = objective(model)
    fit_pwrss = pwrss(model)
    fit_logdet_re, fit_logdet_xx, fit_logdet_total = objective_components(model)
    fit_feval = model.optsum.feval

    input_theta_summary = if isnothing(input_theta)
        nothing
    else
        probe = deepcopy(model)
        obj = if length(input_theta) == 1
            objective!(probe, only(input_theta))
        else
            objective!(probe, input_theta)
        end
        pwrss_input = pwrss(probe)
        logdet_re_input, logdet_xx_input, logdet_total_input = objective_components(probe)
        (obj, pwrss_input, logdet_re_input, logdet_xx_input, logdet_total_input)
    end

    println("{")
    println("  \"model\": ", json_string(model_name), ",")
    println("  \"formula\": ", json_string(string(formula)), ",")
    println("  \"n_rows\": ", nrow(df), ",")
    println("  \"n_subjects\": ", n_subjects, ",")
    println("  \"n_obs_per_subject\": ", n_obs_per_subject, ",")
    println("  \"n_items\": ", model_name == "crossed" ? n_items : "null", ",")
    println("  \"n_sites\": ", model_name == "crossed" ? n_sites : "null", ",")
    println("  \"n_rep\": ", model_name == "crossed" ? n_rep : "null", ",")
    println("  \"data_reaction_sum\": ", json_num(data_reaction_sum), ",")
    println("  \"data_days_sum\": ", json_num(data_days_sum), ",")
    println("  \"seed\": ", seed, ",")
    println("  \"reml\": ", lowercase(string(reml)), ",")
    println("  \"fit_theta\": ", json_array(fit_theta), ",")
    println("  \"fit_beta\": ", json_array(fit_beta), ",")
    println("  \"fit_sigma\": ", json_num(fit_sigma), ",")
    println("  \"fit_objective\": ", json_num(fit_objective), ",")
    println("  \"fit_pwrss\": ", json_num(fit_pwrss), ",")
    println("  \"fit_logdet_re\": ", json_num(fit_logdet_re), ",")
    println("  \"fit_logdet_xx\": ", json_num(fit_logdet_xx), ",")
    println("  \"fit_logdet_total\": ", json_num(fit_logdet_total), ",")
    println("  \"fit_feval\": ", fit_feval, ",")
    println("  \"input_theta\": ", json_null_or_array(input_theta), ",")
    println("  \"objective_at_input_theta\": ", json_null_or_num(isnothing(input_theta_summary) ? nothing : input_theta_summary[1]), ",")
    println("  \"input_theta_pwrss\": ", json_null_or_num(isnothing(input_theta_summary) ? nothing : input_theta_summary[2]), ",")
    println("  \"input_theta_logdet_re\": ", json_null_or_num(isnothing(input_theta_summary) ? nothing : input_theta_summary[3]), ",")
    println("  \"input_theta_logdet_xx\": ", json_null_or_num(isnothing(input_theta_summary) ? nothing : input_theta_summary[4]), ",")
    println("  \"input_theta_logdet_total\": ", json_null_or_num(isnothing(input_theta_summary) ? nothing : input_theta_summary[5]))
    println("}")
end

main()
