

function mean_variance_to_lognormal(mean, variance)
    σ² = log1p(variance / abs2(mean))
    μ  = log(mean) - σ² / 2
    return μ, √σ²
end

function rand_qσ(rng::Random.AbstractRNG, mean, variance)
    μ, σ = mean_variance_to_lognormal(mean, variance)
    return rand(rng, Distributions.LogNormal(μ, σ))
end

function ratio_logpdf_qσ(x_old, x_new, variance)
    μ1, σ1 = mean_variance_to_lognormal(x_new, variance)
    μ2, σ2 = mean_variance_to_lognormal(x_old, variance)
    return Distributions.logpdf(Distributions.LogNormal(μ1, σ1), x_old) - Distributions.logpdf(Distributions.LogNormal(μ2, σ2), x_new)
end

function fast_loglik_ratio_σ(σ_new::Real, σ_old::Real, sum_sq::Integer, log_esf::AbstractVector, k::Integer)
    p = length(log_esf) - 1
    z_old = LogExpFunctions.logsumexp(
        log_esf[k] + σ_old / p * (k - 1)^2
        for k in eachindex(log_esf)
    )
    z_new = LogExpFunctions.logsumexp(
        log_esf[k] + σ_new / p * (k - 1)^2
        for k in eachindex(log_esf)
    )
    (σ_new - σ_old) * sum_sq / p - (z_new - z_old) * k
end

function fast_loglik_ratio_σ_approx(σ_new::Real, σ_old::Real, μ::AbstractVector{<:Real}, sum_sq::Number, k::Integer, obj::AbstractCurieWeissDenominatorApprox)
    p = length(μ)
    find_mode!(obj, μ, σ_old)
    z_old = compute_log_den!(obj, μ, σ_old)
    find_mode!(obj, μ, σ_old)
    z_new = compute_log_den!(obj, μ, σ_new)
    (σ_new - σ_old) * sum_sq / p - (z_new - z_old) * k
end



function sample_cw_μ!(rng::Random.AbstractRNG, μ, σ, sum_scores_x, k, state::CurieWeissStructureInternal{
    <:Any,      #T  <: Distributions.ContinuousUnivariateDistribution,
    <:Any,      #U  <: AbstractFloat,
    <:ExactESF, #V1 <: AbstractESFMethod,
    <:Any,      #V2 <: AbstractESFMethod,
    <:Any,      #W  <: AbstractCurieWeissDenominatorApprox,
    <:Any       #X  <: CurieWeissMHStateσ
})

    prior_α = state.prior_μ_α
    prior_θ = state.prior_μ_β

    p = length(μ)
    r = trues(p)

    log_esf_all = esf_sum_log(μ)
    log_esf     = similar(log_esf_all, length(log_esf_all) - 1)

    log_ts = [σ / p * (i - 1)^2 for i in 1:p+1]
    for i in eachindex(μ)

        r[i] = false
        # log_esf   = esf_sum_log(view(μ, r))
        esf_log_drop!(log_esf, log_esf_all, μ[i])

        # log_esf_0 = .esf_sum_log(μ) # only for testing

        log_c1 = LogExpFunctions.logsumexp(log_esf[k] + log_ts[k]     for k in 1:p)
        log_c2 = LogExpFunctions.logsumexp(log_esf[k] + log_ts[k + 1] for k in 1:p)

        # @assert LogExpFunctions.logaddexp(log_c1, log_c2 + μ[i]) ≈ LogExpFunctions.logsumexp(log_esf_0[k] + log_ts[k] for k in eachindex(log_esf_0))

        # same check as the above on a non-log scale

        # esf = .esf_sum(exp.(μ[1:end .!= i]))
        # esf0 = .esf_sum(exp.(μ))
        # ts0 = [exp(σ / p * (i - 1)^2) for i in 1:p+1]

        # c1 = LinearAlgebra.dot(esf, ts0[1:end - 1])
        # c2 = LinearAlgebra.dot(esf, ts0[2:end])

        # @assert c1 + c2 * exp(μ[i]) ≈ LinearAlgebra.dot(esf0, ts0)

        log_c = log_c2 - log_c1

        sxi = sum_scores_x[i]

        α = sxi
        β = k - sxi + 1
        q = 1

        post_α = α + prior_α

        # r = rand(rng, Distributions.Gamma(β, q))
        # newvalue = rand(rng, Distributions.Gamma(α, 1/r))
        inv_r = rand(rng, Distributions.InverseGamma(β, q))
        post_inv_r  = iszero(prior_θ) ? inv_r : (inv_r * prior_θ) / (inv_r + prior_θ)

        newvalue = rand(rng, Distributions.Gamma(post_α, post_inv_r))

        μ[i] = log(newvalue) - log_c

        r[i] = true

        esf_log_add!(log_esf_all, log_esf, μ[i])


    end

    return μ


end
function sample_cw_μ!(rng::Random.AbstractRNG, μ, σ, sum_scores_x, k, state::CurieWeissStructureInternal{
    <:Any,              #T  <: Distributions.ContinuousUnivariateDistribution,
    <:Any,              #U  <: AbstractFloat,
    <:ApproximateESF,   #V1 <: AbstractESFMethod,
    <:Any,              #V2 <: AbstractESFMethod,
    <:Any,              #W  <: AbstractCurieWeissDenominatorApprox,
    <:Any               #X  <: CurieWeissMHStateσ
})

    obj     = state.obj_den_approx
    prior_α = state.prior_μ_α
    prior_θ = state.prior_μ_β
    p = length(μ)

    # Possible performance, might be better to compute the ESF from scratch
    find_mode!(obj, μ, σ)
    compute_ys!(obj, μ, σ) # needed only if σ was updated?

    for i in eachindex(μ)

        downdate!(obj, μ[i], σ, p)
        log_c1 = get_value(obj)
        log_c2 = get_shifted_value(obj, σ, p, 1)

        log_c = log_c2 - log_c1

        #=
            μᵢ ~ exp(μᵢ * sxᵢ) / (1 + c * exp(μᵢ))
            μᵢ ~ β′(α = sxi, β = k - sxi, 1, q = exp(-log_c)

        =#
        sxi = sum_scores_x[i]

        α = sxi
        β = k - sxi + 1 # NOTE: +1 is not entirely accurate!
        q = 1#exp(-log_c) # underflows, so we use the multiplicative property of the beta prima later on

        post_α      = α + prior_α

        # r = rand(rng, Distributions.Gamma(β, q))
        # newvalue = rand(rng, Distributions.Gamma(α, 1/r))
        inv_r = rand(rng, Distributions.InverseGamma(β, q))
        post_inv_r  = iszero(prior_θ) ? inv_r : (inv_r * prior_θ) / (inv_r + prior_θ)

        newvalue = rand(rng, Distributions.Gamma(post_α, post_inv_r))

        # log(exp(-log_c) * newvalue) ≈ log(newvalue) - log_c
        μ[i] = log(newvalue) - log_c

        update!(obj, μ[i], σ, p)

    end

    return μ

end

function update_proposal_variance!(mh_state::CurieWeissMHStateσ, accepted::Bool)

    # accepted is not really used, but some different schemes make use of it

    (; iteration, n_adapts, acc_target) = mh_state
    if iteration <= n_adapts

        s_σ_old = mh_state.s_σ

        acc_current = mh_state.acceptance_σ / iteration
        s_σ_new = exp(log(s_σ_old) + 10(acc_current - acc_target) / iteration)
        mh_state.s_σ = s_σ_new

        mh_state.iteration += one(mh_state.iteration)
        # @show accepted, acc_target, acc_current, s_σ_old, s_σ_new
    end
end

function sample_cw_σ(rng::Random.AbstractRNG, μ, σ, sum_sq_x, k, state::CurieWeissStructureInternal{
    <:Any,      #T  <: Distributions.ContinuousUnivariateDistribution,
    <:Any,      #U  <: AbstractFloat,
    <:Any,      #V1 <: AbstractESFMethod,
    <:ExactESF, #V2 <: AbstractESFMethod,
    <:Any,      #W  <: AbstractCurieWeissDenominatorApprox,
    <:Any       #X  <: CurieWeissMHStateσ
})

    (; πσ, rand_qσ, ratio_logpdf_qσ, s_σ, iteration, n_adapts, ϕ_σ, acc_target) = state.mh_state_σ

    σⁿ = rand_qσ(rng, σ, s_σ)

    log_esf = esf_sum_log(μ)

    log_acceptance =
        fast_loglik_ratio_σ(σⁿ, σ, sum_sq_x, log_esf, k) +
        # ratio of priors
        Distributions.logpdf(πσ, σⁿ) - Distributions.logpdf(πσ, σ) +
        # ratio of proposals
        ratio_logpdf_qσ(σ, σⁿ, s_σ)

    if rand(rng) <= exp(log_acceptance)
        # cw_state.log_const = log_const_new
        σ_new = σⁿ
        state.mh_state_σ.acceptance_σ += 1
    else
        σ_new = σ
    end

    update_proposal_variance!(state.mh_state_σ, σ_new == σⁿ)

    return σ_new

end

function sample_cw_σ(rng::Random.AbstractRNG, μ, σ, sum_sq_x, k, state::CurieWeissStructureInternal{
    <:Any,            #T  <: Distributions.ContinuousUnivariateDistribution,
    <:Any,            #U  <: AbstractFloat,
    <:Any,            #V1 <: AbstractESFMethod,
    <:ApproximateESF, #V2 <: AbstractESFMethod,
    <:Any,            #W  <: AbstractCurieWeissDenominatorApprox,
    <:Any             #X  <: CurieWeissMHStateσ
})

    obj = state.obj_den_approx
    (; πσ, rand_qσ, ratio_logpdf_qσ, s_σ, iteration, n_adapts, ϕ_σ, acc_target) = state.mh_state_σ

    σⁿ = rand_qσ(rng, σ, s_σ)

    log_acceptance =
        fast_loglik_ratio_σ_approx(σⁿ, σ, μ, sum_sq_x, k, obj) +
        # ratio of priors
        Distributions.logpdf(πσ, σⁿ) - Distributions.logpdf(πσ, σ) +
        # ratio of proposals
        ratio_logpdf_qσ(σ, σⁿ, s_σ)

    if rand(rng) <= exp(log_acceptance)
        # cw_state.log_const = log_const_new
        σ_new = σⁿ
        state.mh_state_σ.acceptance_σ += 1
    else
        σ_new = σ
    end

    update_proposal_variance!(state.mh_state_σ, σ_new == σⁿ)

    return σ_new
end

function sample_curie_weiss(
    x::AbstractMatrix{<:Integer},
    structure::CurieWeissStructure = CurieWeissStructure();
    # MCMC parameters
    n_iter::Integer         = 1000,
    n_warmup::Integer       = min(1000, n_iter ÷ 2),
    n_thinning::Integer     = 1,
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool           = true
)

    all(y -> iszero(y) || isone(y), x) || throw(ArgumentError("x must be a matrix with binary values (0 or 1)"))
    n_iter     >  zero(n_iter)          || throw(ArgumentError("n_iter must be a positive integer"))
    n_warmup   >= zero(n_warmup)        || throw(ArgumentError("n_warmup must be a nonnegative integer"))
    n_thinning >  zero(n_thinning)      || throw(ArgumentError("n_thinning must be a positive integer"))

    p, k = size(x)
    sum_scores_x = vec(sum(x, dims = 2))
    sum_sq_x     = sum(abs2 ∘ sum, eachcol(x))

    internal_structure = to_internal(structure, rng, p, k, n_warmup, false)

    cw_sample = initialize_group_state(rng, structure, p, k, true)
    μs = cw_sample.μ
    σs = cw_sample.σ

    samples = zeros(p + 1, n_iter)

    prog = ProgressMeter.Progress(size(samples, 2), showspeed = true)
    start_time = time()
    prog = ProgressMeter.Progress(n_iter + n_warmup; enabled = verbose, showspeed = true)
    for i in 1:n_iter + n_warmup

        for _ in 1:n_thinning

                 sample_cw_μ!(rng, μs, σs, sum_scores_x, k, internal_structure)
            σs = sample_cw_σ( rng, μs, σs, sum_sq_x,     k, internal_structure)

        end

        if i > n_warmup
            j = i - n_warmup
            samples[eachindex(μs), j] .= μs
            samples[p+1,           j]  = σs

            ProgressMeter.next!(prog)

        end

        ProgressMeter.next!(prog)

    end
    stop_time = time()

    return samples

end

