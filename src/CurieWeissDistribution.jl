struct CurieWeissDistribution{T<:Real, U<:Real, V<:AbstractVector{<:U}} <: AbstractGraphDistribution
    μ::V
    σ::T
end


struct CurieWeissSampler{T, U, V, W} <: AbstractGraphSampler
    d::CurieWeissDistribution{T, U, V}
    log_const::W
end


function Distributions.sampler(d::CurieWeissDistribution{T}) where T
    return CurieWeissSampler(d, log_const(d))
end


Base.length(d::CurieWeissSampler) = length(d.d)
Base.eltype(d::CurieWeissSampler) = eltype(d.d)


Base.length(d::CurieWeissDistribution) = length(d.μ)
Base.eltype(::CurieWeissDistribution) = Int

Distributions.params(d::CurieWeissDistribution) = (d.μ, d.σ)

function Distributions.logpdf(d::CurieWeissDistribution, x::AbstractVector)
    return logpdf_prop(d, x) - log_const(d)
end

function Distributions.loglikelihood(d::CurieWeissDistribution, x::AbstractMatrix)

    μ, σ = Distributions.params(d)

    log_den = log_const(d)

    log_num = σ * sum(abs2 ∘ sum, eachcol(x)) / length(μ)
    for j in eachindex(μ)
        log_num += μ[j] * sum(view(x, j, :))
    end

    return log_num - log_den * size(x, 2)
end

function loglikelihood_suffstats(d::CurieWeissDistribution, sum_scores, sum_sq, k)

    μ, σ = Distributions.params(d)

    log_den = log_const(d)

    log_num = σ * sum_sq / length(μ) + LinearAlgebra.dot(μ, sum_scores)

    return log_num - log_den * k
end

"""
Same as `loglikelihood_suffstats` but drops everything constant w.r.t. μ
"""
function loglikelihood_suffstats_σ(d::CurieWeissDistribution, sum_sq, k)

    μ, σ = Distributions.params(d)

    log_den = log_const(d)

    log_num = σ * sum_sq / length(μ)

    return log_num - log_den * k
end

"""
Proportional to `logpdf(d, x)`, dropping any normalizing constants.
"""
function logpdf_prop(d::CurieWeissDistribution, x::AbstractVector)
    μ, σ = Distributions.params(d)
    # return LinearAlgebra.dot(x, μ) + σ * sum(x)^2
    return LinearAlgebra.dot(x, μ) + σ * abs2(sum(x)) / length(μ)
end

"""
The normalizing constant of `d`.
"""
function log_const(d::CurieWeissDistribution)

    μ, σ = Distributions.params(d)
    p = length(μ)
    # esf_values = esf_sum(exp.(μ))
    log_esf_values = esf_sum_log(μ)
    return LogExpFunctions.logsumexp(
        log_esf_values[i] + σ * (i - 1)^2 / p
        for i in eachindex(log_esf_values)
    )
    # brute force approach
    # return LogExpFunctions.logsumexp(LinearAlgebra.dot(g, μ) + σ * sum(g)^2 for g in BinarySpaceIterator(length(μ)))
end

function log_conditional_prob(gⱼ::Integer, j::Integer, d::CurieWeissDistribution, x::AbstractVector{<:Integer})

    #=
        The logarithm of equation 5.4 in
        Marsman, M., Tanis, C. C., Bechger, T. M., & Waldorp, L. J. (2019).
        Network psychometrics in educational practice: Maximum likelihood estimation of the Curie-Weiss model.
        Theoretical and practical advances in computer-based educational measurement, 93-120.
    =#
    return log_conditional_prob(gⱼ, j, d, sum_all_except_j(x, j)) # ?
end

function log_inclusion_prob(k::Integer, d::CurieWeissDistribution, x::AbstractVector{<:Integer})
    log_conditional_prob(one(k), k, d, x)
end

function sum_all_except_j(x::AbstractVector{<:Integer}, j, init = zero(eltype(x)))
    @inbounds for i in 1:j-1 init += x[i]
    end
    @inbounds for i in j+1:length(x) init += x[i]
    end
    return init
end

function log_conditional_prob(gⱼ::Integer, j::Integer, d::CurieWeissDistribution, s::Integer)

    #=
        basically the same as the other log_conditional_prob but accepts as s the sum of all values except the jth value
    =#

    μ, σ = Distributions.params(d)
    p = length(μ)

    z = μ[j] + σ * (1 + 2s) / p

    return gⱼ * z - LogExpFunctions.log1pexp(z)
end

function log_inclusion_prob(k::Integer, d::CurieWeissDistribution, s::Integer)
    log_conditional_prob(one(k), k, d, s)
end

function compute_log_num_gradient!(result::AbstractVector, x::AbstractMatrix, μ::AbstractVector)
    for i in axes(x, 1)
        result[i] += sum(view(x, i, :))
    end
    result[end] += sum(abs2 ∘ sum, eachcol(x)) / length(μ)
    return result
end

function compute_log_den_gradient!(result::AbstractVector, x::AbstractMatrix, μ::AbstractVector, σ::Number)

    k = size(x, 2)
    exp_μ = exp.(μ)

    esf_values = similar(μ, length(μ) + 1)
    esf_values_no_i = similar(μ)

    esf_sum!(esf_values, exp_μ)
    logden = LogExpFunctions.logsumexp(
        log(esf_values[i]) + σ * (i - 1)^2 / length(μ)
        for i in eachindex(esf_values)
    )

    terms = [
        (i - 1)^2 / length(μ)
        for i in eachindex(esf_values)
    ]

    lognum_σ = LogExpFunctions.logsumexp(
        log(terms[i]) + log(esf_values[i]) + σ * terms[i]
        for i in eachindex(esf_values)
    )

    result[lastindex(result)] -= k * exp(lognum_σ - logden)

    r = trues(length(μ))
    for i in eachindex(μ)
        r[i] = false
        esf_sum!(esf_values_no_i, view(exp_μ, r))
        lognum_μ = LogExpFunctions.logsumexp(
            log(esf_values_no_i[i]) + σ * i^2 / length(μ)
            for i in eachindex(esf_values_no_i)
        )
        result[i] -= k * exp(μ[i] + lognum_μ - logden)
        r[i] = true
    end

    return result

end


function Distributions._rand!(rng::Random.AbstractRNG, d::T, x::AbstractVector) where
    T<:Union{CurieWeissDistribution, CurieWeissSampler}
    if length(d) <= 10
        rand_exhaustive!(rng, d, x)
    else
        rand_Gibbs!(rng, d, x)
    end
end

function rand_exhaustive!(rng::Random.AbstractRNG, d::AbstractGraphDistribution, x::AbstractVector)
    # Possible optimization: this can be done without collect-ing BinarySpaceIterator,
    # there is a way to obtain the nth vector directly from the index
    # using bitstring tricks

    # non allocating version
    # u = rand(rng)
    # s = zero(u)
    # for state in BinarySpaceIterator(k)
    #     s += Distributions.pdf(d, state)
    #     if s >= u
    #         x .= state
    #         return x
    #     end
    # end
    # fill!(x, one(eltype(x)))
    # return x

    k = length(d)
    states = collect(BinarySpaceIterator(k))
    probs = Distributions.pdf.(Ref(d), states)
    index = rand(rng, Distributions.Categorical(probs))
    x .= states[index]
    return x
end

function rand_exhaustive!(rng::Random.AbstractRNG, d::AbstractGraphSampler, x::AbstractVector)
    k = length(d.d)
    states = collect(BinarySpaceIterator(k))
    probs = exp.(logpdf_prop.(Ref(d.d), states) .- d.log_const)
    index = rand(rng, Distributions.Categorical(probs))
    x .= states[index]
    return x
end


function rand_Gibbs!(rng::Random.AbstractRNG, d::AbstractGraphDistribution, x::AbstractVector)

    for i in eachindex(x)
        x[i] = rand(rng, 0:1)
    end

    @inbounds for _ in 1:250
        for j in eachindex(x)
            x[j] = rand(rng) <= exp(log_inclusion_prob(j, d, x)) ? 1 : 0
        end
    end

    return x
end

rand_Gibbs!(rng::Random.AbstractRNG, d::AbstractGraphSampler, x::AbstractVector) = rand_Gibbs!(rng, d.d, x)

# function rand_Gibbs!(rng::Random.AbstractRNG, d::Union{CurieWeissDistribution, CurieWeissScaledDistribution}, x::AbstractVector)
function rand_Gibbs!(rng::Random.AbstractRNG, d::CurieWeissDistribution, x::AbstractVector)

    # this version caches the sum score in s

    @inbounds for i in eachindex(x)
        x[i] = rand(rng, 0:1)
    end

    s = sum(x)
    @inbounds for _ in 1:250
        for j in eachindex(x)
            if rand(rng) <= exp(log_inclusion_prob(j, d, s))
                s += 1 - x[j]
                x[j] = 1
            else
                s -= x[j]
                x[j] = 0
            end
            # x[j] = rand(rng) <= exp(log_inclusion_prob(j, d, x)) ? 1 : 0
        end
    end

    return x
end

"""
Analytically compute the marginal probabilities that x[i] is one for all x
"""
function compute_log_marginal_probs(d::CurieWeissDistribution)

    μ, σ = Distributions.params(d)
    p = length(μ)

    log_esf_values = esf_sum_log(μ)

    log_den = LogExpFunctions.logsumexp(log_esf_values[k + 1] + σ / p * k^2 for k in 0:length(μ))

    result = similar(μ)

    log_esf_values_dropped = similar(μ)

    for i in eachindex(μ)
        esf_log_drop!(log_esf_values_dropped, log_esf_values, μ[i])
        log_num = LogExpFunctions.logsumexp(log_esf_values_dropped[k + 1] + σ / p * k^2 for k in 0:p - 1)
        result[i] = LogExpFunctions.log1mexp(log_num - log_den)
    end
    return result
end

function compute_log_marginal_probs_approx(d::CurieWeissDistribution)

    μ, σ = Distributions.params(d)
    p = length(μ)

    obj = CurieWeissDenominatorApprox(default_no_weights(length(μ)))
    find_mode!(obj, μ, σ)
    log_den = compute_log_den!(obj, μ, σ)

    result = similar(μ)

    rr = trues(p)
    for i in eachindex(μ)
        # possible perforamnce: downdate! and update! used to be used, but compute_log_den!
        # calls compute_log_ys! anyway which defeats the purpose of the update functions

        # downdate!(obj, μ[i], σ, p)

        rr[i] = false
        μv = view(μ, rr)

        find_mode!(obj, μv, σ)
        log_num = compute_log_den!(obj, μv, σ, p)

        rr[i] = true
        # update!(obj, μ[i], σ, p)

        result[i] = LogExpFunctions.log1mexp(log_num - log_den)
    end
    return result
end
