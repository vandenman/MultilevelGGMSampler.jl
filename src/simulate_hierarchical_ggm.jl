#=

    Possible optimization

    do not put the individual level and group level stuff in one struct

abstract type AbstractGroupLevelParameters end
abstract type AbstractIndividualParameters end

struct Parameters{T, U} where {T<:AbstractGroupLevelParameters, AbstractIndividualParameters
    group::T
    individual::U
end

and then some helper dispatches so that you can do act on Parameters and will automatically get the stuff in group or individual
=#

abstract type AbstractParameters end

struct CurieWeissParameters{T<:Real} <: AbstractParameters
    K::Array{T, 3}
    G::BitArray{3}
    μ::Vector{T}
    σ::T
end

struct BernoulliParameters{T<:Real, U} <: AbstractParameters
    K::Array{T, 3}
    G::BitArray{3}
    p::U
end

simulate_hierarchical_ggm(nₖ::Integer, args...) = simulate_hierarchical_ggm(Random.default_rng(), nₖ, args...)

function simulate_hierarchical_ggm(
    rng::Random.AbstractRNG,
    nₖ::T, p::T, k::T,
    # Possible feature: allow for passing specific parameters! perhaps use initial values?
    posterior::GWishart,
    group_structure::AbstractGroupStructure,
    raw_observations::Bool = true
) where {T<:Integer}


    ne = p_to_ne(p)

    df, rate, _ = Distributions.params(posterior)

    dG, graph_parameters = group_structure_to_dist(rng, group_structure, ne, k)
    if dG isa AbstractGraphDistribution
        Gflat = rand(rng, dG, k)
        G = falses(p, p, k)
        for ik in axes(G, 3)
            # G[:, :, ik] = tril_vec_to_sym(view(Gflat, :, ik), -1)
            tril_vec_to_sym!(view(G, :, :, ik), view(Gflat, :, ik), -1)
        end
    elseif group_structure isa BernoulliStructure
        Gflat = rand(rng, dG)
        G = falses(p, p, k)
        for ik in axes(G, 3)
            tril_vec_to_sym!(view(G, :, :, ik), view(Gflat, :, ik), -1)
        end
    else
        G = rand(rng, dG)
    end

    @assert all(LinearAlgebra.issymmetric, eachslice(G, dims=3))

    K = Array{Float64}(undef, p, p, k)

    if raw_observations
        data = similar(K, p, nₖ, k)
    else
        data = similar(K, p, p, k)
    end

    for ik in axes(G, 3)

        K[:, :, ik] = rand(rng, GWishart(df, rate, Graphs.SimpleGraph(G[:, :, ik])))

        Σ = PDMats.PDMat(LinearAlgebra.inv(LinearAlgebra.Symmetric(K[:, :, ik])))

        if raw_observations
            Distributions.rand!(rng, Distributions.MvNormal(Σ),    view(data, :, :, ik))
        else
            Distributions.rand!(rng, Distributions.Wishart(nₖ, Σ), view(data, :, :, ik))
        end
    end

    if group_structure isa CurieWeissStructure# || group_structure isa CurieWeissHierarchicalStructure
        parameters = CurieWeissParameters(K, G, graph_parameters.μ, graph_parameters.σ)
    elseif group_structure isa BernoulliStructure || group_structure isa IndependentStructure
        parameters = BernoulliParameters(K, G, graph_parameters)
    else
        throw(ArgumentError("Unsupported argument passed to `group_structure`"))
    end

    return (; data, #=dims,=# parameters)

end

group_structure_to_dist(s::AbstractGroupStructure, ne::Integer, k::Integer) = group_structure_to_dist(Random.default_rng(), s, ne, k)

"""
Sample a Generalized beta prime random variable using the decomposition in

https://en.wikipedia.org/wiki/Beta_prime_distribution#Compound_gamma_distribution

with α = prior_α and β = prior_θ. The scale parameter q is set to 1.0.

The process is repeated `ne` times.
"""
rand_betaprime(prior_α, prior_θ, ne::Integer) = rand_betaprime(Random.default_rng(), prior_α, prior_θ, ne)
function rand_betaprime(rng::Random.AbstractRNG, prior_α, prior_θ, ne::Integer)

    α = 0.0
    β = 0.01

    q = 1.0
    post_α = α + prior_α
    result = Vector{Float64}(undef, ne)
    safety = 1_000

    sampler_inverse_gamma = Distributions.sampler(Distributions.InverseGamma(β, q))
    for i in eachindex(result)

        post_inv_r = zero(eltype(sampler_inverse_gamma))
        for j in 1:safety
            inv_r = rand(rng, sampler_inverse_gamma)
            post_inv_r  = iszero(prior_θ) ? inv_r : (inv_r * prior_θ) / (inv_r + prior_θ)
            if !isnan(post_inv_r) && isfinite(post_inv_r) && zero(post_inv_r) < post_inv_r
                if !(post_inv_r >= zero(post_inv_r) && post_α >= zero(post_α))
                    @show post_α, post_inv_r, β, q, inv_r
                end
                newvalue = rand(rng, Distributions.Gamma(post_α, post_inv_r))
                result[i] = log(newvalue)
                break
            elseif j == safety
                throw(ArgumentError("Failed to simulate initial values for `μ` from the prior. Try providing values directly."))
            end
        end

    end
    return result
end

function group_structure_to_dist(rng::Random.AbstractRNG, s::CurieWeissStructure, ne::Integer, ::Integer)
    μ = isnothing(s.μ) ? rand_betaprime(rng, s.prior_μ_α, s.prior_μ_β, ne)  : s.μ
    σ = isnothing(s.σ) ? rand(rng, s.πσ)               : s.σ
    dG = CurieWeissDistribution(μ, σ)
    return dG, (;μ, σ)
end

function group_structure_to_dist(rng::Random.AbstractRNG, s::AbstractGroupStructure, ne::Integer, ::Integer)
    μ = isnothing(s.μ) ? rand_dist_len(rng, s.πμ, ne)            : s.μ
    σ = isnothing(s.σ) ? rand_dist_len(rng, s.πσ, p_to_ne(ne))   : s.σ
    dG = to_dist(s, μ, σ)
    return dG, (;μ, σ)
end

function group_structure_to_dist(::Random.AbstractRNG, s::BernoulliStructure, ne::Integer, k::Integer)
    if s.probs isa Number
        d = Distributions.product_distribution([Distributions.Bernoulli(s.probs) for i in 1:ne, j in 1:k])
    else
        if s.probs isa AbstractVector
            d = Distributions.product_distribution([Distributions.Bernoulli(s.probs[i]) for i in 1:ne, j in 1:k])
        else
            d = Distributions.product_distribution([Distributions.Bernoulli(s.probs[i, j]) for i in 1:ne, j in 1:k])
        end
    end
    return d, s.probs
end

function group_structure_to_dist(::Random.AbstractRNG, s::IndependentStructure, ne::Integer, k::Integer)
    probs = [rand() for i in 1:ne, j in 1:k]
    d = Distributions.product_distribution([Distributions.Bernoulli(probs[i, j]) for i in 1:ne, j in 1:k])
    return d, s.probs
end


function σ_to_Σ(p::Integer, σ::AbstractVector)
    Σ = zeros(eltype(σ), p, p)
    c = 1
    @inbounds for j in 1:size(Σ, 1) - 1, i in j+1:size(Σ, 1)
        Σ[i, j] = σ[c]
        Σ[j, i] = Σ[i, j]
        c += 1
    end
    return Σ
end
σ_to_Σ(μ::AbstractVector, σ::AbstractVector) = σ_to_Σ(length(μ), σ)
σ_to_Σ(σ::AbstractVector)                    = σ_to_Σ(ne_to_p(length(σ)), σ)
Σ_to_σ(Σ::AbstractMatrix)                    = tril_to_vec(Σ, -1)

function rand_dist_len(rng::Random.AbstractRNG, dist::Distributions.UnivariateDistribution, desired_length::Integer)
    rand(rng, dist, desired_length)
end
function rand_dist_len(rng::Random.AbstractRNG, dist::Distributions.MultivariateDistribution, desired_length::Integer)
    @assert length(dist) == desired_length
    rand(rng, dist)
end


rand_σ(rng::Random.AbstractRNG, πσ, ne::Integer) = rand_dist_len(rng, πσ, ne)

rand_σ(rng::Random.AbstractRNG, πσ::Distributions.UnivariateDistribution,   ne::Integer) = rand(rng, πσ, ne)
rand_σ(rng::Random.AbstractRNG, πσ::Distributions.MultivariateDistribution,   ::Integer) = rand(rng, πσ)