"""
Computes the prior inclusion probability for including the edge from i to j conditional on the graph G and the prior distribution d.
Fallbacks exist for `DiscreteUnivariateDistribution` and `DiscreteMatrixDistribution`.
For matrix distributions it is advised to add a custom dispatch if an efficient simplification exists because the default generic implementation needs to make a copy of G.
"""
function log_inclusion_prob_prior_G(d::Distributions.DiscreteMatrixDistribution, G::AbstractMatrix{T}, i::Integer, j::Integer, args...) where T<:Integer
    G0 = copy(G) # annoying but necessary in the general case
    G0[i, j] = one(T)
    G0[j, i] = one(T)
    result = Distributions.logpdf(d, G0)
    G0[i, j] = zero(T)
    G0[j, i] = zero(T)
    result -= Distributions.logpdf(d, G0)
    return result
end

function log_inclusion_prob_prior_G(d::Distributions.Distribution{Distributions.ArrayLikeVariate{3}, Distributions.Discrete}, G::AbstractArray{T, 3}, i::Integer, j::Integer, k::Integer) where T<:Integer
    G0 = copy(G) # annoying but necessary in the general case
    G0[i, j, k] = one(T)
    G0[j, i, k] = one(T)
    result = Distributions.logpdf(d, G0)
    G0[i, j, k] = zero(T)
    G0[j, i, k] = zero(T)
    result -= Distributions.logpdf(d, G0)
    return result
end

function log_inclusion_prob_prior_G(d::AbstractGraphDistribution, G::AbstractArray{T, 3}, i::Integer, j::Integer, k::Integer) where T<:Integer

    Gvec = tril_to_vec(view(G, :, :, k))
    e_id = triangle_indices_to_linear_index(i, j, size(G, 1))
    return log_conditional_prob(one(k), e_id, d, Gvec)

end

# univariate assumes that π(G) = prod(π(Gᵢⱼ) for (i, j) in LowerTriangle(G))
function log_inclusion_prob_prior_G(d::Distributions.DiscreteUnivariateDistribution, args...)
    return Distributions.logpdf(d, 1) - Distributions.logpdf(d, 0)
end

function log_inclusion_prob_prior_G(d::Distributions.Bernoulli, args...)
    return LogExpFunctions.logit(d.p)
end

# This struct is a trick to avoid passing information about groups to update_G_wwa!!
# struct ConditionalGraphDistribution{T<:Integer, S<:AbstractArray{<:Integer, 3}, D<:Union{AbstractGraphDistribution, AbstractHierarchicalGraphDistribution}} <: Distributions.DiscreteMatrixDistribution
struct ConditionalGraphDistribution{T<:Integer, S<:AbstractArray{<:Integer, 3}, D<:AbstractGraphDistribution} <: Distributions.DiscreteMatrixDistribution
    k::T
    Gs::S
    d::D
end

function log_inclusion_prob_prior_G(d::ConditionalGraphDistribution, ::AbstractMatrix{<:Integer}, i::Integer, j::Integer)
    log_inclusion_prob_prior_G(d.d, d.Gs, i, j, d.k)
end

function log_inclusion_prob_prior_G(d::ConditionalGraphDistribution, ::AbstractMatrix{<:Integer}, i::Integer, j::Integer, suffstats_state::Vector{Integer})
    log_inclusion_prob_prior_G(d.d, d.Gs, i, j, d.k, suffstats_state)
end

function log_inclusion_prob_prior_G(d::CurieWeissDistribution, Gs::AbstractMatrix{<:Integer}, i::Integer, j::Integer, suffstats_state::Integer)
    # TODO: triangle_indices_to_linear_index is incorrect!
    # NOTE: this is unreachable at the moment, delete the method?
    log_conditional_prob(Gs[i, j], triangle_indices_to_linear_index(i, j, size(Gs, 1)), d, suffstats_state - Gs[i, j])
end

function log_inclusion_prob_prior_G(::IndependentGraphDistribution, ::AbstractMatrix{<:Integer}, ::Integer, ::Integer, suffstats_state::Union{Integer, Nothing})
    return 0.0
end

function logit_inclusion_prob_prior_G(d::CurieWeissDistribution, e_idx::Integer, suffstats_state::Integer, p::Integer)
    μ, σ = Distributions.params(d)
    return μ[e_idx] + σ * (1 + 2 * suffstats_state) / length(μ)
end

function conditional_graph_distribution(d::AbstractGraphDistribution, Gs::AbstractArray{<:Integer, 3}, k::Integer)
    return ConditionalGraphDistribution(k, Gs, d)
end

function conditional_graph_distribution(d::CurieWeissDistribution, ::AbstractArray{<:Integer, 3}, ::Integer)
    return d
end

function conditional_graph_distribution(d::IndependentGraphDistribution, ::AbstractArray{<:Integer, 3}, ::Integer)
    return d
end

function conditional_graph_distribution(d::Distributions.DiscreteDistribution, ::AbstractArray{<:Integer, 3}, ::Integer)
    return d
end

