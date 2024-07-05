abstract type AbstractGraphDistribution <: Distributions.DiscreteMultivariateDistribution end
abstract type AbstractGraphSampler      <: Distributions.Sampleable{Distributions.Multivariate, Distributions.Discrete} end

function log_inclusion_prob(k::Integer, d::AbstractGraphDistribution, g::AbstractVector)
    return log_conditional_prob(one(k), k, d, g)
end

struct IndependentGraphDistribution <: AbstractGraphDistribution
end
