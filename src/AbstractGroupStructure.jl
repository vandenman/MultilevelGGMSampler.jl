abstract type AbstractGroupStructure end

# priors are multivariat and have length of parameters
abstract type AbstractGroupStructureInternal end

struct NoGroupStructure <: AbstractGroupStructureInternal end

struct IndependentStructure <: AbstractGroupStructure
end
struct IndependentStructureInternal <: AbstractGroupStructureInternal
end
to_internal(::IndependentStructure, ::Random.AbstractRNG, ::Integer, ::Integer, ::Integer) = IndependentStructureInternal()

abstract type AbstractESFMethod end
struct ExactESF <: AbstractESFMethod
end
struct ApproximateESF <: AbstractESFMethod
end

Base.@kwdef mutable struct CurieWeissMHStateσ{
    T <: Distributions.ContinuousUnivariateDistribution,
    U, V
}# <: AbstractSamplingMethodInternal
    const   πσ              ::T                 = Distributions.LogNormal()
    const   rand_qσ         ::U                 = rand_qσ
    const   ratio_logpdf_qσ ::V                 = ratio_logpdf_qσ
    const   n_adapts        ::Int
            iteration       ::Int               = 1
            acceptance_σ    ::Int               = 0
            s_σ             ::Float64           = 0.2
    const   ϕ_σ             ::Float64           = 0.75
    const   acc_target      ::Float64           = 0.234#0.15 # 0.234
end

Base.@kwdef struct CurieWeissStructure{
    T  <: Distributions.ContinuousUnivariateDistribution,
    U  <: AbstractFloat,
    V  <: Union{Nothing, AbstractVector{<:Real}},
    W  <: Union{Nothing, Real},
    Y1 <: AbstractESFMethod,
    Y2 <: AbstractESFMethod,
} <: AbstractGroupStructure
    # prior
    πσ::T = Distributions.truncated(Distributions.Normal(), lower = 0) # halfnormal
    prior_μ_α::U = 1.0
    prior_μ_β::U = 1.0
    # state / initial values
    μ::V  = nothing
    σ::W  = nothing
    # ESF method
    μ_esf_method::Y1 = ApproximateESF()
    σ_esf_method::Y2 = ApproximateESF()
end

struct CurieWeissStructureInternal{
    T  <: Distributions.ContinuousUnivariateDistribution,
    U  <: AbstractFloat,
    V1 <: AbstractESFMethod,
    V2 <: AbstractESFMethod,
    W  <: AbstractCurieWeissDenominatorApprox,
    X  <: CurieWeissMHStateσ
} <: AbstractGroupStructureInternal
    # same as CurieWeissStructure
    # prior
    πσ::T
    prior_μ_α::U
    prior_μ_β::U
    # ESF method
    μ_esf_method::V1
    σ_esf_method::V2
    # storage for compute_suffstats_groupstate
    Gs_mat::Matrix{Int}
    sum_k::Vector{Int}
    # approximation
    obj_den_approx::W
    # adaptive MH
    mh_state_σ::X
end

function to_internal(s::CurieWeissStructure, rng::Random.AbstractRNG, p::Integer, k::Integer, n_adapts::Integer, allocate_extra::Bool = true)

    if allocate_extra
        ne = p_to_ne(p)
        Gs_mat = Matrix{Int}(undef, ne, k)
        sum_k  = Vector{Int}(undef, k)
    else
        ne = p
        Gs_mat = Matrix{Int}(undef, 0, 0)
        sum_k  = Vector{Int}(undef, 0)
    end

    obj_den_approx = CurieWeissDenominatorApprox(default_no_weights(ne))
    mh_state_σ = CurieWeissMHStateσ(; n_adapts = n_adapts)
    CurieWeissStructureInternal(s.πσ, s.prior_μ_α, s.prior_μ_β, s.μ_esf_method, s.σ_esf_method, Gs_mat, sum_k, obj_den_approx, mh_state_σ)

end


Base.@kwdef struct BernoulliStructure{T<:Real} <: AbstractGroupStructure
    p::T = 0.5
end


function dist_len(dist::Distributions.UnivariateDistribution, desired_length::Integer)
    # identical to DistributionsAD.filldist(dist, desired_length)
    # but avoids a dependency on DistributionsAD
    Distributions.product_distribution(FillArrays.Fill(dist, desired_length))
end
function dist_len(dist::Distributions.MultivariateDistribution, desired_length::Integer)
    @assert length(dist) == desired_length
    return dist
end

if_nothing_rand(rng, d, x) = isnothing(x) ? rand(rng, d) : x
