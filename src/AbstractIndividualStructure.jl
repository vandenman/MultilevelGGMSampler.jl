abstract type AbstractIndividualStructure end
abstract type AbstractIndividualStructureInternal end
# abstract type AbstractSpikeAndSlabStructure <: AbstractIndividualStructure end
# abstract type AbstractSpikeAndSlabStructureInternal <: AbstractIndividualStructureInternal end

Base.@kwdef struct GWishartStructure{T<:Real, U<:Union{AbstractMatrix{T}, LinearAlgebra.UniformScaling}} <: AbstractIndividualStructure
    prior_df        ::T    = 3.0
    prior_rate      ::U    = LinearAlgebra.I
    delayed_accept  ::Bool = true
    loc_bal         ::Bool = true
    Letac           ::Bool = true
    orig            ::Bool = false
    threaded        ::Bool = false
end

struct GWishartStructureInternal{T<:Real} <: AbstractIndividualStructureInternal
    prior_df        ::T
    post_df         ::T
    post_rate_vec   ::Vector{PDMats.PDMat{T, Matrix{T}}}
    delayed_accept  ::Bool
    loc_bal         ::Bool
    Letac           ::Bool
    orig            ::Bool
    threaded        ::Bool
end

Base.eltype(::GWishartStructureInternal{T}) where T = T

function to_internal(s::GWishartStructure, data)

    n, _, _, sum_of_squares = data
    T = eltype(sum_of_squares)

    prior_df = s.prior_df
    post_df = prior_df + n
    prior_rate = s.prior_rate
    post_rate_vec = Vector{PDMats.PDMat{T, Matrix{T}}}(undef, size(sum_of_squares ,3))
    for ik in axes(sum_of_squares, 3)
        post_rate_vec[ik] = PDMats.PDMat(T.(sum_of_squares[:, :, ik] + prior_rate))
    end

    return GWishartStructureInternal(prior_df, post_df, post_rate_vec, s.delayed_accept, s.loc_bal, s.Letac, s.orig, s.threaded)

end


abstract type SS_SamplingMethod end
struct DirectSampling   <: SS_SamplingMethod end
struct CholeskySampling <: SS_SamplingMethod end

abstract type SS_InversionMethod end
struct Direct_Inv <: SS_InversionMethod end
struct CG_Inv     <: SS_InversionMethod end

Base.@kwdef struct SpikeAndSlabStructure{T<:Real, U<:Union{T, AbstractMatrix{T}}, V<:Union{T, AbstractVector{T}}, W<:SS_SamplingMethod, X <: SS_InversionMethod} <: AbstractIndividualStructure
    σ_spike     ::U    = 0.1
    σ_slab      ::U    = 10.0
    λ           ::V    = 0.1
    threaded    ::Bool = false
    method      ::W    = CholeskySampling()#DataAugmentation()
    inv_method  ::X    = CG_Inv()
end

struct SpikeAndSlabStructureInternal{T<:Real, U<:AbstractMatrix{T}, V<:AbstractVector{T}, W<:SS_SamplingMethod, X <: SS_InversionMethod} <: AbstractIndividualStructureInternal
    σ_spike         ::U
    σ_slab          ::U
    λ               ::V
    sum_of_squares  ::Array{T, 3}
    n               ::Int
    threaded        ::Bool
    method          ::W
    inv_method      ::X
    σ_helper2       ::Matrix{T}
    σ_log_ratio     ::Matrix{T}
end
Base.eltype(::SpikeAndSlabStructureInternal{T}) where T = T

function compute_σ_helper2_and_σ_log_ratio(σ_spike::AbstractMatrix{T}, σ_slab::AbstractMatrix) where T
    p = size(σ_spike, 1)
    σ_helper2   = Matrix{T}(undef, p, p)
    σ_log_ratio = Matrix{T}(undef, p, p)
    for j in axes(σ_helper2, 2)
        for i in axes(σ_helper2, 1)
           σ_helper2[i, j]   = (inv(abs2(σ_spike[i, j])) - inv(abs2(σ_slab[i, j]))) / 2
           σ_log_ratio[i, j] = log(σ_spike[i, j] / σ_slab[i, j])
        end
    end
    return σ_helper2, σ_log_ratio
end

function to_internal(s::SpikeAndSlabStructure, data)

    n, p, k, sum_of_squares = data
    T = eltype(sum_of_squares)

    σ_spike = size(s.σ_spike)  == (p, p) ? T.(s.σ_spike) : FillArrays.Fill(T(s.σ_spike), p, p)
    σ_slab  = size(s.σ_slab)   == (p, p) ? T.(s.σ_slab)  : FillArrays.Fill(T(s.σ_slab),  p, p)

    σ_helper2, σ_log_ratio = compute_σ_helper2_and_σ_log_ratio(σ_spike, σ_slab)

    return SpikeAndSlabStructureInternal(
        σ_spike,
        σ_slab,
        size(s.λ)       ==  p     ? T.(s.λ)       : FillArrays.Fill(T(s.λ), p),
        sum_of_squares,
        n,
        s.threaded,
        s.method,
        s.inv_method,
        σ_helper2,
        σ_log_ratio
    )
end


# Base.@kwdef struct HorseshoeStructure{T<:Real, U<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractSpikeAndSlabStructure
#     local_shrinkage::U = 1.0
#     global_shrinkage::V = 1.0
# end

# struct HorseshoeStructureInternal{T<:Real, U<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractIndividualStructureInternal
#     local_shrinkage::U
#     global_shrinkage::V
#     sum_of_squares::Array{T, 3}
#     n::Int
# end
# Base.eltype(::HorseshoeStructureInternal{T}) where T = T


# function to_internal(s::HorseshoeStructure, data)
#     n, p, k, sum_of_squares = data
#     T = eltype(sum_of_squares)

#     return HorseshoeStructureInternal(
#         maybe_fill(s.local_shrinkage,  (p, p), T, "local_shrinkage"),
#         maybe_fill(s.global_shrinkage, k,      T, "global_shrinkage"),
#         # size(s.local_shrinkage)  == (p, p) ? T.(s.local_shrinkage)  : FillArrays.Fill(T(s.local_shrinkage),  p, p),
#         # size(s.global_shrinkage) ==  k     ? T.(s.global_shrinkage) : FillArrays.Fill(T(s.global_shrinkage), p),
#         data.sum_of_squares,
#         data.n
#     )
# end

function maybe_fill(x, expected_size, T, name)
    size(x) == expected_size && return T.(x)
    x isa Real               && return FillArrays.Fill(T(s.local_shrinkage),  p, p)
    throw(ArgumentError("invalid input for $name"))
end