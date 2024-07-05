#=

    TODO:
        - Support PDMats & LA.I
        - Support adjacency matrix instead of graph

=#
# NOTE: this could also be an adjacency matrix, all we need is to compute neighbors in the rand function
struct GWishart{T<:Real, ST<:Union{PDMats.AbstractPDMat, LinearAlgebra.UniformScaling}, GT<:Integer} <: Distributions.ContinuousMatrixDistribution
    df::T
    S::ST
    G::Graphs.SimpleGraph{GT}
end
GWishart(df::Real, S::AbstractMatrix, G::Graphs.SimpleGraph) = GWishart(df, PDMats.PDMat(S), G)

GWishart(p::Integer, df::Real) = GWishart(df, PDMats.ScalMat(p, 1.0), Graphs.SimpleGraph(p, p ÷ 2))
GWishart(df::Real, S::AbstractMatrix) = GWishart(df, S, Graphs.SimpleGraph(size(S, 1), size(S, 1) ÷ 2))
GWishart(df::Real, G::Graphs.SimpleGraph) = GWishart(df, PDMats.ScalMat(Graphs.nv(G), 1.0), G)

Base.size(d::GWishart)        = size(d.S)

struct InverseGWishart{T<:Real, ST<:Union{PDMats.AbstractPDMat, LinearAlgebra.UniformScaling}, GT<:Integer} <: Distributions.ContinuousMatrixDistribution
    df::T
    S::ST
    G::Graphs.SimpleGraph{GT}
end
InverseGWishart(df::Real, S::AbstractMatrix, G::Graphs.SimpleGraph) = InverseGWishart(df, PDMats.PDMat(S), G)
Base.size(d::InverseGWishart) = size(d.S)
Distributions.params(d::InverseGWishart) = (d.df, d.S, d.G)

function Distributions._rand!(rng::Random.AbstractRNG, d::InverseGWishart, W::AbstractMatrix)

    # Based on Section 2.4 in Lenkoski (2013, arXiv:1304.1350v1).

    # Possible perfomrance: there are some more optimizations possible here (known matrix structure, avoiding inversions), but let's skip these for now
    # probably should use rand_bartlet_A for this!

    _, _, G = Distributions.params(d)

    # does not guarantee psd-ness! could also be sampled from an inverse Wishart though?
    d_wish = Distributions.Wishart(d.df + size(d.G, 1) - 1, d.S)
    # W .= rand(rng, d_wish)
    Distributions.rand!(rng, d_wish, W)

    is_complete(G) && return inv(LinearAlgebra.Symmetric(W))

    Σ = Matrix(inv(LinearAlgebra.Symmetric(W))) # We do not want Σ to be <: Symmetric because that messes with indexing later on, but we do want it to actually be symmetric

    neighbors = Graphs.neighbors.(Ref(G), Graphs.vertices(G))

    return chain!(W, Σ, neighbors)

end



Distributions.params(d::GWishart)        = (d.df, d.S, d.G)

rand_bartlet_A(p::Integer, df::Real) = rand_bartlet_A(Random.default_rng(), p, df)
function rand_bartlet_A(rng::Random.AbstractRNG, p::Integer, df::T) where T<:Real
    L = LinearAlgebra.LowerTriangular(zeros(T, p, p))
    rand_bartlet_A!(rng, L, df)
    return L
end

rand_bartlet_A!(L::Random.AbstractMatrix, df::Real) = rand_bartlet_A!(Random.default_rng(), L, df)
function rand_bartlet_A!(rng::Random.AbstractRNG, L::AbstractMatrix, df::Real)
    @inbounds for j in axes(L, 1)
        L[j, j] = rand(rng, Chi(df - j + 1))
        for i in j+1:size(L, 2)
            L[i, j] = randn(rng)
        end
    end
end

function is_complete(G::Graphs.SimpleGraph)
    nv, ne = Graphs.nv(G), Graphs.ne(G)
    return ne == nv * (nv - one(nv)) ÷ 2
end

# not Base.isempty to avoid type piracy
is_empty(G::Graphs.SimpleGraph) = iszero(Graphs.ne(G))


function Distributions._rand!(rng::Random.AbstractRNG, d::GWishart, W::AbstractMatrix)
    # Possible performance: just sample with inverse Bartlett!
    # temp = Matrix(inv(Symmetric(Distributions.rand(rng, InverseGWishart(d.df, inv(d.S), d.G)))))
    # W .= temp
    # W

    # _, _, G = Distributions.params(d)

    # d_invwish = Distributions.InverseWishart(d.df + size(d.G, 1) - 1, d.S)
    # Distributions.rand!(rng, d_invwish, W)

    # is_complete(G) && return inv(W)

    # Σ = inv(W)

    # neighbors = Graphs.neighbors.(Ref(G), Graphs.vertices(G))

    # chain!(W, Σ, neighbors)

    # return Matrix(inv(W))

    return Matrix(inv(LinearAlgebra.Symmetric(Distributions.rand!(rng, InverseGWishart(d.df, inv(d.S), d.G), W))))
end


function chain!(W, Σ, neighbors)

    copyto!(W, Σ) # Step 1

    # to assess convergence - Possible performance use an uppertriangular view as Wold and a lowertriangular view as Wnew? Since W[j, j] never changes?
    Wold = similar(W)
    Wnew = Vector{Float64}(undef, size(W, 1))
    β̂    = Vector{Float64}(undef, size(W, 1))
    maxWNⱼNⱼchol = Vector{Float64}(undef, maximum(length, neighbors)^2)

    @inbounds for _ in 1:10_000

        copyto!(Wold, W)
        for j in axes(W, 1)

            Nⱼ = neighbors[j]

            Wjj = W[j, j]
            # same optimizations as wwa.cpp
            if isempty(Nⱼ)
                W[:, j] .= 0.0
                W[j, :] .= 0.0
            elseif length(Nⱼ) == size(W, 1) - 1
                W[:, j] = view(Σ, :, j)
                W[j, :] = view(Σ, j, :)
            else

                ΣNⱼj = view(Σ, Nⱼ, j)

                β̂Nⱼ = view(β̂, eachindex(Nⱼ))

                maxWNⱼNⱼchol_v = reshape(view(maxWNⱼNⱼchol, Base.OneTo(length(Nⱼ)^2)), length(Nⱼ), length(Nⱼ))

                # LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(maxWNⱼNⱼchol_v, :U))
                # LinearAlgebra.ldiv!(β̂Nⱼ, LinearAlgebra.Cholesky(maxWNⱼNⱼchol_v, :U, 0), ΣNⱼj)
                fast_ldiv!(β̂Nⱼ, maxWNⱼNⱼchol_v, W, ΣNⱼj, Nⱼ)

                jgemvavx!(Wnew, W, Nⱼ, β̂Nⱼ)

                W[:, j] = Wnew
                W[j, :] = Wnew

            end
            W[j, j] = Wjj
        end

        StatsBase.meanad(W, Wold) <= 1e-8 && return W

    end

    @warn "Did not converge"

    # pray this works
    return W

end

function chain3!(W, Σ, neighbors, buffer0, buffer1, buffer2, buffer3)

    copyto!(W, Σ) # Step 1

    # to assess convergence - use an uppertriangular view as Wold and a lowertriangular view as Wnew? Since W[j, j] never changes?
    Wold = buffer0
    Wnew = buffer1#Vector{Float64}(undef, size(W, 1))
    β̂    = buffer2#Vector{Float64}(undef, size(W, 1))
    maxWNⱼNⱼchol = buffer3#Vector{Float64}(undef, maximum(length, neighbors)^2)

    @inbounds for _ in 1:10_000

        copyto!(Wold, W)
        for j in axes(W, 1)

            Nⱼ = neighbors[j]

            Wjj = W[j, j]
            # same optimizations as wwa.cpp
            if isempty(Nⱼ)
                W[:, j] .= 0.0
                W[j, :] .= 0.0
            elseif length(Nⱼ) == size(W, 1) - 1
                W[:, j] = view(Σ, :, j)
                W[j, :] = view(Σ, j, :)
            else

                ΣNⱼj = view(Σ, Nⱼ, j)

                β̂Nⱼ = view(β̂, eachindex(Nⱼ))

                maxWNⱼNⱼchol_v = reshape(view(maxWNⱼNⱼchol, Base.OneTo(length(Nⱼ)^2)), length(Nⱼ), length(Nⱼ))

                # LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(maxWNⱼNⱼchol_v, :U))
                # LinearAlgebra.ldiv!(β̂Nⱼ, LinearAlgebra.Cholesky(maxWNⱼNⱼchol_v, :U, 0), ΣNⱼj)
                # try
                fast_ldiv!(β̂Nⱼ, maxWNⱼNⱼchol_v, W, ΣNⱼj, Nⱼ)
                # catch e
                    # @show β̂Nⱼ, maxWNⱼNⱼchol_v, W, ΣNⱼj, Nⱼ
                    # throw(error(e))
                # end

                jgemvavx!(Wnew, W, Nⱼ, β̂Nⱼ)

                W[:, j] = Wnew
                W[j, :] = Wnew

            end
            W[j, j] = Wjj
        end

        StatsBase.meanad(W, Wold) <= 1e-6 && return W

    end

    @warn "Did not converge"

    # pray this works
    return W

end


function chain2!(W, Σ, neighbors)

    copyto!(W, Σ) # Step 1

    # to assess convergence - Possible performance: use an uppertriangular view as Wold and a lowertriangular view as Wnew? Since W[j, j] never changes?
    Wold = similar(W)
    Wnew = Vector{Float64}(undef, size(W, 1))
    β̂    = Vector{Float64}(undef, size(W, 1))
    maxWNⱼNⱼchol = Vector{Float64}(undef, maximum(length, neighbors)^2)

    for iter in 1:1000 # used to be 10_000

        copyto!(Wold, W)
        for j in axes(W, 1)

            Nⱼ = neighbors[j]

            Wjj = W[j, j]
            # same optimizations as wwa.cpp
            if isempty(Nⱼ)
                W[:, j] .= 0.0
                W[j, :] .= 0.0
            elseif length(Nⱼ) == size(W, 1) - 1
                W[:, j] = view(Σ, :, j)
                W[j, :] = view(Σ, j, :)
            else

                ΣNⱼj = view(Σ, Nⱼ, j)

                β̂Nⱼ = view(β̂, eachindex(Nⱼ))

                maxWNⱼNⱼchol_v = reshape(view(maxWNⱼNⱼchol, Base.OneTo(length(Nⱼ)^2)), length(Nⱼ), length(Nⱼ))

                # LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(maxWNⱼNⱼchol_v, :U))
                # LinearAlgebra.ldiv!(β̂Nⱼ, LinearAlgebra.Cholesky(maxWNⱼNⱼchol_v, :U, 0), ΣNⱼj)
                fast_ldiv!(β̂Nⱼ, maxWNⱼNⱼchol_v, W, ΣNⱼj, Nⱼ)

                jgemvavx!(Wnew, W, Nⱼ, β̂Nⱼ)

                W[:, j] = Wnew
                W[j, :] = Wnew

            end
            W[j, j] = Wjj
        end

        # tolerance should be sqrt(eps(T))
        StatsBase.meanad(W, Wold) <= 1e-8 && return iter

    end

    @warn "Did not converge"

    # pray this works
    return 1000

end


function copyto_utri!(buffer, source)
    @inbounds for j in axes(source, 2), i in 1:j
        buffer[i, j] = source[i, j]
    end
    buffer
end
function copyto_utri!(buffer, source, idx)
    @inbounds for j in eachindex(idx)
        for i in 1:j
            buffer[i, j] = source[idx[i], idx[j]]
        end
    end
    buffer
end
function fast_ldiv!(y, A_buf, A, x)
    copyto!(y, x)
    copyto_utri!(A_buf, A)
    LinearAlgebra.LAPACK.posv!('U', A_buf, y)
    return y
end
function fast_ldiv!(y, A_buf, A, x, idx)
    copyto!(y, x)
    copyto_utri!(A_buf, A, idx)
    LinearAlgebra.LAPACK.posv!('U', A_buf, y)
    return y
end



"""
Efficient version of `LinearAlgebra.mul!(Wnew, view(W, :, Nⱼ), β̂Nⱼ)` based on https://juliasimd.github.io/LoopVectorization.jl/stable/examples/matrix_vector_ops/
"""
function jgemvavx!(Wnew, W, Nⱼ, β̂Nⱼ)
    #
    LoopVectorization.@turbo for i ∈ eachindex(Wnew)
        temp = zero(eltype(Wnew))
        for j ∈ eachindex(β̂Nⱼ)
            temp += W[i, Nⱼ[j]] * β̂Nⱼ[j]
        end
        Wnew[i] = temp
    end
    return Wnew
end

function rand_with_debuginfo(W, d::GWishart)

    rng = Random.default_rng()
    _, _, G = Distributions.params(d)

    # does not guarantee psd-ness! could also be sampled from an inverse Wishart though?
    d_wish = Distributions.Wishart(d.df + size(d.G, 1) - 1, d.S)
    # W .= rand(rng, d_wish)
    Distributions.rand!(rng, d_wish, W)

    is_complete(G) && return inv(LinearAlgebra.Symmetric(W))

    Σ = Matrix(inv(LinearAlgebra.Symmetric(W))) # We do not want Σ to be <: Symmetric because that messes with indexing later on, but we do want it to actually be symmetric

    neighbors = Graphs.neighbors.(Ref(G), Graphs.vertices(G))

    W_debug = copy(W)
    Σ_debug = copy(Σ)

    iter = chain2!(W, Σ, neighbors)

    debug = (W = W_debug, Σ = Σ_debug, iter = iter)
    return (inv(W), debug)

end

struct KObj{T<:Real}
    K::Matrix{T}
    Kinv::Matrix{T}
    Kchol::Matrix{T}
    buffer1::Vector{T}
    buffer2::Vector{T}
    buffer3::Vector{T}
end
function KObj(K::AbstractMatrix)
    p = size(K, 1)
    KObj(K, similar(K), similar(K), similar(K, p), similar(K, p), similar(K, p^2))
end
function KObj(K::AbstractMatrix, Kinv::AbstractMatrix, Kchol::AbstractMatrix)
    p = size(K, 1)
    KObj(K, Kinv, Kchol, similar(K, p), similar(K, p), similar(K, p^2))
end



function rand_with_extras!(obj, d::GWishart, rng = Random.default_rng())

    _, _, G = Distributions.params(d)

    # return Matrix(inv(LinearAlgebra.Symmetric(Distributions.rand!(rng, InverseGWishart(d.df, inv(d.S), d.G), W))))

    # does not guarantee psd-ness! could also be sampled from an inverse Wishart though?
    d_wish = Distributions.Wishart(d.df + size(d.G, 1) - 1, inv(d.S))
    # W .= rand(rng, d_wish)
    Distributions.rand!(rng, d_wish, obj.Kinv)

    if !is_complete(G)

        copyto!(obj.K, obj.Kinv)
        LinearAlgebra.LAPACK.potrf!('L', obj.K)
        LinearAlgebra.LAPACK.potri!('L', obj.K)
        # this step would not be necessary if chain be adapted to it
        @inbounds for j in 2:size(obj.K, 1), i in 1:j-1
            obj.K[i, j] = obj.K[j, i]
        end
        # Kchol == buffer0
        # args = copy.((obj.Kinv, obj.K, G.fadjlist, obj.Kchol, obj.buffer1, obj.buffer2, obj.buffer3))
        # try
        chain3!(obj.Kinv, obj.K, G.fadjlist, obj.Kchol, obj.buffer1, obj.buffer2, obj.buffer3)
        # catch e
        #     @show args
        # end

    end

    # copyto!(obj.Kchol, obj.Kinv)
    # LinearAlgebra.LAPACK.potrf!('L', obj.Kchol)
    # copyto!(obj.K,  obj.Kchol)
    # LinearAlgebra.LAPACK.potri!('L', obj.K)
    # obj.Kchol[1, 1] = obj.K[1, 1]
    # @inbounds for j in 2:size(obj.K, 1)
    #     for i in 1:j-1
    #         obj.K[i, j] = obj.K[j, i]
    #         obj.Kchol[j, i] = obj.K[j, i]
    #     end
    #     obj.Kchol[j, j] = obj.K[j, j]
    # end
    # LinearAlgebra.LAPACK.potrf!('L', obj.Kchol)

    copyto!(obj.K, obj.Kinv)
    LinearAlgebra.LAPACK.potrf!('L', obj.K)
    LinearAlgebra.LAPACK.potri!('L', obj.K)
    obj.Kchol[1, 1] = obj.K[1, 1]
    @inbounds for j in 2:size(obj.K, 1)
        for i in 1:j-1
            obj.K[i, j] = obj.K[j, i]
            obj.Kchol[j, i] = obj.K[j, i]
        end
        obj.Kchol[j, j] = obj.K[j, j]
    end
    LinearAlgebra.LAPACK.potrf!('L', obj.Kchol)

    # @assert obj.K ≈ inv(obj.Kinv)
    # @assert LinearAlgebra.LowerTriangular(obj.Kchol) ≈ LinearAlgebra.cholesky(obj.K).L

    return obj

end

# function rand_with_buffer(K, Phi, inv_K, d::GWishart)


#     _, _, G = Distributions.params(d)

#     # does not guarantee psd-ness! could also be sampled from an inverse Wishart though?
#     d_wish = Distributions.Wishart(d.df + size(d.G, 1) - 1, d.S)
#     # W .= rand(rng, d_wish)
#     Distributions.rand!(rng, d_wish, W)

#     Distributions.wishart_genA!(rng, A, d.df)
#     PDMats.unwhiten!(d.S, A)

#     is_complete(G) && return inv(LinearAlgebra.Symmetric(W))

#     Σ = Matrix(inv(LinearAlgebra.Symmetric(W))) # We do not want Σ to be <: Symmetric because that messes with indexing later on, but we do want it to actually be symmetric

#     neighbors = Graphs.neighbors.(Ref(G), Graphs.vertices(G))

#     return chain!(W, Σ, neighbors)
#     Distributions.rand!(rng, InverseGWishart(d.df, inv(d.S), d.G), W)
#     return Matrix(inv(LinearAlgebra.Symmetric()))

# end