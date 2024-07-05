function prepare_data(rawdata::Vector{<:AbstractMatrix})

    p, n = size(first(rawdata))
    k = length(rawdata)

    sum_of_squares = Array{eltype(first(rawdata))}(undef, p, p, k)
    for ik in axes(sum_of_squares, 3)
        sum_of_squares[:, :, ik] = StatsBase.scattermat(rawdata[ik]; dims=2)
    end

    return (;n, p, k, sum_of_squares)
end

function prepare_data(rawdata::AbstractArray{T, 3}) where T

    p, n, k = size(rawdata)
    sum_of_squares = similar(rawdata, p, p, k)
    for ik in axes(sum_of_squares, 3)
        LinearAlgebra.mul!(
            view(sum_of_squares, :, :, ik),
            view(rawdata, :, :, ik),
            view(rawdata, :, :, ik)'
        )
    end

    return (;n, p, k, sum_of_squares)
end

function prepare_data(rawdata::AbstractArray{T, 3}, n::Integer) where T

    p, p2, k = size(rawdata)
    p != p2 && throw(ArgumentError("n is passed explicitly but size(rawdata, 1) != size(rawdata, 2)."))
    return (;n, p, k, sum_of_squares = rawdata)
end

function prepare_data(rawdata::AbstractMatrix{T}, n::Integer) where T

    p, p2 = size(rawdata)
    p != p2 && throw(ArgumentError("n is passed explicitly but size(rawdata, 1) != size(rawdata, 2)."))
    k = one(p)
    return (;n, p, k, sum_of_squares = reshape(rawdata, p, p, k))
end
function prepare_data(rawdata::AbstractMatrix{T}) where T
    p, n = size(rawdata)
    k = one(p)
    return (;n, p, k, sum_of_squares = reshape(StatsBase.scattermat(rawdata; dims=2), p, p, k))
end

struct MGGMResults{T<:Number, U, V, W}
    samples_G::BitArray{3}
    samples_K::Array{T, 3}
    groupSamples::U
    stats::V
    metadata::W
end

sample_MGGM(data::Vector{<:AbstractMatrix}, args...; kwargs...) = sample_MGGM(prepare_data(data), args...; kwargs...)
sample_MGGM(data::AbstractArray{<:Real, 3}, args...; kwargs...) = sample_MGGM(prepare_data(data), args...; kwargs...)

function sample_MGGM(
    # rng::Random.AbstractRNG,
    data::NamedTuple, # <- what type should this be?

    individual_structure::AbstractIndividualStructure = GWishartStructure(),
    group_structure::AbstractGroupStructure = CurieWeissStructure();
    # rng
    rng::Random.AbstractRNG = Random.default_rng(),
    # intial values
    init_K = nothing,
    init_G = nothing,
    # MCMC parameters
    n_iter::Integer = 1000,
    n_warmup::Integer = min(1000, n_iter ÷ 2),
    n_thinning::Integer = 1,
    save_individual_precmats::Bool = true,
    save_individual_graphs::Bool = true,
    save_group_samples::Bool = true,
    # what should we allow here? probably multiple input types, e.g., one for everything, one for G vs. rest, one per parameter
    # could also be a callable?
    online_statistics::Function = () -> OnlineStatsBase.Series(OnlineStatsBase.Moments(zeros(4), OnlineStatsBase.EqualWeight(), 0)),
    verbose::Bool = true
)

    (; p, k) = data

    individual_structure_internal = to_internal(individual_structure, data)
    group_structure_internal      = to_internal(group_structure, rng, p, k, n_warmup)
    save_group_samples = save_group_samples && !(group_structure_internal isa IndependentStructureInternal)

    samples = initialize_samples(individual_structure_internal, group_structure, p, k, n_iter, save_individual_precmats, save_individual_graphs, save_group_samples)

    has_online_stats = !isnothing(online_statistics()) && online_statistics() isa OnlineStatsBase.Series
    if has_online_stats
        stats = initialize_online_statistics_new(online_statistics, group_structure_internal, p, k)
    else
        stats = nothing
    end

    individualState = initialize_individual_state(individual_structure_internal, p, k, init_K, init_G)
    group_state      = initialize_group_state(rng, group_structure, p, k)

    start_time = time()
    prog = ProgressMeter.Progress(n_iter + n_warmup; enabled = verbose, showspeed = true)
    for i in 1:n_iter + n_warmup

        for _ in 1:n_thinning

            sample_individual_structure!(rng, individualState, individual_structure_internal, group_structure_internal, group_state)
            sample_group_structure!(rng, group_state, group_structure_internal, individualState.Gs)

        end

        if i > n_warmup
            j = i - n_warmup
            # Could be merged into one function for readability?
            save_individual_state!(samples.individual_samples, individualState, j, save_individual_precmats, save_individual_graphs)
            save_group_samples && save_group_state!(samples.group_samples, group_state, j)

            has_online_stats && update_online_stats!(stats, individualState, group_state, group_structure_internal)

        end

        ProgressMeter.next!(prog)

    end
    stop_time = time()

    return MGGMResults(
        samples.individual_samples.samples_G,
        samples.individual_samples.samples_K,
        samples.group_samples,
        stats,
        (
            n = data.n, p = p, k = k,
            group_structure_internal = group_structure_internal,
            walltime                 = stop_time - start_time
        )
    )

    # return (
    #     samples_G = samples.individual_samples.samples_G,
    #     samples_K = samples.individual_samples.samples_K,
    #     groupSamples = samples.group_samples,
    #     stats = stats,
    #     metadata = (
    #         n = data.n, p = p, k = k,
    #         group_structure_internal = group_structure_internal,
    #         sampling_method          = sampling_method,
    #         walltime                 = stop_time - start_time
    #     )
    # )

end


function initialize_individual_state(
    s::AbstractIndividualStructureInternal,
    p::Integer, k::Integer,
    init_K::Union{Nothing, AbstractArray{T, 3}, AbstractMatrix{T}},
    init_G::Union{Nothing, AbstractArray{U, 3}, AbstractMatrix{U}}
)   where {T<:Real, U<:Integer}

    D = eltype(s)
    if isnothing(init_K)
        Ks = zeros(D, p, p, k)
        for ik in axes(Ks, 3), ip in axes(Ks, 2)
            Ks[ip, ip, ik] = 1.0
        end
    elseif init_K isa AbstractMatrix
        @assert size(init_K) == (p, p)
        Ks = repeat(init_K, 1, 1, k)
    elseif init_K isa AbstractArray{T, 3} where T
        @assert size(init_K) == (p, p, k)
        Ks = copy(init_K)
    end

    if s isa SpikeAndSlabStructureInternal && s.method isa CholeskySampling
        if isnothing(init_K)
            Ls = zeros(D, p, p, k)
            for ik in axes(Ls, 3), ip in axes(Ls, 2)
                Ls[ip, ip, ik] = 1.0
            end
        elseif init_K isa AbstractMatrix
            L_init_K = cholesky(init_K).L
            Ls = repeat(L_init_K, 1, 1, k)
        elseif init_K isa AbstractArray{T, 3} where T
            @assert size(init_K) == (p, p, k)
            Ls = Array{D}(undef, p, p, k)
            for ik in axes(init_K, 3)
                Ls_ik = view(Ls, :, :, ik)
                copyto!(Ls_ik, view(init_K, :, :, ik))
                LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(Ls_ik, :L))
                for (ip, jp) in UpperTriangle(Ls_ik)
                    Ls_ik[ip, jp] = zero(D)
                end
            end
        end
        for ik in axes(Ls, 3)
            Ls_ik = view(Ls, :, :, ik)
            @assert LinearAlgebra.istril(Ls_ik)
        end
    else
        Ls = Array{D}(undef, 0, 0, 0)
    end

    if isnothing(init_G)
        Gs = zeros(Bool, p, p, k)
    elseif init_G isa AbstractMatrix
        @assert size(init_G) == (p, p)
        Gs = Bool.repeat(init_G, 1, 1, k) # NOTE: the different slices are not aliased
    elseif init_G isa AbstractArray{T, 3} where T
        Gs = copy(init_G)
    end

    G_objs = Vector{Graphs.SimpleGraph{Int}}(undef, k)
    # K_objs = Vector{KObj{Float64}}(undef, k)
    for ik in eachindex(G_objs)
        G_objs[ik] = Graphs.SimpleGraph(view(Gs, :, :, ik))
    #     K_objs[ik] = KObj(Ks[:, :, ik]) # it would be nice to not make a copy here
    end


    return (Ks = Ks, Gs = Int.(Gs), Ls = Ls, G_objs = G_objs)

end

function initialize_samples(individual_structure_internal, group_structure, p, k, n_iter, save_individual_precmats, save_individual_graphs, save_group_samples)

    group_samples = if save_group_samples
        initialize_group_samples(group_structure, p, k, n_iter)
    else
        initialize_group_samples(group_structure, zero(p), zero(k), zero(n_iter))
    end
    individual_samples = initialize_individual_samples(individual_structure_internal, p, k, n_iter, save_individual_precmats, save_individual_graphs)
    return (; individual_samples, group_samples)
end

function initialize_individual_samples(individual_structure_internal::AbstractIndividualStructureInternal, p::Integer, k::Integer, n_iter::Integer, save_individual_precmats::Bool, save_individual_graphs::Bool)
    #= # stores full matrices
    T = eltype(individual_structure_internal)
    if save_individual_precmats
        samples_K = zeros(T, p, p, k, n_iter)
    else
        samples_K = zeros(T, 0, 0, 0, 0)
    end
    if save_individual_graphs
        samples_G = BitArray(undef, p, p, k, n_iter)
    else
        samples_G = BitArray(undef, 0, 0, 0, 0)
    end
    =#
    # Stores only lower triangle
    T = eltype(individual_structure_internal)
    ne = p_to_ne(p)
    ne_p = p * (p + 1) ÷ 2
    if save_individual_precmats
        samples_K = zeros(T, ne_p, k, n_iter)
    else
        samples_K = zeros(T, 0, 0, 0)
    end
    if save_individual_graphs
        samples_G = BitArray(undef, ne, k, n_iter)
    else
        samples_G = BitArray(undef, 0, 0, 0)
    end

    return (; samples_K, samples_G)
end

# The two concrete types are the same for now, but they need not be later on
# for example, the GWishart sample may also contain the inverses and choleskys
abstract type AbstractIndividualSample end
mutable struct GWishartSample{T<:Real, U<:Integer} <: AbstractIndividualSample
    Ks::Array{T, 3}
    Gs::Array{U ,3}
end
struct SpikeAndSlabSample{T<:Real, U<:Integer} <: AbstractIndividualSample
    Ks::Array{T, 3}
    Gs::Array{U ,3}
end

abstract type AbstractGroupSample end
mutable struct CurieWeissSample{T<:Real} <: AbstractGroupSample
    const μ::Vector{T}
    σ::T
end
abstract type AbstractGroupSamples end
struct IndependentGroupSamples <: AbstractGroupSamples
end
struct CurieWeissGroupSamples{T} <: AbstractGroupSamples
    μ::Matrix{T}
    σ::Vector{T}
end

function initialize_group_samples(::IndependentStructure, ::Integer, ::Integer, ::Integer)
    return IndependentGroupSamples()
    # return nothing
end

function initialize_group_samples(::CurieWeissStructure, p::Integer, k::Integer, n_iter::Integer)
    σ = zeros(n_iter)
    μ = zeros(p_to_ne(p), n_iter)
    return CurieWeissGroupSamples(μ, σ)
    # return (; μ, σ)
end

function initialize_group_samples(::BernoulliStructure, p::Integer, k::Integer, n_iter::Integer)
    return (;) # empty named tuple
end

function sample_group_structure!(::Random.AbstractRNG, groupState, ::BernoulliStructure, Gs)
    # TODO: implement me!
end

function precmat_array_to_mat(Ks)
    p = size(Ks, 1)
    ne = p * (p + 1) ÷ 2
    mat = similar(Ks, ne, size(Ks, 3))
    return precmat_array_to_mat!(mat, Ks)
end
function precmat_array_to_mat!(mat, Ks)
    for ik in axes(Ks, 3)
        tril_to_vec!(view(mat, :, ik), view(Ks, :, :, ik))
    end
    return mat
end

function graph_array_to_mat(Gs)
    ne = p_to_ne(size(Gs, 1))
    mat = similar(Gs, ne, size(Gs, 3))
    return graph_array_to_mat!(mat, Gs)
end
function graph_array_to_mat!(mat, Gs)
    for ik in axes(Gs, 3)
        tril_to_vec!(view(mat, :, ik), view(Gs, :, :, ik), -1)
    end
    return mat
end

function sample_group_structure!(::Random.AbstractRNG, groupState::Nothing, groupStructure::IndependentStructureInternal, Gs)
    nothing
end

function sample_group_structure!(rng::Random.AbstractRNG, group_state::CurieWeissSample, group_structure::CurieWeissStructureInternal, Gs)

    x = group_structure.Gs_mat
    graph_array_to_mat!(x, Gs)

    k = size(x, 2)

    sum_scores_x = vec(sum(x, dims = 2))
    sum_sq_x     = sum(abs2 ∘ sum, eachcol(x))

    (; μ, σ) = group_state

        sample_cw_μ!(rng, μ, σ, sum_scores_x, k, group_structure)
    σ = sample_cw_σ( rng, μ, σ, sum_sq_x,     k, group_structure)

    group_state.μ .= μ # redundant?
    group_state.σ  = σ

    return group_state

end



function createGraphDistribution(::CurieWeissStructureInternal, groupState::CurieWeissSample, ::Integer)
    CurieWeissDistribution(groupState.μ, groupState.σ)
end

function createGraphDistribution(::Union{NoGroupStructure, IndependentStructureInternal}, ::Any, ::Integer)
    IndependentGraphDistribution()
end


function compute_suffstats_groupstate(::CurieWeissDistribution, Gs)
    Gs_mat = graph_array_to_mat(Gs)
    return vec(sum(Gs_mat, dims = 1))
end

function compute_suffstats_groupstate(::CurieWeissDistribution, Gs, ::Any)
    Gs_mat = graph_array_to_mat(Gs)
    return vec(sum(Gs_mat, dims = 1))
end

function compute_suffstats_groupstate(::CurieWeissDistribution, Gs, group_state::CurieWeissStructureInternal)
    Gs_mat = group_state.Gs_mat
    sum_k  = group_state.sum_k
    graph_array_to_mat!(Gs_mat, Gs)
    @assert all(x-> iszero(x) || isone(x), Gs_mat)
    sum!(sum_k', Gs_mat)
    return sum_k
end

function compute_suffstats_groupstate(::IndependentGraphDistribution, Gs)
    nothing
end

function compute_suffstats_groupstate(::IndependentGraphDistribution, Gs, group_state)
    nothing
end


function update_compute_suffstats_groupstate!(state, ::CurieWeissDistribution, k::Integer, oldvalue::Integer, newvalue::Integer)
    state[k] += newvalue - oldvalue
    # return state
end

function save_individual_state!(individual_samples, individual_state, j::Integer, save_individual_precmats::Bool, save_individual_graphs::Bool)

    # for full dense matrices
    # save_individual_graphs      && (individual_samples.samples_G[:, :, :, j] = individual_state.Gs)
    # save_individual_precmats    && (individual_samples.samples_K[:, :, :, j] = individual_state.Ks)

    # for lower triangle only
    if save_individual_graphs
        @views for ik in axes(individual_state.Gs, 3)
            tril_to_vec!(individual_samples.samples_G[:, ik, j], individual_state.Gs[:, :, ik], -1)
        end
    end
    if save_individual_precmats
        @views for ik in axes(individual_state.Ks, 3)
            tril_to_vec!(individual_samples.samples_K[:, ik, j], individual_state.Ks[:, :, ik])
        end
    end
    return nothing
end

function save_group_state!(groupSamples, groupState::CurieWeissSample, iter::Integer)
    groupSamples.μ[:, iter] = groupState.μ
    groupSamples.σ[iter]    = groupState.σ
    return nothing
end

# default empty fallback
function save_group_state!(groupSamples, groupState, iter::Integer)
end


function initialize_group_state(
    ::Random.AbstractRNG,
    ::IndependentStructure,
    ::Integer, ::Integer,
    ::Bool = false
)
    return nothing
end

function initialize_group_state(
    rng::Random.AbstractRNG,
    s::CurieWeissStructure,
    p::Integer, ::Integer,
    curie_weiss::Bool = false
)

    init_μ, init_σ = s.μ, s.σ

    ne = curie_weiss ? p : p_to_ne(p)

    μ = isnothing(init_μ) ? rand_betaprime(rng, s.prior_μ_α, s.prior_μ_β, ne) : init_μ
    σ = isnothing(init_σ) ? rand(rng, s.πσ) : init_σ

    return CurieWeissSample(μ, σ)
end

function initialize_μ(rng::Random.AbstractRNG, πμ, init_μ, p)
    if isnothing(init_μ)
        return rand_dist_len(rng, πμ, p_to_ne(p))
    else
        @assert length(init_μ) == p_to_ne(p)
        return copy(init_μ)
    end
end

function initialize_Σ(rng::Random.AbstractRNG, πσ, init_Σ, p)
    if isnothing(init_Σ)
        σ = rand_dist_len(rng, πσ, p_to_ne(p))
        return σ_to_Σ(p, σ)
    else

        @assert size(init_Σ) == (p, p)
        # ensure symmetric
        Σ = similar(init_Σ)
        @inbounds for (i, j) in LowerTriangle(init_Σ)
            Σ[i, j] = Σ[j, i] = init_Σ[i, j]
            Σ[i, i] = zero(eltype(init_Σ))
        end
        Σ[end, end] = zero(eltype(init_Σ))
        return Σ
    end
end