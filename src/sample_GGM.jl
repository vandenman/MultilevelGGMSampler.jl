sample_GGM(data::Vector{<:AbstractMatrix}, args...; kwargs...) = sample_GGM(prepare_data(data), args...; kwargs...)
sample_GGM(data::AbstractArray{<:Real, 3}, args...; kwargs...) = sample_GGM(prepare_data(data), args...; kwargs...)

function sample_GGM(
    data::NamedTuple, # <- what type should this be?

    individualStructure::AbstractIndividualStructure = GWishartStructure(),

    ;
    rng::Random.AbstractRNG = Random.default_rng(),
    init_K = nothing,
    init_G = nothing,
    # MCMC parameters
    n_iter::Integer = 1000,
    n_warmup::Integer = min(1000, n_iter ÷ 2),
    n_thinning::Integer = 1,
    save_precmats::Bool = true,
    save_graphs::Bool = true,
    online_statistics::Function = () -> OnlineStatsBase.Series(OnlineStatsBase.Moments(zeros(4), OnlineStatsBase.EqualWeight(), 0)),
    verbose::Bool = true,
    offset::Int = 0
)

    p = data.p
    k = one(p)

    individual_structure_internal = to_internal(individualStructure, data)
    individual_samples = initialize_individual_samples(individual_structure_internal, p, k, n_iter, save_precmats, save_graphs)

    group_structure_internal = NoGroupStructure()
    group_state              = nothing

    has_online_stats = !isnothing(online_statistics()) && online_statistics() isa OnlineStatsBase.Series
    stats = has_online_stats ? initialize_online_statistics_ggm(online_statistics, p) : nothing

    individualState = initialize_individual_state(individual_structure_internal, p, k, init_K, init_G)

    prog = ProgressMeter.Progress(n_iter + n_warmup; enabled = verbose, showspeed = true, offset = offset)
    for i in range(; stop = n_iter + n_warmup)
        for _ in range(; stop = n_thinning)

            sample_individual_structure!(rng, individualState, individual_structure_internal, group_structure_internal, group_state)

        end

        if i > n_warmup
            j = i - n_warmup
            save_individual_state!(individual_samples, individualState, j, save_precmats, save_graphs)
            has_online_stats && update_online_stats!(stats, individualState)
        end

        ProgressMeter.next!(prog)

    end

    return (
        samples_G = individual_samples.samples_G,
        samples_K = individual_samples.samples_K,
        stats     = stats
    )
end