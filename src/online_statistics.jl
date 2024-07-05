function initialize_online_statistics(online_statistics, ::CurieWeissStructureInternal, p::Integer, k::Integer)

    ne_plus_diag = p * (p + 1) ÷ 2
    ne = p_to_ne(p)

    # Possible feature: This structure is very inefficient!
    # Possible feature: not 100% sure about this structure
    return (
        K = [copy(online_statistics) for _ in 1:ne_plus_diag, _ in 1:k],
        μ = [copy(online_statistics) for _ in 1:ne],
        σ = copy(online_statistics)
    )
end

function initialize_online_statistics_new(online_statistics_gen, ::CurieWeissStructureInternal, p::Integer, k::Integer)

    ne_plus_diag = p * (p + 1) ÷ 2
    ne = p_to_ne(p)

    return (
        K = [online_statistics_gen() for _ in 1:ne_plus_diag, _ in 1:k],
        μ = [online_statistics_gen() for _ in 1:ne],
        σ = online_statistics_gen()
    )
end


function update_online_stats!(stats, individualState, group_state#=::CurieWeissSample=#, ::CurieWeissStructureInternal)

    for ik in axes(individualState.Ks, 3)
        ine = 1
        for jp in axes(individualState.Ks, 2)
            for ip in jp:size(individualState.Ks, 1)
        # k_tril_vec = tril_to_vec(view(individualState.Ks, :, :, ik))
                OnlineStatsBase.fit!(stats.K[ine, ik], individualState.Ks[ip, jp, ik])
                ine += 1
            end
        end
    end

    for i in eachindex(group_state.μ)
        OnlineStatsBase.fit!(stats.μ[i], group_state.μ[i])
    end
    OnlineStatsBase.fit!(stats.σ, group_state.σ)

    return stats
end

function initialize_online_statistics_ggm(online_statistics_gen, p::Integer)

    ne_plus_diag = p * (p + 1) ÷ 2

    return (K = [online_statistics_gen() for _ in 1:ne_plus_diag], )
end

function update_online_stats!(stats, individualState)#, group_state#=::CurieWeissSample=#, ::CurieWeissStructureInternal)

    ik = 1
    ine = 1
    for jp in axes(individualState.Ks, 2)
        for ip in jp:size(individualState.Ks, 1)
            OnlineStatsBase.fit!(stats.K[ine, ik], individualState.Ks[ip, jp, ik])
            ine += 1
        end
    end

    return stats

end


function initialize_online_statistics_new(online_statistics_gen, ::IndependentStructureInternal, p::Integer, k::Integer)

    ne_plus_diag = p * (p + 1) ÷ 2
    ne = p_to_ne(p)

    return (
        K = [online_statistics_gen() for _ in 1:ne_plus_diag, _ in 1:k],
    )
end

function update_online_stats!(stats, individualState, group_state#=::CurieWeissSample=#, ::IndependentStructureInternal)

    for ik in axes(individualState.Ks, 3)
        ine = 1
        for jp in axes(individualState.Ks, 2)
            for ip in jp:size(individualState.Ks, 1)
        # k_tril_vec = tril_to_vec(view(individualState.Ks, :, :, ik))
                OnlineStatsBase.fit!(stats.K[ine, ik], individualState.Ks[ip, jp, ik])
                ine += 1
            end
        end
    end

    return stats
end
