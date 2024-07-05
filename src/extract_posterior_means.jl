
function extract_posterior_mean_K(res)
    if isempty(res.samples_K)
        means_k_vec = vec(map(x->OnlineStatsBase.mean(x.stats[1]), res.stats.K))
    else
        if size(res.samples_K, 1) == size(res.samples_K, 2)
            means_k = dropdims(StatsBase.mean(res.samples_K, dims=4); dims=4)
            means_k_vec = vec(mapslices(tril_to_vec, means_k, dims = 1:2))
        else
            means_k_vec = vec(dropdims(StatsBase.mean(res.samples_K, dims=3); dims=3))
        end
    end
    return means_k_vec
end

function extract_posterior_mean_G(res)
    samples_G = res.samples_G
    if size(samples_G, 1) == size(samples_G, 2)
        means_g = dropdims(StatsBase.mean(samples_G, dims=4); dims=4)
        means_g_vec = vec(mapslices(tril_to_vec, means_g, dims = 1:2))
    else
        means_g_vec = vec(dropdims(StatsBase.mean(samples_G, dims=3); dims=3))
    end
    return means_g_vec
end

function extract_posterior_means(res::MGGMResults{<:Any, <:CurieWeissGroupSamples, <:Any, <:Any})
    k, p = res.metadata.k, res.metadata.p
    e1 = p * (p + 1) ÷ 2
    e  = p * (p - 1) ÷ 2
    return (
        K = reshape(extract_posterior_mean_K(res), e1, k),
        G = reshape(extract_posterior_mean_G(res), e, k),
        μ = vec(StatsBase.mean(res.groupSamples.μ, dims=2)),
        σ = StatsBase.mean(res.groupSamples.σ)
    )
end
function extract_posterior_means(res::MGGMResults{<:Any, <:IndependentGroupSamples, <:Any, <:Any})
    k, p = res.metadata.k, res.metadata.p
    e1 = p * (p + 1) ÷ 2
    e  = p * (p - 1) ÷ 2
    return (
        K = reshape(extract_posterior_mean_K(res), e1, k),
        G = reshape(extract_posterior_mean_G(res), e, k)
    )
end

function compute_auc(fpr, tpr)
    # these two checks all agree with compute_auc
    # check 1
    # import QuadGK
    # QuadGK.quadgk(
    #     x -> begin
    #         d = fpr .- x
    #         d[d .> 0] .= -Inf
    #         value, idx = findmax(d)
    #         return tpr[idx]
    #     end,
    #     0, 1
    # )

    # https://stats.stackexchange.com/a/146174
    # height = (tpr[2:end] .+ tpr[1:end - 1]) ./ 2
    # width = diff(fpr)
    # sum(height .* width)

    dfpr = diff(fpr)
    LinearAlgebra.dot(view(tpr, 1:length(tpr)-1), dfpr) + LinearAlgebra.dot(diff(tpr), dfpr) / 2
end

function compute_roc_auc(truth, prediction, thresholds = 100)
    @assert length(truth) == length(prediction)
    roc_values = MLBase.roc(truth, prediction, thresholds)
    tpr = [0.0; reverse!(MLBase.true_positive_rate.(roc_values));  1.0]
    fpr = [0.0; reverse!(MLBase.false_positive_rate.(roc_values)); 1.0]

    auc = compute_auc(fpr, tpr)
    return (; fpr, tpr, auc)
end
