using Test, MultilevelGGMSampler
import Random, StatsBase

# TODO: some of the functions used in this file should be in the package itself, or should be exported!
# - [x] extract_posterior_means
# - [x] compute_roc_auc
# - [ ] graph_array_to_mat
# - [ ] WWA is broken with independent structure

@testset "sample_MGGM" begin

    n, p, k = 1000, 10, 20
    πGW = MultilevelGGMSampler.GWishart(p, 3.0)
    groupstructure = CurieWeissStructure(; σ = 0.15)
    data, parameters = simulate_hierarchical_ggm(n, p, k, πGW, groupstructure)
    save_individual_precmats = false#sizeof(Float64) * p * p * k <= save_limit

    # group_level      = CurieWeissStructure()
    # individual_level = GWishartStructure()

    # group_level      = IndependentStructure()
    # individual_level = GWishartStructure()

    group_level      = CurieWeissStructure()
    individual_level = SpikeAndSlabStructure(;threaded = false)

    for group_level in (CurieWeissStructure(), IndependentStructure())
        for individual_level in (
                SpikeAndSlabStructure(), SpikeAndSlabStructure(;threaded = true),
                GWishartStructure(),     #=GWishartStructure(;threaded = true)=#
            )

            name_group_level      = typeof(group_level).name.wrapper
            name_individual_level = typeof(individual_level).name.wrapper

            @testset "group level = $name_group_level, group level = $name_individual_level, threaded = $(individual_level.threaded)" begin

                res = sample_MGGM(data, individual_level, group_level; n_iter = 1000, n_warmup = 100, save_individual_precmats = save_individual_precmats)

                results = extract_posterior_means(res)
                true_K_vec = vec(dropdims(mapslices(tril_to_vec, parameters.K, dims = 1:2), dims = 2))
                est_K_vec  = vec(results.K)

                @test StatsBase.cor(true_K_vec, est_K_vec) >= .95

                true_G_vec  = vec(MultilevelGGMSampler.graph_array_to_mat(parameters.G))
                means_G_vec = vec(results.G)
                _, _, auc = compute_roc_auc(true_G_vec, means_G_vec)

                @test auc >= .90

                if group_level isa CurieWeissStructure
                    true_μ = parameters.μ
                    est_μ  = results.μ
                    @test StatsBase.cor(true_μ, est_μ) >= .80

                    true_σ = parameters.σ
                    est_σ  = results.σ
                    @test_broken abs(true_σ - est_σ) <= .1
                end

            end

        end
    end
end