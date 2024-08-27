using Test, MultilevelGGMSampler, LinearAlgebra, Distributions
import LogExpFunctions, StatsBase, Random

@testset "Elementary Symmetric Functions" begin

    esf_sum_brute_force(μ, s) = sum(exp(dot(μ, g)) for g in BinarySpaceIterator(length(μ)) if sum(g) == s)

    for k in 1:5

        μ = randn(k)
        replication = esf_sum(exp.(μ))
        expected    = esf_sum_brute_force.(Ref(μ), 0:k)

        @test replication ≈ expected

    end

end


@testset "CurieWeissDistribution" begin

    function rand_dist(::Type{CurieWeissDistribution}, p)
        μ = randn(p)
        σ = abs(randn())
        CurieWeissDistribution(μ, σ)
    end

    n_samples = 5_000
    ps = 2:8

    for D in (CurieWeissDistribution, )#IsingDistribution)

        @testset "$D" begin

            @testset "Random number generation, pdf etc." begin

                for p in ps

                    d = @inferred rand_dist(D, p)

                    all_probs = [
                        pdf(d, x)
                        for x in BinarySpaceIterator(p)
                    ]

                    # test that pdf sums to 1
                    @test sum(all_probs) ≈ 1.0

                    lc = MultilevelGGMSampler.log_const(d)
                    result = LogExpFunctions.logsumexp(
                        logpdf_prop(d, g)
                        for g in BinarySpaceIterator(p)
                    )
                    # test normalizing constant
                    @test result ≈ lc

                    # should always hold
                    @test -lc ≈ logpdf(d, zeros(Int, p))

                    states = collect(BinarySpaceIterator(p))

                    # exhaustively test conditional probabilities
                    seen = Set{Int}()
                    for l in 1:p

                        empty!(seen)
                        idx = 1:p .!= l

                        while length(seen) != length(states)

                            i = first(setdiff(eachindex(states), seen))
                            ic = 0
                            for j in eachindex(states)
                                if j != i
                                    if states[j][idx] == states[i][idx]
                                        ic = j
                                        break
                                    end
                                end
                            end
                            iszero(ic) && throw(error("Should be impossible to get here!"))

                            push!(seen, i)
                            push!(seen, ic)

                            # ensure that i points to the state where the lᵗʰ element is 1 and ic to the state where the lᵗʰ element is 0
                            if iszero(states[i][l])
                                i, ic = ic, i
                            end

                            @assert isone(states[i][l]) && iszero(states[ic][l])

                            manual_log_inclusion_prob = log(all_probs[i] / (all_probs[i] + all_probs[ic]))
                            fast_log_inclusion_prob = MultilevelGGMSampler.log_inclusion_prob(l, d, states[i])

                            @test isapprox(manual_log_inclusion_prob, fast_log_inclusion_prob; atol = 1e-6)

                        end
                    end


                    samples = rand(d, n_samples)

                    @test sum(x->logpdf(d, x), eachcol(samples)) ≈ loglikelihood(d, samples)

                    counts = StatsBase.countmap(eachcol(samples))
                    obs_counts = [haskey(counts, state) ? counts[state] : 0 for state in states]
                    obs_probs = obs_counts ./ sum(obs_counts)

                    # tests observed probabilities against computed ones
                    @test isapprox(obs_probs, all_probs; atol = .1)

                    # only exists for CurieWeissDistribution
                    if d isa CurieWeissDistribution

                        emperical_marginal_probs       = log.(vec(mean(samples, dims = 2)))
                        true_log_marginal_probs        = MultilevelGGMSampler.compute_log_marginal_probs(d)
                        true_log_marginal_probs_approx = MultilevelGGMSampler.compute_log_marginal_probs_approx(d)

                        # tests observed probabilities against theoretical ones
                        @test isapprox(true_log_marginal_probs, emperical_marginal_probs; atol = 1e-1)

                        # tests theoretical probabilities against approximated ones
                        @test isapprox(true_log_marginal_probs, true_log_marginal_probs_approx; atol = 1e-3)

                    end

                    # This test is too variable/ requires too many samples
                    #=
                    # find pairs of complements (e.g., [1, 0, 0] and [0, 0, 0]) and verify conditional probabilties
                    empirical_values   = Float64[]
                    theoretical_values = Float64[]
                    for key in keys(counts)

                        for j in 1:p
                            complement = copy(key)
                            complement[j] = 1 - key[j]
                            if isone(key[j])
                                state_incl, state_excl = key, complement
                            else
                                state_incl, state_excl = complement, key
                            end

                            theoretical_incl_prob = log_inclusion_prob(j, d, state_incl)
                            p1 = counts[state_incl] / n_samples
                            p2 = counts[state_excl] / n_samples
                            empricial_incl_prob   = -LogExpFunctions.log1pexp(p2 - p1)

                            push!(empirical_values, empricial_incl_prob)
                            push!(theoretical_values, theoretical_incl_prob)

                        end
                    end
                    =#
                end

            end

            @testset "Sampling" begin

                esf_methods = (MultilevelGGMSampler.ApproximateESF(), MultilevelGGMSampler.ExactESF())

                n = 5000
                for p in (50, 100, 150)

                    σ = .5
                    μ = randn(p)
                    x = rand(CurieWeissDistribution(μ, σ), n)

                    for μ_esf_method in esf_methods,
                        σ_esf_method in esf_methods

                        # μ_esf_method, σ_esf_method = MultilevelGGMSampler.ApproximateESF(), MultilevelGGMSampler.ApproximateESF()
                        structure = CurieWeissStructure(;
                            μ_esf_method = MultilevelGGMSampler.ApproximateESF(),
                            σ_esf_method = MultilevelGGMSampler.ExactESF()
                        )

                        samples = MultilevelGGMSampler.sample_curie_weiss(x, structure, n_iter = 5_000)
                        μ_est..., σ_est = vec(StatsBase.mean(samples, dims = 2))

                        # @show p, μ_esf_method, σ_esf_method, StatsBase.cor(μ, μ_est), σ_est, σ - σ_est
                        @test StatsBase.cor(μ, μ_est) >= .95
                        @test abs(σ - σ_est) <= .6

                        # test that the adaptive MCMC algorithm reaches the desired acceptation rate
                        mh_state0 = MultilevelGGMSampler.CurieWeissMHStateσ(; n_adapts = 1)
                        desired_acc = 1 - mh_state0.acc_target
                        σ_samples = @view(samples[end, :])
                        observed_acc = StatsBase.mean(σ_samples[2:end] .== σ_samples[1:end-1])

                        @test abs(desired_acc - observed_acc) < .3

                    end

                end

            end

        end
    end
end
