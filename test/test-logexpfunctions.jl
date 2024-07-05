using Test, GGMSampler
import LogExpFunctions

@testset "LogExpFunctions" begin

    function orig_logsumexp_sum(w, y)
        LogExpFunctions.logsumexp(
            a + b
            for (a, b) in zip(w, y)
        )
    end

    function orig_log_sum_exp_sum_shift(w, y, t, shift, σ, k)
        LogExpFunctions.logsumexp(
            wᵢ + yᵢ + shift * 2sqrt(σ / k) * tᵢ
            for (tᵢ, wᵢ, yᵢ) in zip(t, w, y)
        )
    end

    for n in (10, 10, 100, 1000, 10_000)

        w = randn(n)
        y = randn(n)
        t = randn(n)
        shift = randn()
        σ = abs(randn())
        k = 50

        @test orig_logsumexp_sum(w, y)                         ≈ GGMSampler.fast_log_sum_exp_sum(w, y, t, 0.0)
        @test orig_log_sum_exp_sum_shift(w, y, t, shift, σ, k) ≈ GGMSampler.fast_log_sum_exp_sum_shift(w, y, t, 0.0, shift, σ, k)

    end

end