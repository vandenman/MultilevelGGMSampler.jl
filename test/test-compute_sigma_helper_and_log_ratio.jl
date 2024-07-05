using Test, GGMSampler
import Distributions

@testset "Simplifications ratio normal" begin

    ps = (5, 10)
    n_reps = 5

    for p in ps

        σ_spike, σ_slab = abs.(randn(p, p)), abs.(randn(p, p))
        σ_helper2, σ_log_ratio = GGMSampler.compute_σ_helper2_and_σ_log_ratio(σ_spike, σ_slab)

        for _ in 1:n_reps
            x_test = randn()
            for (i, j) in LowerTriangle(σ_spike, false)
                lognum = Distributions.logpdf(Distributions.Normal(0.0, σ_slab[i, j]),  x_test)
                logden = Distributions.logpdf(Distributions.Normal(0.0, σ_spike[i, j]), x_test)
                @test σ_helper2[i, j] * abs2(x_test) + σ_log_ratio[i, j] ≈ lognum - logden
            end
        end
    end

end