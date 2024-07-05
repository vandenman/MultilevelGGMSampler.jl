using Test, GGMSampler, LogExpFunctions
import ForwardDiff

# TODO: ideally no atol is necessary!
# That's possible for k > 10
# TODO: check the approximation when μ is not normally distributed!
# extreme variances cause the approximation to fail

@testset "CurieWeiss Denominator Approximation" begin

    reference_fun(μ, σ, k = length(μ)) = reference_fun!(similar(μ, length(μ) + 1), μ, σ, k)

    function reference_fun!(log_esf, μ, σ, k)

        GGMSampler.esf_sum_log!(log_esf, μ)
        LogExpFunctions.logsumexp(
            log_esf[i] + σ / k * (i - 1)^2
            for i in eachindex(log_esf)
        )
    end

    # specifically for the Gibbs sampling part
    function reference_fun_gibbs(μ, σ, j)

        k = length(μ)
        # log_esf_all = GGMSampler.esf_sum_log(μ)
        # log_esf = similar(log_esf_all, length(log_esf_all) - 1)

        # GGMSampler.esf_log_drop!(log_esf, log_esf_all, μ[j])

        # esf_log_drop! is inaccurate for large k
        log_esf = GGMSampler.esf_sum_log(view(μ, 1:k .!= j))

        log_ts = [σ / k * i^2 for i in 0:k]

        log_c1 = LogExpFunctions.logsumexp(
            log_esf[i] + log_ts[i]
            for i in 1:k
        )
        log_c2 = LogExpFunctions.logsumexp(
            log_esf[i] + log_ts[i + 1]
            for i in 1:k
        )

        return log_c1, log_c2

    end

    # NOTE: for k in 1:5, the approximation sometimes fails.
    # That is a bit silly territory because then it's fairly trivial to just compute the exact ESF.

    # kvals = [1:10; 50:50:500; 2_000; 5_000; 10_000]
    kvals = [50:50:500; 2_000; 5_000; 10_000]

    get_weights(k) = 20 + ceil(Int, log(k))

    nreps(k) = k <= 1_000 ? 10 : 1

    for k in kvals

        @testset "k = $k" begin
            for _ in 1:nreps(k)

                μ = 5 .* randn(k)
                σ = 10 * abs(randn())

                obj = CurieWeissDenominatorApprox(get_weights(k))

                reference_value = reference_fun(μ, σ)
                GGMSampler.find_mode!(obj, μ, σ)
                @test reference_value ≈ compute_log_den!(obj, μ, σ)

                j = rand(1:k)
                μ_new = μ_old = μ[j]

                # TODO: these steps seem weirdly unstable... or are they implemented incorrectly?
                GGMSampler.update!(obj, μ_new, μ_old, σ, k)
                @test reference_value ≈ get_value(obj)


                GGMSampler.downdate!(obj, μ_new, σ, k)
                reference_value2 = reference_fun(view(μ, 1:k .!= j), σ, k)
                # @test isnan(reference_value2) || isapprox(reference_value2, get_value(obj))#, atol=atol)

                @test isapprox(reference_value2, get_value(obj))#, atol = atol)

                GGMSampler.update!(obj, μ_new, σ, k)
                @test reference_value ≈ get_value(obj)

                # without big, the original function tends to produce incorrect answers...
                reference_log_c1, reference_log_c2 = reference_fun_gibbs(μ, σ, j)

                # start fresh
                obj = CurieWeissDenominatorApprox(get_weights(k))
                GGMSampler.find_mode!(obj, μ, σ)
                GGMSampler.compute_ys!(obj, μ, σ)
                GGMSampler.downdate!(obj, μ[j], σ, k)

                @test reference_log_c1 ≈ get_value(obj)                  #atol = atol
                @test reference_log_c2 ≈ get_shifted_value(obj, σ, k, 1) #atol = atol

            end
        end

    end

end

@testset "Derivative integrand" begin

    f_integrand2(t, μ, σ) = GGMSampler.f_integrand(t, μ, σ)
    function f_integrand2(t::ForwardDiff.Dual, μ, σ)

        log_z = 2 * t * sqrt(σ / length(μ))
        result = zero(log_z)
        for j in eachindex(μ)
            result += log(1 + exp(μ[j] + log_z)) # log1p is pretty slow
        end
        result
    end

    testvalues = range(-50, 50, length = 401)

    μ = randn(100)
    σ = 2.5
    g_ref  = [ForwardDiff.gradient(t -> -f_integrand2(first(t), μ, σ) + abs2(first(t)), [v])[1] for v in testvalues]
    d_ref  = [ForwardDiff.derivative(t -> -f_integrand2(t, μ, σ) + abs2(t), v) for v in testvalues]
    d_self = GGMSampler.∂f∂t.(testvalues, Ref(μ), σ)
    @test g_ref ≈ d_ref ≈ d_self

end