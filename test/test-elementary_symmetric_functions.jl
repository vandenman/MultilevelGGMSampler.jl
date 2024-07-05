using Test, MultilevelGGMSampler, LogExpFunctions
import Combinatorics

@testset "esf_sum" begin

    reference_fun(x) = vcat(one(eltype(x)), [sum(prod, Combinatorics.combinations(x, i)) for i in eachindex(x)])


    for n in 1:10
        x = randn(n)
        @test reference_fun(x) ≈ esf_sum(x)
    end
end

@testset "esf_sum_log" begin

    reference_fun(x) = vcat(zero(eltype(x)), log.([sum(prod, Combinatorics.combinations(exp.(x), i)) for i in eachindex(x)]))

    for n in 1:10
        x = randn(n)
        @test reference_fun(x) ≈ MultilevelGGMSampler.esf_sum_log(x)
    end
end

@testset "esf_add" begin
    for n in 2:11
        x = randn(n)

        temp = esf_sum(view(x, 1:n-1))
        @test esf_sum(x) ≈ MultilevelGGMSampler.esf_add(temp, x[n])
    end
end
@testset "esf_drop" begin
    for n in 1:10
        x = randn(n)

        temp = esf_sum(x)
        @test esf_sum(view(x, 1:n-1)) ≈ MultilevelGGMSampler.esf_drop(temp, x[n])
    end
end

@testset "esf_log_add" begin
    for n in 2:11
        x = randn(n)

        temp = MultilevelGGMSampler.esf_sum_log(view(x, 1:n-1))
        @test MultilevelGGMSampler.esf_sum_log(x) ≈ MultilevelGGMSampler.esf_log_add(temp, x[n])
    end
end

@testset "esf_log_drop" begin
    for n in 1:10
        x = randn(n)

        temp = MultilevelGGMSampler.esf_sum_log(x)
        @test MultilevelGGMSampler.esf_sum_log(view(x, 1:n-1)) ≈ MultilevelGGMSampler.esf_log_drop(temp, x[n])
    end
end

# @testset "ESFHelper" begin

#     ns = (1, 2, 3, 5, 20, 50, 100, 200, 300, 2000, 10_000)

#     for n in ns

#         @show n
#         x = abs.(randn(n))
#         t = abs(randn())
#         bx = big.(x)

#         ref_esf = esf_sum(bx)
#         ref_log_esf_t_k_2 = [log(ref_esf[k]) + ((k - 1)^2) * log(t) for k in eachindex(ref_esf)]
#         ref_esf_t_k_2     = [ref_esf[k] * t^((k - 1)^2) for k in eachindex(ref_esf)]

#         obj = MultilevelGGMSampler.ESFHelper(x)
#         esf_t_k_2 = esf_sum_t2(obj, x, t)

#         if n <= 50
#             @test ref_esf_t_k_2 ≈ esf_t_k_2
#         end

#         # should probably simulate these directly
#         log_x = log.(x)
#         log_t = log(t)

#         log_esf_t_k_2 = log_esf_sum_t2(obj, log_x, log_t)

#         if n <= 2000
#             @test ref_log_esf_t_k_2 ≈ log_esf_t_k_2
#         end

#         # the main thing we care about
#         @test LogExpFunctions.logsumexp(ref_log_esf_t_k_2) ≈ LogExpFunctions.logsumexp(log_esf_t_k_2)

#         # if n < 5_000
#         #     step = n <= 20 ? 1 : (n ÷ 20)
#         #     ns_dropped = 1:step:n#range(start = 1, stop = n, length = min(n, 20))
#         #     for i in ns_dropped
#         #         @show n, i

#         #         ref_esf_dropped = esf_sum(view(bx, 1:n .!= i))
#         #         ref_dropped = [log(ref_esf_dropped[k]) + ((k - 1)^2) * log(t) for k in eachindex(ref_esf_dropped)]

#         #         value_dropped = log_esf_sum_t2_drop(obj, log_x[i], log_t)

#         #         if n <= 50
#         #             @test ref_dropped ≈ value_dropped
#         #         end

#         #         @test isapprox(LogExpFunctions.logsumexp(ref_dropped), LogExpFunctions.logsumexp(value_dropped), atol = 1e-4)

#         #         value_added = MultilevelGGMSampler.log_esf_sum_t2_add(obj, log_x[i], log_t)

#         #         if n <= 50
#         #             @test ref_log_esf_t_k_2 ≈ value_added
#         #         end

#         #         @test isapprox(LogExpFunctions.logsumexp(ref_log_esf_t_k_2), LogExpFunctions.logsumexp(value_added), atol = 1e-4)
#         #         # @test LogExpFunctions.logsumexp(ref_log_esf_t_k_2) ≈ LogExpFunctions.logsumexp(value_added)

#         #     end
#         # end
#     end
# end
