using GGMSampler, LinearAlgebra, Test

@testset "Triangular utility functions" begin

    max_p = 10
    max_k = 3

    function rand_sym(p)
        Matrix(Symmetric(randn(p, p)))
    end
    # function rand_lt(nr, nc, k=0)
    #     tril(randn(nr, nc), k)
    # end
    # function rand_ut(nr, nc, k=0)
    #     triu(randn(nr, nc), k)
    # end

    combinations = (
        (tril_to_vec, tril, tril_vec_to_sym),
        (triu_to_vec, triu, triu_vec_to_sym)
    )

    @testset "Type stability" begin

        for p in (0, 4)

            lt_64f = rand_sym(p)
            lt_32f = Float32.(lt_64f)
            lt_16f = Float16.(lt_64f)

            @inferred tril_to_vec(lt_64f)
            @inferred tril_to_vec(lt_32f)
            @inferred tril_to_vec(lt_16f)

            @test tril_to_vec(lt_64f) isa Vector{eltype(lt_64f)}
            @test tril_to_vec(lt_32f) isa Vector{eltype(lt_32f)}
            @test tril_to_vec(lt_16f) isa Vector{eltype(lt_16f)}

            ut_64f = rand_sym(p)
            ut_32f = Float32.(ut_64f)
            ut_16f = Float16.(ut_64f)

            @inferred triu_to_vec(ut_64f)
            @inferred triu_to_vec(ut_32f)
            @inferred triu_to_vec(ut_16f)

            @test tril_to_vec(ut_64f) isa Vector{eltype(ut_64f)}
            @test tril_to_vec(ut_32f) isa Vector{eltype(ut_32f)}
            @test tril_to_vec(ut_16f) isa Vector{eltype(ut_16f)}

        end
    end

    # (method, ref, reconstruct_symmetric) = combinations[1]
    # (method, ref, reconstruct_symmetric) = combinations[2]
    for (method, ref, reconstruct_symmetric) in combinations
        @testset "Correctness of $method" begin

            reference_method = (x, k = 0) -> filter(!(iszero), vec(ref(x, k)))
            for p in 1:max_p
                k_range = min(p, max_k)
                for k in -k_range:k_range

                    x = rand_sym(p)
                    tri_vec = method(x, k)
                    @test tri_vec ≈ reference_method(x, k)

                    if (k <= zero(k) && method == tril_to_vec) || (k >= zero(k) && method == triu_to_vec)
                        sym_mat = reconstruct_symmetric(tri_vec, k)
                        x_new = copy(x)
                        if !iszero(k)
                            kk = abs(k) - 1
                            for k2 in -kk:kk
                                x_new[diagind(x_new, k2)] .= 0.0
                            end
                        end
                        @test sym_mat ≈ x_new
                    end
                end
            end
        end
    end

    # @testset "Reconstructing symmetric matrices"
    # for (method, reference_method) in combinations
    #     @testset "Correctness of $method" begin
    #         for p in 1:max_p
    #             k_range = min(p, max_k)
    #             for k in -k_range:k_range

    #                 x = rand_sym(p)
    #                 @test method(x, k) ≈ reference_method(x, k)

    #             end
    #         end
    #     end
    # end
end
