using Test, MultilevelGGMSampler

@testset "LowerTriangle and UpperTriangle" begin
    for (name, M) in (("LowerTriangle", LowerTriangle), ("UpperTriangle", UpperTriangle))
        for include_diag in (true, false)
            @testset "$name(., $diag)" begin

                @test isempty(M(0))
                @test isempty(M(1))
                @test !isempty(M(1, true))

                @inferred NTuple{2, Int} first(M(2))
                @inferred NTuple{2, Int16} first(M(Int16(2)))

                if M === LowerTriangle && include_diag
                    comparison = >=
                elseif M === LowerTriangle && !include_diag
                    comparison = >
                elseif M === UpperTriangle && include_diag
                    comparison = <=
                elseif M === UpperTriangle && !include_diag
                    comparison = <
                end

                for p in (2, 5, 10, 21, 49)

                    A = zeros(p, p)

                    e = p * (p - 1) ÷ 2

                    idx_manual = Vector{Tuple{Int, Int}}(undef, e + p * include_diag)
                    c = 1
                    for i in CartesianIndices(A)
                        if comparison(i[1], i[2])
                            idx_manual[c] = Tuple(i)
                            c += 1
                        end
                    end

                    idx_fast = collect(M(A, include_diag))
                    @test idx_fast == idx_manual

                    if !include_diag && M === LowerTriangle
                        @test idx_fast ==MultilevelGGMSampler.linear_index_to_lower_triangle_indices.(1:e, p)
                    end

                end

            end
        end
    end
end

@testset "linear_index_to_lower_triangle_indices & triangle_indices_to_linear_index" begin
    for p in 2:10
        kmax = p * (p - 1) ÷ 2
        for k in 1:kmax
            i, j  =MultilevelGGMSampler.linear_index_to_lower_triangle_indices(k, p)
            k_rep =MultilevelGGMSampler.triangle_indices_to_linear_index(i, j, p)
            @test k == k_rep
        end
    end
end