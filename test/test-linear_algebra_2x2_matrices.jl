using Test, MultilevelGGMSampler
import LinearAlgebra

@testset "Linear Algebra for 2x2 matrices" begin

    @testset "eigendecomposition" begin

        for _ in 1:5

            # Dense case

            A = LinearAlgebra.Symmetric(randn(2, 2))
            P = [
                0   1
                1   0
            ]

            ref = LinearAlgebra.inv(A + P)

            rep = MultilevelGGMSampler._inv_2x2_plus_P(Matrix(A))

            @test ref ≈ rep
        end
    end

end