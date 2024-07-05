using Test, MultilevelGGMSampler
import LinearAlgebra

@testset "Linear Algebra for 2x2 matrices" begin

    @testset "eigendecomposition" begin

        sign2(x) = x > zero(x) ? one(x) : -one(x)

        for _ in 1:5

            # Dense case

            A = LinearAlgebra.Symmetric(randn(2, 2))

            eigval_true, eigvec_true = LinearAlgebra.eigen(A)
            eigval_rep, eigvec_rep =MultilevelGGMSampler.eigen_2x2(A)
            @test eigvec_rep * LinearAlgebra.Diagonal(eigval_rep) * eigvec_rep' ≈ A
            @test eigval_true ≈ eigval_rep
            @test eigvec_rep * eigvec_rep' ≈ LinearAlgebra.I
            @test eigvec_rep' * eigvec_rep ≈ LinearAlgebra.I

            @test eigvec_true ≈ eigvec_rep * LinearAlgebra.Diagonal([sign2(eigvec_true[1, 1]) * sign2(eigvec_rep[1, 1]), sign(eigvec_true[2, 2]) * sign2(eigvec_rep[2, 2])])

            # Diagonal case

            A = LinearAlgebra.Symmetric([randn(); 0 ;; 0 ; randn()])

            eigval_true, eigvec_true = LinearAlgebra.eigen(A)
            eigval_rep, eigvec_rep =MultilevelGGMSampler.eigen_2x2(A)
            @test eigvec_rep * LinearAlgebra.Diagonal(eigval_rep) * eigvec_rep' ≈ A
            @test eigval_true ≈ eigval_rep
            @test eigvec_rep * eigvec_rep' ≈ LinearAlgebra.I
            @test eigvec_rep' * eigvec_rep ≈ LinearAlgebra.I

            # too many sign issues
            # @test eigvec_true ≈ eigvec_rep * LinearAlgebra.Diagonal([sign2(eigvec_true[1, 1]) * sign2(eigvec_rep[1, 1]), sign2(eigvec_true[2, 2]) * sign2(eigvec_rep[2, 2])])

        end
    end

end