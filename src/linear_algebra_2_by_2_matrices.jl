"""
Compute the eigendecomposition of a 2x2 matrix.

Reference:
Charles-Alban Deledalle, Loic Denis, Sonia Tabti, Florence Tupin. Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices. [Research Report] Université de Lyon. 2017. hal-01501221

See also [`eigen_2x2!!`](@ref), for a version that does not allocate.
"""
function eigen_2x2(A)
    eigenvalues  = similar(A, 2)
    eigenvectors = similar(A, 2, 2)
    return eigen_2x2!!(eigenvalues, eigenvectors, A)
end

"""
Compute the eigendecomposition of a 2x2 matrix and store the result in a preallocated vector and matrix.

Reference:
Charles-Alban Deledalle, Loic Denis, Sonia Tabti, Florence Tupin. Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices. [Research Report] Université de Lyon. 2017. hal-01501221

See also [`eigen_2x2`](@ref).
"""
function eigen_2x2!!(eigenvalues, eigenvectors, A)

    a, c, _, b = A

    if iszero(c)

        if a < b

            eigenvalues[1] = a
            eigenvalues[2] = b

            eigenvectors[1, 1] = -one(eltype(eigenvectors))
            eigenvectors[2, 1] = zero(eltype(eigenvectors))
            eigenvectors[1, 2] = zero(eltype(eigenvectors))
            eigenvectors[2, 2] =  one(eltype(eigenvectors))

        else

            eigenvalues[1] = b
            eigenvalues[2] = a

            eigenvectors[1, 1] = zero(eltype(eigenvectors))
            eigenvectors[2, 1] =  one(eltype(eigenvectors))
            eigenvectors[1, 2] = -one(eltype(eigenvectors))
            eigenvectors[2, 2] = zero(eltype(eigenvectors))

        end

    else

        δ = sqrt(4 * c^2 + (a - b)^2)
        eigenvalues[1] = (a + b - δ) / 2
        eigenvalues[2] = (a + b + δ) / 2

        eigenvectors[1, 1] = c
        eigenvectors[2, 1] = eigenvalues[1] - a
        eigenvectors[1, 2] = eigenvalues[2] - b
        eigenvectors[2, 2] = c

        # Possible performance: calling LinearAlgebra at this step is a bit silly, but I couldn't be boterhed to write the normalization myself
        LinearAlgebra.normalize!(eigenvectors)
        eigenvectors .*= IrrationalConstants.sqrt2

    end

    return eigenvalues, eigenvectors
end

"""
Given
```julia
    P = [
        0   1
        1   0
    ]
```
`inv_2x2_plus_P!(A)` computes `inv(P + A)` in place for a 2 by 2 matrix.

See also [`inv_2x2_plus_P`](@ref), for a version that does not modify the input.
"""
function inv_2x2_plus_P!(bb)

    een = one(eltype(bb))
    # determinant
    d = inv(bb[1] * bb[4] - (bb[2] + een) * (bb[3] + een))
    temp = bb[1]
    bb[1] = bb[4] * d
    bb[4] = temp  * d
    temp = bb[2]
    bb[2] = -(bb[3] + een) * d
    bb[3] = -(temp + een)  * d

    return bb
end

"""
Given
```julia
    P = [
        0   1
        1   0
    ]
```
`inv_2x2_plus_P(A)` computes `inv(P + A)` for a 2 by 2 matrix.

See also [`inv_2x2_plus_P!`](@ref), for a version that does modifies A in place.
"""
function inv_2x2_plus_P(bb)
    return inv_2x2_plus_P!(copy(bb))
end

# Possible optimization: is A_mul_B! really worth it?
# yes, see bench_woodbury_update.jl
function A_mul_B!(C, A, B)
    # does not take advantage of the symmetry of A
    # from https://juliasimd.github.io/LoopVectorization.jl/latest/examples/matrix_multiplication/
    LoopVectorization.@turbo for n ∈ LoopVectorization.indices((C,B), 2), m ∈ LoopVectorization.indices((C,A), 1)
        Cmn = zero(eltype(C))
        for k ∈ LoopVectorization.indices((A,B), (2,1))
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end


"""
Compute a Woodbury update for the following inverse

inv(A + U * P * U') = inv(A + U * P * U')

with size(A) = (p, p), size(U) = (p, 2), and P = [0 ; 1 ;; 1 ; 0]
"""
function __woodbury_update_rank_2_sym!(invK, U, temp1, temp2)

    A_mul_B!(temp1, invK, U)
    A_mul_B!(temp2, U', temp1)
    # LinearAlgebra.mul!(temp1, invK, U)
    # LinearAlgebra.mul!(temp2, U', temp1)

    inv_2x2_plus_P!(temp2)

    # LinearAlgebra.mul!(U, temp1, temp2)
    A_mul_B!(U, temp1, temp2)
    LinearAlgebra.mul!(invK, U, temp1', -1, 1)

end
