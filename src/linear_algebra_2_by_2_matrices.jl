"""
Given
```julia
    P = [
        0   1
        1   0
    ]
```
`_inv_2x2_plus_P!(A)` computes `inv(P + A)` in place for a 2 by 2 matrix.

The function assumes, but does not check, that A is symmetric and that P + A is invertible.

See also [`_inv_2x2_plus_P`](@ref), for a version that does not modify the input.
"""
function _inv_2x2_plus_P!(bb)

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
`_inv_2x2_plus_P(A)` computes `inv(P + A)` for a 2 by 2 matrix.

The function assumes, but does not check, that A is symmetric and that P + A is invertible.

See also [`_inv_2x2_plus_P!`](@ref), for a version that does modifies A in place.
"""
function _inv_2x2_plus_P(bb)
    return _inv_2x2_plus_P!(copy(bb))
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
function _woodbury_update_rank_2_sym!(invK, U, temp1, temp2)

    A_mul_B!(temp1, invK, U)
    A_mul_B!(temp2, U', temp1)
    # LinearAlgebra.mul!(temp1, invK, U)
    # LinearAlgebra.mul!(temp2, U', temp1)

    _inv_2x2_plus_P!(temp2)

    # LinearAlgebra.mul!(U, temp1, temp2)
    A_mul_B!(U, temp1, temp2)
    LinearAlgebra.mul!(invK, U, temp1', -1, 1)

end
