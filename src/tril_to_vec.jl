# Possible feature: does not work for negative k. Should it?

"""
Convert the lower triangle of a matrix to a vector.

See also [`triu_to_vec`](@ref).
"""
function tril_to_vec(x::AbstractMatrix{T}, k::Integer = 0) where T
    nr, nc = size(x)

    sz = _tril_no_elements(nr, nc , k)
    sz <= zero(sz) && return Vector{T}(undef, 0)
    v = Vector{T}(undef, sz)
    return tril_to_vec!(v, x, k)

end

# NOTE: this used to have {T}, but that means you could not mix bool and integer matrices
# that was inconvenient, so I dropped it. A lot of conversions from int to bool are a bit useless though
function tril_to_vec!(v::AbstractVector, x::AbstractMatrix, k::Integer = 0)
    c = 1
    # @inbounds
    for j in axes(x, 2), i in max(1, j-k):size(x, 1)
        v[c] = x[i, j]
        c += 1
    end
    return v
end

"""
Compute no. elements in triangular matrix of size (nr, nc) including the kth superdiagonal.
k is allowed to be negative.
"""
function _tril_no_elements(nr, nc, k)
    if k > zero(k)
        # subtract upper triangular part from total no. elements
        n  = nc - k - one(k) # extra one for the diagonal
        sz = nr * nc - n * (n + 1) ÷ 2
    else
        # add lower triangular part and rectangular remainder
        n = min(nr + k, nc)
        sz = (n * (n + 1)) ÷ 2 + n * (nr - n + k)
    end
end

"""
Convert the upper triangle of a matrix to a vector.

See also [`tril_to_vec`](@ref).
"""
function triu_to_vec(x::AbstractMatrix{T}, k::Integer = 0) where T
    nr, nc = size(x)

    sz = nr * nc - _tril_no_elements(nr, nc, k - one(k))
    iszero(sz) && return Vector{T}(undef, 0)
    v = Vector{T}(undef, sz)

    c = 1
    # @inbounds
    for j in axes(x, 2), i in 1:min(j-k, size(x, 1))
        v[c] = x[i, j]
        c += 1
    end
    return v
end

# Possible feature: finish these!
function vec_to_tril(x::AbstractMatrix, v::AbstractVector, k::Integer = 0) end
function vec_to_tril!(x::AbstractMatrix, v::AbstractVector, k::Integer = 0) end
function vec_to_triu(v::AbstractVector, k::Integer = 0) end
function vec_to_triu!(x::AbstractMatrix, v::AbstractVector, k::Integer = 0) end

# Possible feature: this function is a special case of tril_vec_to_max(v, nr, nc, k) with nr == nc
# Possible feature: what does it even mean for the symmetric case when k > 0?
function tril_vec_to_sym(v::AbstractVector{T}, k::Integer = 0) where T
    k <= zero(k) || throw(DomainError(k, "should be zero or negative"))

    p = ne_to_p(length(v)) # <- this is wrong, does not account for k
    x = zeros(T, p - k - one(k), p - k - one(k))
    return tril_vec_to_sym!(x, v, k)
end
function tril_vec_to_sym!(x::AbstractMatrix, v::AbstractVector, k::Integer = 0)
    @boundscheck begin
        k <= zero(k) || throw(DomainError(k, "should be zero or negative"))
        p = ne_to_p(length(v))
        size(x) == (p - k - one(k), p - k - one(k)) || throw(ArgumentError("size of x does not match the size implied by v and k."))
    end
    c = 1
    # @inbounds
    for j in axes(x, 2), i in max(1, j-k):size(x, 1)
    # for j in axes(x, 2), i in j+k:size(x, 2)
        x[i, j] = x[j, i] = v[c]
        c += 1
    end
    return x
end

function triu_vec_to_sym(v::AbstractVector{T}, k::Integer = 0) where T
    k >= zero(k) || throw(DomainError(k, "should be zero or positive"))

    p = ne_to_p(length(v)) # <- this is wrong, does not account for k
    x = zeros(T, p + k - one(k), p + k - one(k))
    return triu_vec_to_sym!(x, v, k)
end
function triu_vec_to_sym!(x::AbstractMatrix, v::AbstractVector, k::Integer = 0)
    @boundscheck begin
        k >= zero(k) || throw(DomainError(k, "should be zero or positive"))
        p = ne_to_p(length(v))
        size(x) == (p + k - one(k), p + k - one(k)) || throw(ArgumentError("size of x does not match the size implied by v and k."))
    end
    c = 1
    for j in axes(x, 2), i in 1:min(j-k, size(x, 1))
        x[i, j] = x[j, i] = v[c]
        c += 1
    end
    return x
end
