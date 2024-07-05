#=
    This file is adapted from https://github.com/bdeonovic/ElementarySymmetricFunctions.jl/blob/bb8f9c6140876cdc7afc248db17939b2e74524e2/src/esf.jl#L1
    The code is not reused from the package due to very old compat bounds
    perhaps contact Benjamin to see if we can update this?

=#
function esf_sum!(S::AbstractVector{T}, x::AbstractVector{T}) where {T<:Real}
    Base.require_one_based_indexing(S)
    Base.require_one_based_indexing(x)
    fill!(S, zero(T))
    S[1] = one(T)
    @inbounds for col in 1:length(x)
        for r in 1:col
            row = col - r + 1
            S[row+1] = x[col] * S[row] + S[row+1]
            # S[row+1] = muladd(x[col], S[row], S[row+1])
        end
    end
    return S
end

"""
    esf_sum(x)
Compute the elementary symmetric functions of order k = 1, ..., n
where n = length(x)
# Examples
```julia-repl
julia> esf_sum([3.5118, .6219, .2905, .8450, 1.8648])
6-element Array{Float64,1}:
  1.0
  7.134
 16.9493
 16.7781
  7.05289
 0.999736
```
"""
function esf_sum(x::AbstractVector)
    S = similar(x, length(x) + 1)
    return esf_sum!(S, x)
end

function esf_sum_log!(S::AbstractVector{T}, log_x::AbstractVector{T}) where T <: Real
    Base.require_one_based_indexing(S)
    Base.require_one_based_indexing(log_x)
    fill!(S, -T(Inf))
    S[1] = zero(T)
    @inbounds for col in 1:length(log_x)
        for r in 1:col
            row = col - r + 1
            Sr = S[row] + log_x[col]
            Sr1 = S[row + 1]

            # TODO: can't we really compute this any faster?
            # S[row + 1] = LogExpFunctions.logaddexp(Sr1, Sr)
            S[row + 1] = logaddexp2(Sr1, Sr)

        end
    end
end

function logaddexp2(x::Real, y::Real)
    # Compute max = Base.max(x, y) and diff = x == y ? zero(x - y) : -abs(x - y)
    # in a faster type-stable way
    a, b = promote(x, y)
    if a < b
        diff = a - b
        max = b
    else
        # ensure diff = 0 if a = b = ± Inf
        diff = a == b ? zero(a - b) : b - a
        max = !isnan(b) ? a : b
    end
    return max + log(1 + exp(diff))
end

function esf_sum_log(log_x::AbstractVector)
    S = similar(log_x, length(log_x) + 1)
    esf_sum_log!(S, log_x)
    return S
end

# Update functions. TODO: needs tests!
function esf_add(esf_dropped::AbstractVector{T}, x::Number) where T
    result = similar(esf_dropped, length(esf_dropped) + 1)
    return esf_add!(result, esf_dropped, x)
end

function esf_add!(result::AbstractVector{T}, esf_dropped::AbstractVector{T}, x) where T
    result[1] = one(T)
    for j in 2:length(esf_dropped) # can be done in parallel/ Loopvectorization?
        result[j] = muladd(x, esf_dropped[j-1], esf_dropped[j])
    end
    result[end] = esf_dropped[end] * x
    return result
end

function esf_drop(esf_full::AbstractVector{T}, x::Number) where T
    result = similar(esf_full, length(esf_full) - 1)
    return esf_drop!(result, esf_full, x)
end

function esf_drop!(result::AbstractVector{T}, esf_full::AbstractVector{T}, x) where T
    result[end] = esf_full[end] / x
    for j in length(result):-1:2
        result[j-1] = (esf_full[j] - result[j]) / x
    end
    return result
end

function esf_log_add(esf_dropped::AbstractVector{T}, log_x::Number) where T
    result = similar(esf_dropped, length(esf_dropped) + 1)
    return esf_log_add!(result, esf_dropped, log_x)
end

function esf_log_add!(result::AbstractVector{T}, esf_dropped::AbstractVector{T}, log_x) where T
    result[1] = zero(T)
    for j in 2:length(esf_dropped) # can be done in parallel/ Loopvectorization?
        result[j] = LogExpFunctions.logaddexp(log_x + esf_dropped[j-1], esf_dropped[j])
    end
    result[end] = esf_dropped[end] + log_x
    return result
end

function esf_log_drop(esf_full::AbstractVector{T}, log_x::Number) where T
    result = similar(esf_full, length(esf_full) - 1)
    return esf_log_drop!(result, esf_full, log_x)
end

function esf_log_drop!(result::AbstractVector{T}, esf_full::AbstractVector{T}, log_x) where T
    result[end] = esf_full[end] - log_x
    for j in length(result):-1:2
        result[j-1] = LogExpFunctions.logsubexp(esf_full[j], result[j]) - log_x
    end
    return result
end

