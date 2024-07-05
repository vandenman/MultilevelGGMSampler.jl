#=

    Possible Optimizations:
        - [ ] Vector might be a bit of a weird return type?
        - [ ] state could be a UInt128? Then we could do preallocate a vector for the 0 and 1s and make this

=#

"""
BinarySpaceIterator

Iterate over all combinations of binary numbers.

Examples:
```julia-repl
julia> collect(BinarySpaceIterator(3))
8-element Vector{Vector{Int64}}:
 [0, 0, 0]
 [1, 0, 0]
 [0, 1, 0]
 [1, 1, 0]
 [0, 0, 1]
 [1, 0, 1]
 [0, 1, 1]
 [1, 1, 1]
```
"""
struct BinarySpaceIterator{T<:Integer}
    length::T
end

Base.IteratorSize(::Type{BinarySpaceIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{BinarySpaceIterator}) = Base.HasEltype()

Base.eltype(::BinarySpaceIterator{T}) where T = Vector{T}
# Base.eltype(iter::BinarySpaceIterator{T}) where T = Vector{T}(undef, iter.length)

Base.length(iter::BinarySpaceIterator) = 2^iter.length

# copy is needed to make collect work
function Base.iterate(iter::BinarySpaceIterator{T}) where T
    val = zeros(T, iter.length)
    return val, copy(val)
end

function Base.iterate(::BinarySpaceIterator{T}, state) where T
    finished = true
    for i in eachindex(state)
        if iszero(state[i])
            state[i] = one(T)
            finished = false
            break
        else
            state[i] = zero(T)
        end
    end
    # @show finished
    finished && return nothing
    return (state, copy(state))

end

Base.isdone(::BinarySpaceIterator{T}, state) where T = all(isone, state)