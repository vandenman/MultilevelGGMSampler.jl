struct LowerTriangle{T<:Integer}
    n::T
    diag::Bool
    function LowerTriangle(n::T, diag::Bool = false) where T<:Integer
        new{T}(n, diag)
    end
end
LowerTriangle(x::AbstractMatrix, diag = false) = LowerTriangle(size(x, 1), diag)

Base.IteratorSize(::Type{LowerTriangle}) = Base.HasLength()
Base.IteratorEltype(::Type{LowerTriangle}) = Base.HasEltype()

Base.eltype(::LowerTriangle{T}) where T = NTuple{2, T}

Base.length(iter::LowerTriangle) = iter.n * (iter.n - 1) ÷ 2 + iter.diag * iter.n

function Base.iterate(iter::LowerTriangle{T}) where T
    return iszero(length(iter)) ? nothing : (
        (one(T) + !iter.diag * one(T), one(T)),
        (one(T) + !iter.diag * one(T), one(T))
    )
end

function Base.iterate(iter::LowerTriangle{T}, state) where T

    i, j = state
    if i == iter.n
        if j == iter.n - one(T) * !iter.diag
            return nothing
        else
            return (
                (j+1+!iter.diag, j+1),
                (j+1+!iter.diag, j+1)
            )

        end
    end

    return ((i+1, j), (i+1, j))
end

struct UpperTriangle{T<:Integer}
    n::T
    diag::Bool
    function UpperTriangle(n::T, diag::Bool = false) where T<:Integer
        new{T}(n, diag)
    end
end
UpperTriangle(x::AbstractMatrix, diag = false) = UpperTriangle(size(x, 1), diag)

Base.IteratorSize(::Type{UpperTriangle}) = Base.HasLength()
Base.IteratorEltype(::Type{UpperTriangle}) = Base.HasEltype()

Base.eltype(::UpperTriangle{T}) where T = NTuple{2, T}

Base.length(iter::UpperTriangle) = iter.n * (iter.n - 1) ÷ 2 + iter.diag * iter.n

function Base.iterate(iter::UpperTriangle{T}) where T
    return iszero(length(iter)) ? nothing : (
        (one(T), one(T) + !iter.diag * one(T)),
        (one(T), one(T) + !iter.diag * one(T))
    )
end

function Base.iterate(iter::UpperTriangle{T}, state) where T

    i, j = state
    if i == j - one(T) * !iter.diag
        if j == iter.n
            return nothing
        else
            return (
                (one(T), j+1),
                (one(T), j+1)
            )

        end
    end

    return ((i+1, j), (i+1, j))
end
