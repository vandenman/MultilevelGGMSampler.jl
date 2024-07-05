import OnlineStatsBase

struct OnCounter{T<:Integer} <: OnlineStatsBase.OnlineStat{AbstractVector}
    value::Vector{T}
    OnCounter(value) = new{eltype(value)}(value)
end

function OnCounter(p::Integer, n::Integer = 0)
    value = zeros(UInt32, p + 1)
    value[end] = n
    OnCounter(value)
end

function OnlineStatsBase._fit!(o::OnCounter, y)

    @boundscheck length(y) + 1 == length(o.value)

    @inbounds for i in eachindex(y)
        o.value[i] += y[i]
    end
    o.value[end] += 1
    return o
end

struct PairWiseOn{T<:Integer} <: OnlineStatsBase.OnlineStat{AbstractVector}
    value::Vector{T}
    PairWiseOn(value) = new{eltype(value)}(value)
end

struct PairWiseOff{T<:Integer} <: OnlineStatsBase.OnlineStat{AbstractVector}
    value::Vector{T}
    PairWiseOff(value) = new{eltype(value)}(value)
end

struct PairWiseOnOff{T<:Integer} <: OnlineStatsBase.OnlineStat{AbstractVector}
    value::Matrix{T}
    PairWiseOnOff(value) = new{eltype(value)}(value)
end

function PairWiseOn(p::Integer, n::Integer = 0)
    value = zeros(UInt32, p * (p - one(p)) ÷ 2 + one(p))
    value[end] = n
    PairWiseOn(value)
end

function PairWiseOff(p::Integer, n::Integer = 0)
    value = zeros(UInt32, p * (p - one(p)) ÷ 2 + one(p))
    value[end] = n
    PairWiseOff(value)
end

function PairWiseOnOff(p::Integer, n::Integer = 0)
    value = zeros(UInt32, 2, p * (p - one(p)) ÷ 2 + one(p))
    value[end] = n
    PairWiseOnOff(value)
end

function OnlineStatsBase._fit!(o::PairWiseOn, y)

    Base.require_one_based_indexing(y)
    @boundscheck length(y) * (length(y) - 1) ÷ 2 + 1 == length(o.value)

    n = length(y)
    c = 1
    # @inbounds
    for i in 1:n-1, j in i+1:n
        o.value[c] += y[i] * y[j]
        c += 1
    end
    o.value[end] += 1
    return o
end

function OnlineStatsBase._fit!(o::PairWiseOff, y)

    Base.require_one_based_indexing(y)
    @boundscheck length(y) * (length(y) - 1) ÷ 2 + 1 == length(o.value)

    n = length(y)
    c = 1
    # @inbounds
    for i in 1:n-1, j in i+1:n
        o.value[c] += (1 - y[i]) * (1 - y[j])
        c += 1
    end
    o.value[end] += 1
    return o
end

function OnlineStatsBase._fit!(o::PairWiseOnOff, y)

    Base.require_one_based_indexing(y)
    @boundscheck length(y) * (length(y) - 1) ÷ 2 + 2 == length(o.value)

    n = length(y)
    c = 1
    # @inbounds
    for i in 1:n-1, j in i+1:n
        o.value[1, c] +=      y[i]  *      y[j]
        o.value[2, c] += (1 - y[i]) * (1 - y[j])
        c += 1
    end
    o.value[1, end] += 1
    return o
end

OnlineStatsBase.value(o::OnCounter)     = o.value[1:end-1]
OnlineStatsBase.value(o::PairWiseOn)    = o.value[1:end-1]
OnlineStatsBase.value(o::PairWiseOff)   = o.value[1:end-1]
OnlineStatsBase.value(o::PairWiseOnOff) = o.value[:, 1:end-1]

OnlineStatsBase.nobs(o::OnCounter)     = o.value[end]
OnlineStatsBase.nobs(o::PairWiseOn)    = o.value[end]
OnlineStatsBase.nobs(o::PairWiseOff)   = o.value[end]
OnlineStatsBase.nobs(o::PairWiseOnOff) = o.value[1, end]

# OnlineStatsBase.fit!(o::OnCounter{I},     y::T) where {I, T} = (_fit!(o, y); return o)
# OnlineStatsBase.fit!(o::PairWiseOn{I},    y::T) where {I, T} = (_fit!(o, y); return o)
# OnlineStatsBase.fit!(o::PairWiseOff{I},   y::T) where {I, T} = (_fit!(o, y); return o)
# OnlineStatsBase.fit!(o::PairWiseOnOff{I}, y::T) where {I, T} = (_fit!(o, y); return o)
