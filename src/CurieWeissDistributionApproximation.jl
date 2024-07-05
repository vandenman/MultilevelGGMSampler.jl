#=

    TODO:

        - [ ] the treading needs to be controlled via dispatch!

=#

abstract type AbstractCurieWeissDenominatorApprox end

struct CurieWeissDenominatorApprox{T<:Real} <: AbstractCurieWeissDenominatorApprox
    ys           ::Vector{T}
    ts           ::Vector{T}
    log_weights  ::Vector{T}
    mode         ::Base.RefValue{T} # always length 1, but I'd like to keep the struct immutable
    thread       ::Bool
end

function CurieWeissDenominatorApprox(::Type{T}, nweights::Integer, mode::T = zero(T), thread::Bool = true) where T<:Real

    ts, log_weights = FastGaussQuadrature.gausshermite(nweights)
    ys = similar(ts)
    log_weights .= log.(log_weights)

    CurieWeissDenominatorApprox{T}(ys, ts, log_weights, Ref(mode), thread)

end
CurieWeissDenominatorApprox(nweights::Integer, mode::Float64 = 0.0, thread::Bool = true) = CurieWeissDenominatorApprox(Float64, nweights, mode, thread)

default_no_weights(k::Integer) = 20 + ceil(Int, log(k))#max(5, ceil(Int, 4sqrt(k)))

compute_ys!(obj::CurieWeissDenominatorApprox, μ::AbstractVector, σ, p = length(μ)) = compute_ys!(obj.ys, obj.ts, μ, σ, p, zero(σ), obj.mode[], obj.thread)
function compute_ys!(ys::AbstractVector, ts::AbstractVector, μ::AbstractVector, σ::T, p::Integer = length(μ), scaled_offset = zero(T), mode::T = zero(T), thread::Bool = true) where T
    LoopVectorization.@turbo thread=true warn_check_args=false for i in eachindex(ys, ts)

        log_z = 2 * (ts[i] + mode) * sqrt(σ / p) + scaled_offset

        result = zero(log_z)
        for j in eachindex(μ)
            log_value = μ[j] + log_z
            # 33.23111882352963 is from LogExpFunctions.log1pexp (we don't care about the other branches)
            result += ifelse(log_value > 33.23111882352963, log_value, log(1 + exp(μ[j] + log_z))) # log1p is pretty slow
        end

        ys[i] = result
    end
end
function compute_log_den!(obj::CurieWeissDenominatorApprox{T}, μ::AbstractVector{T}, σ::T, p::Integer = length(μ)) where T
    compute_ys!(obj, μ, σ, p)
    return get_value(obj)
end

# function compute_log_den!(obj::CurieWeissDenominatorApprox, μ::AbstractVector, σ::ForwardDiff.Dual, p::Integer = length(μ))
#     # NOTE: this branch only exists to make things work with ForwardDiff. It is also unused, so maybe drop support?
#     ys = Vector{typeof(σ)}(undef, length(obj.ys))
#     compute_ys!(ys, obj.ts, μ, σ, p)
#     return LogExpFunctions.logsumexp(w + y for (w, y) in zip(obj.log_weights, ys)) - log(IrrationalConstants.sqrtπ)
#     # return get_value(obj)
# end


function downdate!(obj::CurieWeissDenominatorApprox, μ_rem::Real, σ::Real, p::Integer)
    (; ys, ts) = obj
    mode = obj.mode[]
    LoopVectorization.@turbo thread=true warn_check_args=true for i in eachindex(ys, ts)
        log_z = 2 * (ts[i] + mode) * sqrt(σ / p)
        ys[i] -= log(1 + exp(μ_rem + log_z))
    end
end

function update!(obj::CurieWeissDenominatorApprox, μ_new::Real, σ::Real, p::Integer)
    (; ys, ts) = obj
    mode = obj.mode[]
    LoopVectorization.@turbo thread=true warn_check_args=true for i in eachindex(ys, ts)
        log_z = 2 * (ts[i] + mode) * sqrt(σ / p)
        ys[i] += log(1 + exp(μ_new + log_z))
    end
end

function update!(obj::CurieWeissDenominatorApprox, μ_new::Real, μ_rem::Real, σ::Real, p::Integer)
    (; ys, ts) = obj
    LoopVectorization.@turbo thread=true for i in eachindex(ys, ts)
        log_z = 2 * ts[i] * sqrt(σ / p)
        ys[i] += log((1 + exp(μ_new + log_z)) / (1 + exp(μ_rem + log_z)))
    end
end

#=
@inline get_value(obj::CurieWeissDenominatorApprox) = LogExpFunctions.logsumexp(w + y for (w, y) in zip(obj.log_weights, obj.ys)) - log(IrrationalConstants.sqrtπ)
@inline function get_shifted_value(obj::CurieWeissDenominatorApprox, σ, k::Integer, shift::Real)
    LogExpFunctions.logsumexp(
        w + y + shift * 2sqrt(σ / k) * t
        for (t, w, y) in zip(obj.ts, obj.log_weights, obj.ys)
    ) - log(IrrationalConstants.sqrtπ)
end
=#

function get_value(obj::AbstractCurieWeissDenominatorApprox)
    fast_log_sum_exp_sum(obj.log_weights, obj.ys, obj.ts, obj.mode[]) - log(IrrationalConstants.sqrtπ)
end

function get_shifted_value(obj::CurieWeissDenominatorApprox, σ, k::Integer, shift::Real)
    fast_log_sum_exp_sum_shift(obj.log_weights, obj.ys, obj.ts, obj.mode[], shift, σ, k) - log(IrrationalConstants.sqrtπ)
end


function fast_log_sum_exp_sum(w, y, t, mode)
    # m = maximum(a + b for (a, b) in zip(w, y))
    # f(x, y) = exp(x + y - m)
    # return m + log(LoopVectorization.vmapreduce(f, +, w, y))
    m = maximum(wᵢ + yᵢ + -(2 * tᵢ * mode + mode^2) for (tᵢ, wᵢ, yᵢ) in zip(t, w, y))
    f(tᵢ, wᵢ, yᵢ) = exp(wᵢ + yᵢ + -(2 * tᵢ * mode + mode^2) - m)
    return m + log(LoopVectorization.vmapreduce(f, +, t, w, y))
end

function fast_log_sum_exp_sum_shift(w, y, t, mode, shift, σ, k)
    m = maximum(
        wᵢ + yᵢ + shift * 2sqrt(σ / k) * (tᵢ + mode) -(2 * tᵢ * mode + mode^2)
        for (tᵢ, wᵢ, yᵢ) in zip(t, w, y)
    )

    f(tᵢ, wᵢ, yᵢ) = exp(wᵢ + yᵢ + shift * 2sqrt(σ / k) * (tᵢ + mode) -(2 * tᵢ * mode + mode^2) - m)
    return m + log(LoopVectorization.vmapreduce(f, +, t, w, y))
end

# function fast_log_sum_exp_sum_shift(w, y, t, shift, σ, k)
#     m = maximum(
#         wᵢ + yᵢ + shift * 2sqrt(σ / k) * tᵢ
#         for (tᵢ, wᵢ, yᵢ) in zip(t, w, y)
#     )

#     f(tᵢ, wᵢ, yᵢ) = exp(wᵢ + yᵢ + shift * 2sqrt(σ / k) * tᵢ - m)
#     return m + log(LoopVectorization.vmapreduce(f, +, t, w, y))
# end


# function fast_log_sum_exp_sum_shift(w, y, t, mode, shift, σ, k, offset)
#     m = maximum(
#         wᵢ + yᵢ + shift * 2sqrt(σ / k) * tᵢ + offset
#         for (tᵢ, wᵢ, yᵢ) in zip(t, w, y)
#     )

#     f(tᵢ, wᵢ, yᵢ) = exp(wᵢ + yᵢ + shift * 2sqrt(σ / k) * tᵢ + offset - m)
#     return m + log(LoopVectorization.vmapreduce(f, +, t, w, y))
# end

#region find mode
function find_mode!(obj::CurieWeissDenominatorApprox, μ, σ)
    new_mode = find_mode(μ, σ, obj.mode[], obj.thread)
    obj.mode[] = new_mode
    obj
end

function find_mode0(μ, σ, x0, thread::Bool = true)
    Optim.optimize(
        t -> -f_integrand(first(t), μ, σ, thread) + abs2(first(t)),
        (g, t) -> g[1] = ∂f∂t(first(t), μ, σ, thread),
        [x0],
        Optim.GradientDescent())
end
find_mode(μ, σ, x0, thread) = find_mode0(μ, σ, x0, thread) |> Optim.minimizer |> first

"""
Integrand such that

``
Σ_{s=0}^K e_s(μ) exp(σ s²) =
∫_ℝ f_integrand(t, μ, σ) dt
``
where `e_s(μ)` is the `s`-th elementary symmetric function of `exp.(μ)`.

`f_integrand` is defined as

``
exp(-t²)∏_{s=1}^K (1+exp(μ_s + 2 * t * √(σ/K)).
``
"""
function f_integrand(t, μ, σ, thread::Bool = true)

    log_z = 2 * t * sqrt(σ / length(μ))
    result = zero(log_z)
    LoopVectorization.@turbo thread=true warn_check_args=false for j in eachindex(μ)
        result += log(1 + exp(μ[j] + log_z)) # log1p is pretty slow
    end
    result
end

"""
First derivative of f_integrand
"""
function ∂f∂t(t, μ, σ, thread::Bool = true)
    temp = 2 * sqrt(σ / length(μ))
    log_z = t * temp
    result = zero(log_z)
    LoopVectorization.@turbo thread=true warn_check_args=false for j in eachindex(μ)
        result += 1 / (1 + exp(μ[j] + log_z))
    end
    return -temp * (length(μ) - result) + 2 * t
end

"""
Second derivative of f_integrand
"""
function ∂²f∂²t(t, μ, σ, thread::Bool = true)
    log_z = 2 * t * sqrt(σ / length(μ))
    result = zero(log_z)
    LoopVectorization.@turbo thread=true warn_check_args=false for j in eachindex(μ)
        result += 1 / (1 + cosh(μ[j] + log_z))
    end
    p = length(μ)
    return -2σ / p * result + 2
end


#endregion