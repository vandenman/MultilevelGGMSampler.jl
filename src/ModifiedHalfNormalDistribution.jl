struct ModifiedHalfNormal{T} <: Distributions.ContinuousUnivariateDistribution
    α::T
    β::T
    γ::T
    ModifiedHalfNormal{T}(α::T, β::T, γ::T) where T<:Real = new{T}(α, β, γ)
end

function ModifiedHalfNormal(α::T, β::T, γ::T; check_args::Bool=true) where {T <: Real}
    Distributions.@check_args ModifiedHalfNormal (α, α > zero(α)) (α, α > zero(α)) (β, β > zero(β))
    ModifiedHalfNormal{T}(α, β, γ)
end
ModifiedHalfNormal(α::Number, β::Number, γ::Number; check_args::Bool = true) = ModifiedHalfNormal(promote(α, β, γ)...; check_args = check_args)

Distributions.params(d::ModifiedHalfNormal) = (d.α, d.β, d.γ)

Base.minimum(::ModifiedHalfNormal{T}) where T = zero(T)
Base.maximum(::ModifiedHalfNormal{T}) where T = T(Inf)

Distributions.insupport(::ModifiedHalfNormal, x::Real) = x >= zero(x)

function Random.rand(rng::Random.AbstractRNG, d::ModifiedHalfNormal)

    α, β, γ = Distributions.params(d)

    # special cases
    iszero(γ) && return sqrt(rand(rng, Distributions.Gamma(α / 2, inv(β))))
    isone(α)  && return rand(rng, Distributions.truncated(Distributions.Normal(γ / 2β, inv(sqrt(2β))), 0, Inf))

    # 0 < α < 1
    # return _rand_Gao_Wang(rng, α, β, -γ)


    # condition_sun_alg_1(α, β, γ) && return sun_alg_1(rng, α, β, γ)

    try
        return _rand_Gao_Wang(rng, α, β, -γ)
    catch e
        @show α, β, γ
        throw(e)
    end

    # return _rand_Gao_Wang(rng, α, β, -γ)
    # return _rand_mhn_α_ge_0_β_ge_0_γ_le_0(rng, d)

end


# Random.sampler(d::ModifiedHalfNormalDistribution)

function Distributions.logpdf(d::ModifiedHalfNormal, x::Real)
    logpdf_prop(d, x) - log_const(d)
end

function logpdf_prop(d::ModifiedHalfNormal, x::Real)

    α, β, γ = Distributions.params(d)

    result = #=(α / 2) * log(β) +=# (α - one(α)) * log(x) + (
        -β * abs2(x) + γ * x
      )

    Distributions.insupport(d, x) || return oftype(result, -Inf)
    iszero(x) && return oftype(result, -Inf)
    isinf(x)  && return oftype(result, -Inf)

    return result

end

function log_const(d::ModifiedHalfNormal)

    α, β, γ = Distributions.params(d)

    temp1 = (one(α) + α) / 2
    temp2 = α / 2

    # Mathematica's answer to the normalizing constant uses confluent HypergeometricFunctions.
    # a direct implementation of the Fox-Wright Psi function might be more accurate but boils down to the same thing
    gamma_pos = γ > zero(γ)
    logterm_1 = log(abs(γ))+ SpecialFunctions.loggamma(temp1) + log(HypergeometricFunctions._₁F₁(temp1, 3 / 2, abs2(γ) / (4β)))
    logterm_2 = log(β) / 2 + SpecialFunctions.loggamma(temp2) + log(HypergeometricFunctions._₁F₁(temp2, 1 / 2, abs2(γ) / (4β)))
    return IrrationalConstants.Loghalf() - temp1 * log(β) + (gamma_pos ? LogExpFunctions.logaddexp(logterm_2, logterm_1) : LogExpFunctions.logsubexp(logterm_2, logterm_1))

end

# function exp_const(d::ModifiedHalfNormal)

#     α, β, γ = Distributions.params(d)


#     temp1 = (one(α) + α) / 2
#     temp2 = α / 2
#     term_1 =       γ * SpecialFunctions.gamma(temp1) * HypergeometricFunctions._₁F₁(temp1, 3 / 2, abs2(γ) / (4β))
#     term_2 = sqrt(β) * SpecialFunctions.gamma(temp2) * HypergeometricFunctions._₁F₁(temp2, 1 / 2, abs2(γ) / (4β))
#     return (1 / 2) * β ^ (-temp1) * (term_1 + term_2)

# end


# these two are outdated!
@inline function _gap_setup(tl, logf_m)
    expl = exp(tl)
    logf_tl  = λ * tl - expl * (    expl + β)
    dlogf_tl = λ      - expl * (2 * expl + β)
    gap = logf_m - logf_tl
    return gap, tl, expl, logf_tl, dlogf_tl
end
@inline function _gap_helper(gap, tl, dlogf_tl, logf_m, λ, β, gap_lb = 0.46, gap_ub = 2.49, gap_diff = one(gap))
    while gap < gap_lb || gap > gap_ub
        tl = tl + (gap - gap_diff) / dlogf_tl
        expl = exp(tl)
        logf_tl  = λ * tl - expl * (expl + β)
        dlogf_tl = λ - expl * (2 * expl + β)
        gap = logf_m - logf_tl
    end
    return gap, tl, expl, logf_tl, dlogf_tl
end

@inline function _gap(start, logf_m, λ_m1, β)
    tr = start
    logf_tr = λ_m1 * log(tr) - (tr + β) * tr
    dlogf_tr = λ_m1 / tr - 2*tr - β
    gap = logf_m - logf_tr

    # iszero(gap - 1) && throw(error("failure in _gap: gap - 1 == 0"))
    # iszero(gap - 1) && @warn "failure in _gap: gap - 1 == 0"

    count = 0
    safety = 100_000
    while gap < 0.46 || gap > 2.49
        tr = tr + (gap - 1) / dlogf_tr
        logf_tr = λ_m1 * log(tr) - (tr + β) * tr
        dlogf_tr = λ_m1 / tr - 2*tr - β
        gap = logf_m - logf_tr

        count += 1
        count > safety && throw(error("failure in _gap: 100_000 iterations passed"))
    end
    return gap, tr, logf_tr, dlogf_tr
end

@inline function _gap2(tl, logf_m, λ, β, gap_diff = one(tl), gap_lb = 0.46, gap_ub = 2.49)
    expl = exp(tl)
    logf_tl  = λ * tl - expl * (    expl + β)
    dlogf_tl = λ      - expl * (2 * expl + β)
    gap = logf_m - logf_tl

    # iszero(gap - gap_diff) && throw(error("failure in _gap2: gap - gap_diff == 0"))
    # iszero(gap - gap_diff) && @warn "failure in _gap2: gap - gap_diff == 0"

    count = 0
    safety = 100_000
    while gap < gap_lb || gap > gap_ub
        tl = tl + (gap - gap_diff) / dlogf_tl
        expl = exp(tl)
        logf_tl  = λ * tl - expl * (expl + β)
        dlogf_tl = λ - expl * (2 * expl + β)
        gap = logf_m - logf_tl

        count += 1
        count > safety && throw(error("failure in _gap2: 100_000 iterations passed"))
    end
    return gap, tl, expl, logf_tl, dlogf_tl
end

function _error_rand_Gao_Wang()
    throw(error("An internal while loop for sampling from the ModifiedHalfNormal Distribution took more than 100 000 iterations."))
end

function _rand_Gao_Wang(rng::Random.AbstractRNG, λ, α, β)

    # λ, α, β = Distributions.params(d)

    root_α = sqrt(α)
    β = β / root_α
    β² = abs2(β)

    # if β < zero(β)
    #     expm = (sqrt(β² + 8λ) - β) / 4
    # else
    #     expm = 2λ / (sqrt(β² + 8λ) + β)
    # end

    # mmm = log(expm)
    # logf_m   = λ * mmm - expm * (    expm + β)
    # dlogf_m  = λ       - expm * (2 * expm + β)
    # ddlogf_m = -expm * (4 * expm + β)

    if λ >= one(λ)

        λ_m1 = λ - one(λ)
        DD = β² + 8λ_m1

        if β < zero(β)
            mmm = (sqrt(DD) - β) / 4
        else
            mmm = 2λ_m1 / (sqrt(DD) + β)
        end

        logf_m = λ_m1 * log(mmm) - (mmm + β) * mmm
        ddlogf_m = -2 - λ_m1 / abs2(mmm)
        inc = sqrt(-2 / ddlogf_m)

        gap, tr, logf_tr, dlogf_tr = _gap(mmm + inc, logf_m, λ_m1, β)

        om1 = -1 / dlogf_tr
        pr = tr - om1 * gap
        ppr = pr + om1 * log(om1)

        if ((λ_m1 <= 1.718282) && ((β >= zero(β)) || (abs2(DD) <= 109.97 * λ_m1)))
            pl = zero(β)
            om2 = pr
            om3 = zero(β)
        else
            gap, tl, logf_tl, dlogf_tl = _gap(mmm - inc, logf_m, λ_m1, β)

            pl = tl + gap / dlogf_tl
            prb = exp(-dlogf_tl * pl)
            om3 = (1 - prb) / dlogf_tl
            om2 = pr - pl
        end
        om = om1 + om2 + om3

        count, safety = 0, 100_000
        while true
            U = rand(rng, Distributions.Uniform(zero(om), om))
            if U < om1
                X = ppr - om1 * log(om1 - U)
                condition = λ_m1 * log(X) - (X + β) * X - logf_m - dlogf_tr * (X - pr)
            elseif U < om1 + om2
                X = pl + U - om1
                condition = λ_m1 * log(X) - (X + β) * X - logf_m
            else
                X = pl + log1p(dlogf_tl * (U - om)) / dlogf_tl
                condition = λ_m1 * log(X) - (X + β) * X - logf_m - dlogf_tl * (X - pl)
            end
            if log(rand(rng)) <= condition
                return X / root_α
            end
            count += 1
            count >= safety && _error_rand_Gao_Wang()

        end
    else # λ < 1

        if β < zero(β)
            expm = (sqrt(β² + 8λ) - β) / 4
        else
            expm = 2λ / (sqrt(β² + 8λ) + β)
        end

        mmm = log(expm)
        logf_m   = λ * mmm - expm * (    expm + β)
        ddlogf_m = -expm * (4 * expm + β)

        if β >= zero(β) # T-0

            inc = sqrt(-2 / ddlogf_m)
            gap, tl, expl, logf_tl, dlogf_tl = _gap2(mmm - inc, logf_m, λ, β)

            om1 = 1 / dlogf_tl
            pl = tl + om1 * gap

            gap, tr, exp_tr, logf_tr, dlogf_tr = _gap2(mmm + inc, logf_m, λ, β)

            om3 = -1 / dlogf_tr
            pr = tr - om3 * gap
            om2 = pr - pl
            om = om1 + om2 + om3
            ppl = pl - om1 * log(om1)
            ppr = pr + om3 * log(om3)

            count, safety = 0, 100_000
            while true
                U = rand(rng, Distributions.Uniform(zero(om), om))
                if U < om1
                    X = ppl + om1 * log(U)
                    expX = exp(X)
                    condition = λ * X - expX * (expX + β) - logf_m - dlogf_tl * (X - pl)
                elseif U < om1 + om2
                    X = pl + U - om1
                    expX = exp(X)
                    condition = λ * X - expX * (expX + β) - logf_m
                else
                    X = ppr - om3 * log(om - U)
                    expX = exp(X)
                    condition = λ * X - expX * (expX + β) - logf_m - dlogf_tr * (X - pr)
                end
                if log(rand(rng)) <= condition
                    return expX / root_α
                end
                count += 1
                count >= safety && _error_rand_Gao_Wang()

            end

        elseif ((λ >= 0.5) || (β >= 2 * sqrt(1 - 2 * λ) - 2)) ## T_{-1/2}

            inc = sqrt(-2.772589 / ddlogf_m)
            gap, tl, expl, logf_tl, dlogf_tl = _gap2(mmm - inc, logf_m, λ, β, 1.386294, 0.93, 1.99)

            halfdlogf_tl = dlogf_tl / 2
            om1 =  exp(-gap / 2) / halfdlogf_tl
            pl = tl + 1 / halfdlogf_tl - om1

            gap, tr, exp_tr, logf_tr, dlogf_tr = _gap2(mmm + inc, logf_m, λ, β, 1.386294, 0.93, 1.99)

            halfdlogf_tr = dlogf_tr / 2
            om3 = -exp(-gap / 2) / halfdlogf_tr
            pr = tr + 1 / halfdlogf_tr + om3
            om2 = pr - pl
            om = om1 + om2 + om3

            ppl = pl + om1
            ppr = pr - om3
            om1_s = abs2(om1)
            om3_s = abs2(om3)

            count, safety = 0, 100_000
            while true
                U = rand(rng, Distributions.Uniform(zero(om), om))

                if (U < om1)
                    X = ppl - om1_s / U
                    expX = exp(X)
                    condition = exp(λ * X - expX * (expX + β) - logf_tl) * (halfdlogf_tl * (X - tl) - 1)^2
                elseif (U < om1 + om2)
                    X = pl + U - om1
                    expX = exp(X)
                    condition = exp(λ * X - expX * (expX + β) - logf_m)
                else
                    X = ppr + om3_s / (om - U)
                    expX = exp(X)
                    condition = exp(λ * X - expX * (expX + β) - logf_tr) * (halfdlogf_tr * (X - tr) - 1)^2
                end

                if rand(rng) <= condition
                    return expX / root_α
                end

                count += 1
                count >= safety && _error_rand_Gao_Wang()

            end
        else

            expxast = -β / 4
            xast = log(expxast)
            logf_xast = λ * xast - expxast * (expxast + β)
            rho = logf_m - logf_xast

            if rho > 2.49

                gap, tl, exptl, logf_tl, dlogf_tl = _gap2(log(-β / 3), logf_m, λ, β)

                pl = tl + gap / dlogf_tl
                exptl_ast = -β / 2 - exptl
                rho_ast = -exptl_ast * (exptl_ast + β)
                max_k = ceil(Int, rho_ast)

                x_seq = Float64[]
                logf_seq = Float64[]
                delta = zero(β)
                rho_ast = 2 * rho_ast / max_k

                count, safety = 0, 100_000
                while true

                    delta = delta + rho_ast
                    exp_xk = delta / (sqrt(β² - 2 * delta) - β)

                    xk = log(exp_xk)
                    logf_xk = λ * xk - exp_xk * (exp_xk + β)

                    # @show delta, xk, logf_xk
                    push!(x_seq, xk)
                    push!(logf_seq, logf_xk)

                    logf_xk <= logf_tl + dlogf_tl * (xk - tl) && break

                    count += 1
                    count >= safety && _error_rand_Gao_Wang()

                end
                # @show x_seq

                max_k = length(x_seq)
                if max_k > one(max_k)
                    lm_seq = diff(logf_seq) ./ diff(x_seq)
                    om = diff(exp.(logf_seq .- logf_m))
                    lm_seq = pushfirst!(lm_seq, λ)
                    om = [exp(logf_seq[1] - logf_m); om] ./ lm_seq
                else
                    # Possible performance: type stability!
                    lm_seq = λ
                    om = exp.(logf_seq .- logf_m) ./ lm_seq
                end

                max_xk = x_seq[max_k]
                om_J = (1 - exp(dlogf_tl * (max_xk - tl) - gap)) / dlogf_tl
                push!(om, om_J)

                gap, tr, exp_tr, logf_tr, dlogf_tr = _gap2(mmm + sqrt(-2 / ddlogf_m), logf_m, λ, β)

                pr = tr + gap / dlogf_tr

                push!(om, pr - pl, -1 / dlogf_tr)
                om_cum = cumsum(om)
                om_sum = sum(om) # Possible performance: sometimes this is negative!?

                if om_sum < zero(om_sum)
                    @show om
                end

                count, safety = 0, 100_000
                while true
                    U = rand(rng, Distributions.Uniform(zero(om_sum), om_sum))
                    k = findfirst(>(U), om_cum)

                    if isone(k)
                        X = x_seq[k] + log(U / om[k]) / lm_seq[k]
                        expX = exp(X)
                        condition = λ * X - expX * (expX + β) - logf_seq[k] - lm_seq[k] * (X - x_seq[k])
                    elseif k <= max_k
                        tmp = exp(logf_seq[k - 1] - logf_seq[k])
                        X = x_seq[k] + log((U - om_cum[k - 1]) / om[k] * (1 - tmp) + tmp) / lm_seq[k]
                        expX = exp(X)
                        condition = λ * X - expX * (expX + β) - logf_seq[k] - lm_seq[k] * (X - x_seq[k])
                    elseif (k == max_k + 1)
                        tmp = exp(-dlogf_tl * (pl - max_xk))
                        X = pl + log((U - om_cum[k-1]) / om[k] * (1 - tmp) + tmp) / dlogf_tl
                        expX = exp(X)
                        condition = λ * X - expX * (expX + β) - logf_m - dlogf_tl * (X - pl)
                    elseif k == max_k + 2
                        X = pl + (pr - pl) * (U - om_cum[k - 1]) / om[k]
                        expX = exp(X)
                        condition = λ * X - expX * (expX + β) - logf_m
                    else
                        X = pr - om[k] * log((om_cum[k] - U) / om[k])
                        expX = exp(X)
                        condition = λ * X - expX * (expX + β) - logf_m - dlogf_tr * (X - pr)
                    end

                    if log(rand(rng)) <= condition
                        return expX / root_α
                    end
                    count += 1
                    count >= safety && _error_rand_Gao_Wang()
                end

            else

                rho_ast = logf_xast - λ * xast
                max_k   = ceil(Int, rho_ast)
                rho_seq = range(2 * rho_ast / max_k, 2 * rho_ast, length = max_k)
                exp_x   =  rho_seq ./ (sqrt.(β² .- 2 .* rho_seq) .- β)
                x_seq = log.(exp_x)
                x_seq[max_k] = xast
                logf_seq = λ .* x_seq .- exp_x.* (exp_x .+ β)

                if max_k > one(max_k)
                    lm_seq = diff(logf_seq) ./ diff(x_seq)
                    om = diff(exp.(logf_seq .- logf_m))
                    lm_seq = pushfirst!(lm_seq, λ)
                    om = [exp(logf_seq[1] - logf_m); om] ./ lm_seq
                else
                    lm_seq = λ
                    om = exp.(logf_seq .- logf_m) ./ lm_seq
                end
                max_xk = xast

                gap, tr, exp_tr, logf_tr, dlogf_tr = _gap2(mmm + sqrt(-2 / ddlogf_m), logf_m, λ, β)

                pr = tr + gap / dlogf_tr

                push!(om, pr - max_xk, -1 / dlogf_tr)
                om_cum = cumsum(om)
                om_sum = sum(om)

                # Possible performance: this looks identical to the sampler in the statement above!
                count, safety = 0, 100_000
                while true
                    U = rand(rng, Distributions.Uniform(zero(om_sum), om_sum))
                    k = findfirst(>(U), om_cum)

                    if isone(k)
                        X = x_seq[k] + log(U / om[k]) / lm_seq[k]
                        expX = exp(X)
                        condition = λ * X - expX * (expX + β) - logf_seq[k] - lm_seq[k] * (X - x_seq[k])
                    elseif k <= max_k
                        tmp = exp(logf_seq[k - 1] - logf_seq[k])
                        X = x_seq[k] + log((U - om_cum[k - 1]) / om[k] * (1 - tmp) + tmp) / lm_seq[k]
                        expX = exp(X)
                        condition = λ * X - expX * (expX + β) - logf_seq[k] - lm_seq[k] * (X - x_seq[k])
                    elseif (k == max_k + 1)
                        X = max_xk + (pr - max_xk) * (U - om_cum[k-1]) / om[k]
                        expX = exp(X)
                        condition = λ * X - expX * (expX + β) - logf_m
                    else
                        X = pr - om[k] * log((om_cum[k] - U) / om[k])
                        expX = exp(X)
                        condition = λ * X - expX * (expX + β) - logf_m - dlogf_tr * (X - pr)
                    end

                    if log(rand(rng)) <= condition
                        return expX / root_α
                    end

                    count += 1
                    count >= safety && _error_rand_Gao_Wang()
                end

            end
        end

    end
end

condition_sun_alg_1(α, β, γ) = α > one(α) && β > zero(β) && γ > zero(γ)

function compute_log_K1(μ, α, β, γ)
    α_min_1 = α - one(α)
    return (log(2) + log(pi) / 2 + (α - 1) * log(
        sqrt(β) * α_min_1 / (2β * μ - γ)
    ) + (
        -α_min_1 + β * abs2(μ)
    ))
    # this ignores the Fox-Wright term common to both K1 and K2
end
function compute_log_K2(δ, α, β, γ)
    return α * log(β) / 2 + SpecialFunctions.loggamma(α / 2) + abs2(γ) / (4 * (β - δ)) - α * log(δ) / 2
    # this ignores the Fox-Wright term common to both K1 and K2
end

function sun_alg_1(rng, α, β, γ)

    root_β = sqrt(β)
    γ = γ / root_β
    β = one(β)

    α_min_1 = α - one(α)
    μ_opt = (γ + sqrt(abs2(γ) + 8 * α_min_1 * β)) / 4β
    δ_opt = β + (abs2(γ) - γ * sqrt(abs2(γ) + 8α * β)) / 4α

    if iszero(β - δ_opt) || compute_log_K1(μ_opt, α, β, γ) > compute_log_K2(δ_opt, α, β, γ)

        for _ in 1:100_000
            X = Random.rand(rng, Distributions.Normal(μ_opt, 1 / 2β))
            U = Random.rand(rng)

            if X > zero(X) && log(U) < α_min_1 * log(X / μ_opt) + (2 * β * μ_opt - γ) * (μ_opt - X)
                return X * root_β
            end
        end

    else

        for _ in 1:100_000
            T = Random.rand(rng, Distributions.Gamma(α / 2, δ_opt))
            X = sqrt(T)
            U = rand(rng)

            if log(U) < -(β - δ_opt) * T + γ * X - abs2(γ) / (4 * (β - δ_opt))
                return X * root_β
            end
        end
    end
    _error_rand_Gao_Wang()
end
