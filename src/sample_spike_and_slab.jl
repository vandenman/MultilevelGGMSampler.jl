
# Would be nice to reuse the main part here between spike and slab and graphical horseshoe

function sample_individual_structure!(
    rng::Random.AbstractRNG,
    individual_state,
    individual_structure_internal::SpikeAndSlabStructureInternal,
    group_structure_internal::AbstractGroupStructureInternal,
    group_state
)

    # the update step for the group structure could be pulled out of the update for the Ks
    if individual_structure_internal.threaded
        sample_ss_threaded(rng, individual_state, individual_structure_internal, group_structure_internal, group_state)
    else
        sample_ss_serial(rng, individual_state, individual_structure_internal, group_structure_internal, group_state)
    end

end

function sample_ss_serial(
    rng::Random.AbstractRNG,
    individual_state,
    individual_structure_internal::SpikeAndSlabStructureInternal{<:Any, <:Any, <:Any, SS_METHOD, INV_METHOD},
    group_structure_internal::AbstractGroupStructureInternal,
    groupState
) where {SS_METHOD <: SS_SamplingMethod, INV_METHOD <: SS_InversionMethod}

    p = size(individual_state.Ks, 1)
    k = size(individual_state.Ks, 3)
    σ_prior_mat = similar(individual_state.Ks, p, p)
    cache = setup_cache(individual_state, SS_METHOD, INV_METHOD)

    πG = createGraphDistribution(group_structure_internal, groupState, k)
    # Infiltrator.@infiltrate
    # suffstats_groupstate = compute_suffstats_groupstate(πG, individual_state.Gs)
    suffstats_groupstate = compute_suffstats_groupstate(πG, individual_state.Gs, group_structure_internal)

    # suffstats_groupstate = compute_suffstats_groupstate(πG, individual_state.Gs, group_structure_internal)

    for ik in axes(individual_state.Ks, 3)

        setup_σ_prior_mat!(σ_prior_mat, ik, individual_state, individual_structure_internal)

        sample_ss_precmat_one_participant!(rng, ik, individual_state, σ_prior_mat, individual_structure_internal, cache, SS_METHOD, INV_METHOD)
        # sample_ss_indicators_one_participant!(rng, ik, individual_state, suffstats_groupstate, πG, individual_structure_internal)
        sample_ss_indicators_one_participant_logit!(rng, ik, individual_state, suffstats_groupstate, πG, individual_structure_internal)

    end
end

function sample_ss_threaded(
    rng::Random.AbstractRNG,
    individual_state,
    individual_structure_internal::SpikeAndSlabStructureInternal{<:Any, <:Any, <:Any, SS_METHOD, INV_METHOD},
    group_structure_internal::AbstractGroupStructureInternal,
    groupState
) where {SS_METHOD <: SS_SamplingMethod, INV_METHOD <: SS_InversionMethod}

    p = size(individual_state.Ks, 1)
    k = size(individual_state.Ks, 3)

    πG = createGraphDistribution(group_structure_internal, groupState, k)
    # suffstats_groupstate = compute_suffstats_groupstate(πG, individual_state.Gs)
    suffstats_groupstate = compute_suffstats_groupstate(πG, individual_state.Gs, group_structure_internal)
    @assert isnothing(suffstats_groupstate) || length(suffstats_groupstate) == k
    # the parallization is based on # https://julialang.org/blog/2023/07/PSA-dont-use-threadid/#better_fix_work_directly_with_tasks

    # customize this as needed. More tasks have more overhead, but better load balancing
    tasks_per_thread = 5#1
    chunk_size = max(1, k ÷ (tasks_per_thread * Threads.nthreads()))
    data_chunks = Iterators.partition(axes(individual_state.Ks, 3), chunk_size)

    blas_threads = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)

    tasks = map(data_chunks) do chunk

        Threads.@spawn begin

            σ_prior_mat = similar(individual_state.Ks, p, p)
            cache = setup_cache(individual_state, SS_METHOD, INV_METHOD)

            for ik in chunk

                setup_σ_prior_mat!(σ_prior_mat, ik, individual_state, individual_structure_internal)

                sample_ss_precmat_one_participant!(rng, ik, individual_state, σ_prior_mat, individual_structure_internal, cache, SS_METHOD, INV_METHOD)
                # sample_ss_indicators_one_participant!(rng, ik, individual_state, suffstats_groupstate, πG, individual_structure_internal)
                sample_ss_indicators_one_participant_logit!(rng, ik, individual_state, suffstats_groupstate, πG, individual_structure_internal)

            end
        end
    end

    fetch.(tasks)

    LinearAlgebra.BLAS.set_num_threads(blas_threads)

end

#region sample precision matrix
function sample_ss_precmat_one_participant!(rng, ik, individual_state, σ_prior_mat, individual_structure_internal, cache, ::Type{DirectSampling}, ::Type{<:SS_InversionMethod})

    Ks = individual_state.Ks
    Gs = individual_state.Gs
    S = individual_structure_internal.sum_of_squares
    n = individual_structure_internal.n

    σ_slab  = individual_structure_internal.σ_slab
    σ_spike = individual_structure_internal.σ_spike
    λ       = individual_structure_internal.λ


    (; r, v12, ω12, μ_ω12, L_ω12, invK, K11inv, Δ, U, temp1, temp2) = cache

    copyto!(invK, view(Ks, :, :, ik))
    chol = LinearAlgebra.cholesky!(invK)
    LinearAlgebra.inv!(chol)

    for ip in axes(Ks, 2)

        r[ip] = false
        r[isone(ip) ? size(Ks, 1) : ip - 1] = true

        # Q: don't we update only two spots in v12?? A: No, there can be more spots.
        c = 1
        for jp in eachindex(r)
            !r[jp] && continue
            v12[c] = isone(Gs[jp, ip, ik]) ? σ_slab[jp]^2 : σ_spike[jp]^2
            c += 1
        end
        # copy_by_r!(v12, view(σ_prior_mat, ip), r)

        s12 = view(S, r, ip, ik)
        s22 = S[ip, ip, ik]

#=
        K11inv = LinearAlgebra.Symmetric(LinearAlgebra.inv(LinearAlgebra.Symmetric(view(Ks, r, r))))
        C = inv((s22 + λ[ip]) * K11inv + LinearAlgebra.inv(LinearAlgebra.Diagonal(v12)))
        μ = -C * s12
        ω12 = rand(rng, Distributions.MvNormal(μ, C))

        v = rand(rng, Distributions.Gamma(n / 2 + 1, 2 / (s22 + λ[ip])))
        ω22 = v + LinearAlgebra.dot(ω12, K11inv, ω12)

        Ks[ip, r, ik] = ω12
        Ks[r, ip, ik] = ω12
        Ks[ip, ip, ik] = ω22
=#


        GGMSampler.inv_submatrix!(K11inv, invK, ip)

        # check = LinearAlgebra.UpperTriangular(K11inv) ≈ LinearAlgebra.UpperTriangular(inv(LinearAlgebra.Symmetric(Ks[r, r, ik])))
        # if !check
        #     Kk = Ks[:, :, ik]
        #     @show invK, K11inv, ip, Ks[r, r, ik], Kk
        #     @assert check
        # end
        # @assert LinearAlgebra.isposdef(LinearAlgebra.Symmetric(K11inv))

        # inv(a[p] * inv(K[-p, -p]) + Diagonal(1 ./ v[-p]))

        # L_ω12 .= (s22 + λ[ip]) .* K11inv .+ LinearAlgebra.Diagonal(1 ./ v12)
        for ii in eachindex(L_ω12, K11inv)
            L_ω12[ii] = (s22 + λ[ip]) * K11inv[ii]
        end
        for ii in axes(L_ω12, 1)
            L_ω12[ii, ii] += 1 / v12[ii]
        end

        LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(L_ω12)) # Q: why does this sometimes fail? A: Numerical accuracy
        LinearAlgebra.LAPACK.trtri!('U', 'N', L_ω12)

        uL_ω12 = LinearAlgebra.UpperTriangular(L_ω12)
        # @assert uL_ω12 * uL_ω12' ≈ LinearAlgebra.Symmetric(inv(LinearAlgebra.Symmetric(
        #     (s22 + λ[ip]) * K11inv + LinearAlgebra.Diagonal(1 ./ v12)
        # )))

        LinearAlgebra.mul!(ω12, uL_ω12', -s12) # Q: does the - allocate? A: Yes. Possible performance: use mul! with a alpha=-1, as in the line below
        # LinearAlgebra.mul!(ω12, uL_ω12',  s12, -1, false)
        LinearAlgebra.mul!(μ_ω12, uL_ω12, ω12)

        # Σ_ω12 = LinearAlgebra.Symmetric(inv(LinearAlgebra.Symmetric(
        #     (s22 + λ[ip]) * K11inv + LinearAlgebra.Diagonal(1 ./ v12)
        # )))
        # Distributions.rand!(rng, Distributions.MvNormal(μ_ω12, Σ_ω12), ω12)
        # @assert μ_ω12 ≈ -Σ_ω12 * s12
        # if !check
        #     @show μ_ω12, Σ_ω12, s12, uL_ω12
        # end

        # based on https://github.com/JuliaStats/Distributions.jl/blob/630e1c908b82bc2702cc7f29616cd22e5f94ab04/src/multivariate/mvnormal.jl#L276-L280
        Random.randn!(rng, ω12)
        LinearAlgebra.lmul!(LinearAlgebra.UpperTriangular(L_ω12), ω12)
        ω12 .+= μ_ω12

        # Q: could also preallocate the samplers since everything here is constant?
        # A: yes, it safes an if statement, and a type stability. But it's not a big deal
        v = rand(rng, Distributions.Gamma(n / 2 + 1, 2 / (s22 + σ_prior_mat[ip, ip])))

        ω22 = v + LinearAlgebra.dot(ω12, LinearAlgebra.Symmetric(K11inv), ω12)

        # the statement below the loop is equivalent, but the broadcasting allocates
        lp = 1
        for kp in axes(Ks, 2)
            !r[kp] && continue

            Δ[kp] = ω12[lp] - Ks[ip, kp, ik]
            lp += 1

        end
        # Δ[r]  .= ω12 .- Ks[ip, r, ik]
        Δ[ip] = (ω22 - Ks[ip, ip, ik]) / 2
        # Δ[ip] = (K1[ip, ip] - K0[ip, ip]) / 2

        # K0 = Ks[:, :, ik]
        Ks[ip, r, ik] = ω12
        Ks[r, ip, ik] = ω12
        Ks[ip, ip, ik] = ω22
        # K1 = Ks[:, :, ik]

        # @assert LinearAlgebra.isposdef(LinearAlgebra.Symmetric(K1))


        _update_U!(U, Δ, ip)

        # For some reason, this is not always numerically accurate
        _woodbury_update_rank_2_sym!(invK, U, temp1, temp2)

        # check = invK ≈ inv(Ks[:, :, ik])
        # if !check
        #     @show invK0, U0#, temp10, temp20
        #     # _woodbury_update_rank_2_sym!(invK0, U0, temp10, temp20)
        #     @assert invK ≈ inv(LinearAlgebra.Symmmetric(Ks[:, :, ik]))
        # end

    end

end

"""
Express symmetric update as rank two update.

Let `A` be a p by p symmetric matrix, `Δ` be a vector of difference values of length p, `U` a preallocated matrix of size p x 2,
and `j` an index in 1:p. Then `_update_U!(U, Δ, j)` fills in U such that this update
```
temp = A[j, j]
A[:, j] .+= Δ
A[j, :] .+= Δ
```
is equivalent to
```
P = [
    0   1
    1   0
]
A += U * P * U'
```
This is useful for rank two updates of inverses and Cholesky decompositions.

```examples
A = randn(4, 4)
Δ = randn(4)
j = 3

U = similar(A, 4, 2)
GGMSampler._update_U!(U, Δ, j)

A2 = copy(A)
A2[:, j] += Δ
A2[j, :] += Δ

P = [
    0   1
    1   0
]
A + U * P * U' ≈ A2

```
"""
function _update_U!(U, Δ, j)
    # @inbounds begin
        U[:, 2] .= Δ
        for i in 1:j-1
            U[i, 1]  = 0.0
        end
        U[j, 1]  = 1.0
        for i in j+1:size(U, 1)
            U[i, 1]  = 0.0
        end
    # end
end

function sample_ss_precmat_one_participant!(rng, ik, individual_state, σ_mat, individual_structure_internal, cache, ::Type{CholeskySampling}, ::Type{Direct_Inv})

    # unpack cache
    Ks = individual_state.Ks
    Gs = individual_state.Gs
    Ls = individual_state.Ls


    n       = individual_structure_internal.n
    S       = view(individual_structure_internal.sum_of_squares, :, :, ik)

    (new_v_0, potential_vector_0, precision_matrix_0, precision_pseudo_chol_0, _, _, _) = cache

    L_current = view(Ls, :, :, ik)
    p = size(L_current, 1)

    for row_idx in axes(L_current, 1)

        not_i = 1:p .!= row_idx
        rr = range(stop = row_idx - 1)
        @views begin
            A1 = L_current[not_i, rr]
            A2 = L_current[row_idx+1:p, row_idx]
            A3 = L_current[row_idx+1:p, rr]

            D1 = LinearAlgebra.Diagonal(σ_mat[not_i,       row_idx])
            D2 = LinearAlgebra.Diagonal(σ_mat[row_idx+1:p, row_idx])
        end

        old_u = L_current[row_idx, row_idx]

        if row_idx > 1

            # to examine the code
            # Infiltrator.@infiltrate

            # without adjusting the scaling w.r.t n
            # precision_matrix = LinearAlgebra.Symmetric((S[row_idx, row_idx] + σ_mat[row_idx, row_idx]) * LinearAlgebra.I + 2 * A1' * D1 * A1)
            # potential_vector = -(2 * A3' * D2 * A2 * old_u + A1' * S[row_idx, not_i])

            # new_v = rand(rng, Distributions.MvNormal(potential_vector, precision_matrix))
            # new_v = precision_matrix \ new_v

            # decreases with n
            # StatsBase.mean_and_var(vec(precision_matrix \ rand(Distributions.MvNormal(potential_vector, precision_matrix), 1000)))
            # increases with n
            # StatsBase.mean_and_var(vec(rand(Distributions.MvNormal(potential_vector, precision_matrix), 1000)))

            precision_matrix = (1 / n) * LinearAlgebra.I * LinearAlgebra.Symmetric((S[row_idx, row_idx] + σ_mat[row_idx, row_idx]) * LinearAlgebra.I + 2 * A1' * D1 * A1)
            potential_vector = -(2 * A3' * D2 * A2 * old_u + A1' * S[row_idx, not_i]) ./ sqrt(n)

            # precision_matrix \ potential_vector
            # (precision_matrix2) \ potential_vector2 .* (1 / sqrt(n))
            # inv(precision_matrix) * precision_matrix * inv(precision_matrix)

            # inv(precision_matrix / sqrt(n)) * precision_matrix / n * inv(precision_matrix / sqrt(n))

            # inv(sqrt(n) .* precision_matrix2) * precision_matrix2 * inv(sqrt(n) .* precision_matrix2)
            # inv(sqrt(n) .* precision_matrix2) * potential_vector2

            new_v = rand(rng, Distributions.MvNormal(potential_vector, precision_matrix))
            new_v = (precision_matrix \ new_v) .* (1 / sqrt(n))

            # decreases with n
            # StatsBase.mean_and_var(vec(precision_matrix2 \ rand(Distributions.MvNormal(potential_vector, precision_matrix), 1000) .* (1 / sqrt(n))))
            # constant-ish with n
            # StatsBase.mean_and_var(vec(                    rand(Distributions.MvNormal(potential_vector, precision_matrix), 1000)))

            L_current[row_idx, rr] .= new_v

        end

        new_v = L_current[row_idx, rr]
        L_current[row_idx, row_idx] = sample_cholesky_diagonal(rng, new_v, A2, A3, D2, σ_mat, S, n, p, row_idx)

    end

    # update K
    LinearAlgebra.mul!(view(Ks, :, :, ik), LinearAlgebra.LowerTriangular(L_current), LinearAlgebra.LowerTriangular(L_current)')

end

function sample_cholesky_diagonal(rng, new_v, A2, A3, D2, σ_mat, S, n, p, row_idx)

    α = n + (p - row_idx + 1)
    # β =  (1 / 2) * S[row_idx, row_idx] + 1 / σ_mat[row_idx, row_idx] + 2 * LinearAlgebra.dot(A2, D2, A2)
    β =  (1 / 2) * (S[row_idx, row_idx] + σ_mat[row_idx, row_idx] + 2 * LinearAlgebra.dot(A2, D2, A2))
    γ = -(1 / 2) * (4 * A2' * D2 * A3 * new_v + 2 * LinearAlgebra.dot(view(S, row_idx+1:p, row_idx), A2))

    return rand(rng, GGMSampler.ModifiedHalfNormal(α, β, γ))
end

function sample_cholesky_diagonal(rng, new_v, A2, A3, D2, σ_mat, S, n, p, row_idx, temp_mem)

    α = n + (p - row_idx + 1)
    # β =  (1 / 2) * S[row_idx, row_idx] + 1 / σ_mat[row_idx, row_idx] + 2 * LinearAlgebra.dot(A2, D2, A2)
    β =  (1 / 2) * (S[row_idx, row_idx] + σ_mat[row_idx, row_idx] + 2 * LinearAlgebra.dot(A2, D2, A2))
    LinearAlgebra.mul!(temp_mem, A3,  new_v)
    for i in eachindex(temp_mem)
        temp_mem[i] *= D2[i, i]
    end
    temp_val = LinearAlgebra.dot(A2', temp_mem)
    γ  = -(1 / 2) * (4 * temp_val              + 2 * LinearAlgebra.dot(view(S, row_idx+1:p, row_idx), A2))

    return rand(rng, GGMSampler.ModifiedHalfNormal(α, β, γ))
end

function sample_ss_precmat_one_participant!(rng, ik, individual_state, σ_mat, individual_structure_internal, cache, ::Type{CholeskySampling}, ::Type{CG_Inv})

    Ks = individual_state.Ks
    Gs = individual_state.Gs
    Ls = individual_state.Ls

    S = view(individual_structure_internal.sum_of_squares, :, :, ik)
    # S = individual_structure_internal.sum_of_squares
    n = individual_structure_internal.n

    L_current = view(Ls, :, :, ik)
    @assert LinearAlgebra.istril(L_current)
    # @show L_current

    # unpack cache

    (new_v_0, potential_vector_0, precision_matrix_0, precision_pseudo_chol_0, tempA, tempB, temp_randn,
        solver) = cache

    p = size(L_current, 1)

    for row_idx in axes(L_current, 1)

        not_i = 1:p .!= row_idx
        rr = range(stop = row_idx - 1)
        @views begin
            A1 = L_current[not_i, rr]
            A2 = L_current[row_idx+1:p, row_idx]
            A3 = L_current[row_idx+1:p, rr]

            D1 = LinearAlgebra.Diagonal(σ_mat[not_i,       row_idx])
            D2 = LinearAlgebra.Diagonal(σ_mat[row_idx+1:p, row_idx])
        end

        old_u = L_current[row_idx, row_idx]

        # These are a bit ugly, but necessary for performance
        A1a  = view(L_current, rr,            rr)
        A1b  = view(L_current, row_idx + 1:p, rr)

        # D1a = Diagonal(@view σ_mat[rr, row_idx])
        # D1b = Diagonal(@view σ_mat[row_idx + 1:p, row_idx])

        # setup memory helpers
        new_v = view(new_v_0, rr)
        rr2 = range(stop = (row_idx - 1)^2)
        potential_vector = view(potential_vector_0, rr)
        precision_matrix = reshape(view(precision_matrix_0, rr2), row_idx - 1, row_idx - 1)
        # precision_matrix = view(precision_matrix_0, rr, rr)
        precision_pseudo_chol = reshape(view(precision_pseudo_chol_0, range(stop = (row_idx - 1) * (p - 1))), row_idx - 1, p - 1)

        solver = Krylov.CgSolver(
            reshape(view(tempA, rr2), row_idx - 1, row_idx - 1),
            view(tempB, rr)
        )

        #= the main idea is given N(mean, sd) do
            sample z₁ ~ N(0, 1)
                then A1' * sqrt.(D1) * z₁ ~ N(0, A1' * D1 * A1)
            sample z₂ ~ N(0, S[row_idx, row_idx] + σ_mat[row_idx, row_idx])
            then z₁ + A1' * sqrt.(D1) * z₁ ~ N(0, S[row_idx, row_idx] + σ_mat[row_idx, row_idx] + A1' * D1 * A1)
            let C = inv(S[row_idx, row_idx] + σ_mat[row_idx, row_idx] + A1' * D1 * A1)
            then
                C * (z₁ + A1' * sqrt.(D1) * z₁) ~ N(0, C)
        =#
        # temp_randn_v serves as helper memory here
        temp_randn_v   = view(temp_randn, eachindex(new_v))
        temp_potential = view(temp_randn, eachindex(A2))

        LinearAlgebra.mul!(temp_potential, D2, A2)
        LinearAlgebra.mul!(potential_vector, LinearAlgebra.LowerTriangular(A1a)', view(S, row_idx, rr))
        LinearAlgebra.mul!(potential_vector, A1b', view(S, row_idx, row_idx+1:p), one(eltype(potential_vector)), one(eltype(potential_vector)))
        LinearAlgebra.mul!(potential_vector, A3', temp_potential, -2old_u, -one(eltype(potential_vector)))
        potential_vector

        # at this point we have
        # @assert isapprox(potential_vector, -(2 * A3' * D2 * A2 * old_u + A1' * S[row_idx, not_i]), atol = 1e-5)
        # Infiltrator.@infiltrate !isapprox(potential_vector, -(2 * A3' * D2 * A2 * old_u + A1' * S[row_idx, not_i]), atol = 1e-5)

        # sample temp_randn ~ N(0, D1)
        Random.randn!(temp_randn)
        for i in eachindex(temp_randn)
            temp_randn[i] *= sqrt(D1[i, i]) / sqrt(n)
        end

        LinearAlgebra.mul!(new_v, LinearAlgebra.LowerTriangular(A1a)', view(temp_randn, rr))
        LinearAlgebra.mul!(new_v, A1b',                  view(temp_randn, row_idx:p - 1), one(eltype(new_v)), one(eltype(new_v)))
        # at this point we have
        # @assert new_v ≈ A1' * temp_randn
        # Random.randn!(rng, temp_randn_v)
        for i in eachindex(new_v)
            new_v[i] += sqrt(S[row_idx, row_idx] + σ_mat[row_idx, row_idx]) / sqrt(n) * randn(rng) + potential_vector[i] / sqrt(n)
            # new_v[i] += sqrt(S[row_idx, row_idx] + σ_mat[row_idx, row_idx]) * temp_randn_v[i] + potential_vector[i]
            # new_v[i] = muladd(sqrt(S[row_idx, row_idx] + σ_mat[row_idx, row_idx]), temp_randn_v[i], new_v[i] + potential_vector[i])
        end

        LinearAlgebra.mul!(precision_pseudo_chol, A1', D1, 2.0, 0.0)
        ppcA = view(precision_pseudo_chol, :, rr)
        ppcB = view(precision_pseudo_chol, :, row_idx:p-1)
        # fast_copy_lv!(precision_matrix, L_current) # CartesianIndices(precision_matrix) equals the view that is A1a

        fast_copy!(precision_matrix, L_current) # CartesianIndices(precision_matrix) equals the view that is A1a
        LinearAlgebra.lmul!(LinearAlgebra.UpperTriangular(ppcA), precision_matrix)

        # the above two lines are identical to
        # mul!(precision_matrix, UpperTriangular(ppcA), LowerTriangular(A1a))
        # but then mul! first copies A1a to precision matrix with copyto! and then uses in place matrix multiplication
        # the copyto! hits an inefficient path because IndexStyle(precision_matrix) == IndexLinear and IndexStyle(A1a) == IndexCartesian

        LinearAlgebra.mul!(precision_matrix, ppcB, A1b, one(eltype(precision_matrix)), one(eltype(precision_matrix)))

        # at this point we have
        # @assert precision_matrix ≈ 2 * A1' * D1 * A1

        # XT_D_X_lv!(precision, A1a, A1b, D1a, D1b)

        for i in axes(precision_matrix, 1)
            precision_matrix[i, i] += S[row_idx, row_idx] + σ_mat[row_idx, row_idx]
        end

        # at this point we have
        # @assert precision_matrix ≈ LinearAlgebra.Symmetric((S[row_idx, row_idx] + σ_mat[row_idx, row_idx]) * LinearAlgebra.I + 2 * A1' * D1 * A1)

        # copyto!(temp_mat, precision_matrix)
        # Preconditioners.UpdatePreconditioner!(preconditioner, temp_mat)
        # Krylov.cg!(solver, precision_matrix, new_v)#, M = preconditioner)#, view(L_current, row_idx, rr))

        old_v = view(L_current, row_idx, rr)
        precision_matrix ./= n
        # to dispatch to symv! and not gemv!, but this appears to be slower...
        # precision_matrix_sym = LinearAlgebra.Symmetric(precision_matrix)
        # Krylov.cg!(solver, precision_matrix_sym, new_v, old_v)
        Krylov.cg!(solver, precision_matrix, new_v, old_v)

        # if length(new_v) > 0 && !isapprox(new_v, precision_matrix * Krylov.solution(solver) ; atol = 1e-4)
        #     # fallback for when Krylov fails to converge:
        #     @show solver.stats.solved
        #     if !solver.stats.solved
        #         @show solver
        #     end
        #     u = LinearAlgebra.Symmetric(precision_matrix) \ new_v
        #     copyto!(new_v, u)
        # else
            copyto!(new_v, Krylov.solution(solver))
            new_v ./= sqrt(n)
        # end

        # if any(isnan, new_v)
        #     @show solver.stats.solved
        #     # @show precision_matrix, new_v, old_v
        # end


        L_current[row_idx, rr] .= new_v

        temp_mem = view(potential_vector_0, range(; stop = p - row_idx))
        L_current[row_idx, row_idx] = sample_cholesky_diagonal(rng, new_v, A2, A3, D2, σ_mat, S, n, p, row_idx, temp_mem)
        # L_current[row_idx, row_idx] = sample_cholesky_diagonal(rng, new_v, A2, A3, D2, σ_mat, S, n, p, row_idx)

    end

    # update K
    LinearAlgebra.mul!(view(Ks, :, :, ik), LinearAlgebra.LowerTriangular(L_current), LinearAlgebra.LowerTriangular(L_current)')

end
#endregion

#region sample indicators
function sample_ss_indicators_one_participant!(rng, ik, individual_state, suffstats_groupstate, πG, individual_structure_internal)

    Ks = individual_state.Ks
    Gs = individual_state.Gs

    p = size(Gs, 1)
    k = size(Gs, 3)

    σ_slab  = individual_structure_internal.σ_slab
    σ_spike = individual_structure_internal.σ_spike

    πGik  = conditional_graph_distribution(πG, Gs, ik) # the same thing for all ik for the Curie Weiss???
    for ip in 1:p-1, jp in ip+1:p

        lognum = Distributions.logpdf(Distributions.Normal(0.0, σ_slab[jp, ip]),  Ks[jp, ip, ik])
        logden = Distributions.logpdf(Distributions.Normal(0.0, σ_spike[jp, ip]), Ks[jp, ip, ik])

        log_prior_ratio = log_inclusion_prob_prior_G(πGik, view(Gs, :, :, ik), ip, jp, suffstats_groupstate[k])
        # prob_1 = num / (num + den)
        # logprob_1 = log(1 / (1 + den / num))
        # prob_1 = num*p(num) / (num*p(num) + den*p(den))
        # logprob_1 = log(1 / (1 + (den*p(den)) / (num*p(num))))
        # logprob_1 = log(1 / (1 + den / num * p(den)/p(num)))
        logprob_1 = -LogExpFunctions.log1pexp(logden - lognum  - log_prior_ratio)

        z = log(rand(rng)) <= logprob_1
        update_compute_suffstats_groupstate!(suffstats_groupstate, πG, ik, Gs[ip, jp, ik], z)
        Gs[ip, jp, ik] = Gs[jp, ip, ik] = z
    end
end

function sample_ss_indicators_one_participant_logit!(rng, ik, individual_state, suffstats_groupstate, πG, individual_structure_internal)

    Ks = individual_state.Ks
    Gs = individual_state.Gs

    p = size(Gs, 1)
    σ_helper2   = individual_structure_internal.σ_helper2
    σ_log_ratio = individual_structure_internal.σ_log_ratio

    πGik  = conditional_graph_distribution(πG, Gs, ik)
    e_idx = 1

    for ip in 1:p-1, jp in ip+1:p

        #=
            derivation of what is computed here
            logit(num*π₁ / (num*π₁ + den*π₀))
            log(num*π₁ / den*π₀)
            log(num / den)  + log(π₁ / π₀)
            ↑ logit likehood  ↑ logit prior
        =#
        logit_likelihood_1 = σ_helper2[jp, ip] * abs2(Ks[jp, ip, ik]) + σ_log_ratio[jp, ip]

        logit_prior_1      = logit_inclusion_prob_prior_G(πGik, e_idx, suffstats_groupstate[ik] - Gs[jp, ip, ik], p)
        logit_prob_1       = logit_likelihood_1 + logit_prior_1

        z = rand(rng, Distributions.BernoulliLogit(logit_prob_1))

        update_compute_suffstats_groupstate!(suffstats_groupstate, πG, ik, Gs[ip, jp, ik], z)
        Gs[ip, jp, ik] = Gs[jp, ip, ik] = z

        e_idx += 1

    end
end

function sample_ss_indicators_one_participant_logit!(rng, ik, individual_state, suffstats_groupstate::Nothing, πG::IndependentGraphDistribution, individual_structure_internal)

    Ks = individual_state.Ks
    Gs = individual_state.Gs

    p = size(Gs, 1)
    σ_helper2   = individual_structure_internal.σ_helper2
    σ_log_ratio = individual_structure_internal.σ_log_ratio

    for ip in 1:p-1, jp in ip+1:p


        logit_prob_1    = σ_helper2[jp, ip] * abs2(Ks[jp, ip, ik]) + σ_log_ratio[jp, ip]
        z = rand(rng, Distributions.BernoulliLogit(logit_prob_1))

        #=

        lognum     = Distributions.logpdf(Distributions.Normal(0.0, individual_structure_internal.σ_slab[jp, ip]),  Ks[jp, ip, ik])
        logden     = Distributions.logpdf(Distributions.Normal(0.0, individual_structure_internal.σ_spike[jp, ip]), Ks[jp, ip, ik])
        log_prob_1 = -LogExpFunctions.log1pexp(logden - lognum)
        # last ones are equal
        @show log_prob_1, logit_prob_1, LogExpFunctions.logit(exp(log_prob_1))
        stop(error("end"))
        =#

        Gs[ip, jp, ik] = Gs[jp, ip, ik] = z

    end
end


function sample_ss_indicators_all_participants_logit!(
    rng::Random.AbstractRNG, individual_state,

    individualStructureInternal::SpikeAndSlabStructureInternal,#{<:Any, <:Any, <:Any, CholeskySampling, T},
    groupStructureInternal::AbstractGroupStructureInternal,
    groupState
)

    Ks = individual_state.Ks
    Gs = individual_state.Gs
    p = size(Ks, 1)
    k = size(Ks, 3)

    # precomputed coefficients of σ_slab and σ_spike
    σ_helper2   = individualStructureInternal.σ_helper2
    σ_log_ratio = individualStructureInternal.σ_log_ratio

    πG = createGraphDistribution(groupStructureInternal, groupState, k)
    suffstats_groupstate = compute_suffstats_groupstate(πG, Gs)

    # Possible performance: we could thread here, controlled by the group-level prior!
    for ik in axes(Ks, 3)
        πGik  = conditional_graph_distribution(πG, Gs, ik)
        for ip in 1:p-1, jp in ip+1:p

            log_prior_ratio = log_inclusion_prob_prior_G(πGik, view(Gs, :, :, ik), ip, jp, suffstats_groupstate[ik])
            logit_prob_1    = σ_helper2[jp, ip] * abs2(Ks[jp, ip, ik]) + σ_log_ratio[jp, ip] + log_prior_ratio
            z = rand(rng, Distributions.BernoulliLogit(logit_prob_1))

            update_compute_suffstats_groupstate!(suffstats_groupstate, πG, ik, Gs[ip, jp, ik], z)
            Gs[ip, jp, ik] = Gs[jp, ip, ik] = z
        end
    end
end


#endregion

#region setup_cache
function setup_cache(individual_state, ::Type{DirectSampling}, ::Type{<:SS_InversionMethod})

    Ks = individual_state.Ks
    p = size(Ks, 1)

    return (;
        r       = trues(p),
        v12     = similar(Ks, p - 1),

        ω12     = similar(Ks, p - 1),
        μ_ω12   = similar(Ks, p - 1),
        L_ω12   = similar(Ks, p - 1, p - 1),
        invK    = similar(Ks, p,     p),
        K11inv  = similar(Ks, p - 1, p - 1),

        # for low rank update of matrix inverse
        Δ       = similar(Ks, p),
        U       = zeros(eltype(Ks), p, 2),
        temp1   = zeros(eltype(Ks), p, 2),
        temp2   = zeros(eltype(Ks), 2, 2)
    )

end

function setup_cache(individual_state, ::Type{CholeskySampling}, ::Type{Direct_Inv})

    Ls = individual_state.Ls
    p = size(Ls, 1)

    tempA                     = similar(Ls, p,     p)
    tempB                     = similar(Ls, p)
    return (
        new_v                 = similar(Ls, p - 1),
        potential_vector      = similar(Ls, p - 1),
        precision_matrix      = similar(Ls, p - 1, p - 1),
        precision_pseudo_chol = similar(Ls, p - 1, p - 1),
        tempA                 = tempA,
        tempB                 = tempB,
        temp_randn            = similar(Ls, p - 1),
        solver                = Krylov.CgSolver(tempA, tempB)
    )

end

function setup_cache(individual_state, ::Type{CholeskySampling}, ::Type{CG_Inv})

    Ls = individual_state.Ls
    p = size(Ls, 1)

    tempA                     = similar(Ls, p,     p)
    tempB                     = similar(Ls, p)
    return (
        new_v                 = similar(Ls, p - 1),
        potential_vector      = similar(Ls, p - 1),
        precision_matrix      = similar(Ls, p - 1, p - 1),
        precision_pseudo_chol = similar(Ls, p - 1, p - 1),
        tempA                 = tempA,
        tempB                 = tempB,
        temp_randn            = similar(Ls, p - 1),
        solver                = Krylov.CgSolver(tempA, tempB)
    )

end
#endregion

#region setup_σ_prior_mat!
function setup_σ_prior_mat!(σ_prior_mat, ik, individual_state, individual_structure_internal)
    setup_σ_prior_mat!(
        σ_prior_mat,
        individual_structure_internal.σ_slab,
        individual_structure_internal.σ_spike,
        individual_structure_internal.λ,
        individual_state.Gs,
        ik
    )
end

function setup_σ_prior_mat!(σ_prior_mat, σ_slab::FillArrays.FillMatrix, σ_spike::FillArrays.FillMatrix, λ::FillArrays.FillVector, Gs, ik)
    for jp in axes(Gs, 2)
        # σ_prior_mat[jp, jp] = inv(λ[jp])
        σ_prior_mat[jp, jp] = λ[jp]
        for ip in jp+1:size(Gs, 1)
            if isone(Gs[ip, jp, ik])
                σ_prior_mat[ip, jp] = inv(abs2(σ_slab[ip, jp]))
            else
                σ_prior_mat[ip, jp] = inv(abs2(σ_spike[ip, jp]))
            end
            σ_prior_mat[jp, ip] = σ_prior_mat[ip, jp]
        end
    end
    return σ_prior_mat
end

function setup_σ_prior_mat!(σ_prior_mat, σ_slab::T, σ_spike::T, λ::T, Gs, ik) where T<: Number
    for jp in axes(Gs, 2)
        # σ_prior_mat[jp, jp] = inv(λ)
        σ_prior_mat[jp, jp] = λ
        for ip in jp+1:size(Gs, 1)
            if isone(Gs[ip, jp, ik])
                σ_prior_mat[ip, jp] = inv(abs2(σ_slab))
            else
                σ_prior_mat[ip, jp] = inv(abs2(σ_spike))
            end
            σ_prior_mat[jp, ip] = σ_prior_mat[ip, jp]
        end
    end
    return σ_prior_mat
end

function setup_σ_prior_mat(σ_slab, σ_spike, λ, Gs, ik)
    σ_prior_mat = Matrix{eltype(σ_slab)}(undef, size(Gs, 1), size(Gs, 2))
    setup_σ_prior_mat!(σ_prior_mat, σ_slab, σ_spike, λ, Gs, ik)
end
#endregion

#Possible performance: document why these are needed and include a benchmark (so that this can ideally be removed...)
# IIRC, it has to do with both A and B being views.
function fast_copy!(A, B)
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        A[i, j] = B[i, j]
    end
end
