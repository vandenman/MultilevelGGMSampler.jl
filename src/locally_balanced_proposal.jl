function locally_balanced_proposal_adjugate!!(
    log_Q::AbstractMatrix{T},Q::AbstractMatrix{T},
    K::AbstractMatrix{T},
    adj::AbstractMatrix{<:Integer},
    n_e::Integer,
    πG::Distributions.DiscreteDistribution,
    df_prior::Real,
    rate::AbstractMatrix{T},
    suffstats_groupstate::Integer,
    Letac::Bool = true
) where {T<:Real}

    #=
    Compute the locally balanced proposal from
    Zanella (2019, doi:10.1080/01621459.2019.1585255)
    =#

    #=
        Possible performance:

        - [x] consider using a log scale?
        - [x] is it even necessary to form the adjugate explicitly? could just do + logdet_K where needed
        - [x] do we even need the sign_ functions? A: yes we do need one of them
        - [x] use precomputed inv(K) & Phi
        - [x] no allocations
        - [x] inbounds
        - [ ] update Kinv and Kchol after updating K
                    - do we even need to update Kchol? the determinant remain unchanged by the transformation no?
                    - could also cache the determinant!
                    - actually, a permuted version of Kchol is updated so we do need to keep it around
                - [ ] lowrankupdate for Kinv
                - [ ] lowrankupdate for Kchol
        - [ ] paralell version over k to initialize the queue
    =#

    p = size(adj, 1)
    fill!(log_Q, T(-Inf))

    # Possible performance: pass group_structure_internal to reuse memory
    # suffstats_groupstate = compute_suffstats_groupstate(πG, parent(adj))#, group_structure_internal)
    # ik = last(parentindices(adj)) # or pass this directly?

    log_q_add = log_proposal_G_es(p, n_e + 1, n_e)
    log_q_rm  = log_proposal_G_es(p, n_e - 1, n_e)

    inv_K = inv(K)
    log_det_K = LinearAlgebra.logdet(K)
    # Tullio.@tullio log_adj_K[i, j] := log_det_K + log(abs(inv_K[i, j]))
    # sign_adj_K = 2 .* signbit.(inv_K) .- 1

    # inv_K_sub = Array{T}(undef, p-1, p-1)
    # log_adj_K_sub  = Array{eltype(K)}(undef, p-1, p-1)

    # for debugging on the non-log scale
    # adj_K = LinearAlgebra.det(big.(K)) * inv_K
    # adj_K_sub = Array{eltype(adj_K)}(undef, p-1, p-1)

    @inbounds for i in 1:p-1

        log_adj_K_ii = log_det_K + log(abs(inv_K[i, i]))
        # inv_K_sub .= (inv_K - inv_K[:, j1] * inv_K[:, j1]' / inv_K[j1, j1])[1:p .!= j1, 1:p .!= j1]
        # inv_submatrix!(inv_K_sub, inv_K, i)
        # log_adj_K_sub  .= log_adj_K[j1, j1] .+ log.(abs.(inv_K_sub))
        # Tullio.@tullio log_adj_K_sub[ii, jj] = log_adj_K[$j1, $j1] + log(abs(inv_K_sub[ii, jj]))

        # for debugging on the non-log scale
        # adj_K_sub .= inv_K_sub * adj_K[j1, j1]

        for j in i+1:p

            inv_K_sub_j1j1 = compute_inv_submatrix_i(inv_K, i, j, j)
            # @assert inv_K_sub[j - 1, j - 1] ≈ inv_K_sub_j1j1
            # inv_K_sub_j1j1 = inv_K_sub[j - 1, j - 1]
            log_adj_K_ij = log_det_K + log(abs(inv_K[i, j]))
            log_adj_K_jj = log_det_K + log(abs(inv_K[j, j]))
            log_abs_inv_K_sub_jj = log_adj_K_ii + log(abs(inv_K_sub_j1j1))
            # log_abs_inv_K_sub_jj = log_adj_K_ii + log(abs(inv_K_sub[j - 1, j - 1]))

            Phi_p1p1 = exp((log_adj_K_jj - log_abs_inv_K_sub_jj) / 2)
            Phi_pp1  = (2 * signbit(inv_K[i, j]) - 1) * exp(log_adj_K_ij - (log_adj_K_jj + log_abs_inv_K_sub_jj) / 2)

            # @assert Phi_p1p1 ≈ exp((log_adj_K[j, j] - log_abs_inv_K_sub_jj) / 2)
            # @assert Phi_pp1  ≈ sign_adj_K[i, j] * exp(log_adj_K[i, j] - (log_adj_K[j, j] + log_abs_inv_K_sub_jj) / 2)

            # Phi_p1p1 = exp((log_adj_K[j, j] - log_abs_inv_K_sub_jj) / 2)
            # Phi_p1p1 = exp((log_adj_K[j2, j2] - log_adj_K_sub[j2-1, j2-1]) / 2)
            # Phi_pp1  = sign_adj_K[i, j] * exp(log_adj_K[i, j] - (log_adj_K[j, j] + log_abs_inv_K_sub_jj) / 2)
            # Phi_pp1  = sign_adj_K[j1, j2] * exp(log_adj_K[j1, j2] - (log_adj_K[j2, j2] + log_adj_K_sub[j2-1, j2-1]) / 2)

            K_pp1    = K[i, j]

            # for debugging on the non-log scale
            # Phi_p1p1_big = sqrt(adj_K[j2, j2] / adj_K_sub[j2-1, j2-1])
            # Phi_pp1_big  = -adj_K[j1, j2] / sqrt(adj_K[j2, j2] * adj_K_sub[j2-1, j2-1])

            # if !(Float64(Phi_p1p1_big) ≈ Phi_p1p1) || !(Float64(Phi_pp1_big) ≈ Phi_pp1)
            #     @show i, j, Float64(Phi_p1p1_big), Phi_p1p1, Float64(Phi_pp1_big), Phi_pp1
            #     @assert Float64(Phi_p1p1_big) ≈ Phi_p1p1
            #     @assert Float64(Phi_pp1_big)  ≈ Phi_pp1
            # end

            if isone(adj[i, j])
                exponent = -1;
                log_Q[i, j] = log_q_rm;
            else
                exponent = 1;
                log_Q[i, j] = log_q_add;
            end

            # Infiltrator.@infiltrate

            log_Q[i, j] += log_balancing_function(exponent * (
                # TODO: just pass e_idx?
                log_inclusion_prob_prior_G(πG, adj, i, j, suffstats_groupstate) +
                # log_inclusion_prob_prior_G(πG, adj, i, j) +
                    log_N_tilde2(
                        Phi_p1p1,
                        Phi_pp1,
                        K_pp1,
                        rate[j, j],
                        rate[i, j]
                    ) + (Letac ? log_norm_ratio_Letac(adj, i, j, df_prior) : 0.0)
            ));

        end
    end

    normalize_Q!(log_Q, Q)

    return Q, log_Q

end

function locally_balanced_proposal_adjugate!!(
    log_Q::AbstractMatrix{T}, Q::AbstractMatrix{T},
    Kobj::KObj{T},
    adj::AbstractMatrix{<:Integer},
    n_e::Integer,
    πG::Distributions.DiscreteDistribution,
    df_prior::Real,
    rate::AbstractMatrix{T},
    Letac::Bool = true
) where {T<:Real}

    p = size(adj, 1)
    fill!(log_Q, T(-Inf))

    log_q_add = log_proposal_G_es(p, n_e + 1, n_e)
    log_q_rm  = log_proposal_G_es(p, n_e - 1, n_e)

    K     = Kobj.K
    inv_K = Kobj.Kinv
    Kchol = Kobj.Kchol
    log_det_K = 2 * LinearAlgebra.logdet(LinearAlgebra.LowerTriangular(Kchol))

    @inbounds for i in 1:p-1

        log_adj_K_ii = log_det_K + log(abs(inv_K[i, i]))

        for j in i+1:p

            inv_K_sub_j1j1 = compute_inv_submatrix_i(inv_K, i, j, j)
            log_adj_K_ij = log_det_K + log(abs(inv_K[i, j]))
            log_adj_K_jj = log_det_K + log(abs(inv_K[j, j]))
            log_abs_inv_K_sub_jj = log_adj_K_ii + log(abs(inv_K_sub_j1j1))

            Phi_p1p1 = exp((log_adj_K_jj - log_abs_inv_K_sub_jj) / 2)
            Phi_pp1  = (2 * signbit(inv_K[i, j]) - 1) * exp(log_adj_K_ij - (log_adj_K_jj + log_abs_inv_K_sub_jj) / 2)

            K_pp1    = K[i, j]

            if isone(adj[i, j])
                exponent = -1;
                log_Q[i, j] = log_q_rm;
            else
                exponent = 1;
                log_Q[i, j] = log_q_add;
            end

            log_Q[i, j] += log_balancing_function(exponent * (
                log_inclusion_prob_prior_G(πG, adj, i, j) +
                    + log_N_tilde2(
                        Phi_p1p1,
                        Phi_pp1,
                        K_pp1,
                        rate[j, j],
                        rate[i, j]
                    ) + (Letac ? log_norm_ratio_Letac(adj, i, j, df_prior) : 0.0)
            ));

        end
    end

    normalize_Q!(log_Q, Q)

    return Q, log_Q

end

function normalize_Q!(log_Q, Q)
    log_Q .-= maximum(log_Q)
    Q .= exp.(log_Q)
    sum_Q = sum(Q)
    Q ./= sum_Q
    log_Q .-= log(sum_Q)
end

"""
Computes a single element of inv_submatrix! at `[ii, jj]`.
The caller is responsible for ensuring that `ii` and `jj` are not equal to `i` and that they are in bounds.
"""

function compute_inv_submatrix_i(inv_K, i, ii, jj)
    return inv_K[ii, jj] - inv_K[ii, i] * inv_K[jj, i] / inv_K[i, i]
    # return @inbounds (inv_K[ii, jj] - inv_K[ii, i] * inv_K[jj, i] / inv_K[i, i])
end

"""
Compute the inverse of K without the ith column and row.
Based on the Sherman–Morrison formula and https://math.stackexchange.com/a/4100775/
Assumes that size(inv_K_sub) == size(inv_K) .- 1
Only computes the upper triangle for i >= j, since it assumes the result is symmetric.
"""
function inv_submatrix!(inv_K_sub, inv_K, i)

    # NOTE: iteration order is suboptimal!
    p = size(inv_K, 1)
    r1 = 1:i-1
    r2 = i+1:p
    @inbounds for ii in r1
        for jj in ii:i-1
            inv_K_sub[ii, jj] = (inv_K[ii, jj] - inv_K[ii, i] * inv_K[jj, i] / inv_K[i, i])
        end
        for jj in r2
            inv_K_sub[ii, jj-1] = (inv_K[ii, jj] - inv_K[ii, i] * inv_K[jj, i] / inv_K[i, i])
        end
    end
    @inbounds for ii in r2
        # for jj in r1
        #     inv_K_sub[ii-1, jj] = (inv_K[ii, jj] - inv_K[ii, i] * inv_K[jj, i] / inv_K[i, i])
        # end
        for jj in ii:p
            # @show ii, jj
            inv_K_sub[ii-1, jj-1] = (inv_K[ii, jj] - inv_K[ii, i] * inv_K[jj, i] / inv_K[i, i])
        end
    end
    return inv_K_sub
end

function inv_submatrix(inv_K::AbstractMatrix, i::Integer)
    inv_K_sub = similar(inv_K, size(inv_K, 1) - 1, size(inv_K, 2) - 1)
    return inv_submatrix!(inv_K_sub, inv_K, i)
end

# NOTE: unussed, only there for comparison with the other approaches
function locally_balanced_proposal(
    K::AbstractMatrix{<:Real},
    adj::AbstractMatrix{<:Integer},
    n_e::Integer,
    πG::Distributions.DiscreteDistribution,
    df_0::Real,
    rate::AbstractMatrix{<:Real},
    Letac::Bool = true
)

    #=
    Compute the locally balanced proposal from
    Zanella (2019, doi:10.1080/01621459.2019.1585255)
    =#
    p = size(adj, 1)
    log_Q = fill(-Inf64, p, p)
    # The matrix `Phi` is specified here to avoid repeated expensive memory (de)allocation.
    # blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi;
    Phi = similar(log_Q)

    log_q_add = log(proposal_G_es(p, n_e + 1, n_e))
    log_q_rm  = log(proposal_G_es(p, n_e - 1, n_e))

    # Fused triangular loop based on
    # https://stackoverflow.com/a/33836073/5216563 as OnepMP does not support
    # collapsing triangular loops.
    #pragma omp parallel for schedule(static) private(Phi)
    perm = Vector{Int}(undef, p)
    perm_inv = Vector{Int}(undef, p)
    for e_id in 1:p_to_ne(p)

        # this doesn't loop sequentially which is a bit odd. However, these two snippets seem to match
        # linear_index_to_triangle_index_0based.(0:p * (p - 1) ÷ 2 - 1, p)
        # linear_index_to_triangle_index.(1:p * (p - 1) ÷ 2, p)
        i, j = linear_index_to_triangle_index(e_id, p)

        perm, perm_inv = permute_e_last(i, j, p)
        K_perm = permute_mat(K, perm_inv)
        # @assert LinearAlgebra.isposdef(K_perm) && LinearAlgebra.isposdef(K)

        # There should be a way to do this without entirely recomputing the cholesky?
        # Phi = copy(K_perm)
        # LinearAlgebra.cholesky!(Phi)
        Phi = LinearAlgebra.cholesky(K_perm).L

        if isone(adj[i, j])
            exponent = -1;
            log_Q[i, j] = log_q_rm;
        else
            exponent = 1;
            log_Q[i, j] = log_q_add;
        end

        # @show i, j, log_Q[i, j], perm, perm_inv
        log_Q[i, j] += log_balancing_function(exponent * (
            log_inclusion_prob_prior_G(πG, adj, i, j) +
                + log_N_tilde(
                    Phi, rate[perm_inv[p],     perm_inv[p]],
                         rate[perm_inv[p - 1], perm_inv[p]]
                ) + (Letac ? log_norm_ratio_Letac(adj, i, j, df_0) : 0.0)
        ));
        # @show log_Q[i, j]
    end

    log_Q .-= maximum(log_Q)
    Q = exp.(log_Q)
    sum_Q = sum(Q)
    return Q ./ sum_Q, log_Q .- log(sum_Q)

end
