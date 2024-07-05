function update_G_wwa!!(
    rng::Random.AbstractRNG,
    K::AbstractMatrix,
    G::AbstractMatrix,
    rate::AbstractMatrix,
    df::Real,
    πG::Distributions.DiscreteDistribution,
    df_0::Real,
    n_edge_updates::Integer,
    # order of these should match wwaMCMC
    approx::Bool = false, delayed_accept::Bool = true,
    loc_bal::Bool = true, Letac::Bool = true,
    bufferData = nothing,
    threaded::Bool = false
)

    #=
            - [x] this function not should take a GWishart as argument
            - [ ] should this function take an adjacency matrix, or a graph?
                - if a Graph, DGW is also modified in place!
                - perhaps the function should just take K, DGW?
            - [ ] clearly distinghuish between adj::Matrix and G::SimpleGraph
            - [ ] adj should be modified in place -- already done?
            - [ ] everything that is posterior/ prior needs a postfix
            - [x] K should be modified in place
            - [x] this function should not return anything
            - [ ] use proper names for things
            - [x] add rng as argument!
    =#

    # df, rate, G = Distributions.params(DGW)
    # adj = Graphs.adjacency_matrix(G)

    Gr = Graphs.SimpleGraph(G) # can we do without this?
    n_e = Graphs.ne(Gr)

    # sample from ℙ(K | G, data)
    K .= Distributions.rand(rng, GWishart(df, rate, Gr))
    # Possible performance: this should not be necessary!
    for i in axes(K, 1), j in i+1:size(K, 2)
        K[i, j] = K[j, i]
    end
    # Distributions.rand!(GWishart(df, rate, Gr), K)

    @assert LinearAlgebra.isposdef(K)

    # sample from ℙ(G | K, data)
    for _ in 1:n_edge_updates

        n_e = update_single_edge(
            rng,
            K, G, n_e, πG, df, df_0, rate,
            approx, delayed_accept, loc_bal, Letac,
            bufferData,
            threaded
        )

    end

end

function update_single_edge(
    rng::Random.AbstractRNG,
    K::AbstractMatrix{<:Real},
    adj,
    n_e::Integer,
    πG::Distributions.DiscreteDistribution,
    df::Real,
    df_0::Real,
    rate::AbstractMatrix{<:Real},
    approx::Bool = false, delayed_accept::Bool = true,
    loc_bal::Bool = true, Letac::Bool = true,
    bufferData = nothing,
    threaded::Bool = false
)

    #=

        Possible performance:

            - [ ] this function could use a lot more subfunctions! (search subfunction)
            - [ ] add rng as argument!
            - [ ] add !! for mutation
            - [ ] does this even mutate K? if yes, then it should be updated accordingly in update_G_wwa!

    =#

    if approx && delayed_accept
        throw(ArgumentError("`approx` and `delayed_accept` cannot be true simultaneously."))
    end

    p = size(adj, 1)
    max_e = p_to_ne(p)
    accept = true

    # Q, log_Q, Phis = bufferData.Q, bufferData.log_Q, bufferData.Phis
    Q, log_Q = bufferData.Q, bufferData.log_Q

    if loc_bal
        locally_balanced_proposal_adjugate!!(
            log_Q, Q,
            K, adj, n_e, πG, df_0, rate, Letac
        )

        i, j = 1, 1
        tmp_sum = 0.0
        u = rand(rng)

        while tmp_sum < u
            if j == p
                i += 1
                j = i
            end
            j += 1
            tmp_sum += Q[i, j]
        end

        add = iszero(adj[i, j])
        log_q_y_x = log_Q[i, j]

    else

        # subfunction
        # Decide whether to propose an edge addition or removal.
        if iszero(n_e)
            add = true
        elseif n_e == max_e
            add = false
        else
            add = rand(rng) < 0.5
        end

        # int row_sum
        row_sum_cum = 0
        i = 0

        if add

            e_id = rand(rng, 1:max_e - n_e)
            i, j = 0, 0
            current_sum = 0
            for l in 1:p-1, k in l+1:p
                current_sum += (1 - adj[k, l])
                if current_sum == e_id
                    j, i = k, l
                    break
                end
            end
            @assert iszero(adj[i, j])

        else

            e_id = rand(rng, 1:n_e)
            i, j = 0, 0
            current_sum = 0
            for l in 1:p-1, k in l+1:p
                current_sum += adj[k, l]
                if current_sum == e_id
                    j, i = k, l
                    break
                end
            end
            @assert isone(adj[i, j])

        end

        n_e_tilde = n_e + 2 * add - 1
        log_q_y_x = log(proposal_G_es(p, n_e_tilde, n_e));
        log_q_x_y = log(proposal_G_es(p, n_e, n_e_tilde));

    end

    exponent = 2 * add - 1
    n_e_tilde = n_e + exponent

    # TODO: nothing is wrong here!
    K_perm, Phi, perm, perm_inv, rate_1p, rate_pp, log_prior_ratio, log_N_tilde_post  = common_stuff(i, j, exponent, K, adj, rate, df, πG, rng, nothing)
    Phi_12_cur = Phi[p, p - 1]

    if loc_bal
        # Compute the reverse transition probability `log_q_x_y`.

        # Update `Phi[p, p - 1]` according to the proposal.
        if add
            Phi[p, p - 1] = randn(rng) / sqrt(rate_pp) - Phi[p - 1, p - 1] * rate_1p / rate_pp
        else
            Phi[p, p - 1] = -sum(view(Phi, p-1, 1:p-2) .* view(Phi, p, 1:p-2)) / Phi[p - 1, p - 1]
            # Phi[p - 1, p - 2] = -sum(submatrix(Phi, p - 2, 0, 1, p - 2) % submatrix(Phi, p - 1, 0, 1, p - 2)) / Phi(p - 2, p - 2)
        end

        adj[i, j] = add
        adj[j, i] = add

        @assert LinearAlgebra.isposdef(K)
        update_K_from_Phi(p, j, perm, K, Phi)
        @assert LinearAlgebra.isposdef(K)

        locally_balanced_proposal_adjugate!!(
            log_Q, Q,
            K, adj, n_e, πG, df_0, rate, Letac
        )

        log_q_x_y = log_Q[i, j]

    end

    accept = compute_acceptance(
        i, j, adj, add, exponent, df_0, log_q_y_x, log_q_x_y, perm_inv,
        log_prior_ratio, log_N_tilde_post, delayed_accept, approx, Letac, rng
    )

    @assert LinearAlgebra.isposdef(K)

    if accept # Update the graph.
        adj[i, j] = add
        adj[j, i] = add
        n_e = n_e_tilde
    elseif loc_bal # Revert any update in `adj` and `Phi`.
        adj[i, j] = !add
        adj[j, i] = !add
        Phi[p, p - 1] = Phi_12_cur
    end

    if !loc_bal
        # Update `Phi[p, p - 1]`.
        if isone(adj[i, j])  # The graph contains (`i`, `j`) after updating.
            Phi[p, p - 1] = randn(rng) / sqrt(rate_pp) - Phi[p - 1, p - 1] * rate_1p / rate_pp
        else # The graph does not contain (`i`, `j`) after updating.
            Phi[p, p - 1] = -sum(
                    view(Phi, p-1, 1:p - 2) .* view(Phi, p, 1:p - 2)
                ) / Phi[p - 1, p - 1]
        end
    end

    update_K_from_Phi(p, j, perm, K, Phi)

    return n_e

end

function log_N_tilde(
    Phi::AbstractMatrix{<:Real},
    rate_perm_11::Real,
    rate_perm_21::Real
)
    #=
    The log of the function N from
    Cheng & Lengkoski (2012, page 2314, doi:10.1214/12-EJS746)
    `rate_perm_11` and `rate_perm_21` contain the element in the last row and
    column and the element just above it in the rate matrix, respectively.
    =#
    p = size(Phi, 1)

    return log(Phi[p - 1, p - 1]) + 0.5*(
        -log(rate_perm_11) + rate_perm_11 * (
            # Possible performance: could be written as an inner product? compare to dot()
            -(view(Phi, p-1, 1:p-2)' * view(Phi, p, 1:p-2)
            ) / Phi[p - 1, p - 1] + Phi[p - 1, p - 1] * rate_perm_21 / rate_perm_11
        )^2
    )

end

function log_N_tilde2(
    Phi_p1p1,
    Phi_pp1,
    K_pp1,
    rate_perm_11::Real,
    rate_perm_21::Real
)
    #=
    The log of the function N from
    Cheng & Lengkoski (2012, page 2314, doi:10.1214/12-EJS746)
    `rate_perm_11` and `rate_perm_21` contain the element in the last row and
    column and the element just above it in the rate matrix, respectively.
    =#

    return log(Phi_p1p1) + 0.5*(
        -log(rate_perm_11) + rate_perm_11 * (
            -(K_pp1 / Phi_p1p1 - Phi_pp1) + Phi_p1p1 * rate_perm_21 / rate_perm_11
        )^2
    )

end

function proposal_G_es(
    p::T,
    n_e_tilde::T,
    n_e::T
)#=::Real=# where T<:Integer

    # Proposal transition probability from `G` to `G_tilde` based on edge counts
    max_e = p_to_ne(p)

    if iszero(n_e) || n_e == max_e
        return 1.0 / max_e
    elseif n_e > n_e_tilde
        return 0.5 / n_e
    else
        return 0.5 / (max_e - n_e)
    end
end

function log_proposal_G_es(
    p::T,
    n_e_tilde::T,
    n_e::T
)#=::Real=# where T<:Integer

    # Log proposal transition probability from `G` to `G_tilde` based on edge counts
    max_e = p_to_ne(p)

    if iszero(n_e) || n_e == max_e
        return -log(max_e)
    elseif n_e > n_e_tilde
        return IrrationalConstants.loghalf - log(n_e)
    else
        return IrrationalConstants.loghalf - log(max_e - n_e)
    end
end

function permute_e_last!!(
    perm::Vector{T}, perm_inv::Vector{T},
    i::T, j::T, p::T
) where T<:Integer

    #=
    Permute the nodes such that edge `(i, j)` becomes edges (p - 1, p).
    This function returns the permutation and inverse permutation.
    =#
    @boundscheck begin
        length(perm) == length(perm_inv) == p || throw(BoundsError("Incorrect arguments passed to permute_e_last!!"))
    end

    copyto!(perm, 1:p)

    # Permute the nodes involved in `e`.
    @inbounds if i != p - 1

        perm[i] = p - 1

        if j == p - 1
            perm[p - 1] = p
            perm[p    ] = i
        else
            perm[p - 1] = i
            perm[p    ] = j
            perm[j] = p
        end
    end

    # This makes no sense... perm_inv == perm?
    @inbounds for l in eachindex(perm_inv)
        perm_inv[perm[l]] = l
    end

    return perm, perm_inv
end

function permute_e_last(
    i::T, j::T, p::T
) #= ::Tuple{Vector{Int}, Vector{Int}}=# where T<:Integer

    #=
    Permute the nodes such that edge `(i, j)` becomes edges (p - 1, p).
    This function returns the permutation and inverse permutation.
    =#

    perm = Vector{Int}(undef, p)
    perm_inv = Vector{Int}(undef, p)
    return permute_e_last!!(perm, perm_inv, i, j, p)

    # Permute the nodes involved in `e`.
    if i != p - 1

        perm[i] = p - 1;

        if j == p - 1
            perm[p - 1] = p;
            perm[p    ] = i;
        else
            perm[p - 1] = i;
            perm[p    ] = j;
            perm[j] = p;
        end
    end

    for l in 1:p
        perm_inv[perm[l]] = l
    end

    # if !allunique(perm_inv) || !allunique(perm)
    #     @warn "bad stuff" i, j, p
    # end

    return perm, perm_inv

end

function permute_mat(
    mat::AbstractMatrix{<:Real},
    perm::Vector{Int}
) # :: AbstractMatrix{<:Real}

    # Possible performance: what does this function do?!
    # it looks like it's just mat[perm, perm] in a very complicated way!

    # A = Matrix(reshape(1.:16, 4, 4))
    # A .= (A .+ A') ./ 2.0
    # permute_mat(A, [1, 2, 3, 4])
    # permute_mat(A, [1, 2, 4, 3])
    # permute_mat(A, [2, 3, 4, 1])

    from = Vector{Int}(undef, 0)
    to   = Vector{Int}(undef, 0)

    for i in eachindex(perm)
        if perm[i] != i
            push!(from, perm[i])
            push!(to,   i)
        end
    end

    perm_rows = mat[from, :]
    perm_rows[:, to] .= perm_rows[:, from]
    mat_perm = copy(mat)
    mat_perm[to, :] .= perm_rows
    mat_perm[:, to] .= perm_rows'
    # if mat_perm != mat[perm, perm]
        # @show perm, mat_perm, mat
        # println(perm)
        @assert mat_perm ≈ mat[perm, perm]
    # end
    return mat_perm

end

function log_norm_ratio_Letac(
    adj::AbstractMatrix{<:Integer}, i::Integer, j::Integer, df_prior::Real
) # :: Real
    #=
    Log of the approximation of the ratio of normalizing constants of the
    G-Wishart prior distributions from Letac et al. (2018, arXiv:1706.04416v2).
    The ratio is evaluated at the graph given by adjacency matrix `adj` with
    edge (`i`, `j`) absent (numerator) divided by the same graph with
    (`i`, `j`) present (denominator).
    `df_0` is the degrees of freedom of the G-Wishart prior distribution.
    =#
    # `n_paths` is the number of paths of length 2 that connect nodes `i` and
    # `j`.

    n_paths = LinearAlgebra.dot(view(adj, :, i), view(adj, :, j))

    v1, _ = SpecialFunctions.logabsgamma((df_prior + n_paths)       / 2)
    v2, _ = SpecialFunctions.logabsgamma((df_prior + (n_paths + 1)) / 2)
    return log(0.5) - 0.5 * log(pi) + v1 - v2
    # return IrrationalConstants.loghalf + IrrationalConstants.logπ / 2 + v1 - v2

end


function log_balancing_function(x::Real) # ::Real
    return -LogExpFunctions.log1pexp.(-x)
end

function linear_index_to_triangle_index_0based(e_id, p)

    i = e_id ÷ p
    j = e_id % p

    if i >= j
        i = p - i - one(p) - one(p)
        j = p - j - one(p)
    end
    return i, j
end

function linear_index_to_triangle_index(e_id, p)
    return linear_index_to_triangle_index_0based(e_id - one(e_id), p) .+ one(e_id)

    # i = (e_id ÷ p)# + 1
    # j = (e_id % p)

    # if i >= j
    #     i = p - i - 1
    #     j = p - j
    # end
    # return i, j
end

function update_K_from_Phi(
    p::Integer,
    j::Integer,
    perm::AbstractVector{<:Integer},
    K::AbstractMatrix{<:Real},
    Phi::AbstractMatrix{<:Real}
) # ::Nothing
    #=
    Update the precision matrix `K` in place according to a new Phi.

    The matrix `Phi` has been permuted such that indices `i` and `j` become
    indices p-1 and p. Matrix `K` is unpermuted.
    =#
    K_vec = Phi * view(Phi, p, :)
    K_vec = K_vec[perm];  # Undo the permutation.
    # Q: why not j+1? A: becaus the calling function should have adjusted that already
    K[:, j] .= K_vec
    K[j, :] .= K_vec # needs a transpose?
    return # avoid returning the last expression
end

function update_KObj_from_Phi(
    p::Integer,
    j::Integer,
    perm::AbstractVector{<:Integer},
    K_obj::KObj{<:Real},
    Phi::AbstractMatrix{<:Real}
) # ::Nothing
    #=
    Update the precision matrix `K` in place according to a new Phi.

    The matrix `Phi` has been permuted such that indices `i` and `j` become
    indices p-1 and p. Matrix `K` is unpermuted.
    =#

    # update K
    K_vec = Phi * view(Phi, p, :)
    K_vec = K_vec[perm];  # Undo the permutation.

    Δ = K_obj.K[:, j] - K_vec

    K_obj.K[:, j] .= K_vec
    K_obj.K[j, :] .= K_vec

    # Possible performance: low-rank updates!
    # @assert LinearAlgebra.issymmetric(K_obj.K)
    K_obj.Kinv .= Matrix(inv(LinearAlgebra.Symmetric(K_obj.K)))
    # if !LinearAlgebra.isposdef(K_obj.K)
    #     @show K_obj.K
    #     @assert LinearAlgebra.isposdef(K_obj.K)
    # end
    K_obj.Kchol .= LinearAlgebra.cholesky(LinearAlgebra.Symmetric(K_obj.K)).L
    # @assert LinearAlgebra.issymmetric(K_obj.K)
    return

end

function update_inv(A, u, v)
    # Sherman-Morrison update
    A - A * u * v' * A ./ (1 + LinearAlgebra.dot(v, A, u))
end


#for checking, 0-based
function submatrix_blaze(M::AbstractMatrix, r, c, nr, nc)
    return view(
        M,
        r + 1 : r + 1 + nr,
        c + 1 : c + 1 + nc
    )
end

@inline p_to_ne(p::Integer)  = p * (p - 1) ÷ 2
@inline ne_to_p(ne::Integer) = (1 + isqrt(1 + 8ne)) ÷ 2