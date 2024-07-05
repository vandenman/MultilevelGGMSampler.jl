#=
    it would be even better to do this step with WWA!
    that means:
        1. compute Q | K, G for all people in parallel (double parallel perhaps!)
        2. sequentially compute the proposed edges from Q one by one,
            2a. once edge (i, j) is selected for person k, need to renormalize Q for the next people.
        3. form K_new, G_new for all people, perhaps in parallel
        4. compute Q_new | K_new, G_new for all people in parallel,
            5a. do not compute permutations that need to be set to 0 anyway.
        5. compute the acceptance probabilities and possible updates in parallel
        it's probably best to do 2 in a randomized order

    also, perhaps not return the edge id but just the pair i, j?
    we're not really memory bound
=#


function sample_individual_structure!(
    rng::Random.AbstractRNG,
    individual_state,
    individual_structure_internal::GWishartStructureInternal,
    group_structure_internal::AbstractGroupStructureInternal,
    group_state
)

    Ks     = individual_state.Ks
    Gs     = individual_state.Gs
    G_objs = individual_state.G_objs
    (; prior_df, post_df, post_rate_vec, delayed_accept, loc_bal, Letac, orig, threaded) = individual_structure_internal

    p = size(Ks, 1)
    k = size(Ks, 3)

    n_edge_updates = p
    if orig

        bufferData = (
            Q     = threaded ? zeros(0, 0) : zeros(p, p),
            log_Q = threaded ? zeros(0, 0) : zeros(p, p)
        )
        # Possible performance: the code below should be used when threadings is disabled!
        # Could also be created once as long as σ would be a reference?
        πG = createGraphDistribution(individual_structure_internal, group_state, k)
        for ik in eachindex(post_rate_vec)

            πGk = conditional_graph_distribution(πG, Gs, ik)

            K = view(Ks, :, :, ik)
            G = view(Gs, :, :, ik)

            update_G_wwa!!(
                rng,
                K, G, post_rate_vec[ik], post_df, πGk,
                prior_df, n_edge_updates,
                false, # approx hardcoded to false like in wwa.py
                delayed_accept, loc_bal, Letac,
                # delayed_accept, loc_bal, Letac,
                bufferData
                # threaded
            )
        end
    else

        sample_K!(Ks, G_objs, post_rate_vec, post_df, rng, threaded)

        if loc_bal
            # TODO: do not use the channel approach but just threading
            if threaded
                sample_G_WWA_channel!!!(    Gs, G_objs, Ks, n_edge_updates, group_state, group_structure_internal, post_rate_vec, post_df, prior_df, rng, threaded)
                # sample_G_WWA_channel!!!(    Gs, G_objs, K_objs, n_edge_updates, groupState, post_rate_vec, post_df, prior_df, rng, threaded)
                # sample_G_WWA_threaded2!!!(    Gs, G_objs, K_objs, n_edge_updates, groupState, post_rate_vec, post_df, prior_df, rng, threaded)
            else
                sample_G_WWA!!!(    Gs, G_objs, Ks, n_edge_updates, group_state, group_structure_internal, post_rate_vec, post_df, prior_df, rng, threaded)
                # sample_G_WWA!!!(    Gs, G_objs, K_objs, n_edge_updates, groupState, post_rate_vec, post_df, prior_df, rng, threaded)
            end
        else
            sample_G_uniform!!!(Gs, G_objs, Ks, n_edge_updates, group_state, post_rate_vec, post_df, prior_df, rng, threaded)
        end
    end

end

function common_stuff(i, j, exponent, K, adj, rate, df_post, πG, rng, suffstats_groupstate)

    p = size(K, 1)
    perm, perm_inv = permute_e_last(i, j, p)
    # @show i, j, p

    @assert LinearAlgebra.issymmetric(K)
    K_perm = permute_mat(K, perm_inv)
    Phi_perm = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(K_perm, :L)).L

    rate_pp          = rate[perm_inv[p    ], perm_inv[p]]
    rate_1p          = rate[perm_inv[p - 1], perm_inv[p]]
    log_N_tilde_post = log_N_tilde(Phi_perm, rate_pp, rate_1p)
    log_prior_ratio  = exponent * log_inclusion_prob_prior_G(πG, adj, i, j, isnothing(suffstats_groupstate) ? 0 : suffstats_groupstate[last(parentindices(adj))])

    Phi_perm[p, p] = sqrt(rand(rng, Distributions.Chisq(df_post)) / rate_pp) # Possible performance: chi distribution!

    return K_perm, Phi_perm, perm, perm_inv, rate_1p, rate_pp, log_prior_ratio, log_N_tilde_post
end

function propose_edge_uniform!!!(
    adj,
    G_obj::Graphs.SimpleGraph,
    K::AbstractMatrix{<:Real},
    edge::NTuple{2, <:Integer},
    rate::AbstractMatrix{<:Real},
    df_post::Real,
    df_prior::Real,
    πG::Distributions.DiscreteDistribution,
    rng::Random.AbstractRNG,
    add_or_remove::Bool,
    approx::Bool = false, delayed_accept::Bool = true,
    Letac::Bool = true
)

    accept = true

    p   = Graphs.nv(G_obj)
    n_e = Graphs.ne(G_obj)

    k = last(parentindices(adj))
    i, j = edge
    add = iszero(adj[i, j])
    @assert add == add_or_remove

    exponent = 2 * add - 1
    n_e_tilde = n_e + exponent
    log_q_y_x = log(proposal_G_es(p, n_e_tilde, n_e))
    log_q_x_y = log(proposal_G_es(p, n_e, n_e_tilde))

    K_perm, Phi, perm, perm_inv, rate_1p, rate_pp, log_prior_ratio, log_N_tilde_post  = common_stuff(i, j, exponent, K, adj, rate, df_post, πG, rng, suffstats_groupstate[k])

    accept = compute_acceptance(
        i, j, adj, add, exponent, df_prior, log_q_y_x, log_q_x_y, perm_inv,
        log_prior_ratio, log_N_tilde_post, delayed_accept, approx, Letac, rng
    )

    @assert LinearAlgebra.isposdef(K)

    if accept # Update the graph.
        adj[i, j] = add
        adj[j, i] = add

        if add
            suffstats_groupstate[k] += 1
            Graphs.add_edge!(G_obj, i, j)
        else
            suffstats_groupstate[k] -= 1
            Graphs.rem_edge!(G_obj, i, j)
        end
    end

    if isone(adj[i, j])  # The graph contains (`i`, `j`) after updating.
        Phi[p, p - 1] = randn(rng) / sqrt(rate_pp) - Phi[p - 1, p - 1] * rate_1p / rate_pp
    else # The graph does not contain (`i`, `j`) after updating.
        Phi[p, p - 1] = -sum(
                view(Phi, p-1, 1:p - 2) .* view(Phi, p, 1:p - 2)
                # submatrix(Phi_0_tilde, p - 2, 0, 1, p - 2) % submatrix(Phi_0_tilde, p - 1, 0, 1, p - 2)
            ) / Phi[p - 1, p - 1]
    end
    # end

    update_K_from_Phi(p, j, perm, K, Phi) # WvdB: Not required if `loc_bal and accept`?

end

function sample_K!(Ks::AbstractArray{<:Real, 3}, G_objs, rate_post_vec, df_post, rng, threaded)

    if threaded
        Threads.@threads for ik in axes(Ks, 3)

            K = view(Ks, :, :, ik)
            rate = rate_post_vec[ik]
            G_obj = G_objs[ik]

            # Distributions.rand!(rng, GWishart(df_post, rate, G_obj), K)
            K .= Distributions.rand(rng, GWishart(df_post, rate, G_obj))

        end
    else
        for ik in axes(Ks, 3)

            K = view(Ks, :, :, ik)
            rate = rate_post_vec[ik]
            G_obj = G_objs[ik]

            # Distributions.rand!(rng, GWishart(df_post, rate, G_obj), K)
            K .= Distributions.rand(rng, GWishart(df_post, rate, G_obj))

            # @assert LinearAlgebra.isposdef(K)
            # Distributions.rand!(rng, GWishart(df_post, rate, G_obj), K)

        end
    end
    return Ks
end

function compute_available_edges(e_ids_in_use, max_e, G, add)#, t)
    available = BitSet(1:max_e)
    for j in eachindex(e_ids_in_use)
        delete!(available, e_ids_in_use[j])
    end
    id = 1
    p = size(G, 1)
    # can probably do some better boolean arithmetic here, xor(add, iszero(Gs[i, j, k]))? with a ⊻ b?
    for j in 1:p-1, i in j+1:p
        #   add     isone(Gs[i, j, k])  desired outcome
        #   T       T                   T
        #   T       F                   F
        #   F       T                   F
        #   F       F                   T
        #   satisfied by add === isone(Gs[i, j, k])
        # if add && isone(G[i, j])
        #     delete!(available, id)
        # elseif !add && iszero(G[i, j])
        #     @show i, j, id
        #     delete!(available, id)
        # end

        add === isone(G[i, j]) && delete!(available, id)
        id += 1
    end
    return available
end

function schedule_egdes_uniform(Gs::AbstractArray{<:Integer, 3}, rng = Random.default_rng())

    p, k = size(Gs, 1), size(Gs, 3)
    max_e = p_to_ne(p)
    e_ids = zeros(Int, k)                    # linear index to the edges
    edges = Vector{NTuple{2, Int}}(undef, k) # pair of row and column indices
    add_or_remove = BitVector(undef, k)
    for ik in 1:k

        e_ids_in_use = view(e_ids, 1:ik-1)
        G = view(Gs, :, :, ik)

        n_e = sum(G) ÷ 2
        # not sure if this is the wisest proposal scheme... isn't it better to just sample an edge at random?
        add =
            if iszero(n_e)      true
            elseif n_e == max_e false
            else                rand(rng) < 0.5
            end

        available = compute_available_edges(e_ids_in_use, max_e, G, add)

        # can happen, but only once
        if isempty(available)
            add = !add
            available = compute_available_edges(e_ids_in_use, max_e, G, add)
        end

        @assert !isempty(available)
        @assert isdisjoint(available, e_ids_in_use)

        proposed_edge = rand(rng, available)
        # e_ids[ik] = proposed_edge
        j, i = linear_index_to_lower_triangle_indices(proposed_edge, p)
        @assert iszero(G[i, j]) == add
        edges[ik] = (i, j)
        add_or_remove[ik] = add

    end
    return edges, add_or_remove
end

function sample_G_uniform!!!(Gs, G_objs, Ks, n_edge_updates, groupState, rate_post_vec, df_post, df_prior, rng, threaded)

    # TODO: this function is entirely incorrect!
    # TODO: the threaded version should also use a channel... this is incorrect.
    πG = createGraphDistribution(groupState, size(Gs, 3))

    for _ in 1:n_edge_updates

        edges, add_or_remove = schedule_egdes_uniform(Gs, rng)

        if threaded

            Threads.@threads for ik in eachindex(edges)

                K     = view(Ks, :, :, ik)
                G     = view(Gs, :, :, ik) # NOTE: cannot be if this can be updated in parallel! Actually, it can now that it is an integer matrix
                G_obj = G_objs[ik]
                rate  = rate_post_vec[ik]
                edge  = edges[ik]
                πGk = conditional_graph_distribution(πG, Gs, ik)

                propose_edge_uniform!!!(G, G_obj, K, edge, rate, df_post, df_prior, πGk, rng, add_or_remove[ik])

            end
        else

            for ik in eachindex(edges)

                K     = view(Ks, :, :, ik)
                G     = view(Gs, :, :, ik)
                G_obj = G_objs[ik]
                rate  = rate_post_vec[ik]
                edge  = edges[ik]
                πGk = conditional_graph_distribution(πG, Gs, ik)

                propose_edge_uniform!!!(G, G_obj, K, edge, rate, df_post, df_prior, πGk, rng, add_or_remove[ik])

                @assert Matrix(Graphs.adjacency_matrix(G_obj)) == G
            end
        end
    end
end

function sample_G_WWA!!!(
    Gs::AbstractArray{<:Integer, 3},
    G_objs,
    Ks::AbstractArray{T, 3},
    n_edge_updates::Integer,
    groupGraphState,
    group_structure_internal::AbstractGroupStructureInternal,
    rate_post_vec,
    df_post::Real,
    df_prior::Real,
    rng::Random.AbstractRNG = Random.default_rng(),
    threaded::Bool = false,
    approx::Bool = false,
    delayed_accept::Bool = true,
    Letac::Bool = true
) where {T<:Real}

    p = size(Gs, 1)
    k = size(Gs, 3)
    edges = Vector{NTuple{2, Int}}(undef, k)

    πG = createGraphDistribution(group_structure_internal, groupGraphState, k)
    suffstats_groupstate = compute_suffstats_groupstate(πG, Gs, group_structure_internal)

    # buffers for locally_balanced_proposal_givens
    Qs     = Array{T}(undef, p, p, k)
    log_Qs = Array{T}(undef, p, p, k)

    # executor = threaded ? FLoops.ThreadedEx() : FLoops.SequentialEx()

    # 1. Compute Q & log_Q for 1:k
    for up in 1:n_edge_updates
        for ik in 1:k

            K     = view(Ks, :, :, ik)
            adj   = view(Gs, :, :, ik)
            rate  = rate_post_vec[ik]
            πGik  = conditional_graph_distribution(πG, Gs, ik)
            n_e   = Graphs.ne(G_objs[ik])
            @assert sum(adj) ÷ 2 == n_e

            Q     = view(Qs,     :, :, ik)
            log_Q = view(log_Qs, :, :, ik)

            locally_balanced_proposal_adjugate!!(
                log_Q, Q,
                K, adj, n_e, πGik, df_prior, rate,
                isnothing(suffstats_groupstate) ? 0 : suffstats_groupstate[k],
                Letac
            )

        end

        # 2. sequentially sample proposals from Q, without considering previously selected edges
        # TODO: this fails when there are more people than edges!
        for ik in 1:k

            Q = view(Qs, :, :, ik)

            threaded && renormalize_Q!(Q, view(edges, 1:ik-1))

            i, j = sample_edge_from_Q(Q, rng)

            edges[ik] = (i, j)

        end

        # 3. update edges can now be done in parallel
        for ik in 1:k

            K     = view(Ks, :, :, ik)
            adj   = view(Gs, :, :, ik)
            πGik  = conditional_graph_distribution(πG, Gs, ik)
            rate  = rate_post_vec[ik]
            G_obj = G_objs[ik]

            i, j = edges[ik]

            add = iszero(adj[i, j])
            exponent = 2add - 1

            @assert sum(adj) ÷ 2 == Graphs.ne(G_obj)
            n_e       = Graphs.ne(G_obj)
            n_e_tilde = n_e + exponent
            @assert n_e_tilde >= zero(n_e_tilde)

            # 3. update G, K, and Phi according to the proposed edges
            K_perm, Phi, perm, perm_inv, rate_1p, rate_pp, log_prior_ratio, log_N_tilde_post = common_stuff(i, j, exponent, K, adj, rate, df_post, πGik, rng, suffstats_groupstate)
            Phi_12_cur = Phi[p, p - 1]

            # 4. compute the inverse proposal probability log_q_x_y
            if add
                Phi[p, p - 1] = randn(rng) / sqrt(rate_pp) - Phi[p - 1, p - 1] * rate_1p / rate_pp
            else
                Phi[p, p - 1] = -sum(view(Phi, p-1, 1:p-2) .* view(Phi, p, 1:p-2)) / Phi[p - 1, p - 1]
                # Phi[p - 1, p - 2] = -sum(submatrix(Phi, p - 2, 0, 1, p - 2) % submatrix(Phi, p - 1, 0, 1, p - 2)) / Phi(p - 2, p - 2)
            end

            adj[i, j] = add
            adj[j, i] = add

            update_K_from_Phi(p, j, perm, K, Phi)

            Q     = view(Qs,     :, :, ik)
            log_Q = view(log_Qs, :, :, ik)

            log_q_adjust = threaded ? normalization_factor_log_Q(log_Q, view(edges, 1:ik-1)) : zero(T)
            log_q_y_x = log_Q[i, j] - log_q_adjust

            locally_balanced_proposal_adjugate!!(
                log_Q, Q,
                K, adj, n_e, πGik, df_prior, rate,
                isnothing(suffstats_groupstate) ? 0 : suffstats_groupstate[k],
                Letac
            )

            log_q_adjust = normalization_factor_log_Q(log_Q, view(edges, 1:ik-1))
            log_q_x_y = log_Q[i, j] - log_q_adjust

            # 5. compute the acceptance probability
            accept = compute_acceptance(
                i, j, adj, add, exponent, df_prior, log_q_y_x, log_q_x_y, perm_inv,
                log_prior_ratio, log_N_tilde_post, delayed_accept, approx, Letac, rng
            )
            # accept = rand(rng, Bool)

            if accept # Update the graph.
                adj[i, j] = add
                adj[j, i] = add
                if add
                    Graphs.add_edge!(G_obj, i, j)
                else
                    Graphs.rem_edge!(G_obj, i, j)
                end
            else
                # Revert any update in `adj` and `Phi`.
                adj[i, j] = !add
                adj[j, i] = !add

                Phi[p, p - 1] = Phi_12_cur
                update_K_from_Phi(p, j, perm, K, Phi)
            end


        end
    end
    # Possible performance:
    # 3. do the part in common with the random edge stuff for 1:k, possibly in parallel
    # 4. compute the reverse transition probability log_q_x_y for 1:k, possibly in parallel
    # 5. compute the acceptance probabilities, possibly in parallel
    # note that 3-5 can be done in a single loop, and they must be done sequentially
    # however, 4 can be parallized over p^2 and k
    # it is possible to nest @threads though so maybe that would be an option?
    # that would mean we don't have to store all permutations.
    # initially, just do 1 loop without parallelizing the locally_balanced_proposal step
    # also, without threading the original approach is probably better than what we do now.


end

function compute_acceptance_async(
    acceptance_queue,
    acceptance_done_queue,
    Gs::AbstractArray{<:Integer, 3},
    G_objs,
    Ks::AbstractArray{<:Real, 3},
    Qs,
    log_Qs,
    πG,
    rate_post_vec,
    df_post::Real,
    df_prior::Real,
    rng::Random.AbstractRNG = Random.default_rng(),
    approx::Bool = false,
    delayed_accept::Bool = true,
    Letac::Bool = true
)

    for job_id in acceptance_queue

        ik, i, j, edges_for_this_k = job_id

        K     = view(Ks, :, :, ik)
        adj   = view(Gs, :, :, ik)
        πGik  = conditional_graph_distribution(πG, Gs, ik)
        rate  = rate_post_vec[ik]
        G_obj = G_objs[ik]
        p = size(K, 1)

        add = iszero(adj[i, j])
        exponent = 2add - 1

        @assert sum(adj) ÷ 2 == Graphs.ne(G_obj)
        n_e       = Graphs.ne(G_obj)
        n_e_tilde = n_e + exponent
        @assert n_e_tilde >= zero(n_e_tilde)

        # 3. update G, K, and Phi according to the proposed edges
        K_perm, Phi, perm, perm_inv, rate_1p, rate_pp, log_prior_ratio, log_N_tilde_post = common_stuff(i, j, exponent, K, adj, rate, df_post, πGik, rng)
        Phi_12_cur = Phi[p, p - 1]

        # 4. compute the inverse proposal probability log_q_x_y
        if add
            Phi[p, p - 1] = randn(rng) / sqrt(rate_pp) - Phi[p - 1, p - 1] * rate_1p / rate_pp
        else
            Phi[p, p - 1] = -sum(view(Phi, p-1, 1:p-2) .* view(Phi, p, 1:p-2)) / Phi[p - 1, p - 1]
            # Phi[p - 1, p - 2] = -sum(submatrix(Phi, p - 2, 0, 1, p - 2) % submatrix(Phi, p - 1, 0, 1, p - 2)) / Phi(p - 2, p - 2)
        end

        adj[i, j] = add
        adj[j, i] = add

        update_K_from_Phi(p, j, perm, K, Phi)

        Q     = view(Qs,     :, :, ik)
        log_Q = view(log_Qs, :, :, ik)

        log_q_adjust = normalization_factor_log_Q(log_Q, edges_for_this_k)
        log_q_y_x = log_Q[i, j] - log_q_adjust

        locally_balanced_proposal_adjugate!!(
            log_Q, Q,
            K, adj, n_e, πGik, df_prior, rate, Letac
        )

        log_q_adjust = normalization_factor_log_Q(log_Q, edges_for_this_k)
        log_q_x_y = log_Q[i, j] - log_q_adjust

        # 5. compute the acceptance probability
        accept = compute_acceptance(
            i, j, adj, add, exponent, df_prior, log_q_y_x, log_q_x_y, perm_inv,
            log_prior_ratio, log_N_tilde_post, delayed_accept, approx, Letac, rng
        )
        # accept = rand(rng, Bool)

        if accept # Update the graph.
            adj[i, j] = add
            adj[j, i] = add
            if add
                Graphs.add_edge!(G_obj, i, j)
            else
                Graphs.rem_edge!(G_obj, i, j)
            end
        else
            # Revert any update in `adj` and `Phi`.
            adj[i, j] = !add
            adj[j, i] = !add

            Phi[p, p - 1] = Phi_12_cur
            update_K_from_Phi(p, j, perm, K, Phi)
        end

        put!(acceptance_done_queue, (ik, i, j))
    end
end

function renormalize_Q!(Q::AbstractMatrix{T}, edges) where {T <: Real}
    isempty(edges) && return Q
    value = zero(T)
    for (i, j) in edges
        value += Q[i, j]
        Q[i, j] = zero(T)
    end
    return Q ./= (one(T) - value)
end

function normalization_factor_log_Q(log_Q::AbstractMatrix{T}, edges) where {T <: Real}
    isempty(edges) && return zero(T)
    return LogExpFunctions.logsumexp(log_Q[i, j] for (i, j) in edges)
end

function compute_acceptance(
    i::Integer, j::Integer,
    adj::AbstractMatrix{<:Integer},
    add::Bool,
    exponent::Int,
    df_prior::Real,
    log_q_y_x::Real,
    log_q_x_y::Real,
    perm_inv::AbstractVector{<:Integer},
    log_prior_ratio::Real,
    log_N_tilde_post::Real,
    delayed_accept::Bool,
    approx::Bool,
    Letac::Bool,
    rng::Random.AbstractRNG
)

    if delayed_accept || approx
        log_target_ratio_approx = log_prior_ratio + exponent*(
            log_N_tilde_post# + Letac * log_norm_ratio_Letac(adj, i, j, df_0)
        )
        if Letac
            log_target_ratio_approx += exponent * log_norm_ratio_Letac(adj, i, j, df_prior)
        end

        log_g_x_y = min(zero(log_q_x_y), log_q_x_y - log_q_y_x + log_target_ratio_approx)
        # log_g_x_y = min(0.0, log_q_x_y - log_q_y_x + log_target_ratio_approx)

        # Step 2
        accept = log(rand(rng)) < log_g_x_y

    end

    if accept && !approx
        if delayed_accept
            log_q_star_y_x = log_g_x_y + log_q_y_x
            log_q_star_x_y = min(log_q_x_y, log_q_y_x - log_target_ratio_approx)
        else
            log_q_star_y_x = log_q_y_x
            log_q_star_x_y = log_q_x_y
        end

        # Step 3
        # Exchange algorithm to avoid evaluation of normalization constants
        p = size(adj, 1)
        adj_perm = permute_mat(adj, perm_inv)
        adj_perm[p - 1, p] = add
        adj_perm[p, p - 1] = add

        G_perm = Graphs.SimpleGraph(adj_perm)

        K_0_tilde = rand(rng, GWishart(df_prior, PDMats.ScalMat(p, 1.0), G_perm))

        Phi_0_tilde = LinearAlgebra.cholesky(K_0_tilde).L

        log_N_tilde_prior = log(Phi_0_tilde[p - 1, p - 1]) + 0.5 * ((
            K_0_tilde[p-1, p] - Phi_0_tilde[p, p - 1] * Phi_0_tilde[p-1, p-1]
            ) / Phi_0_tilde[p - 1, p - 1])^2

        log_target_ratio = log_prior_ratio + exponent*(
            log_N_tilde_post - log_N_tilde_prior
        )

        accept = log(rand(rng)) < log_q_star_x_y - log_q_star_y_x + log_target_ratio
    end

    return accept

end

function sample_edge_from_Q(Q::AbstractMatrix{T}, rng = Random.default_rng()) where {T <: Real}
    # based on WvdB, could be more julian. Does not enumerate in column major order
    # note that i < j, i being the row and j the column
    i, j = 1, 1
    p = size(Q, 1)
    u = rand(rng)
    tmp_sum = zero(T)

    # NOTE: probably it's fine to flip the indices
    while tmp_sum < u
        if j == p
            i += 1
            j = i
        end
        j += 1
        if i > p || j > p
            @show Q, u
            @assert i <= p && j <= p
        end
        tmp_sum += Q[i, j]
    end
    return i, j
end

function linear_index_to_lower_triangle_indices(k::Integer, n::Integer)
    i = n - 1 - floor(Int,sqrt(-8*k + 4*n*(n-1) + 1)/2 - 0.5)
    j = k + i + ( (n-i+1)*(n-i) - n*(n-1) )÷2
    return j, i
end
function triangle_indices_to_linear_index(i::Integer, j::Integer, p::Integer)
    # Possible fix: should this not be
    # j * (j - 1) ÷ 2 + i  + (j - 1) * (p - j)
    return j*(j-1) ÷ 2 + i - 1 + (j - 1) * (p - j - 1)
end

# function get_indices_of_nth_edge(adj, edge_idx::Integer, add::Bool)
#     p = size(adj, 1)
#     count = zero(edge_idx)
#     for j in 1:p-1, i in j+1:p
#         #   add     isone(Gs[i, j, k])  desired outcome
#         #   T       T                   0
#         #   T       F                   1
#         #   F       T                   1
#         #   F       F                   0
#         #   satisfied by add !== isone(Gs[i, j, k])
#         count += add !== isone(adj[i, j])
#         count == edge_idx && return (i, j)
#     end
#     return (0, 0)
# end