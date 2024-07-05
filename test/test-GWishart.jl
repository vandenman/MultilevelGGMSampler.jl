#=

    TODO:

=#

# run the commented code on the first try, make sure that
# LD_LIBRARY_PATH is set in settings.json > terminal.integrated.env.linux
# ENV["R_HOME"] = "/home/don/R/custom_R/R-4.2.2"
# import Pkg
# Pkg.build("RCall")

#=

using RCall

R"""
renv::activate("R")
if (!require("BDgraph")) {
  renv::install("BDgraph")
  renv::snapshot()
}
"""

using MultilevelGGMSampler, Test, StatsBase
import Graphs, Distributions, PDMats, LinearAlgebra

function linear_to_cartesian(k)
    i = (sqrt(1 + 8k) - 1) / 2
    i0 = floor(Int, i)
    i == i0 && return (i0, i0)
    j = k - i0 * (i0 + 1) ÷ 2
    i0 += 1
    return i0, j
end

function run_BDgraph(g_mat, df, rate, n = 10_000)
    result = rcopy(R"BDgraph::rgwish(n = $n, adj = $g_mat, b = $df, D = $rate)")::Array{Float64, 3}
    return result
end

@testset "chain! works as intended" begin

    verify_sparity(A, adj) = all(xor(iszero(adj[i, j]), !isapprox(A[i, j], 0.0; atol = 1e-6)) for i in 1:size(adj, 1)-1 for j in i+1:size(adj, 1))

    on_ci = haskey(ENV, "CI") ? ENV["CI"] == "true" : false
    ps_values = on_ci ? (3:10, 100, 1000) : (3:8)
    for ps in ps_values
        for p in ps

        id = zeros(p, p)
        id[diagind(id)] .= 1.0

        W = rand(Distributions.Wishart(p, id))
        Σ = inv(W)

        G = Graphs.SimpleGraph(p, p)

        neighbors = Graphs.neighbors.(Ref(G), Graphs.vertices(G))

        MultilevelGGMSampler.chain!(W, Σ, neighbors)

        iW = inv(W)
        adj = Matrix(Graphs.adjacency_matrix(G))

        # TODO: why does it fail? maybe there are near zero elements that shouldn't be
        @show p
        @test verify_sparity(iW, adj)

        end
    end


end

@testset "comparison with BDgraph" begin

    n_samples = 20_000
    on_ci = haskey(ENV, "CI") ? ENV["CI"] == "true" : false

    p_values = on_ci ? (4:8) : (4:2:8)
    for p in 4:8

        @testset "p = $p" begin

            max_edges = p * (p - 1) ÷ 2
            n_e_values = on_ci ? (0:max_edges) : range(0, max_edges, step = max(1, max_edges ÷ 4))

            for n_e in n_e_values

                @show p, n_e, max_edges

                g = Graphs.SimpleGraph(p, n_e)
                g_mat = Matrix(Graphs.adjacency_matrix(g))
                df = p

                rate = rand(Distributions.Wishart(p, PDMats.ScalMat(p, 1.0)))
                samples_BDgraph = run_BDgraph(g_mat, df, Matrix(rate), n_samples)

                d = GWishart(df, rate, g)
                samples_self = Array{Float64}(undef, p, p, n_samples)
                for i in axes(samples_self, 3)
                    samples_self[:, :, i] .= rand(d)
                end

                @test isapprox(
                    mean(samples_self,    dims = 3),
                    mean(samples_BDgraph, dims = 3),
                    atol = 1e-1, rtol = 1e-1
                )

                if n_e == max_edges

                    theoreticalDist = Wishart(df + p - 1, inv(LinearAlgebra.cholesky(rate)))
                    @test isapprox(
                        mean(samples_self, dims = 3),
                        mean(theoreticalDist),
                        atol = 1e-1, rtol = 1e-1
                    )

                end

                @test isapprox(
                    std(samples_self, dims = 3),
                    std(samples_BDgraph, dims = 3),
                    atol = 1e-1, rtol = 1e-1
                )

                probs = .1:.1:.99
                for e in Graphs.edges(g)

                    i, j = e.dst, e.src#linear_to_cartesian(k)
                    qs_BDgraph = quantile(view(samples_BDgraph, i, j, :), probs)
                    qs_self    = quantile(view(samples_BDgraph, i, j, :), probs)

                    @test isapprox(qs_BDgraph, qs_self, atol = 1e-1, rtol = 1e-1)

                end
            end
        end
    end
end

=#