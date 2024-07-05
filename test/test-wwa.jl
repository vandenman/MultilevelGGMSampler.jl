#TODO: rewrite with sample_MGGM()
#=
using Test, Distributions, Random, LinearAlgebra

@testset "test that wwa is reproducible" begin


    n, p = 100, 5

    # simulate some data
    g_init = Matrix(Graphs.adjacency_matrix(Graphs.SimpleGraph(p, p ÷ 2)))
    Σ = rand(Wishart(p, Matrix(Diagonal(ones(p)))))
    μ = zeros(p)
    data = permutedims(rand(MvNormal(μ, Σ), n))

    # run twice with same rng
    rng = Random.MersenneTwister(42)
    res1 = wwaMCMC(rng, g_init, data, 10)
    rng = Random.MersenneTwister(42)
    res2 = wwaMCMC(rng, g_init, data, 10)
    @test res1 == res2

end
=#
