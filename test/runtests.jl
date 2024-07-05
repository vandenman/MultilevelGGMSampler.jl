using MultilevelGGMSampler
using Test
import Random

# following R conventions (I know) everything is sourced that starts with "test-" and ends with ".jl"
const test_dir = basename(pwd()) == "MultilevelGGMSampler.jl" ? joinpath(pwd(), "test") : pwd()
const tests = joinpath.(test_dir, filter!(x->startswith(x, "test-") && endswith(x, ".jl"), readdir(test_dir)))

const on_ci = haskey(ENV, "CI") ? ENV["CI"] == "true" : false

@testset "MultilevelGGMSampler" begin

	for t in tests
		@testset "Test $t" begin
			Random.seed!(42)
			include("$t")
		end
	end
end