using Documenter, MultilevelGGMSampler

makedocs(
	sitename="MultilevelGGMSampler.jl",
	modules  = [MultilevelGGMSampler],
	format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
	warnonly = true
)

deploydocs(;
	repo = "github.com/vandenman/MultilevelGGMSampler.git"
)