# MultilevelGGMSampler.jl

[![Build Status](https://github.com/vandenman/MultilevelGGMSampler.jl/workflows/runtests/badge.svg)](https://github.com/vandenman/MultilevelGGMSampler.jl/actions)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://vandenman.github.io/MultilevelGGMSampler.jl/latest/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://vandenman.github.io/MultilevelGGMSampler.jl/stable/)

A Julia package for estimating Multilevel Gaussian Graphical Models using Bayesian inference.

## Installation

This package is not (yet) registered. To install it, start julia and press `]` to enter the package manager. Then run
```julia-repl
pkg> add https://github.com/vandenman/MultilevelGGMSampler.jl
```

## Multilevel Models

## Supported Individual Level Models
- Spike and Slab
- G-Wishart

## Supported Group Level Models
- None (assumes independence)
- Curie-Weiss distribution

## Example

```julia
using MultilevelGGMSampler

# time points, nodes, participants
t, p, k = 1000, 10, 20

# used to sample individual level precision matrices
πGW = GWishart(p, 3.0)

# define the group-level model
group_structure = CurieWeissStructure()

# simulate data
data, parameters = simulate_hierarchical_ggm(n, p, k, πGW, groupstructure)

# specify the individual-level model
SpikeAndSlabStructure(;threaded = true)

# run the Gibbs sampler
results = sample_MGGM(data, individual_level, group_level; n_iter = 2000, n_warmup = 1000)

# extract the posterior means
results = extract_posterior_means(res)
```
