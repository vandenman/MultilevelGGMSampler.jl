module MultilevelGGMSampler

#=

    Naming convention & abbreviations

        n  -> no. time points/ sample size per participant
        p  -> no. nodes
        k  -> no. participants
        ne -> no. edges

    Iterators

        i* -> 1st iterator of *, e.g., ip iterates 1:p, ik iterates 1:k
        j* -> 2nd iterator of *
        alternatively, i1*, i2*

    Casing

        PascalCase for types
        snake_case for everything else, with a slight preference to omit the _ if the result remains legible
        nonexported functions start with an underscore ('_')
        nonexported types do not start with an underscore

=#

# stdlib
import LinearAlgebra, Random

import Distributions, FillArrays, Graphs, LogExpFunctions, PDMats, SpecialFunctions, StatsBase, IrrationalConstants
import ProgressMeter

import MLBase
import HypergeometricFunctions # free, already a dependency of Distributions

import OnlineStatsBase

import Krylov

# heavy dependencies, can we do without them?
import LoopVectorization
import Optim # could write a custom GradientDescent or Newton's method, might be faster since the value, derivative, and hessian can be computed simultaneously

# for approximations to esf
import FastGaussQuadrature


include("BinarySpaceIterator.jl")
include("LowerTriangleIterator.jl")
include("tril_to_vec.jl")

include("linear_algebra_2_by_2_matrices.jl")

include("elementary_symmetric_functions.jl")

include("AbstractGraphDistribution.jl")

include("GWishartDistributions.jl")
include("ModifiedHalfNormalDistribution.jl")

include("CurieWeissDistribution.jl")
include("CurieWeissDistributionApproximation.jl")

include("AbstractGroupStructure.jl")

include("sample_CurieWeissDistribution.jl")

include("AbstractIndividualStructure.jl")

include("locally_balanced_proposal.jl")

include("simulate_hierarchical_ggm.jl")

include("log_inclusion_prob_prior_G.jl")

include("wwa.jl")
include("sample_GWishart.jl")
include("sample_spike_and_slab.jl")

include("initial_values.jl")
include("online_statistics.jl")
include("sample_MGGM.jl")
include("sample_GGM.jl")

include("extract_posterior_means.jl")

include("pairwise_on_off_stat.jl")

export
    # Distributions -- types
    ModifiedHalfNormal,
    GWishart,
    AbstractGraphDistribution,
    CurieWeissDistribution,

    # Distributions -- methods
    logpdf_prop,
    log_const,

    # elementary symmetric functions
    esf_sum,
    esf_sum!,
    esf_sum_log,
    esf_sum_log!,

    # approximations to elementary symmetric functions for CurieWeissDistribution
    # should these be exported?
    CurieWeissDenominatorApprox,
    compute_log_den!,
    compute_ys!,
    update!,
    downdate!,
    default_no_weights,
    get_value,
    get_shifted_value,

    # utilities
    BinarySpaceIterator,
    LowerTriangle,
    UpperTriangle,
    tril_to_vec,
    triu_to_vec,
    tril_vec_to_sym,
    triu_vec_to_sym,

    # post processing
    extract_posterior_means,
    compute_roc_auc,

    # simulate data
    simulate_hierarchical_ggm,

    # MCMC -- methods
    sample_MGGM,
    sample_GGM,
    prepare_data,
    sample_curie_weiss,

    # MCMC -- types for specifying individual structure
    GWishartStructure,
    SpikeAndSlabStructure,

    # MCMC -- types for specifying group structure
    AbstractGroupStructure,
    CurieWeissStructure,
    BernoulliStructure,
    IndependentStructure,

    # onlinestats -- custom stuff for binary parameters
    OnCounter,
    PairWiseOn,
    PairWiseOff,
    PairWiseOnOff



end
