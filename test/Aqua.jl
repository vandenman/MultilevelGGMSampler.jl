using Aqua
import ForwardDiff, StatsBase
Aqua.test_all(
    MultilevelGGMSampler,
    ambiguities=(exclude=[ForwardDiff.Dual, StatsBase.PValue, StatsBase.TestStat], broken=true),
)