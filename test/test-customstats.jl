using Test, MultilevelGGMSampler
import OnlineStats

@testset "custom stats" begin

    opts = (
        (
            type = MultilevelGGMSampler.OnCounter,
            name = "OnCounter",
            ref  = x -> vec(sum(x, dims = 1)),
        ),
        (
            type = MultilevelGGMSampler.PairWiseOn,
            name = "PairWiseOn",
            ref  = x -> MultilevelGGMSampler.tril_to_vec(x' * x, -1),
        ),
        (
            type = MultilevelGGMSampler.PairWiseOff,
            name = "PairWiseOff",
            ref  = x -> MultilevelGGMSampler.tril_to_vec((1 .- x)' * (1 .- x), -1),
        ),
        (
            type = MultilevelGGMSampler.PairWiseOnOff,
            name = "PairWiseOnOff",
            ref  = x -> begin
                [
                    MultilevelGGMSampler.tril_to_vec(x' * x, -1) ;;
                    MultilevelGGMSampler.tril_to_vec((1 .- x)' * (1 .- x), -1)
                ]'
            end
        )
    )

    # obj = opts[4]

    max_k = 5
    nn    = 3
    ns = zeros(Int, nn)
    for i in 1:nn
        ns[i] = rand(1:5)
    end
    ntot = sum(ns)

    all_data_int = rand(0:1, sum(ns), max_k)
    for type in (BitMatrix, Matrix{Int})
        all_data = convert(type, all_data_int)
        for obj in opts
            @testset "Stat: $(obj.name), type $type" begin
                for k in 2:max_k

                    o = obj.type(k)

                    i = 1
                    nseen = 0
                    for l in eachindex(ns)
                        for j in 1:ns[l]
                            OnlineStats.fit!(o, view(all_data, i, 1:k))
                            i += 1
                        end

                        nseen += ns[l]
                        @test Int.(OnlineStats.value(o)) == obj.ref(view(all_data, 1:nseen, 1:k))
                        @test OnlineStats.nobs(o) == nseen
                    end
                end
            end
        end
    end
end
