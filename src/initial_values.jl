
"""
See https://stats.stackexchange.com/a/311596
"""
function intersection_normal_pdfs(σ₁, σ₂)
    σ₁², σ₂² = σ₁^2, σ₂^2
    A = -1 / σ₁² + 1 / σ₂²
    B = zero(A)
    C = log(σ₂² / σ₁²)
    Δ = B^2 - 4*A*C

    return (B + √Δ) / 2A, (B - √Δ) / 2A

end
intersect, _ = intersection_normal_pdfs(.1, 10)
Distributions.logpdf(Distributions.Normal(0, .1), intersect) - Distributions.logpdf(Distributions.Normal(0, 10), intersect)

function find_initial_values_K(prep_data)
    init_K = similar(prep_data.sum_of_squares)
    for ik in axes(init_K, 3)
        init_K_ik = view(init_K, :, :, ik)
        obs_s = LinearAlgebra.Symmetric(view(prep_data.sum_of_squares, :, :, ik))
        copyto!(init_K_ik, obs_s)
        init_K_ik ./= prep_data.n
        if LinearAlgebra.isposdef(init_K_ik)
            chol_s = LinearAlgebra.cholesky!(init_K_ik)
            LinearAlgebra.inv!(chol_s)
        else # This is probably suboptimal, perhaps LDLT decomposition and make negative ds a small positive value?
            init_K_ik .= LinearAlgebra.pinv(init_K_ik)
        end
    end
    return init_K
end

function find_initial_values_G(init_K, σ_slab, σ_spike)
    lb, ub = intersection_normal_pdfs(σ_spike, σ_slab)
    if ub < lb
        lb, ub = ub, lb
    end

    init_G = Array{Int}(undef, size(init_K))
    for index in CartesianIndices(init_K)
        init_G[index] = !(lb <= init_K[index] <= ub)
    end

    return init_G
end
