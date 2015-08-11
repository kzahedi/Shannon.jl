module Shannon

using StatsBase

include("distribution.jl")
include("entropy.jl")
include("binning.jl")
include("mi.jl")
include("te.jl")
include("mc.jl")

export KL, PI, MI, TE, MC

export entropy

export bin_vector, bin_matrix, bin_value
export unbin_value, unbin_matrix, unbin_vector
export combine_binned_matrix, combine_binned_vector
export combine_and_relabel_binned_matrix
export unary_of_matrix
export relabel

function KL(p::Vector{Float64}, q::Vector{Float64}; base=2)
  @assert (length(p) == length(q)) "Size mismatch"
  sum([ (p[i] != 0 && q[i] != 0)? p[i] * log(base, p[i]/q[i]) : 0 for i=1:length(p)])
end


end # module
