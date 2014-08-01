module Shannon

# The entropy functions (ChaoShen, Dirichlet, MillerMadow) were copied from the
# R package "entropy", by Jean Hausser and Korbinian Strimmer. For details, see
# http://cran.r-project.org/web/packages/entropy/index.html

using StatsBase

export KL, PI, MI, entropy
export bin_vector, bin_matrix
export combine_binned_matrix, combine_binned_vector
export combine_and_relabel_binned_matrix
export unary_of_matrix
export relabel

"unary"

function KL(p::Vector{Float64}, q::Vector{Float64}; base=2)
  @assert (length(p) == length(q)) "Size mismatch"
  sum([ (p[i] != 0 && q[i] != 0)? p[i] * log(base, p[i]/q[i]) : 0 for i=1:length(p)])
end

# predictive information
PI(data::Vector{Int64}; base=2, mode="emperical") = MI(hcat(data[1:end-1], data[2:end]), base, mode)

# mutual information
function MI(data::Matrix{Int64}; base=2, mode="emperical", pseudocount=0)

  max = maximum(data)
  px  = fe1p(data[:,1])
  py  = fe1p(data[:,2])
  pxy = fe2p(hcat(data[:,[1:2]]))

  r = 0
  for x=1:length(px)
    for y=1:length(py)
      if px[x] != 0.0 && py[y] != 0.0 && pxy[x,y] > 0.0
        r = r + pxy[x,y] * (log(base, pxy[x,y]) - log(base, px[x]*py[y]))
      end
    end
  end
  r
end

entropy_emperical(p::Vector{Float64}, base::Number) = sum([ p[x] > 0 ? (-p[x] * log(base, p[x])) : 0 for x=1:size(p)[1]])

function entropy_chaoshen(data::Vector{Int64}, base::Number)
  c = counts(data, 1:maximum(data))
  n = sum(c)
  p = c ./ n
  s = sum(p)
  p = p ./ s # to be sure

  # code copied form R entropy package
  f1 = sum([i == 1? 1 : 0 for i in c]) # number of singletons
  if (f1 == n)
    f1 = n-1                # avoid C=0
  end

  C  = 1 - f1/n              # estimated coverage
  pa = C .* p                # coverage adjusted empirical frequencies
  la = (1-(1-pa).^n)         # probability to see a bin (species) in the sample
  -sum(pa.*log(base, pa)./la) # Chao-Shen (2003) entropy estimatojr
end

function entropy_millermadow(data::Vector{Int64}, base::Number)
  c = counts(data, 1:maximum(data))
  n = sum(c)                         # total number of counts
  m = sum([i > 0? 1 : 0 for i in c]) # number of bins with non-zero counts
  p = c ./ n
  s = sum(p)
  p = p ./ s

  # bias-corrected empirical estimate
  entropy_emperical(p, base) + (m-1)/(2*n)
end

function entropy(data::Vector{Int64}; base=2, mode="emperical", pseudocount=0)
  known_mode = (mode == "emperical" ||
                mode == "ChaoShen"  ||
                mode == "Dirichlet" ||
                mode == "MillerMadow")
  @assert known_mode "Mode may be any of the following: [\"emperical\", \"ChaoShen\", \"Dirichlet\", \"MillerMadow\"]"

  p = []

  r = nothing

  if     mode == "emperical"
    p = fe1p(data)
    r = entropy_emperical(p, base) 
  elseif mode == "ChaoShen"
    r = entropy_chaoshen(data, base)
  elseif mode == "Dirichlet"
    p = fe1pd(data, pseudocount)
    r = entropy_emperical(p, base)
  elseif mode == "MillerMadow"
    r = entropy_millermadow(data, base)
  #= else =# # Not needed. Caught by assertion
  end
  r
end

function bin_value(v::Float64, min::Float64, max::Float64, bins::Int64)
  f = maximum([minimum([1.0, (v-min) / (max - min)]), 0.0])
  g = maximum([f * bins, 1.0])
  int64(g)
end

function bin_vector(vec::Vector{Float64}, min::Float64, max::Float64, bins::Int64)
  bf(v::Float64) = bin_value(v, min, max, bins)
  map(bf, vec)
end

function bin_matrix(m::Matrix{Float64}, min::Float64, max::Float64, bins::Int64)
  r = zeros(size(m))
  for i=1:size(m)[1]
    r[i,:] = bin_vector(squeeze(m[i,:],1), min, max, bins)
  end
  convert(Matrix{Int64}, r)
end

function combine_binned_vector(v::Vector{Int64}, bins::Int64)
  r = 0
  for i = 1:length(v)
    r += v[i] * bins^(i-1)
  end
  convert(Int64, r)
end

function combine_binned_matrix(v::Matrix{Int64})
  l = size(v)[1]
  r = zeros(l)
  m = maximum(v)
  for i = 1:l
    r[i] = combine_binned_vector(squeeze(v[i,:],1), m)
  end
  convert(Vector{Int64}, r)
end

relabel(v::Vector{Int64}) = indexin(v, unique(v))

combine_and_relabel_binned_matrix(data::Matrix{Int64}) = relabel(combine_binned_matrix(data))

unary_of_matrix(data::Matrix{Float64}, min::Float64, max::Float64, bins::Int64) = combine_and_relabel_binned_matrix(bin_matrix(data, min, max, bins))

fe1ph(v::Vector{Int64})   = hist(v)[2] ./ size(v)[1]

function fe2p(v::Matrix{Int64}) # frequency estimation of one dimensional probability
  m1 = maximum(v[:,1])
  m2 = maximum(v[:,2])
  r  = counts(v[:,1], v[:,2], (1:m1, 1:m2))
  l  = size(v)[1]
  r = r ./ l
  # just to get rid of the numerical inaccuracies and make sure its a probability distribution
  s = sum(r)
  r ./ s
end

# frequency estimation of one dimensional probability
function fe1p(v::Vector{Int64})
  m = maximum(v)
  l  = size(v)[1]
  r  = counts(v, 1:m)
  r = r ./ l
  # just to get rid of the numerical inaccuracies and make sure its a probability distribution
  s = sum(r)
  r ./ s
end

function fe1pd(v::Vector{Int64}, d::Number) # Dirichlet
  r = counts(v, 1:maximum(v))
  r = r .+ d
  l = sum(r)
  r = r ./ l
  s = sum(r) # just to be sure
  r ./ s
end

end # module
