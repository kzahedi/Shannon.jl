module Shannon

export KL, MI
export bin_value, bin_vector, bin_matrix
export combine_binned_vector, combine_binned_matrix
export relabel
export fe1p, fe1ph, fe2p

function KL(p::Vector{Float64}, q::Vector{Float64})
  @assert (length(p) == length(q)) "Size mismatch"
  r = 0
  for i=1:length(p)
    if p[i] != 0 && q[i] != 0
      r = r + p[i] * log2(p[i]/q[i])
    end
  end
  r
end

# mutual information
function MI(data::Vector{Int64})
  max = maximum(data)
  px  = fe1p(data[1:end-1])
  py  = fe1p(data[2:end])
  pxy = fe2p(hcat(data[1:end-1], data[2:end]))

  r = 0
  for x=1:length(px)
    for y=1:length(py)
      if abs(px[x]) > 0.00000000001 && abs(py[y]) > 0.00000000001 && abs(pxy[x,y]) > 0.00000000001
        r = r + pxy[x,y] * (log2(pxy[x,y]) - log2(px[x]*py[y]))
      end
    end
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
  l  = length(v)
  r  = counts(v, 1:m)
  r = r ./ l
  # just to get rid of the numerical inaccuracies and make sure its a probability distribution
  s = sum(r)
  r ./ s
end

end # module
