export bin_vector, bin_matrix, bin_value
export unbin_value, unbin_matrix, unbin_vector
export combine_binned_matrix, combine_binned_vector
export combine_and_relabel_binned_matrix
export unary_of_matrix
export relabel

bin_value(v::Float64, bins::Int64, min=-1.0, max=1.0) = round(Int64, minimum([1+floor((v-min) / (max - min) * (bins-0.1)), bins]))

bin_vector(vec::Vector{Float64}, min::Float64, max::Float64, bins::Int64) = map(v->bin_value(v, bins, min, max), vec)

unbin_vector(vec::Vector{Float64}, min::Float64, max::Float64, bins::Int64; mode="centre") = map(v->unbin_value(v, bins, min, max, mode=mode), vec)

function unbin_value(v::Int64, bins::Int64, min=-1.0, max=1.0; mode="centre")
  known_mode = (mode == "centre" ||
                mode == "lower"  ||
                mode == "upper")
  @assert known_mode "Mode may be any of the following: [\"centre\", \"lower\", \"upper\"]"

  delta = (max - min) / Float64(bins)
  u = (v - 1) * delta + min

  if     mode == "centre"
    return u + 0.5 * delta
  elseif mode == "upper"
    return u + delta
  end
  return u
end

function bin_matrix(m::Matrix{Float64}, min::Float64, max::Float64, bins::Int64)
  r = zeros(size(m))
  for i=1:size(m)[1]
    r[i,:] = bin_vector(squeeze(m[i,:],1), min, max, bins)
  end
  convert(Matrix{Int64}, r)
end

function unbin_matrix(m::Matrix{Float64}, min::Float64, max::Float64, bins::Int64; mode="centre")
  r = zeros(size(m))
  for i=1:size(m)[1]
    r[i,:] = unbin_vector(squeeze(m[i,:],1), min, max, bins, mode=mode)
  end
  convert(Matrix{Int64}, r)
end

function combine_binned_vector(v::Vector{Int64}, bins::Int64)
  convert(Int64, sum([v[i] * bins^(i-1) for i = 1:length(v)]))
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

relabel(v::Vector{Int64}) = indexin(v, sort(unique(v)))

combine_and_relabel_binned_matrix(data::Matrix{Int64}) = relabel(combine_binned_matrix(data))

unary_of_matrix(data::Matrix{Float64}, min::Float64, max::Float64, bins::Int64) = combine_and_relabel_binned_matrix(bin_matrix(data, min, max, bins))
