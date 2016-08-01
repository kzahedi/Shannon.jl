include("distribution.jl")
include("const.jl")

export entropy

function entropy_ML(data::Vector{Int64}, base::Number)
  p = fe1p(data)
  return -sum([ p[x] > 0 ? (p[x] * log(base, p[x])) : 0 for x=1:size(p)[1]])
end

# implemented from [1] (see below)
function entropy_MLBC(data::Vector{Int64}, base::Number)
  p = fe1p(data)
  n = float(size(data)[1])
  S = float(size(p)[1])
  H = -sum([ p[x] > ϵ ? (p[x] * log(base, p[x])) : 0 for x=1:size(p)[1]])
  return H + (S-1) / (2.0 * n)
end

# implemented from [1] (see below)
function entropy_HT(data::Vector{Int64}, base::Number)
  p = fe1p(data)
  n = size(data)[1]
  return -sum([ p[x] > ϵ ? ((p[x] * log(base, p[x])) / (1.0 - ((1.0 - p[x])^n))) : 0 for x=1:size(p)[1]])
end

# implemented from [1] (see below)
function entropy_CS(data::Vector{Int64}, base::Number)
  m = maximum(v)
  n  = size(v)[1]
  c  = counts(v, 1:m)
  c = c ./ n
  # just to get rid of the numerical inaccuracies and make sure its a probability distribution
  s = sum(c)
  p = c ./ s
  C = 1.0 - float(sum(filter(x == 1, c))) / float(n)
  p = p .* C
  return -sum([ p[x] > ϵ ? ((p[x] * log(base, p[x])) / (1.0 - ((1.0 - p[x])^l))) : 0 for x=1:size(p)[1]])
end

function entropy(data::Vector{Int64}; base=2, mode="ML")
  modes  = ["ML", "Maximum Likelihood", 
            "MLBC", "Maximum Likelihood with Bias Correction",
            "Horovitz-Thompson", "HT",
            "ChaoShen", "Chao-Shen", "CS"]
  umodes = map(x->uppercase(x), modes)
  known_mode = uppercase(mode) in umodes
  pmodes = map(x->", $x",modes)
  pmodes = foldl(*, pmodes[1][3:end], pmodes[2:end])
  @assert known_mode "Mode may be any of the following: [$pmodes]."

  if uppercase(mode) in umodes[1:2]
    return entropy_ML(data, base) 
  elseif uppercase(mode) in umodes[3:4]
    return entropy_MLBC(data, base)
  elseif uppercase(mode) in umodes[5:6]
    return entropy_HT(data, base)
  elseif uppercase(mode) in umodes[7:9]
    return entropy_CS(data, base)
  end
  return nothing
end


# [1] A. Chao and T.-J. Shen. Nonparametric estimation of shannon’s index of diversity when there are unseen species in sample. Environmental and Ecological Statistics, 10(4):429–443, 2003.

