export fe1ph, fe2p, fe1p

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


