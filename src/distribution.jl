fe1ph(v::Vector{Int64})   = hist(v)[2] ./ size(v)[1]

function fe2p(v::Matrix{Int64}) # frequency estimation of one dimensional probability
  @assert (size(v)[2] == 2) "fe2p requires two columns."
  m1 = maximum(v[:,1])
  m2 = maximum(v[:,2])
  r  = counts(v[:,1], v[:,2], (1:m1, 1:m2))
  l  = size(v)[1]
  r  = r ./ l
  # just to get rid of the numerical inaccuracies and make sure its a probability distribution
  s  = sum(r)
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

function fe3p(data::Matrix{Int64})
  @assert (size(data)[2] == 3) "fe3p required three columns."
  x = data[:,1]
  y = data[:,2]
  z = data[:,3]

  mx = maximum(x)
  my = maximum(y)
  mz = maximum(z)

  p  = zeros(mx, my, mz)
  for index = 1:length(x)
    p[x[index], y[index], z[index]] = p[x[index], y[index], z[index]] + 1.0
  end
  p ./ float(length(x))
end

# frequency estimation of conditional probability distribution p(x|y,z)
function fe3c2p(data::Matrix{Int64})
  @assert (size(data)[2] == 3) "fe3c2p required three columns."

  pxyz = fe3p(data)
  pyz  = fe2p(data[:,[2:3]])

  for x = 1:size(data)[1]
    for y = 1:size(data)[2]
      for z = 1:size(data)[3]
        pxyz[x,y,z] = pxyz[x,y,z] / pyz[y,z]
      end
    end
  end
  pxyz
end

function fe2c1p(data::Matrix{Int64})
  @assert (size(data)[2] == 3) "fe2c1p required two columns."
  pxy = fe2p(data)
  py  = fe1p(data[:,1])
  @assert (mininum(py) > 0.0) "p(y) must have full support (no zero-valued entries)"
  for x=1:size(pxy)[1]
    for y=1:size(pxy)[2]
      pxy[x,y] = pxy[x,y] / py[y]
    end
  end
  pxy
end
