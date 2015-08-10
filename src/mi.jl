include("distribution.jl")

export MI, PI

ϵ = 0.0000001

# predictive information
PI(data::Vector{Int64}; base=2, mode="ML") = MI(hcat(data[1:end-1], data[2:end]), base=base, mode=mode)

function MI_ML(data::Matrix{Int64}; base=2)
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

# mutual information
function MI(data::Matrix{Int64}; base=2, mode="ML", pseudocount=0)

  modes      = ["ML", "Maximum Likelihood"]
  umodes     = map(x->uppercase(x), modes)
  known_mode = uppercase(mode) in umodes
  pmodes     = map(x->", $x",modes)
  pmodes     = foldl(*, pmodes[1][3:end], pmodes[2:end])
  @assert known_mode "Mode may be any of the following: [$pmodes]."

  if uppercase(mode) in umodes[1:2]
    MI_ML(data, base=base)
  end

end

# MI(X;Y) = ∑ p(x,y) * log( p(x,y) / (p(x) * p(y)))
function MI(pxy::Matrix{Float64}; base=2)
  px = sum(pxy,1)
  py = sum(pxy,2)
  r  = 0
  for x = 1:size(pxy)[1]
    for y = 1:size(pxy)[2]
      if abs(px[x]) > ϵ && abs(py[y]) > ϵ && abs(pxy[x,y]) > ϵ
        r = r + pxy[x,y] * (log(base, pxy[x,y]) - log(base, px[x] * py[y]))
      end
    end
  end
  return r
end

# MI(X;Y|Z) = ∑ p(x,y,z) * log( p(x,y|z) / (p(x|z) * p(y|z)))
function CMI(pxyz::Array{Float64,3}; base=2)
  r = 0
  pz  = sum(pxyz, (1,2))
  pxz = sum(pxyz, 2)
  pyz = sum(pxyz, 1)
  for x = 1:size(pxyz)[1]
    for y = 1:size(pxyz)[2]
      for z = 1:size(pxyz)[3]
        if abs(pxyz[x,y,z]) > ϵ && abs(pz[z]) > ϵ && abs(pxz[x,z]) > ϵ && abs(pyz[y,z]) > ϵ
           r = r + pxyz[x,y,z] * ( log(base, pxyz[x,y,z] / pz[z]) - log(base, pxz[x,z] / pz[z] * pyz[y,z] / pz[z]))
         end
      end
    end
  end
  return r
end
