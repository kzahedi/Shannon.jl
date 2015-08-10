include("distribution.jl")

export TE

function TE_ML(data::Matrix{Int64}; base=2) # D_KL(p(x|y,z)||p(x|y))

  pxyz    = fe3p(data)
  px_c_yz = fe3c2p(data)
  px_c_y  = fe2c1p(data[:,[1:2]])

  r = 0
  for x=1:size(data)[1]
    for y=1:size(data)[2]
      for z=1:size(data)[3]
        if px_c_yz[x, y, z] != 0.0 && px_c_y[x, y] != 0.0
          r = r + pxyz[x, y, z] * (log(base, px_c_yz[x, y, z]) - log(base, px_c_y[x, y]))
        end
      end
    end
  end
  r
end

# Transfer Entropy
function TE(data::Matrix{Int64}; base=2, mode="ML", pseudocount=0)

  modes      = ["ML", "Maximum Likelihood"]
  umodes     = map(x->uppercase(x), modes)
  known_mode = uppercase(mode) in umodes
  pmodes     = map(x->", $x",modes)
  pmodes     = foldl(*, pmodes[1][3:end], pmodes[2:end])
  @assert known_mode "Mode may be any of the following: [$pmodes]."

  if uppercase(mode) in umodes[1:2]
    return TE_ML(data, base)
  end
end
