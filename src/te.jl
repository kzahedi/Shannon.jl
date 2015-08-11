include("distribution.jl")

export TE

ϵ = 0.0000001

function TE_ML(data::Matrix{Int64}; base=2) # D_KL(p(x|y,z)||p(x|y))

  x2         = data[2:end,1]
  x1         = data[1:end-1,1]
  y1         = data[1:end-1,2]
  x_max      = maximum(data[:,1])
  y_max      = maximum(data[:,2])
  px2x1y1    = Shannon.fe3p(hcat(x2,x1,y1))
  px1y1      = sum(px2x1y1, 1)
  px2x1      = sum(px2x1y1, 3)
  px1        = sum(px2x1y1, (1,3))

  px2_c_x1y1 = zeros(x_max, x_max, y_max)
  px2_c_x1   = zeros(x_max, x_max)

  for x2_index = 1:x_max
    for x1_index = 1:x_max
      px2_c_x1[x2_index, x1_index] = px2x1[x2_index, x1_index, 1] / px1[1, x1_index, 1]
      for y1_index = 1:y_max
        px2_c_x1y1[x2_index, x1_index, y1_index] = px2x1y1[x2_index, x1_index, y1_index] / px1y1[1, x1_index, y1_index]
      end
    end
  end

  r = 0
  for x2_index = 1:x_max
    for x1_index = 1:x_max
      for y1_index = 1:y_max
        if abs(px2x1y1[x2_index, x1_index, y1_index]) > ϵ && abs(px2_c_x1y1[x2_index, x1_index, y1_index]) > ϵ && abs(px2_c_x1[x2_index, x1_index]) > ϵ
          r = r + px2x1y1[x2_index, x1_index, y1_index] * (log(base, px2_c_x1y1[x2_index, x1_index, y1_index]) - log(base, px2_c_x1[x2_index, x1_index]))
      end
    end
  end
  return r
end

# Transfer Entropy
function TE(data::Matrix{Int64}; base=2, mode="ML", pseudocount=0)
  @assert (size(data)[2] == 2) "TE required two columns (X,Y) to calcualte TE(X->Y)."

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
