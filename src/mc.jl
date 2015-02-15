include("distribution.jl")
include("te.jl")

export MC

# data[w', w, a] -> D_KL(p(w'|w,a)||p(w'|a))
function MC_MCW(data::Matrix{Int64}; base=2) = TE(data, base=base) / log(base, maximum(hcat(data[:,1], data[:,2])))

# data[w', w, a] -> D_KL(p(w'|w,a)||p(w'|w))
function MC_MCA(data::Matrix{Int64}; base=2) = TE(hcat(data[:,1], data[:,3], data[:,2]), base=base) / log(base, maximum(hcat(data[:,1], data[:,2])))

# data[s', s, a] -> D_KL(p(s'|do(a))||p(s'|do(s)))
function C_A(data::Matrix{Int64})
  psa = fe2p(hcat(data[:,2], data[:,3]))

  0
end

# data[s', s, a] -> D_KL(p(s'|s,a)||p(s'|a))
function C_W(data::Matrix{Int64})
  0
end

# Morphological Computation 
function MC(data::Matrix{Int64}; base=2, mode="ML", pseudocount=0)
  modes      = ["MCA", "MC as effect of A",
                "MCW", "MC as effect of W",
                "ASOC A",
                "ASOC W",
                "C A",
                "C W"
                ]
  umodes     = map(x->uppercase(x), modes)
  known_mode = uppercase(mode) in umodes
  pmodes     = map(x->", $x",modes)
  pmodes     = foldl(*, pmodes[1][3:end], pmodes[2:end])
  @assert known_mode "Mode may be any of the following: [$pmodes]."

  if uppercase(mode) in umodes[1:2]
    return MC_MCA(data, base) 
  elseif uppercase(mode) in umodes[3:4]
    return MC_MCW(data, base) 
  elseif uppercase(mode) in umodes[5]
    return MC_MCA(data, base) 
  elseif uppercase(mode) in umodes[6]
    return MC_MCW(data, base) 
  elseif uppercase(mode) in umodes[7]
    return C_A(data, base) 
  elseif uppercase(mode) in umodes[8]
    return C_W(data, base) 
  end
end


