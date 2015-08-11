include("distribution.jl")

ϵ = 0.0000001

# data[w', w, a] -> |MI(W';W|A)|
function MC_W(data::Matrix{Int64}; base=2)
  @assert (size(data)[2] == 2) "MC_W required two columns (W,A)."
  w2      = data[2:end,1]
  w1      = data[1:end-1,1]
  a1      = data[1:end-1,2]
  m       = maximum(data[:,1])
  pw2w1a1 = fe3p(hcat(w2,w1,a1)) # add possibility to use other estimators
  return CMI(pw2w1a1, base=base) / log(base,m)
end

# data[w', w, a] -> 1 - |MI(W';A|W)|
function MC_A(data::Matrix{Int64}; base=2)
  @assert (size(data)[2] == 2) "MC_W required two columns (W,A)."
  w2      = data[2:end,1]
  w1      = data[1:end-1,1]
  a1      = data[1:end-1,2]
  m       = maximum(data[:,1])
  pw2a1w1 = fe3p(hcat(w2,a1,w1)) # add possibility to use other estimators
  return 1.0 - (CMI(pw2a1w1, base=base) / log(base,m))
end

# CA = ∑ p(s,a) p(s'|do(a)) log (p(s'|do(a)) / p(s'|do(s)))
function C_A(data::Matrix{Int64}; base=2)
  @assert (size(data)[2] == 2) "C_A required two columns (S,A)."
  s_max   = maximum(data[:,1])
  a_max   = maximum(data[:,2])
  s2      = data[2:end,1]
  s1      = data[1:end-1,1]
  a1      = data[1:end-1,2]
  m       = maximum(data[:,1])
  ps2s1a1 = fe3p(hcat(s2,s1,a1)) # add possibility to use other estimators

  ps1a1   = sum(ps2s1a1, 1)
  pa1     = sum(ps2s1a1, (1,2))
  ps1     = sum(ps2s1a1, (1,3))

  # p(s'|s,a) = p(s',s,a) / p(s,a)
  ps2_c_s1a1 = zeros(size(ps2s1a1))
  for s2_index = 1:s_max
    for s1_index = 1:s_max
      for a1_index = 1:a_max
        if abs(ps1a1[1, s1_index, a1_index]) > ϵ
          ps2_c_s1a1[s2_index, s1_index, a1_index] = ps2s1a1[s2_index, s1_index, a1_index] / ps1a1[1, s1_index, a1_index]
        end
      end
    end
  end

  # p(s'| do(a)) = ∑ p(s'|s,a) * p(s)
  ps2_do_a1 = zeros(s_max, a_max)
  for s2_index = 1:s_max
    for a1_index = 1:a_max
      for s1_index = 1:s_max
        ps2_do_a1[s2_index, a1_index] = ps2_do_a1[s2_index, a1_index] + ps2_c_s1a1[s2_index, s1_index, a1_index] * ps1[1,s1_index, 1]
      end
    end
  end

  # p(a|s) = p(s,a) / p(s)
  pa1_c_s1 = zeros(a_max, s_max)
  for a1_index = 1:a_max
    for s1_index = 1:s_max
      if abs(ps1[s1_index]) > ϵ
        pa1_c_s1[a1_index, s1_index] = ps1a1[1, s1_index, a1_index] / ps1[1, s1_index, 1]
      end
    end
  end

  # p(s'|do(s)) = ∑ p(a|s) p(s'|do(a))
  ps2_do_s1 = zeros(s_max, s_max)
  for s2_index = 1:s_max
    for s1_index = 1:s_max
      for a1_index = 1:a_max
        ps2_do_s1[s2_index, s1_index] = ps2_do_s1[s2_index, s1_index] + pa1_c_s1[a1_index, s1_index] * ps2_do_a1[s2_index, a1_index]
      end
    end
  end


  # CA = ∑ p(s,a) p(s'|do(a)) log (p(s'|do(a)) / p(s'|do(s)))
  r = 0.0
  for s2_index = 1:s_max
    for s1_index = 1:s_max
      for a1_index = 1:a_max
        if abs(ps1a1[1, s1_index, a1_index]) > ϵ && abs(ps2_do_a1[s2_index, a1_index]) > ϵ && abs(ps2_do_s1[s2_index, s1_index]) > ϵ
          r = r + ps1a1[1, s1_index, a1_index] * ps2_do_a1[s2_index, a1_index] * (log(base, ps2_do_a1[s2_index, a1_index]) - log(base, ps2_do_s1[s2_index, s1_index]))
        end
      end
    end
  end
  return r / log(base, s_max)
end

# data[s', s, a] -> D_KL(p(s'|s,a)||p(s'|a))
function C_W(data::Matrix{Int64}; base=2)
  @assert (size(data)[2] == 2) "C_A required two columns (S,A)."
  s_max   = maximum(data[:,1])
  a_max   = maximum(data[:,2])
  s2      = data[2:end,1]
  s1      = data[1:end-1,1]
  a1      = data[1:end-1,2]
  m       = maximum(data[:,1])
  ps2s1a1 = fe3p(hcat(s2,s1,a1)) # add possibility to use other estimators

  ps1     = sum(ps2s1a1, (1,3))
  pa1     = sum(ps2s1a1, (1,2))
  ps2s1   = sum(ps2s1a1, 3)
  ps1a1   = sum(ps2s1a1, 1)
  ps2a1   = sum(ps2s1a1, 2)

  # p(s'|s) = p(s',s) / p(s)
  ps2_c_s1 = zeros(s_max, s_max)
  for s2_index = 1:s_max
    for s1_index = 1:s_max
      if abs(ps1[s1_index]) > ϵ
        ps2_c_s1[s2_index, s1_index] = ps2s1[s2_index, s1_index] / ps1[s1_index]
      end
    end
  end

  # p(s'|a) = p(s',a) / p(a)
  ps2_c_a1 = zeros(s_max, a_max)
  for s2_index = 1:s_max
    for a1_index = 1:a_max
      if abs(pa1[a1_index]) > ϵ
        ps2_c_a1[s2_index, a1_index] = ps2a1[s2_index, 1, a1_index] / pa1[a1_index]
      end
    end
  end

  # p(a|s) = p(a,s) / p(s)
  pa1_c_s1 = zeros(a_max, s_max)
  for a1_index = 1:a_max
    for s1_index = 1:s_max
      if abs(ps1[1, s1_index, 1]) > ϵ
        pa1_c_s1[a1_index, s1_index] = ps1a1[1, s1_index, a1_index] / ps1[1, s1_index, 1]
      end
    end
  end

  #̂̂p^(s'|s) = ∑ p(s'|a) p(a|s)
  phat_s2_c_s1 = zeros(s_max, s_max)
  for s2_index = 1:s_max
    for s1_index = 1:s_max
      for a1_index = 1:a_max
        phat_s2_c_s1[s2_index, s1_index] = ps2_c_a1[s2_index, a1_index] * pa1_c_s1[a1_index, s1_index]
      end
    end
  end

  r = 0
  for s1_index = 1:s_max
    for s2_index = 1:s_max
      if abs(ps2s1[s2_index, s1_index, 1]) > ϵ && abs(ps2_c_s1[s2_index, s1_index]) > ϵ && abs(phat_s2_c_s1[s2_index, s1_index]) > ϵ
        r = r + ps2s1[s2_index, s1_index, 1] * (log(base, ps2_c_s1[s2_index, s1_index]) - log(base, phat_s2_c_s1[s2_index, s1_index]))
      end
    end
  end

  return r / log(base, s_max)
end

# Morphological Computation
function quantify(data::Matrix{Int64}; base=2, mode="MC A", pseudocount=0)
  modes      = ["MC A",
  "MC W",
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
    return MC_A(data, base)
  elseif uppercase(mode) in umodes[3:4]
    return MC_W(data, base)
  elseif uppercase(mode) in umodes[5]
    return MC_A(data, base)
  elseif uppercase(mode) in umodes[6]
    return MC_W(data, base)
  elseif uppercase(mode) in umodes[7]
    return C_A(data, base)
  elseif uppercase(mode) in umodes[8]
    return C_W(data, base)
  end
end
