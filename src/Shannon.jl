module Shannon

export KL

function KL(p::Vector{Float64}, q::Vector{Float64})
  r = 0
  for i=1:length(p)
    if p[i] != 0 && q[i] != 0
      r = r + p[i] * log2(p[i]/q[i])
    end
  end
  r
end

end # module
