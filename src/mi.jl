include("distribution.jl")

export MI, PI

# predictive information
PI(data::Vector{Int64}; base=2, mode="emperical") = MI(hcat(data[1:end-1], data[2:end]), base=base, mode=mode)

# mutual information
function MI(data::Matrix{Int64}; base=2, mode="emperical", pseudocount=0)

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


