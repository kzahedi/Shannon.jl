include("distribution.jl")

export entropy

function entropy_emperical(data::Vector{Int64}, base::Number)
  p = fe1p(data)
  sum([ p[x] > 0 ? (-p[x] * log(base, p[x])) : 0 for x=1:size(p)[1]])
end

#= function entropy_chaoshen(p::Vector{Int64}, base::Number) =#
  #= p = fe1p(data) =#
  #= sum([ p[x] > 0 ? (-p[x] * log(base, p[x])) : 0 for x=1:size(p)[1]]) =#
#= end =#

function entropy(data::Vector{Int64}; base=2, mode="emperical", pseudocount=0)
  known_mode = (mode == "emperical")
  @assert known_mode "Mode may be any of the following: [\"emperical\"]."
  entropy_emperical(data, base) 

  #= known_mode = (mode == "emperical" || mode == "ChaoShen")  =#
  #= @assert known_mode "Mode may be any of the following: [\"emperical\", \"ChaoShen\"]." =#

  #= if mode == "emperical" =#
    #= entropy_emperical(data, base)  =#
  #= elseif mode == "ChaoShen" =#
    #= entropy_chaoshen(data, base) =#
  #= end =#
end
