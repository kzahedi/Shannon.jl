using Shannon
using Base.Test

v = [0:10] / 10
r = [1,1:10]

@test r == bin_vector(v, 0.0, 1.0, 10)

r = zeros((10,5))
i = 1
for j=1:10
  for k=1:5
    r[j,k]=i
    i = i+1
  end
end
v = r ./ 50
r = convert(Matrix{Int64}, r)

@test r == bin_matrix(v, 0.0, 1.0, 50)

c = [[1 0 0 0], [1 2 0 0], [1 2 3 0], [1 2 3 4]]
r = [1, 1 + 2 * 4^1, 1 + 2*4^1 + 3*4^2, 1 + 2*4^1 + 3*4^2 + 4*4^3]

@test 1    == combine_binned_vector(squeeze(c[1,:],1), 10)
@test 21   == combine_binned_vector(squeeze(c[2,:],1), 10)
@test 321  == combine_binned_vector(squeeze(c[3,:],1), 10)
@test 4321 == combine_binned_vector(squeeze(c[4,:],1), 10)

@test r[1] == combine_binned_vector(squeeze(c[1,:],1), 4)
@test r[2] == combine_binned_vector(squeeze(c[2,:],1), 4)
@test r[3] == combine_binned_vector(squeeze(c[3,:],1), 4)
@test r[4] == combine_binned_vector(squeeze(c[4,:],1), 4)

@test r    == combine_binned_matrix(c)

#= println(v) =#
#= vb = bin_matrix(v, -1.0, 1.0, 10) =#
#= cb = combine_binned_matrix(vb) =#
#= println(cb) =#
