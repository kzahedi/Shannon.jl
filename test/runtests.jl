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

@test 0           == entropy([1,1,1,1,1])
@test abs(log(2,6) - entropy([1,2,3,4,5,6])) < 0.0000001
@test abs(log(4,6) - entropy([1,2,3,4,5,6], base=4)) < 0.0000001

@test abs(5.540741 - entropy([1,2,3,4,5,6], mode="ChaoShen")) < 0.00001
@test abs(5.540741 - entropy([1,2,3,4,5,6], mode="ChaoShen", base=2)) < 0.00001
@test abs(1.667929 - entropy([1,2,3,4,5,6], mode="ChaoShen", base=10)) < 0.00001
@test abs(3.840549 - entropy([1,2,3,4,5,6], mode="ChaoShen", base=e)) < 0.00001

@test abs(1.632631 - entropy([1,2,2,3,3,3,4,4,4,4,5,6], mode="Dirichlet", pseudocount=0,base=e)) < 0.00001
@test abs(2.355389 - entropy([1,2,2,3,3,3,4,4,4,4,5,6], mode="Dirichlet", pseudocount=0)) < 0.00001

println(entropy([1,2,3,4,5,6], mode="MillerMadow", base=e))
@test abs(2.208426 - entropy([1,2,3,4,5,6], mode="MillerMadow", base=e)) < 0.00001
