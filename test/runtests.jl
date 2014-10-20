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

# entropy tests
@test 0           == entropy([1,1,1,1,1])
@test abs(log(2,6) - entropy([1,2,3,4,5,6])) < 0.0000001
@test abs(log(4,6) - entropy([1,2,3,4,5,6], base=4)) < 0.0000001

#= @test abs(5.540741 - entropy([1,2,3,4,5,6], mode="ChaoShen")) < 0.00001 =#
#= @test abs(5.540741 - entropy([1,2,3,4,5,6], mode="ChaoShen", base=2)) < 0.00001 =#
#= @test abs(1.667929 - entropy([1,2,3,4,5,6], mode="ChaoShen", base=10)) < 0.00001 =#
#= @test abs(3.840549 - entropy([1,2,3,4,5,6], mode="ChaoShen", base=e)) < 0.00001 =#


# test basic functions - bin a value
@test 1  == bin_value(-1.0, 10)
@test 1  == bin_value(-0.9, 10)
@test 1  == bin_value(-0.8, 10)
@test 2  == bin_value(-0.7, 10)
@test 2  == bin_value(-0.6, 10)
@test 3  == bin_value(-0.5, 10)
@test 3  == bin_value(-0.4, 10)
@test 4  == bin_value(-0.3, 10)
@test 4  == bin_value(-0.2, 10)
@test 5  == bin_value(-0.1, 10)
@test 5  == bin_value(0.0,  10)
@test 6  == bin_value(0.1,  10)
@test 6  == bin_value(0.2,  10)
@test 7  == bin_value(0.3,  10)
@test 7  == bin_value(0.4,  10)
@test 8  == bin_value(0.5,  10)
@test 8  == bin_value(0.6,  10)
@test 9  == bin_value(0.7,  10)
@test 9  == bin_value(0.8,  10)
@test 10 == bin_value(0.9,  10)
@test 10 == bin_value(1.0,  10)

# test basic functions - bin a vector
@test [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10] == bin_vector([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], -1.0, 1.0, 10)
@test [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10] != bin_vector([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], -1.0, 1.0, 10)

# test basic functions - bin a matrix
@test [[1, 2, 3] [4, 5, 6] [7, 8, 9]] == bin_matrix([[-1.0, -0.7, -0.5] [-0.3, -0.1, 0.1] [0.3, 0.5, 0.7]], -1.0, 1.0, 10)
@test [[2, 2, 3] [4, 5, 6] [7, 8, 9]] != bin_matrix([[-1.0, -0.7, -0.5] [-0.3, -0.1, 0.1] [0.3, 0.5, 0.7]], -1.0, 1.0, 10)

# test inverse binning
@test abs(-1.0 - unbin_value(1,  10, -1.0, 1.0, mode="lower")) < 0.0001
@test abs(-0.8 - unbin_value(2,  10, -1.0, 1.0, mode="lower")) < 0.0001
@test abs(-0.6 - unbin_value(3,  10, -1.0, 1.0, mode="lower")) < 0.0001
@test abs(-0.4 - unbin_value(4,  10, -1.0, 1.0, mode="lower")) < 0.0001
@test abs(-0.2 - unbin_value(5,  10, -1.0, 1.0, mode="lower")) < 0.0001
@test abs( 0.0 - unbin_value(6,  10, -1.0, 1.0, mode="lower")) < 0.0001
@test abs( 0.2 - unbin_value(7,  10, -1.0, 1.0, mode="lower")) < 0.0001
@test abs( 0.4 - unbin_value(8,  10, -1.0, 1.0, mode="lower")) < 0.0001
@test abs( 0.6 - unbin_value(9,  10, -1.0, 1.0, mode="lower")) < 0.0001
@test abs( 0.8 - unbin_value(10, 10, -1.0, 1.0, mode="lower")) < 0.0001

@test abs(-0.8 - unbin_value(1,  10, -1.0, 1.0, mode="upper")) < 0.0001
@test abs(-0.6 - unbin_value(2,  10, -1.0, 1.0, mode="upper")) < 0.0001
@test abs(-0.4 - unbin_value(3,  10, -1.0, 1.0, mode="upper")) < 0.0001
@test abs(-0.2 - unbin_value(4,  10, -1.0, 1.0, mode="upper")) < 0.0001
@test abs(-0.0 - unbin_value(5,  10, -1.0, 1.0, mode="upper")) < 0.0001
@test abs( 0.2 - unbin_value(6,  10, -1.0, 1.0, mode="upper")) < 0.0001
@test abs( 0.4 - unbin_value(7,  10, -1.0, 1.0, mode="upper")) < 0.0001
@test abs( 0.6 - unbin_value(8,  10, -1.0, 1.0, mode="upper")) < 0.0001
@test abs( 0.8 - unbin_value(9,  10, -1.0, 1.0, mode="upper")) < 0.0001
@test abs( 1.0 - unbin_value(10, 10, -1.0, 1.0, mode="upper")) < 0.0001

@test abs(-0.9 - unbin_value(1,  10, -1.0, 1.0, mode="centre")) < 0.0001
@test abs(-0.7 - unbin_value(2,  10, -1.0, 1.0, mode="centre")) < 0.0001
@test abs(-0.5 - unbin_value(3,  10, -1.0, 1.0, mode="centre")) < 0.0001
@test abs(-0.3 - unbin_value(4,  10, -1.0, 1.0, mode="centre")) < 0.0001
@test abs(-0.1 - unbin_value(5,  10, -1.0, 1.0, mode="centre")) < 0.0001
@test abs( 0.1 - unbin_value(6,  10, -1.0, 1.0, mode="centre")) < 0.0001
@test abs( 0.3 - unbin_value(7,  10, -1.0, 1.0, mode="centre")) < 0.0001
@test abs( 0.5 - unbin_value(8,  10, -1.0, 1.0, mode="centre")) < 0.0001
@test abs( 0.7 - unbin_value(9,  10, -1.0, 1.0, mode="centre")) < 0.0001
@test abs( 0.9 - unbin_value(10, 10, -1.0, 1.0, mode="centre")) < 0.0001

