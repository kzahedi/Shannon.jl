# Shannon
A collection of quantifications related to Shannon's information theory and methods to discretise data.

### Example

    using Shannon
    xy = hcat([sin(x) + randn() * .1 for x=0:0.01:2pi], [cos(x) + randn() * .1 for x=0:0.01:2pi])
    bxy = bin_matrix(xy, -1.0, 1.0, 10)
    c=combine_binned_matrix(bxy)
    c=relabel(c) # remove unused bin from the data vector
    H = entropy(c)
    I = MI(bxy)
    
## Entropy estimators
The estimators were taken from Jean Hausser's and Korbinian Strimmer's R package _entropy_. For more information about their work, please have a look [here](http://cran.r-project.org/web/packages/entropy/index.html).

    entropy(x,mode="empirical") [ this is the default ]
    entropy(x,mode="ChaoShen")
    entropy(x,mode="Dirichlet", pseudocount=10)
    entropy(x,mode="MillerMadow")

_x_ is a of type *Vector{Int64}*
	
### Setting the base

    entropy(x,base=2) [ this is the default ]
    entropy(x,mode="MillerMadow", base=10)
    
## Mutual Information estimators
Currently, the _emperical_ estimator is implemented, but different bases can be used:

    MI(xy, base=2) [ this is the default ]
    MI(xy, base=10)

_xy_ is a two-dimensional matrix with _n_ rows and two columns.

## Predictive Information
This in an implementation of the one-step predictive information, which is given by the mutual information of consecutive data points. If x is the data vector, then:

    PI(x) = MI(hcat(x[1:end-1], x[2:end]))
    PI(x,[base],[mode]) = MI(x[1:end-1], x[2:end], base, mode)
    
## Kullback-Leibler Divergence
This function calculates the KL-Divergence on two probability distributions, and is essentially given by:

    KL(p,q)=  sum([(p[i] != 0 && q[i] != 0)? p[i] * log(base, p[i]/q[i]) : 0 for i=1:length(p)])

_p,q_ must be valid probability distributions, i.e.

    x >= 0 for x in p
    y >= 0 for y in q
    sum(p) == sum(q) == 1.0
