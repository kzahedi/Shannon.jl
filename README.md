# Shannon.jl
A collection of quantifications related to Shannon's information theory and methods to discretise data.

## Example

    using Shannon
    xy = hcat([sin(x) + randn() * .1 for x=0:0.01:2pi], [cos(x) + randn() * .1 for x=0:0.01:2pi])
    bxy = bin_matrix(xy, -1.0, 1.0, 10)
    c=combine_binned_matrix(bxy)
    c=relabel(c)
    H = entropy(c)
    I = MI(bxy)

A faster way is to call 

    unary_of_matrix(xy, -1.0, 1.0, 10)
   
which is a short cut for the lines below

    bxy = bin_matrix(xy, -1.0, 1.0, 10)
    c=combine_binned_matrix(bxy)
    c=relabel(c)
    
## Entropy estimators
The estimators are implemented from the following list of publications:

[1] A. Chao and T.-J. Shen. Nonparametric estimation of shannon’s index of diversity when there are unseen species in sample. Environmental and Ecological Statistics, 10(4):429–443, 2003.

and the function call is

    entropy(data, base=2, mode="ML")

where

<table>
<tr> <td> **data** </td> <td> is the discrete data (*Vector{Int64}*)</td></tr>
<tr> <td valign=top> **mode** </td> <td> determines which estimator should be used (see below). It is *not* case-sensitive </td> </tr>
<tr> <td> **base** </td>  <td> determines the base of the logarithm </td> </tr>
 </table>

###Maximum Likelihood Estimator
This is the default estimator.

    entropy(data)
    entropy(data, mode="ML")
    entropy(data, mode="Maximum Likelihood")

###Maximum Likelihood Estimator with Bias Correction (implemented from [1])
    
    entropy(data, mode="MLBC")
    entropy(data, mode="Maximum Likelihood with Bias Compensation")
    
###Horovitz-Thompson Estimator (implemented from [1])
    

    entropy(data, mode="HT")
    entropy(data, mode="Horovitz-Thompson")


###Chao-Shen Estimator (implemented from [1])


    entropy(data, mode="CS")
    entropy(data, mode="Chao-Shen")
    entropy(data, mode="ChaoShen")


#### Setting the base

    entropy(data, base=2) [ this is the default ]
    entropy(data, mode="HT", base=10)
    
## Mutual Information estimators
Currently, only the _maximum likelihood estimator_ is implemented. It can be used with different bases:

    MI(xy, base=2) [ this is the default ]
    MI(xy, base=10)

**xy** is a two-dimensional matrix with **n** rows and two columns.

## Predictive Information
This in an implementation of the one-step predictive information, which is given by the mutual information of consecutive data points. If x is the data vector, then:

    PI(x) = MI(hcat(x[1:end-1], x[2:end]))
    PI(x,[base],[mode]) = MI(x[1:end-1], x[2:end], base, mode)
    
## Kullback-Leibler Divergence
This function calculates the KL-Divergence on two probability distributions, and is essentially given by:

    KL(p,q)=  sum([(p[i] != 0 && q[i] != 0)? p[i] * log(base, p[i]/q[i]) : 0 for i=1:length(p)])

**p**,**q** must be valid probability distributions, i.e.

    x >= 0 for x in p
    y >= 0 for y in q
    sum(p) == sum(q) == 1.0
