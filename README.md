# A Sampling Based Method for Tensor Ring

# Decomposition[3]

## Osman Asif Malik, Stephen Becker

## 1 Summary

Dimensionality reduction is an essential technique for multiway large-scale data,
i.e., tensor. Tensor ring (TR) decomposition has become popular due to its high
representation ability and flexibility[1].
In this paper, presented an estimation of usage of the leverage scores to attain
a method which has complexity sublinear in the number of input tensor entries.
Finally, our algorithms show superior performance in deep learning dataset compression.

## 2  Tensor-Ring Decomposition

Core-tensors could be estimated by arg min||TR(G(n))−X||F.
There are two common approaches to this fitting problem - SVD based and alternating least-squares (ALS) based, which is the method this paper is focused.
SVD methods are in general faster, however they are suffer higher reconstruction-
error.



The fitting objective can be solved iteratively by solving least-squares for each
core-tensor 

To reduce the size of least-squares problem in each internal iteration, the paper
offers sketch each core-tensor by matrix S, and proves that if J is bigger enough

Practically, the S matrix is never explicitly constructed, but realization of index
vector is sampled to construct the matrices



The main issue of this paper, in orientation of our course, is the using of leverage scoring is used to improve the
convergence time (iterations and computation time), while keeping the accuracy of ALS methods.


## References

 Osman Asif Malik and Stephen Becker.A Sampling Based Method for Ten-
sor Ring Decomposition. 2020. arXiv:2010.08581 [math.NA].
