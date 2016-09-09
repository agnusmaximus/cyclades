# cyclades

This repository contains code for Cyclades, a general framework for
parallelizing stochastic optimization algorithms in a shared memory
setting. See https://arxiv.org/abs/1605.09721 for more information.

Here we implement SGD, SVRG and SAGA for sparse stochastic gradient
descent methods on applications including matrix completion, graph
eigenvalues, word embeddings and least squares.

# Building
After cloning the repository, be sure to fetch the gflags submodule by doing
```c++
git submodule init && git submodule update
```
After the submodule fetches use cmake
```c++
cmake .
```
Then make
```c++
make
```