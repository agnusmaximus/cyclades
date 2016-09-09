# cyclades

This repository contains code for Cyclades, a general framework for
parallelizing stochastic optimization algorithms in a shared memory
setting. See https://arxiv.org/abs/1605.09721 for more information.

Here we implement SGD, SVRG and SAGA for sparse stochastic gradient
descent methods applied to applications including matrix completion,
graph eigenvalues, word embeddings and least squares.

# Building
After cloning the repository, fetch the gflags submodule with
```c++
git submodule init && git submodule update
```
After the submodule fetches use cmake to generate a build file
```c++
cmake .
```
Then make to compile
```c++
make
```

# Fetching data
TODO

# Running

After compilation, a single executable called cyclades will be
built. There are numerous flags that control the specifics of
execution, such as learning rate, training type, number of epochs to
run, etc.

To see a list of flags that can be set, run
```c++
./cyclades --help
```

A quick example to run after compiling and fetching the data is
```c++
./cyclades   --print_loss_per_epoch  --print_partition_time  --n_threads=2 --learning_rate=1e-2  -matrix_completion  -cyclades_trainer  -cyclades_batch_size=800 -n_epochs=20 -sgd --data_file="data/movielens/ml-1m/movielens-1m.data"
```
