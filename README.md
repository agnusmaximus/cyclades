# Cyclades

This repository contains code for Cyclades, a general framework for
parallelizing stochastic optimization algorithms in a shared memory
setting. See https://arxiv.org/abs/1605.09721 for more information.

Here we implement SGD, SVRG and SAGA for sparse stochastic gradient
descent methods applied to problems including matrix completion, graph
eigenvalues, word embeddings and least squares.

# Overview

Cyclades is a general framework for parallelizing stochastic
optimization algorithms in a shared memory setting. By partitioning
the conflict graph of datapoints into batches of non-conflicting
updates, serializability can be maintained under execution of multiple
cores.

<div align="center"><img src="https://raw.github.com/agnusmaximus/cyclades/master/images/Cyclades.png" height="400" width="410" ></div>

Cyclades carefully samples updates, then finds conflict-groups, and
allocates them across cores. Then, each core asynchronously updates
the shared model, without incurring any read/write conflicts. This is
possible by processing all the conflicting updates within the same
core. After the processing of a batch is completed, the above is
repeated, for as many iterations as required.

# Experiments

Maintaining serializability confers numerous benefits, and the
additional overhead of partitioning the conflict graph does not hinder
performance too much. In fact, in some cases the avoidance of conflicts and
the slightly better cache behavior of Cyclades leads to better
performance.

<div align="center"><img src="https://raw.github.com/agnusmaximus/cyclades/master/images/Matrix%20Completion%208%20threads%20-%20Movielens%2010m.png" height="400" width="410" ></div>
<em> Here we see that Cyclades initially starts slower than Hogwild, but due to cache locality and avoiding conflicts ends up being slightly faster. Both training methods were run for the same number of epochs in this plot, with the same learning rate. </em>

Additionally, for various variance reduction algorithms we find that
Cyclades' serial equivalance allows it to outperform Hogwild in terms
of convergence.

For full experiment details please refer to the paper.

# Building
After cloning the repository, fetch the gflags submodule with
```c++
git submodule init && git submodule update
```
After the submodule fetches use cmake to generate a build file
```c++
mkdir build && cmake ..
```
Then make to compile
```c++
make
```

# Fetching data
To fetch all experiment data, from the project home directory, run
```c++
cd data && sh fetch_all_data.sh && cd ..
```

# Running

After compilation, a single executable called cyclades will be
built. There are numerous flags that control the specifics of
execution, such as learning rate, training type, number of epochs to
run, etc.

To see a list of flags that can be set, run
```c++
./cyclades --help
```

A quick example to run after compiling and fetching the data is (run from the home directory)
```c++
./build/cyclades   --print_loss_per_epoch  --print_partition_time  --n_threads=2 --learning_rate=1e-2  -matrix_completion  -cyclades_trainer  -cyclades_batch_size=800 -n_epochs=20 -sgd --data_file="data/movielens/ml-1m/movielens_1m.data"
```
