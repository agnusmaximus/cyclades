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

<div align="center"><img src="https://raw.github.com/agnusmaximus/cyclades/master/images/Cyclades.png" height="400" width="490" ></div>

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

<div align="center"><img src="https://raw.github.com/agnusmaximus/cyclades/master/images/Matrix%20Completion%20Speedup.png" width="350" height="300" /></div>

<em> Cyclades initially starts slower than Hogwild due to the overhead
of partitioning the conflict graph. But by having better locality and
avoiding conflicts Cyclades ends up slightly faster in terms of
running time. In the plots both training methods were run for the same
number of epochs, with the same learning rate. Note this graph was
generated using the "custom" updater to optimize for
performance. </em>

Additionally, for various variance reduction algorithms we find that
Cyclades' serial equivalance allows it to outperform Hogwild in terms
of convergence.

<div align="center"><img
src="https://raw.github.com/agnusmaximus/cyclades/master/images/SAGA%20Least%20Squares%202%20threads%20-%20NH2010.png"
height="400" width="525" ></div> <em> On multithread SAGA, the
serializability of Cyclades allows it to use a larger stepsize than
Hogwild. With higher stepsizes, Hogwild diverges due to conflicts. </em>

For full experiment details please refer to the paper.

# Building
<em> Note that compilation requires [git](https://git-scm.com/), [make](https://www.gnu.org/software/make/), [cmake](https://cmake.org/), [OpenMP](http://openmp.org/wp/), and [wget](https://www.gnu.org/software/wget/).
     Additionally, on a Mac, [Xcode Command Line Tools](http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/) and [ClangOMP++](https://clang-omp.github.io/) are required.
</em>

After cloning the repository, cd into the project directory
```c++
cd cyclades
```
Fetch the gflags submodule with
```c++
git submodule init && git submodule update
```
After the submodule fetches use cmake to generate a build file.

On Linux do
```c++
cmake .
```
While on Mac OS X do
```c++
cmake -DCMAKE_CXX_COMPILER=clang-omp++ .
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
./cyclades   --print_loss_per_epoch  --print_partition_time  --n_threads=2 --learning_rate=1e-2  -matrix_completion  -cyclades_trainer  -cyclades_batch_size=800 -n_epochs=20 -sparse_sgd --data_file="data/movielens/ml-1m/movielens_1m.data"
```

# Guide On Writing Custom Models

   Writing a model that can be optimized using Hogwild and Cyclades is
   straightforward. The two main classes that need to be overridden by
   the user are the `Model` and `Datapoint` classes, which capture all
   necessary information required for optimization.

   The `Model` class is a wrapper around the user-defined model data,
   specifying methods that operate on the model (such as computing
   gradients and loss).  The `Datapoint` class is a wrapper around the
   individual data elements used to train the model.

   After defining the `Datapoint` and `Model` subclasses, the user can
   run Cyclades/Hogwild by including "run.h" and calling
   `Run<CustomModel, CustomDatapoint>()`.

## Data File Reading / Data File Format

   The data file specified by the `--data_file` flag should contain
   information to initialize the model, as well as the individual data
   points that are used for training.

   The first line of the data file is fed to the constructor of the
   model, and each subsequent line is used to instantiate separate
   instances of the `Datapoint` class.

   For example, suppose we are writing the custom model class
   `MyCustomModel` and the custom data point class
   `MyCustomDatapoint` and the data file contains

   ```c++
   1 2
   1
   2
   3
   4
   5
   ```

   This would result in the model being instantiated as
   `MyCustomModel("1 2")` and the creation of five separate instances
   of the data point class: `MyCustomDatapoint("1")`,
   `MyCustomDatapoint("2")`, ... , `MyCustomDatapoint("5")`. Note that
   the inputs are strings.

   The `MyCustomDatapoint(const std::string &input_line)` and
   `MyCustomModel(const std::string &input_line)` constructors can
   then be defined by the user to specify how to initialize the
   objects using the given data file inputs. For example the model
   data input may specify the dimension of the model, and the
   constructor may use this information to pre-allocate enough memory
   to hold it.

   It is important to note that the user must manage the underlying
   data behind their custom model / datapoint classes. For the model,
   the underlying raw model data should be captured by a
   `std::vector<double>`.

## Defining the Datapoint Subclass

The following virtual methods of `Datapoint` are required to be overridden.

#### `Datapoint(const std::string &input_line, int order)`

The constructor for the subclass of Datapoint. The `order` argument
should be passed in to the superclass constructor call. For example:
`CustomDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) { ... }`

##### Args:

* <b>input_line</b> - The line of input from the data file.
* <b>order</b> - The order in which this data point appears in the shuffled permutation of data points.

---

#### `virtual std::vector<double> & GetWeights()`

Return a vector<double> of weights where the i'th weight in the returned vector corresponds to the i'th coordinate of GetCoordinates().

---

#### `virtual std::vector<int> & GetCoordinates()`

Return a vector<double> of coordinates where the i'th coordinate of the returned vector corresponds to the i'th weight of GetWeights().

---

## Defining the Model Subclass

The following virtual methods of `Model` are required to be overridden.

---

##### `Model(const std::string &input_line)`

The constructor for the subclass of Model.

###### Args:

* <b>input_line</b> - first line of the data file.

---

##### `virtual double ComputeLoss(const std::vector<Datapoint *> &datapoints)`

Compute loss value given a list of data points.

###### Args:

* <b>datapoints</b> - vector of pointers to the user defined data
  point structure. If the custom data point structure has methods not
  listed in the `Model` parent class you may need to cast the pointer
  to a pointer of the custom class.

---

##### `virtual int NumParameters()`

Return the number of coordinates of the model.

---

##### `virtual int CoordinateSize()`

Return the size of the coordinate vectors of the model. For scalar
coordinates, return 1.

---

##### `virtual std::vector<double> & ModelData()`

Return a reference to the underlying data. ModelData().size() should
be NumParameters() * CoordinateSize().

---

<b>For the following gradient related methods, we formulate the gradient at a datapoint x, model coordinate j as [∇f(x)]_j = λ_j * x_j − κ_j + h_bar_j(x).</b>

---

##### `virtual void PrecomputeCoefficients(Datapoint *datapoint, Gradient *g, std::vector<double> &local_model)`

Do any sort of precomputation (E.G: computing dot product) on a
datapoint before calling methods for computing lambda, kappa and
h_bar. Note that PrecomputeCoefficients is called by multiple threads.

###### Args:
* <b>datapoint</b> - Data point to precompute gradient information.
* <b>g</b> - Gradient object for storing any precomputed data. This is passed
  to the h_bar method afterwards. The relevant Gradient attribute is g->coeffs, a vector<double>
  to store arbitrary data. Note that g->coeffs is initially size 0, so in PrecomputeCoefficients the
  user needs to resize this vector according to their needs. Gradient objects are thread local
  objects that are re-used. Thus, g->coeffs may contain junk precompute info from a previous iteration.
* <b>cur_model</b> - a vector of doubles that contains the raw data of the
  model to precompute gradient information.

---

##### `virtual void H_bar(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model)`

Write to output h_bar_j of [∇f(x)]_j = λ_j * x_j − κ_j + h_bar_j(x). Note that this function is called by multiple threads.

###### Args:
* <b>coordinate</b> - The model coordinate j for which h_bar_j should be computed.
* <b>out</b> - Reference to vector<double> to which the value of h_bar should be written to.
* <b>g</b> - Gradient object which contains the precomputed data previously set by PrecomputeCoefficients.
  Further note that g->datapoint is a pointer to the data point whose gradient is being computed (which is the data point
  that was used by PrecomputeCoefficients to precompute gradient information).
* <b>local_model</b> - The raw data of the model for which lambda should be computed for.

---

##### `virtual void Lambda(int coordinate, double &out, std::vector<double> &local_model)`

Write to output the λ_j coefficient of the gradient equation [∇f(x)]_j
= λ_j * x_j − κ_j + h_bar_j(x). Note that this function is called by multiple
threads.

###### Args:
* <b>coordinate</b> - The model coordinate j for which λ_j should be computed.
* <b>out</b> - Reference to scalar double to which the value of lambda should be written to.
* <b>local_model</b> - The raw data of the model for which lambda should be computed for.

---

##### `virtual void Kappa(int coordinate, std::vector<double> &out, std::vector<double> &local_model)`

Write to output the κ_j coefficient of the gradient equation [∇f(x)]_j = λ_j * x_j − κ_j + h_bar_j(x). Note that Kappa is called by multiple
threads.

###### Args:
* <b>coordinate</b> - The model coordinate j for which κ_j should be computed.
* <b>out</b> - Reference to vector<double> to which the value of kappa should be written to.
* <b>local_model</b> - The raw data of the model for which lambda should be computed for.

---

## Example: Sparse Least Squares

Here we show how to define a custom model and datapoint class to solve
the classic least squares problem.

In the least squares problem we are interested in minimizing the
function `||Ax-b||^2`. A data point in this sense is a row of `A`
which we refer to as `a_i`, and the model we are optimizing is
`x`. Note that minimizing `||Ax-b||^2` is equivalent to minimizing
`sum (dot(a_i, x) - b)^2`.

For the purposes of this example, we will name our least squares model
`SimpleLSModel` and our least squares datapoint
`SimpleLSDatapoint`. The full source code is in the examples
directory.

### Starting off

To begin using Cyclades, we must include the "run.h" and "defines.h"
files in the src directory. Furthermore, to make use of gflags, we
must also initialize gflags.

```c++
#include "../src/run.h"
#include "../src/defines.h"

int main(int argc, char **argv) {
    // Initialize gflags
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}
```

After implementing `SimpleLSDatapoint` and `SimpleLSModel` we can call
```c++
Run<SimpleLSModel, SimpleLSDatapoint>();
```
to solve the least squares problem with the passed in command line flags.

### Data File Format / Data File Reading

To store the `A` matrix and `b` label vector, we will use the following format.
```c++
line 1 (input to model constructor) : the dimension of the x model vector
line 2...n (input to datapoint constructor) : m index_of_nnz_1 value_of_nnz_1 ... index_of_nnz_m value_of_nnz_m label
```

For the purposes of this example we will use the following data input:
```c++
10
1 0 1 1
1 1 1 2
1 2 1 3
1 3 1 4
1 4 1 5
1 5 1 6
1 6 1 7
1 7 1 8
1 8 1 9
1 9 1 10
```

This is equivalent to solving the problem with
```c++
A = 1 0 0 0 0 0 0 0 0 0       b = 1
    0 1 0 0 0 0 0 0 0 0           2
    0 0 1 0 0 0 0 0 0 0           3
    0 0 0 1 0 0 0 0 0 0           4
    0 0 0 0 1 0 0 0 0 0           5
    0 0 0 0 0 1 0 0 0 0           6
    0 0 0 0 0 0 1 0 0 0           7
    0 0 0 0 0 0 0 1 0 0           8
    0 0 0 0 0 0 0 0 1 0           9
    0 0 0 0 0 0 0 0 0 1           10
```

### Defining `SimpleLSDatapoint`

First we subclass `Datapoint`, and keep a `weights` vector, a
`coordinates` vector, and a `label` double.

```c++
class SimpleLSDatapoint : public Datapoint {
public:
    std::vector<double> weights;
    std::vector<int> coordinates;
    double label;
};
```

`weights[i]` will contain the value of the row of `A` at the index
specified `coordinates[i]`. This is a sparse representation of a row
of a matrix. `label` will be the corresponding `b_i` of the row
specified by the data point. The Cyclades framework does not require a
`label` variable, but it will be used by the model when computing gradients.

To read the datapoint values from a line of the input file, we
implement the constructor to read according to the format. Ignore the
`order` parameter.

```c++
SimpleLSDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
    // Create a string stream from the input line.
    // This lets us read from the line as if reading via cin.
    std::stringstream in(input_line);

    // We expect a sparse row from the A matrix, a_i, as well as the
    // corresponding value of b_i. The first value of the line is the number
    // of nnz values in the row of the matrix.
    // E.g:
    // n index_1 value_1 index_2 value_2 .... index_n value_n label
    int n;
    in >> n;
    weights.resize(n);
    coordinates.resize(n);
    for (int i = 0; i < n; i++) {
        in >> coordinates[i];
        in >> weights[i];
    }
    in >> label;
}
```

Finally, we fill in the required `GetWeights`, `GetCoordinates()` and
`GetNumCoordinateTouches()` methods.
```c++
std::vector<double> & GetWeights() override {
    return weights;
}

std::vector<int> & GetCoordinates() override {
    return coordinates;
}

int GetNumCoordinateTouches() override {
    return coordinates.size();
}
```

### Defining `SimpleLSModel`

First we subclass `Model`, and define `x`, the raw data containing the
model we are trying to optimize.

```c++
class SimpleLSModel : public Model {
public:
    std::vector<double> x;
};
```
Initialization with the data input line from the constructor is similar to the
process in `SimpleLSDatapoint`.

```c++
SimpleLSModel(const std::string &input_line) {
    // Create a string stream from the input line.
    // This lets us read from the line as if reading via cin.
    std::stringstream in(input_line);

    // We expect a single integer describing the number
    // of coordinates of the model.
    int num_coordinates;
    in >> num_coordinates;

    // Preallocate the x model and randomly initialize.
    x.resize(num_coordinates);
    for (int i = 0; i < x.size(); i++) {
        x[i] = ((double)rand()/(double)RAND_MAX);
    }
}
```

Here we additionally allocate enough memory to hold the model `x` and
furthermore randomly initialize its values.

It is convenient to define a dot product operation which computes the
dot product between a row specified by `SimpleLSDatapoint` and the
model.

```c++
double dot(SimpleLSDatapoint *a_i, std::vector<double> &x) {
    double product = 0;
    for (int i = 0; i < a_i->GetNumCoordinateTouches(); i++) {
        int index = a_i->GetCoordinates()[i];
        double value = a_i->GetWeights()[i];
        product += value * x[index];
    }
    return product;
}
```

Computing the loss is iterating through the data points and summing up
`(dot(a_i, x) - b)^2`.
```c++
// Minimize loss = sum (a_i * x - b)^2.
double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
    double loss = 0;

    // Note that ComputeLoss is called by a SINGLE thread.
    // It is possible to parallelize this via
    // #pragma omp parallel for
    for (int i = 0; i < datapoints.size(); i++) {
        SimpleLSDatapoint *a_i = (SimpleLSDatapoint *)datapoints[i];
        double b_i = a_i->label;
        double dot_product = dot(a_i, x);
        loss += (dot_product - b_i) * (dot_product - b_i);
    }

    // Let's also print out the model to see if we are on the right track.
    std::cout << "Model Parameters: " << std::endl;
    for (int i = 0; i < NumParameters(); i++) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    return loss / datapoints.size();
}
```
