#ifndef _DEFINES_
#define _DEFINES_

#include <math.h>
#include <omp.h>
#include <cstdlib>
#include <thread>
#include <map>
#include <set>
#include <cstring>
#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include "Datapoint/Datapoint.h"
#include "Gradient/Gradient.h"
#include "DatasetReader.h"
#include "DatapointPartitions/DatapointPartitions.h"
#include "Partitioner/Partitioner.h"
#include "Partitioner/BasicPartitioner.h"
#include "Model/Model.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#include <sys/time.h>
int clock_gettime(int /*clk_id*/, struct timespec* t) {
    struct timeval now;
    int rv = gettimeofday(&now, NULL);
    if (rv) return rv;
    t->tv_sec  = now.tv_sec;
    t->tv_nsec = now.tv_usec * 1000;
    return 0;
}
#define CLOCK_MONOTONIC 0
#endif

class Timer {
public:
    struct timespec _start;
    struct timespec _end;
    Timer(){
        clock_gettime(CLOCK_MONOTONIC, &_start);
    }
    virtual ~Timer(){}
    inline void Restart(){
        clock_gettime(CLOCK_MONOTONIC, &_start);
    }
    inline float Elapsed(){
        clock_gettime(CLOCK_MONOTONIC, &_end);
        return (_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec) / 1000000000.0;
    }
};

void pin_to_core(size_t core) {
#ifdef _GNU_SOURCE
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

DEFINE_string(data_file, "blank", "Input data file.");
DEFINE_int32(n_epochs, 100, "Number of passes of data in training.");
DEFINE_int32(n_threads, 2, "Number of threads in parallel during training.");
DEFINE_double(learning_rate, .001, "Learning rate.");
DEFINE_bool(print_loss_per_epoch, false, "Should compute and print loss every epoch.");
DEFINE_bool(print_partition_time, false, "Should print time taken to distribute datapoints across threads.");

// Flags for training types.
DEFINE_bool(cache_efficient_hogwild_trainer, false, "Hogwild training method with cache friendly datapoint ordering (parallel).");
DEFINE_bool(cyclades_trainer, false, "Cyclades training method (parallel).");
DEFINE_bool(hogwild_trainer, false, "Hogwild training method (parallel).");

// Flags for updating types.
DEFINE_bool(sgd, false, "Use the SGD update method.");
DEFINE_bool(svrg, false, "Use the SVRG update method.");

// Flags for application types.
DEFINE_bool(matrix_completion, false, "Matrix completion application type.");
DEFINE_bool(dense_least_squares, false, "Dense least squares application type.");
DEFINE_bool(word_embeddings, false, "W2V application type.");
DEFINE_bool(matrix_inverse, false, "Matrix inverse application type.");
DEFINE_bool(least_squares, false, "Sparse least squares application type.");
DEFINE_bool(ising_gibbs, false, "Ising gibbs model computation.");

// Flag for tuning learning rate.
DEFINE_bool(tune_learning_rate, false, "Tune learning rate for given problem.");
DEFINE_double(tune_lr_lower_bound, 1e-20, "Lower bound for tuning learning rate.");
DEFINE_double(tune_lr_upper_bound, 10, "Upper bound for tuning learning rate.");
DEFINE_double(tune_epsilon, 1e-10, "Stops tuning learning rate when lower_bound and upper_bound are within this value.");
DEFINE_double(tune_stepfactor, 10, "Factor of tuning stepsize.");

// MISC flags.
DEFINE_int32(random_range, 100, "Range of random numbers for initializing the model.");

#include "Updater/Updater.h"
#include "Updater/SGDUpdater.h"
#include "Updater/SVRGUpdater.h"

#include "Partitioner/CycladesPartitioner.h"
#include "Partitioner/DFSCachePartitioner.h"
//#include "Partitioner/GraphCutsCachePartitioner.h"
#include "Trainer/Trainer.h"
#include "Trainer/CycladesTrainer.h"
#include "Trainer/HogwildTrainer.h"
#include "Trainer/CacheEfficientHogwildTrainer.h"

#include "Datapoint/MCDatapoint.h"
#include "Model/MCModel.h"

#include "Datapoint/WordEmbeddingsDatapoint.h"
#include "Model/WordEmbeddingsModel.h"

#include "Datapoint/MatrixInverseDatapoint.h"
#include "Model/MatrixInverseModel.h"

#include "Datapoint/LSDatapoint.h"
#include "Model/LSModel.h"

/*#include "Datapoint/GibbsDatapoint.h"
#include "Model/IsingGibbsModel.h"*/

#endif
