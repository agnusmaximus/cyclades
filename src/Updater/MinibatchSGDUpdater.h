#ifndef _MINIBATCHSGD_UPDATER_
#define _MINIBATCHSGD_UPDATER_

#include "Updater.h"

DEFINE_int32(minibatch_batch_size, 100, "Minibatch batch size.");

template <class GRADIENT_CLASS>
class MinibatchSGDUpdater : public Updater<GRADIENT_CLASS> {
private:
    int n_threads;
    GRADIENT_CLASS *thread_gradients;

 public:
    MinibatchSGDUpdater(int n_threads) : Updater<GRADIENT_CLASS>() {
	thread_gradients = new GRADIENT_CLASS[n_threads];
	this->n_threads = n_threads;
    }

    ~MinibatchSGDUpdater() {
	delete [] thread_gradients;
    }

    // Main update method.
    virtual void Update(Model *model, Datapoint *datapoint, int thread_num) {
	std::cerr << "MinibatchSGDUpdater: the single datapoint update method should not be called." << std::endl;
	exit(0);
    }

    void UpdateMultiple(Model *model, DatapointPartitions &partitions, int meta_batch, int thread_num) {
	// Note: Meta batch is the current batch used potentially by cyclades / hogwild.
	int meta_batch_size = partitions.NumDatapointsInBatch(thread_num, meta_batch);

	// Clear sum of gradients.
	thread_gradients[thread_num].Clear();

	for (int i = 0; i < meta_batch_size; i += FLAGS_minibatch_batch_size) {
	    for (int index = 0; index < std::min(i+FLAGS_minibatch_batch_size, meta_batch_size); index++) {
		GRADIENT_CLASS cur_gradient;
		Datapoint *datapoint = partitions.GetDatapoint(thread_num, meta_batch, index);
		model->ComputeGradient(datapoint, &cur_gradient);
		thread_gradients[thread_num].Add(cur_gradient);
	    }
	    model->ApplyGradient(&thread_gradients[thread_num]);
	}
    }
};

#endif