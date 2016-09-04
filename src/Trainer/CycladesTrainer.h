#ifndef _CYCLADES_TRAINER_
#define _CYCLADES_TRAINER_

class CycladesTrainer : public Trainer {
private:
    int *thread_batch;

    // A parallel method where each thread waits
    // until all other threads are on the same batch.
    void WaitForThreadsTilBatch(int thread, int batch) {
	thread_batch[thread] = batch;
	bool waiting_for_threads = true;
	while (waiting_for_threads) {
	    waiting_for_threads = false;
	    for (int t = 0; t < FLAGS_n_threads; t++) {
		if (thread_batch[t] < batch) {
		    waiting_for_threads = true;
		    break;
		}
	    }
	}
    }

    void ClearThreadBatchIndices() {
	memset(thread_batch, 0, sizeof(int) * FLAGS_n_threads);
    }

    void DebugPrintPartitions(DatapointPartitions &p) {
	for (int i = 0; i < p.NumBatches(); i++) {
	    std::cout << "Batch " << i << std::endl;
	    for (int j = 0; j < FLAGS_n_threads; j++) {
		std::cout << "Thread " << j << ": ";
		for (int k = 0; k < p.NumDatapointsInBatch(j, i); k++) {
		    if (k != 0) std::cout << " ";
		    std::cout << p.GetDatapoint(j, i, k)->GetOrder();
		}
		std::cout << std::endl;
	    }
	}
    }

public:
    CycladesTrainer() {
	thread_batch = new int[FLAGS_n_threads];
	memset(thread_batch, 0, sizeof(int) * FLAGS_n_threads);
    }
    ~CycladesTrainer() {
	delete [] thread_batch;
    }

    TrainStatistics Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater *updater) override {
	// Partitions.
	CycladesPartitioner partitioner(model);
	Timer partition_timer;
	DatapointPartitions partitions = partitioner.Partition(datapoints, FLAGS_n_threads);
	if (FLAGS_print_partition_time) {
	    this->PrintPartitionTime(partition_timer);
	}

	model->SetUpWithPartitions(partitions);

	// Default batch ordering.
	std::vector<int> batch_ordering(partitions.NumBatches());
	for (int i = 0; i < partitions.NumBatches(); i++) {
	    batch_ordering[i] = i;
	}

	// Default datapoint processing ordering.
	// [thread][batch][index].
	std::vector<std::vector<std::vector<int> > > per_batch_datapoint_order(FLAGS_n_threads);
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    per_batch_datapoint_order[thread].resize(partitions.NumBatches());
	    for (int batch = 0; batch < partitions.NumBatches(); batch++) {
		per_batch_datapoint_order[thread][batch].resize(partitions.NumDatapointsInBatch(thread, batch));
		for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
		    per_batch_datapoint_order[thread][batch][index] = index;
		}
	    }
	}

	// Keep track of statistics of training.
	TrainStatistics stats;

	// Train.
	Timer gradient_timer;
	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {

	    this->EpochBegin(epoch, gradient_timer, model, datapoints, &stats);

	    // Random batch ordering generation.
	    if (FLAGS_random_batch_processing) {
		for (int i = 0; i < partitions.NumBatches(); i++) {
		    batch_ordering[i] = rand() % partitions.NumBatches();
		}
	    }

	    // Random per batch datapoint processing.
	    if (FLAGS_random_per_batch_datapoint_processing) {
		for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		    for (int batch = 0; batch < partitions.NumBatches(); batch++) {
			for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
			    per_batch_datapoint_order[thread][batch][index] = rand() % partitions.NumDatapointsInBatch(thread, batch);
			}
		    }
		}
	    }

	    updater->EpochBegin();

#pragma omp parallel for schedule(static, 1)
	    for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		for (int batch_count = 0; batch_count < partitions.NumBatches(); batch_count++) {
		    int batch = batch_ordering[batch_count];
		    WaitForThreadsTilBatch(thread, batch_count);
		    for (int index_count = 0; index_count < partitions.NumDatapointsInBatch(thread, batch); index_count++) {
			int index = per_batch_datapoint_order[thread][batch][index_count];
			updater->Update(model, partitions.GetDatapoint(thread, batch, index));
		    }
		}
	    }
	    updater->EpochFinish();
	    ClearThreadBatchIndices();
	}
	return stats;
    }
};

#endif
