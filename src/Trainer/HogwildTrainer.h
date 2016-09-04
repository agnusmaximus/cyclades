#ifndef _HOGWILD_TRAINER_
#define _HOGWILD_TRAINER_

class HogwildTrainer : public Trainer {
public:
    HogwildTrainer() {}
    ~HogwildTrainer() {}

    TrainStatistics Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater *updater) override {
	// Partition.
	BasicPartitioner partitioner;
	Timer partition_timer;
	DatapointPartitions partitions = partitioner.Partition(datapoints, FLAGS_n_threads);
	if (FLAGS_print_partition_time) {
	    this->PrintPartitionTime(partition_timer);
	}

	model->SetUpWithPartitions(partitions);

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

	    // Random per batch datapoint processing.
	    if (FLAGS_random_per_batch_datapoint_processing) {
		for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		    int batch = 0;
		    for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
			per_batch_datapoint_order[thread][batch][index] = rand() % partitions.NumDatapointsInBatch(thread, batch);
		    }
		}
	    }

	    updater->EpochBegin();

#pragma omp parallel for schedule(static, 1)
	    for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		int batch = 0; // Hogwild only has 1 batch.
		for (int index_count = 0; index_count < partitions.NumDatapointsInBatch(thread, batch); index_count++) {
		    int index = per_batch_datapoint_order[thread][batch][index_count];
		    updater->Update(model, partitions.GetDatapoint(thread, batch, index));
		}
	    }
	    updater->EpochFinish();
	}
	return stats;
    }
};

#endif
