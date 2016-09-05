#ifndef _CACHE_EFFICIENT_HOGWILD_TRAINER_
#define _CACHE_EFFICIENT_HOGWILD_TRAINER_

class CacheEfficientHogwildTrainer : public Trainer {
public:
    CacheEfficientHogwildTrainer() {}
    ~CacheEfficientHogwildTrainer() {}

    TrainStatistics Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater *updater) override {
	// Partition.
	DFSCachePartitioner partitioner;
	Timer partition_timer;
	DatapointPartitions partitions = partitioner.Partition(datapoints, FLAGS_n_threads);
	if (FLAGS_print_partition_time) {
	    this->PrintPartitionTime(partition_timer);
	}

	model->SetUpWithPartitions(partitions);
	updater->SetUpWithPartitions(partitions);

	TrainStatistics stats;

	// Train.
	Timer gradient_timer;
	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {
	    this->EpochBegin(epoch, gradient_timer, model, datapoints, &stats);

	    updater->EpochBegin();

#pragma omp parallel for schedule(static, 1)
	    for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		for (int batch = 0; batch < partitions.NumBatches(); batch++) {
		    for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
			updater->Update(model, partitions.GetDatapoint(thread, batch, index));
		    }
		}
	    }
	    updater->EpochFinish();
	}
	return stats;
    }
};

#endif
