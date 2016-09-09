
/*
* Copyright 2016 [See AUTHORS file for list of authors]
*
*    Licensed under the Apache License, Version 2.0 (the "License");
*    you may not use this file except in compliance with the License.
*    You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS,
*    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*    See the License for the specific language governing permissions and
*    limitations under the License.
*/

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
