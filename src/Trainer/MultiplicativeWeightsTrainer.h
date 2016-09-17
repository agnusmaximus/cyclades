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
#ifndef _MULTIPLICATIVE_WEIGHTS_TRAINER_
#define _MULTIPLICATIVE_WEIGHTS_TRAINER_

DEFINE_double(epsilon, .1, "A penalty constant for multiplicative weights updates.");
DEFINE_double(frequency, .1, "How frequently to judge the effectiveness of a datapoint's gradient?");

class MultiplicativeWeightsTrainer : public Trainer {
protected:
    std::vector<double> p_distribution;

    int ChooseDatapointIndex(std::vector<double> &p_distribution) {
	double random_number = rand() / (double)RAND_MAX;
	double t = 0;
	for (int i = 0; i < p_distribution.size(); i++) {
	    if (t >= random_number) return i;
	    t += p_distribution[i];
	}
	return p_distribution.size()-1;
    }

    void NormalizeDistribution(std::vector<double> &p_distribution) {
	double sum = 0;
	for (int i = 0; i < p_distribution.size(); i++) {
	    sum += p_distribution[i];
	}
	for (int i = 0; i < p_distribution.size(); i++) {
	    p_distribution[i] /= sum;
	}
    }

    void UpdateDatapoint(Model *model, int datapoint_index, Datapoint *datapoint, const std::vector<Datapoint *> &datapoints, Updater *updater, int epoch) {
	bool should_check = rand() % 100 < FLAGS_frequency * 100;
	double start_loss, end_loss;
	if (should_check) {
	    start_loss = model->ComputeLoss(datapoints);
	}
	updater->Update(model, datapoint);
	if (should_check) {
	    end_loss = model->ComputeLoss(datapoints);
	    double cost = (end_loss - start_loss);
	    if (cost > 0) {
		p_distribution[datapoint_index] *= pow(1-FLAGS_epsilon, cost);
		NormalizeDistribution(p_distribution);
	    }
	}
    }

public:
    MultiplicativeWeightsTrainer() {}
    ~MultiplicativeWeightsTrainer() {}

    TrainStatistics Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater *updater) override {
	// Make sure there is only 1 thread.
	if (FLAGS_n_threads != 1) {
	    std::cout << "MultiplicativeWeightsTrainer: Only single threaded allowed." << std::endl;
	    exit(0);
	}

	// Initialize probability distribution of each datapoint.
	p_distribution.resize(datapoints.size());
	std::fill(p_distribution.begin(), p_distribution.end(), 1/(double)datapoints.size());

	// Partition.
	BasicPartitioner partitioner;
	Timer partition_timer;
	DatapointPartitions partitions = partitioner.Partition(datapoints, FLAGS_n_threads);
	if (FLAGS_print_partition_time) {
	    this->PrintPartitionTime(partition_timer);
	}

	model->SetUpWithPartitions(partitions);
	updater->SetUpWithPartitions(partitions);

	// Keep track of statistics of training.
	TrainStatistics stats;

	// Train.
	Timer gradient_timer;
	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {

	    this->EpochBegin(epoch, gradient_timer, model, datapoints, &stats);
	    updater->EpochBegin();

	    for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		int batch = 0; // MultiplicativeWeights only has 1 batch.
		for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
		    int chosen_datapoint_index = ChooseDatapointIndex(p_distribution);
		    UpdateDatapoint(model, chosen_datapoint_index, partitions.GetDatapoint(thread, batch, chosen_datapoint_index), datapoints, updater, epoch);
		}
	    }
	    updater->EpochFinish();
	}
	return stats;
    }
};

#endif
