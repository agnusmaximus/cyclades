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

// Sample call: ./cyclades -matrix_inverse -n_threads=2  -cyclades_trainer  -cyclades_batch_size=500  -learning_rate=.000001 --print_partition_time -n_epochs=20 -sgd -print_loss_per_epoch --data_file="data/nh2010/nh2010.data"

#include <iostream>
#include "defines.h"
#include <iomanip>

template<class MODEL_CLASS, class DATAPOINT_CLASS>
TrainStatistics RunOnce() {
    // Initialize model and datapoints.
    Model *model;
    std::vector<Datapoint *> datapoints;
    DatasetReader::ReadDataset<MODEL_CLASS, DATAPOINT_CLASS>(FLAGS_data_file, datapoints, model);
    model->SetUp(datapoints);

    // Shuffle the datapoints and assign the order.
    std::random_shuffle(datapoints.begin(), datapoints.end());
    for (int i = 0; i < datapoints.size(); i++) {
	datapoints[i]->SetOrder(i+1);
    }

    // Create updater.
    Updater *updater = NULL;
    if (FLAGS_dense_linear_sgd) {
	updater = new DenseLinearSGDUpdater(model, datapoints);
    }
    else if (FLAGS_sparse_sgd) {
	updater = new SparseSGDUpdater(model, datapoints);
    }
    else if (FLAGS_svrg) {
	updater = new SVRGUpdater(model, datapoints);
    }
    else if (FLAGS_saga) {
	updater = new SAGAUpdater(model, datapoints);
    }
    else if (FLAGS_custom) {
	updater = new CustomUpdater(model, datapoints);
    }
    if (!updater) {
	std::cerr << "Main: updater class not chosen." << std::endl;
	exit(0);
    }

    // Create trainer depending on flag.
    Trainer *trainer = NULL;
    if (FLAGS_cache_efficient_hogwild_trainer) {
	trainer = new CacheEfficientHogwildTrainer();
    }
    if (FLAGS_cyclades_trainer) {
	trainer = new CycladesTrainer();
    }
    else if (FLAGS_hogwild_trainer) {
	trainer = new HogwildTrainer();
    }
    if (!trainer) {
	std::cerr << "Main: training method not chosen." << std::endl;
	exit(0);
    }

    TrainStatistics stats = trainer->Train(model, datapoints, updater);

    // Delete trainer.
    delete trainer;

    // Delete model and datapoints.
    delete model;
    for_each(datapoints.begin(), datapoints.end(), std::default_delete<Datapoint>());

    // Delete updater.
    delete updater;

    return stats;
}

// Method to tune the learning rate.
template<class MODEL_CLASS, class DATAPOINT_CLASS>
void TuneLearningRate() {

    double best_stepsize = -1;
    double best_score = DBL_MAX;

    for (double cur_stepsize = FLAGS_tune_lr_upper_bound; cur_stepsize >= FLAGS_tune_lr_lower_bound; cur_stepsize /= FLAGS_tune_stepfactor) {
	FLAGS_learning_rate = cur_stepsize;
	TrainStatistics cur_stats = RunOnce<MODEL_CLASS, DATAPOINT_CLASS>();
	std::cout << "Trainer: (learning_rate: " << cur_stepsize << ") Loss from " << cur_stats.losses[0] << " -> " << cur_stats.losses[cur_stats.losses.size()-1] << std::endl;
	if (cur_stats.losses[cur_stats.losses.size()-1] < best_score) {
	    best_score = cur_stats.losses[cur_stats.losses.size()-1];
	    best_stepsize = cur_stepsize;
	}
    }
    double increment = (best_stepsize * FLAGS_tune_stepfactor - best_stepsize / FLAGS_tune_stepfactor) / FLAGS_tune_stepfactor;
    for (double cur_learning_rate = best_stepsize / FLAGS_tune_stepfactor; cur_learning_rate < best_stepsize * FLAGS_tune_stepfactor; cur_learning_rate += increment) {
	FLAGS_learning_rate = cur_learning_rate;
	TrainStatistics cur_stats = RunOnce<MODEL_CLASS, DATAPOINT_CLASS>();
	std::cout << "Trainer: (learning_rate: " << cur_learning_rate << ") Loss from " << cur_stats.losses[0] << " -> " << cur_stats.losses[cur_stats.losses.size()-1] << std::endl;
	if (cur_stats.losses[cur_stats.losses.size()-1] < best_score) {
	    best_score = cur_stats.losses[cur_stats.losses.size()-1];
	    best_stepsize = cur_learning_rate;
	}
    }
    std::cout << "Best stepsize: " << best_stepsize << " Lowest loss: " << best_score << std::endl;
}

template<class MODEL_CLASS, class DATAPOINT_CLASS>
void Run() {
    if (!FLAGS_tune_learning_rate) {
	TrainStatistics stats = RunOnce<MODEL_CLASS, DATAPOINT_CLASS>();
    }
    else {
	// Tune the learning rate.
	TuneLearningRate<MODEL_CLASS, DATAPOINT_CLASS>();
    }
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_matrix_completion) {
	Run<MCModel, MCDatapoint>();
    }
    else if (FLAGS_word_embeddings) {
	Run<WordEmbeddingsModel, WordEmbeddingsDatapoint>();
    }
    else if (FLAGS_matrix_inverse) {
	Run<MatrixInverseModel, MatrixInverseDatapoint>();
    }
    else if (FLAGS_least_squares) {
	Run<LSModel, LSDatapoint>();
    }
    /*else if (FLAGS_ising_gibbs) {
	Run<IsingGibbsModel, GibbsDatapoint>();
	}*/
}
