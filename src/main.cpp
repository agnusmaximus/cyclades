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

template<class MODEL_CLASS, class DATAPOINT_CLASS, class CUSTOM_UPDATER=SparseSGDUpdater, class CUSTOM_TRAINER=CycladesTrainer>
TrainStatistics RunOnce() {
    // Initialize model and datapoints.
    Model *model;
    std::vector<Datapoint *> datapoints;
    DatasetReader::ReadDataset<MODEL_CLASS, DATAPOINT_CLASS>(FLAGS_data_file, datapoints, model);
    model->SetUp(datapoints);

    // Shuffle the datapoints and assign the order.
    if (FLAGS_shuffle_datapoints) {
      std::random_shuffle(datapoints.begin(), datapoints.end());
    }
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
    else {
	updater = new CUSTOM_UPDATER(model, datapoints);
    }

    // Create trainer depending on flag.
    Trainer *trainer = NULL;
    if (FLAGS_cache_efficient_hogwild_trainer) {
	trainer = new CacheEfficientHogwildTrainer();
    }
    else if (FLAGS_cyclades_trainer) {
	trainer = new CycladesTrainer();
    }
    else if (FLAGS_hogwild_trainer) {
	trainer = new HogwildTrainer();
    }
    else {
	trainer = new CUSTOM_TRAINER();
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

template<class MODEL_CLASS, class DATAPOINT_CLASS, class CUSTOM_UPDATER=SparseSGDUpdater, class CUSTOM_TRAINER=CycladesTrainer>
void Run() {
    TrainStatistics stats = RunOnce<MODEL_CLASS, DATAPOINT_CLASS, CUSTOM_UPDATER, CUSTOM_TRAINER>();
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_matrix_completion) {
	Run<MCModel, MCDatapoint>();
    }
    else if (FLAGS_word_embeddings) {
	Run<WordEmbeddingsModel, WordEmbeddingsDatapoint, WordEmbeddingsSGDUpdater>();
    }
    else if (FLAGS_matrix_inverse) {
	Run<MatrixInverseModel, MatrixInverseDatapoint>();
    }
    else if (FLAGS_least_squares) {
	Run<LSModel, LSDatapoint>();
    }
    else if (FLAGS_fast_matrix_completion) {
	Run<MCModel, MCDatapoint, FastMCSGDUpdater>();
    }
}
