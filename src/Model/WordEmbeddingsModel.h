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

#ifndef _WORDEMBEDDINGSMODEL_
#define _WORDEMBEDDINGSMODEL_

#include <sstream>
#include "../DatapointPartitions/DatapointPartitions.h"
#include "Model.h"

DEFINE_int32(vec_length, 30, "Length of word embeddings vector in w2v.");

class WordEmbeddingsModel : public Model {
 private:
    std::vector<double> model;
    double C;
    std::vector<double > c_sum_mult1, c_sum_mult2;
    int n_words;
    int w2v_length;

    void InitializePrivateModel() {
	for (int i = 0; i < n_words; i++) {
	    for (int j = 0; j < w2v_length; j++) {
		model[i*w2v_length+j] = ((double)rand()/(double)RAND_MAX);
	    }
	}
    }

    void Initialize(const std::string &input_line) {
	// Expected input_line format: n_words.
	std::stringstream input(input_line);
	input >> n_words;
	w2v_length = FLAGS_vec_length;

	// Allocate memory.
	model.resize(n_words * w2v_length);
	C = 0;

	// Initialize private model.
	InitializePrivateModel();
    }

 public:
    WordEmbeddingsModel(const std::string &input_line) {
	Initialize(input_line);
    }

    ~WordEmbeddingsModel() {
    }

    void SetUpWithPartitions(DatapointPartitions &partitions) override {
	// Initialize C_sum_mult variables.
	c_sum_mult1.resize(FLAGS_n_threads);
	c_sum_mult2.resize(FLAGS_n_threads);
	// First calculate number of datapoints per thread.
	int max_datapoints = 0;
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    int n_datapoints_for_thread = 0;
	    for (int batch = 0; batch < partitions.NumBatches(); batch++) {
		n_datapoints_for_thread += partitions.NumDatapointsInBatch(thread, batch);
	    }
	    c_sum_mult1[thread] = 0;
	    c_sum_mult2[thread] = 0;
	}
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;
#pragma omp parallel for num_threads(FLAGS_n_threads) reduction(+:loss)
	for (int i = 0; i < datapoints.size(); i++) {
	    Datapoint *datapoint = datapoints[i];
	    const std::vector<double> &labels = datapoint->GetWeights();
	    const std::vector<int> &coordinates = datapoint->GetCoordinates();
	    double weight = labels[0];
	    int x = coordinates[0];
	    int y = coordinates[1];
	    double cross_product = 0;
	    for (int j = 0; j < w2v_length; j++) {
		cross_product += (model[x*w2v_length+j]+model[y*w2v_length+j]) *
		    (model[y*w2v_length+j]+model[y*w2v_length+j]);
	    }
	    loss += weight * (log(weight) - cross_product - C) * (log(weight) - cross_product - C);
	}
	return loss / datapoints.size();
    }

    void EpochFinish() {
	// Update C based on C_sum_mult.
	double C_A = 0, C_B = 0;
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    C_A += c_sum_mult1[thread];
	    C_B += c_sum_mult2[thread];
	    c_sum_mult1[thread] = 0;
	    c_sum_mult2[thread] = 0;
	}

	C = C_A / C_B;
    }

    int CoordinateSize() override {
	return w2v_length;
    }

    int NumParameters() override {
	return n_words;
    }

    std::vector<double> & ModelData() override {
	return model;
    }

    void PrecomputeCoefficients(Datapoint *datapoint, Gradient *g, std::vector<double> &local_model) override {
	if (g->coeffs.size() != 1) g->coeffs.resize(1);
	const std::vector<double> &labels = datapoint->GetWeights();
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	int coord1 = coordinates[0];
	int coord2 = coordinates[1];
	double weight = labels[0];
	double norm = 0;
	for (int i = 0; i < w2v_length; i++) {
	    norm += (local_model[coord1*w2v_length+i] + local_model[coord2*w2v_length+i]) *
		(local_model[coord1*w2v_length+i] + local_model[coord2*w2v_length+i]);
	}
	g->coeffs[0] = 2 * weight * (log(weight) - norm - C);

	// Do some extra computation for C.
	c_sum_mult1[omp_get_thread_num()] += weight * (log(weight) - norm);
	c_sum_mult2[omp_get_thread_num()] += weight;
    }

    virtual void Mu(int coordinate, double &out, std::vector<double> &local_model) override {
    }

    virtual void Nu(int coordinate, std::vector<double> &out, std::vector<double> &local_model) override {
    }

    virtual void H_bar(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model) override {
	int c1 = g->datapoint->GetCoordinates()[0];
	int c2 = g->datapoint->GetCoordinates()[1];
	for (int i = 0; i < w2v_length; i++) {
	    out[i] = (2 * g->coeffs[0] * (local_model[c1*w2v_length+i] + local_model[c2*w2v_length+i]));
	}
    }
};


#endif
