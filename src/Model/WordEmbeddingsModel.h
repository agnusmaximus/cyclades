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
    std::vector<std::vector<double> > c_sum_mult1, c_sum_mult2;
    std::vector<double> c_thread_index_tracker;
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
	c_thread_index_tracker.resize(FLAGS_n_threads);
	// First calculate number of datapoints per thread.
	int max_datapoints = 0;
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    int n_datapoints_for_thread = 0;
	    for (int batch = 0; batch < partitions.NumBatches(); batch++) {
		n_datapoints_for_thread += partitions.NumDatapointsInBatch(thread, batch);
	    }
	    c_sum_mult1[thread].resize(n_datapoints_for_thread);
	    c_sum_mult2[thread].resize(n_datapoints_for_thread);
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
	    for (int index = 0; index < c_sum_mult1[thread].size(); index++) {
		C_A += c_sum_mult1[thread][index];
		C_B += c_sum_mult2[thread][index];
	    }
	}
	C = C_A / C_B;

	// Reset c sum index tracker.
	std::fill(c_thread_index_tracker.begin(), c_thread_index_tracker.end(), 0);
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
	    norm += (model[coord1*w2v_length+i] + model[coord2*w2v_length+i]) *
		(model[coord1*w2v_length+i] + model[coord2*w2v_length+i]);
	}
	g->coeffs[0] = 2 * weight * (log(weight) - norm - C);

	// Do some extra computation for C.
	int index = c_thread_index_tracker[omp_get_thread_num()]++;
	c_sum_mult1[omp_get_thread_num()][index] = weight * (log(weight) - norm);
	c_sum_mult2[omp_get_thread_num()][index] = weight;
    }

    virtual void Mu(int coordinate, double &out, std::vector<double> &local_model) override {
	out = 0;
    }

    virtual void Nu(int coordinate, std::vector<double> &out, std::vector<double> &local_model) override {
	std::fill(out.begin(), out.end(), 0);
    }

    virtual void H(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model) override {
	int c1 = g->datapoint->GetCoordinates()[0];
	int c2 = g->datapoint->GetCoordinates()[1];
	for (int i = 0; i < w2v_length; i++) {
	    out[i] = -1 * (2 * g->coeffs[0] * (model[c1*w2v_length+i] + model[c2*w2v_length+i]));
	}
    }
};


#endif
