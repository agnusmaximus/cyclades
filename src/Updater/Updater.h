#ifndef _UPDATER_
#define _UPDATER_

#include "../DatapointPartitions/DatapointPartitions.h"
#include "../Gradient/Gradient.h"

class Updater {
protected:
    Model *model;
    Gradient *thread_gradients;
    std::vector<Datapoint *> datapoints;
    int n_threads;
    std::vector<int> bookkeeping;
    // [thread][name][2d_vector].
    std::vector<std::map<std::string, std::vector<std::vector<double> > > > thread_local_2d_vectors;
    std::vector<std::map<std::string, std::vector<double> > > thread_local_1d_vectors;

    virtual double H(int coordinate, int index_into_coordinate_vector, Gradient *g) = 0;
    virtual double Nu(int coordinate, int index_into_coordinate_vector, Gradient *g) = 0;
    virtual double Mu(int coordinate, Gradient *g) = 0;
    // Expect that calls to Nu, Mu, and H are ready for all coordinates touched by datapoint.
    virtual void ComputeGradient(Datapoint *datapoint, Gradient *g) = 0;
    // Expect that calls to Nu and Mu are ready for all coordinates.
    virtual void ComputeAllNuAndMu(Gradient *g) = 0;

    virtual void ApplyGradient(Datapoint *datapoint, Gradient *g) {
	std::vector<double> &model_data = model->ModelData();
	int coordinate_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    double mu = Mu(index, g);
	    for (int j = 0; j < coordinate_size; j++) {
		model_data[index * coordinate_size + j] = (1 - mu) * model_data[index * coordinate_size + j]
		    - Nu(index, j, g)
		    + H(index, j, g);
	    }
	}
    }

    void CatchUp(Datapoint *datapoint, Gradient *g, int order, std::vector<int> &bookkeeping) {
	// Optimize by quick returning if nu and mu are zero.
	std::vector<double> &model_data = model->ModelData();
	int coordinate_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    int diff = order - bookkeeping[index] - 1;
	    if (diff < 0) diff = 0;
	    double geom_sum = 0;
	    double mu = Mu(index, g);
	    if (mu != 0) {
		geom_sum = ((1 - pow(1 - mu, diff+1)) / (1 - (1 - mu))) - 1;
	    }
	    for (int j = 0; j < coordinate_size; j++) {
		model_data[index * coordinate_size + j] =
		    pow(1 - mu, diff) * model_data[index * coordinate_size + j]
		    - Nu(index, j, g) * geom_sum;
	    }
	}
    }

    void FinalCatchUp(std::vector<int> &bookkeeping) {
	int coordinate_size = model->CoordinateSize();
	std::vector<double> &model_data = model->ModelData();
#pragma omp parallel num_threads(FLAGS_n_threads)
	{
	    Gradient *g = &thread_gradients[omp_get_thread_num()];
	    ComputeAllNuAndMu(g);
	    std::vector<double> nu(coordinate_size);
#pragma omp for
	    for (int i = 0; i < model->NumParameters(); i++) {
		int diff = model->NumParameters() - bookkeeping[i];
		double geom_sum = 0, mu = Mu(i, g);
		if (mu != 0) {
		    geom_sum = ((1 - pow(1 - mu, diff+1)) / (1 - (1 - mu))) - 1;
		}
		for (int j = 0; j < coordinate_size; j++) {
		    model_data[i * coordinate_size + j] =
			pow(1 - mu, diff) * model_data[i * coordinate_size + j]
			- Nu(i, j, g) * geom_sum;
		}
	    }
	}
    }

    void RegisterThreadLocal2dVector(std::string name, int n_rows, int n_columns) {
	for (int i = 0; i <FLAGS_n_threads; i++) {
	    thread_local_2d_vectors[i][name].resize(n_rows, std::vector<double>(n_columns, 0));
	}
    }

    void RegisterThreadLocal1dVector(std::string name, int n_columns) {
	for (int i = 0; i <FLAGS_n_threads; i++) {
	    thread_local_1d_vectors[i][name].resize(n_columns, 0);
	}
    }

    std::vector<std::vector<double> > & GetThreadLocal2dVector(std::string name) {
	return thread_local_2d_vectors[omp_get_thread_num()][name];
    }

    std::vector<double> & GetThreadLocal1dVector(std::string name) {
	return thread_local_1d_vectors[omp_get_thread_num()][name];
    }


public:
    Updater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) {
	// Create gradients for each thread.
	thread_gradients = new Gradient[n_threads];
	this->n_threads = n_threads;
	for (int thread = 0; thread < n_threads; thread++) {
	    thread_gradients[thread] = Gradient();
	    thread_gradients[thread].SetUp(model);
	}
	this->model = model;

	// Create thread local vectors for each thread.
	thread_local_2d_vectors.resize(n_threads);
	thread_local_1d_vectors.resize(n_threads);

	// Set up bookkeping.
	this->datapoints = datapoints;
	for (int i = 0; i < model->NumParameters(); i++) {
	    bookkeeping.push_back(0);
	}
    }
    Updater() {}
    virtual ~Updater() {
	delete [] thread_gradients;
    }

    // Main update method.
    virtual void Update(Model *model, Datapoint *datapoint, int thread_num) {
	thread_gradients[thread_num].Clear();
	ComputeGradient(datapoint, &thread_gradients[thread_num]);
        CatchUp(datapoint, &thread_gradients[thread_num], datapoint->GetOrder(), bookkeeping);
	ApplyGradient(datapoint, &thread_gradients[thread_num]);
	for (const auto &coordinate : datapoint->GetCoordinates()) {
	    bookkeeping[coordinate] = datapoint->GetOrder();
	}
    }

    // Called before epoch begins.
    virtual void EpochBegin() {
	model->EpochBegin();
    }

    // Called when the epoch ends.
    virtual void EpochFinish() {
	FinalCatchUp(bookkeeping);
	std::fill(bookkeeping.begin(), bookkeeping.end(), 0);
	model->EpochFinish();
    }
};

#endif
