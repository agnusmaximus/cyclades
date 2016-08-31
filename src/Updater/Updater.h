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


    virtual double H(int coordinate, int index_into_coordinate_vector, Gradient *g) = 0;
    virtual double Nu(int coordinate, int index_into_coordinate_vector, Gradient *g) = 0;
    virtual double Mu(int coordinate, Gradient *g) = 0;

    virtual void ComputeGradient(Model *model, Datapoint *datapoint, Gradient *g, bool preprocess=true) = 0;

    virtual void ApplyGradient(Model *model, Datapoint *datapoint, Gradient *g) {
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

    virtual void CatchUp(Model *model, Datapoint *datapoint, Gradient *g, int order, std::vector<int> &bookkeeping) {
	if (!model->NeedsCatchup()) return;
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
	ComputeGradient(model, datapoint, &thread_gradients[thread_num]);
        CatchUp(model, datapoint, &thread_gradients[thread_num], datapoint->GetOrder(), bookkeeping);
	ApplyGradient(model, datapoint, &thread_gradients[thread_num]);
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
	for (const auto &datapoint : datapoints) {
	    ComputeGradient(model, datapoint, &thread_gradients[0], false);
	    CatchUp(model, datapoint, &thread_gradients[0], model->NumParameters()+1, bookkeeping);
	    for (const auto &coordinate : datapoint->GetCoordinates()) {
		bookkeeping[coordinate] = model->NumParameters()+1;
	    }
	}

	std::fill(bookkeeping.begin(), bookkeeping.end(), 0);
	model->EpochFinish();
    }
};

#endif
