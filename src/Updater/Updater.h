#ifndef _UPDATER_
#define _UPDATER_

#include "../DatapointPartitions/DatapointPartitions.h"
#include "../Gradient/LinearGradient.h"

class Updater {
private:
    Model *model;
    LinearGradient *thread_gradients;
    std::vector<Datapoint *> datapoints;
    int n_threads;
    std::vector<int> bookkeeping;

    virtual void ComputeGradient(Model *model, Datapoint *datapoint, LinearGradient *g) = 0;

    virtual void ApplyGradient(Model *model, Datapoint *datapoint, LinearGradient *g) {
	std::vector<double> &model_data = model->ModelData();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    for (int j = 0; j < model->CoordinateSize(); j++) {
		model_data[index * model->CoordinateSize() + j] = (1 - g->mu) * model_data[index * model->CoordinateSize() + j]
		    - g->nu[j]
		    + g->h * model_data[index * model->CoordinateSize() + j];
	    }
	}
    }

    virtual void CatchUp(Model *model, Datapoint *datapoint, LinearGradient *g, int order, std::vector<int> &bookkeeping) {
	std::vector<double> &model_data = model->ModelData();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    int diff = order - bookkeeping[index] - 1;
	    double geom_sum = 0;
	    if (g->mu != 0) {
		geom_sum = ((1 - pow(g->mu, diff+1)) / (1 - g->mu)) - 1;
	    }
	    for (int j = 0; j < model->CoordinateSize(); j++) {
		model_data[index * model->CoordinateSize() + j] =
		    pow(1 - g->mu, diff) * model_data[index * model->CoordinateSize() + j]
		    - g->nu[j] * geom_sum;
	    }
	}
    }

public:
    Updater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) {
	// Create gradients for each thread.
	thread_gradients = new LinearGradient[n_threads];
	this->n_threads = n_threads;
	for (int thread = 0; thread < n_threads; thread++) {
	    thread_gradients[thread] = LinearGradient();
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
	model->EpochFinish();
	for (const auto &datapoint : datapoints) {
	    LinearGradient g;
	    g.SetUp(model);
	    model->Nu(datapoint, g.nu);
	    model->Mu(datapoint, g.mu);
	    CatchUp(model, datapoint, &g, model->NumParameters()+1, bookkeeping);
	    for (const auto &coordinate : datapoint->GetCoordinates()) {
		bookkeeping[coordinate] = model->NumParameters()+1;
	    }
	}
	std::fill(bookkeeping.begin(), bookkeeping.end(), 0);
    }
};

#endif
