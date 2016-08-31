#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SGDUpdater : public Updater {
protected:
    std::vector<double> nu, mu, h;

    virtual void ComputeGradient(Model *model, Datapoint *datapoint, Gradient *g, int thread) {
	g->datapoint = datapoint;
	model->PrecomputeCoefficients(datapoint, g, thread);
	int coord_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    mu[index] = model->Mu(index);
	    for (int j = 0; j < coord_size; j++) {
		nu[index*coord_size+j] = model->Nu(index, j);
		h[index*coord_size+j] = model->H(index, j, g);
	    }
	}
    }

    double H(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return -FLAGS_learning_rate * h[coordinate * model->CoordinateSize() + index_into_coordinate_vector];
    }

    double Nu(int coordinate, int index_into_coordinate_vector) {
	return nu[coordinate * model->CoordinateSize() + index_into_coordinate_vector] * FLAGS_learning_rate;
    }

    double Mu(int coordinate) {
	return mu[coordinate] * FLAGS_learning_rate;
    }

 public:
    SGDUpdater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) : Updater(model, datapoints, n_threads) {
	nu.resize(model->NumParameters() * model->CoordinateSize());
	mu.resize(model->NumParameters());
	h.resize(model->NumParameters() * model->CoordinateSize());
    }

    ~SGDUpdater() {
    }
};

#endif
