#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SGDUpdater : public Updater {
protected:
    virtual void ComputeGradient(Model *model, Datapoint *datapoint, Gradient *g, int thread) {
	g->datapoint = datapoint;
	model->PrecomputeCoefficients(datapoint, g, thread);
	int coord_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    g->mu[index] = model->Mu(index);
	    for (int j = 0; j < coord_size; j++) {
		g->nu[index*coord_size+j] = model->Nu(index, j);
		g->h[index*coord_size+j] = model->H(index, j, g);
	    }
	}
    }

    double H(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return -FLAGS_learning_rate * g->h[coordinate * model->CoordinateSize() + index_into_coordinate_vector];
    }

    double Nu(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return g->nu[coordinate * model->CoordinateSize() + index_into_coordinate_vector] * FLAGS_learning_rate;
    }

    double Mu(int coordinate, Gradient *g) {
	return g->mu[coordinate] * FLAGS_learning_rate;
    }

 public:
    SGDUpdater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) : Updater(model, datapoints, n_threads) {}

    ~SGDUpdater() {
    }
};

#endif
