#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SGDUpdater : public Updater {
protected:
    virtual void ComputeGradient(Model *model, Datapoint *datapoint, Gradient *g) {
	g->datapoint = datapoint;
	model->PrecomputeCoefficients(datapoint, g);
	int coord_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->Mu(index, g->mu[index]);
	    model->Nu(index, g->nu[index]);
	    model->H(index, g->h[index], g);
	}
    }

    double H(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return -FLAGS_learning_rate * g->h[coordinate][index_into_coordinate_vector];
    }

    double Nu(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return g->nu[coordinate][index_into_coordinate_vector] * FLAGS_learning_rate;
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
