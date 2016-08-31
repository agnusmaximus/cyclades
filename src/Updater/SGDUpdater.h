#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SGDUpdater : public Updater {
protected:

    void ComputeGradient(Model *model, Datapoint *datapoint, Gradient *g) {
	model->PrecomputeCoefficients(datapoint, g);
    }

    double H(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return -FLAGS_learning_rate * model->H(coordinate, index_into_coordinate_vector, g);
    }

    double Nu(int coordinate, int index_into_coordinate_vector) {
	return model->Nu(coordinate, index_into_coordinate_vector) * FLAGS_learning_rate;
    }

    double Mu(int coordinate) {
	return model->Mu(coordinate) * FLAGS_learning_rate;
    }

 public:
   SGDUpdater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) : Updater(model, datapoints, n_threads) {}

    ~SGDUpdater() {
    }
};

#endif
