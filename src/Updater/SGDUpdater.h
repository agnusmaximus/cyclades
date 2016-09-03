#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SGDUpdater : public Updater {
protected:
    virtual void ComputeAllNuAndMu(Gradient *g) {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &nu = g->Get2dVector("nu");
	std::vector<double> &mu = g->Get1dVector("mu");
	for (int i = 0; i < model->NumParameters(); i++) {
	    model->Mu(i, mu[i], cur_model);
	    model->Nu(i, nu[i], cur_model);
	}
    }

    virtual void ComputeGradient(Datapoint *datapoint, Gradient *g) {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &nu = g->Get2dVector("nu");
	std::vector<std::vector<double> > &h = g->Get2dVector("h");
	std::vector<double> &mu = g->Get1dVector("mu");

	g->datapoint = datapoint;
	model->PrecomputeCoefficients(datapoint, g, cur_model);
	int coord_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->Mu(index, mu[index], cur_model);
	    model->Nu(index, nu[index], cur_model);
	    model->H(index, h[index], g, cur_model);
	}
    }

    double H(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return -FLAGS_learning_rate * g->Get2dVector("h")[coordinate][index_into_coordinate_vector];
    }

    double Nu(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return g->Get2dVector("nu")[coordinate][index_into_coordinate_vector] * FLAGS_learning_rate;
    }

    double Mu(int coordinate, Gradient *g) {
	return g->Get1dVector("mu")[coordinate] * FLAGS_learning_rate;
    }

 public:
    SGDUpdater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) : Updater(model, datapoints, n_threads) {
	Register1dVector("mu", model->NumParameters());
	Register2dVector("nu", model->NumParameters(), model->CoordinateSize());
	Register2dVector("h", model->NumParameters(), model->CoordinateSize());
    }

    ~SGDUpdater() {
    }
};

#endif
