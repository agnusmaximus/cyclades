#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SGDUpdater : public Updater {
protected:
    void PrepareNu(std::vector<int> &coordinates) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &nu = GetThreadLocal2dVector("nu");
	for (int i = 0; i < coordinates.size(); i++) {
	    int index = coordinates[i];
	    model->Nu(index, nu[index], cur_model);
	}
    }

    void PrepareMu(std::vector<int> &coordinates) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<double> &mu = GetThreadLocal1dVector("mu");
	for (int i = 0; i < coordinates.size(); i++) {
	    int index = coordinates[i];
	    model->Mu(index, mu[index], cur_model);
	}
    }

    void PrepareH(Datapoint *datapoint, Gradient *g) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &h = GetThreadLocal2dVector("h");
	model->PrecomputeCoefficients(datapoint, g, cur_model);
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->H(index, h[index], g, cur_model);
	}
    }

    double H(int coordinate, int index_into_coordinate_vector) {
	return GetThreadLocal2dVector("h")[coordinate][index_into_coordinate_vector] * FLAGS_learning_rate;
    }

    double Nu(int coordinate, int index_into_coordinate_vector) {
	return GetThreadLocal2dVector("nu")[coordinate][index_into_coordinate_vector] * FLAGS_learning_rate;
    }

    double Mu(int coordinate) {
	return GetThreadLocal1dVector("mu")[coordinate] * FLAGS_learning_rate;
    }

 public:
    SGDUpdater(Model *model, std::vector<Datapoint *> &datapoints) : Updater(model, datapoints) {
	RegisterThreadLocal1dVector("mu", model->NumParameters());
	RegisterThreadLocal2dVector("nu", model->NumParameters(), model->CoordinateSize());
	RegisterThreadLocal2dVector("h", model->NumParameters(), model->CoordinateSize());
    }

    ~SGDUpdater() {
    }
};

#endif
