#ifndef _SVRG_UPDATER_
#define _SVRG_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SVRGUpdater : public Updater {
protected:
    std::vector<double> model_copy;

    virtual void ComputeAllNuAndMu(Model *model, Gradient *g) {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<double> &lambda = g->Get1dVector("lambda");
	for (int i = 0; i < model->NumParameters(); i++) {
	    model->Mu(i, lambda[i], cur_model);
	}
    }

    virtual void ComputeGradient(Model *model, Datapoint *datapoint, Gradient *g) {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &h_x = g->Get2dVector("h_x");
	std::vector<std::vector<double> > &h_y = g->Get2dVector("h_y");
	std::vector<double> &lambda = g->Get1dVector("lambda");

	g->datapoint = datapoint;
	model->PrecomputeCoefficients(datapoint, g, cur_model);
	int coord_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->Mu(index, lambda[index], cur_model);
	    model->H(index, h_x[index], g, cur_model);
	}
	model->PrecomputeCoefficients(datapoint, g, model_copy);
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->H(index, h_y[index], g, model_copy);
	}
    }

    double H(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return -FLAGS_learning_rate * (g->Get2dVector("h_x")[coordinate][index_into_coordinate_vector] -
				       g->Get2dVector("h_y")[coordinate][index_into_coordinate_vector]);

    }

    double Nu(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return FLAGS_learning_rate * (g->Get2dVector("g")[coordinate][index_into_coordinate_vector] -
				      g->Get1dVector("lambda")[coordinate] * model_copy[coordinate*model->CoordinateSize()+index_into_coordinate_vector]);
    }

    double Mu(int coordinate, Gradient *g) {
	return g->Get1dVector("lambda")[coordinate] * FLAGS_learning_rate;
    }

 public:
 SVRGUpdater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) : Updater(model, datapoints, n_threads) {
	Register2dVector("g", model->NumParameters(), model->CoordinateSize());
	Register1dVector("lambda", model->NumParameters());
	Register2dVector("h_x", model->NumParameters(), model->CoordinateSize());
	Register2dVector("h_y", model->NumParameters(), model->CoordinateSize());
	model_copy.resize(model->ModelData().size());
    }

    ~SVRGUpdater() {
    }

    void EpochFinish() override {
	Updater::EpochFinish();

	// Make a copy of the model every epoch.
	model_copy = model->ModelData();

	// Compute average sum of gradients of the model_copy.
	Gradient *grad = &thread_gradients[omp_get_thread_num()];
	//std::vector<std::vector<double> > &g = grad->Get2dVector("g");
	//for (int i = 0; i < datapoints.size(); i++) {
	//}
    }
};

#endif
