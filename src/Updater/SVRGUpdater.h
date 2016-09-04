#ifndef _SVRG_UPDATER_
#define _SVRG_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SVRGUpdater : public Updater {
protected:
    std::vector<double> model_copy;

    virtual void ComputeAllNuAndMu(Gradient *g) {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<double> &lambda = GetThreadLocal1dVector("lambda");
	for (int i = 0; i < model->NumParameters(); i++) {
	    model->Mu(i, lambda[i], cur_model);
	}
    }

    virtual void ComputeGradient(Datapoint *datapoint, Gradient *g) {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &h_x = GetThreadLocal2dVector("h_x");
	std::vector<std::vector<double> > &h_y = GetThreadLocal2dVector("h_y");
	std::vector<double> &lambda = GetThreadLocal1dVector("lambda");

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
	return -FLAGS_learning_rate * (GetThreadLocal2dVector("h_x")[coordinate][index_into_coordinate_vector] -
				       GetThreadLocal2dVector("h_y")[coordinate][index_into_coordinate_vector]);
    }

    double Nu(int coordinate, int index_into_coordinate_vector, Gradient *g) {
	return FLAGS_learning_rate * (GetGlobal2dVector("g")[coordinate][index_into_coordinate_vector] -
				      GetThreadLocal1dVector("lambda")[coordinate] * model_copy[coordinate*model->CoordinateSize()+index_into_coordinate_vector]);
    }

    double Mu(int coordinate, Gradient *g) {
	return GetThreadLocal1dVector("lambda")[coordinate] * FLAGS_learning_rate;
    }

 public:
 SVRGUpdater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) : Updater(model, datapoints, n_threads) {
	RegisterGlobal2dVector("g", model->NumParameters(), model->CoordinateSize());
	RegisterThreadLocal1dVector("lambda", model->NumParameters());
	RegisterThreadLocal2dVector("h_x", model->NumParameters(), model->CoordinateSize());
	RegisterThreadLocal2dVector("h_y", model->NumParameters(), model->CoordinateSize());
	model_copy.resize(model->ModelData().size());

	RegisterThreadLocal2dVector("kappa", model->NumParameters(), model->CoordinateSize());
    }

    ~SVRGUpdater() {
    }

    void EpochBegin() override {
	Updater::EpochBegin();

	// Make a copy of the model every epoch.
	model_copy = model->ModelData();

	// Compute average sum of gradients of the model_copy.
	Gradient *grad = &thread_gradients[omp_get_thread_num()];
	std::vector<std::vector<double> > &g = GetGlobal2dVector("g");
	for (auto & v : g)
	    std::fill(v.begin(), v.end(), 0);

	std::vector<std::vector<double> > &nu = GetThreadLocal2dVector("kappa");
	std::vector<double> &mu = GetThreadLocal1dVector("lambda");
	std::vector<std::vector<double> > &h = GetThreadLocal2dVector("h_x");

	int n_coords = model->NumParameters();
	int coordinate_size = model->CoordinateSize();
	for (int dp = 0; dp < datapoints.size(); dp++) {
	    Datapoint *datapoint = datapoints[dp];

	    for (auto & v : h)
		std::fill(v.begin(), v.end(), 0);
	    for (auto & v : nu)
		std::fill(v.begin(), v.end(), 0);
	    std::fill(mu.begin(), mu.end(), 0);

	    model->PrecomputeCoefficients(datapoint, grad, model->ModelData());

	    // Compute the origial sgd gradients needed in order to sum them.
	    for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
		int index = datapoint->GetCoordinates()[i];
		model->Mu(index, mu[index], model->ModelData());
		model->Nu(index, nu[index], model->ModelData());
		model->H(index, h[index], grad, model->ModelData());
	    }

	    // Calc the gradients.
	    for (int i = 0; i < n_coords; i++) {
		for (int j = 0; j < coordinate_size; j++) {
		    g[i][j] += (mu[i] * model_copy[i*coordinate_size+j] + nu[i][j] + h[i][j]) / datapoints.size();
		}
	    }

	    // Sum the gradients now
	}
    }
};

#endif
