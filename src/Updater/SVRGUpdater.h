#ifndef _SVRG_UPDATER_
#define _SVRG_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SVRGUpdater : public Updater {
protected:
    std::vector<double> model_copy;
    // Vectors for computing SVRG related data.
    REGISTER_THREAD_LOCAL_1D_VECTOR(lambda);
    REGISTER_THREAD_LOCAL_2D_VECTOR(h_x);
    REGISTER_THREAD_LOCAL_2D_VECTOR(h_y);
    REGISTER_GLOBAL_1D_VECTOR(g);

    // Vectors for computing the sum of gradients (g).
    REGISTER_THREAD_LOCAL_2D_VECTOR(g_nu);
    REGISTER_THREAD_LOCAL_1D_VECTOR(g_mu);
    REGISTER_THREAD_LOCAL_2D_VECTOR(g_h);
    REGISTER_GLOBAL_1D_VECTOR(g_bookkeeping);

    void PrepareMu(std::vector<int> &coordinates) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<double> &lambda = GET_THREAD_LOCAL_VECTOR(lambda);
	for (int i = 0; i < coordinates.size(); i++) {
	    int index = coordinates[i];
	    model->Mu(index, lambda[index], cur_model);
	}
    }

    void PrepareNu(std::vector<int> &coordinates) override {
    }

    void PrepareH(Datapoint *datapoint, Gradient *g) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &h_x = GET_THREAD_LOCAL_VECTOR(h_x);
	std::vector<std::vector<double> > &h_y = GET_THREAD_LOCAL_VECTOR(h_y);

	g->datapoint = datapoint;
	model->PrecomputeCoefficients(datapoint, g, cur_model);
	int coord_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->H(index, h_x[index], g, cur_model);
	}
	model->PrecomputeCoefficients(datapoint, g, model_copy);
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->H(index, h_y[index], g, model_copy);
	}
    }

    double H(int coordinate, int index_into_coordinate_vector) {
	return FLAGS_learning_rate * (GET_THREAD_LOCAL_VECTOR(h_x)[coordinate][index_into_coordinate_vector] -
				      GET_THREAD_LOCAL_VECTOR(h_y)[coordinate][index_into_coordinate_vector]);
    }

    double Nu(int coordinate, int index_into_coordinate_vector) {
	return FLAGS_learning_rate * (GET_GLOBAL_VECTOR(g)[coordinate*model->CoordinateSize()+index_into_coordinate_vector] -
				      GET_THREAD_LOCAL_VECTOR(lambda)[coordinate] * model_copy[coordinate*model->CoordinateSize()+index_into_coordinate_vector]);
    }

    double Mu(int coordinate) {
	return GET_THREAD_LOCAL_VECTOR(lambda)[coordinate] * FLAGS_learning_rate;
    }

 public:
 SVRGUpdater(Model *model, std::vector<Datapoint *> &datapoints) : Updater(model, datapoints) {
	INITIALIZE_GLOBAL_1D_VECTOR(g, model->NumParameters() * model->CoordinateSize());
	INITIALIZE_THREAD_LOCAL_1D_VECTOR(lambda, model->NumParameters());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h_x, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h_y, model->NumParameters(), model->CoordinateSize());
	model_copy.resize(model->ModelData().size());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(g_nu, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_THREAD_LOCAL_1D_VECTOR(g_mu, model->NumParameters());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(g_h, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_GLOBAL_1D_VECTOR(g_bookkeeping, model->NumParameters());
    }

    ~SVRGUpdater() {
    }

    void EpochBegin() override {
	Updater::EpochBegin();

	// Make a copy of the model every epoch.
	model_copy = model->ModelData();

	// Clear the sum of gradients.
	std::vector<double> &g = GET_GLOBAL_VECTOR(g);
	std::fill(g.begin(), g.end(), 0);

	// Compute average sum of gradients on the model copy.
	// Essentially perform SGD on it.
	std::vector<std::vector<double> > &g_nu = GET_THREAD_LOCAL_VECTOR(g_nu);
	std::vector<double> &g_mu = GET_THREAD_LOCAL_VECTOR(g_mu);
	std::vector<std::vector<double> > &g_h = GET_THREAD_LOCAL_VECTOR(g_h);
	std::vector<double> &g_bookkeeping = GET_GLOBAL_VECTOR(g_bookkeeping);
	std::fill(g_bookkeeping.begin(), g_bookkeeping.end(), 0);
	int coord_size = model->CoordinateSize();

	// Compute average sum of gradients of the model_copy.
	/*Gradient *grad = &thread_gradients[omp_get_thread_num()];
	std::vector<std::vector<double> > &g = GET_GLOBAL_VECTOR(g);
	for (auto & v : g)
	    std::fill(v.begin(), v.end(), 0);

	std::vector<std::vector<double> > &nu = GET_THREAD_LOCAL_VECTOR(kappa);
	std::vector<double> &mu = GET_THREAD_LOCAL_VECTOR(lambda);
	std::vector<std::vector<double> > &h = GET_THREAD_LOCAL_VECTOR(h_x);

	int n_coords = model->NumParameters();
	int coordinate_size = model->CoordinateSize();
	for (int dp = 0; dp < datapoints.size(); dp++) {
	    Datapoint *datapoint = datapoints[dp];

	    for (auto & v : h)
		std::fill(v.begin(), v.end(), 0);
	    for (auto & v : nu)
		std::fill(v.begin(), v.end(), 0);
	    std::fill(mu.begin(), mu.end(), 0);
	    std::fill(grad->coeffs.begin(), grad->coeffs.end(), 0);

	    model->PrecomputeCoefficients(datapoint, grad, model->ModelData());

	    // Compute the origial sgd gradients needed in order to sum them.
	    //for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    for (int i = 0; i < n_coords; i++) {
		model->Mu(i, mu[i], model->ModelData());
		model->Nu(i, nu[i], model->ModelData());
		model->H(i, h[i], grad, model->ModelData());
	    }

	    // Calc the gradients.
	    for (int i = 0; i < n_coords; i++) {
		for (int j = 0; j < coordinate_size; j++) {
		    g[i][j] += (mu[i] * model_copy[i*coordinate_size+j] + nu[i][j] - h[i][j]) / datapoints.size();
		}
	    }
	    }*/
    }
};

#endif
