#ifndef _SAGA_UPDATER_
#define _SAGA_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SAGAUpdater: public Updater {
 protected:

    // Data structures for capturing the gradient.
    REGISTER_THREAD_LOCAL_2D_VECTOR(h);

    // SAGA data structures.
    REGISTER_GLOBAL_2D_VECTOR(sum_gradients);
    REGISTER_GLOBAL_2D_VECTOR(prev_gradients);

    void PrepareNu(std::vector<int> &coordinates) override {
	// Assuming gradients are sparse, nu should be 0.
    }

    void PrepareMu(std::vector<int> &coordinates) override {
	// We also assume mu is 0.
    }

    void PrepareH(Datapoint *datapoint, Gradient *g) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &h = GET_THREAD_LOCAL_VECTOR(h);
	model->PrecomputeCoefficients(datapoint, g, cur_model);
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->H(index, h[index], g, cur_model);
	}
    }

    double H(int coordinate, int index_into_coordinate_vector) override {
	return 0;
    }

    double Nu(int coordinate, int index_into_coordinate_vector) override {
	return 0;
    }

    double Mu(int coordinate) override {
	return 0;
    }


 public:
    SAGAUpdater(Model *model, std::vector<Datapoint *>&datapoints): Updater(model, datapoints) {
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h, model->NumParameters(), model->CoordinateSize());

	INITIALIZE_GLOBAL_2D_VECTOR(sum_gradients, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_GLOBAL_2D_VECTOR(prev_gradients, model->NumParameters(), model->CoordinateSize());
    }

    ~SAGAUpdater() {}

};

#endif
