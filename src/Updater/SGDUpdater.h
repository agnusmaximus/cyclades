#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/LinearGradient.h"

class SGDUpdater : public Updater {
private:

    void ComputeGradient(Model *model, Datapoint *datapoint, LinearGradient *g) {
	model->Nu(datapoint, g->nu);
	model->Mu(datapoint, g->mu);
	model->H(datapoint, g->h);
	for (int i = 0; i < g->nu.size(); i++) {
	    g->nu[i] *= FLAGS_learning_rate;
	}
	g->mu *= FLAGS_learning_rate;
	g->h *= -FLAGS_learning_rate;
    }

 public:
   SGDUpdater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) : Updater(model, datapoints, n_threads) {}

    ~SGDUpdater() {
    }
};

#endif
