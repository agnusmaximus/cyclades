#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/LinearGradient.h"

class SGDUpdater : public Updater {
private:

    void ComputeGradient(Model *model, Datapoint *datapoint, LinearGradient *g) {
	g->nu_zero = model->Nu(datapoint, g->nu);
	g->mu_zero = model->Mu(datapoint, g->mu);
	g->h_zero = model->H(datapoint, g->h);
	if (!g->nu_zero) {
	    for (int i = 0; i < g->nu.size(); i++) {
		g->nu[i] *= FLAGS_learning_rate;
	    }
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
