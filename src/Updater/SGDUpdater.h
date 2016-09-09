
/*
* Copyright 2016 [See AUTHORS file for list of authors]
*
*    Licensed under the Apache License, Version 2.0 (the "License");
*    you may not use this file except in compliance with the License.
*    You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS,
*    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*    See the License for the specific language governing permissions and
*    limitations under the License.
*/
#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SGDUpdater : public Updater {
protected:
    REGISTER_THREAD_LOCAL_1D_VECTOR(mu);
    REGISTER_THREAD_LOCAL_2D_VECTOR(nu);
    REGISTER_THREAD_LOCAL_2D_VECTOR(h);

    void PrepareNu(std::vector<int> &coordinates) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &nu = GET_THREAD_LOCAL_VECTOR(nu);
	for (int i = 0; i < coordinates.size(); i++) {
	    int index = coordinates[i];
	    model->Nu(index, nu[index], cur_model);
	}
    }

    void PrepareMu(std::vector<int> &coordinates) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<double> &mu = GET_THREAD_LOCAL_VECTOR(mu);
	for (int i = 0; i < coordinates.size(); i++) {
	    int index = coordinates[i];
	    model->Mu(index, mu[index], cur_model);
	}
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

    double H(int coordinate, int index_into_coordinate_vector) {
	return GET_THREAD_LOCAL_VECTOR(h)[coordinate][index_into_coordinate_vector] * FLAGS_learning_rate;
    }

    double Nu(int coordinate, int index_into_coordinate_vector) {
	return GET_THREAD_LOCAL_VECTOR(nu)[coordinate][index_into_coordinate_vector] * FLAGS_learning_rate;
    }

    double Mu(int coordinate) {
	return GET_THREAD_LOCAL_VECTOR(mu)[coordinate] * FLAGS_learning_rate;
    }

 public:
    SGDUpdater(Model *model, std::vector<Datapoint *> &datapoints) : Updater(model, datapoints) {
	INITIALIZE_THREAD_LOCAL_1D_VECTOR(mu, model->NumParameters());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(nu, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h, model->NumParameters(), model->CoordinateSize());
    }

    ~SGDUpdater() {
    }
};

#endif
