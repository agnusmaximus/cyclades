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

#ifndef _CUSTOMUPDATER_
#define _CUSTOMUPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class CustomUpdater : public Updater {
protected:

    void PrepareNu(std::vector<int> &coordinates) override {
    }

    void PrepareMu(std::vector<int> &coordinates) override {
    }

    void PrepareH(Datapoint *datapoint, Gradient *g) override {
    }

    double H(int coordinate, int index_into_coordinate_vector) {
	return 0;
    }

    double Nu(int coordinate, int index_into_coordinate_vector) {
	return 0;
    }

    double Mu(int coordinate) {
	return 0;
    }

    void CatchUp(int index, int diff) override {
	model->CustomCatchUp(index, diff);
    }

    void Update(Model *model, Datapoint *datapoint) override {
	int thread_num = omp_get_thread_num();
	thread_gradients[thread_num].Clear();
	thread_gradients[thread_num].datapoint = datapoint;

	// Catch up.
        CatchUpDatapoint(datapoint);

	// Prepare and apply gradient.
	model->CustomPrepareGradient(datapoint, &thread_gradients[thread_num]);
	model->CustomApplyGradient(datapoint, &thread_gradients[thread_num]);

	// Update bookkeeping.
	for (const auto &coordinate : datapoint->GetCoordinates()) {
	    bookkeeping[coordinate] = datapoint->GetOrder();
	}
    }

 public:
    CustomUpdater(Model *model, std::vector<Datapoint *> &datapoints) : Updater(model, datapoints) {
    }

    ~CustomUpdater() {
    }
};

#endif
