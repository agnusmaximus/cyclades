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
#ifndef _MODEL_
#define _MODEL_

#include "../DatapointPartitions/DatapointPartitions.h"

class Model {
 public:
    Model() {}
    Model(const std::string &input_line) {}
    virtual ~Model() {}

    // Computes loss on the model
    virtual double ComputeLoss(const std::vector<Datapoint *> &datapoints) = 0;

    // Do some set up with the model and datapoints before running gradient descent.
    virtual void SetUp(const std::vector<Datapoint *> &datapoints) {}

    // Do some set up with the model given partitioning scheme before running the trainer.
    virtual void SetUpWithPartitions(DatapointPartitions &partitions) {}

    // Do any sort of extra computation at the beginning of an epoch.
    virtual void EpochBegin() {}

    // Do any sort of extra computation at the end of an epoch.
    virtual void EpochFinish() {}

    // Return the number of parameters of the model.
    virtual int NumParameters() = 0;

    // Return the size (the # of doubles) of a single coordinate.
    virtual int CoordinateSize() = 0;

    // Return data to actual model.
    virtual std::vector<double> & ModelData() = 0;

    // The following are for updates of the form:
    // x_j = (1 - mu_j)x_j - nu_j + h_ij(x_S_i)
    // Where h_ij(x) = 0 for j not in S_i.
    virtual void PrecomputeCoefficients(Datapoint *datapoint, Gradient *g, std::vector<double> &local_model) = 0;
    virtual void Lambda(int coordinate, double &out, std::vector<double> &local_model) = 0;
    virtual void Kappa(int coordinate, std::vector<double> &out, std::vector<double> &local_model) = 0;
    virtual void H_bar(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model) = 0;

    // The following methods are for custom gradient creation.
    virtual void CustomCatchUp(int index, int diff) {
	std::cout << "Model.h: CustomCatchUp is not implemented." << std::endl;
	exit(0);
    }

    virtual void CustomPrepareGradient(Datapoint *datapoint, Gradient *gradient) {
	std::cout << "Model.h: CustomPrepareGradient is not implemented." << std::endl;
	exit(0);
    }

    virtual void CustomApplyGradient(Datapoint *datapoint, Gradient *gradient) {
	std::cout << "Model.h: CustomApplyGradient is not implemented." << std::endl;
	exit(0);
    }
};

#endif
