
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
#ifndef _GIBBSDATAPOINT_
#define _GIBBSDATAPOINT_

#include <tuple>
#include <sstream>
#include "Datapoint.h"

class GibbsDatapoint : public Datapoint {
private:
    std::vector<double> weights; // Remains empty.
    std::vector<int> neighbors;

    void Initialize(const std::string &input_line) {
	std::stringstream input(input_line);

	// Expect format:
	// Coord# prior neighbor1 neighbor2 ... neighborn.
	input >> coord;
	input >> tendency;
	while (input) {
	    int neighbor;
	    input >> neighbor;
	    if (!input) break;
	    neighbors.push_back(neighbor);
	}
    }

 public:
    int coord;
    int tendency;

    GibbsDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }

    std::vector<double> & GetWeights() override {
	return weights;
    }

    std::vector<int> & GetCoordinates() override {
	return neighbors;
    }

    int GetNumCoordinateTouches() override {
	return neighbors.size();
    }
};

#endif
