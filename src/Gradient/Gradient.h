#ifndef _GRADIENT_
#define _GRADIENT_

#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>

class Model;
class Datapoint;

class Gradient {
 public:
    std::vector<double> nu, mu, h;
    std::vector<double> coeffs;
    Datapoint *datapoint;

    Gradient() {}
    virtual ~Gradient() {}

    virtual void Clear() {
	std::fill(coeffs.begin(), coeffs.end(), 0);
	datapoint = NULL;
    }

    void SetUp(Model *model);
};

#endif
