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
    std::vector<std::vector<double> > nu, h;
    std::vector<double> mu;
    std::vector<double> coeffs;
    Datapoint *datapoint;

    Gradient() {}
    virtual ~Gradient() {}

    virtual void Clear() {
	datapoint = NULL;
    }

    void SetUp(Model *model);
};

#endif
