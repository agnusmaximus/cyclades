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
    std::vector<double> coeffs;
    std::map<std::string, std::vector<std::vector<double> > > vectors_2d;
    std::map<std::string, std::vector<double> > vectors_1d;
    Datapoint *datapoint;

    Gradient() {}
    virtual ~Gradient() {}

    virtual void Clear() {
	datapoint = NULL;
    }

    void Register2dVector(std::string name, int n_rows, int n_columns) {
	vectors_2d[name].resize(n_rows, std::vector<double>(n_columns, 0));
    }

    void Register1dVector(std::string name, int n_rows) {
	vectors_1d[name].resize(n_rows, 0);
    }

    std::vector<std::vector<double> > & Get2dVector(std::string name) {
	return vectors_2d.find(name)->second;
    }

    std::vector<double> & Get1dVector(std::string name) {
	return vectors_1d.find(name)->second;
    }

    void SetUp(Model *model) {

    }
};

#endif
