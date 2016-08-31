#ifndef _GRADIENT_
#define _GRADIENT_

class Model;

class Gradient {
 public:
    std::vector<double> coeffs;

    Gradient() {}
    virtual ~Gradient() {}

    virtual void Clear() {
	std::fill(coeffs.begin(), coeffs.end(), 0);
    }

    virtual void SetUp(Model *model) {
	Clear();
    }
};

#endif
