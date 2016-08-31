#ifndef _LINEARGRADIENT_
#define _LINEARGRADIENT_

// Gradient of updates of the form:
// x_j = (1 - mu_j)x_j - nu_j + h_ij*x_S_i
// Where h_ij = 0 when j not in S_i.
class LinearGradient : public Gradient {
 public:
    double mu, h;
    std::vector<double> nu;
    bool mu_zero, h_zero, nu_zero;
    Datapoint *datapoint;

    void Clear() override {
	datapoint = NULL;
	mu = h = 0;
	std::fill(nu.begin(), nu.end(), 0);
	mu_zero = h_zero = nu_zero = true;
    }

    void SetUp(Model *model) override {
	Clear();
	nu.resize(model->CoordinateSize());
    }

    LinearGradient() {}

    ~LinearGradient() {
    }
};

#endif
