#include "Gradient.h"
#include "../Model/Model.h"

void Gradient::SetUp(Model *model) {
    Clear();
    //nu.resize(model->NumParameters() * model->CoordinateSize());
    //mu.resize(model->NumParameters());
    //h.resize(model->NumParameters() * model->CoordinateSize());
    mu.resize(model->NumParameters());
    nu.resize(model->NumParameters(), std::vector<double>(model->CoordinateSize(), 0));
    h.resize(model->NumParameters(), std::vector<double>(model->CoordinateSize(), 0));
}
