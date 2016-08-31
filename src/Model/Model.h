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
    // x_j = (1 - mu_j)x_j - nu_j + h_ij*x_S_i
    // Where h_ij = 0 for j not in S_i.
    virtual void Mu(Datapoint *datapoint, double &mu_out) = 0;
    virtual void Nu(Datapoint *datapoint, std::vector<double> &nu_out) = 0;
    virtual void H(Datapoint *datapoint, double &h_out) = 0;
};

#endif
