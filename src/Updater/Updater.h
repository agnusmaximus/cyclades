#ifndef _UPDATER_
#define _UPDATER_

#include "../DatapointPartitions/DatapointPartitions.h"
#include "../Gradient/Gradient.h"

// Some ugly macros to declare extra thread-local / global 1d/2d vectors.
// This avoids the use of std::maps, which are very inefficient.
// Gives around a 2-3x speedup over using maps.
#define REGISTER_THREAD_LOCAL_1D_VECTOR(NAME) std::vector<std::vector<double> > NAME ## _LOCAL_
#define REGISTER_THREAD_LOCAL_2D_VECTOR(NAME) std::vector<std::vector<std::vector<double> > > NAME ## _LOCAL_

#define INITIALIZE_THREAD_LOCAL_1D_VECTOR(NAME, N_COLUMNS) {NAME##_LOCAL_.resize(FLAGS_n_threads); for (int i = 0; i < FLAGS_n_threads; i++) NAME ## _LOCAL_[i].resize(N_COLUMNS, 0);}
#define INITIALIZE_THREAD_LOCAL_2D_VECTOR(NAME, N_ROWS, N_COLUMNS) {NAME##_LOCAL_.resize(FLAGS_n_threads); for (int i = 0; i < FLAGS_n_threads; i++) NAME ## _LOCAL_[i].resize(N_ROWS, std::vector<double>(N_COLUMNS, 0));}

#define GET_THREAD_LOCAL_VECTOR(NAME) NAME ## _LOCAL_[omp_get_thread_num()]

#define REGISTER_GLOBAL_1D_VECTOR(NAME) std::vector<double> NAME ## _GLOBAL_
#define REGISTER_GLOBAL_2D_VECTOR(NAME) std::vector<std::vector<double> > NAME ## _GLOBAL_

#define INITIALIZE_GLOBAL_1D_VECTOR(NAME, N_COLUMNS) {NAME ## _GLOBAL_.resize(N_COLUMNS, 0);}
#define INITIALIZE_GLOBAL_2D_VECTOR(NAME, N_ROWS, N_COLUMNS) {NAME ## _GLOBAL_.resize(N_ROWS, std::vector<double>(N_COLUMNS, 0));}

#define GET_GLOBAL_VECTOR(NAME) NAME ## _GLOBAL_

class Updater {
protected:
    // Keep a reference of the model and datapoints, and partition ordering.
    Model *model;
    std::vector<Datapoint *> datapoints;
    DatapointPartitions *datapoint_partitions;

    // Have an array of Gradient objects (stores extra info for Model processing).
    // Have 1 per thread to avoid conflicts.
    Gradient *thread_gradients;
    std::vector<int> bookkeeping;

    // A reference to all_coordinates, which indexes all the coordinates of the model.
    std::vector<int> all_coordinates;

    // The following datastructures are used to store extra 1d/2d vectors
    // on the fly, which may be needed by a subclass.
    // [thread][name][2d_vector].
    std::vector<std::map<std::string, std::vector<std::vector<double> > > > thread_local_2d_vectors;
    std::vector<std::map<std::string, std::vector<double> > > thread_local_1d_vectors;
    // [name][2d_vector].
    std::map<std::string, std::vector<std::vector<double> > > global_2d_vectors;
    std::map<std::string, std::vector<double > > global_1d_vectors;

    // H, Nu and Mu for updates.
    virtual double H(int coordinate, int index_into_coordinate_vector) = 0;
    virtual double Nu(int coordinate, int index_into_coordinate_vector) = 0;
    virtual double Mu(int coordinate) = 0;

    // After calling PrepareNu/Mu/H, for the given coordinates, we expect that
    // calls to Nu/Mu/H are ready.
    virtual void PrepareNu(std::vector<int> &coordinates) = 0;
    virtual void PrepareMu(std::vector<int> &coordinates) = 0;
    virtual void PrepareH(Datapoint *datapoint, Gradient *g) = 0;

    virtual void ApplyGradient(Datapoint *datapoint) {
	std::vector<double> &model_data = model->ModelData();
	int coordinate_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    double mu = Mu(index);
	    for (int j = 0; j < coordinate_size; j++) {
		model_data[index * coordinate_size + j] = (1 - mu) * model_data[index * coordinate_size + j]
		    - Nu(index, j)
		    + H(index, j);
	    }
	}
    }

    virtual void CatchUpSingle(int index, int diff) {
	if (diff < 0) diff = 0;
	double geom_sum = 0;
	double mu = Mu(index);
	if (mu != 0) {
	    geom_sum = ((1 - pow(1 - mu, diff+1)) / (1 - (1 - mu))) - 1;
	}
	for (int j = 0; j < model->CoordinateSize(); j++) {
	    model->ModelData()[index * model->CoordinateSize() + j] =
		pow(1 - mu, diff) * model->ModelData()[index * model->CoordinateSize() + j]
		- Nu(index, j) * geom_sum;
	}
    }

    virtual void CatchUp(Datapoint *datapoint) {
	std::vector<double> &model_data = model->ModelData();
	int coordinate_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    int diff = datapoint->GetOrder() - bookkeeping[index] - 1;
	    CatchUpSingle(index, diff);
	}
    }

    virtual void FinalCatchUp() {
	int coordinate_size = model->CoordinateSize();
	std::vector<double> &model_data = model->ModelData();
#pragma omp parallel num_threads(FLAGS_n_threads)
	{
	    PrepareNu(all_coordinates);
	    PrepareMu(all_coordinates);
#pragma omp for
	    for (int i = 0; i < model->NumParameters(); i++) {
		int diff = model->NumParameters() - bookkeeping[i];
		CatchUpSingle(i, diff);
	    }
	}
    }

public:
    Updater(Model *model, std::vector<Datapoint *> &datapoints) {
	// Create gradients for each thread.
	thread_gradients = new Gradient[FLAGS_n_threads];
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    thread_gradients[thread] = Gradient();
	    thread_gradients[thread].SetUp(model);
	}
	this->model = model;

	// Create thread local vectors for each thread.
	thread_local_2d_vectors.resize(FLAGS_n_threads);
	thread_local_1d_vectors.resize(FLAGS_n_threads);

	// Set up bookkeping.
	this->datapoints = datapoints;
	for (int i = 0; i < model->NumParameters(); i++) {
	    bookkeeping.push_back(0);
	}

	// Keep an array that has integers 1...n_coords.
	for (int i = 0; i < model->NumParameters(); i++) {
	    all_coordinates.push_back(i);
	}
    }

    Updater() {}
    virtual ~Updater() {
	delete [] thread_gradients;
    }

    // Could be useful to get partitioning info.
    virtual void SetUpWithPartitions(DatapointPartitions &partitions) {
	datapoint_partitions = &partitions;
    }

    // Main update method, which is run by multiple threads.
    virtual void Update(Model *model, Datapoint *datapoint) {
	int thread_num = omp_get_thread_num();
	thread_gradients[thread_num].Clear();
	thread_gradients[thread_num].datapoint = datapoint;

	// First prepare Nu and Mu for catchup since they are independent of the the model.
	PrepareNu(datapoint->GetCoordinates());
	PrepareMu(datapoint->GetCoordinates());
        CatchUp(datapoint);

	// After catching up, prepare H and apply the gradient.
	PrepareH(datapoint, &thread_gradients[thread_num]);
	ApplyGradient(datapoint);

	// Update bookkeeping.
	for (const auto &coordinate : datapoint->GetCoordinates()) {
	    bookkeeping[coordinate] = datapoint->GetOrder();
	}
    }

    // Called before epoch begins.
    virtual void EpochBegin() {
	model->EpochBegin();
    }

    // Called when the epoch ends.
    virtual void EpochFinish() {
	FinalCatchUp();
	std::fill(bookkeeping.begin(), bookkeeping.end(), 0);
	model->EpochFinish();
    }
};

#endif
