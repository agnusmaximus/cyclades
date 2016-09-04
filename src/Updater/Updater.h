#ifndef _UPDATER_
#define _UPDATER_

#include "../DatapointPartitions/DatapointPartitions.h"
#include "../Gradient/Gradient.h"

class Updater {
protected:
    // Keep a reference of the model and datapoints.
    Model *model;
    std::vector<Datapoint *> datapoints;

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

    void CatchUp(Datapoint *datapoint) {
	// Optimize by quick returning if nu and mu are zero.
	std::vector<double> &model_data = model->ModelData();
	int coordinate_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    int diff = datapoint->GetOrder() - bookkeeping[index] - 1;
	    if (diff < 0) diff = 0;
	    double geom_sum = 0;
	    double mu = Mu(index);
	    if (mu != 0) {
 		geom_sum = ((1 - pow(1 - mu, diff+1)) / (1 - (1 - mu))) - 1;
	    }
	    for (int j = 0; j < coordinate_size; j++) {
		model_data[index * coordinate_size + j] =
		    pow(1 - mu, diff) * model_data[index * coordinate_size + j]
		    - Nu(index, j) * geom_sum;
	    }
	}
    }

    void FinalCatchUp() {
	int coordinate_size = model->CoordinateSize();
	std::vector<double> &model_data = model->ModelData();
#pragma omp parallel num_threads(FLAGS_n_threads)
	{
	    Gradient *g = &thread_gradients[omp_get_thread_num()];
	    PrepareNu(all_coordinates);
	    PrepareMu(all_coordinates);
#pragma omp for
	    for (int i = 0; i < model->NumParameters(); i++) {
		int diff = model->NumParameters() - bookkeeping[i];
		double geom_sum = 0, mu = Mu(i);
		if (mu != 0) {
		    geom_sum = ((1 - pow(1 - mu, diff+1)) / (1 - (1 - mu))) - 1;
		}
		for (int j = 0; j < coordinate_size; j++) {
		    model_data[i * coordinate_size + j] =
			pow(1 - mu, diff) * model_data[i * coordinate_size + j]
			- Nu(i, j) * geom_sum;
		}
	    }
	}
    }

    void RegisterGlobal2dVector(std::string name, int n_rows, int n_columns) {
	global_2d_vectors[name].resize(n_rows, std::vector<double>(n_columns, 0));
    }

    void RegisterGlobal1dVector(std::string name, int n_cols) {
	global_1d_vectors[name].resize(n_cols, 0);
    }

    std::vector<std::vector<double> > & GetGlobal2dVector(std::string name) {
	return global_2d_vectors[name];
    }

    std::vector<double> & GetGlobal1dVector(std::string name) {
	return global_1d_vectors[name];
    }

    void RegisterThreadLocal2dVector(std::string name, int n_rows, int n_columns) {
	for (int i = 0; i <FLAGS_n_threads; i++) {
	    thread_local_2d_vectors[i][name].resize(n_rows, std::vector<double>(n_columns, 0));
	}
    }

    void RegisterThreadLocal1dVector(std::string name, int n_columns) {
	for (int i = 0; i <FLAGS_n_threads; i++) {
	    thread_local_1d_vectors[i][name].resize(n_columns, 0);
	}
    }

    std::vector<std::vector<double> > & GetThreadLocal2dVector(std::string name) {
	return thread_local_2d_vectors[omp_get_thread_num()][name];
    }

    std::vector<double> & GetThreadLocal1dVector(std::string name) {
	return thread_local_1d_vectors[omp_get_thread_num()][name];
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

    // Main update method, which is run by multiple threads.
    virtual void Update(Model *model, Datapoint *datapoint) {
	int thread_num = omp_get_thread_num();
	thread_gradients[thread_num].Clear();

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
