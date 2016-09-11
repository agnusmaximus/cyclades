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
#ifndef _CYCLADES_PARTITIONER_
#define _CYCLADES_PARTITIONER_

DEFINE_int32(cyclades_batch_size, 5000, "Batch size for cyclades.");

#include "../DatapointPartitions/DatapointPartitions.h"
#include "Partitioner.h"
#include <unordered_map>

class CycladesPartitioner : public Partitioner {
private:
    int model_size;
    int **tree;

    int UnionFind(int a, int *p) {
	int root = a;
	while (p[a] != a) {
	    a = p[a];
	}
	while (root != a) {
	    int root2 = p[root];
	    p[root] = a;
	    root = root2;
	}
	return a;
    }

    void ComputeCC(const std::vector<Datapoint *> & datapoints, int start_index, int end_index,
		   std::unordered_map<int, std::vector<Datapoint *>> &components, int *tree) {
	// Initialize tree for union find.
	for (int i = 0; i < model_size + FLAGS_cyclades_batch_size; i++) {
	    tree[i] = i;
	}

	// CC Computation.
	for (int i = start_index; i < end_index; i++) {
	    Datapoint *point = datapoints[i];
	    int target = UnionFind(i-start_index, tree);
	    for (auto const & coordinate : point->GetCoordinates()) {
		int coordinate_src = UnionFind(coordinate + end_index-start_index, tree);
		tree[coordinate_src] = target;
	    }
	}

	for (int i = 0; i < end_index-start_index; i++) {
	    components[UnionFind(i, tree)].push_back(datapoints[i+start_index]);
	}
    }

public:
    CycladesPartitioner(Model *model) : Partitioner() {
	model_size = model->NumParameters();
	tree = new int *[FLAGS_n_threads];
	for (int i = 0; i < FLAGS_n_threads; i++) {
	    tree[i] = new int[model_size + FLAGS_cyclades_batch_size];
	}
    }
    ~CycladesPartitioner() {
	for (int i = 0; i < FLAGS_n_threads; i++) {
	    delete [] tree[i];
	}
	delete [] tree;
    };

    // Basic partitioner return partition with 1 batch, each thread gets an equal
    // split of a shuffled portion of the datapoints.
    DatapointPartitions Partition(const std::vector<Datapoint *> &datapoints, int n_threads) {

	DatapointPartitions partitions(n_threads);

	// Shuffle the datapoints.
	std::vector<Datapoint *> datapoints_copy(datapoints);

	// Calculate overall number of batches.
	int num_total_batches = ceil((double)datapoints_copy.size() / (double)FLAGS_cyclades_batch_size);

	// Process FLAGS_cyclades_batch_size pointer per iteration, computing CCS on them.
	std::vector<std::unordered_map<int, std::vector<Datapoint *>>> components(num_total_batches);
	#pragma omp parallel for
	for (int datapoint_count = 0; datapoint_count < datapoints_copy.size(); datapoint_count += FLAGS_cyclades_batch_size) {
	    // Current batch index.
	    int batch_index = datapoint_count / FLAGS_cyclades_batch_size;
	    int start = datapoint_count;
	    int end = std::min(datapoint_count + FLAGS_cyclades_batch_size, (int)datapoints_copy.size());

	    // Compute components.
	    ComputeCC(datapoints_copy, start, end,
		      components[batch_index],
		      tree[omp_get_thread_num()]);
	}

	// Load balance the connected components (load balance within the batch, not across it).
	for (int batch = 0; batch < num_total_batches; batch++) {
	    for (std::unordered_map<int, std::vector<Datapoint *>>::iterator it = components[batch].begin();
		 it != components[batch].end(); it++) {
		partitions.AddDatapointsToLeastLoadedThread(it->second);
	    }
	    partitions.StartNewBatch();
	}

	return partitions;
    }
};

#endif
