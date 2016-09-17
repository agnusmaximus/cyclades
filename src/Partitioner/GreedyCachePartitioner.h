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
#ifndef _GREEDY_PARTITIONER_
#define _GREEDY_PARTITIONER_

#include <unordered_map>
#include "Partitioner.h"

class GreedyCachePartitioner : public Partitioner {
 protected:

    int CalculateOverlap(std::unordered_map<int, bool> &first, Datapoint *second) {
	int result = 0;
	for (const auto & coordinate : second->GetCoordinates()) {
	    if (first.find(coordinate) != first.end()) {
		result++;
	    }
	}
	return result;
    }

    void FindDatapointWithMaximumOverlap(Datapoint *datapoint,
					 std::unordered_map<int, bool> &coords,
					 const std::vector<Datapoint *> &datapoints,
					 int &max_overlap, int &index_of_result,
					 std::unordered_map<int, bool> &used_datapoints) {
	int best_overlap = -1, best_index = -1;
	for (int i = 0; i < datapoints.size(); i++) {
	    if (datapoints[i] != datapoint &&
		used_datapoints.find(i) == used_datapoints.end()) {
		int cur_overlap = CalculateOverlap(coords, datapoints[i]);
		if (cur_overlap > best_overlap) {
		    best_overlap = cur_overlap;
		    best_index = i;
		}
	    }
	}
	max_overlap = best_overlap;
	index_of_result = best_index;
    }
 public:
    GreedyCachePartitioner() {};
    ~GreedyCachePartitioner() {};

    // Assumptions: datapoints orders (id) are continuous and in order.
    DatapointPartitions Partition(const std::vector<Datapoint *> &datapoints, int n_threads) {
	// This only works for 1 thread for now!
	if (FLAGS_n_threads > 1) {
	    std::cout << "GreedyCachePartitioner.h: Only 1 threaded is valid for now." << std::endl;
	    exit(0);
	}

	// Keep track of a map of distinct coordinate accesses accessed by datapoint.
	std::vector<std::unordered_map<int, bool> > datapoint_coordinate_accesses(datapoints.size());
	for (int i = 0; i < datapoints.size(); i++) {
	    for (const auto &coordinate : datapoints[i]->GetCoordinates()) {
		datapoint_coordinate_accesses[i][coordinate] = 1;
	    }
	}

	// Note the cache permutation stores the indices of the elements.
	std::vector<int> cache_permutation;
	std::unordered_map<int, bool> used_datapoints;

	// Find the pair of datapoints that produces the best overlap.
	int global_max_overlap = 0, best_first, best_second;
	for (int i = 0; i < datapoints.size(); i++) {
	    int cur_max_overlap = 0;
	    int index_of_second_datapoint = 0;
	    FindDatapointWithMaximumOverlap(datapoints[i], datapoint_coordinate_accesses[i],
					    datapoints, cur_max_overlap, index_of_second_datapoint, used_datapoints);
	    if (cur_max_overlap > global_max_overlap) {
		best_first = i;
		best_second = index_of_second_datapoint;
	    }
	}
	cache_permutation.push_back(best_first);
	cache_permutation.push_back(best_second);
	used_datapoints[best_first] = 1;
	used_datapoints[best_second] = 1;

	// Greedly append datapoint that produces best overlap.
	for (int i = 2; i < datapoints.size(); i++) {
	    int max_overlap = 0;
	    int next_index = 0;
	    FindDatapointWithMaximumOverlap(datapoints[cache_permutation[i-1]],
					    datapoint_coordinate_accesses[cache_permutation[i-1]],
					    datapoints, max_overlap, next_index, used_datapoints);
	    cache_permutation.push_back(next_index);
	    used_datapoints[next_index] = 1;
	}

	if (cache_permutation.size() != datapoints.size()) {
	    std::cout << "GreedyCachePartitioner.h: Something went wrong... Sizes of permutation and datapoint set don't match." << std::endl;
	    exit(0);
	}

	// Number of datapoints. To be used as graph node id offset.
	int n_datapoints = datapoints.size();
	DatapointPartitions partitions(n_threads);

	for (int i = 0; i < datapoints.size(); i++) {
	    partitions.AddDatapointToThread(datapoints[cache_permutation[i]], 0);
	}

	return partitions;
    }
};

#endif
