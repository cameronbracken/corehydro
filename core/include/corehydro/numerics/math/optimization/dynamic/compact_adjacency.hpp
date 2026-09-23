// ported from: Numerics/Mathematics/Optimization/Dynamic/CompactAdjacency.cs @ 7e8e8d1
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/optimization/dynamic/edge.hpp"

namespace corehydro::numerics::math::optimization {

// Stable compressed sparse row view over the network edges. Parallel arrays preserve original
// edge order within each node bucket, so relaxation and exact tie behavior match the list form.
struct CompactAdjacency {
    int node_count = 0;
    std::vector<int> row_start;
    std::vector<int> from_node;
    std::vector<int> to_node;
    std::vector<float> weight;
    std::vector<int> edge_index;
    // Empty when built from caller-supplied incoming lists.
    std::vector<int> source_position;

    static CompactAdjacency from_edges(const std::vector<Edge>& edges, int node_count,
                                       bool group_by_end_node,
                                       const std::string& parameter_name);

    static CompactAdjacency from_incoming_lists(
        const std::vector<std::vector<Edge>>& edges_to_nodes, int node_count,
        const std::string& parameter_name);
};

}  // namespace corehydro::numerics::math::optimization
