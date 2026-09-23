// ported from: Numerics/Mathematics/Optimization/Dynamic/Dijkstra.cs @ 7e8e8d1
#pragma once

namespace corehydro::numerics::math::optimization {

struct Edge {
    int from_index = 0;
    int to_index = 0;
    float weight = 0.0f;
    int index = 0;

    Edge() = default;
    Edge(int from_node_index, int to_node_index, float edge_weight, int edge_index)
        : from_index(from_node_index),
          to_index(to_node_index),
          weight(edge_weight),
          index(edge_index) {}
};

}  // namespace corehydro::numerics::math::optimization
