// ported from: Numerics/Mathematics/Optimization/Dynamic/Dijkstra.cs @ 7e8e8d1
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/optimization/dynamic/compact_adjacency.hpp"
#include "corehydro/numerics/math/optimization/dynamic/edge.hpp"
#include "corehydro/numerics/math/optimization/dynamic/indexed_min_heap.hpp"

namespace corehydro::numerics::math::optimization {

inline CompactAdjacency CompactAdjacency::from_edges(const std::vector<Edge>& edges,
                                                     int count, bool group_by_end_node,
                                                     const std::string& parameter_name) {
    CompactAdjacency out;
    out.node_count = count;
    out.row_start.assign(static_cast<std::size_t>(count + 1), 0);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const Edge& edge = edges[i];
        if (edge.from_index < 0 || edge.from_index >= count || edge.to_index < 0 ||
            edge.to_index >= count) {
            int bad = edge.from_index < 0 || edge.from_index >= count ? edge.from_index
                                                                      : edge.to_index;
            throw std::invalid_argument(parameter_name + ": edge at position " +
                                        std::to_string(i) + " references node index " +
                                        std::to_string(bad) + " outside [0, " +
                                        std::to_string(count) + ")");
        }
        if (edge.index < 0)
            throw std::invalid_argument(parameter_name + ": edge at position " +
                                        std::to_string(i) + " has negative edge index " +
                                        std::to_string(edge.index));
        int bucket = group_by_end_node ? edge.to_index : edge.from_index;
        ++out.row_start[static_cast<std::size_t>(bucket + 1)];
    }
    for (int n = 0; n < count; ++n)
        out.row_start[static_cast<std::size_t>(n + 1)] +=
            out.row_start[static_cast<std::size_t>(n)];

    std::vector<int> cursor(static_cast<std::size_t>(count), 0);
    out.from_node.resize(edges.size());
    out.to_node.resize(edges.size());
    out.weight.resize(edges.size());
    out.edge_index.resize(edges.size());
    out.source_position.resize(edges.size());
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const Edge& edge = edges[i];
        int bucket = group_by_end_node ? edge.to_index : edge.from_index;
        int slot = out.row_start[static_cast<std::size_t>(bucket)] +
                   cursor[static_cast<std::size_t>(bucket)]++;
        std::size_t k = static_cast<std::size_t>(slot);
        out.from_node[k] = edge.from_index;
        out.to_node[k] = edge.to_index;
        out.weight[k] = edge.weight;
        out.edge_index[k] = edge.index;
        out.source_position[k] = static_cast<int>(i);
    }
    return out;
}

inline CompactAdjacency CompactAdjacency::from_incoming_lists(
    const std::vector<std::vector<Edge>>& lists, int count,
    const std::string& parameter_name) {
    CompactAdjacency out;
    out.node_count = count;
    out.row_start.assign(static_cast<std::size_t>(count + 1), 0);
    int edge_count = 0;
    for (int n = 0; n < count; ++n) {
        edge_count += static_cast<int>(lists[static_cast<std::size_t>(n)].size());
        out.row_start[static_cast<std::size_t>(n + 1)] = edge_count;
    }
    out.from_node.resize(static_cast<std::size_t>(edge_count));
    out.to_node.resize(static_cast<std::size_t>(edge_count));
    out.weight.resize(static_cast<std::size_t>(edge_count));
    out.edge_index.resize(static_cast<std::size_t>(edge_count));
    for (int n = 0; n < count; ++n) {
        int slot = out.row_start[static_cast<std::size_t>(n)];
        for (const Edge& edge : lists[static_cast<std::size_t>(n)]) {
            if (edge.from_index < 0 || edge.from_index >= count || edge.to_index < 0 ||
                edge.to_index >= count)
                throw std::invalid_argument(parameter_name +
                                            ": incoming list contains an out-of-range node");
            if (edge.index < 0)
                throw std::invalid_argument(parameter_name +
                                            ": incoming list contains a negative edge index");
            std::size_t k = static_cast<std::size_t>(slot++);
            out.from_node[k] = edge.from_index;
            out.to_node[k] = edge.to_index;
            out.weight[k] = edge.weight;
            out.edge_index[k] = edge.index;
        }
    }
    return out;
}

namespace dijkstra {

inline constexpr int NEXT_NODE = 0;
inline constexpr int EDGE_INDEX = 1;
inline constexpr int COST = 2;
using ResultTable = std::vector<std::array<float, 3>>;

namespace detail {

inline int node_count_from_edges(const std::vector<Edge>& edges) {
    if (edges.empty())
        throw std::invalid_argument(
            "The node count cannot be derived from an empty edge list; provide node_count "
            "explicitly.");
    int maximum = 0;
    for (const Edge& edge : edges) {
        if (edge.from_index > maximum) maximum = edge.from_index;
        if (edge.to_index > maximum) maximum = edge.to_index;
    }
    return maximum + 1;
}

inline int resolve_node_count(const std::vector<Edge>& edges, int node_count) {
    if (node_count == -1) return node_count_from_edges(edges);
    if (node_count < 1)
        throw std::out_of_range(
            "The node count must be positive, or -1 to derive it from the edges.");
    return node_count;
}

inline CompactAdjacency build_incoming_adjacency(
    const std::vector<Edge>& edges, int node_count,
    const std::vector<std::vector<Edge>>* provided_lists) {
    if (provided_lists != nullptr &&
        static_cast<int>(provided_lists->size()) == node_count)
        return CompactAdjacency::from_incoming_lists(*provided_lists, node_count, "edges");
    return CompactAdjacency::from_edges(edges, node_count, true, "edges");
}

}  // namespace detail

inline bool path_exists(const ResultTable& table, int node_index) {
    if (node_index < 0 || node_index >= static_cast<int>(table.size()))
        throw std::out_of_range("The node index is outside the result table.");
    const float cost = table[static_cast<std::size_t>(node_index)][COST];
    return !(std::isinf(cost) && cost > 0.0f);
}

inline bool try_get_path(const ResultTable& table, int start_node_index,
                         std::vector<int>& path_edge_indices, float& total_cost) {
    int rows = static_cast<int>(table.size());
    if (start_node_index < 0 || start_node_index >= rows)
        throw std::out_of_range("The start node index is outside the result table.");
    path_edge_indices.clear();
    total_cost = table[static_cast<std::size_t>(start_node_index)][COST];
    if (std::isinf(total_cost) && total_cost > 0.0f) return false;
    int node = start_node_index;
    int steps = 0;
    while (table[static_cast<std::size_t>(node)][EDGE_INDEX] >= 0.0f) {
        if (++steps > rows)
            throw std::invalid_argument(
                "The result table does not converge to a destination; it may be inconsistent.");
        path_edge_indices.push_back(
            static_cast<int>(table[static_cast<std::size_t>(node)][EDGE_INDEX]));
        int next = static_cast<int>(table[static_cast<std::size_t>(node)][NEXT_NODE]);
        if (next < 0 || next >= rows)
            throw std::invalid_argument(
                "The result table routes to a node outside the table; it may be inconsistent.");
        node = next;
    }
    if (table[static_cast<std::size_t>(node)][COST] != 0.0f)
        throw std::invalid_argument(
            "The result table walk ended away from a destination; it may be inconsistent.");
    return true;
}

inline std::optional<std::vector<int>> get_path(const ResultTable& table,
                                                int start_node_index) {
    std::vector<int> path;
    float cost = 0.0f;
    if (!try_get_path(table, start_node_index, path, cost)) return std::nullopt;
    return path;
}

inline void run_to_exhaustion(const CompactAdjacency& adjacency,
                              const std::vector<float>* weight_override,
                              std::vector<int>& next, std::vector<int>& edge_indexes,
                              std::vector<float>& dist, std::vector<int>& state,
                              IndexedMinHeap& heap) {
    while (heap.count() > 0) {
        int current = 0;
        float cost = 0.0f;
        heap.remove_min(current, cost);
        if (state[static_cast<std::size_t>(current)] == 1) continue;
        state[static_cast<std::size_t>(current)] = 1;
        int row_end = adjacency.row_start[static_cast<std::size_t>(current + 1)];
        for (int k = adjacency.row_start[static_cast<std::size_t>(current)]; k < row_end; ++k) {
            std::size_t slot = static_cast<std::size_t>(k);
            int from = adjacency.from_node[slot];
            float weight = weight_override == nullptr
                               ? adjacency.weight[slot]
                               : (*weight_override)[static_cast<std::size_t>(
                                     adjacency.source_position[slot])];
            float new_cost = cost + weight;
            if (new_cost < dist[static_cast<std::size_t>(from)]) {
                dist[static_cast<std::size_t>(from)] = new_cost;
                if (state[static_cast<std::size_t>(from)] != 2) {
                    heap.add(from, new_cost);
                    state[static_cast<std::size_t>(from)] = 2;
                } else {
                    heap.decrease_key(from, new_cost);
                }
                next[static_cast<std::size_t>(from)] = adjacency.to_node[slot];
                edge_indexes[static_cast<std::size_t>(from)] = adjacency.edge_index[slot];
            }
        }
    }
}

inline void solve_core(const CompactAdjacency& adjacency,
                       const std::vector<float>* weight_override, int destination_index,
                       std::vector<int>& next, std::vector<int>& edge_indexes,
                       std::vector<float>& dist, std::vector<int>& state,
                       IndexedMinHeap& heap) {
    int n = adjacency.node_count;
    next.assign(static_cast<std::size_t>(n), -1);
    edge_indexes.assign(static_cast<std::size_t>(n), -1);
    dist.assign(static_cast<std::size_t>(n), std::numeric_limits<float>::infinity());
    state.assign(static_cast<std::size_t>(n), 0);
    heap.clear();
    next[static_cast<std::size_t>(destination_index)] = destination_index;
    dist[static_cast<std::size_t>(destination_index)] = 0.0f;
    heap.add(destination_index, 0.0f);
    state[static_cast<std::size_t>(destination_index)] = 2;
    run_to_exhaustion(adjacency, weight_override, next, edge_indexes, dist, state, heap);
}

inline void solve_nearest_core(const CompactAdjacency& adjacency,
                               const std::vector<float>* weight_override,
                               const std::vector<int>& destinations,
                               std::vector<int>& next, std::vector<int>& edge_indexes,
                               std::vector<float>& dist, std::vector<int>& state,
                               IndexedMinHeap& heap) {
    int n = adjacency.node_count;
    next.assign(static_cast<std::size_t>(n), -1);
    edge_indexes.assign(static_cast<std::size_t>(n), -1);
    dist.assign(static_cast<std::size_t>(n), std::numeric_limits<float>::infinity());
    state.assign(static_cast<std::size_t>(n), 0);
    heap.clear();
    for (int destination : destinations) {
        if (state[static_cast<std::size_t>(destination)] == 2) continue;
        next[static_cast<std::size_t>(destination)] = destination;
        dist[static_cast<std::size_t>(destination)] = 0.0f;
        heap.add(destination, 0.0f);
        state[static_cast<std::size_t>(destination)] = 2;
    }
    run_to_exhaustion(adjacency, weight_override, next, edge_indexes, dist, state, heap);
}

inline void solve_merged_core(const CompactAdjacency& adjacency,
                              const std::vector<float>* weight_override,
                              const std::vector<int>& destinations,
                              std::vector<int>& next, std::vector<int>& edge_indexes,
                              std::vector<float>& dist, std::vector<int>& state,
                              IndexedMinHeap& heap, std::vector<int>& best_next,
                              std::vector<int>& best_edge, std::vector<float>& best_dist) {
    int n = adjacency.node_count;
    best_next.assign(static_cast<std::size_t>(n), -1);
    best_edge.assign(static_cast<std::size_t>(n), -1);
    best_dist.assign(static_cast<std::size_t>(n), std::numeric_limits<float>::infinity());
    for (int destination : destinations) {
        solve_core(adjacency, weight_override, destination, next, edge_indexes, dist, state, heap);
        for (int j = 0; j < n; ++j) {
            std::size_t i = static_cast<std::size_t>(j);
            if (dist[i] < best_dist[i]) {
                best_next[i] = next[i];
                best_edge[i] = edge_indexes[i];
                best_dist[i] = dist[i];
            }
        }
    }
}

inline ResultTable write_table(const std::vector<int>& next,
                               const std::vector<int>& edge_indexes,
                               const std::vector<float>& dist) {
    ResultTable table(next.size());
    for (std::size_t i = 0; i < next.size(); ++i)
        table[i] = {static_cast<float>(next[i]), static_cast<float>(edge_indexes[i]), dist[i]};
    return table;
}

inline void validate_destinations(const std::vector<int>& destinations, int node_count,
                                  bool require_nonempty) {
    if (require_nonempty && destinations.empty())
        throw std::invalid_argument("At least one destination index is required.");
    for (int destination : destinations)
        if (destination < 0 || destination >= node_count)
            throw std::out_of_range("A destination index is outside the network.");
}

inline ResultTable solve(const std::vector<Edge>& edges, int destination_index,
                         int node_count = -1,
                         const std::vector<std::vector<Edge>>* incoming = nullptr) {
    int n = detail::resolve_node_count(edges, node_count);
    validate_destinations({destination_index}, n, true);
    CompactAdjacency adjacency = detail::build_incoming_adjacency(edges, n, incoming);
    std::vector<int> next, edge_indexes, state;
    std::vector<float> dist;
    IndexedMinHeap heap(n);
    solve_core(adjacency, nullptr, destination_index, next, edge_indexes, dist, state, heap);
    return write_table(next, edge_indexes, dist);
}

inline ResultTable solve(const std::vector<Edge>& edges, const std::vector<int>& destinations,
                         int node_count = -1,
                         const std::vector<std::vector<Edge>>* incoming = nullptr) {
    int n = detail::resolve_node_count(edges, node_count);
    validate_destinations(destinations, n, false);
    CompactAdjacency adjacency = detail::build_incoming_adjacency(edges, n, incoming);
    std::vector<int> next, edge_indexes, state, best_next, best_edge;
    std::vector<float> dist, best_dist;
    IndexedMinHeap heap(n);
    solve_merged_core(adjacency, nullptr, destinations, next, edge_indexes, dist, state, heap,
                      best_next, best_edge, best_dist);
    return write_table(best_next, best_edge, best_dist);
}

inline ResultTable solve_nearest(const std::vector<Edge>& edges,
                                 const std::vector<int>& destinations,
                                 int node_count = -1) {
    int n = detail::resolve_node_count(edges, node_count);
    validate_destinations(destinations, n, true);
    CompactAdjacency adjacency = CompactAdjacency::from_edges(edges, n, true, "edges");
    std::vector<int> next, edge_indexes, state;
    std::vector<float> dist;
    IndexedMinHeap heap(n);
    solve_nearest_core(adjacency, nullptr, destinations, next, edge_indexes, dist, state, heap);
    return write_table(next, edge_indexes, dist);
}

}  // namespace dijkstra
}  // namespace corehydro::numerics::math::optimization
