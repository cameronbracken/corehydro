// ported from: Numerics/Mathematics/Optimization/Dynamic/Network.cs @ 7e8e8d1
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/optimization/dynamic/dijkstra.hpp"

namespace corehydro::numerics::math::optimization {

// Compiled network topology for repeated shortest-path solves. Edges and destinations are copied
// at construction, and custom weights overlay the original edge order without rebuilding.
class Network {
   public:
    Network(std::vector<Edge> edges, std::vector<int> destination_indices)
        : destination_indices_(std::move(destination_indices)), edges_(std::move(edges)) {
        if (edges_.empty()) throw std::invalid_argument("At least one edge is required.");
        if (destination_indices_.empty())
            throw std::invalid_argument("At least one destination index is required.");
        int maximum = 0;
        for (const Edge& edge : edges_) {
            if (edge.from_index > maximum) maximum = edge.from_index;
            if (edge.to_index > maximum) maximum = edge.to_index;
        }
        node_count_ = maximum + 1;
        incoming_adjacency_ = CompactAdjacency::from_edges(edges_, node_count_, true, "edges");
        outgoing_adjacency_ = CompactAdjacency::from_edges(edges_, node_count_, false, "edges");
        is_destination_.assign(static_cast<std::size_t>(node_count_), false);
        for (int destination : destination_indices_) {
            if (destination < 0 || destination >= node_count_)
                throw std::out_of_range("A destination index is outside the network.");
            is_destination_[static_cast<std::size_t>(destination)] = true;
        }
        incoming_edges_ = materialize_lists(incoming_adjacency_);
        outgoing_edges_ = materialize_lists(outgoing_adjacency_);
    }

    const std::vector<int>& destination_indices() const { return destination_indices_; }
    const std::vector<std::vector<Edge>>& incoming_edges() const { return incoming_edges_; }
    const std::vector<std::vector<Edge>>& outgoing_edges() const { return outgoing_edges_; }
    int node_count() const { return node_count_; }

    dijkstra::ResultTable solve(int destination_index) const {
        validate_destinations({destination_index});
        std::vector<int> next, edge_indexes, state;
        std::vector<float> dist;
        IndexedMinHeap heap(node_count_);
        dijkstra::solve_core(incoming_adjacency_, nullptr, destination_index, next, edge_indexes,
                             dist, state, heap);
        return dijkstra::write_table(next, edge_indexes, dist);
    }

    dijkstra::ResultTable solve(const std::vector<int>& destinations) const {
        validate_destinations(destinations);
        std::vector<int> next, edge_indexes, state, best_next, best_edge;
        std::vector<float> dist, best_dist;
        IndexedMinHeap heap(node_count_);
        dijkstra::solve_merged_core(incoming_adjacency_, nullptr, destinations, next,
                                    edge_indexes, dist, state, heap, best_next, best_edge,
                                    best_dist);
        return dijkstra::write_table(best_next, best_edge, best_dist);
    }

    dijkstra::ResultTable solve(const std::vector<float>& edge_weights) const {
        validate_weights(edge_weights);
        std::vector<int> next, edge_indexes, state, best_next, best_edge;
        std::vector<float> dist, best_dist;
        IndexedMinHeap heap(node_count_);
        dijkstra::solve_merged_core(incoming_adjacency_, &edge_weights, destination_indices_,
                                    next, edge_indexes, dist, state, heap, best_next, best_edge,
                                    best_dist);
        return dijkstra::write_table(best_next, best_edge, best_dist);
    }

    dijkstra::ResultTable solve_nearest() const {
        std::vector<int> next, edge_indexes, state;
        std::vector<float> dist;
        IndexedMinHeap heap(node_count_);
        dijkstra::solve_nearest_core(incoming_adjacency_, nullptr, destination_indices_, next,
                                     edge_indexes, dist, state, heap);
        return dijkstra::write_table(next, edge_indexes, dist);
    }

    dijkstra::ResultTable solve_nearest(const std::vector<float>& edge_weights) const {
        validate_weights(edge_weights);
        std::vector<int> next, edge_indexes, state;
        std::vector<float> dist;
        IndexedMinHeap heap(node_count_);
        dijkstra::solve_nearest_core(incoming_adjacency_, &edge_weights, destination_indices_,
                                     next, edge_indexes, dist, state, heap);
        return dijkstra::write_table(next, edge_indexes, dist);
    }

    std::optional<std::vector<int>> get_path(const std::vector<int>& edges_to_remove,
                                             int start_node_index) const {
        validate_start(start_node_index);
        std::unordered_set<int> removed(edges_to_remove.begin(), edges_to_remove.end());
        return find_detour_path(removed, start_node_index, nullptr);
    }

    std::vector<int> get_path(const std::vector<int>& edges_to_remove, int start_node_index,
                              const dijkstra::ResultTable& existing_results) const {
        return get_path_impl(edges_to_remove, start_node_index, existing_results, nullptr);
    }

    std::vector<int> get_path(const std::vector<int>& edges_to_remove, int start_node_index,
                              const dijkstra::ResultTable& existing_results,
                              const std::vector<float>& edge_weights) const {
        validate_weights(edge_weights);
        return get_path_impl(edges_to_remove, start_node_index, existing_results, &edge_weights);
    }

   private:
    static std::vector<std::vector<Edge>> materialize_lists(
        const CompactAdjacency& adjacency) {
        std::vector<std::vector<Edge>> lists(static_cast<std::size_t>(adjacency.node_count));
        for (int n = 0; n < adjacency.node_count; ++n) {
            int start = adjacency.row_start[static_cast<std::size_t>(n)];
            int end = adjacency.row_start[static_cast<std::size_t>(n + 1)];
            auto& list = lists[static_cast<std::size_t>(n)];
            list.reserve(static_cast<std::size_t>(end - start));
            for (int k = start; k < end; ++k) {
                std::size_t slot = static_cast<std::size_t>(k);
                list.emplace_back(adjacency.from_node[slot], adjacency.to_node[slot],
                                  adjacency.weight[slot], adjacency.edge_index[slot]);
            }
        }
        return lists;
    }

    void validate_destinations(const std::vector<int>& destinations) const {
        for (int destination : destinations)
            if (destination < 0 || destination >= node_count_)
                throw std::out_of_range("A destination index is outside the network.");
    }

    void validate_weights(const std::vector<float>& weights) const {
        if (weights.size() != edges_.size())
            throw std::invalid_argument("The weight count must equal the edge count.");
    }

    void validate_start(int start_node_index) const {
        if (start_node_index < 0 || start_node_index >= node_count_)
            throw std::out_of_range("The start node index is outside the network.");
    }

    void validate_table(const dijkstra::ResultTable& table) const {
        if (static_cast<int>(table.size()) != node_count_)
            throw std::invalid_argument("The result table has the wrong row count.");
    }

    std::vector<int> get_path_impl(const std::vector<int>& edges_to_remove,
                                   int start_node_index,
                                   const dijkstra::ResultTable& existing_results,
                                   const std::vector<float>* edge_weights) const {
        validate_table(existing_results);
        validate_start(start_node_index);
        const float existing_cost =
            existing_results[static_cast<std::size_t>(start_node_index)][2];
        if (std::isinf(existing_cost) && existing_cost > 0.0f)
            return {};
        std::unordered_set<int> removed(edges_to_remove.begin(), edges_to_remove.end());
        auto recorded = dijkstra::get_path(existing_results, start_node_index);
        if (recorded.has_value()) {
            bool blocked = false;
            for (int edge : *recorded)
                if (removed.find(edge) != removed.end()) {
                    blocked = true;
                    break;
                }
            if (!blocked) return *recorded;
        }
        auto detour = find_detour_path(removed, start_node_index, edge_weights);
        return detour.value_or(std::vector<int>{});
    }

    std::optional<std::vector<int>> find_detour_path(
        const std::unordered_set<int>& removed, int start_node_index,
        const std::vector<float>* edge_weights) const {
        if (is_destination_[static_cast<std::size_t>(start_node_index)])
            return std::vector<int>{};
        std::vector<float> dist(static_cast<std::size_t>(node_count_),
                                std::numeric_limits<float>::infinity());
        std::vector<int> state(static_cast<std::size_t>(node_count_), 0);
        std::vector<int> previous_slot(static_cast<std::size_t>(node_count_), -1);
        IndexedMinHeap heap(node_count_);
        dist[static_cast<std::size_t>(start_node_index)] = 0.0f;
        heap.add(start_node_index, 0.0f);
        state[static_cast<std::size_t>(start_node_index)] = 2;

        int reached = -1;
        while (heap.count() > 0) {
            int current = 0;
            float cost = 0.0f;
            heap.remove_min(current, cost);
            if (state[static_cast<std::size_t>(current)] == 1) continue;
            state[static_cast<std::size_t>(current)] = 1;
            if (is_destination_[static_cast<std::size_t>(current)]) {
                reached = current;
                break;
            }
            int end = outgoing_adjacency_.row_start[static_cast<std::size_t>(current + 1)];
            for (int k = outgoing_adjacency_.row_start[static_cast<std::size_t>(current)];
                 k < end; ++k) {
                std::size_t slot = static_cast<std::size_t>(k);
                if (removed.find(outgoing_adjacency_.edge_index[slot]) != removed.end()) continue;
                int to = outgoing_adjacency_.to_node[slot];
                float weight = edge_weights == nullptr
                                   ? outgoing_adjacency_.weight[slot]
                                   : (*edge_weights)[static_cast<std::size_t>(
                                         outgoing_adjacency_.source_position[slot])];
                float new_cost = cost + weight;
                if (new_cost < dist[static_cast<std::size_t>(to)]) {
                    dist[static_cast<std::size_t>(to)] = new_cost;
                    if (state[static_cast<std::size_t>(to)] != 2) {
                        heap.add(to, new_cost);
                        state[static_cast<std::size_t>(to)] = 2;
                    } else {
                        heap.decrease_key(to, new_cost);
                    }
                    previous_slot[static_cast<std::size_t>(to)] = k;
                }
            }
        }
        if (reached < 0) return std::nullopt;
        std::vector<int> path;
        int node = reached;
        while (node != start_node_index) {
            int slot_value = previous_slot[static_cast<std::size_t>(node)];
            std::size_t slot = static_cast<std::size_t>(slot_value);
            path.push_back(outgoing_adjacency_.edge_index[slot]);
            node = outgoing_adjacency_.from_node[slot];
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    std::vector<std::vector<Edge>> outgoing_edges_;
    std::vector<std::vector<Edge>> incoming_edges_;
    int node_count_ = 0;
    std::vector<int> destination_indices_;
    std::vector<Edge> edges_;
    CompactAdjacency incoming_adjacency_;
    CompactAdjacency outgoing_adjacency_;
    std::vector<bool> is_destination_;
};

}  // namespace corehydro::numerics::math::optimization
