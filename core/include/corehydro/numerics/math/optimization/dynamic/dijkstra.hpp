// ported from: Numerics/Mathematics/Optimization/Dynamic/Dijkstra.cs @ 2a0357a
//
// The `Edge` struct and Dijkstra's shortest-path solver, run BACKWARDS from a destination: the
// result table answers "from node i, which neighbour do I step to, along which edge, at what
// remaining cost" for every node at once, which is what a routing consumer wants.
//
// Transcription notes:
//
// 1. WEIGHTS ARE `float`, NOT `double`. C# declares `float Weight`, allocates `float[nNodes, 3]`
//    and accumulates `float newCost = cost + edge.Weight`, and its tests assert the table by
//    EXACT equality (`Assert.AreEqual(3f, result[2, 2])`), so the port keeps `float` end to end;
//    widening to double belongs at the toolbox boundary, not here. This is not a theoretical
//    concern: MEASURED against the real Numerics library at 2a0357a, a 10-hop chain of 0.1f edges
//    reports node costs of 0.90000009536743164, 0.80000007152557373, ... , which the port
//    reproduces bit-for-bit and a double-precision port would not (see the SUPPLEMENT block in
//    core/tests/test_network_optimization.cpp, which pins that graph).
//
// 2. The result table is C#'s `float[nNodes, 3]` -> `std::vector<std::array<float, 3>>`, with the
//    same column order. C# keeps NEXT_NODE / EDGE_INDEX / COST as PRIVATE consts and its tests
//    index with bare literals; the port exposes them (still by the C# names) because the
//    ported consumers next door -- Network and the toolbox group -- and the tests here read the
//    table by column, and a named column is the difference between a readable port and three
//    magic numbers.
//
// 3. `List<Edge>[] edgesToNodes` -> `std::vector<std::vector<Edge>>` passed by const pointer, so
//    the caller-supplied cache stays optional exactly as C#'s `= null` default is. C#'s per-node
//    entries are null until first appended to and the loop guards `if (edgesToNodes[current] ==
//    null) continue;`; an empty vector iterates zero times, so the port needs no guard and the
//    behavior is identical.
//
// 4. C# REASSIGNS the `edgesToNodes` parameter when it is null or the wrong length. A const
//    pointer parameter cannot be reassigned, so the port builds a local and re-seats a
//    `resolved` pointer at it. Same effect, same condition (`== null || Length != nNodes`).
//
// 5. Both overloads build their cache from `edge.ToIndex` -- INCLUDING the multi-destination one,
//    whose parameter is named `edgesFromNodes` while it is populated with incoming edges exactly
//    like the single-destination `edgesToNodes`. The naming is inconsistent upstream; the
//    behavior is not, and the port transcribes the behavior as written rather than "fixing" a
//    parameter name that a caller may be passing by position.
//
// 6. severed: `Console.WriteLine($"Node{i} is unreachable from destination {destinationIndex}")`,
//    the trailing loop over unreached nodes in the single-destination overload. It is a console
//    diagnostic with no effect on the returned table, and a library that prints to stdout is not
//    something an R or Python caller can be handed. The unreachable state remains fully
//    observable through the table's positive-infinity COST column (that is what `path_exists`
//    reads, and what four of the eight C# tests assert).
//
// 7. `edges.Max(o => Math.Max(o.FromIndex, o.ToIndex)) + 1` computes the node count when the
//    caller passes -1. LINQ `Max` on an empty sequence throws InvalidOperationException
//    ("Sequence contains no elements."); the port mirrors that as std::runtime_error carrying the
//    same message rather than silently returning an empty table.
//
// 8. The heap is constructed with a hard-coded capacity of 10000 in C#, independent of the node
//    count. Transcribed as written -- a graph with more than 10000 nodes reachable in one pass
//    would throw "Heap is full." there exactly as it does here.
//
// 9. A `node_count` SMALLER than the graph needs is a CHECKED error, not undefined behavior. C#
//    allocates every array at `nNodes` and then indexes them by node index without checking --
//    `edgesToNodes[edge.ToIndex]`, `resultTable[destinationIndex, ...]`,
//    `nodeWeightToDestination[from]` -- so the CLR raises IndexOutOfRangeException ("Index was
//    outside the bounds of the array."). `std::vector::operator[]` would be undefined behavior
//    instead, which is a silent wrong answer at best, so `detail::checked_node_index` guards
//    every one of those sites and throws std::out_of_range carrying the C# message.
//
//    The guard is IN PLACE, at each indexing site, and deliberately NOT hoisted into one up-front
//    sweep of the edge list, because C#'s throw is LAZY: an out-of-range index on an edge the
//    search never relaxes never throws. MEASURED against the real Numerics library at 2a0357a
//    (probe: /tmp/getpath_probe):
//      Solve({(0,1),(1,5)}, 0, nodeCount: 2)      THREW System.IndexOutOfRangeException
//      Solve({(0,1),(1,5)}, new[]{0}, 2)          THREW System.IndexOutOfRangeException
//      Solve({(5,1),(0,1)}, 1, nodeCount: 2)      THREW System.IndexOutOfRangeException
//      Solve({(0,1)},       5, nodeCount: 2)      THREW System.IndexOutOfRangeException
//      Solve({(0,1),(7,0)}, 1, nodeCount: 2)      THREW System.IndexOutOfRangeException
//      Solve({(0,1),(7,1)}, 0, nodeCount: 2)      RETURNED [(0,-1,0), (-1,-1,Inf)]
//    The last line is the lazy case: node 1 is never reached from destination 0, so the edge
//    carrying the out-of-range FromIndex 7 is never relaxed and C# returns a table. An up-front
//    sweep would reject it, which is why there is not one here. The R and Python `shortest_path()`
//    wrappers ARE stricter (they reject any `node_count` below `max(from, to) + 1` up front, so
//    the user gets a message naming the argument rather than this one from inside the solver);
//    that asymmetry is deliberate and documented on both of them.
#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/optimization/dynamic/binary_heap.hpp"

namespace corehydro::numerics::math::optimization {

// Struct that represents an edge in a network. An edge contains information on the start node,
// end node, edge weight, and edge index.
struct Edge {
    // Node index at start of edge.
    int from_index;
    // Node index at end of edge.
    int to_index;
    // Weight (or Cost) of transversing the edge.
    float weight;
    // Index of the edge, often used as an index to the edge source (e.g., road segment).
    int index;

    // C# structs are default-initialized (all fields zero); see binary_heap.hpp note 3.
    Edge() : from_index(0), to_index(0), weight(0.0f), index(0) {}

    Edge(int from_node_index, int to_node_index, float edge_weight, int edge_index)
        : from_index(from_node_index),
          to_index(to_node_index),
          weight(edge_weight),
          index(edge_index) {}
};

// C# `public static class Dijkstra`; a namespace of free functions is the C++ spelling of a
// static class.
namespace dijkstra {

// C#'s three private column constants, by their C# names (see dijkstra.hpp note 2).
inline constexpr int NEXT_NODE = 0;
inline constexpr int EDGE_INDEX = 1;
inline constexpr int COST = 2;

// The result table type: C#'s `float[nNodes, 3]`.
using ResultTable = std::vector<std::array<float, 3>>;

namespace detail {

// Stands in for the CLR's own bounds check on every `nNodes`-sized array C# indexes by a node
// index; see note 9 for the measured C# behavior this reproduces, including why it is called at
// each indexing site rather than once up front.
inline std::size_t checked_node_index(int index, int n_nodes) {
    if (index < 0 || index >= n_nodes) {
        throw std::out_of_range("Index was outside the bounds of the array.");
    }
    return static_cast<std::size_t>(index);
}

// C# `edges.Max(o => Math.Max(o.FromIndex, o.ToIndex)) + 1`; see note 7.
inline int node_count_from_edges(const std::vector<Edge>& edges) {
    if (edges.empty()) throw std::runtime_error("Sequence contains no elements.");
    int max_index = edges[0].from_index > edges[0].to_index ? edges[0].from_index
                                                            : edges[0].to_index;
    for (const auto& edge : edges) {
        if (edge.from_index > max_index) max_index = edge.from_index;
        if (edge.to_index > max_index) max_index = edge.to_index;
    }
    return max_index + 1;
}

// C#'s `edgesToNodes[edge.ToIndex] ??= new List<Edge>(); ...Add(edge);` build loop, shared by
// both overloads (which write it out identically).
inline std::vector<std::vector<Edge>> build_edges_to_nodes(const std::vector<Edge>& edges,
                                                           int n_nodes) {
    std::vector<std::vector<Edge>> edges_to_nodes(
        static_cast<std::size_t>(n_nodes > 0 ? n_nodes : 0));
    for (const auto& edge : edges) {
        edges_to_nodes[checked_node_index(edge.to_index, n_nodes)].push_back(edge);
    }
    return edges_to_nodes;
}

}  // namespace detail

// May be a useful call in LifeSim -> GetPath(). Follows the logic that is implemented in the
// Solve method.
inline bool path_exists(const ResultTable& result_table, int node_index) {
    const float cost =
        result_table[detail::checked_node_index(node_index,
                                                static_cast<int>(result_table.size()))][COST];
    // C# `!float.IsPositiveInfinity(...)`.
    return !(std::isinf(cost) && cost > 0.0f);
}

// Solves the shortest path from every node in the network of edges to a given destination.
//   edges           Edges, or segments, that make up the network.
//   destination_index  Index of the destination node.
//   node_count      Optional number of nodes in the network. If not provided (-1) it will be
//                   calculated internally.
//   edges_to_nodes  Optional list of incoming edges from each node in the network. If not
//                   provided or mismatched with edges it will be calculated internally.
// Returns a lookup table of shortest paths from any given node.
inline ResultTable solve(const std::vector<Edge>& edges, int destination_index,
                         int node_count = -1,
                         const std::vector<std::vector<Edge>>* edges_to_nodes = nullptr) {
    // Set optional parameters if required.
    const int n_nodes = (node_count == -1) ? detail::node_count_from_edges(edges) : node_count;

    std::vector<std::vector<Edge>> local_edges_to_nodes;
    const std::vector<std::vector<Edge>>* resolved = edges_to_nodes;
    if (resolved == nullptr || static_cast<int>(resolved->size()) != n_nodes) {
        local_edges_to_nodes = detail::build_edges_to_nodes(edges, n_nodes);
        resolved = &local_edges_to_nodes;
    }

    // Prepare results table with destination defined.
    const std::size_t table_rows = static_cast<std::size_t>(n_nodes > 0 ? n_nodes : 0);
    ResultTable result_table(table_rows);
    // 0 - Node hasn't been scanned yet, 1 - Node has been solved for, 2 - Node has been scanned
    // into heap but not solved for.
    std::vector<int> node_state(table_rows, 0);
    std::vector<float> node_weight_to_destination(table_rows);

    // Initialize all nodes are unreachable
    for (int i = 0; i < n_nodes; i++) {
        auto& row = result_table[static_cast<std::size_t>(i)];
        row[NEXT_NODE] = -1;
        row[EDGE_INDEX] = -1;
        row[COST] = std::numeric_limits<float>::infinity();
        node_weight_to_destination[static_cast<std::size_t>(i)] =
            std::numeric_limits<float>::infinity();
    }

    BinaryHeap<Edge> heap(10000);

    const std::size_t destination = detail::checked_node_index(destination_index, n_nodes);
    auto& destination_row = result_table[destination];
    destination_row[NEXT_NODE] = static_cast<float>(destination_index);  // Tail
    destination_row[EDGE_INDEX] = -1;                                    // edge index
    destination_row[COST] = 0;                                           // Cumulative Weight
    node_weight_to_destination[destination] = 0;
    heap.add(BinaryHeap<Edge>::Node(0, destination_index,
                                    Edge(destination_index, destination_index, 0, -1)));
    node_state[destination] = 2;

    while (heap.count() > 0) {
        auto node = heap.remove_min();
        int current = node.index;
        float cost = node.weight;

        if (node_state[static_cast<std::size_t>(current)] == 1) continue;

        node_state[static_cast<std::size_t>(current)] = 1;

        for (const auto& edge : (*resolved)[static_cast<std::size_t>(current)]) {
            int from = edge.from_index;
            int to = edge.to_index;
            float new_cost = cost + edge.weight;

            // C# reads `nodeWeightToDestination[from]` here unchecked; see note 9. This is the
            // site whose lazy throw the up-front alternative would get wrong.
            const std::size_t from_node = detail::checked_node_index(from, n_nodes);
            if (new_cost < node_weight_to_destination[from_node]) {
                node_weight_to_destination[from_node] = new_cost;
                auto new_node = BinaryHeap<Edge>::Node(new_cost, from, edge);

                if (node_state[from_node] != 2) {
                    heap.add(new_node);
                    node_state[from_node] = 2;
                } else {
                    heap.decrease_key(new_node);
                }

                auto& row = result_table[from_node];
                row[NEXT_NODE] = static_cast<float>(to);
                row[EDGE_INDEX] = static_cast<float>(edge.index);
                row[COST] = new_cost;
            }
        }
    }
    // severed here: the C# unreachable-node Console.WriteLine loop (see note 6).
    return result_table;
}

// Solves the shortest path from every node in the network of edges to a set of destinations,
// keeping the cheaper of the per-destination paths.
//   edges              Edges, or segments, that make up the network.
//   destination_indices  Indices of the destination nodes.
//   node_count         Optional number of nodes in the network.
//   edges_from_nodes   Optional list of incoming edges from each node (upstream's parameter name;
//                      see note 5).
inline ResultTable solve(const std::vector<Edge>& edges,
                         const std::vector<int>& destination_indices, int node_count = -1,
                         const std::vector<std::vector<Edge>>* edges_from_nodes = nullptr) {
    // Set optional parameters if required.
    const int n_nodes = (node_count == -1) ? detail::node_count_from_edges(edges) : node_count;

    std::vector<std::vector<Edge>> local_edges_from_nodes;
    const std::vector<std::vector<Edge>>* resolved = edges_from_nodes;
    if (resolved == nullptr || static_cast<int>(resolved->size()) != n_nodes) {
        local_edges_from_nodes = detail::build_edges_to_nodes(edges, n_nodes);
        resolved = &local_edges_from_nodes;
    }

    ResultTable result_table(static_cast<std::size_t>(n_nodes > 0 ? n_nodes : 0));
    for (int i = 0; i < n_nodes; i++) {
        auto& row = result_table[static_cast<std::size_t>(i)];
        row[NEXT_NODE] = -1;
        row[EDGE_INDEX] = -1;
        row[COST] = std::numeric_limits<float>::infinity();
    }

    for (std::size_t i = 0; i < destination_indices.size(); i++) {
        int destination_index = destination_indices[i];
        auto partial_result = solve(edges, destination_index, n_nodes, resolved);
        for (int j = 0; j < n_nodes; j++) {
            // Keep better path
            auto& row = result_table[static_cast<std::size_t>(j)];
            const auto& partial_row = partial_result[static_cast<std::size_t>(j)];
            if (partial_row[COST] < row[COST]) {
                row[NEXT_NODE] = partial_row[NEXT_NODE];
                row[EDGE_INDEX] = partial_row[EDGE_INDEX];
                row[COST] = partial_row[COST];
            }
        }
    }
    return result_table;
}

}  // namespace dijkstra

}  // namespace corehydro::numerics::math::optimization
