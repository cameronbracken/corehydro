// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `network` group's dispatch arms over the dynamic-programming trio
// (numerics/math/optimization/dynamic/{binary_heap,dijkstra,network}.hpp). Unlike every other
// toolbox group, this one's subject is a GRAPH rather than a series, so the toolbox convention
// (flat double vectors as `data`, scalars and flags in `options`) carries the edge list as FOUR
// parallel data vectors -- `[from, to, weight, index]`, all of the same length, one element per
// edge -- with the destination node indices in `options.destinations`.
//
// PRECISION, and it is load-bearing (see dijkstra.hpp note 1): the ported layer accumulates edge
// costs in `float`, because C# does. This header is the boundary where the widening happens: the
// incoming `weight` doubles are narrowed to `float` on the way in, and the result table's floats
// are widened back to double on the way out. `from`, `to` and `index` are whole numbers carried
// through a double vector and cast back to `int`, which is exact for every graph a caller can
// realistically build (a double holds every integer below 2^53) but is checked anyway, since
// silently truncating a node index would corrupt a graph rather than fail.
//
// The three methods:
//
//   dijkstra              the free solver. Uses the SINGLE-destination C# overload when exactly
//                         one destination is given and the multi-destination one otherwise,
//                         mirroring which overload DijkstraTesting.cs calls for each of its
//                         graphs. The two agree element for element on one destination (the
//                         multi form seeds an all-infinite table and keeps whichever partial
//                         result is cheaper, and one partial result is cheaper than infinity
//                         exactly where it is finite), so this is a fidelity choice, not a
//                         behavioral one.
//   network_solve         the same solve routed through `Network`, which caches the incoming-edge
//                         lists in its constructor and hands them to the free solver.
//   network_solve_weights `Network::Solve(float[] edgeWeights)`. Preserved as an oracle of an
//                         UPSTREAM DEFECT, not offered as a feature: the custom weights have no
//                         effect, because the re-weighted edge array is passed alongside the
//                         stale cached incoming-edge lists, which is where the solver reads its
//                         weights from. It therefore returns the ORIGINAL-weight table. See
//                         network.hpp note 5 and docs/upstream-csharp-issues.md. The user-facing
//                         `shortest_path()` verb in both packages does NOT call this method for
//                         exactly that reason -- a caller who wants different weights passes
//                         them as `weight`.
//
// `node_count` is an option only on `dijkstra` (C#'s own optional `nNodes` parameter, default
// -1 meaning "derive it from the edge list"). `Network` derives its own node count in its
// constructor and its `Solve` overloads have no such parameter, so supplying `node_count` to
// either network_* method is an error rather than a silently ignored key.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/optimization/dynamic/dijkstra.hpp"
#include "corehydro/numerics/math/optimization/dynamic/network.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

namespace nwopt = corehydro::numerics::math::optimization;

// A whole number carried across the toolbox boundary in a double vector, checked rather than
// silently truncated (see this header's precision note).
inline int network_index(const std::string& method, const char* what, double v) {
    if (!(std::floor(v) == v) || !std::isfinite(v))
        throw std::runtime_error("toolbox method 'network." + method + "' needs a whole number " +
                                 "for " + what + "; got " + std::to_string(v));
    if (v < 0)
        throw std::runtime_error("toolbox method 'network." + method + "' needs a non-negative " +
                                 what + "; got " + std::to_string(v));
    return static_cast<int>(v);
}

// Reassembles the edge list from the four parallel data vectors.
inline std::vector<nwopt::Edge> to_edges(const std::string& method,
                                         const std::vector<std::vector<double>>& data) {
    const std::vector<double>& from = data_at(data, 0, "network", method);
    const std::vector<double>& to = data_at(data, 1, "network", method);
    const std::vector<double>& weight = data_at(data, 2, "network", method);
    const std::vector<double>& index = data_at(data, 3, "network", method);
    if (to.size() != from.size() || weight.size() != from.size() || index.size() != from.size())
        throw std::runtime_error("toolbox method 'network." + method +
                                 "' needs from/to/weight/index vectors of the same length; got " +
                                 std::to_string(from.size()) + ", " + std::to_string(to.size()) +
                                 ", " + std::to_string(weight.size()) + ", " +
                                 std::to_string(index.size()));
    if (from.empty())
        throw std::runtime_error("toolbox method 'network." + method +
                                 "' needs at least one edge");
    std::vector<nwopt::Edge> edges;
    edges.reserve(from.size());
    for (std::size_t i = 0; i < from.size(); ++i) {
        edges.emplace_back(network_index(method, "from", from[i]),
                           network_index(method, "to", to[i]),
                           static_cast<float>(weight[i]),
                           network_index(method, "index", index[i]));
    }
    return edges;
}

// The `destinations` option: a JSON array of node indices, or a bare number for one.
inline std::vector<int> to_destinations(const std::string& method, const JsonValue& options) {
    if (!options.contains("destinations"))
        throw std::runtime_error("toolbox method 'network." + method +
                                 "' needs a 'destinations' option");
    const JsonValue& d = options.at("destinations");
    std::vector<int> out;
    if (d.type() == JsonValue::Type::Array) {
        for (const JsonValue& v : d.items())
            out.push_back(network_index(method, "destination", v.as_double()));
    } else {
        out.push_back(network_index(method, "destination", d.as_double()));
    }
    if (out.empty())
        throw std::runtime_error("toolbox method 'network." + method +
                                 "' needs at least one destination");
    return out;
}

// Serializes the result table back to a ToolboxResult, row-major, with `dims = {n_nodes, 3}` and
// the column order NEXT_NODE / EDGE_INDEX / COST -- the C# column order, unchanged.
inline ToolboxResult result_table_result(const nwopt::dijkstra::ResultTable& table) {
    ToolboxResult r;
    r.dims = {static_cast<int>(table.size()), 3};
    r.values.reserve(table.size() * 3);
    for (const auto& row : table)
        for (int c = 0; c < 3; ++c) r.values.push_back(static_cast<double>(row[c]));
    return r;
}

// C# indexes the result table with the destination index and lets the CLR raise
// IndexOutOfRangeException when it is past the end; `std::vector::operator[]` would be undefined
// behavior instead, so the boundary checks it. `n_nodes` is whatever the chosen method will
// derive: the `node_count` option for `dijkstra`, and `max(from, to) + 1` otherwise.
inline void check_destinations(const std::string& method, const std::vector<int>& destinations,
                               int n_nodes) {
    for (int d : destinations)
        if (d >= n_nodes)
            throw std::runtime_error("toolbox method 'network." + method + "' destination " +
                                     std::to_string(d) + " is out of range for a network of " +
                                     std::to_string(n_nodes) + " nodes");
}

inline ToolboxResult run_network(const std::string& method,
                                 const std::vector<std::vector<double>>& data,
                                 const JsonValue& options) {
    std::vector<nwopt::Edge> edges = to_edges(method, data);
    std::vector<int> destinations = to_destinations(method, options);

    if (method == "dijkstra") {
        int node_count = options.value_or("node_count", -1);
        check_destinations(method, destinations,
                           node_count == -1 ? nwopt::dijkstra::detail::node_count_from_edges(edges)
                                            : node_count);
        if (destinations.size() == 1)
            return result_table_result(
                nwopt::dijkstra::solve(edges, destinations[0], node_count));
        return result_table_result(nwopt::dijkstra::solve(edges, destinations, node_count));
    }

    if (method == "network_solve" || method == "network_solve_weights") {
        if (options.contains("node_count"))
            throw std::runtime_error(
                "toolbox method 'network." + method +
                "' takes no 'node_count' option: Network derives its own node count from the "
                "edge list");
        nwopt::Network network(edges, destinations);
        check_destinations(method, destinations, network.node_count());

        if (method == "network_solve") {
            if (destinations.size() == 1) return result_table_result(network.solve(destinations[0]));
            return result_table_result(network.solve(destinations));
        }

        // network_solve_weights: the quirk-preserving arm (see this header's method list).
        const std::vector<double>& weights = data_at(data, 4, "network", method);
        if (weights.size() != edges.size())
            throw std::runtime_error("toolbox method 'network.network_solve_weights' needs one "
                                     "edge weight per edge; got " +
                                     std::to_string(weights.size()) + " for " +
                                     std::to_string(edges.size()) + " edges");
        std::vector<float> edge_weights(weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i)
            edge_weights[i] = static_cast<float>(weights[i]);
        return result_table_result(network.solve(edge_weights));
    }

    throw std::runtime_error("unknown network method: " + method);
}

}  // namespace corehydro::numerics::support::detail
