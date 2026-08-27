// ported from: Numerics/Mathematics/Optimization/Dynamic/Network.cs @ 2a0357a
//
// A network of edges used for shortest path optimization applications: it caches the incoming and
// outgoing edge lists once and forwards the three `Solve` overloads to `dijkstra::solve` over that
// cache, and it carries the two `GetPath` alternate-route methods.
//
// READ THIS BEFORE CHANGING ANYTHING HERE. The shipped C# `Network` is unreachable code: EVERY
// constructor call throws, so no C# caller can ever have used the class and no upstream test
// covers it (Test_Numerics has no Network test class; DijkstraTesting.cs's class is
// `ShortestPathTesting` and all eight of its methods call `Dijkstra.Solve` directly). Three
// distinct upstream defects are involved, all three MEASURED against upstream/Numerics @ 2a0357a
// with a throwaway console app referencing the real Numerics.csproj -- see notes 1, 5 and 6, and
// the full write-up in docs/upstream-csharp-issues.md. One of the three is DIVERGED FROM (note 1,
// because a class that cannot be constructed has no behavior to be faithful to); the other two are
// reproduced exactly, quirks and all.
//
// Transcription notes:
//
// 1. DIVERGENCE, deliberate and measured -- the constructor's array sizing. C# computes
//    `max = Math.Max(edges.Max(FromIndex), edges.Max(ToIndex))` and then allocates
//    `new List<Edge>[max]` for BOTH direction caches, one element short: `max` is by construction
//    an index some edge actually uses, so the build loop below indexes at `max` and runs off the
//    end. MEASURED, on the smallest possible graph:
//
//      $ cd /tmp/getpath_probe && dotnet run -c Release
//      === ctor(one edge 0->0)
//        THREW System.IndexOutOfRangeException: Index was outside the bounds of the array.
//        STACK at Numerics.Mathematics.Optimization.Network..ctor(Edge[] edges,
//               Int32[] destinationIndices) in .../Dynamic/Network.cs:line 41
//
//    The same run shows there is no escape by padding the graph, since the padding node is itself
//    the new maximum. The commented-out constructor block immediately above the defect
//    (`if (_edges[i].ToIndex > _nodeCount) { _nodeCount = _edges[i].ToIndex; }` ... `//// Add one
//    to the count for the index offset. //_nodeCount += 1;`) states the intended sizing outright,
//    so the port allocates `max + 1` and sets `node_count_ = max + 1`. That is the ONLY change;
//    it is not silent (it is written up in docs/upstream-csharp-issues.md with the probe output);
//    and it is independently checkable, because with it `network.solve(d)` returns exactly what
//    `dijkstra::solve(edges, d)` returns, which IS oracle-able against the real C# library. The
//    ctest oracles come from a copy of Network.cs carrying this one patch and nothing else,
//    compiled against the real Numerics assembly.
//
// 2. A knock-on of the same defect: C# leaves `_nodeCount` at the `0` its first line assigns
//    (the two lines that would have raised it are inside the commented-out block), and every
//    `Solve` overload passes that 0 down as `Dijkstra.Solve`'s `nodeCount`. Since 0 is not the
//    -1 sentinel, the solver would take it literally, rebuild a zero-length cache and throw:
//
//      === Dijkstra.Solve(edges, 1, nodeCount: 0)
//        THREW System.IndexOutOfRangeException ... in .../Dynamic/Dijkstra.cs:line 123
//
//    so even a correctly sized ctor would leave `Solve` broken. Note 1's `node_count_ = max + 1`
//    fixes both at once, which is why they are one divergence rather than two.
//
// 3. `List<Edge>[]` -> `std::vector<std::vector<Edge>>`, as in dijkstra.hpp note 3: C#'s per-node
//    entries are null until first appended to and the code null-guards them, while an empty vector
//    iterates zero times, so the port drops the guards. `float[,]` -> `dijkstra::ResultTable`,
//    the same `std::vector<std::array<float, 3>>` the solver returns.
//
// 4. `List<int>?` (null for "no path") -> `std::optional<std::vector<int>>`. The two GetPath
//    overloads differ in their no-path return -- the first returns null, the second an EMPTY
//    list -- and both are transcribed as written.
//
// 5. UPSTREAM DEFECT, reproduced on purpose -- `Solve(float[] edgeWeights)` ignores its own
//    argument. It builds a re-weighted copy of the edge array, then passes the STALE
//    `_incomingEdges` cache alongside it; `Dijkstra.Solve` uses a supplied cache whenever its
//    length matches the node count, so it never looks at the re-weighted array and the custom
//    weights have no effect at all. MEASURED in the patched C#: feeding all-ones weights to the
//    DijkstraTesting routing grid returns that grid's ORIGINAL cost column (8, 5, 6, 4, 5, ...),
//    not the unit-weight one (4, 3, 3, 2, 1, ...). The port reproduces this exactly -- the same
//    cache-length test in `dijkstra::solve` produces it for free -- and pins it in
//    `core/tests/test_network_optimization.cpp`. DO NOT "fix" it here; a consumer that wants
//    custom weights must call `dijkstra::solve` on the re-weighted edges itself.
//
// 6. UPSTREAM DEFECT, reproduced on purpose -- `GetPath` cannot return a path. Both overloads
//    filter edges with `Array.BinarySearch(edgesToRemove, edge)`, where `edgesToRemove` is `int[]`
//    and `edge` is an `Edge` struct. That binds `Array.BinarySearch(Array, object)`, which boxes
//    the Edge and asks an `Int32` to compare itself against it. MEASURED:
//
//      === Array.BinarySearch(int[]{0,1,2}, (object)Edge)
//        THREW System.InvalidOperationException: Failed to compare two elements in the array.
//        INNER System.ArgumentException: Object must be of type Int32.
//
//    so any call carrying a non-empty removal list dies on the first edge it examines -- which is
//    every call the method exists to serve. A ZERO-LENGTH binary search never invokes the
//    comparer (`Array.BinarySearch(int[0], (object)Edge)` returns -1), so an empty removal list
//    gets through, and then the outer `do { ... } while (heap.Count == 0)` -- which loops only
//    while the heap is EMPTY, i.e. exits after one pass or throws "Heap is empty." -- means
//    `foundPath` is still false at the end. MEASURED on the patched C#, the complete set of
//    observable outcomes for both overloads is: throw, null, or an empty list. Never a path.
//    `detail::binary_search_edge` below is the port of that expression: it reproduces the
//    measured behavior (-1 when the list is empty, the .NET exception message otherwise) instead
//    of pretending the line does something it does not. The method is transcribed structurally so
//    upstream diffs still map, and is SEVERED from the R/Python surface.
//
// 7. `Array.Sort(edgesToRemove)` sorts the CALLER's array in place. The port takes the removal
//    list by value, so the caller's vector is untouched; nothing observable depends on the
//    difference, since the sorted array is only ever read by note 6's expression.
//
// 8. Node-indexed reads inside `get_path` go through `.at()`, not `operator[]`. The transcribed
//    logic can compute an out-of-range node index -- e.g. `tempIndex = (int)Solve(...)[tempIndex,
//    NEXT_NODE]` is -1 for an unreachable node -- which in C# raises IndexOutOfRangeException and
//    in C++ would be undefined behavior. `.at()` throws `std::out_of_range` instead, keeping the
//    C# failure mode rather than inventing a memory bug. The `solve` overloads use plain indexing
//    like the rest of the port; the constructor's build loop is hardened the same way, since
//    a negative node index there would otherwise write outside the cache.
//
// 9. `Solve(startNodeIndex)` is re-evaluated on EVERY reference inside the first `GetPath`
//    overload -- a full Dijkstra run per table lookup, a dozen times per iteration. Transcribed as
//    written; it is observable only as wasted work, and hoisting it would be the kind of silent
//    improvement this port does not make.
#pragma once
#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/optimization/dynamic/binary_heap.hpp"
#include "corehydro/numerics/math/optimization/dynamic/dijkstra.hpp"

namespace corehydro::numerics::math::optimization {

namespace detail {

// C# `Array.BinarySearch(int[] edgesToRemove, Edge edge)` -- the `(Array, object)` overload. See
// network.hpp note 6: a zero-length search returns ~0 == -1 without comparing anything, and any
// other length throws when Int32 is asked to compare itself against a boxed Edge.
inline int binary_search_edge(const std::vector<int>& edges_to_remove, const Edge&) {
    if (edges_to_remove.empty()) return -1;
    throw std::runtime_error("Failed to compare two elements in the array.");
}

// C# `Array.BinarySearch(int[] edgesToRemove, int edgeIndex)` -- the GENERIC
// `BinarySearch<T>(T[], T)` overload, which is the well-formed sibling of the one above and is
// what the two `(int)resultTable[..., EDGE_INDEX]` call sites bind. Returns the match index, or
// the bitwise complement of the insertion point when absent, exactly as .NET does.
inline int binary_search_index(const std::vector<int>& sorted, int value) {
    const auto it = std::lower_bound(sorted.begin(), sorted.end(), value);
    const int position = static_cast<int>(it - sorted.begin());
    if (it != sorted.end() && *it == value) return position;
    return ~position;
}

}  // namespace detail

// A network of edges used for shortest path optimization applications.
class Network {
   public:
    // Creates a new network from the specified edges and destination indices.
    //   edges                The edges that define the network.
    //   destination_indices  The destination node indices.
    Network(std::vector<Edge> edges, std::vector<int> destination_indices)
        : edges_(std::move(edges)), destination_indices_(std::move(destination_indices)) {
        // C# `edges.Select(x => x.FromIndex).Max()`; LINQ Max throws on an empty sequence.
        if (edges_.empty()) throw std::runtime_error("Sequence contains no elements.");
        int max1 = edges_[0].from_index;
        for (const auto& edge : edges_) {
            if (edge.from_index > max1) max1 = edge.from_index;
        }
        int max2 = edges_[0].to_index;
        for (const auto& edge : edges_) {
            if (edge.to_index > max2) max2 = edge.to_index;
        }
        const int max = max1 > max2 ? max1 : max2;
        // DIVERGENCE (note 1): C# is `_nodeCount = 0;` and `new List<Edge>[max]`, both of which
        // make the class unconstructible. `max + 1` is what the commented-out ctor block upstream
        // says was intended, and is the only change in this file.
        node_count_ = max + 1;
        // .at() below, and the non-negative resize, for the reason in note 8: a negative node
        // index raises IndexOutOfRangeException in C# and would be undefined behavior here.
        incoming_edges_.resize(node_count_ > 0 ? static_cast<std::size_t>(node_count_) : 0);
        outgoing_edges_.resize(node_count_ > 0 ? static_cast<std::size_t>(node_count_) : 0);
        for (std::size_t i = 0; i < edges_.size(); i++) {
            incoming_edges_.at(static_cast<std::size_t>(edges_[i].to_index)).push_back(edges_[i]);
            outgoing_edges_.at(static_cast<std::size_t>(edges_[i].from_index))
                .push_back(edges_[i]);
        }
    }

    // The destination node indices for shortest path computation.
    const std::vector<int>& destination_indices() const { return destination_indices_; }

    // The incoming edges for each node, indexed by node index.
    const std::vector<std::vector<Edge>>& incoming_edges() const { return incoming_edges_; }

    // The outgoing edges for each node, indexed by node index.
    const std::vector<std::vector<Edge>>& outgoing_edges() const { return outgoing_edges_; }

    // The edges that define the network (C#'s private `_edges`, exposed because the toolbox
    // boundary needs to hand them back and nothing upstream forbids it).
    const std::vector<Edge>& edges() const { return edges_; }

    // The number of nodes in the network (C#'s private `_nodeCount`; see note 1).
    int node_count() const { return node_count_; }

    // Solves the shortest path from all nodes to the specified destination.
    dijkstra::ResultTable solve(int destination_index) const {
        return dijkstra::solve(edges_, destination_index, node_count_, &incoming_edges_);
    }

    // Solves the shortest path from all nodes to the specified destinations.
    dijkstra::ResultTable solve(const std::vector<int>& destination_indices) const {
        return dijkstra::solve(edges_, destination_indices, node_count_, &incoming_edges_);
    }

    // Solves the shortest path using custom edge weights.
    //
    // WARNING, and it is upstream's, not the port's: the custom weights have NO EFFECT. See
    // note 5 -- the re-weighted array is built and then bypassed, because the cached incoming
    // edges passed alongside it already match the node count.
    dijkstra::ResultTable solve(const std::vector<float>& edge_weights) const {
        std::vector<Edge> edges(edges_.size());
        for (std::size_t i = 0; i < edges_.size(); i++) {
            edges[i] = Edge(edges_[i].from_index, edges_[i].to_index, edge_weights[i],
                            edges_[i].index);
        }
        //
        return dijkstra::solve(edges, destination_indices_, node_count_, &incoming_edges_);
    }

    // Finds an alternative path avoiding the specified edges.
    //   edges_to_remove   Edge indices to exclude from the path.
    //   start_node_index  The starting node index.
    // Returns a list of edge indices forming the alternative path, or null if no path exists.
    //
    // SEVERED from the R/Python surface and CANNOT RETURN A PATH -- see note 6. Kept so upstream
    // diffs map, and so a caller that reaches it gets upstream's own failure rather than silence.
    std::optional<std::vector<int>> get_path(std::vector<int> edges_to_remove,
                                             int start_node_index) const {
        std::vector<int> node_state(static_cast<std::size_t>(node_count_), 0);
        std::vector<float> node_weight_to_destination(static_cast<std::size_t>(node_count_), 0.0f);
        BinaryHeap<Edge> heap(100000);

        // backwards Dijkstra
        dijkstra::ResultTable result_table(static_cast<std::size_t>(node_count_));
        result_table.at(static_cast<std::size_t>(start_node_index))[0] =
            static_cast<float>(start_node_index);
        result_table.at(static_cast<std::size_t>(start_node_index))[1] = 0;
        result_table.at(static_cast<std::size_t>(start_node_index))[2] = 0;
        node_state.at(static_cast<std::size_t>(start_node_index)) = 1;

        int previous_value = start_node_index;
        int node_index;
        bool found_path = false;

        std::sort(edges_to_remove.begin(), edges_to_remove.end());

        // Loading up the heap starting from destination
        for (const Edge& edge : incoming_edges_.at(static_cast<std::size_t>(previous_value))) {
            if (detail::binary_search_edge(edges_to_remove, edge) < 0) {
                if (previous_value == edge.from_index) {
                    node_index = edge.to_index;
                } else {
                    node_index = edge.from_index;
                }
                switch (node_state.at(static_cast<std::size_t>(node_index))) {
                    case 0: {  // it has not been scanned yet
                        BinaryHeap<Edge>::Node input_node(edge.weight, node_index, edge);
                        heap.add(input_node);
                        node_state.at(static_cast<std::size_t>(node_index)) = 2;
                        node_weight_to_destination.at(static_cast<std::size_t>(node_index)) =
                            input_node.weight;
                        break;
                    }
                    case 1:  // do nothing it has already been solved for
                        break;
                    case 2:  // it has been scanned but not solved
                        if (node_weight_to_destination.at(static_cast<std::size_t>(node_index)) >
                            edge.weight) {
                            BinaryHeap<Edge>::Node input_node2(edge.weight, node_index, edge);
                            node_weight_to_destination.at(static_cast<std::size_t>(node_index)) =
                                input_node2.weight;
                            heap.replace(input_node2);
                        }
                        break;
                }
            }
        }
        // if n = 0, then no roads to escape to
        if (heap.count() == 0) return std::nullopt;

        float temp_weight;
        int temp_index;
        // C# `float FoundDistance = 99999999;` -- the int literal is converted to float
        // and rounds to exactly 1e8f; written as a float literal here so the port does
        // not rely on an implicit narrowing conversion.
        float found_distance = 99999999.0f;
        int potential_to_index = 0;

        BinaryHeap<Edge>::Node result_node;
        float cumulative_weight = 0;

        do {
            result_node = heap.remove_min();

            if (solve(start_node_index).at(static_cast<std::size_t>(result_node.index))[0] == 0)
                continue;

            if (result_node.weight +
                    solve(start_node_index).at(static_cast<std::size_t>(result_node.index))[2] <
                found_distance) {
                previous_value = result_node.index;
                node_state.at(static_cast<std::size_t>(result_node.index)) = 1;
                node_weight_to_destination.at(static_cast<std::size_t>(result_node.index)) =
                    result_node.weight;

                for (const Edge& edge :
                     incoming_edges_.at(static_cast<std::size_t>(previous_value))) {
                    auto& row = result_table.at(static_cast<std::size_t>(result_node.index));
                    if (edge.to_index == result_node.index) {
                        row[0] = static_cast<float>(edge.from_index);
                    } else {
                        row[0] = static_cast<float>(edge.to_index);
                    }

                    row[1] = static_cast<float>(edge.index);
                    row[2] = result_node.weight;

                    if (solve(start_node_index).at(static_cast<std::size_t>(edge.to_index))[0] ==
                        static_cast<float>(edge.from_index)) {
                        if (detail::binary_search_edge(edges_to_remove, edge) < 0) {
                            if (previous_value == edge.from_index) {
                                node_index = edge.to_index;
                            } else {
                                node_index = edge.from_index;
                            }

                            switch (node_state.at(static_cast<std::size_t>(node_index))) {
                                case 0:  // has not been scanned yet
                                    cumulative_weight = edge.weight + result_node.weight;
                                    heap.add(BinaryHeap<Edge>::Node(cumulative_weight, node_index,
                                                                    edge));
                                    node_state.at(static_cast<std::size_t>(node_index)) = 2;
                                    node_weight_to_destination.at(
                                        static_cast<std::size_t>(node_index)) = cumulative_weight;
                                    break;
                                case 1:
                                    break;
                                case 2:
                                    if (node_weight_to_destination.at(
                                            static_cast<std::size_t>(node_index)) >
                                        cumulative_weight) {
                                        node_weight_to_destination.at(
                                            static_cast<std::size_t>(node_index)) =
                                            cumulative_weight;
                                        heap.replace(BinaryHeap<Edge>::Node(cumulative_weight,
                                                                            node_index, edge));
                                    }
                                    break;
                            }
                        }
                    } else if (edge.from_index != start_node_index &&
                               solve(start_node_index)
                                       .at(static_cast<std::size_t>(edge.from_index))[0] ==
                                   static_cast<float>(result_node.index)) {
                        // Already on the lookup table going forwards
                        for (const Edge& edge2 :
                             incoming_edges_.at(static_cast<std::size_t>(previous_value))) {
                            if (detail::binary_search_edge(edges_to_remove, edge2) < 0) {
                                if (previous_value == edge2.from_index) {
                                    node_index = edge2.to_index;
                                } else {
                                    node_index = edge2.from_index;
                                }

                                switch (node_state.at(static_cast<std::size_t>(node_index))) {
                                    case 0:
                                        heap.add(BinaryHeap<Edge>::Node(cumulative_weight,
                                                                        node_index, edge));
                                        node_state.at(static_cast<std::size_t>(node_index)) = 2;
                                        node_weight_to_destination.at(
                                            static_cast<std::size_t>(node_index)) =
                                            cumulative_weight;
                                        break;
                                    case 1:
                                        break;
                                    case 2:
                                        if (node_weight_to_destination.at(
                                                static_cast<std::size_t>(node_index)) >
                                            cumulative_weight) {
                                            node_weight_to_destination.at(
                                                static_cast<std::size_t>(node_index)) =
                                                cumulative_weight;
                                            heap.replace(BinaryHeap<Edge>::Node(
                                                cumulative_weight, node_index, edge));
                                        }
                                        break;
                                }
                            }
                        }
                    } else {
                        // Potential new path, check path viability
                        temp_weight =
                            solve(start_node_index)
                                .at(static_cast<std::size_t>(result_node.index))[2];
                        temp_index = result_node.index;

                        do {
                            if (detail::binary_search_index(
                                    edges_to_remove,
                                    static_cast<int>(
                                        solve(start_node_index)
                                            .at(static_cast<std::size_t>(temp_index))[1])) >= 0) {
                                for (const Edge& edge3 :
                                     incoming_edges_.at(static_cast<std::size_t>(previous_value))) {
                                    if (detail::binary_search_edge(edges_to_remove, edge3) < 0) {
                                        if (previous_value == edge3.from_index) {
                                            node_index = edge3.to_index;
                                        } else {
                                            node_index = edge3.from_index;
                                        }

                                        switch (node_state.at(
                                            static_cast<std::size_t>(node_index))) {
                                            case 0:
                                                heap.add(BinaryHeap<Edge>::Node(cumulative_weight,
                                                                                node_index, edge));
                                                node_state.at(
                                                    static_cast<std::size_t>(node_index)) = 2;
                                                node_weight_to_destination.at(
                                                    static_cast<std::size_t>(node_index)) =
                                                    cumulative_weight;
                                                break;
                                            case 1:
                                                break;
                                            case 2:
                                                if (node_weight_to_destination.at(
                                                        static_cast<std::size_t>(node_index)) >
                                                    cumulative_weight) {
                                                    node_weight_to_destination.at(
                                                        static_cast<std::size_t>(node_index)) =
                                                        cumulative_weight;
                                                    heap.replace(BinaryHeap<Edge>::Node(
                                                        cumulative_weight, node_index, edge));
                                                }
                                                break;
                                        }
                                    }
                                }
                                break;
                            }
                            temp_weight = solve(start_node_index)
                                              .at(static_cast<std::size_t>(temp_index))[2];
                            temp_index = static_cast<int>(
                                solve(start_node_index).at(static_cast<std::size_t>(temp_index))[0]);
                        } while (temp_weight == 0);

                        if (temp_weight == 0) {
                            found_distance =
                                result_node.weight +
                                solve(start_node_index)
                                    .at(static_cast<std::size_t>(result_node.index))[2];
                            potential_to_index = result_node.index;
                            found_path = true;
                        }
                    }
                }
            }
        } while (heap.count() == 0);

        // Check to see if a destination was reached, if so then create a path to the nearest
        // destination
        if (found_path) {
            std::vector<int> updated_path;
            float temp_len = result_table.at(static_cast<std::size_t>(potential_to_index))[2];
            int temp_edge = static_cast<int>(
                result_table.at(static_cast<std::size_t>(potential_to_index))[1]);
            int temp_node = potential_to_index;

            while (temp_len == 0) {
                updated_path.push_back(temp_edge);
                temp_node =
                    static_cast<int>(result_table.at(static_cast<std::size_t>(temp_node))[0]);
                temp_edge =
                    static_cast<int>(result_table.at(static_cast<std::size_t>(temp_node))[1]);
                temp_len = result_table.at(static_cast<std::size_t>(temp_node))[2];
            }

            std::reverse(updated_path.begin(), updated_path.end());

            temp_len =
                solve(start_node_index).at(static_cast<std::size_t>(potential_to_index))[2];
            temp_edge = static_cast<int>(
                solve(start_node_index).at(static_cast<std::size_t>(potential_to_index))[1]);
            temp_node = potential_to_index;

            while (temp_len == 0) {
                updated_path.push_back(temp_edge);
                temp_node = static_cast<int>(
                    solve(start_node_index).at(static_cast<std::size_t>(temp_node))[0]);
                temp_edge = static_cast<int>(
                    solve(start_node_index).at(static_cast<std::size_t>(temp_node))[1]);
                temp_len =
                    solve(start_node_index).at(static_cast<std::size_t>(temp_node))[2];
            }

            return updated_path;
        } else {
            return std::nullopt;
        }
    }

    // Finds an alternative path avoiding the specified edges, using a pre-computed results table.
    //   edges_to_remove         Edge indices to exclude from the path.
    //   start_node_index        The starting node index.
    //   existing_results_table  A pre-computed shortest path results table.
    // Returns a list of edge indices forming the alternative path, or an EMPTY list if no path
    // exists (upstream's two overloads disagree on this; see note 4).
    //
    // SEVERED from the R/Python surface and CANNOT RETURN A PATH -- see note 6.
    std::optional<std::vector<int>> get_path(
        std::vector<int> edges_to_remove, int start_node_index,
        const dijkstra::ResultTable& existing_results_table) const {
        std::vector<int> node_state(static_cast<std::size_t>(node_count_), 0);
        std::vector<float> node_weight_to_destination(static_cast<std::size_t>(node_count_), 0.0f);
        BinaryHeap<Edge> heap(100000);
        int node_index;

        // backwards Dijkstra
        dijkstra::ResultTable result_table(static_cast<std::size_t>(node_count_));
        result_table.at(static_cast<std::size_t>(start_node_index))[0] =
            static_cast<float>(start_node_index);
        result_table.at(static_cast<std::size_t>(start_node_index))[1] = 0;
        result_table.at(static_cast<std::size_t>(start_node_index))[2] = 0;
        node_state.at(static_cast<std::size_t>(start_node_index)) = 1;

        int previous_value = start_node_index;
        bool found_path = false;

        std::sort(edges_to_remove.begin(), edges_to_remove.end());

        // Loading up the heap starting from destination
        for (const Edge& edge : incoming_edges_.at(static_cast<std::size_t>(previous_value))) {
            if (detail::binary_search_edge(edges_to_remove, edge) < 0) {
                if (previous_value == edge.from_index) {
                    node_index = edge.to_index;
                } else {
                    node_index = edge.from_index;
                }
                switch (node_state.at(static_cast<std::size_t>(node_index))) {
                    case 0: {  // it has not been scanned yet
                        BinaryHeap<Edge>::Node input_node(edge.weight, node_index, edge);
                        heap.add(input_node);
                        node_state.at(static_cast<std::size_t>(node_index)) = 2;
                        node_weight_to_destination.at(static_cast<std::size_t>(node_index)) =
                            input_node.weight;
                        break;
                    }
                    case 1:  // do nothing it has already been solved for
                        break;
                    case 2:  // it has been scanned but not solved
                        if (node_weight_to_destination.at(static_cast<std::size_t>(node_index)) >
                            edge.weight) {
                            BinaryHeap<Edge>::Node input_node2(edge.weight, node_index, edge);
                            node_weight_to_destination.at(static_cast<std::size_t>(node_index)) =
                                input_node2.weight;
                            heap.replace(input_node2);
                        }
                        break;
                }
            }
        }

        // if n = 0 then no roads to escape to
        if (heap.count() == 0) return std::nullopt;

        float temp_weight;
        int temp_index;
        // C# `float FoundDistance = 99999999;` -- the int literal is converted to float
        // and rounds to exactly 1e8f; written as a float literal here so the port does
        // not rely on an implicit narrowing conversion.
        float found_distance = 99999999.0f;
        int potential_to_index = 0;

        BinaryHeap<Edge>::Node result_node;
        float cumulative_weight = 0;

        do {
            result_node = heap.remove_min();

            if (existing_results_table.at(static_cast<std::size_t>(result_node.index))[0] == 0)
                continue;

            if (result_node.weight +
                    existing_results_table.at(static_cast<std::size_t>(result_node.index))[2] <
                found_distance) {
                previous_value = result_node.index;
                node_state.at(static_cast<std::size_t>(result_node.index)) = 1;
                node_weight_to_destination.at(static_cast<std::size_t>(result_node.index)) =
                    result_node.weight;

                for (const Edge& edge :
                     incoming_edges_.at(static_cast<std::size_t>(previous_value))) {
                    auto& row = result_table.at(static_cast<std::size_t>(result_node.index));
                    if (edge.to_index == result_node.index) {
                        row[0] = static_cast<float>(edge.from_index);
                    } else {
                        row[0] = static_cast<float>(edge.to_index);
                    }

                    row[1] = static_cast<float>(edge.index);
                    row[2] = result_node.weight;

                    if (existing_results_table.at(static_cast<std::size_t>(edge.to_index))[0] ==
                        static_cast<float>(edge.from_index)) {
                        if (detail::binary_search_edge(edges_to_remove, edge) < 0) {
                            if (previous_value == edge.from_index) {
                                node_index = edge.to_index;
                            } else {
                                node_index = edge.from_index;
                            }

                            switch (node_state.at(static_cast<std::size_t>(node_index))) {
                                case 0:  // has not been scanned yet
                                    cumulative_weight = edge.weight + result_node.weight;
                                    heap.add(BinaryHeap<Edge>::Node(cumulative_weight, node_index,
                                                                    edge));
                                    node_state.at(static_cast<std::size_t>(node_index)) = 2;
                                    node_weight_to_destination.at(
                                        static_cast<std::size_t>(node_index)) = cumulative_weight;
                                    break;
                                case 1:
                                    break;
                                case 2:
                                    if (node_weight_to_destination.at(
                                            static_cast<std::size_t>(node_index)) >
                                        cumulative_weight) {
                                        node_weight_to_destination.at(
                                            static_cast<std::size_t>(node_index)) =
                                            cumulative_weight;
                                        heap.replace(BinaryHeap<Edge>::Node(cumulative_weight,
                                                                            node_index, edge));
                                    }
                                    break;
                            }
                        }
                    }
                }
            } else if (heap.count() != 0) {
                for (const Edge& edge :
                     incoming_edges_.at(static_cast<std::size_t>(previous_value))) {
                    if (existing_results_table.at(static_cast<std::size_t>(edge.from_index))[0] ==
                        static_cast<float>(result_node.index)) {
                        if (detail::binary_search_edge(edges_to_remove, edge) < 0) {
                            if (previous_value == edge.from_index) {
                                node_index = edge.to_index;
                            } else {
                                node_index = edge.from_index;
                            }

                            switch (node_state.at(static_cast<std::size_t>(node_index))) {
                                case 0:  // has not been scanned yet
                                    cumulative_weight = edge.weight + result_node.weight;
                                    heap.add(BinaryHeap<Edge>::Node(cumulative_weight, node_index,
                                                                    edge));
                                    node_state.at(static_cast<std::size_t>(node_index)) = 2;
                                    node_weight_to_destination.at(
                                        static_cast<std::size_t>(node_index)) = cumulative_weight;
                                    break;
                                case 1:
                                    break;
                                case 2:
                                    if (node_weight_to_destination.at(
                                            static_cast<std::size_t>(node_index)) >
                                        cumulative_weight) {
                                        node_weight_to_destination.at(
                                            static_cast<std::size_t>(node_index)) =
                                            cumulative_weight;
                                        heap.replace(BinaryHeap<Edge>::Node(cumulative_weight,
                                                                            node_index, edge));
                                    }
                                    break;
                            }
                        }
                    }
                }
            } else {
                // check viability of route
                temp_weight =
                    existing_results_table.at(static_cast<std::size_t>(result_node.index))[2];
                temp_index = result_node.index;

                do {
                    // check to see if the current route has a blocked segment
                    if (detail::binary_search_index(
                            edges_to_remove,
                            static_cast<int>(existing_results_table.at(
                                static_cast<std::size_t>(temp_index))[1])) >= 0) {
                        for (const Edge& edge :
                             incoming_edges_.at(static_cast<std::size_t>(previous_value))) {
                            if (previous_value == edge.from_index) {
                                node_index = edge.to_index;
                            } else {
                                node_index = edge.from_index;
                            }

                            switch (node_state.at(static_cast<std::size_t>(node_index))) {
                                case 0:  // has not been scanned yet
                                    cumulative_weight = edge.weight + result_node.weight;
                                    heap.add(BinaryHeap<Edge>::Node(cumulative_weight, node_index,
                                                                    edge));
                                    node_state.at(static_cast<std::size_t>(node_index)) = 2;
                                    node_weight_to_destination.at(
                                        static_cast<std::size_t>(node_index)) = cumulative_weight;
                                    break;
                                case 1:
                                    break;
                                case 2:
                                    if (node_weight_to_destination.at(
                                            static_cast<std::size_t>(node_index)) >
                                        cumulative_weight) {
                                        node_weight_to_destination.at(
                                            static_cast<std::size_t>(node_index)) =
                                            cumulative_weight;
                                        heap.replace(BinaryHeap<Edge>::Node(cumulative_weight,
                                                                            node_index, edge));
                                    }
                                    break;
                            }
                        }
                    }
                    temp_weight =
                        existing_results_table.at(static_cast<std::size_t>(temp_index))[2];
                    temp_index = static_cast<int>(
                        existing_results_table.at(static_cast<std::size_t>(temp_index))[0]);
                } while (temp_weight == 0);

                if (temp_weight == 0) {
                    found_distance =
                        result_node.weight +
                        existing_results_table.at(static_cast<std::size_t>(result_node.index))[2];
                    potential_to_index = result_node.index;
                    found_path = true;
                }
            }
        } while (heap.count() == 0);

        // Check to see if the destination was reached, if so then create a path to the nearest
        // destination
        if (found_path) {
            std::vector<int> updated_path;
            float temp_len = result_table.at(static_cast<std::size_t>(potential_to_index))[2];
            int temp_edge = static_cast<int>(
                result_table.at(static_cast<std::size_t>(potential_to_index))[1]);
            int temp_node = potential_to_index;

            while (temp_len == 0) {
                updated_path.push_back(temp_edge);
                temp_node =
                    static_cast<int>(result_table.at(static_cast<std::size_t>(temp_node))[0]);
                temp_edge =
                    static_cast<int>(result_table.at(static_cast<std::size_t>(temp_node))[1]);
                temp_len = result_table.at(static_cast<std::size_t>(temp_node))[2];
            }

            // updatedPath.Add(startingEdge);
            std::reverse(updated_path.begin(), updated_path.end());

            temp_len =
                existing_results_table.at(static_cast<std::size_t>(potential_to_index))[2];
            temp_edge = static_cast<int>(
                existing_results_table.at(static_cast<std::size_t>(potential_to_index))[1]);
            temp_node = potential_to_index;

            while (temp_len == 0) {
                updated_path.push_back(temp_edge);
                // NOTE, upstream, transcribed as written: this walks the COST column (index 2) as
                // if it were a node index, where the sibling overload above walks NEXT_NODE.
                temp_node = static_cast<int>(
                    existing_results_table.at(static_cast<std::size_t>(temp_node))[2]);
                temp_edge = static_cast<int>(
                    existing_results_table.at(static_cast<std::size_t>(temp_node))[1]);
                temp_len = existing_results_table.at(static_cast<std::size_t>(temp_node))[2];
            }

            return updated_path;
        } else {
            return std::vector<int>();
        }
    }

   private:
    std::vector<Edge> edges_;
    std::vector<int> destination_indices_;
    std::vector<std::vector<Edge>> incoming_edges_;
    std::vector<std::vector<Edge>> outgoing_edges_;
    int node_count_ = 0;
};

}  // namespace corehydro::numerics::math::optimization
