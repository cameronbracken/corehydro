// Transcribed C# oracle tests for the dynamic-programming network layer (P3 Task 8):
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Dynamic/BinaryHeapTesting.cs @ 2a0357a
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Dynamic/DijkstraTesting.cs @ 2a0357a
//
// All 17 upstream [TestMethod]s (9 BinaryHeapTesting + 8 ShortestPathTesting, the class inside
// DijkstraTesting.cs) are transcribed in C# file order with their exact literals. These are
// internal-support ports validated against the C# test oracles themselves; the public-surface
// fixtures come later in the phase. Skipped upstream methods: none. Despite the file name and
// `SimpleNetworkRouting`'s method name, NOTHING in DijkstraTesting.cs constructs a `Network` --
// all eight methods call `Dijkstra.Solve` directly -- so none of them had to wait for that class.
//
// A clearly-marked SUPPLEMENT at the bottom (not from either C# file) covers the two things the
// 17 transcriptions leave unguarded: the single-precision arithmetic, and `Dijkstra.PathExists`.
//
// Between the two sits a NETWORK block (P3 Task 9), also supplement-class: `Network.cs` has no
// upstream test class either, and the shipped C# `Network` cannot even be constructed, so its
// oracles were measured off a minimally patched copy of the C# class compiled against the real
// Numerics assembly. That block's own comment explains the patch and carries the probe transcript.
//
// EXACT float EQUALITY, on purpose. The C# tests assert `Assert.AreEqual(3f, result[2, 2])` with
// no delta -- MSTest's generic overload compares floats exactly -- and
// `float.IsPositiveInfinity(...)`. The port asserts the same way (CHECK_EQ on `float`,
// `std::isinf` plus a sign test), because the whole point of keeping the weights in single
// precision is that the accumulation is observable. An epsilon comparison here would hide the
// very thing the C# tests pin.
//
// Where the C# writes `Assert.AreEqual(3, result[1, 0])` -- an `int` literal against a `float`
// element -- C#'s generic overload infers T = float and widens the literal, so the port writes
// the literal as `3.0f` and compares floats. Nothing about the comparison changes.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/math/optimization/dynamic/binary_heap.hpp"
#include "corehydro/numerics/math/optimization/dynamic/dijkstra.hpp"
#include "corehydro/numerics/math/optimization/dynamic/network.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

using corehydro::numerics::math::optimization::BinaryHeap;
using corehydro::numerics::math::optimization::Edge;
using corehydro::numerics::math::optimization::Network;
using corehydro::numerics::sampling::MersenneTwister;
namespace dijkstra = corehydro::numerics::math::optimization::dijkstra;

namespace {

// ====================== BinaryHeap (BinaryHeapTesting.cs) ======================

// Checking weights on heap.
void heap_test1() {
    // Node weights
    std::vector<float> weights = {.3f, .5f, 32.f, 15.f, 12.f, .01f, -4.f};

    BinaryHeap<double> heap(30);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        heap.add(BinaryHeap<double>::Node(weights[i], static_cast<int>(i),
                                          static_cast<double>(i) * .5));
    }

    CHECK_EQ(heap.remove_min().weight, weights[6]);
    CHECK_EQ(heap.remove_min().weight, weights[5]);
    CHECK_EQ(heap.remove_min().weight, weights[0]);
    CHECK_EQ(heap.remove_min().weight, weights[1]);
    CHECK_EQ(heap.remove_min().weight, weights[4]);
    CHECK_EQ(heap.remove_min().weight, weights[3]);
    CHECK_EQ(heap.remove_min().weight, weights[2]);
}

// Random heap and weight allotment.
void heap_test2() {
    // Random Node weights
    std::vector<float> weights(1000);

    MersenneTwister randy(12345);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>(randy.next_double());
    }
    // Add to the heap
    BinaryHeap<double> heap(static_cast<int>(weights.size()));
    for (std::size_t i = 0; i < weights.size(); ++i) {
        heap.add(BinaryHeap<double>::Node(weights[i], static_cast<int>(i),
                                          static_cast<double>(i) * .5));
    }

    std::sort(weights.begin(), weights.end());
    // Compare
    for (std::size_t i = 0; i < weights.size(); ++i) {
        CHECK_EQ(weights[i], heap.remove_min().weight);
    }
}

// Making sure heap is ordering correctly.
void heap_test3() {
    // Node weights
    std::vector<float> weights = {.3f, .5f, 32.f, 15.f, 12.f, .01f, -4.f};

    BinaryHeap<double> heap(30);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        heap.add(BinaryHeap<double>::Node(weights[i], static_cast<int>(i),
                                          static_cast<double>(i) * .5));
    }

    for (std::size_t i = 0; i < weights.size(); ++i) {
        heap.replace(
            BinaryHeap<double>::Node(weights[i], static_cast<int>(i), static_cast<double>(i)));
    }

    std::sort(weights.begin(), weights.end());
    // Compare
    for (std::size_t i = 0; i < weights.size(); ++i) {
        CHECK_EQ(weights[i], heap.remove_min().weight);
    }
}

// Testing edge case.
void heap_test4() {
    // Node weights
    std::vector<float> weights = {.3f, .5f, 32.f, 15.f, 12.f, .01f, -4.f};

    BinaryHeap<double> heap(30);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        heap.add(BinaryHeap<double>::Node(weights[i], static_cast<int>(i),
                                          static_cast<double>(i) * .5));
    }

    for (std::size_t i = 0; i < weights.size(); ++i) {
        heap.replace(
            BinaryHeap<double>::Node(weights[i], static_cast<int>(i), static_cast<double>(i) * 5));
    }

    // Compare (the C# `Assert.AreNotEqual(heap.RemoveMin().Value, weights[i])` compares the
    // stored double against the UNSORTED weight at the same loop index)
    for (std::size_t i = 0; i < weights.size(); ++i) {
        CHECK_TRUE(heap.remove_min().value != static_cast<double>(weights[i]));
    }
}

// Decrease key for ordering
void decrease_key_test() {
    BinaryHeap<std::string> heap(10);
    heap.add(BinaryHeap<std::string>::Node(10.f, 1, "A"));
    heap.add(BinaryHeap<std::string>::Node(20.f, 2, "B"));
    heap.add(BinaryHeap<std::string>::Node(30.f, 3, "C"));

    heap.decrease_key(BinaryHeap<std::string>::Node(5.f, 3, "C"));  // now should be top

    auto node = heap.remove_min();
    CHECK_EQ(3, node.index);
    CHECK_EQ(5.f, node.weight);
}

// Edge case for heap capacity.
void heap_capacity_exceeded_test() {
    CHECK_THROWS_MSG(
        ([] {
            BinaryHeap<int> heap(3);
            heap.add(BinaryHeap<int>::Node(1.f, 1, 1));
            heap.add(BinaryHeap<int>::Node(2.f, 2, 2));
            heap.add(BinaryHeap<int>::Node(3.f, 3, 3));
            heap.add(BinaryHeap<int>::Node(4.f, 4, 4));  // exceeds capacity
        }()),
        "Heap is full.");
}

// Testing RemoveMin() is getting called correctly on heap.
void remove_min_from_empty_heap_test() {
    CHECK_THROWS_MSG(
        ([] {
            BinaryHeap<int> heap(10);
            heap.remove_min();  // should throw
        }()),
        "Heap is empty.");
}

// Using heap and Decrease key and name of the test explains it.
void replace_with_higher_weight_should_do_nothing_test() {
    BinaryHeap<int> heap(10);
    heap.add(BinaryHeap<int>::Node(5.f, 1, 1));
    heap.decrease_key(BinaryHeap<int>::Node(10.f, 1, 1));  // Should not update weight

    CHECK_EQ(5.f, heap.remove_min().weight);  // weight should still be 5
}

// Edge case to check ordering is correct with negative weights.
void ordering_with_negative_weights_test() {
    BinaryHeap<std::string> heap(5);
    heap.add(BinaryHeap<std::string>::Node(-10.f, 1, "A"));
    heap.add(BinaryHeap<std::string>::Node(-20.f, 2, "B"));
    heap.add(BinaryHeap<std::string>::Node(0.f, 3, "C"));

    CHECK_TRUE(heap.remove_min().value == "B");
    CHECK_TRUE(heap.remove_min().value == "A");
    CHECK_TRUE(heap.remove_min().value == "C");
}

// ================= Dijkstra (DijkstraTesting.cs, class ShortestPathTesting) =================

// A positive-infinity element, the port's `float.IsPositiveInfinity`.
bool is_positive_infinity(float value) { return std::isinf(value) && value > 0.f; }

// Testing a something that cost doesn't really matter.
void simple_edge_graph_cost() {
    std::vector<Edge> edges;
    edges.push_back(Edge(0, 1, 2, 0));
    edges.push_back(Edge(0, 2, 4, 2));
    edges.push_back(Edge(1, 2, 1, 2));
    edges.push_back(Edge(1, 3, 7, 3));
    edges.push_back(Edge(2, 3, 3, 4));
    edges.push_back(Edge(4, 0, 1, 5));

    auto result = dijkstra::solve(edges, 3, 6);

    CHECK_EQ(0.f, result[3][dijkstra::COST]);
    CHECK_EQ(3.f, result[2][dijkstra::COST]);
    CHECK_EQ(4.f, result[1][dijkstra::COST]);
    CHECK_EQ(6.f, result[0][dijkstra::COST]);
    CHECK_EQ(7.f, result[4][dijkstra::COST]);
    CHECK_TRUE(is_positive_infinity(result[5][dijkstra::COST]));
}

// Simple network run, testing to see if algorithm chooses the lowest cost path as it should.
void simple_network_routing() {
    // Simple Network Node Setup
    // 0 - 1 - 2 - 3 - 4
    // |   | \ |   |   |
    // 5 - 6 - 7 - 8 - 9

    std::vector<Edge> edges;
    edges.push_back(Edge(0, 5, 1, 0));
    edges.push_back(Edge(0, 1, 30, 1));

    edges.push_back(Edge(1, 0, 30, 1));
    edges.push_back(Edge(1, 2, 1, 2));
    edges.push_back(Edge(1, 6, 15, 3));
    edges.push_back(Edge(1, 7, 2, 4));

    edges.push_back(Edge(2, 1, 1, 2));
    edges.push_back(Edge(2, 3, 5, 5));
    edges.push_back(Edge(2, 7, 5, 6));

    edges.push_back(Edge(3, 2, 5, 5));
    edges.push_back(Edge(3, 8, 2, 7));
    edges.push_back(Edge(3, 4, 1, 8));

    edges.push_back(Edge(4, 3, 1, 8));
    edges.push_back(Edge(4, 9, 30, 9));

    edges.push_back(Edge(5, 0, 1, 0));
    edges.push_back(Edge(5, 6, 3, 10));

    edges.push_back(Edge(6, 5, 3, 10));
    edges.push_back(Edge(6, 1, 15, 3));
    edges.push_back(Edge(6, 7, 1, 11));

    edges.push_back(Edge(7, 6, 1, 11));
    edges.push_back(Edge(7, 1, 2, 4));
    edges.push_back(Edge(7, 2, 5, 6));
    edges.push_back(Edge(7, 8, 1, 12));

    edges.push_back(Edge(8, 7, 1, 12));
    edges.push_back(Edge(8, 3, 2, 7));
    edges.push_back(Edge(8, 9, 2, 13));

    edges.push_back(Edge(9, 8, 2, 13));
    edges.push_back(Edge(9, 4, 30, 9));

    auto result = dijkstra::solve(edges, 9);

    // Algorithm is choosing the next node that yields the shortest paths
    CHECK_EQ(5.f, result[0][dijkstra::NEXT_NODE]);
    CHECK_EQ(8.f, result[0][dijkstra::COST]);

    CHECK_EQ(7.f, result[1][dijkstra::NEXT_NODE]);
    CHECK_EQ(5.f, result[1][dijkstra::COST]);

    CHECK_EQ(1.f, result[2][dijkstra::NEXT_NODE]);
    CHECK_EQ(6.f, result[2][dijkstra::COST]);

    CHECK_EQ(8.f, result[3][dijkstra::NEXT_NODE]);
    CHECK_EQ(4.f, result[3][dijkstra::COST]);

    CHECK_EQ(3.f, result[4][dijkstra::NEXT_NODE]);
    CHECK_EQ(5.f, result[4][dijkstra::COST]);

    CHECK_EQ(6.f, result[5][dijkstra::NEXT_NODE]);
    CHECK_EQ(7.f, result[5][dijkstra::COST]);

    CHECK_EQ(7.f, result[6][dijkstra::NEXT_NODE]);
    CHECK_EQ(4.f, result[6][dijkstra::COST]);

    CHECK_EQ(8.f, result[7][dijkstra::NEXT_NODE]);
    CHECK_EQ(3.f, result[7][dijkstra::COST]);

    CHECK_EQ(9.f, result[8][dijkstra::NEXT_NODE]);
    CHECK_EQ(2.f, result[8][dijkstra::COST]);

    CHECK_EQ(9.f, result[9][dijkstra::NEXT_NODE]);
    CHECK_EQ(0.f, result[9][dijkstra::COST]);
}

// Testing edges with with bidirectionality.
void bidirectional_routing() {
    std::vector<Edge> edges;
    edges.push_back(Edge(0, 1, 6, 0));
    edges.push_back(Edge(0, 3, 1, 1));

    edges.push_back(Edge(1, 0, 6, 0));
    edges.push_back(Edge(1, 2, 5, 2));
    edges.push_back(Edge(1, 3, 2, 3));
    edges.push_back(Edge(1, 4, 2, 4));

    edges.push_back(Edge(2, 1, 5, 2));
    edges.push_back(Edge(2, 4, 5, 5));

    edges.push_back(Edge(3, 0, 1, 1));
    edges.push_back(Edge(3, 1, 2, 3));
    edges.push_back(Edge(3, 4, 1, 6));

    edges.push_back(Edge(4, 1, 2, 4));
    edges.push_back(Edge(4, 2, 5, 5));
    edges.push_back(Edge(4, 3, 1, 6));

    auto result = dijkstra::solve(edges, 4);

    CHECK_EQ(3.f, result[0][dijkstra::NEXT_NODE]);
    CHECK_EQ(2.f, result[0][dijkstra::COST]);

    CHECK_EQ(4.f, result[1][dijkstra::NEXT_NODE]);
    CHECK_EQ(2.f, result[1][dijkstra::COST]);

    CHECK_EQ(4.f, result[2][dijkstra::NEXT_NODE]);
    CHECK_EQ(5.f, result[2][dijkstra::COST]);

    CHECK_EQ(4.f, result[3][dijkstra::NEXT_NODE]);
    CHECK_EQ(1.f, result[3][dijkstra::COST]);

    CHECK_EQ(4.f, result[4][dijkstra::NEXT_NODE]);
    CHECK_EQ(0.f, result[4][dijkstra::COST]);
}

// Testing that a disconnected node returns a positive infinity.
void disconnected_nodes_test() {
    std::vector<Edge> edges = {
        Edge(0, 1, 1, 0),
        Edge(1, 2, 1, 1),
        // Node 3 is disconnected
    };

    auto result = dijkstra::solve(edges, 2, 4);

    CHECK_EQ(2.f, result[1][dijkstra::NEXT_NODE]);
    CHECK_EQ(1.f, result[1][dijkstra::COST]);

    CHECK_EQ(1.f, result[0][dijkstra::NEXT_NODE]);
    CHECK_EQ(2.f, result[0][dijkstra::COST]);

    // Unreachable node 3 should remain with default values
    CHECK_TRUE(is_positive_infinity(result[3][dijkstra::COST]));
}

// Simple multiple dest path
void multiple_dest_shared_path() {
    // Graph:
    // 0 - 1 - 2
    //     |
    //     3
    std::vector<Edge> edges = {Edge(0, 1, 1, 0), Edge(1, 0, 3, 1), Edge(1, 2, 1, 2),
                               Edge(2, 1, 2, 3), Edge(1, 3, 3, 4)};

    auto result = dijkstra::solve(edges, std::vector<int>{0, 3}, 4);

    CHECK_EQ(0.f, result[1][dijkstra::NEXT_NODE]);
    CHECK_EQ(3.f, result[1][dijkstra::COST]);
    CHECK_EQ(1.f, result[2][dijkstra::NEXT_NODE]);
    CHECK_EQ(5.f, result[2][dijkstra::COST]);
}

// Checking paths are indeed disconnected.
void disconnected_component() {
    // Graph:
    // 0 - 1      2 - 3
    std::vector<Edge> edges = {Edge(0, 1, 1, 0), Edge(1, 0, 3, 1), Edge(2, 3, 1, 2)};
    auto result = dijkstra::solve(edges, std::vector<int>{0, 3}, 4);
    CHECK_EQ(0.f, result[1][dijkstra::NEXT_NODE]);
    CHECK_EQ(3.f, result[1][dijkstra::COST]);
    CHECK_EQ(3.f, result[2][dijkstra::NEXT_NODE]);
    CHECK_EQ(1.f, result[2][dijkstra::COST]);
}

// Checking disconnected with 1 destination.
void disconnected_component2() {
    std::vector<Edge> edges = {
        Edge(0, 1, 1, 0),
        Edge(1, 0, 3, 1),
        Edge(2, 3, 1, 2),
    };

    auto result = dijkstra::solve(edges, std::vector<int>{0}, 4);
    CHECK_TRUE(is_positive_infinity(result[2][dijkstra::COST]));
}

// Testing two destinations when they are connected by an edge.
void triangle_path() {
    // Graph:
    // 0 <-> 1
    // 1 -> 2
    // 2 -> 0
    std::vector<Edge> edges = {Edge(0, 1, 1, 0), Edge(1, 0, 4, 1), Edge(1, 2, 1, 2),
                               Edge(2, 0, 10, 3)};

    auto result = dijkstra::solve(edges, std::vector<int>{0, 2}, 3);
    CHECK_EQ(0.f, result[0][dijkstra::NEXT_NODE]);
    CHECK_EQ(0.f, result[0][dijkstra::COST]);
    CHECK_EQ(2.f, result[1][dijkstra::NEXT_NODE]);
    CHECK_EQ(1.f, result[1][dijkstra::COST]);
    CHECK_EQ(2.f, result[2][dijkstra::NEXT_NODE]);
}

// ========================= Network (NO upstream test class) =========================
// `Network.cs` has no test class anywhere in Test_Numerics -- DijkstraTesting.cs's class is
// `ShortestPathTesting` and all eight of its methods call `Dijkstra.Solve` directly. So
// everything below is SUPPLEMENT-CLASS coverage: no C# test literal exists for any of it, and
// every expected value was READ OFF the real C# library rather than derived (see below).
//
// Getting an oracle at all took a detour, because the shipped C# `Network` CANNOT BE
// CONSTRUCTED -- its constructor sizes both edge-list arrays to `max` (the largest node index)
// rather than `max + 1`, so the very edge that attains that index runs off the end. MEASURED
// against upstream/Numerics @ 2a0357a:
//
//   $ cd /tmp/getpath_probe && dotnet run -c Release
//   === ctor(one edge 0->0)
//     THREW System.IndexOutOfRangeException: Index was outside the bounds of the array.
//     STACK at Numerics.Mathematics.Optimization.Network..ctor(Edge[] edges, Int32[]
//            destinationIndices) in .../Dynamic/Network.cs:line 41
//
// The port therefore carries ONE documented, measured divergence -- the `max + 1` sizing and the
// `_nodeCount` the commented-out ctor block shows was intended -- and the oracles below come from
// a copy of Network.cs patched with exactly that one change, compiled against the real Numerics
// assembly (/tmp/getpath_probe/PatchedNetwork.cs). See network.hpp's transcription notes and
// docs/upstream-csharp-issues.md for the full write-up. Nothing else in the class is changed, so
// every number below is the real C# algorithm's output.

// DijkstraTesting.SimpleNetworkRouting's 10-node bidirectional grid, the graph all the Network
// oracles below were measured on.
//   0 - 1 - 2 - 3 - 4
//   |   | \ |   |   |
//   5 - 6 - 7 - 8 - 9
std::vector<Edge> routing_grid() {
    return {Edge(0, 5, 1, 0),  Edge(0, 1, 30, 1), Edge(1, 0, 30, 1), Edge(1, 2, 1, 2),
            Edge(1, 6, 15, 3), Edge(1, 7, 2, 4),  Edge(2, 1, 1, 2),  Edge(2, 3, 5, 5),
            Edge(2, 7, 5, 6),  Edge(3, 2, 5, 5),  Edge(3, 8, 2, 7),  Edge(3, 4, 1, 8),
            Edge(4, 3, 1, 8),  Edge(4, 9, 30, 9), Edge(5, 0, 1, 0),  Edge(5, 6, 3, 10),
            Edge(6, 5, 3, 10), Edge(6, 1, 15, 3), Edge(6, 7, 1, 11), Edge(7, 6, 1, 11),
            Edge(7, 1, 2, 4),  Edge(7, 2, 5, 6),  Edge(7, 8, 1, 12), Edge(8, 7, 1, 12),
            Edge(8, 3, 2, 7),  Edge(8, 9, 2, 13), Edge(9, 8, 2, 13), Edge(9, 4, 30, 9)};
}

void check_table(const dijkstra::ResultTable& actual, const float (&expected)[10][3]) {
    for (std::size_t i = 0; i < 10; ++i) {
        CHECK_EQ(expected[i][0], actual[i][dijkstra::NEXT_NODE]);
        CHECK_EQ(expected[i][1], actual[i][dijkstra::EDGE_INDEX]);
        CHECK_EQ(expected[i][2], actual[i][dijkstra::COST]);
    }
}

// The ctor's caching, and the one divergence. C# reports `IncomingEdges.Length == 9` only in the
// counterfactual where the indexing did not throw first; the patched C# reports 10, which is what
// the port produces.
void network_constructor_caches_both_directions() {
    Network network(routing_grid(), std::vector<int>{9});

    CHECK_EQ(std::size_t{10}, network.incoming_edges().size());
    CHECK_EQ(std::size_t{10}, network.outgoing_edges().size());
    CHECK_EQ(std::size_t{1}, network.destination_indices().size());
    CHECK_EQ(9, network.destination_indices()[0]);

    // Node 0 is entered from 1 and 5, and leaves to 5 and 1 (see the ASCII grid above).
    CHECK_EQ(std::size_t{2}, network.incoming_edges()[0].size());
    CHECK_EQ(1, network.incoming_edges()[0][0].from_index);
    CHECK_EQ(5, network.incoming_edges()[0][1].from_index);
    CHECK_EQ(std::size_t{2}, network.outgoing_edges()[0].size());
    CHECK_EQ(5, network.outgoing_edges()[0][0].to_index);
    CHECK_EQ(1, network.outgoing_edges()[0][1].to_index);

    // Every edge lands in exactly one incoming and one outgoing bucket.
    std::size_t incoming_total = 0, outgoing_total = 0;
    for (const auto& bucket : network.incoming_edges()) incoming_total += bucket.size();
    for (const auto& bucket : network.outgoing_edges()) outgoing_total += bucket.size();
    CHECK_EQ(std::size_t{28}, incoming_total);
    CHECK_EQ(std::size_t{28}, outgoing_total);
}

// C# `Network(edges, ...)` on an empty edge array hits LINQ `Max` on an empty sequence:
//   === ctor(no edges)
//     THREW System.InvalidOperationException: Sequence contains no elements
void network_constructor_rejects_an_empty_edge_set() {
    CHECK_THROWS_MSG(Network(std::vector<Edge>{}, std::vector<int>{0}),
                     "Sequence contains no elements.");
}

// Both index overloads are thin forwarders to `Dijkstra.Solve` over the cached incoming edges.
// The 30 values are the patched C#'s verbatim output; the run also confirmed
// `Solve(9) == Dijkstra.Solve(grid, 9)` element for element ("identical to free Dijkstra.Solve:
// True"), which is the whole point of the class.
void network_solve_single_destination_matches_csharp() {
    Network network(routing_grid(), std::vector<int>{9});
    const float expected[10][3] = {{5, 0, 8},  {7, 4, 5},  {1, 2, 6},  {8, 7, 4}, {3, 8, 5},
                                   {6, 10, 7}, {7, 11, 4}, {8, 12, 3}, {9, 13, 2}, {9, -1, 0}};
    check_table(network.solve(9), expected);

    // ... and it is the same table the free function produces.
    auto free_table = dijkstra::solve(routing_grid(), 9);
    check_table(free_table, expected);
}

void network_solve_multiple_destinations_matches_csharp() {
    Network network(routing_grid(), std::vector<int>{0, 9});
    const float expected[10][3] = {{0, -1, 0}, {7, 4, 5},  {1, 2, 6},  {8, 7, 4}, {3, 8, 5},
                                   {0, 0, 1},  {5, 10, 4}, {8, 12, 3}, {9, 13, 2}, {9, -1, 0}};
    check_table(network.solve(std::vector<int>{0, 9}), expected);
}

// UPSTREAM DEFECT, reproduced on purpose (network.hpp note 5): `Solve(float[] edgeWeights)`
// builds a re-weighted edge array and then hands `Dijkstra.Solve` the STALE `_incomingEdges`
// cache, whose length already matches nNodes -- so the solver never looks at the re-weighted
// array and the custom weights are silently ignored. MEASURED in the patched C#: feeding
// all-ones weights to this 30-and-15-weighted grid returns the ORIGINAL table, not the
// unit-weight one.
void network_solve_with_weights_ignores_its_own_argument() {
    auto grid = routing_grid();
    Network network(grid, std::vector<int>{9});

    std::vector<float> ones(grid.size(), 1.0f);
    const float unchanged[10][3] = {{5, 0, 8},  {7, 4, 5},  {1, 2, 6},  {8, 7, 4}, {3, 8, 5},
                                    {6, 10, 7}, {7, 11, 4}, {8, 12, 3}, {9, 13, 2}, {9, -1, 0}};
    check_table(network.solve(ones), unchanged);

    // What an honest custom-weight solve would have returned, for contrast (also measured in C#,
    // by calling the free `Dijkstra.Solve` on a genuinely re-weighted edge array).
    std::vector<Edge> unit_edges;
    for (const auto& edge : grid) {
        unit_edges.push_back(Edge(edge.from_index, edge.to_index, 1.0f, edge.index));
    }
    const float honest[10][3] = {{1, 1, 4},  {7, 4, 3},  {3, 5, 3},  {4, 8, 2}, {9, 9, 1},
                                 {6, 10, 4}, {7, 11, 3}, {8, 12, 2}, {9, 13, 1}, {9, -1, 0}};
    check_table(dijkstra::solve(unit_edges, std::vector<int>{9}), honest);
}

// UPSTREAM DEFECT (network.hpp note 6). `Array.BinarySearch(int[] edgesToRemove, Edge edge)`
// binds the `(Array, object)` overload, which boxes the Edge and asks an Int32 to compare itself
// against it. MEASURED, both in isolation and through the patched `GetPath`:
//   === Array.BinarySearch(int[]{0,1,2}, (object)Edge)
//     THREW System.InvalidOperationException: Failed to compare two elements in the array.
//     INNER System.ArgumentException: Object must be of type Int32.
// so ANY call carrying a non-empty removal list dies on the first edge it examines.
void get_path_throws_on_a_non_empty_removal_list() {
    Network network(routing_grid(), std::vector<int>{9});
    CHECK_THROWS_MSG(network.get_path(std::vector<int>{0}, 0),
                     "Failed to compare two elements in the array.");
    CHECK_THROWS_MSG(network.get_path(std::vector<int>{0}, 0, network.solve(9)),
                     "Failed to compare two elements in the array.");
}

// The complement: a ZERO-LENGTH binary search never invokes the comparer, so an empty removal
// list is the one input that gets past the throw. MEASURED: `Array.BinarySearch(int[0],
// (object)Edge)` returns -1, and the two patched-C# GetPath overloads then return, respectively,
// null and an EMPTY list -- never a path. Taken together with the case above, that is the
// complete observable behavior of both overloads: throw, null, or empty.
void get_path_with_an_empty_removal_list_never_returns_a_path() {
    Network network(routing_grid(), std::vector<int>{9});

    // === GetPath(empty, 0)  ->  returned null
    CHECK_TRUE(!network.get_path(std::vector<int>{}, 0).has_value());
    // === GetPath(empty, 9)  ->  returned null
    CHECK_TRUE(!network.get_path(std::vector<int>{}, 9).has_value());

    // === GetPath(empty, 0, table)  ->  returned []
    auto with_table = network.get_path(std::vector<int>{}, 0, dijkstra::solve(routing_grid(), 9));
    CHECK_TRUE(with_table.has_value());
    CHECK_EQ(std::size_t{0}, with_table->size());
}

// A start node with no incoming edges cannot load the heap at all, so the removal list is never
// consulted and both overloads short-circuit before the defect above.
//   === GetPath(new[]{0}, isolated node) on a padded graph  ->  returned null
void get_path_returns_null_when_the_start_node_has_no_incoming_edges() {
    auto edges = routing_grid();
    edges.push_back(Edge(10, 0, 1, 14));  // node 10 emits an edge and receives none
    Network network(edges, std::vector<int>{9});

    CHECK_TRUE(!network.get_path(std::vector<int>{0}, 10).has_value());
    CHECK_TRUE(!network.get_path(std::vector<int>{}, 10).has_value());
}

// ============================== SUPPLEMENT ==============================
// NOT from either C# test file. Two things the 17 transcriptions above leave completely
// unguarded, both measured against the REAL Numerics library at 2a0357a rather than derived:
//
//   (a) SINGLE PRECISION. Every weight in the 8 Dijkstra tests is a small integer, and float and
//       double agree exactly on those, so nothing above can tell a `float` port from a `double`
//       one -- MEASURED: widening the port's Node weight, Edge weight and result table to double
//       leaves all 1,084 transcribed checks green (55 of them Dijkstra's) and fails exactly six
//       checks, all of them in this function. The
//       graph is a 10-hop chain of 0.1f edges against a two-hop 0.3f + 0.7f bypass; the chain
//       accumulates to 1.0000001 in single precision while the bypass lands on exactly 1, so the
//       intermediate costs are single-precision partial sums and nothing else.
//
//   (b) `Dijkstra.PathExists`, which no upstream test calls at all.
//
//   (c) a `node_count` too small for the graph, added by the branch review that caught the port
//       answering it with an out-of-bounds write. See that function's own comment.
//
// The 36 values below are the verbatim G17 output of a scratch dotnet console app
// (/tmp/netprobe, referencing upstream/Numerics/Numerics/Numerics.csproj at 2a0357a) running
// this exact graph through the real `Dijkstra.Solve(edges, 10, 12)`. The port reproduces all 36
// bit-for-bit.
void fractional_weights_reproduce_csharp_single_precision() {
    std::vector<Edge> edges;
    for (int i = 0; i < 10; ++i) edges.push_back(Edge(i, i + 1, 0.1f, i));
    edges.push_back(Edge(0, 11, 0.3f, 10));
    edges.push_back(Edge(11, 10, 0.7f, 11));

    auto result = dijkstra::solve(edges, 10, 12);

    // node 0 routes over the 0.3f + 0.7f bypass, which in single precision costs exactly 1,
    // beating the ten-hop 0.1f chain's 1.0000001.
    CHECK_EQ(11.f, result[0][dijkstra::NEXT_NODE]);
    CHECK_EQ(10.f, result[0][dijkstra::EDGE_INDEX]);
    CHECK_EQ(1.f, result[0][dijkstra::COST]);

    CHECK_EQ(0.90000009536743164f, result[1][dijkstra::COST]);
    CHECK_EQ(0.80000007152557373f, result[2][dijkstra::COST]);
    CHECK_EQ(0.70000004768371582f, result[3][dijkstra::COST]);
    CHECK_EQ(0.60000002384185791f, result[4][dijkstra::COST]);
    CHECK_EQ(0.5f, result[5][dijkstra::COST]);
    CHECK_EQ(0.40000000596046448f, result[6][dijkstra::COST]);
    CHECK_EQ(0.30000001192092896f, result[7][dijkstra::COST]);
    CHECK_EQ(0.20000000298023224f, result[8][dijkstra::COST]);
    CHECK_EQ(0.10000000149011612f, result[9][dijkstra::COST]);
    CHECK_EQ(0.f, result[10][dijkstra::COST]);
    CHECK_EQ(0.69999998807907104f, result[11][dijkstra::COST]);

    for (int i = 1; i <= 9; ++i) {
        CHECK_EQ(static_cast<float>(i + 1), result[i][dijkstra::NEXT_NODE]);
        CHECK_EQ(static_cast<float>(i), result[i][dijkstra::EDGE_INDEX]);
    }
    CHECK_EQ(10.f, result[10][dijkstra::NEXT_NODE]);
    CHECK_EQ(-1.f, result[10][dijkstra::EDGE_INDEX]);
    CHECK_EQ(10.f, result[11][dijkstra::NEXT_NODE]);
    CHECK_EQ(11.f, result[11][dijkstra::EDGE_INDEX]);
}

// `Dijkstra.PathExists` is untested upstream. It is a one-line read of the COST column, so the
// oracle is the same DisconnectedNodesTest table the C# already pins.
void path_exists_reads_the_cost_column() {
    std::vector<Edge> edges = {
        Edge(0, 1, 1, 0),
        Edge(1, 2, 1, 1),
        // Node 3 is disconnected
    };
    auto result = dijkstra::solve(edges, 2, 4);
    CHECK_TRUE(dijkstra::path_exists(result, 0));
    CHECK_TRUE(dijkstra::path_exists(result, 1));
    CHECK_TRUE(dijkstra::path_exists(result, 2));
    CHECK_TRUE(!dijkstra::path_exists(result, 3));
}

// A `node_count` too small for the graph. Added after a branch review found that the port
// answered it with an out-of-bounds WRITE (AddressSanitizer: heap-buffer-overflow in
// build_edges_to_nodes) where C# raises IndexOutOfRangeException -- a silent wrong table in R and
// a crash in Python. Every expectation below is the verbatim behavior of the real Numerics
// library at 2a0357a, measured with a scratch dotnet console app (/tmp/getpath_probe) and
// transcribed here; see dijkstra.hpp note 9 for the full probe transcript.
//
// The last two cases are the ones that decide HOW the guard is written: C#'s throw is LAZY (it
// is the CLR bounds-checking an indexing expression, not a validation pass), so an out-of-range
// index on an edge the search never relaxes must still RETURN a table. An up-front sweep of the
// edge list would pass every other case here and fail that one.
void a_node_count_below_the_graph_throws_like_csharp() {
    const std::vector<Edge> to_out_of_range = {Edge(0, 1, 1, 0), Edge(1, 5, 1, 1)};
    CHECK_THROWS_MSG(dijkstra::solve(to_out_of_range, 0, 2),
                     "Index was outside the bounds of the array.");
    const std::vector<int> destinations = {0};
    CHECK_THROWS_MSG(dijkstra::solve(to_out_of_range, destinations, 2),
                     "Index was outside the bounds of the array.");

    // FromIndex out of range, on an edge the search DOES relax.
    const std::vector<Edge> from_out_of_range = {Edge(5, 1, 1, 0), Edge(0, 1, 1, 1)};
    CHECK_THROWS_MSG(dijkstra::solve(from_out_of_range, 1, 2),
                     "Index was outside the bounds of the array.");
    const std::vector<Edge> from_out_of_range2 = {Edge(0, 1, 1, 0), Edge(7, 0, 1, 1)};
    CHECK_THROWS_MSG(dijkstra::solve(from_out_of_range2, 1, 2),
                     "Index was outside the bounds of the array.");

    // A destination past the end of the table.
    const std::vector<Edge> one_edge = {Edge(0, 1, 1, 0)};
    CHECK_THROWS_MSG(dijkstra::solve(one_edge, 5, 2),
                     "Index was outside the bounds of the array.");
    // ... and PathExists, which C# indexes the same way.
    auto table = dijkstra::solve(one_edge, 1, 2);
    CHECK_THROWS_MSG(dijkstra::path_exists(table, 2),
                     "Index was outside the bounds of the array.");

    // THE LAZY CASE. Node 1 is never reached from destination 0, so the edge carrying the
    // out-of-range FromIndex 7 is never relaxed and C# returns this exact table.
    const std::vector<Edge> never_relaxed = {Edge(0, 1, 1, 0), Edge(7, 1, 1, 1)};
    auto lazy = dijkstra::solve(never_relaxed, 0, 2);
    CHECK_EQ(std::size_t(2), lazy.size());
    CHECK_EQ(0.f, lazy[0][dijkstra::NEXT_NODE]);
    CHECK_EQ(-1.f, lazy[0][dijkstra::EDGE_INDEX]);
    CHECK_EQ(0.f, lazy[0][dijkstra::COST]);
    CHECK_EQ(-1.f, lazy[1][dijkstra::NEXT_NODE]);
    CHECK_EQ(-1.f, lazy[1][dijkstra::EDGE_INDEX]);
    CHECK_TRUE(std::isinf(lazy[1][dijkstra::COST]) && lazy[1][dijkstra::COST] > 0.f);
}

}  // namespace

int main() {
    // BinaryHeapTesting.cs
    heap_test1();
    heap_test2();
    heap_test3();
    heap_test4();
    decrease_key_test();
    heap_capacity_exceeded_test();
    remove_min_from_empty_heap_test();
    replace_with_higher_weight_should_do_nothing_test();
    ordering_with_negative_weights_test();
    // DijkstraTesting.cs (class ShortestPathTesting)
    simple_edge_graph_cost();
    simple_network_routing();
    bidirectional_routing();
    disconnected_nodes_test();
    multiple_dest_shared_path();
    disconnected_component();
    disconnected_component2();
    triangle_path();
    // Network (P3 Task 9) -- no upstream test class; oracles measured off the real C# library
    network_constructor_caches_both_directions();
    network_constructor_rejects_an_empty_edge_set();
    network_solve_single_destination_matches_csharp();
    network_solve_multiple_destinations_matches_csharp();
    network_solve_with_weights_ignores_its_own_argument();
    get_path_throws_on_a_non_empty_removal_list();
    get_path_with_an_empty_removal_list_never_returns_a_path();
    get_path_returns_null_when_the_start_node_has_no_incoming_edges();
    // Supplement (P3 Task 8): single precision, and the untested PathExists
    fractional_weights_reproduce_csharp_single_precision();
    path_exists_reads_the_cost_column();
    a_node_count_below_the_graph_throws_like_csharp();
    return chtest::summary("test_network_optimization");
}
