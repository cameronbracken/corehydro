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
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

using corehydro::numerics::math::optimization::BinaryHeap;
using corehydro::numerics::math::optimization::Edge;
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
    // Supplement (P3 Task 8): single precision, and the untested PathExists
    fractional_weights_reproduce_csharp_single_precision();
    path_exists_reads_the_cost_column();
    return chtest::summary("test_network_optimization");
}
