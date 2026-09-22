// C++ coverage for Numerics v2.2.0 dynamic shortest paths, ported from:
//   Test_Numerics/Mathematics/Optimization/Dynamic/Test_{BinaryHeap,IndexedMinHeap,Network,
//   ShortestPath}.cs @ 7e8e8d1
// The real upstream suite is also run as a separate gate. These checks pin the shared C++ public
// contracts, exact float routing tables, host-facing validation, and dense-graph capacity.
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/math/optimization/dynamic/binary_heap.hpp"
#include "corehydro/numerics/math/optimization/dynamic/dijkstra.hpp"
#include "corehydro/numerics/math/optimization/dynamic/indexed_min_heap.hpp"
#include "corehydro/numerics/math/optimization/dynamic/network.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

using corehydro::numerics::math::optimization::BinaryHeap;
using corehydro::numerics::math::optimization::Edge;
using corehydro::numerics::math::optimization::IndexedMinHeap;
using corehydro::numerics::math::optimization::Network;
namespace dijkstra = corehydro::numerics::math::optimization::dijkstra;
using corehydro::numerics::sampling::MersenneTwister;

namespace {

std::vector<Edge> triangle() {
    return {{0, 1, 1.0f, 0}, {1, 0, 4.0f, 1}, {1, 2, 1.0f, 2}, {2, 0, 10.0f, 3}};
}

void check_row(const dijkstra::ResultTable& table, int row, float next, float edge, float cost) {
    CHECK_EQ(table[static_cast<std::size_t>(row)][0], next);
    CHECK_EQ(table[static_cast<std::size_t>(row)][1], edge);
    CHECK_EQ(table[static_cast<std::size_t>(row)][2], cost);
}

void binary_heap_contract() {
    using Heap = BinaryHeap<int>;
    Heap heap(5);
    heap.add({1.0f, 1, 10});
    heap.add({2.0f, 2, 20});
    heap.add({3.0f, 3, 30});
    heap.replace({10.0f, 1, 11});
    CHECK_EQ(heap.remove_min().index, 2);
    CHECK_EQ(heap.remove_min().index, 3);
    auto last = heap.remove_min();
    CHECK_EQ(last.index, 1);
    CHECK_EQ(last.value, 11);

    heap.replace({0.0f, 99, 0});
    CHECK_EQ(heap.count(), 0);
    heap.decrease_key({-2.0f, 4, 40});
    CHECK_EQ(heap.remove_min().index, 4);
    CHECK_THROWS(heap.remove_min());

    Heap full(1);
    full.add({0.0f, 0, 0});
    CHECK_THROWS(full.add({1.0f, 1, 1}));
}

void binary_heap_random_order() {
    std::vector<float> weights(1000);
    MersenneTwister random(12345);
    BinaryHeap<int> heap(static_cast<int>(weights.size()));
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>(random.next_double());
        heap.add({weights[i], static_cast<int>(i), static_cast<int>(i)});
    }
    std::sort(weights.begin(), weights.end());
    for (float expected : weights) CHECK_EQ(heap.remove_min().weight, expected);
}

void indexed_heap_contract() {
    IndexedMinHeap heap(4);
    heap.add(0, 4.0f);
    heap.add(1, 3.0f);
    heap.add(2, 2.0f);
    heap.decrease_key(0, 1.0f);
    int node = -1;
    float weight = 0.0f;
    heap.remove_min(node, weight);
    CHECK_EQ(node, 0);
    CHECK_EQ(weight, 1.0f);
    heap.decrease_key(3, -1.0f);
    heap.remove_min(node, weight);
    CHECK_EQ(node, 3);
    heap.clear();
    CHECK_EQ(heap.count(), 0);
    CHECK_THROWS(heap.remove_min(node, weight));

    IndexedMinHeap full(1);
    full.add(0, 0.0f);
    CHECK_THROWS(full.add(0, 1.0f));
}

void single_and_merged_solves() {
    auto edges = triangle();
    auto table = dijkstra::solve(edges, 0, 3);
    check_row(table, 0, 0.0f, -1.0f, 0.0f);
    check_row(table, 1, 0.0f, 1.0f, 4.0f);
    check_row(table, 2, 0.0f, 3.0f, 10.0f);

    auto merged = dijkstra::solve(edges, std::vector<int>{0, 2}, 3);
    check_row(merged, 0, 0.0f, -1.0f, 0.0f);
    check_row(merged, 1, 2.0f, 2.0f, 1.0f);
    check_row(merged, 2, 2.0f, -1.0f, 0.0f);

    auto repeated = dijkstra::solve_nearest(edges, std::vector<int>{0, 0, 2}, 3);
    for (std::size_t i = 0; i < repeated.size(); ++i)
        for (int j = 0; j < 3; ++j) CHECK_EQ(repeated[i][j], merged[i][j]);

    auto none = dijkstra::solve(edges, std::vector<int>{}, 3);
    for (const auto& row : none) {
        CHECK_EQ(row[0], -1.0f);
        CHECK_EQ(row[1], -1.0f);
        CHECK_TRUE(std::isinf(row[2]));
    }
}

void tie_and_float_contracts() {
    std::vector<Edge> ties{{0, 1, 1.0f, 10}, {0, 2, 1.0f, 11}};
    auto first = dijkstra::solve(ties, std::vector<int>{1, 2}, 3);
    auto second = dijkstra::solve(ties, std::vector<int>{2, 1}, 3);
    CHECK_EQ(first[0][0], 1.0f);
    CHECK_EQ(second[0][0], 2.0f);

    std::vector<Edge> chain;
    for (int i = 0; i < 10; ++i) chain.emplace_back(i, i + 1, 0.1f, i);
    auto table = dijkstra::solve(chain, 10, 11);
    CHECK_EQ(table[0][2], 1.0000001192092896f);
    CHECK_EQ(table[1][2], 0.90000009536743164f);
}

void path_reconstruction() {
    std::vector<Edge> edges{{0, 1, 1.0f, 7}, {1, 2, 2.0f, 8}, {3, 4, 1.0f, 9}};
    auto table = dijkstra::solve(edges, 2, 5);
    CHECK_TRUE(dijkstra::path_exists(table, 0));
    CHECK_TRUE(!dijkstra::path_exists(table, 4));
    auto path = dijkstra::get_path(table, 0);
    CHECK_TRUE(path.has_value());
    CHECK_EQ(path->size(), std::size_t{2});
    CHECK_EQ((*path)[0], 7);
    CHECK_EQ((*path)[1], 8);
    auto destination = dijkstra::get_path(table, 2);
    CHECK_TRUE(destination.has_value());
    CHECK_TRUE(destination->empty());
    CHECK_TRUE(!dijkstra::get_path(table, 4).has_value());

    auto inconsistent = table;
    inconsistent[0][0] = 99.0f;
    CHECK_THROWS(dijkstra::get_path(inconsistent, 0));
    inconsistent = table;
    inconsistent[0] = {0.0f, 7.0f, 1.0f};
    CHECK_THROWS(dijkstra::get_path(inconsistent, 0));
    CHECK_THROWS(dijkstra::path_exists(table, -1));
    auto negative_infinity = table;
    negative_infinity[4][2] = -std::numeric_limits<float>::infinity();
    CHECK_TRUE(dijkstra::path_exists(negative_infinity, 4));
}

std::vector<Edge> routing_grid() {
    return {{0, 5, 1, 0},   {0, 1, 30, 1}, {1, 0, 30, 1}, {1, 2, 1, 2},
            {1, 6, 15, 3},  {1, 7, 2, 4},  {2, 1, 1, 2},   {2, 3, 5, 5},
            {2, 7, 5, 6},   {3, 2, 5, 5},  {3, 8, 2, 7},   {3, 4, 1, 8},
            {4, 3, 1, 8},   {4, 9, 30, 9}, {5, 0, 1, 0},   {5, 6, 3, 10},
            {6, 5, 3, 10},  {6, 1, 15, 3}, {6, 7, 1, 11},  {7, 6, 1, 11},
            {7, 1, 2, 4},   {7, 2, 5, 6},  {7, 8, 1, 12},  {8, 7, 1, 12},
            {8, 3, 2, 7},   {8, 9, 2, 13}, {9, 8, 2, 13},  {9, 4, 30, 9}};
}

void routing_grid_oracle() {
    auto table = dijkstra::solve(routing_grid(), 9);
    const float expected[10][3] = {{5, 0, 8},  {7, 4, 5},  {1, 2, 6},  {8, 7, 4},
                                   {3, 8, 5},  {6, 10, 7}, {7, 11, 4}, {8, 12, 3},
                                   {9, 13, 2}, {9, -1, 0}};
    for (int row = 0; row < 10; ++row)
        for (int column = 0; column < 3; ++column)
            CHECK_EQ(table[static_cast<std::size_t>(row)][column], expected[row][column]);
}

void validation_contracts() {
    auto edges = triangle();
    CHECK_THROWS(dijkstra::solve({}, 0));
    CHECK_THROWS(dijkstra::solve(edges, 0, 0));
    CHECK_THROWS(dijkstra::solve(edges, 3, 3));
    CHECK_THROWS(dijkstra::solve({Edge{-1, 0, 1.0f, 0}}, 0, 1));
    CHECK_THROWS(dijkstra::solve({Edge{0, 1, 1.0f, -1}}, 0, 2));
    CHECK_THROWS(dijkstra::solve_nearest(edges, {}, 3));

    std::vector<Edge> special{{0, 1, -1.0f, 0},
                              {1, 2, std::numeric_limits<float>::infinity(), 1},
                              {0, 2, std::numeric_limits<float>::quiet_NaN(), 2}};
    auto table = dijkstra::solve(special, 2, 3);
    CHECK_TRUE(std::isinf(table[0][2]));
}

void network_solve_contracts() {
    auto edges = triangle();
    Network network(edges, {0, 2});
    CHECK_EQ(network.node_count(), 3);
    CHECK_EQ(network.incoming_edges().size(), std::size_t{3});
    CHECK_EQ(network.outgoing_edges().size(), std::size_t{3});

    edges[0].weight = 99.0f;
    auto original = network.solve(std::vector<int>{0, 2});
    CHECK_EQ(original[0][2], 0.0f);
    CHECK_EQ(original[1][2], 1.0f);

    std::vector<float> ones(4, 1.0f);
    auto weighted = network.solve(ones);
    check_row(weighted, 0, 0.0f, -1.0f, 0.0f);
    check_row(weighted, 1, 0.0f, 1.0f, 1.0f);
    check_row(weighted, 2, 2.0f, -1.0f, 0.0f);
    auto nearest = network.solve_nearest();
    for (std::size_t i = 0; i < nearest.size(); ++i)
        CHECK_EQ(nearest[i][2], original[i][2]);

    CHECK_THROWS(network.solve(std::vector<float>{1.0f}));
    CHECK_THROWS(network.solve(3));
    CHECK_THROWS(Network({}, {0}));
    CHECK_THROWS(Network(triangle(), {}));
}

void network_detours() {
    std::vector<Edge> edges{{0, 1, 1.0f, 5}, {1, 3, 1.0f, 6}, {0, 2, 2.0f, 7},
                            {2, 3, 2.0f, 8}, {0, 2, 3.0f, 7}};
    Network network(edges, {3});
    auto direct = network.get_path({}, 0);
    CHECK_TRUE(direct.has_value());
    CHECK_EQ(*direct, (std::vector<int>{5, 6}));
    auto detour = network.get_path({5}, 0);
    CHECK_TRUE(detour.has_value());
    CHECK_EQ(*detour, (std::vector<int>{7, 8}));
    auto duplicate_removed = network.get_path({7}, 0);
    CHECK_TRUE(duplicate_removed.has_value());
    CHECK_EQ(*duplicate_removed, (std::vector<int>{5, 6}));
    auto at_destination = network.get_path({5}, 3);
    CHECK_TRUE(at_destination.has_value());
    CHECK_TRUE(at_destination->empty());
    CHECK_TRUE(!network.get_path({5, 7}, 0).has_value());

    auto table = network.solve(3);
    CHECK_EQ(network.get_path({}, 0, table), (std::vector<int>{5, 6}));
    CHECK_EQ(network.get_path({5}, 0, table), (std::vector<int>{7, 8}));

    std::vector<float> weights{10.0f, 1.0f, 1.0f, 1.0f, 2.0f};
    auto custom_table = network.solve(weights);
    CHECK_EQ(network.get_path({7}, 0, custom_table, weights),
             (std::vector<int>{5, 6}));
}

void dense_capacity_contract() {
    constexpr int n = 40;
    std::vector<Edge> edges;
    int index = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) edges.emplace_back(i, j, static_cast<float>((i + j) % 7 + 1), index++);
    auto table = dijkstra::solve(edges, n - 1, n);
    CHECK_EQ(table.size(), std::size_t{n});
    CHECK_EQ(table[static_cast<std::size_t>(n - 1)][2], 0.0f);
}

}  // namespace

int main() {
    binary_heap_contract();
    binary_heap_random_order();
    indexed_heap_contract();
    single_and_merged_solves();
    tie_and_float_contracts();
    path_reconstruction();
    routing_grid_oracle();
    validation_contracts();
    network_solve_contracts();
    network_detours();
    dense_capacity_contract();
    return chtest::summary("network_optimization");
}
