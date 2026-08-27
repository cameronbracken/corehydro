// corehydro ADDITION -- shared .NET sort port, no single upstream C# counterpart.
//
// A faithful port of .NET's `ArraySortHelper<T>.IntrospectiveSort` plus the
// `System.Double.CompareTo` ordering it is driven by. `List<T>.Sort(Comparison<T>)` and
// `Array.Sort(T[], Comparison<T>)` both route through it, and it is a
// deterministic-but-NOT-stable introspective sort: for tied keys it produces a specific
// permutation that neither `std::sort` nor `std::stable_sort` reproduces. Wherever a ported class
// sorts a COMPARISON-carrying array whose keys can tie, that permutation is oracle-visible and
// this is the sort to use.
//
// Two ported call sites need it today:
//   - RMC.BestFit `DataFrame::calculate_plotting_positions` (the legacy above/below tie
//     permutation in the Hirsch-Stedinger plotting positions -- see
//     models/data_frame/data_frame_plotting.hpp's file header for why that permutation is
//     load-bearing). This is where the code originally lived and where it was verified against
//     dotnet 10 output.
//   - Numerics `KNearestNeighbors` (the `kNNItem[]` distance sort; equidistant training points
//     tie exactly in upstream's own `Test_GetNeighbors_MultiRow`).
//
// P5 Task 1 moved the block here VERBATIM from `corehydro::models::detail` so the `numerics/`
// layer can reach it without depending on `models/`; `data_frame_plotting.hpp` now includes this
// header and re-exports the names into its own `detail` namespace, so every pre-existing call
// site is untouched and the plotting-position fixtures are the regression proof that the move
// changed nothing.
//
// NOT for primitive sorts. `Array.Sort(double[])` / `Array.Sort(int[])` sort VALUES, where tied
// elements are indistinguishable and no permutation is observable, so `std::sort` is both correct
// and faster there. Use this only for a comparison over a struct or a key/payload pair.
#pragma once
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace corehydro::numerics::utilities {

// C# System.Double.CompareTo semantics: NaN sorts before everything and equals NaN.
inline int compare_double(double x, double y) {
    if (x < y) return -1;
    if (x > y) return 1;
    if (x == y) return 0;
    if (std::isnan(x)) return std::isnan(y) ? 0 : -1;
    return 1;
}

// --- .NET introsort port (see the file header). `Compare` returns an int with C#
// Comparison<T> semantics: negative / zero / positive. ---

template <typename T, typename Compare>
inline void dn_swap_if_greater(T* keys, Compare& cmp, int i, int j) {
    if (cmp(keys[i], keys[j]) > 0) std::swap(keys[i], keys[j]);
}

template <typename T, typename Compare>
inline void dn_insertion_sort(T* keys, int length, Compare& cmp) {
    for (int i = 0; i < length - 1; i++) {
        T t = keys[i + 1];
        int j = i;
        while (j >= 0 && cmp(t, keys[j]) < 0) {
            keys[j + 1] = keys[j];
            j--;
        }
        keys[j + 1] = t;
    }
}

template <typename T, typename Compare>
inline void dn_down_heap(T* keys, int i, int n, Compare& cmp) {
    T d = keys[i - 1];
    while (i <= n / 2) {
        int child = 2 * i;
        if (child < n && cmp(keys[child - 1], keys[child]) < 0) child++;
        if (!(cmp(d, keys[child - 1]) < 0)) break;
        keys[i - 1] = keys[child - 1];
        i = child;
    }
    keys[i - 1] = d;
}

template <typename T, typename Compare>
inline void dn_heap_sort(T* keys, int length, Compare& cmp) {
    int n = length;
    for (int i = n / 2; i >= 1; i--) dn_down_heap(keys, i, n, cmp);
    for (int i = n; i > 1; i--) {
        std::swap(keys[0], keys[i - 1]);
        dn_down_heap(keys, 1, i - 1, cmp);
    }
}

template <typename T, typename Compare>
inline int dn_pick_pivot_and_partition(T* keys, int length, Compare& cmp) {
    int hi = length - 1;
    // Median-of-three pivot: sort lo/mid/hi, pivot = mid, parked at hi - 1.
    int middle = hi >> 1;
    dn_swap_if_greater(keys, cmp, 0, middle);
    dn_swap_if_greater(keys, cmp, 0, hi);
    dn_swap_if_greater(keys, cmp, middle, hi);
    T pivot = keys[middle];
    std::swap(keys[middle], keys[hi - 1]);
    int left = 0, right = hi - 1;
    while (left < right) {
        while (cmp(keys[++left], pivot) < 0) {
        }
        while (cmp(pivot, keys[--right]) < 0) {
        }
        if (left >= right) break;
        std::swap(keys[left], keys[right]);
    }
    if (left != hi - 1) std::swap(keys[left], keys[hi - 1]);
    return left;
}

template <typename T, typename Compare>
inline void dn_intro_sort(T* keys, int length, int depth_limit, Compare& cmp) {
    const int kIntrosortSizeThreshold = 16;
    int partition_size = length;
    while (partition_size > 1) {
        if (partition_size <= kIntrosortSizeThreshold) {
            if (partition_size == 2) {
                dn_swap_if_greater(keys, cmp, 0, 1);
                return;
            }
            if (partition_size == 3) {
                dn_swap_if_greater(keys, cmp, 0, 1);
                dn_swap_if_greater(keys, cmp, 0, 2);
                dn_swap_if_greater(keys, cmp, 1, 2);
                return;
            }
            dn_insertion_sort(keys, partition_size, cmp);
            return;
        }
        if (depth_limit == 0) {
            dn_heap_sort(keys, partition_size, cmp);
            return;
        }
        depth_limit--;
        int p = dn_pick_pivot_and_partition(keys, partition_size, cmp);
        dn_intro_sort(keys + p + 1, partition_size - (p + 1), depth_limit, cmp);
        partition_size = p;
    }
}

// C# List<T>.Sort(Comparison<T>): introsort with depth limit 2*(floor(log2(n)) + 1).
template <typename T, typename Compare>
inline void dotnet_list_sort(std::vector<T>& items, Compare cmp) {
    if (items.size() <= 1) return;
    unsigned n = static_cast<unsigned>(items.size());
    int log2n = 0;
    while (n >>= 1) log2n++;
    dn_intro_sort(items.data(), static_cast<int>(items.size()), 2 * (log2n + 1), cmp);
}

}  // namespace corehydro::numerics::utilities
