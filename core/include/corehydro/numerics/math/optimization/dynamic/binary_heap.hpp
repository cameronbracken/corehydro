// ported from: Numerics/Mathematics/Optimization/Dynamic/BinaryHeap.cs @ 2a0357a
//
// An implicit binary min-heap over a fixed-capacity array, with an index-to-position map so a
// node's key can be decreased in place. It exists for the shortest-path solver next door
// (dijkstra.hpp); the C# doc comment names Dijkstra as the motivating consumer and cites
// http://opendatastructures.org/versions/edition-0.1e/ods-java/10_1_BinaryHeap_Implicit_Bi.html.
//
// Transcription notes:
//
// 1. WEIGHTS ARE `float`, NOT `double`, throughout this file and dijkstra.hpp. C# declares
//    `public float Weight` on the node, and the solver next door accumulates path costs through
//    it in `float`. Single-precision rounding is observable in the shortest-path result table --
//    measured, not assumed; see dijkstra.hpp's note 1 -- so widening here would silently change
//    the numbers the oracle pins. The toolbox boundary is where the widening to double belongs.
//
// 2. `Dictionary<int, int> _positionMap` becomes `std::unordered_map<int, int>`. Only membership,
//    lookup, insert and erase are used -- never iteration -- so the container's ordering is not
//    observable and the mapping is exact.
//
// 3. `Node[] _heap` (fixed length `heapSize`, default-initialized) becomes a `std::vector<Node>`
//    sized in the constructor and never resized; `add` mirrors C#'s capacity check against the
//    array LENGTH rather than growing. Node therefore needs a default constructor, which C#
//    structs get for free (all fields zero/default).
//
// 4. Exception-type mapping for this file: C# `InvalidOperationException` -> `std::runtime_error`
//    (the repo's established mapping), with the message strings transcribed verbatim ("Heap is
//    full.", "Heap is empty.").
//
// 5. `Add` calls `BubbleUp(_n)` BEFORE incrementing `_n`. BubbleUp never reads `_n`, so the order
//    is immaterial -- transcribed as written anyway.
//
// 6. `Replace` scans linearly for the matching index and only ever bubbles UP, so replacing with a
//    heavier node leaves the heap out of order. That is upstream behavior, exercised by
//    BinaryHeapTesting's HeapTest3/HeapTest4, and is not "fixed" here.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace corehydro::numerics::math::optimization {

// C# `public class BinaryHeap<T>`. T is the payload carried alongside the priority; the solver
// stores the Edge that produced the tentative cost.
template <typename T>
class BinaryHeap {
   public:
    // Represents a node in the binary heap with a weight, index, and value.
    struct Node {
        // The weight (priority) of the node.
        float weight;
        // The index identifier of the node.
        int index;
        // The value stored in the node.
        T value;

        // C# structs are default-initialized inside `new Node[heapSize]`; see header note 3.
        Node() : weight(0.0f), index(0), value() {}

        // Creates a new node with the specified weight, index, and value.
        Node(float node_weight, int node_index, T node_value)
            : weight(node_weight), index(node_index), value(std::move(node_value)) {}
    };

    // Creates a new binary heap with the specified maximum size.
    explicit BinaryHeap(int heap_size)
        : heap_(heap_size <= 0 ? 0 : static_cast<std::size_t>(heap_size)) {}

    // The number of nodes in the heap.
    int count() const { return n_; }

    // Add a node to the heap.
    void add(const Node& node) {
        if (n_ >= static_cast<int>(heap_.size())) throw std::runtime_error("Heap is full.");

        heap_[static_cast<std::size_t>(n_)] = node;
        // Map the index to the position in the heap array (for Replace method)
        position_map_[node.index] = n_;
        bubble_up(n_);
        n_++;
    }

    // Remove the minimum (top) node from the heap.
    Node remove_min() {
        if (n_ == 0) throw std::runtime_error("Heap is empty.");

        Node min = heap_[0];
        position_map_.erase(min.index);

        n_--;

        if (n_ > 0) {
            heap_[0] = heap_[static_cast<std::size_t>(n_)];
            position_map_[heap_[0].index] = 0;  // Update position map
            bubble_down(0);
        }
        return min;
    }

    // Updates the distance (priority) of a node if a shorter path is found.
    void decrease_key(const Node& new_node) {
        auto it = position_map_.find(new_node.index);
        if (it == position_map_.end()) {
            add(new_node);
            return;
        }
        const int position = it->second;
        // No need to decrease key if the new weight is not smaller.
        if (new_node.weight >= heap_[static_cast<std::size_t>(position)].weight) return;

        heap_[static_cast<std::size_t>(position)] = new_node;
        bubble_up(position);
    }

    // Replace a node that has the same index value as the new node.
    void replace(const Node& new_node) {
        for (int i = 0; i < n_; i++) {
            if (heap_[static_cast<std::size_t>(i)].index == new_node.index) {
                heap_[static_cast<std::size_t>(i)] = new_node;
                bubble_up(i);
                break;
            }
        }
    }

   private:
    // Putting the new item in the first vacant cell in the array. Then move it up in the heap
    // based on its value compared to its parent.
    void bubble_up(int i) {
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (heap_[static_cast<std::size_t>(i)].weight >=
                heap_[static_cast<std::size_t>(parent)].weight)
                break;

            // Swap
            std::swap(heap_[static_cast<std::size_t>(i)], heap_[static_cast<std::size_t>(parent)]);

            position_map_[heap_[static_cast<std::size_t>(i)].index] = i;
            position_map_[heap_[static_cast<std::size_t>(parent)].index] = parent;
            i = parent;
        }
    }

    // Used in heap deletion. Compares the parent nodes with child nodes in subtree.
    void bubble_down(int i) {
        while (true) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int smallest = i;

            if (left < n_ && heap_[static_cast<std::size_t>(left)].weight <
                                 heap_[static_cast<std::size_t>(smallest)].weight)
                smallest = left;
            if (right < n_ && heap_[static_cast<std::size_t>(right)].weight <
                                  heap_[static_cast<std::size_t>(smallest)].weight)
                smallest = right;
            if (smallest == i) break;

            std::swap(heap_[static_cast<std::size_t>(i)],
                      heap_[static_cast<std::size_t>(smallest)]);
            position_map_[heap_[static_cast<std::size_t>(i)].index] = i;
            position_map_[heap_[static_cast<std::size_t>(smallest)].index] = smallest;

            i = smallest;
        }
    }

    std::vector<Node> heap_;
    std::unordered_map<int, int> position_map_;

    int n_ = 0;  // Number of nodes.
};

}  // namespace corehydro::numerics::math::optimization
