// ported from: Numerics/Mathematics/Optimization/Dynamic/IndexedMinHeap.cs @ 7e8e8d1
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace corehydro::numerics::math::optimization {

// Fixed-capacity indexed min-heap for dense node identifiers. The shortest-path state machine
// keeps at most one live entry per node, so node count is an exact capacity.
class IndexedMinHeap {
   public:
    explicit IndexedMinHeap(int capacity)
        : weights_(static_cast<std::size_t>(capacity)),
          nodes_(static_cast<std::size_t>(capacity)),
          position_(static_cast<std::size_t>(capacity), -1) {}

    int count() const { return count_; }

    void clear() {
        for (int i = 0; i < count_; ++i)
            position_[static_cast<std::size_t>(nodes_[static_cast<std::size_t>(i)])] = -1;
        count_ = 0;
    }

    void add(int node, float weight) {
        if (count_ >= static_cast<int>(nodes_.size()))
            throw std::runtime_error("Heap is full.");
        int slot = count_++;
        sift_up(slot, node, weight);
    }

    void remove_min(int& node, float& weight) {
        if (count_ == 0) throw std::runtime_error("Heap is empty.");
        node = nodes_[0];
        weight = weights_[0];
        position_[static_cast<std::size_t>(node)] = -1;
        --count_;
        if (count_ > 0)
            sift_down(0, nodes_[static_cast<std::size_t>(count_)],
                      weights_[static_cast<std::size_t>(count_)]);
    }

    void decrease_key(int node, float weight) {
        int slot = position_[static_cast<std::size_t>(node)];
        if (slot < 0) {
            add(node, weight);
            return;
        }
        if (weight >= weights_[static_cast<std::size_t>(slot)]) return;
        sift_up(slot, node, weight);
    }

   private:
    void sift_up(int slot, int node, float weight) {
        while (slot > 0) {
            int parent = (slot - 1) >> 1;
            if (weight >= weights_[static_cast<std::size_t>(parent)]) break;
            weights_[static_cast<std::size_t>(slot)] =
                weights_[static_cast<std::size_t>(parent)];
            nodes_[static_cast<std::size_t>(slot)] = nodes_[static_cast<std::size_t>(parent)];
            position_[static_cast<std::size_t>(nodes_[static_cast<std::size_t>(slot)])] = slot;
            slot = parent;
        }
        weights_[static_cast<std::size_t>(slot)] = weight;
        nodes_[static_cast<std::size_t>(slot)] = node;
        position_[static_cast<std::size_t>(node)] = slot;
    }

    void sift_down(int slot, int node, float weight) {
        while (true) {
            int left = 2 * slot + 1;
            if (left >= count_) break;
            int right = left + 1;
            int smallest =
                right < count_ && weights_[static_cast<std::size_t>(right)] <
                                      weights_[static_cast<std::size_t>(left)]
                    ? right
                    : left;
            if (weights_[static_cast<std::size_t>(smallest)] >= weight) break;
            weights_[static_cast<std::size_t>(slot)] =
                weights_[static_cast<std::size_t>(smallest)];
            nodes_[static_cast<std::size_t>(slot)] = nodes_[static_cast<std::size_t>(smallest)];
            position_[static_cast<std::size_t>(nodes_[static_cast<std::size_t>(slot)])] = slot;
            slot = smallest;
        }
        weights_[static_cast<std::size_t>(slot)] = weight;
        nodes_[static_cast<std::size_t>(slot)] = node;
        position_[static_cast<std::size_t>(node)] = slot;
    }

    std::vector<float> weights_;
    std::vector<int> nodes_;
    std::vector<int> position_;
    int count_ = 0;
};

}  // namespace corehydro::numerics::math::optimization
