// ported from: Numerics/Machine Learning/Support/DecisionNode.cs @ 2a0357a
//
// One node of a decision tree: either an internal split (`feature_index` / `threshold` / two
// children) or a leaf (`value` / `is_leaf_node`). Upstream is a plain mutable property bag with
// the same defaults, so this is a struct rather than a class.
//
// One transcription note: C# `Left`/`Right` are nullable references into a garbage-collected
// heap. `std::shared_ptr` is the faithful C++ spelling -- a tree is built bottom-up by returning
// child nodes from a recursive function, so the parent does not own its children at construction
// time, and a raw owning pointer would need a hand-written destructor that recurses (and would
// blow the stack on a degenerate 100-deep tree the same way the builder does). `shared_ptr` also
// makes a copied tree share its nodes, matching C#'s reference semantics: `DecisionTree::root()`
// hands out the same nodes the tree walks.
#pragma once
#include <limits>
#include <memory>

namespace corehydro::numerics::machine_learning {

struct DecisionNode {
    // The feature index.
    int feature_index = -1;

    // The threshold used to split the node.
    double threshold = std::numeric_limits<double>::quiet_NaN();

    // Nodes to the left of the threshold.
    std::shared_ptr<DecisionNode> left = nullptr;

    // Nodes to the right of the threshold.
    std::shared_ptr<DecisionNode> right = nullptr;

    // The leaf node value.
    double value = std::numeric_limits<double>::quiet_NaN();

    // Determines if this is a leaf node.
    bool is_leaf_node = false;
};

}  // namespace corehydro::numerics::machine_learning
