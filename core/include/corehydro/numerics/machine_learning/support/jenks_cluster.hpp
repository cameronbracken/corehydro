// ported from: Numerics/Machine Learning/Support/JenksCluster.cs @ 2a0357a
//
// One cluster (class) of a Jenks natural-breaks classification: an inclusive index range into the
// SORTED input array plus the summary statistics of the values it covers.
//
// Two transcription notes:
//
// 1. `Variance` reads `Count`, which is derived from `StartIndex`/`EndIndex`, and the C#
//    constructor sets both index properties BEFORE the Welford loop, so `Count` is already
//    correct when the last line uses it. The C++ constructor keeps that order for the same
//    reason -- computing `variance_` from a member that has not been assigned yet would silently
//    read garbage.
// 2. `Variance` is the SAMPLE (N-1) variance and is 0 for a singleton cluster, while
//    `SumOfSquaredDeviations` is the raw Welford `m2`. They are not the same quantity and only
//    the latter feeds `JenksNaturalBreaks::GoodnessOfVarianceFit`. Upstream computes both from
//    one Welford pass rather than the two-pass formula, so a large-magnitude cluster does not
//    lose precision to cancellation; keep the recurrence.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace corehydro::numerics::machine_learning {

class JenksCluster {
   public:
    // Creates a new Jenks cluster over `data[start_index .. end_index]`, inclusive. `data` must
    // be sorted (the caller, JenksNaturalBreaks::estimate, guarantees it).
    JenksCluster(const std::vector<double>& data, int start_index, int end_index)
        : start_index_(start_index), end_index_(end_index) {
        if (start_index < 0 || end_index < start_index ||
            end_index >= static_cast<int>(data.size()))
            throw std::out_of_range("JenksCluster: index range out of bounds");

        min_value_ = data[static_cast<std::size_t>(start_index)];
        max_value_ = data[static_cast<std::size_t>(end_index)];

        // Compute summary statistics using Welford's algorithm for numerical stability.
        double sum = 0;
        double mean = 0;
        double m2 = 0;
        for (int i = start_index; i <= end_index; i++) {
            double x = data[static_cast<std::size_t>(i)];
            sum += x;
            int k = i - start_index + 1;
            double delta = x - mean;
            mean += delta / k;
            double delta2 = x - mean;
            m2 += delta * delta2;
        }

        sum_ = sum;
        average_ = mean;
        sum_of_squared_deviations_ = m2;
        variance_ = count() <= 1 ? 0.0 : m2 / (count() - 1);
    }

    int start_index() const { return start_index_; }
    int end_index() const { return end_index_; }
    int count() const { return end_index_ - start_index_ + 1; }
    double min_value() const { return min_value_; }
    double max_value() const { return max_value_; }
    double sum() const { return sum_; }
    double average() const { return average_; }
    double variance() const { return variance_; }
    double sum_of_squared_deviations() const { return sum_of_squared_deviations_; }

   private:
    int start_index_;
    int end_index_;
    double min_value_ = 0.0;
    double max_value_ = 0.0;
    double sum_ = 0.0;
    double average_ = 0.0;
    double variance_ = 0.0;
    double sum_of_squared_deviations_ = 0.0;
};

}  // namespace corehydro::numerics::machine_learning
