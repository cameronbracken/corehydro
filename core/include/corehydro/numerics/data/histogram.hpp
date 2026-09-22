// ported from: Numerics/Data/Statistics/Histogram.cs @ 7e8e8d1
//
// Bins a sample into a Histogram, used by ParameterResults to summarize MCMC posterior
// draws. Ports the full public surface except the internal deserialization ctor (took
// pre-built bin data + no raw sample -- an XML/JSON-restore path, not a math one) and
// Bin's Equals/GetHashCode/Clone (object-identity plumbing the C++ value type doesn't
// need; Bin is returned/copied by value here, matching what Clone() achieves in C#).
//
// The two public constructors share identical bin-construction logic in the C# source
// (only the NumberOfBins line differs); factored into a private build_bins() helper here --
// same operations, same order, byte-identical behavior, just not re-typed twice.
//
// v2.1.4 sync (Numerics 33dc1af): FIXED, not mirrored -- AddData(double)'s "expand the
// first/last bin to cover an out-of-range point" branches used to be DEAD CODE: the method
// called GetBinIndexOf(data) UNCONDITIONALLY before checking either branch, and
// GetBinIndexOf throws for any value strictly outside [LowerBound, UpperBound], so a point
// that would need the histogram to "auto-adapt" (per the class's own XML doc comment)
// threw instead of adapting. Upstream's fix moves the interior GetBinIndexOf call into an
// `else` branch reached only when `data` is strictly between the current bounds, and
// additionally distinguishes "at the boundary" (`data == LowerBound`/`UpperBound`, a true
// no-op) from "beyond the boundary" (`data < LowerBound`/`data > UpperBound`, which now
// actually widens the endpoint bin and updates LowerBound/UpperBound) via a nested `if`;
// either way the endpoint bin's Frequency is incremented. This port mirrors that same
// nested-if structure below, INCLUDING one upstream asymmetry transcribed verbatim: the
// lower-extend branch marks `bins_sorted_ = false` (a stale flag, since extending an
// endpoint outward never changes the bins' relative order -- Bin::compare_to only
// compares bounds pairwise, and widening the first/last bin never crosses another bin's
// bounds), but the upper-extend branch does NOT -- this asymmetry is harmless (any
// re-sort would be a no-op) but is not "fixed" here, matching the C# exactly. See
// docs/upstream-csharp-issues.md (marked RESOLVED) and
// fixtures/special_functions/histogram.json's `adapt_*` cases (ported from the new v2.1.4
// Test_AddData_AdaptsEndpointBins).
// v2.2.0 changes only the ArgumentException constructor argument order in Bin.CompareTo;
// C++ already exposes the intended message and has no ParamName field to mirror.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/search.hpp"

namespace corehydro::numerics::data {

class Histogram {
   public:
    // A histogram bin: [lower_bound, upper_bound) with an integer frequency count.
    class Bin {
       public:
        Bin(double lower_bound, double upper_bound, int frequency = 0)
            : lower_bound(lower_bound), upper_bound(upper_bound), frequency(frequency) {
            if (lower_bound > upper_bound)
                throw std::out_of_range("The upper bound must be greater than the lower bound.");
            if (frequency < 0) throw std::out_of_range("The count must be non-negative.");
        }

        double lower_bound;
        double upper_bound;
        int frequency;

        double midpoint() const { return (upper_bound + lower_bound) / 2.0; }

        // Mirrors C#'s IComparable<Bin>.CompareTo: 0 if bit-for-bit equal bounds, +1 if
        // this bin is lower than `other`, -1 otherwise; throws if the bins overlap
        // (excluding a shared edge). Used only by sort_bins()'s ordering comparator below.
        int compare_to(const Bin& other) const {
            if (upper_bound > other.lower_bound && lower_bound < other.lower_bound)
                throw std::invalid_argument("The bins cannot be overlapping.");
            if (upper_bound == other.upper_bound && lower_bound == other.lower_bound) return 0;
            if (other.upper_bound <= lower_bound) return 1;
            return -1;
        }
    };

    // Constructs a histogram from `data`, using the Rice Rule to set the bin count.
    explicit Histogram(const std::vector<double>& data) {
        number_of_bins_ =
            static_cast<int>(std::ceil(2.0 * std::pow(static_cast<double>(data.size()), 1.0 / 3.0)) + 1.0);
        init_bounds(data);
        build_bins();
        add_data(data);
    }

    // Constructs a histogram with a specific bin count; the histogram limits are derived
    // from `data`.
    Histogram(const std::vector<double>& data, int number_of_bins) : number_of_bins_(number_of_bins) {
        init_bounds(data);
        build_bins();
        add_data(data);
    }

    double lower_bound() const { return lower_bound_; }
    double upper_bound() const { return upper_bound_; }
    int number_of_bins() const { return number_of_bins_; }
    double bin_width() const { return bin_width_; }

    // The total sample count (sum of every bin's frequency).
    int data_count() const {
        int count = 0;
        for (const auto& b : bins_) count += b.frequency;
        return count;
    }

    // A copy of the bin at `index` (bins are kept sorted lazily; see sort_bins()).
    Bin bin(int index) const {
        sort_bins();
        return bins_[static_cast<std::size_t>(index)];
    }

    double mean() const {
        int total = 0;
        double sum = 0.0;
        for (const auto& b : bins_) {
            sum += b.midpoint() * b.frequency;
            total += b.frequency;
        }
        return total == 0 ? 0.0 : sum / total;
    }

    double median() const {
        int total = 0;
        for (const auto& b : bins_) total += b.frequency;
        if (total == 0) return std::numeric_limits<double>::quiet_NaN();
        int half_total = total / 2;
        std::size_t m = 0;
        int v = 0;
        while (m < bins_.size()) {
            v += bins_[m].frequency;
            if (v >= half_total) break;
            m += 1;
        }
        return bins_[m].midpoint();
    }

    double mode() const {
        std::size_t m = 0;
        int cur_max = 0;
        for (std::size_t i = 0; i < bins_.size(); ++i) {
            if (bins_[i].frequency > cur_max) {
                cur_max = bins_[i].frequency;
                m = i;
            }
        }
        return bins_[m].midpoint();
    }

    double standard_deviation() const {
        double stddev = 0.0;
        int total = 0;
        double m = mean();
        for (const auto& b : bins_) {
            int vals = b.frequency;
            double diff = b.midpoint() - m;
            stddev += diff * diff * vals;
            total += vals;
        }
        return total == 0 ? 0.0 : std::sqrt(stddev / total);
    }

    // Appends a bin to the bin list (marks the cached sort/limits stale).
    void add_bin(const Bin& bin) {
        bins_.push_back(bin);
        bins_sorted_ = false;
    }

    // Adds one data value to the histogram. If `value` falls strictly outside the current
    // [lower_bound(), upper_bound()] range, the first/last bin (and the histogram's own
    // lower_bound()/upper_bound()) widen to cover it (see the file header for the v2.1.4
    // fix this ports); an exact boundary match is a no-op widen but still increments the
    // endpoint bin's frequency; an interior value increments whichever bin contains it.
    void add_data(double value) {
        sort_bins();
        if (value <= lower_bound_) {
            if (value < lower_bound_) {
                bins_.front().lower_bound = value;
                lower_bound_ = value;
                bins_sorted_ = false;
            }
            bins_.front().frequency += 1;
        } else if (value >= upper_bound_) {
            if (value > upper_bound_) {
                bins_.back().upper_bound = value;
                upper_bound_ = value;
            }
            bins_.back().frequency += 1;
        } else {
            int index = get_bin_index_of(value);
            if (index >= 0 && index < number_of_bins_) {
                bins_[static_cast<std::size_t>(index)].frequency += 1;
            }
        }
    }

    // Adds a sequence of data values to the histogram.
    void add_data(const std::vector<double>& data) {
        for (double d : data) add_data(d);
    }

    // Returns the index of the bin containing `value`. Throws if `value` falls outside the
    // histogram's overall [lower_bound(), upper_bound()] limits.
    int get_bin_index_of(double value) const {
        sort_bins();
        if (value < bins_.front().lower_bound || value > bins_.back().upper_bound)
            throw std::invalid_argument("The value is not contained with the histogram limits.");
        int idx = search::bisection(value, bin_limits_);
        int n = static_cast<int>(bin_limits_.size());
        return idx < 0 ? 0 : (idx >= n ? n - 1 : idx);
    }

   private:
    // Shared by both public constructors: derives lower_bound_/upper_bound_/bin_width_ from
    // `data` given number_of_bins_ already set.
    void init_bounds(const std::vector<double>& data) {
        if (data.empty()) throw std::invalid_argument("Sequence contains no elements.");
        lower_bound_ = *std::min_element(data.begin(), data.end());
        upper_bound_ = *std::max_element(data.begin(), data.end());
        if (upper_bound_ == lower_bound_) upper_bound_ = lower_bound_ + 1.0;
        bin_width_ = (upper_bound_ - lower_bound_) / number_of_bins_;
    }

    // Shared by both public constructors: lays out number_of_bins_ equal-width bins
    // spanning [lower_bound_, upper_bound_], guaranteeing the last bin's upper bound is
    // exactly upper_bound_ (not a rounded multiple of bin_width_).
    void build_bins() {
        double xl = lower_bound_;
        double xu = xl + bin_width_;
        add_bin(Bin(xl, xu));
        for (int i = 1; i < number_of_bins_; ++i) {
            xl = xu;
            xu = xl + bin_width_;
            add_bin(Bin(xl, xu));
        }
        bins_.back().upper_bound = upper_bound_;
    }

    // Sorts the bins (by Bin::compare_to) and rebuilds the lower-bound search index if
    // stale. `bins_`/`bin_limits_`/`bins_sorted_` are mutable so const readers (bin(),
    // get_bin_index_of()) can lazily sort, matching the C# SortBins() call pattern.
    void sort_bins() const {
        if (!bins_sorted_) {
            std::sort(bins_.begin(), bins_.end(),
                      [](const Bin& a, const Bin& b) { return a.compare_to(b) < 0; });
            bin_limits_.clear();
            for (const auto& b : bins_) bin_limits_.push_back(b.lower_bound);
            bins_sorted_ = true;
        }
    }

    double lower_bound_ = 0.0;
    double upper_bound_ = 0.0;
    int number_of_bins_ = 0;
    double bin_width_ = 0.0;
    mutable std::vector<Bin> bins_;
    mutable std::vector<double> bin_limits_;
    mutable bool bins_sorted_ = false;
};

}  // namespace corehydro::numerics::data
