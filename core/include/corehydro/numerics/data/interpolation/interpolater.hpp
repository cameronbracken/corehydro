// ported from: Numerics/Data/Interpolation/Support/Interpolater.cs @ 2a0357a
//
// Abstract base for 1D interpolation. Owns the shared search machinery -- sequential,
// bisection, and hunt search over the sorted x-value list -- that Linear (and, through
// it, Bilinear) uses to locate the interpolation interval, plus the SearchStart/
// correlated-call memory that lets repeated nearby lookups reuse the hunt search's
// exponential expansion.
//
// CubicSpline and Polynomial interpolation (cubic_spline.hpp, polynomial.hpp) both derive
// from this base too. The standalone Search.cs utility class is a separate, DIFFERENT
// algorithm from the search machinery here (free functions with boundary-sentinel returns,
// not member functions with correlated-call state); only the subset its callers need
// (SNIS's Sequential overload, Histogram's Bisection overload) is ported, in search.hpp.
//
// Quirk transcribed verbatim (not "fixed"): `deltaStart = Math.Min(1, (int)Math.Pow(Count,
// 0.25))` always evaluates to 1 for any Count >= 2 (pow(2, 0.25) already truncates to >= 1,
// and Math.Min caps at 1) -- see docs/upstream-csharp-issues.md.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/sort_order.hpp"

namespace corehydro::numerics::data {

class Interpolater {
   public:
    Interpolater(std::vector<double> x_values, std::vector<double> y_values,
                 SortOrder sort_order = SortOrder::Ascending)
        : x_values_(std::move(x_values)), y_values_(std::move(y_values)), sort_order_(sort_order) {
        count_ = static_cast<int>(x_values_.size());
        if (static_cast<int>(y_values_.size()) != count_)
            throw std::invalid_argument("The x and y lists must be the same length.");
        if (count_ < 2)
            throw std::invalid_argument("The x list is too small. It must have at least 2 values.");
        for (int i = 1; i < count_; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            if (x_values_[ui] == x_values_[ui - 1])
                throw std::invalid_argument("All x values should be unique.");
            if (sort_order == SortOrder::Ascending && x_values_[ui] < x_values_[ui - 1])
                throw std::invalid_argument("The x values are not in ascending order.");
            if (sort_order == SortOrder::Descending && x_values_[ui] > x_values_[ui - 1])
                throw std::invalid_argument("The x values are not in descending order.");
        }
        delta_start_ = std::min(1, static_cast<int>(std::pow(static_cast<double>(count_), 0.25)));
    }

    virtual ~Interpolater() = default;

    int count() const { return count_; }

    int search_start() const { return search_start_; }
    void set_search_start(int value) { search_start_ = value; }

    bool use_smart_search() const { return use_smart_search_; }
    void set_use_smart_search(bool value) { use_smart_search_ = value; }

    const std::vector<double>& x_values() const { return x_values_; }
    const std::vector<double>& y_values() const { return y_values_; }

    SortOrder sort_order() const { return sort_order_; }

    // Given a value x, returns an interpolated value at the interval starting at `index`.
    virtual double base_interpolate(double x, int index) const = 0;

    std::vector<double> interpolate(const std::vector<double>& x) const {
        std::vector<double> values(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) values[i] = interpolate(x[i]);
        return values;
    }

    double interpolate(double x) const { return base_interpolate(x, search(x)); }

    // Search for the lower bound of the interpolation interval; updates the correlated-call
    // flag and remembers the search location for the next call (mutable: logically a cache,
    // not user-visible state, so search()/*_search() stay const like the rest of the API).
    int search(double x) const {
        int start;
        if (use_smart_search_) {
            start = correlated_ ? hunt_search(x) : bisection_search(x);
        } else {
            start = sequential_search(x);
        }
        correlated_ = std::abs(start - search_start_) > delta_start_ ? false : true;
        search_start_ = (start < 0 || start >= count_) ? 0 : start;
        return start;
    }

    int sequential_search(double x) const {
        int jl = search_start_;
        bool ascending = sort_order_ == SortOrder::Ascending;
        if ((ascending && x < x_values_[0]) || (!ascending && x > x_values_[0])) {
            return 0;
        } else if ((ascending && x > x_values_[static_cast<std::size_t>(count_ - 1)]) ||
                   (!ascending && x < x_values_[static_cast<std::size_t>(count_ - 1)])) {
            return count_ - 2;
        } else if ((ascending && x < x_values_[static_cast<std::size_t>(search_start_)]) ||
                   (!ascending && x > x_values_[static_cast<std::size_t>(search_start_)])) {
            jl = 0;
        }
        for (int i = jl; i < count_; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            if ((ascending && x <= x_values_[ui]) || (!ascending && x >= x_values_[ui])) {
                jl = i - 1;
                break;
            }
        }
        return jl;
    }

    int bisection_search(double x) const {
        int ju = count_ - 1, jl = 0;
        bool ascnd = sort_order_ == SortOrder::Ascending;
        while (ju - jl > 1) {
            int jm = (ju + jl) >> 1;
            if ((x >= x_values_[static_cast<std::size_t>(jm)]) == ascnd)
                jl = jm;
            else
                ju = jm;
        }
        return jl;
    }

    int hunt_search(double x) const {
        int jl = search_start_, ju, inc = 1;
        bool ascnd = sort_order_ == SortOrder::Ascending;
        if (jl < 0 || jl > count_ - 1) {
            jl = 0;
            ju = count_ - 1;
        } else {
            if ((x >= x_values_[static_cast<std::size_t>(jl)]) == ascnd) {
                for (;;) {
                    ju = jl + inc;
                    if (ju >= count_ - 1) {
                        ju = count_ - 1;
                        break;
                    } else if ((x < x_values_[static_cast<std::size_t>(ju)]) == ascnd) {
                        break;
                    } else {
                        jl = ju;
                        inc += inc;
                    }
                }
            } else {
                ju = jl;
                for (;;) {
                    jl = jl - inc;
                    if (jl <= 0) {
                        jl = 0;
                        break;
                    } else if ((x >= x_values_[static_cast<std::size_t>(jl)]) == ascnd) {
                        break;
                    } else {
                        ju = jl;
                        inc += inc;
                    }
                }
            }
        }
        while (ju - jl > 1) {
            int jm = (ju + jl) >> 1;
            if ((x >= x_values_[static_cast<std::size_t>(jm)]) == ascnd)
                jl = jm;
            else
                ju = jm;
        }
        return jl;
    }

   protected:
    std::vector<double> x_values_;
    std::vector<double> y_values_;

   private:
    int count_ = 0;
    mutable int search_start_ = 0;
    int delta_start_ = 0;
    mutable bool correlated_ = false;
    bool use_smart_search_ = true;
    SortOrder sort_order_ = SortOrder::Ascending;
};

}  // namespace corehydro::numerics::data
