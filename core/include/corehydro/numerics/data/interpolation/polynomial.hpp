// ported from: Numerics/Data/Interpolation/Polynomial.cs @ 2a0357a
//
// Polynomial interpolation of a given order via Neville's algorithm.
//
// Porting hazard #1: the C# `BaseInterpolate` writes the `Error` property (the running dy of
// Neville's tableau -- the error estimate of the MOST RECENT interpolate() call) as a side
// effect, even though the ported base signature is
// `double base_interpolate(double x, int index) const` -- const because Interpolater's own
// contract, and every other override (Linear, CubicSpline), never mutates state through it.
// error_ is therefore `mutable`: logically a cache of the last call's error estimate (the same
// pattern as Interpolater's own search_start_/correlated_ caches), not user-visible
// construction state, so the const signature stays intact.
//
// Porting hazard #2: the C# `XValues.ToArray().Subset(jl)` (an array-slice extension with no
// C++ standard-library equivalent) becomes a plain index-offset loop below: every `xa(i)`/
// `ya(i)` reads `x_values()[jl + i]`/`y_values()[jl + i]` directly instead of slicing a copy
// first. `jl + mm <= count()` always holds (mirroring the C# invariant): the constructor
// already rejects `order >= count()`, so the `jl = 0` fallback branch has `mm <= count()`, and
// the `jl = index` branch is only taken when `index + mm <= count()`.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/interpolater.hpp"
#include "corehydro/numerics/data/interpolation/sort_order.hpp"

namespace corehydro::numerics::data {

class Polynomial : public Interpolater {
   public:
    // order: the polynomial order. There are order + 1 terms for each polynomial function.
    Polynomial(int order, std::vector<double> x_values, std::vector<double> y_values,
               SortOrder sort_order = SortOrder::Ascending)
        : Interpolater(std::move(x_values), std::move(y_values), sort_order), order_(order) {
        if (order >= count())
            throw std::invalid_argument(
                "order: The order must be less than the length of the x value list.");
    }

    // The error estimate for the most recent call to the interpolation function.
    double error() const { return error_; }

    // The polynomial order. There are order + 1 terms for each polynomial function.
    int order() const { return order_; }
    void set_order(int value) { order_ = value; }

    // Given a value x, this routine returns an interpolated value y, and stores an error
    // estimate in error_. The return value is obtained by (order + 1)-point polynomial
    // interpolation on the subrange x[jl..jl + order].
    double base_interpolate(double x, int index) const override {
        if (index < 0 || index >= count()) index = 0;
        int mm = order_ + 1, ns = 0;
        double y, den, dif, dift, ho, hp, w;
        int jl = index + mm > count() ? 0 : index;
        const std::vector<double>& xv = x_values();
        const std::vector<double>& yv = y_values();
        auto xa = [&](int i) { return xv[static_cast<std::size_t>(jl + i)]; };
        auto ya = [&](int i) { return yv[static_cast<std::size_t>(jl + i)]; };
        std::vector<double> c(static_cast<std::size_t>(mm));
        std::vector<double> d(static_cast<std::size_t>(mm));
        dif = std::abs(x - xa(0));
        // Here we find the index ns of the closest table entry
        for (int i = 0; i < mm; ++i) {
            if ((dift = std::abs(x - xa(i))) < dif) {
                ns = i;
                dif = dift;
            }
            // and initialize the tableau of c's and d's.
            c[static_cast<std::size_t>(i)] = ya(i);
            d[static_cast<std::size_t>(i)] = ya(i);
        }
        // This is the initial approximation to y.
        y = ya(ns);
        --ns;
        // For each column in the tableau, we loop over the current c's and d's and update them.
        for (int m = 1; m < mm; ++m) {
            for (int i = 0; i < mm - m; ++i) {
                std::size_t ui = static_cast<std::size_t>(i);
                ho = xa(i) - x;
                hp = xa(i + m) - x;
                w = c[ui + 1] - d[ui];
                den = ho - hp;
                den = w / den;
                // Here the c's and d's are updated.
                d[ui] = hp * den;
                c[ui] = ho * den;
            }
            error_ = (2 * (ns + 1) < (mm - m)) ? c[static_cast<std::size_t>(ns + 1)]
                                                : d[static_cast<std::size_t>(ns--)];
            y += error_;
            // After each column in the tableau is completed, we decide which correction, c or
            // d, we want to add to our accumulating value of y, i.e., which path to take
            // through the tableau -- forking up or down. We do this in such a way as to take
            // the most "straight line" route through the tableau to its apex, updating ns
            // accordingly to keep track of where we are. This route keeps the partial
            // approximations centered (insofar as possible) on the target x. The last dy added
            // is thus the error indication.
        }
        return y;
    }

   private:
    int order_;
    mutable double error_ = 0.0;
};

}  // namespace corehydro::numerics::data
