// ported from: Numerics/Data/Interpolation/CubicSpline.cs @ 2a0357a
//
// Natural cubic spline interpolation. SetSecondDerivatives always uses the C# yp1 = ypn =
// 1e99 sentinel, which signals the natural-boundary condition (zero second derivative at
// both ends) -- C# never re-parameterizes yp1/ypn, so this port doesn't either; the two
// locals stay hardcoded rather than becoming constructor arguments. The constructor calls
// set_second_derivatives() immediately, exactly like C#, and the method stays public and
// re-callable: per the C# doc comment, a caller who mutates the x- or y-values afterward
// must call it again to refresh y2_.
#pragma once
#include <vector>

#include "corehydro/numerics/data/interpolation/interpolater.hpp"
#include "corehydro/numerics/data/interpolation/sort_order.hpp"

namespace corehydro::numerics::data {

class CubicSpline : public Interpolater {
   public:
    CubicSpline(std::vector<double> x_values, std::vector<double> y_values,
                SortOrder sort_order = SortOrder::Ascending)
        : Interpolater(std::move(x_values), std::move(y_values), sort_order) {
        set_second_derivatives();
    }

    // Auxiliary routine to set the second derivatives. If you make changes to the x- or
    // y-values, then you need to call this routine afterwards.
    void set_second_derivatives() {
        // Stores y2_[0...n-1] with second derivatives of the interpolating function at the
        // tabulated points. yp1/ypn are hardcoded above 0.99e99, so both boundary conditions
        // below always take the "natural spline" branch (zero second derivative on that
        // boundary), matching C# exactly.
        double yp1 = 1e99, ypn = 1e99;
        int n = count();
        const std::vector<double>& x = x_values();
        const std::vector<double>& y = y_values();
        y2_.assign(static_cast<std::size_t>(n), 0.0);
        std::vector<double> u(static_cast<std::size_t>(n - 1), 0.0);
        double p, qn, sig, un;
        if (yp1 > 0.99e99) {
            y2_[0] = 0.0;
            u[0] = 0.0;
        } else {
            y2_[0] = -0.5;
            u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
        }
        for (int i = 1; i < n - 1; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            sig = (x[ui] - x[ui - 1]) / (x[ui + 1] - x[ui - 1]);
            p = sig * y2_[ui - 1] + 2.0;
            y2_[ui] = (sig - 1.0) / p;
            u[ui] = (y[ui + 1] - y[ui]) / (x[ui + 1] - x[ui]) -
                    (y[ui] - y[ui - 1]) / (x[ui] - x[ui - 1]);
            u[ui] = (6.0 * u[ui] / (x[ui + 1] - x[ui - 1]) - sig * u[ui - 1]) / p;
        }
        if (ypn > 0.99e99) {
            qn = 0.0;
            un = 0.0;
        } else {
            std::size_t last = static_cast<std::size_t>(n - 1), prev = static_cast<std::size_t>(n - 2);
            qn = 0.5;
            un = (3.0 / (x[last] - x[prev])) * (ypn - (y[last] - y[prev]) / (x[last] - x[prev]));
        }
        std::size_t last = static_cast<std::size_t>(n - 1);
        y2_[last] = (un - qn * u[last - 1]) / (qn * y2_[last - 1] + 1.0);
        for (int k = n - 2; k >= 0; --k) {
            std::size_t uk = static_cast<std::size_t>(k);
            y2_[uk] = y2_[uk] * y2_[uk + 1] + u[uk];
        }
    }

    double base_interpolate(double x, int index) const override {
        if (index < 0 || index >= count()) index = 0;
        int klo = index, khi = index + 1;
        std::size_t uklo = static_cast<std::size_t>(klo), ukhi = static_cast<std::size_t>(khi);
        const std::vector<double>& xv = x_values();
        const std::vector<double>& yv = y_values();
        double h = xv[ukhi] - xv[uklo];
        double a = (xv[ukhi] - x) / h;
        double b = (x - xv[uklo]) / h;
        double y = a * yv[uklo] + b * yv[ukhi] +
                   ((a * a * a - a) * y2_[uklo] + (b * b * b - b) * y2_[ukhi]) * (h * h) / 6.0;
        return y;
    }

   private:
    std::vector<double> y2_;
};

}  // namespace corehydro::numerics::data
