// ported from: Numerics/Mathematics/Integration/Integration.cs @ 2a0357a
//
// Static numerical-integration methods. `GaussLegendre20` was ported first (M8) -- it is what
// the uncertain-data (measurement-error) branch of the UnivariateDistribution stationary
// likelihood integrates with. This file now carries the full static surface: `GaussLegendre`
// (10-point), `GaussLegendre20`, `TrapezoidalRule`, `SimpsonsRule`, and `Midpoint`. (The adaptive
// integrators live in their own files upstream and here -- see adaptive_gauss_kronrod.hpp,
// adaptive_simpsons_rule.hpp, adaptive_simpsons_rule_2d.hpp, adaptive_gauss_lobatto.hpp.)
//
// GaussLegendre/GaussLegendre20 nodes are roots of the corresponding Legendre polynomial;
// weights are the corresponding Christoffel numbers (Abramowitz and Stegun 1964, Table 25.4).
// Values are verbatim from the C# source.
#pragma once
#include <functional>

namespace corehydro::numerics::math::integration {

class Integration {
   public:
    // Returns the integral of `f` between `a` and `b` by ten-point Gauss-Legendre integration.
    static double gauss_legendre(const std::function<double(double)>& f, double a, double b) {
        static constexpr double x[5] = {0.1488743389816312, 0.4333953941292472,
                                        0.6794095682990244, 0.8650633666889845,
                                        0.9739065285171717};
        static constexpr double w[5] = {0.2955242247147529, 0.2692667193099963,
                                        0.2190863625159821, 0.1494513491505806,
                                        0.0666713443086881};
        double xm = 0.5 * (b + a);
        double xr = 0.5 * (b - a);
        double s = 0;
        for (int j = 0; j < 5; j++) {
            double dx = xr * x[j];
            s += w[j] * (f(xm + dx) + f(xm - dx));
        }
        s *= xr;  // C# `return s *= xr;`
        return s;
    }

    // Returns the integral of `f` between `a` and `b` by twenty-point Gauss-Legendre
    // integration (C# `GaussLegendre20`, line 62). Exact for polynomials of degree 39 or
    // less; 10 symmetric node pairs (20 function evaluations total).
    static double gauss_legendre20(const std::function<double(double)>& f, double a, double b) {
        static constexpr double x[10] = {
            0.0765265211334973338, 0.2277858511416450781, 0.3737060887154195607,
            0.5108670019508270981, 0.6360536807265150254, 0.7463319064601507926,
            0.8391169718222188234, 0.9122344282513259059, 0.9639719272779137912,
            0.9931285991850949247};
        static constexpr double w[10] = {
            0.1527533871307258507, 0.1491729864726037467, 0.1420961093183820514,
            0.1316886384491766269, 0.1181945319615184174, 0.1019301198172404351,
            0.0832767415767047487, 0.0626720483341090636, 0.0406014298003869413,
            0.0176140071391521183};
        double xm = 0.5 * (b + a);
        double xr = 0.5 * (b - a);
        double s = 0;
        for (int j = 0; j < 10; j++) {
            double dx = xr * x[j];
            s += w[j] * (f(xm + dx) + f(xm - dx));
        }
        s *= xr;  // C# `return s *= xr;`
        return s;
    }

    // Numerical integration using the Trapezoidal Rule. `steps` is the number of integration
    // steps (default 2).
    static double trapezoidal_rule(const std::function<double(double)>& f, double a, double b,
                                    int steps = 2) {
        double h = (b - a) / steps;
        double x = a;
        double sum = 0.5 * (f(a) + f(b));
        for (int i = 1; i <= steps - 1; i++) {
            x += h;
            sum += f(x);
        }
        return h * sum;
    }

    // Numerical integration using Simpson's Rule. `steps` is the number of integration steps
    // (default 2).
    static double simpsons_rule(const std::function<double(double)>& f, double a, double b,
                                 int steps = 2) {
        double h = (b - a) / steps;
        double sum1 = f(a + h / 2.0);
        double sum2 = 0.0;
        for (int i = 1; i <= steps - 1; i++) {
            sum1 += f(a + h * i + h / 2.0);
            sum2 += f(a + h * i);
        }
        return h / 6.0 * (f(a) + f(b) + 4.0 * sum1 + 2.0 * sum2);
    }

    // Numerical integration using the Midpoint method. `steps` is the number of integration
    // steps (default 2).
    static double midpoint(const std::function<double(double)>& f, double a, double b,
                            int steps = 2) {
        double h = (b - a) / steps;
        double x = a + h / 2.0;
        double sum = 0;
        for (int i = 1; i <= steps; i++) {
            sum += f(x);
            x += h;
        }
        return h * sum;
    }
};

}  // namespace corehydro::numerics::math::integration
