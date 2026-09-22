// ported from: Numerics/Mathematics/Special Functions/Erf.cs @ 7e8e8d1
//
// Error function and its inverse, mirroring the C# Erf class method-for-method.
//
// Erf.Function(x)     → erf::function(x)     : regularised error function
// Erf.Erfc(x)         → erf::erfc(x)         : complementary error function
// Erf.InverseErf(y)   → erf::inverse_erf(y)  : inverse error function
// Erf.InverseErfc(y)  → erf::inverse_erfc(y) : inverse complementary error function
//
// Rationale: inverse_erf / inverse_erfc are not in the C++ standard library and are
// required by TruncatedNormal and other future distributions. The forward functions
// (function / erfc) are ported for bit-fidelity where a distribution's oracle traces
// through Gamma.LowerIncomplete(0.5, x*x).
//
// These functions are INTERNAL CORE MATH — not exposed to the R or Python packages.
#pragma once
#include <cmath>
#include <stdexcept>

#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::special::erf {

namespace detail {

// Wichura AS241 — standard-normal quantile (r8_normal_01_cdf_inverse).
// Same algorithm used by Normal.StandardZ in the C# source. Duplicated here
// to avoid a dependency on the distributions layer.
inline double r8poly_value(int n, const double a[], double x) {
    double v = 0.0;
    for (int i = n - 1; 0 <= i; --i) v = v * x + a[i];
    return v;
}

inline double standard_z(double p) {
    static const double a[8] = {3.3871328727963666080,    1.3314166789178437745e+2,
                                 1.9715909503065514427e+3,  1.3731693765509461125e+4,
                                 4.5921953931549871457e+4,  6.7265770927008700853e+4,
                                 3.3430575583588128105e+4,  2.5090809287301226727e+3};
    static const double b[8] = {1.0,                       4.2313330701600911252e+1,
                                 6.8718700749205790830e+2,  5.3941960214247511077e+3,
                                 2.1213794301586595867e+4,  3.9307895800092710610e+4,
                                 2.8729085735721942674e+4,  5.2264952788528545610e+3};
    static const double c[8] = {1.42343711074968357734,    4.63033784615654529590,
                                 5.76949722146069140550,    3.64784832476320460504,
                                 1.27045825245236838258,    2.41780725177450611770e-1,
                                 2.27238449892691845833e-2, 7.74545014278341407640e-4};
    static const double d[8] = {1.0,                       2.05319162663775882187,
                                 1.67638483018380384940,    6.89767334985100004550e-1,
                                 1.48103976427480074590e-1, 1.51986665636164571966e-2,
                                 5.47593808499534494600e-4, 1.05075007164441684324e-9};
    static const double e[8] = {6.65790464350110377720,    5.46378491116411436990,
                                 1.78482653991729133580,    2.96560571828504891230e-1,
                                 2.65321895265761230930e-2, 1.24266094738807843860e-3,
                                 2.71155556874348757815e-5, 2.01033439929228813265e-7};
    static const double f[8] = {1.0,                       5.99832206555887937690e-1,
                                 1.36929880922735805310e-1, 1.48753612908506148525e-2,
                                 7.86869131145613259100e-4, 1.84631831751005468180e-5,
                                 1.42151175831644588870e-7, 2.04426310338993978564e-15};
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (1.0 <= p) return std::numeric_limits<double>::infinity();
    double q = p - 0.5;
    double r, value;
    if (std::fabs(q) <= 0.425) {
        r = 0.180625 - q * q;
        value = q * r8poly_value(8, a, r) / r8poly_value(8, b, r);
    } else {
        r = q < 0.0 ? p : 1.0 - p;
        r = std::sqrt(-std::log(r));
        if (r <= 5.0) {
            r = r - 1.6;
            value = r8poly_value(8, c, r) / r8poly_value(8, d, r);
        } else {
            r = r - 5.0;
            value = r8poly_value(8, e, r) / r8poly_value(8, f, r);
        }
        if (q < 0.0) value = -value;
    }
    return value;
}

}  // namespace detail

// Computes the error function erf(x) = 2/sqrt(π) ∫₀ˣ e^(−t²) dt.
// Implementation: Gamma.LowerIncomplete(0.5, x*x) with sign, exactly as in the C# source.
inline double function(double x) {
    if (x < 0.0) return -corehydro::numerics::math::special::lower_incomplete(0.5, x * x);
    return corehydro::numerics::math::special::lower_incomplete(0.5, x * x);
}

// Computes the complementary error function without tail cancellation.
inline double erfc(double x) {
    return std::erfc(x);
}

// Computes the inverse error function: given y = erf(x), returns x.
// Implementation: Normal.StandardZ(0.5*y + 0.5) * sqrt(2) / 2, exactly as in the C# source.
inline double inverse_erf(double y) {
    double s = detail::standard_z(0.5 * y + 0.5);
    return s * corehydro::numerics::kSqrt2 / 2.0;
}

// Computes the inverse complementary error function: given y = erfc(x), returns x.
// Implementation: -Normal.StandardZ(0.5*y) / sqrt(2), preserving small upper-tail probabilities.
inline double inverse_erfc(double y) {
    return -detail::standard_z(0.5 * y) / corehydro::numerics::kSqrt2;
}

}  // namespace corehydro::numerics::math::special::erf
