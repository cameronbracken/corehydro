// ported from: Numerics/Mathematics/Special Functions/Gamma.cs @ 7e8e8d1
//             + Numerics/Mathematics/Special Functions/Evaluate.cs (PolynomialRev)
//
// Gamma function (Cephes), Lanczos approximation, and digamma. Algorithms and
// coefficient tables are copied verbatim from the C# source so results match the
// oracle values bit-closely.
#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>

#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::special {

namespace detail {

// Wichura AS241 inverse standard-normal CDF. Ported from Normal.r8_normal_01_cdf_inverse.
// Needed by inverse_lower_incomplete / inverse_upper_incomplete (they call Normal.StandardZ).
inline double normal_standard_z(double p) {
    // Boundary guards match r8_normal_01_cdf_inverse: p<=0 -> -inf, p>=1 -> +inf.
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return  std::numeric_limits<double>::infinity();
    static constexpr double a[8] = {3.3871328727963666080,    1.3314166789178437745e+2,
                                    1.9715909503065514427e+3,  1.3731693765509461125e+4,
                                    4.5921953931549871457e+4,  6.7265770927008700853e+4,
                                    3.3430575583588128105e+4,  2.5090809287301226727e+3};
    static constexpr double b[8] = {1.0,                       4.2313330701600911252e+1,
                                    6.8718700749205790830e+2,  5.3941960214247511077e+3,
                                    2.1213794301586595867e+4,  3.9307895800092710610e+4,
                                    2.8729085735721942674e+4,  5.2264952788528545610e+3};
    static constexpr double c[8] = {1.42343711074968357734,    4.63033784615654529590,
                                    5.76949722146069140550,    3.64784832476320460504,
                                    1.27045825245236838258,    2.41780725177450611770e-1,
                                    2.27238449892691845833e-2, 7.74545014278341407640e-4};
    static constexpr double d[8] = {1.0,                       2.05319162663775882187,
                                    1.67638483018380384940,    6.89767334985100004550e-1,
                                    1.48103976427480074590e-1, 1.51986665636164571966e-2,
                                    5.47593808499534494600e-4, 1.05075007164441684324e-9};
    static constexpr double e[8] = {6.65790464350110377720,    5.46378491116411436990,
                                    1.78482653991729133580,    2.96560571828504891230e-1,
                                    2.65321895265761230930e-2, 1.24266094738807843860e-3,
                                    2.71155556874348757815e-5, 2.01033439929228813265e-7};
    static constexpr double f[8] = {1.0,                       5.99832206555887937690e-1,
                                    1.36929880922735805310e-1, 1.48753612908506148525e-2,
                                    7.86869131145613259100e-4, 1.84631831751005468180e-5,
                                    1.42151175831644588870e-7, 2.04426310338993978564e-15};
    constexpr double split1 = 0.425, split2 = 5.0;
    constexpr double const1 = 0.180625, const2 = 1.6;
    double q, r, val;
    q = p - 0.5;
    if (std::fabs(q) <= split1) {
        r = const1 - q * q;
        double num = 0.0, den = 0.0;
        for (int i = 7; i >= 0; --i) { num = num * r + a[i]; den = den * r + b[i]; }
        val = q * num / den;
    } else {
        r = (q < 0.0) ? p : 1.0 - p;
        r = std::sqrt(-std::log(r));
        if (r <= split2) {
            r -= const2;
            double num = 0.0, den = 0.0;
            for (int i = 7; i >= 0; --i) { num = num * r + c[i]; den = den * r + d[i]; }
            val = num / den;
        } else {
            r -= split2;
            double num = 0.0, den = 0.0;
            for (int i = 7; i >= 0; --i) { num = num * r + e[i]; den = den * r + f[i]; }
            val = num / den;
        }
        if (q < 0.0) val = -val;
    }
    return val;
}

// Horner evaluation: coeffs[0]*x^n + ... + coeffs[n]  (mirrors Evaluate.PolynomialRev).
inline double polynomial_rev(const double* coeffs, double x, int n) {
    double value = coeffs[0];
    for (int i = 1; i <= n; ++i) {
        value *= x;
        value += coeffs[i];
    }
    return value;
}

inline constexpr double kLanczosP[9] = {
    0.99999999999980993,    676.5203681218851,     -1259.1392167224028,
    771.32342877765313,     -176.61502916214059,   12.507343278686905,
    -0.13857109526572012,   0.0000099843695780195716, 0.00000015056327351493116};

inline constexpr double kGammaP[7] = {
    1.60119522476751861407E-4, 1.19135147006586384913E-3, 1.04213797561761569935E-2,
    4.76367800457137231464E-2, 2.07448227648435975150E-1, 4.94214826801497100753E-1,
    9.99999999999999996796E-1};

inline constexpr double kGammaQ[8] = {
    -2.31581873324120129819E-5, 5.39605580493303397842E-4, -4.45641913851797240494E-3,
    1.18139785222060435552E-2,  3.58236398605498653373E-2, -2.34591795718243348568E-1,
    7.14304917030273074085E-2,  1.00000000000000000320E0};

inline constexpr double kStir[5] = {7.87311395793093628397E-4, -2.29549961613378126380E-4,
                                    -2.68132617805781232825E-3, 3.47222221605458667310E-3,
                                    8.33333333333482257126E-2};

}  // namespace detail

// Forward declaration: Lanczos uses Function for the reflection branch.
double function(double x);

inline double stirling(double x) {
    constexpr double MAXSTIR = 143.01608;
    double w = 1.0 / x;
    double y = std::exp(x);
    w = 1.0 + w * detail::polynomial_rev(detail::kStir, w, 4);
    if (x > MAXSTIR) {
        double v = std::pow(x, 0.5 * x - 0.25);
        if (std::isinf(v) && std::isinf(y)) {
            y = std::numeric_limits<double>::infinity();
        } else {
            y = v * (v / y);
        }
    } else {
        y = std::pow(x, x - 0.5) / y;
    }
    y = kSqrt2PI * y * w;
    return y;
}

// The Gamma function (Cephes implementation).
inline double function(double x) {
    double p, z;
    double q = std::fabs(x);

    if (q > 33.0) {
        if (x < 0.0) {
            p = std::floor(q);
            if (p == q) throw std::overflow_error("gamma overflow");
            z = q - p;
            if (z > 0.5) {
                p += 1.0;
                z = q - p;
            }
            z = q * std::sin(kPi * z);
            if (z == 0.0) throw std::overflow_error("gamma overflow");
            z = std::fabs(z);
            z = kPi / (z * stirling(q));
            return -z;
        } else {
            return stirling(x);
        }
    }

    z = 1.0;
    while (x >= 3.0) {
        x -= 1.0;
        z *= x;
    }
    while (x < 0.0) {
        if (x > -1.0E-9) return z / ((1.0 + kEuler * x) * x);
        z /= x;
        x += 1.0;
    }
    while (x < 2.0) {
        if (x == 0.0) throw std::runtime_error("gamma singularity");
        if (x < 1.0E-9) return z / ((1.0 + kEuler * x) * x);
        z /= x;
        x += 1.0;
    }
    if (x == 2.0 || x == 3.0) return z;
    x -= 2.0;
    p = detail::polynomial_rev(detail::kGammaP, x, 6);
    q = detail::polynomial_rev(detail::kGammaQ, x, 7);
    return z * p / q;
}

// Lanczos approximation of the Gamma function.
inline double lanczos(double x) {
    constexpr int g = 7;
    if (x < 0.5) {
        return kPi / (std::sin(kPi * x) * function(1.0 - x));
    }
    x -= 1.0;
    double y = detail::kLanczosP[0];
    for (int i = 1; i < g + 2; ++i) y += detail::kLanczosP[i] / (x + i);
    double t = x + g + 0.5;
    return kSqrt2PI * std::pow(t, x + 0.5) * std::exp(-t) * y;
}

// Digamma (Euler's psi), algorithm AS103.
inline double digamma(double X) {
    constexpr double SMALL = 0.000000001;
    constexpr double CRIT = 13.0;
    constexpr double C1 = 0.083333333333333329, C2 = -0.0083333333333333332,
                     C3 = 0.003968253968253968, C4 = -0.0041666666666666666,
                     C5 = 0.007575757575757576, C6 = -0.021092796092796094,
                     C7 = 0.083333333333333329, D1 = -0.57721566490153287;
    if (X <= 0.0) throw std::out_of_range("digamma: X must be > 0");
    double dg = 0.0;
    if (X > SMALL) {
        double Y = X;
        while (Y < CRIT) {
            dg -= 1.0 / Y;
            Y += 1.0;
        }
        dg += std::log(Y) - 0.5 / Y;
        Y = 1.0 / (Y * Y);
        double sum = ((((((C7 * Y + C6) * Y + C5) * Y + C4) * Y + C3) * Y + C2) * Y + C1) * Y;
        dg -= sum;
    } else {
        dg = D1 - 1.0 / X;
    }
    return dg;
}

// log_gamma: Lanczos-based log(Gamma(z)), ported from Gamma.LogGamma(z).
// Coefficient table (GammaDk) and algorithm mirror the C# source exactly.
namespace detail {
inline constexpr double kGammaDk[11] = {
    2.48574089138753565546e-5,  1.05142378581721974210,    -3.45687097222016235469,
    4.51227709466894823700,    -2.98285225323576655721,     1.05639711577126713077,
   -1.95428773191645869583e-1,  1.70970543404441224307e-2, -5.71926117404305781283e-4,
    4.63399473359905636708e-6, -2.71994908488607703910e-9};
}  // namespace detail

inline double log_gamma(double z) {
    constexpr int kGammaN = 10;
    constexpr double kGammaR = 10.900511;
    constexpr double kLnPi = 1.1447298858494001741434273513530587116472948129153;
    constexpr double kLogTwoSqrtEOverPi = 0.6207822376352452223455184457816472122518527279025978;
    if (z < 0.5) {
        double s = detail::kGammaDk[0];
        for (int i = 1; i <= kGammaN; ++i) s += detail::kGammaDk[i] / (i - z);
        return kLnPi - std::log(std::sin(kPi * z)) - std::log(s) - kLogTwoSqrtEOverPi
               - (0.5 - z) * std::log((0.5 - z + kGammaR) / corehydro::numerics::kE);
    } else {
        double s = detail::kGammaDk[0];
        for (int i = 1; i <= kGammaN; ++i) s += detail::kGammaDk[i] / (z + i - 1.0);
        return std::log(s) + kLogTwoSqrtEOverPi
               + (z - 0.5) * std::log((z - 0.5 + kGammaR) / corehydro::numerics::kE);
    }
}

// Regularized incomplete gamma integral. This is the AS239 surface retained separately from the
// Cephes lower_incomplete implementation below because upstream exposes both algorithms.
inline double incomplete(double x, double alpha) {
    constexpr double eps = 1e-12;
    constexpr int max_iterations = 100000;
    constexpr double overflow = 1e30;
    constexpr double underflow_log = -180.0;
    constexpr double hill_threshold = 10000.0;

    if (alpha <= 0.0) throw std::out_of_range("incomplete: alpha must be positive");
    if (x < 0.0) throw std::out_of_range("incomplete: x must be non-negative");
    if (x == 0.0) return 0.0;

    if (alpha > hill_threshold) {
        const double r = 1.0 / std::sqrt(alpha);
        double z = (x - alpha) * r;
        double term = z * z;
        double sum = 0.5 * term;
        for (int i = 1; i <= 12; ++i) {
            term = -term * z * r;
            sum += term / (static_cast<double>(i) + 2.0);
            if (term <= eps) break;
        }
        const double ww = 2.0 * sum;
        double w = std::sqrt(ww);
        if (x < alpha) w = -w;
        const double h1 = 1.0 / 3.0;
        const double h2 = -w / 36.0;
        const double h3 = (-ww + 13.0) / 1620.0;
        const double h4 = (42.0 * ww + 119.0) * w / 38880.0;
        z = (((h4 * r + h3) * r + h2) * r + h1) * r + w;
        return 0.5 + 0.5 * std::erf(z * 0.70710678118654757);
    }

    if (x > 1.0 && x >= alpha) {
        double a = 1.0 - alpha;
        double b = a + x + 1.0;
        double term = 0.0;
        double pn1 = 1.0;
        double pn2 = x;
        double pn3 = x + 1.0;
        double pn4 = x * b;
        double ratio = pn3 / pn4;
        for (int i = 1; i <= max_iterations; ++i) {
            a += 1.0;
            b += 2.0;
            term += 1.0;
            const double an = a * term;
            const double pn5 = b * pn3 - an * pn1;
            const double pn6 = b * pn4 - an * pn2;
            if (pn6 != 0.0) {
                const double rn = pn5 / pn6;
                const double difference = std::fabs(ratio - rn);
                if (difference <= eps && difference <= eps * rn) break;
                ratio = rn;
            }
            pn1 = pn3;
            pn2 = pn4;
            pn3 = pn5;
            pn4 = pn6;
            if (std::fabs(pn5) >= overflow) {
                pn1 /= overflow;
                pn2 /= overflow;
                pn3 /= overflow;
                pn4 /= overflow;
            }
        }
        const double argument = alpha * std::log(x) - x - log_gamma(alpha) + std::log(ratio);
        return argument >= underflow_log ? 1.0 - std::exp(argument) : 1.0;
    }

    double sum = 1.0;
    double term = 1.0;
    double a = alpha;
    for (int i = 1; i <= max_iterations; ++i) {
        a += 1.0;
        term = term * x / a;
        sum += term;
        if (term <= eps) break;
    }
    const double argument = alpha * std::log(x) - x - log_gamma(alpha) + std::log(sum / alpha);
    return argument >= underflow_log ? std::exp(argument) : 0.0;
}

// trigamma: ψ'(x), second derivative of log-gamma.
// Ported from Gamma.Trigamma (algorithm AS121).
inline double trigamma(double X) {
    constexpr double A = 0.0001;
    constexpr double B = 5.0;
    constexpr double B2 =  0.1666666667;
    constexpr double B4 = -0.03333333333;
    constexpr double B6 =  0.02380952381;
    constexpr double B8 = -0.03333333333;
    if (X <= 0.0) throw std::out_of_range("trigamma: X must be > 0");
    double Z = X;
    double result;
    if (X <= A) {
        result = 1.0 / X / X;
    } else {
        result = 0.0;
        while (Z < B) {
            result += 1.0 / Z / Z;
            Z += 1.0;
        }
        double Y = 1.0 / Z / Z;
        result += 0.5 * Y + (1.0 + Y * (B2 + Y * (B4 + Y * (B6 + Y * B8)))) / Z;
    }
    return result;
}

// upper_incomplete: regularized upper incomplete gamma Q(a, x).
// Ported from Gamma.UpperIncomplete — continued-fraction algorithm.
inline double upper_incomplete(double a, double x);

// lower_incomplete: regularized lower incomplete gamma P(a, x).
// Ported from Gamma.LowerIncomplete — series expansion algorithm.
inline double lower_incomplete(double a, double x) {
    constexpr double kLogMax = 709.782712893384;
    constexpr double kDoubleEpsilon = 0.000000000000000111022302462516;
    if (a <= 0.0) return 1.0;
    if (x <= 0.0) return 0.0;
    if (x > 1.0 && x > a) return 1.0 - upper_incomplete(a, x);
    double ax = a * std::log(x) - x - log_gamma(a);
    if (ax < -kLogMax) return 0.0;
    ax = std::exp(ax);
    double r = a;
    double c = 1.0;
    double ans = 1.0;
    do {
        r += 1.0;
        c *= x / r;
        ans += c;
    } while (c / ans > kDoubleEpsilon);
    return ans * ax / a;
}

inline double upper_incomplete(double a, double x) {
    constexpr double kLogMax = 709.782712893384;
    constexpr double kDoubleEpsilon = 0.000000000000000111022302462516;
    constexpr double big = 4.5035996273705e+15;
    constexpr double biginv = 0.000000000000000222044604925031;
    if (x <= 0.0 || a <= 0.0) return 1.0;
    if (x < 1.0 || x < a) return 1.0 - lower_incomplete(a, x);
    if (std::isinf(x) && x > 0.0) return 0.0;
    double ax = a * std::log(x) - x - log_gamma(a);
    if (ax < -kLogMax) return 0.0;
    ax = std::exp(ax);
    // continued fraction
    double y = 1.0 - a;
    double z = x + y + 1.0;
    double c = 0.0;
    double pkm2 = 1.0, qkm2 = x;
    double pkm1 = x + 1.0, qkm1 = z * x;
    double ans = pkm1 / qkm1;
    double t;
    do {
        c += 1.0; y += 1.0; z += 2.0;
        double yc = y * c;
        double pk = pkm1 * z - pkm2 * yc;
        double qk = qkm1 * z - qkm2 * yc;
        double r;
        if (qk != 0.0) {
            r = pk / qk;
            t = std::fabs((ans - r) / r);
            ans = r;
        } else {
            t = 1.0;
        }
        pkm2 = pkm1; pkm1 = pk;
        qkm2 = qkm1; qkm1 = qk;
        if (std::fabs(pk) > big) {
            pkm2 *= biginv; pkm1 *= biginv;
            qkm2 *= biginv; qkm1 *= biginv;
        }
    } while (t > kDoubleEpsilon);
    return ans * ax;
}

// inverse_upper_incomplete: x = Q^{-1}(a, y), i.e. UpperIncomplete(a, x) ≈ y.
// inverse_lower_incomplete: x = P^{-1}(a, y), i.e. LowerIncomplete(a, x) ≈ y.
// Both port Gamma.Inverse (private) via the public wrappers in Gamma.cs.

namespace detail {
// Core inverse: finds x such that UpperIncomplete(a, x) ≈ y.
// Mirrors Gamma.Inverse exactly (Newton + interval halving).
// Normal.StandardZ (called by C# Gamma.Inverse) clamps p==0 to double.Epsilon and
// p==1 to 1-double.Epsilon before calling r8_normal_01_cdf_inverse; we replicate that
// here so boundary inputs yield the same large-finite results as the C# oracle.
inline double gamma_inverse(double a, double y) {
    constexpr double kLogMax = 709.782712893384;
    constexpr double kDoubleEpsilon = 0.000000000000000111022302462516;
    double x0 = std::numeric_limits<double>::max();
    double yl = 0.0;
    double x1 = 0.0;
    double yh = 1.0;
    double dithresh = 5.0 * kDoubleEpsilon;
    // initial approximation — clamp y like Normal.StandardZ to match C# oracle behavior
    // for boundary inputs (y==0 → denorm_min, y==1 → 1-denorm_min).
    double yc = y;
    if (yc == 0.0) yc = std::numeric_limits<double>::denorm_min();
    if (yc == 1.0) yc = 1.0 - std::numeric_limits<double>::denorm_min();
    double d = 1.0 / (9.0 * a);
    double yy = 1.0 - d - normal_standard_z(yc) * std::sqrt(d);
    double x = a * yy * yy * yy;
    double lgm = log_gamma(a);
    for (int i = 0; i <= 9; ++i) {
        if (x > x0 || x < x1) goto ihalve;
        yy = upper_incomplete(a, x);
        if (yy < yl || yy > yh) goto ihalve;
        if (yy < y) { x0 = x; yl = yy; }
        else        { x1 = x; yh = yy; }
        d = (a - 1.0) * std::log(x) - x - lgm;
        if (d < -kLogMax) goto ihalve;
        d = -std::exp(d);
        d = (yy - y) / d;
        if (std::fabs(d / x) < kDoubleEpsilon) return x;
        x = x - d;
    }
ihalve:
    d = 0.0625;
    if (x0 == std::numeric_limits<double>::max()) {
        if (x <= 0.0) x = 1.0;
        while (x0 == std::numeric_limits<double>::max() && !std::isnan(x) && !std::isinf(x)) {
            x = (1.0 + d) * x;
            yy = upper_incomplete(a, x);
            if (yy < y) { x0 = x; yl = yy; break; }
            d += d;
        }
    }
    d = 0.5;
    double dir = 0.0;
    for (int i = 0; i <= 399; ++i) {
        double t = x1 + d * (x0 - x1);
        if (std::isnan(t)) break;
        x = t;
        yy = upper_incomplete(a, x);
        lgm = (x0 - x1) / (x1 + x0);
        if (std::fabs(lgm) < dithresh) break;
        lgm = (yy - y) / y;
        if (std::fabs(lgm) < dithresh) break;
        if (x <= 0.0) break;
        if (yy >= y) {
            x1 = x; yh = yy;
            if (dir < 0.0) { dir = 0.0; d = 0.5; }
            else if (dir > 1.0) { d = 0.5 * d + 0.5; }
            else { d = (y - yl) / (yh - yl); }
            dir += 1.0;
        } else {
            x0 = x; yl = yy;
            if (dir > 0.0) { dir = 0.0; d = 0.5; }
            else if (dir < -1.0) { d = 0.5 * d; }
            else { d = (y - yl) / (yh - yl); }
            dir -= 1.0;
        }
    }
    if (x == 0.0 || std::isnan(x)) throw std::runtime_error("gamma_inverse: failed to converge");
    return x;
}
}  // namespace detail

// inverse_upper_incomplete: x such that UpperIncomplete(a, x) ≈ y.
// Mirrors Gamma.InverseUpperIncomplete → Gamma.Inverse(a, y).
inline double inverse_upper_incomplete(double a, double y) {
    return detail::gamma_inverse(a, y);
}

// inverse_lower_incomplete: x such that LowerIncomplete(a, x) ≈ y.
// Mirrors Gamma.InverseLowerIncomplete → Gamma.Inverse(a, 1-y).
inline double inverse_lower_incomplete(double a, double y) {
    return detail::gamma_inverse(a, 1.0 - y);
}

}  // namespace corehydro::numerics::math::special
