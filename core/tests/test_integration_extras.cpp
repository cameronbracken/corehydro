// ported from: Test_Numerics/Mathematics/Integration/{Integrands,Test_SimpsonsRule,
// Test_TrapezoidalRule,Test_AdaptiveSimpsonsRule,Test_AdaptiveSimpsonsRule2D,
// Test_AdaptiveGaussLobatto,Test_Integration}.cs @ 2a0357a
//
// Covers the five deterministic integrators (SimpsonsRule, TrapezoidalRule,
// AdaptiveSimpsonsRule, AdaptiveGaussLobatto, AdaptiveSimpsonsRule2D) and the Integration static
// methods (GaussLegendre, GaussLegendre20, TrapezoidalRule, SimpsonsRule, Midpoint).
//
// The four 1D integrators share the same seven C# test methods (FX3/Cosine/Sine/FXX/FXXX/Gamma/
// CVaR), transcribed once as `run_1d_suite<Integrator>()` and instantiated four times, mirroring
// the four upstream test classes' identical bodies. C# tolerances (typically 1E-3 absolute)
// govern throughout.
//
// The `SumOfNormals`/`mu20`/`sigma20` N-dimensional family (Integrands.cs) is deferred to Task 5
// (Monte Carlo integration); `SumOfNormals2D` is ported here with just the two `mu20`/`sigma20`
// entries it actually reads (index 0 and 1), since Test_AdaptiveSimpsonsRule2D.Test_SumOfTwoNormals
// needs it and does not touch the rest of the 20-dimensional arrays.
#include <cmath>
#include <functional>
#include <string>

#include "check.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/integration/adaptive_gauss_lobatto.hpp"
#include "corehydro/numerics/math/integration/adaptive_simpsons_rule.hpp"
#include "corehydro/numerics/math/integration/adaptive_simpsons_rule_2d.hpp"
#include "corehydro/numerics/math/integration/integration.hpp"
#include "corehydro/numerics/math/integration/simpsons_rule.hpp"
#include "corehydro/numerics/math/integration/trapezoidal_rule.hpp"

namespace integ = corehydro::numerics::math::integration;
namespace dist = corehydro::numerics::distributions;

// --- Integrands.cs (the subset this task needs) -----------------------------------------------

// The integral of x^3, should equal 0.25.
static double FX3(double x) { return std::pow(x, 3.0); }

// The integral of Cos(x), should equal ~1.6829419.
static double Cosine(double x) { return std::cos(x); }

// The integral of Sin(x), should equal ~0.459697694131.
static double Sine(double x) { return std::sin(x); }

// The integral of a 2nd-order polynomial, should equal 57.
static double FXX(double x) { return 0.5 + 24.0 * x + 3.0 * x * x; }

// The integral of a 3rd-order polynomial, should equal 89.
static double FXXX(double x) { return 0.5 + 24.0 * x + 3.0 * x * x + 8.0 * x * x * x; }

// The integral of Pi. Should equal ~3.14.
static double PI2D(double x, double y) { return (x * x + y * y < 1.0) ? 1.0 : 0.0; }

// Test function for the sum of 2 normal distributions (Integrands.SumOfNormals2D). Uses only
// mu20[0]/sigma20[0] and mu20[1]/sigma20[1] from the upstream 20-dimensional arrays.
static double SumOfNormals2D(double p1, double p2) {
    constexpr double mu0 = 10.0, sigma0 = 2.0;
    constexpr double mu1 = 30.0, sigma1 = 15.0;
    double result = 0.0;
    result += mu0 + sigma0 * dist::Normal::standard_z(p1);
    result += mu1 + sigma1 * dist::Normal::standard_z(p2);
    return result;
}

// --- shared 1D suite (FX3/Cosine/Sine/FXX/FXXX/Gamma/CVaR), run for all four 1D integrators ----

template <typename Integrator1D>
static void run_1d_suite() {
    // Test_FX3
    {
        Integrator1D r(FX3, 0.0, 1.0);
        r.integrate();
        CHECK_NEAR(r.result(), 0.25, 1e-3);
    }
    // Test_Cosine
    {
        Integrator1D r(Cosine, -1.0, 1.0);
        r.integrate();
        CHECK_NEAR(r.result(), 1.6829419, 1e-3);
    }
    // Test_Sine
    {
        Integrator1D r(Sine, 0.0, 1.0);
        r.integrate();
        CHECK_NEAR(r.result(), 0.459697694131, 1e-3);
    }
    // Test_FXX
    {
        Integrator1D r(FXX, 0.0, 2.0);
        r.integrate();
        CHECK_NEAR(r.result(), 57.0, 1e-3);
    }
    // Test_FXXX
    {
        Integrator1D r(FXXX, 0.0, 2.0);
        r.integrate();
        CHECK_NEAR(r.result(), 89.0, 1e-3);
    }
    // Test_Gamma
    {
        dist::GammaDistribution gamma_dist(10.0, 5.0);
        Integrator1D r([&](double x) { return x * gamma_dist.pdf(x); },
                      gamma_dist.inverse_cdf(1e-16), gamma_dist.inverse_cdf(1.0 - 1e-16));
        r.integrate();
        CHECK_NEAR(r.result(), 50.0, 1e-3);
    }
    // Test_CVaR
    {
        dist::LnNormal ln(10.0, 2.0);
        double alpha = 0.99;
        Integrator1D r([&](double x) { return x * ln.pdf(x); }, ln.inverse_cdf(alpha),
                      ln.inverse_cdf(1.0 - 1e-16));
        r.integrate();
        double result = r.result() / (1.0 - alpha);
        double true_result = std::exp(ln.mu() + 0.5 * ln.sigma() * ln.sigma()) / (1.0 - alpha) *
                             (1.0 - dist::Normal::standard_cdf(dist::Normal::standard_z(alpha) -
                                                               ln.sigma()));
        CHECK_NEAR(result, true_result, 1e-3);
    }
}

int main() {
    // --- SimpsonsRule (Test_SimpsonsRule.cs) ---------------------------------------------------
    run_1d_suite<integ::SimpsonsRule>();

    // --- TrapezoidalRule (Test_TrapezoidalRule.cs) ---------------------------------------------
    run_1d_suite<integ::TrapezoidalRule>();

    // --- AdaptiveSimpsonsRule (Test_AdaptiveSimpsonsRule.cs) ------------------------------------
    run_1d_suite<integ::AdaptiveSimpsonsRule>();

    // --- AdaptiveGaussLobatto (Test_AdaptiveGaussLobatto.cs) ------------------------------------
    run_1d_suite<integ::AdaptiveGaussLobatto>();

    // --- AdaptiveSimpsonsRule2D (Test_AdaptiveSimpsonsRule2D.cs) --------------------------------
    // Test_PI
    {
        integ::AdaptiveSimpsonsRule2D asr2d(PI2D, -1.0, 1.0, -1.0, 1.0);
        asr2d.integrate();
        double true_result = 3.14;
        CHECK_NEAR(asr2d.result(), true_result, 1e-3 * true_result);
    }
    // Test_SumOfTwoNormals
    {
        integ::AdaptiveSimpsonsRule2D asr2d(SumOfNormals2D, 1e-15, 1.0 - 1e-15, 1e-15,
                                            1.0 - 1e-15);
        asr2d.integrate();
        double true_result = 40.0;
        CHECK_NEAR(asr2d.result(), true_result, 1e-3 * true_result);
    }
    // Test_XPlusY
    {
        std::function<double(double, double)> func = [](double x, double y) { return x + y; };
        integ::AdaptiveSimpsonsRule2D asr2d(func, -1.0, 1.0, -1.0, 1.0);
        asr2d.integrate();
        CHECK_NEAR(asr2d.result(), 0.0, 1e-5);
    }
    // Test_XSquaredPlusYSquared
    {
        std::function<double(double, double)> func = [](double x, double y) {
            return x * x + y * y;
        };
        integ::AdaptiveSimpsonsRule2D asr2d(func, -1.0, 1.0, -1.0, 1.0);
        asr2d.integrate();
        CHECK_NEAR(asr2d.result(), 2.6666667, 1e-6);
    }

    // --- Integration statics (Test_Integration.cs) ----------------------------------------------
    // Test_GaussLegendre
    {
        double e = integ::Integration::gauss_legendre(FX3, 0.0, 1.0);
        CHECK_NEAR(e, 0.25, 1e-3);
    }
    // Test_GaussLegendre20
    {
        double e1 = integ::Integration::gauss_legendre20(FX3, 0.0, 1.0);
        CHECK_NEAR(e1, 0.25, 1e-14);

        double e2 = integ::Integration::gauss_legendre20(
                [](double x) { return std::cos(x); }, 0.0, 1.0);
        CHECK_NEAR(e2, std::sin(1.0), 1e-14);

        double e3 = integ::Integration::gauss_legendre20(
                [](double x) { return std::log(x); }, 1.0, 2.0);
        CHECK_NEAR(e3, 2.0 * std::log(2.0) - 1.0, 1e-14);
    }
    // Test_TrapezoidalRule
    {
        double e = integ::Integration::trapezoidal_rule(FX3, 0.0, 1.0, 1000);
        CHECK_NEAR(e, 0.25, 1e-3);
    }
    // Test_SimpsonsRule
    {
        double e = integ::Integration::simpsons_rule(FX3, 0.0, 1.0, 1000);
        CHECK_NEAR(e, 0.25, 1e-3);
    }
    // Test_MidPoint
    {
        double e = integ::Integration::midpoint(FX3, 0.0, 1.0, 1000);
        CHECK_NEAR(e, 0.25, 1e-3);
    }

    return chtest::summary("integration_extras");
}
