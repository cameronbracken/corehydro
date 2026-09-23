// Task B4 -- distribution moment machinery for the Bulletin 17C family.
//
// Upstream test transcription: the ONLY relevant upstream test across
// Test_Numerics/Distributions/Univariate/ is Test_Exponential.cs:158 Test_EXP_Partials
// (QuantileGradient). Grep-verified: Test_Normal.cs, Test_GammaDistribution.cs,
// Test_PearsonTypeIII.cs, Test_LogPearsonTypeIII.cs, Test_LnNormal.cs, and
// Test_LogNormal.cs contain NO gradient/partial tests, and NO ConditionalMoments or
// ParametersFromMoments/MomentsFromParameters tests exist anywhere upstream. The
// GEV/GLO/GPA/GNO/Kappa4 gradient tests exercise classes already ported (or out of
// scope for B4) and are deliberately skipped here.
//
// Everything else below is an analytic closed-form or self-consistency check, with
// tolerances justified inline. These members are internal support until B12 emits
// fixtures; emitter corroboration of this numeric surface lands at B12, so hardcoded
// oracles in this C++-only ctest are acceptable (per the B4 brief).
#include <cmath>
#include <limits>
#include <vector>

#include "corehydro/numerics/distributions/exponential.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/gumbel.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/log_pearson_type_iii.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/pearson_type_iii.hpp"
#include "corehydro/numerics/tools.hpp"
#include "check.hpp"

using namespace corehydro::numerics;
using namespace corehydro::numerics::distributions;

// Relative-tolerance check built on CHECK_NEAR (expected must be nonzero).
#define CHECK_REL(actual, expected, rtol)                                   \
    do {                                                                    \
        double _exp_rel = (expected);                                       \
        CHECK_NEAR((actual), _exp_rel, std::fabs(_exp_rel) * (rtol));       \
    } while (0)

namespace {

// Standard normal pdf/cdf helpers for the closed-form truncated-normal expectations,
// computed independently of the distribution classes under test.
double phi(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi); }
double Phi(double x) { return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))); }

bool all_nan(const std::vector<double>& v) {
    if (v.size() != 4) return false;
    for (double x : v)
        if (!std::isnan(x)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Transcribed upstream test: Test_Exponential.cs:158 Test_EXP_Partials
// ---------------------------------------------------------------------------

// Exponential(30314.48, 18907.87): dQ/dLocation(0.99) = 1.0, dQ/dScale(0.99) = 4.60517,
// rel 0.01 (the C# asserts (actual-true)/true < 0.01; transcribed as |rel| <= 0.01).
void exp_partials_transcribed() {
    Exponential EXP(30314.48, 18907.87);
    double dQdLocation = EXP.quantile_gradient(0.99)[0];
    double dQdScale = EXP.quantile_gradient(0.99)[1];
    CHECK_REL(dQdLocation, 1.0, 0.01);
    CHECK_REL(dQdScale, 4.60517, 0.01);
}

// ---------------------------------------------------------------------------
// QuantileGradient -- analytic + finite-difference cross-checks
// ---------------------------------------------------------------------------

// Normal: Q(p) = mu + sigma z(p), so dQ/dmu = 1 and dQ/dsigma = z(p) exactly. The FD
// cross-check against inverse_cdf uses a central difference (error O(h^2) ~ 1e-10 at
// h = 1e-5 * scale); rel 1e-7 leaves two orders of margin.
void normal_quantile_gradient() {
    Normal n(100.0, 15.0);
    for (double p : {0.1, 0.5, 0.9, 0.99}) {
        auto g = n.quantile_gradient(p);
        CHECK_EQ(g[0], 1.0);
        CHECK_EQ(g[1], Normal::standard_z(p));
        // FD wrt mu and sigma
        double h = 1e-5 * 15.0;
        double dmu = (Normal(100.0 + h, 15.0).inverse_cdf(p) -
                      Normal(100.0 - h, 15.0).inverse_cdf(p)) / (2.0 * h);
        double dsig = (Normal(100.0, 15.0 + h).inverse_cdf(p) -
                       Normal(100.0, 15.0 - h).inverse_cdf(p)) / (2.0 * h);
        CHECK_REL(g[0], dmu, 1e-7);
        if (p != 0.5) CHECK_REL(g[1], dsig, 1e-7);  // z(0.5) = 0: nothing to compare
    }
}

// Exponential: Q(p) = xi - alpha log(1-p): dQ/dxi = 1, dQ/dalpha = -log(1-p) exactly;
// FD tolerance as for Normal.
void exponential_quantile_gradient() {
    Exponential e(10.0, 5.0);
    for (double p : {0.1, 0.5, 0.99}) {
        auto g = e.quantile_gradient(p);
        CHECK_EQ(g[0], 1.0);
        CHECK_EQ(g[1], -std::log1p(-p));
        double h = 1e-6 * 5.0;
        double dalpha = (Exponential(10.0, 5.0 + h).inverse_cdf(p) -
                         Exponential(10.0, 5.0 - h).inverse_cdf(p)) / (2.0 * h);
        CHECK_REL(g[1], dalpha, 1e-7);
    }
}

// Gamma: the analytic gradient goes through the Cornish-Fisher FrequencyFactorKp
// approximation, so agreement with an FD of the exact inverse_cdf is limited by the
// CF accuracy, not the FD. At kappa = 100 (skew = 0.2) the CF error is < 1e-6
// (measured ~1e-7); rel 1e-5 per the brief's guidance for gradient FD checks.
void gamma_quantile_gradient_fd() {
    double theta = 2.0, kappa = 100.0;
    GammaDistribution g(theta, kappa);
    for (double p : {0.1, 0.9, 0.99}) {
        auto grad = g.quantile_gradient(p);
        double ht = 1e-5 * theta;
        double dtheta = (GammaDistribution(theta + ht, kappa).inverse_cdf(p) -
                         GammaDistribution(theta - ht, kappa).inverse_cdf(p)) / (2.0 * ht);
        double hk = 1e-4 * kappa;
        double dkappa = (GammaDistribution(theta, kappa + hk).inverse_cdf(p) -
                         GammaDistribution(theta, kappa - hk).inverse_cdf(p)) / (2.0 * hk);
        CHECK_REL(grad[0], dtheta, 1e-5);
        CHECK_REL(grad[1], dkappa, 1e-5);
    }
}

// FrequencyFactorKp: near-zero skew returns the standard normal Z variate exactly
// (the C# |C| < 1e-4 branch).
void frequency_factor_kp_small_skew() {
    for (double p : {0.1, 0.5, 0.9, 0.999}) {
        CHECK_EQ(GammaDistribution::frequency_factor_kp(0.0, p), Normal::standard_z(p));
        CHECK_EQ(GammaDistribution::frequency_factor_kp(5e-5, p), Normal::standard_z(p));
    }
}

// FrequencyFactorKp vs the exact standardized Gamma quantile K = (Q(p) - mean)/sd.
// Cornish-Fisher branch (skew 0.2): measured error ~1e-7; assert 1e-5.
// Modified Wilson-Hilferty branch (skew 3): Kirby (1972) polynomial fit; measured
// error 1.4% at p = 0.99 (and at the skew-2 branch boundary CF/WH straddle the exact
// value at ~+/-0.5% each, confirming approximation error, not a port bug); assert 3e-2.
void frequency_factor_kp_vs_exact() {
    {
        double kappa = 100.0, theta = 2.0;  // skew = 2/sqrt(kappa) = 0.2
        GammaDistribution g(theta, kappa);
        for (double p : {0.1, 0.9, 0.99}) {
            double k_exact = (g.inverse_cdf(p) - g.mean()) / g.standard_deviation();
            CHECK_REL(GammaDistribution::frequency_factor_kp(0.2, p), k_exact, 1e-5);
        }
    }
    {
        double kappa = 4.0 / 9.0, theta = 1.0;  // skew = 3
        GammaDistribution g(theta, kappa);
        for (double p : {0.9, 0.99}) {
            double k_exact = (g.inverse_cdf(p) - g.mean()) / g.standard_deviation();
            CHECK_REL(GammaDistribution::frequency_factor_kp(3.0, p), k_exact, 3e-2);
        }
    }
}

// PartialKp is the exact term-by-term derivative of the CF polynomial for |skew| <= 2,
// so a central FD of frequency_factor_kp wrt skew must match to FD accuracy (~1e-9);
// assert rel 1e-6. For |skew| > 2 the C# DEFINES PartialKp as
// NumericalDerivative.Derivative(Kp, skew, 1e-4) -- reproduce that call in the test and
// require exact agreement.
void partial_kp_vs_fd() {
    for (double p : {0.9, 0.99}) {
        double skew = 0.7, h = 1e-5;
        double fd = (GammaDistribution::frequency_factor_kp(skew + h, p) -
                     GammaDistribution::frequency_factor_kp(skew - h, p)) / (2.0 * h);
        CHECK_REL(GammaDistribution::partial_kp(skew, p), fd, 1e-6);

        double skew2 = 2.5, h2 = 1e-4;
        double fd2 = (GammaDistribution::frequency_factor_kp(skew2 + h2, p) -
                      GammaDistribution::frequency_factor_kp(skew2 - h2, p)) / (2.0 * h2);
        CHECK_REL(GammaDistribution::partial_kp(skew2, p), fd2, 1e-12);
    }
}

// PT3 QuantileGradientForMoments = [1, Kp(skew,p), sigma*PartialKp(skew,p)] vs a
// central FD of the exact inverse_cdf wrt each moment. At skew 0.2 the CF error
// dominates the FD error; measured ~1e-7 on dQ/dsigma and ~1e-5 on dQ/dgamma
// (the CF derivative is one order less accurate than the CF value); assert 1e-5 / 1e-3.
void pt3_quantile_gradient_for_moments_fd() {
    double mu = 100.0, sigma = 10.0, skew = 0.2;
    PearsonTypeIII p3(mu, sigma, skew);
    for (double p : {0.1, 0.9, 0.99}) {
        auto g = p3.quantile_gradient_for_moments(p);
        CHECK_EQ(g[0], 1.0);
        double hm = 1e-5 * mu;
        double dmu = (PearsonTypeIII(mu + hm, sigma, skew).inverse_cdf(p) -
                      PearsonTypeIII(mu - hm, sigma, skew).inverse_cdf(p)) / (2.0 * hm);
        CHECK_REL(g[0], dmu, 1e-7);
        double hs = 1e-5 * sigma;
        double dsigma = (PearsonTypeIII(mu, sigma + hs, skew).inverse_cdf(p) -
                         PearsonTypeIII(mu, sigma - hs, skew).inverse_cdf(p)) / (2.0 * hs);
        CHECK_REL(g[1], dsigma, 1e-5);
        double hg = 1e-5;
        double dgamma = (PearsonTypeIII(mu, sigma, skew + hg).inverse_cdf(p) -
                         PearsonTypeIII(mu, sigma, skew - hg).inverse_cdf(p)) / (2.0 * hg);
        CHECK_REL(g[2], dgamma, 1e-3);
    }
}

// ---------------------------------------------------------------------------
// ConditionalMoments -- closed forms + override-vs-base-virtual agreement
// ---------------------------------------------------------------------------
// The base virtual is a 300-bin midpoint integration whose error dominates every
// override-vs-base comparison (the overrides are closed forms, accurate to near
// machine precision). Measured integration error on these windows: ~2e-5 relative on
// m1, up to ~5.4e-4 on the higher central moments (worst near a support bound), so
// override-vs-base comparisons assert rel 1e-3 (~2x margin over the worst measured
// error; a porting mistake -- wrong centering, normalization, or sign -- shifts these
// values by orders of magnitude more). Closed-form m1 checks assert rel 1e-12 (pure
// double arithmetic) and pin the mean independently of the integration reference.

void normal_conditional_moments() {
    double mu = 5.0, sigma = 2.0, a = 4.0, b = 9.0;
    Normal n(mu, sigma);
    auto m = n.conditional_moments(a, b);

    // Truncated-normal closed forms about the UNCONDITIONAL mean, computed
    // independently here.
    double as = (a - mu) / sigma, bs = (b - mu) / sigma;
    double Z = Phi(bs) - Phi(as);
    double lambda = (phi(as) - phi(bs)) / Z;
    double m1 = mu + sigma * lambda;
    CHECK_REL(m[0], m1, 1e-12);
    double delta = (as * phi(as) - bs * phi(bs)) / Z;
    CHECK_REL(m[1], sigma * sigma * (1.0 + delta), 1e-12);

    // Override vs the base virtual's stratified integration on the same window.
    auto mb = n.UnivariateDistributionBase::conditional_moments(a, b);
    for (int i = 0; i < 4; ++i) CHECK_REL(m[i], mb[i], 1e-3);

    // Guards: a >= b.
    CHECK_TRUE(all_nan(n.conditional_moments(4.0, 4.0)));
}

void exponential_conditional_moments() {
    double xi = 10.0, alpha = 5.0;
    Exponential e(xi, alpha);

    // Closed-form truncated-exponential mean on [12, 40] (y = x - xi on [2, 30]):
    // E[Y | A<Y<B] = ((A+alpha) e^{-A/a} - (B+alpha) e^{-B/a}) / (e^{-A/a} - e^{-B/a}).
    double A = 2.0, B = 30.0;
    double eA = std::exp(-A / alpha), eB = std::exp(-B / alpha);
    double m1_closed = xi + ((A + alpha) * eA - (B + alpha) * eB) / (eA - eB);
    auto m = e.conditional_moments(12.0, 40.0);
    CHECK_REL(m[0], m1_closed, 1e-12);

    // Memorylessness: for b effectively at +infinity, E[X | X > a] = a + alpha.
    auto mm = e.conditional_moments(13.0, 1e9);
    CHECK_REL(mm[0], 13.0 + alpha, 1e-12);

    // Override vs base virtual on a finite window.
    auto mb = e.UnivariateDistributionBase::conditional_moments(12.0, 40.0);
    for (int i = 0; i < 4; ++i) CHECK_REL(m[i], mb[i], 1e-3);

    CHECK_TRUE(all_nan(e.conditional_moments(40.0, 12.0)));
    // Interval entirely left of the support: B <= 0 after shifting.
    CHECK_TRUE(all_nan(e.conditional_moments(1.0, 9.0)));
}

void gamma_conditional_moments_vs_base() {
    GammaDistribution g(2.0, 9.0);  // mean 18, sd 6
    auto m = g.conditional_moments(9.0, 30.0);
    auto mb = g.UnivariateDistributionBase::conditional_moments(9.0, 30.0);
    for (int i = 0; i < 4; ++i) CHECK_REL(m[i], mb[i], 1e-3);
    CHECK_TRUE(all_nan(g.conditional_moments(30.0, 9.0)));
}

void ln_normal_conditional_moments() {
    // Real-space mean 100, sd 40.
    LnNormal ln(100.0, 40.0);
    double mu = ln.mu(), sigma = ln.sigma();

    double a = 60.0, b = 220.0;
    auto m = ln.conditional_moments(a, b);

    // Closed-form truncated-lognormal mean (independent computation):
    // E[X | a<X<b] = e^{mu+s^2/2} (Phi(beta-s) - Phi(alpha-s)) / (Phi(beta)-Phi(alpha)).
    double as = (std::log(a) - mu) / sigma, bs = (std::log(b) - mu) / sigma;
    double Z = Phi(bs) - Phi(as);
    double m1_closed = std::exp(mu + 0.5 * sigma * sigma) * (Phi(bs - sigma) - Phi(as - sigma)) / Z;
    CHECK_REL(m[0], m1_closed, 1e-12);

    auto mb = ln.UnivariateDistributionBase::conditional_moments(a, b);
    for (int i = 0; i < 4; ++i) CHECK_REL(m[i], mb[i], 1e-3);

    CHECK_TRUE(all_nan(ln.conditional_moments(b, a)));
}

void pt3_conditional_moments_vs_base() {
    // Moderate positive skew: the smooth blend sits on the exact truncated-Gamma path
    // (|skew| = 0.8 is far above the WH/Normal switch points at 0.10 / 1e-3).
    {
        PearsonTypeIII p3(100.0, 10.0, 0.8);
        auto m = p3.conditional_moments(85.0, 130.0);
        auto mb = p3.UnivariateDistributionBase::conditional_moments(85.0, 130.0);
        for (int i = 0; i < 4; ++i) CHECK_REL(m[i], mb[i], 1e-3);
    }
    // Negative skew (upper-bounded support, Beta < 0 branch).
    {
        PearsonTypeIII p3(100.0, 10.0, -0.8);
        auto m = p3.conditional_moments(80.0, 118.0);
        auto mb = p3.UnivariateDistributionBase::conditional_moments(80.0, 118.0);
        for (int i = 0; i < 4; ++i) CHECK_REL(m[i], mb[i], 1e-3);
        CHECK_TRUE(all_nan(p3.conditional_moments(118.0, 80.0)));
    }
}

// Base-virtual sanity on a distribution WITHOUT an override (Gumbel): over nearly the
// full support the conditional moments reproduce the unconditional Mean/Variance.
// With 300 midpoint bins spanning ~30 sd the integration error was measured at
// ~2e-5 relative on m1 and ~2e-4 on m2; assert 1e-3 (margin over measurement).
void base_virtual_sanity() {
    Gumbel gum(100.0, 20.0);
    double sd = gum.standard_deviation();
    double a = gum.mean() - 10.0 * sd, b = gum.mean() + 20.0 * sd;
    auto m = gum.conditional_moments(a, b);
    CHECK_REL(m[0], gum.mean(), 1e-3);
    CHECK_REL(m[1], gum.variance(), 1e-3);

    // a >= b returns the NaN quadruple.
    CHECK_TRUE(all_nan(gum.conditional_moments(5.0, 5.0)));
    CHECK_TRUE(all_nan(gum.conditional_moments(7.0, 5.0)));

    // Zero-probability window (base virtual; LogNormal has NO override): mass below
    // the support is zero, so totalProb <= 0 -> NaN quadruple.
    LogNormal lg(3.0, 0.5);
    CHECK_TRUE(all_nan(lg.conditional_moments(-5.0, -1.0)));
}

// ---------------------------------------------------------------------------
// IMomentEstimation round trips
// ---------------------------------------------------------------------------
// Normal and PearsonTypeIII parameterize BY their moments, so
// parameters_from_moments(moments_from_parameters(theta)) == theta holds exactly
// (rel 1e-10 asserted). LnNormal / LogNormal / LogPearsonTypeIII have asymmetric
// moment spaces in the C# (parameters and moments live in different spaces); the C#
// source governs, so those tests assert the self-consistency relations the C# actually
// defines -- see each test and the task report.

void normal_moment_round_trip() {
    for (auto theta : {std::vector<double>{100.0, 15.0}, std::vector<double>{-3.0, 0.5}}) {
        Normal n;
        auto m = n.moments_from_parameters(theta);
        CHECK_REL(m[0], theta[0], 1e-10);
        CHECK_REL(m[1], theta[1], 1e-10);
        CHECK_NEAR(m[2], 0.0, 1e-15);   // Normal skew
        CHECK_REL(m[3], 3.0, 1e-15);    // Normal kurtosis
        auto back = n.parameters_from_moments(m);
        CHECK_EQ(static_cast<int>(back.size()), 2);
        CHECK_REL(back[0], theta[0], 1e-10);
        CHECK_REL(back[1], theta[1], 1e-10);
    }
}

void pt3_moment_round_trip() {
    // Includes a negative-skew point (brief requirement).
    for (auto theta : {std::vector<double>{100.0, 10.0, 1.2}, std::vector<double>{50.0, 5.0, -0.8}}) {
        PearsonTypeIII p3;
        auto m = p3.moments_from_parameters(theta);
        CHECK_REL(m[0], theta[0], 1e-10);
        CHECK_REL(m[1], theta[1], 1e-10);
        CHECK_REL(m[2], theta[2], 1e-10);
        // Kurtosis = 3 + 6/alpha with alpha = 4/skew^2 = 3 + 1.5*skew^2.
        CHECK_REL(m[3], 3.0 + 1.5 * theta[2] * theta[2], 1e-10);
        auto back = p3.parameters_from_moments(m);
        CHECK_EQ(static_cast<int>(back.size()), 3);
        for (int i = 0; i < 3; ++i) CHECK_REL(back[i], theta[i], 1e-10);
    }
}

void lp3_moment_methods() {
    // C# LP3 ParametersFromMoments == moments.Subset(0, 2): pure passthrough.
    LogPearsonTypeIII lp3;
    auto back = lp3.parameters_from_moments({2.5, 0.25, -0.4, 3.3});
    CHECK_EQ(static_cast<int>(back.size()), 3);
    CHECK_EQ(back[0], 2.5);
    CHECK_EQ(back[1], 0.25);
    CHECK_EQ(back[2], -0.4);

    // C# LP3 MomentsFromParameters returns the REAL-SPACE moments of X (not the
    // log-space parameters), so the strict round trip does NOT hold upstream; assert
    // the relation the C# defines: mfp(theta) == {Mean, SD, Skew, Kurt} of the
    // distribution built from theta. Includes a negative-skew point.
    for (auto theta : {std::vector<double>{3.0, 0.5, 0.3}, std::vector<double>{2.5, 0.25, -0.4}}) {
        LogPearsonTypeIII d(theta[0], theta[1], theta[2]);
        auto m = lp3.moments_from_parameters(theta);
        CHECK_REL(m[0], d.mean(), 1e-12);
        CHECK_REL(m[1], d.standard_deviation(), 1e-12);
        CHECK_REL(m[2], d.skewness(), 1e-12);
        CHECK_REL(m[3], d.kurtosis(), 1e-12);
    }
}

void ln_normal_moment_round_trip() {
    // LnNormal parameters are the REAL-SPACE mean/sd; moments_from_parameters returns
    // real-space moments, so its first two entries reproduce theta (rel 1e-10)...
    std::vector<double> theta{10.0, 3.0};
    LnNormal ln(theta[0], theta[1]);
    auto m = ln.moments_from_parameters(theta);
    CHECK_REL(m[0], theta[0], 1e-10);
    CHECK_REL(m[1], theta[1], 1e-10);
    // ...but parameters_from_moments maps to the LOG-SPACE (mu, sigma) pair (the C#
    // duplicates DirectMethodOfMoments), so the C#-defined round trip lands on the
    // internal log-space parameters, not on theta.
    auto back = ln.parameters_from_moments(m);
    CHECK_REL(back[0], ln.mu(), 1e-10);
    CHECK_REL(back[1], ln.sigma(), 1e-10);
    // Degenerate moments guard: sd <= 0 returns the NaN pair.
    auto bad = ln.parameters_from_moments({10.0, 0.0});
    CHECK_TRUE(std::isnan(bad[0]) && std::isnan(bad[1]));
}

void log_normal_moment_round_trip() {
    // LogNormal (base 10): parameters are the base-10 log-space (mu, sigma).
    // MomentsFromParameters returns real-space moments and ParametersFromMoments converts
    // the natural-log result into the configured base, so both parameters round trip.
    std::vector<double> theta{3.0, 0.5};
    LogNormal lg(theta[0], theta[1]);
    auto m = lg.moments_from_parameters(theta);
    CHECK_REL(m[0], lg.mean(), 1e-12);
    CHECK_REL(m[1], lg.standard_deviation(), 1e-12);
    auto back = lg.parameters_from_moments(m);
    CHECK_REL(back[0], theta[0], 1e-10);
    CHECK_REL(back[1], theta[1], 1e-10);
}

}  // namespace

int main() {
    exp_partials_transcribed();
    normal_quantile_gradient();
    exponential_quantile_gradient();
    gamma_quantile_gradient_fd();
    frequency_factor_kp_small_skew();
    frequency_factor_kp_vs_exact();
    partial_kp_vs_fd();
    pt3_quantile_gradient_for_moments_fd();
    normal_conditional_moments();
    exponential_conditional_moments();
    gamma_conditional_moments_vs_base();
    ln_normal_conditional_moments();
    pt3_conditional_moments_vs_base();
    base_virtual_sanity();
    normal_moment_round_trip();
    pt3_moment_round_trip();
    lp3_moment_methods();
    ln_normal_moment_round_trip();
    log_normal_moment_round_trip();
    return chtest::summary("test_moment_machinery");
}
