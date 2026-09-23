// Standalone tests for the DataFrame bootstrap / resampling surface (A3):
// next_integers overloads #2/#3, DataFrame::JackKnife, DataFrame::Resample,
// DataFrame::BootstrapDataFrame, and DataFrame::shift_distribution.
//
// Upstream oracle reality (verified): DataFrameTests.cs contains NO JackKnife /
// Resample / BootstrapDataFrame tests, and there is no DataFrameBootstrapTests.cs. The one
// bootstrap-adjacent upstream test file exercises the BootstrapDiagnostics DTO, which is
// ported in A8, not A3. So this C++-only ctest is built from:
//   1. structural / analytic invariants transcribed from the C# method bodies
//      (RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataFrame.cs @ fc28c0c, lines
//      2059-2543; Numerics/Utilities/ExtensionMethods.cs @ 7e8e8d1, lines 78/94),
//   2. seeded MersenneTwister determinism (the C# stream-parity guarantee), and
//   3. draw-replication: a parallel same-seed MersenneTwister reproduces the exact
//      inverse_cdf(u) sequence the methods consume, checked to rel 1e-9.
//
// Skipped upstream test methods (documented in the A3 report): BootstrapDiagnostics DTO
// tests -> A8; DataFrameConcurrencyTests -> the port is single-threaded by design.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/generalized_beta.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/pert.hpp"
#include "corehydro/numerics/distributions/student_t.hpp"
#include "corehydro/numerics/distributions/triangular.hpp"
#include "corehydro/numerics/distributions/truncated_normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/distributions/binomial.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"
#include "check.hpp"

using corehydro::models::DataFrame;
using corehydro::models::ExactData;
using corehydro::models::ExactSeries;
using corehydro::models::IntervalData;
using corehydro::models::IntervalSeries;
using corehydro::models::ThresholdData;
using corehydro::models::ThresholdSeries;
using corehydro::models::UncertainData;
using corehydro::models::UncertainSeries;
using corehydro::numerics::sampling::MersenneTwister;
using corehydro::numerics::utilities::next_integers;
namespace dist = corehydro::numerics::distributions;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// Relative-tolerance floating comparison (rel 1e-9, the deterministic-point policy).
void check_rel(double actual, double expected, const char* what, double rel = 1e-9) {
    double denom = std::max(1.0, std::fabs(expected));
    CHECK_NEAR(actual, expected, rel * denom);
    (void)what;
}

// ── next_integers overloads #2/#3 ──────────────────────────────────────────────

void test_next_integers_min_max_length() {
    MersenneTwister mt(12345);
    std::vector<int> v = next_integers(mt, 5, 10, 200);
    CHECK_EQ(v.size(), static_cast<std::size_t>(200));
    bool all_in_range = true;
    for (int x : v)
        if (x < 5 || x >= 10) all_in_range = false;
    CHECK_TRUE(all_in_range);
}

void test_next_integers_min_max_length_seeded_parity() {
    MersenneTwister a(999);
    MersenneTwister b(999);
    std::vector<int> va = next_integers(a, 0, 1000, 128);
    std::vector<int> vb = next_integers(b, 0, 1000, 128);
    CHECK_TRUE(va == vb);
}

void test_next_integers_replace_true_matches_four_arg() {
    // replace == true branch is identical to the 4-arg overload (same draws).
    MersenneTwister a(7);
    MersenneTwister b(7);
    std::vector<int> va = next_integers(a, 3, 17, 50);
    std::vector<int> vb = next_integers(b, 3, 17, 50, true);
    CHECK_TRUE(va == vb);
}

void test_next_integers_replace_false_distinct() {
    MersenneTwister mt(2024);
    std::vector<int> v = next_integers(mt, 0, 10, 10, false);
    CHECK_EQ(v.size(), static_cast<std::size_t>(10));
    std::set<int> distinct(v.begin(), v.end());
    CHECK_EQ(distinct.size(), static_cast<std::size_t>(10));  // all distinct
    bool all_in_range = true;
    for (int x : v)
        if (x < 0 || x >= 10) all_in_range = false;
    CHECK_TRUE(all_in_range);
}

void test_next_integers_replace_false_throws_when_too_long() {
    MersenneTwister mt(1);
    CHECK_THROWS(next_integers(mt, 0, 3, 5, false));
}

// ── shift_distribution ─────────────────────────────────────────────────────────

// Every additive AND multiplicative arm with positive means shifts the distribution's
// mean exactly to newCenter (location shift / pure scaling both map mean -> newCenter).
void test_shift_distribution_additive_arms_move_mean() {
    const double nc = 25.0;

    dist::Normal normal(10.0, 2.0);
    check_rel(DataFrame::shift_distribution(normal, nc)->mean(), nc, "Normal");

    dist::TruncatedNormal tn(10.0, 2.0, 5.0, 15.0);
    check_rel(DataFrame::shift_distribution(tn, nc)->mean(), nc, "TruncatedNormal");

    dist::StudentT st(10.0, 2.0, 5.0);
    check_rel(DataFrame::shift_distribution(st, nc)->mean(), nc, "StudentT");

    dist::LnNormal lnn(10.0, 2.0);
    check_rel(DataFrame::shift_distribution(lnn, nc)->mean(), nc, "LnNormal");

    dist::Uniform u(0.0, 20.0);
    check_rel(DataFrame::shift_distribution(u, nc)->mean(), nc, "Uniform");

    dist::Triangular t(0.0, 10.0, 20.0);
    check_rel(DataFrame::shift_distribution(t, nc)->mean(), nc, "Triangular");

    dist::Pert p(0.0, 10.0, 20.0);
    check_rel(DataFrame::shift_distribution(p, nc)->mean(), nc, "Pert");

    dist::GeneralizedBeta gb(2.0, 2.0, 0.0, 20.0);
    check_rel(DataFrame::shift_distribution(gb, nc)->mean(), nc, "GeneralizedBeta");
}

void test_shift_distribution_multiplicative_arms_preserve_cv() {
    const double nc = 25.0;

    // LogNormal: the C# adds a natural-log(ratio) to Mu, but Numerics LogNormal is base-10
    // parameterized, so the mean does NOT land on newCenter (a faithful-to-C# quirk); the
    // CV (a function of sigma only, which is unchanged) IS preserved. We assert only CV
    // preservation, the invariant the brief states for the multiplicative arms.
    dist::LogNormal ln(1.0, 0.5);
    double cv_ln = ln.standard_deviation() / ln.mean();
    auto sln = DataFrame::shift_distribution(ln, nc);
    check_rel(sln->standard_deviation() / sln->mean(), cv_ln, "LogNormal CV", 1e-7);

    // Gamma: the scale shift (theta * ratio) is a true scaling, so mean = kappa*theta
    // scales linearly to newCenter AND the CV = 1/sqrt(kappa) is preserved.
    dist::GammaDistribution g(3.0, 4.0);  // theta, kappa
    double cv_g = g.standard_deviation() / g.mean();
    auto sg = DataFrame::shift_distribution(g, nc);
    check_rel(sg->mean(), nc, "Gamma mean");
    check_rel(sg->standard_deviation() / sg->mean(), cv_g, "Gamma CV", 1e-7);
}

void test_shift_distribution_nan_inf_guard_returns_clone() {
    dist::Normal normal(10.0, 2.0);
    // shift = NaN - 10 = NaN -> clone (mean unchanged).
    check_rel(DataFrame::shift_distribution(normal, kNaN)->mean(), 10.0, "NaN guard");
    // shift = Inf - 10 = Inf -> clone (mean unchanged).
    check_rel(DataFrame::shift_distribution(normal, kInf)->mean(), 10.0, "Inf guard");
}

void test_shift_distribution_default_arm_clones() {
    // Binomial is not in the switch -> default clone (mean = n*p unchanged).
    dist::Binomial binom(0.5, 10);
    check_rel(DataFrame::shift_distribution(binom, 25.0)->mean(), 5.0, "default clone");
}

// ── JackKnife ──────────────────────────────────────────────────────────────────

DataFrame make_exact_frame() {
    DataFrame df;
    ExactSeries es;
    es.add(ExactData(0, 10.0));
    es.add(ExactData(1, 20.0));
    es.add(ExactData(2, 30.0));
    df.set_exact_series(std::move(es));
    return df;
}

void test_jackknife_removes_exact_ordinate_and_leaves_parent() {
    DataFrame df = make_exact_frame();
    DataFrame jk = df.JackKnife(1);

    CHECK_EQ(jk.exact_series().count(), static_cast<std::size_t>(2));
    // index 1 (value 20) is gone; 0 and 2 remain unchanged.
    bool found_removed = false;
    for (std::size_t i = 0; i < jk.exact_series().count(); i++)
        if (jk.exact_series()[i].index() == 1) found_removed = true;
    CHECK_TRUE(!found_removed);
    // Parent untouched (jackknife returns a new frame).
    CHECK_EQ(df.exact_series().count(), static_cast<std::size_t>(3));
}

void test_jackknife_threshold_decrement() {
    // Threshold window [0,9], NumberAbove=2 -> above region [0,2), below region [2,9].
    DataFrame df;
    ThresholdSeries ts;
    ThresholdData td(0, 9, 100.0);
    td.set_number_above(2);
    td.set_number_below(8);
    ts.add(td);
    df.set_threshold_series(std::move(ts));

    // index 0 falls in the above region -> NumberAbove decremented.
    DataFrame jk_above = df.JackKnife(0);
    CHECK_EQ(jk_above.threshold_series()[0].number_above(), 1);
    CHECK_EQ(jk_above.threshold_series()[0].number_below(), 8);

    // index 5 falls in the below region -> NumberBelow decremented.
    DataFrame jk_below = df.JackKnife(5);
    CHECK_EQ(jk_below.threshold_series()[0].number_above(), 2);
    CHECK_EQ(jk_below.threshold_series()[0].number_below(), 7);

    // Parent untouched.
    CHECK_EQ(df.threshold_series()[0].number_above(), 2);
    CHECK_EQ(df.threshold_series()[0].number_below(), 8);
}

// ── Resample ───────────────────────────────────────────────────────────────────

void test_resample_preserves_count_and_membership() {
    DataFrame df;
    ExactSeries es;
    for (int i = 0; i < 5; i++) es.add(ExactData(i, 100.0 + i));
    df.set_exact_series(std::move(es));

    MersenneTwister mt(42);
    DataFrame rs = df.Resample(mt);

    // Sampling with replacement preserves the full-time-series count.
    CHECK_EQ(static_cast<int>(rs.full_time_series().size()),
             static_cast<int>(df.full_time_series().size()));

    // Every resampled exact value is drawn from the parent, and indices are sorted.
    std::set<double> parent_values;
    for (std::size_t i = 0; i < df.exact_series().count(); i++)
        parent_values.insert(df.exact_series()[i].value());
    bool all_members = true;
    bool sorted_indices = true;
    int prev = std::numeric_limits<int>::min();
    for (std::size_t i = 0; i < rs.exact_series().count(); i++) {
        if (parent_values.find(rs.exact_series()[i].value()) == parent_values.end())
            all_members = false;
        if (rs.exact_series()[i].index() < prev) sorted_indices = false;
        prev = rs.exact_series()[i].index();
    }
    CHECK_TRUE(all_members);
    CHECK_TRUE(sorted_indices);
}

void test_resample_seeded_determinism() {
    DataFrame df;
    ExactSeries es;
    for (int i = 0; i < 8; i++) es.add(ExactData(i, 100.0 + 3.0 * i));
    df.set_exact_series(std::move(es));

    MersenneTwister a(2718);
    MersenneTwister b(2718);
    DataFrame ra = df.Resample(a);
    DataFrame rb = df.Resample(b);

    CHECK_EQ(ra.exact_series().count(), rb.exact_series().count());
    bool identical = ra.exact_series().count() == rb.exact_series().count();
    for (std::size_t i = 0; identical && i < ra.exact_series().count(); i++) {
        if (ra.exact_series()[i].index() != rb.exact_series()[i].index() ||
            ra.exact_series()[i].value() != rb.exact_series()[i].value())
            identical = false;
    }
    CHECK_TRUE(identical);
}

// ── BootstrapDataFrame ───────────────────────────────────────────────────────────

void test_bootstrap_exact_equals_inverse_cdf_of_draw() {
    DataFrame df = make_exact_frame();
    dist::Normal fitted(50.0, 10.0);

    MersenneTwister mt(555);
    DataFrame bs = df.BootstrapDataFrame(fitted, mt);

    CHECK_EQ(bs.exact_series().count(), static_cast<std::size_t>(3));

    // Replicate the draw sequence: one next_double() per exact ordinate, in order.
    MersenneTwister rep(555);
    for (std::size_t i = 0; i < bs.exact_series().count(); i++) {
        double u = rep.next_double();
        double expected = fitted.inverse_cdf(u);
        check_rel(bs.exact_series()[i].value(), expected, "bootstrap exact");
        CHECK_EQ(bs.exact_series()[i].index(), static_cast<int>(i));
    }
}

void test_bootstrap_interval_reclassification_respects_bounds() {
    DataFrame df;
    IntervalSeries is;
    is.add(IntervalData(0, 40.0, 50.0, 60.0));
    df.set_interval_series(std::move(is));
    dist::Normal fitted(50.0, 10.0);

    MersenneTwister mt(777);
    DataFrame bs = df.BootstrapDataFrame(fitted, mt);
    CHECK_EQ(bs.interval_series().count(), static_cast<std::size_t>(1));

    // Replicate: exact loop (none), then one draw for the single interval.
    MersenneTwister rep(777);
    double u = rep.next_double();
    double sim = fitted.inverse_cdf(u);
    const IntervalData& out = bs.interval_series()[0];
    if (sim < 40.0) {
        check_rel(out.upper_value(), 40.0, "interval left-censored upper");
        CHECK_TRUE(out.lower_value() < out.upper_value());
    } else if (sim > 60.0) {
        check_rel(out.lower_value(), 60.0, "interval right-censored lower");
        CHECK_TRUE(out.lower_value() < out.upper_value());
    } else {
        check_rel(out.lower_value(), 40.0, "interval preserved lower");
        check_rel(out.upper_value(), 60.0, "interval preserved upper");
    }
    // Valid interval ordering in all branches.
    CHECK_TRUE(out.lower_value() <= out.value() && out.value() <= out.upper_value());
}

void test_bootstrap_threshold_systematic_number_below_zero() {
    // A fully-covered window: exact data at every index below a very high threshold, so
    // ProcessThresholdSeries pins NumberBelow to 0 -> the bin is cloned, not resampled.
    DataFrame df;
    ExactSeries es;
    for (int i = 0; i <= 4; i++) es.add(ExactData(i, 10.0 + i));
    df.set_exact_series(std::move(es));
    ThresholdSeries ts;
    ThresholdData td(0, 4, 1.0e6);  // threshold value far above the data
    td.set_number_above(0);
    ts.add(td);
    df.set_threshold_series(std::move(ts));
    df.process_threshold_series();
    CHECK_EQ(df.threshold_series()[0].number_below(), 0);  // fully covered

    dist::Normal fitted(12.0, 3.0);
    MersenneTwister mt(31);
    DataFrame bs = df.BootstrapDataFrame(fitted, mt);
    CHECK_EQ(bs.threshold_series().count(), static_cast<std::size_t>(1));
    // Systematic clone path: no spurious NumberAbove from Binomial resampling.
    CHECK_EQ(bs.threshold_series()[0].number_above(), 0);
    CHECK_EQ(bs.threshold_series()[0].number_below(), 0);
}

void test_bootstrap_threshold_historical_nabove_in_range() {
    // A historical window with unobserved years (NumberBelow > 0, no overlapping data).
    DataFrame df;
    ThresholdSeries ts;
    ThresholdData td(0, 9, 50.0);  // duration 10
    td.set_number_above(1);
    td.set_number_below(9);
    ts.add(td);
    df.set_threshold_series(std::move(ts));

    dist::Normal fitted(50.0, 10.0);  // p = 1 - CDF(50) = 0.5
    int n = 10 - 1;                   // Duration - NumberAbove

    // Run many seeds; nAbove (post-process) must always stay within [0, n].
    bool in_range = true;
    for (std::uint32_t seed = 1; seed <= 200; seed++) {
        MersenneTwister mt(seed);
        DataFrame bs = df.BootstrapDataFrame(fitted, mt);
        int na = bs.threshold_series()[0].number_above();
        if (na < 0 || na > n) in_range = false;
    }
    CHECK_TRUE(in_range);
}

void test_bootstrap_seeded_determinism() {
    DataFrame df = make_exact_frame();
    dist::Normal fitted(50.0, 10.0);

    MersenneTwister a(4096);
    MersenneTwister b(4096);
    DataFrame ba = df.BootstrapDataFrame(fitted, a);
    DataFrame bb = df.BootstrapDataFrame(fitted, b);

    bool identical = ba.exact_series().count() == bb.exact_series().count();
    for (std::size_t i = 0; identical && i < ba.exact_series().count(); i++)
        if (ba.exact_series()[i].value() != bb.exact_series()[i].value()) identical = false;
    CHECK_TRUE(identical);
}

}  // namespace

int main() {
    // next_integers overloads
    test_next_integers_min_max_length();
    test_next_integers_min_max_length_seeded_parity();
    test_next_integers_replace_true_matches_four_arg();
    test_next_integers_replace_false_distinct();
    test_next_integers_replace_false_throws_when_too_long();

    // shift_distribution
    test_shift_distribution_additive_arms_move_mean();
    test_shift_distribution_multiplicative_arms_preserve_cv();
    test_shift_distribution_nan_inf_guard_returns_clone();
    test_shift_distribution_default_arm_clones();

    // JackKnife
    test_jackknife_removes_exact_ordinate_and_leaves_parent();
    test_jackknife_threshold_decrement();

    // Resample
    test_resample_preserves_count_and_membership();
    test_resample_seeded_determinism();

    // BootstrapDataFrame
    test_bootstrap_exact_equals_inverse_cdf_of_draw();
    test_bootstrap_interval_reclassification_respects_bounds();
    test_bootstrap_threshold_systematic_number_below_zero();
    test_bootstrap_threshold_historical_nabove_in_range();
    test_bootstrap_seeded_determinism();

    return chtest::summary("data_frame_bootstrap");
}
