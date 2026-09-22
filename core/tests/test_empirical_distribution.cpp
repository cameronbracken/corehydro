// C++-only ctest for the v2.1.4 Numerics `EmpiricalDistribution` ValidateData wave
// (EmpiricalDistribution.cs @ 2a0357a): at least two ordinates, matching x/p lengths,
// nondecreasing x (strictX = false, so duplicate/tied x values are now valid), finite
// probabilities in [0, 1], and probabilities strictly monotonic in the DECLARED direction
// (ascending -> ordinary CDF, the default; descending -> survival-function encoding, flipped
// via 1-p -- an explicit opt-in via the constructor's `p_descending` argument, NOT
// auto-detected from the data; see empirical_distribution.hpp's header note).
//
// These mirror the new cases in
// `Numerics/Test_Numerics/Distributions/Univariate/Test_EmpiricalDistribution.cs`
// (@ 2a0357a): `Test_NonStrictDuplicateXValues_AreValidForCdfAndInverseCdf`,
// `Test_DecreasingXValues_ThrowWhenUsed`, `Test_NonFiniteProbability_ThrowsWhenUsed`, and
// `Test_DescendingProbabilityOrder_SupportsCdfAndInverseCdf`. The "valid" cases (duplicate-x,
// descending-probability-order) are ALSO pinned as oracle fixture cases in
// fixtures/distributions/univariate/empirical_distribution.json (parameters_valid + exact
// cdf/quantile values reproduced against the real C#); they are additionally exercised here
// directly for fast, dependency-free coverage of the underlying interpolation code path.
// `test_descending_probability_without_declaring_is_invalid` is a corehydro-only regression
// guard (no direct C# test) added after the dotnet oracle gate caught an earlier auto-detect
// design reproducing a false positive for this exact case.
//
// `test_length_mismatch_throws_eagerly_at_construction` pins the ONE eager rule: C#'s
// SetParameters(x, p) throws ArgumentException immediately on a length mismatch, before
// ValidateData runs -- unlike every other rule (including "too few points," asserted
// immediately below it), which stays lazy. A code-review pass on this task caught an earlier
// version of this port folding the length check into the lazy parameters_valid()/throw-on-use
// path, which silently diverges from C# (construction succeeds in the port, throws in C#).
//
// The "throws when used" cases (decreasing-x, non-finite probability) have NO equivalent in
// fixtures/README.md's schema: the generic runner's `parameters_valid` assertion (`mode:
// "bool"`) can check that construction leaves the distribution invalid, but there is no
// "assert this method call throws" assertion shape (documented gap from the T3 distribution-
// validation wave), and calling cdf()/inverse_cdf() on an invalid case directly from the JSON
// fixture would abort the C++/R/Python test binaries with an uncaught exception. So those two
// behaviors are pinned here instead, following the `test_mixture.cpp` C++-only-ctest
// precedent; the fixture's own cases only assert `parameters_valid: false` for the same
// invalid inputs (see the `decreasing_x_invalid` / `non_finite_probability_invalid` cases).
#include <cmath>
#include <limits>
#include <vector>

#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "check.hpp"

using corehydro::numerics::distributions::EmpiricalDistribution;
using corehydro::numerics::distributions::EmpiricalTransform;
using corehydro::numerics::data::paired_data::ExtrapolationSides;

namespace {

// Mirrors Test_NonStrictDuplicateXValues_AreValidForCdfAndInverseCdf: duplicate (nondecreasing,
// not strictly increasing) x-values are valid now that strictX = false.
void test_duplicate_x_valid() {
    EmpiricalDistribution dist({100.0, 100.0, 125.0, 150.0, 200.0},
                               {0.1, 0.2, 0.45, 0.7, 0.95});
    CHECK_TRUE(dist.parameters_valid());
    double cdf = dist.cdf(125.0);
    CHECK_TRUE(!std::isnan(cdf) && !std::isinf(cdf));
    CHECK_TRUE(cdf >= 0.0 && cdf <= 1.0);
    CHECK_NEAR(dist.inverse_cdf(0.45), 125.0, 1e-12);
}

// Mirrors Test_DecreasingXValues_ThrowWhenUsed: a strictly-decreasing step anywhere in x makes
// the whole distribution invalid; CDF/InverseCDF throw lazily (C# ArgumentOutOfRangeException;
// this port's convention is std::invalid_argument -- see empirical_distribution.hpp header).
void test_decreasing_x_throws_when_used() {
    EmpiricalDistribution dist({100.0, 150.0, 125.0}, {0.1, 0.5, 0.9});
    CHECK_TRUE(!dist.parameters_valid());
    CHECK_THROWS(dist.cdf(125.0));
    CHECK_THROWS(dist.inverse_cdf(0.5));
}

// Mirrors Test_NonFiniteProbability_ThrowsWhenUsed: a NaN (or +-Infinity) probability anywhere
// invalidates the distribution; CDF throws lazily.
void test_non_finite_probability_throws_when_used() {
    EmpiricalDistribution dist({100.0, 125.0, 150.0},
                               {0.1, std::numeric_limits<double>::quiet_NaN(), 0.9});
    CHECK_TRUE(!dist.parameters_valid());
    CHECK_THROWS(dist.cdf(125.0));
    CHECK_THROWS(dist.inverse_cdf(0.5));
}

// Out-of-[0,1] but finite probabilities are invalid too (the ValidateData range check is
// distinct from the NaN/Infinity check -- both are asserted so neither branch is unreachable).
void test_out_of_range_probability_invalid() {
    EmpiricalDistribution dist({100.0, 125.0, 150.0}, {0.1, 1.5, 0.9});
    CHECK_TRUE(!dist.parameters_valid());
    CHECK_THROWS(dist.cdf(125.0));
}

// A length mismatch between x and p throws EAGERLY, at construction -- mirrors C#'s
// SetParameters(x, p) throwing ArgumentException immediately, before ValidateData (and thus
// before every other rule) ever runs. Reproduced against the real C#: SetParameters({100,125,150},
// {0.1,0.9}) throws in C#, so a port that only set parameters_valid() = false here (and let
// construction succeed) would silently diverge.
void test_length_mismatch_throws_eagerly_at_construction() {
    bool threw = false;
    try {
        EmpiricalDistribution mismatched({100.0, 125.0, 150.0}, {0.1, 0.9});
    } catch (...) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

// Fewer-than-two ordinates, by contrast, IS one of the lazy ValidateData rules: construction
// succeeds with parameters_valid() == false, and cdf()/inverse_cdf() throw only when used.
void test_too_few_points_invalid_lazily() {
    EmpiricalDistribution single({100.0}, {0.5});
    CHECK_TRUE(!single.parameters_valid());
    CHECK_THROWS(single.cdf(100.0));
}

// Probabilities that are neither strictly ascending nor strictly descending are invalid.
void test_non_monotonic_probability_invalid() {
    EmpiricalDistribution dist({100.0, 125.0, 150.0, 175.0}, {0.1, 0.5, 0.3, 0.9});
    CHECK_TRUE(!dist.parameters_valid());
    CHECK_THROWS(dist.cdf(125.0));
}

// Mirrors Test_DescendingProbabilityOrder_SupportsCdfAndInverseCdf: ascending x with strictly
// descending p, EXPLICITLY declared via the p_descending constructor argument (mirroring C#'s
// `SetParameters(x, p, SortOrder.Ascending, SortOrder.Descending)`), is a valid survival-
// function encoding; CDF/InverseCDF flip via 1-p internally.
void test_descending_probability_order_supported() {
    EmpiricalDistribution dist({100.0, 150.0, 200.0}, {0.9, 0.5, 0.1},
                               EmpiricalTransform::NormalZ, /*p_descending=*/true);
    CHECK_TRUE(dist.parameters_valid());
    CHECK_NEAR(dist.cdf(150.0), 0.5, 1e-12);
    CHECK_NEAR(dist.inverse_cdf(0.5), 150.0, 1e-12);
}

// The direction is DECLARED, not auto-detected: the same descending p array constructed with
// the default p_descending = false (mirroring the plain SetParameters(x, p) overload's
// hardcoded SortOrder.Ascending) is INVALID -- confirmed against the real C# via the dotnet
// oracle gate, which rejects this exact case with "Y values must increase."
void test_descending_probability_without_declaring_is_invalid() {
    EmpiricalDistribution dist({100.0, 150.0, 200.0}, {0.9, 0.5, 0.1});
    CHECK_TRUE(!dist.parameters_valid());
    CHECK_THROWS(dist.cdf(150.0));
}

void test_sided_extrapolation() {
    EmpiricalDistribution ascending({1.0, 2.0}, {0.25, 0.75}, EmpiricalTransform::None);
    CHECK_NEAR(ascending.inverse_cdf(1e-16), 1.0, 0.0);
    ascending.set_extrapolation(ExtrapolationSides::Both);
    CHECK_NEAR(ascending.inverse_cdf(1e-16), 0.5, 1e-12);
    CHECK_NEAR(ascending.inverse_cdf(1.0 - 1e-16), 2.5, 1e-12);
    CHECK_NEAR(ascending.cdf(0.0), 0.0, 0.0);
    CHECK_NEAR(ascending.cdf(3.0), 1.0, 0.0);

    EmpiricalDistribution descending({1.0, 2.0}, {0.75, 0.25}, EmpiricalTransform::None,
                                     /*p_descending=*/true);
    descending.set_extrapolation(ExtrapolationSides::Both);
    CHECK_NEAR(descending.inverse_cdf(1e-16), 0.5, 1e-12);
    CHECK_NEAR(descending.inverse_cdf(1.0 - 1e-16), 2.5, 1e-12);

    auto clone = ascending.clone();
    CHECK_NEAR(clone->inverse_cdf(1e-16), 0.5, 1e-12);
}

}  // namespace

int main() {
    test_duplicate_x_valid();
    test_decreasing_x_throws_when_used();
    test_non_finite_probability_throws_when_used();
    test_out_of_range_probability_invalid();
    test_length_mismatch_throws_eagerly_at_construction();
    test_too_few_points_invalid_lazily();
    test_non_monotonic_probability_invalid();
    test_descending_probability_order_supported();
    test_descending_probability_without_declaring_is_invalid();
    test_sided_extrapolation();
    return chtest::summary("test_empirical_distribution");
}
