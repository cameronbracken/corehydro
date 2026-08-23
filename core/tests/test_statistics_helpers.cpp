// Transcribed from: upstream/Numerics/Numerics/Utilities/Tools.cs (Pow, lines 157-179) and
// upstream/Numerics/Numerics/Data/Statistics/Statistics.cs (MeanVariance line 262,
// Percentile(IList<double>, IList<double>, bool) line 573, RanksInPlace(double[], out double[])
// line 674) @ 2a0357a. P4 Task 1 -- the four small helpers HypothesisTests (P4 Task 2) needs and
// this port had deliberately omitted; see the header notes in tools.hpp / statistics.hpp for the
// full fidelity discussion.
#include <limits>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/tools.hpp"
#include "check.hpp"

namespace bfdata = corehydro::numerics::data;

namespace {

// tools::pow against Tools.Pow's hand-written binary exponentiation edge cases.
void test_pow() {
    CHECK_EQ(corehydro::numerics::pow(2.0, 0), 1.0);
    CHECK_EQ(corehydro::numerics::pow(1.0, -5), 1.0);
    CHECK_EQ(corehydro::numerics::pow(-1.0, 3), -1.0);
    CHECK_EQ(corehydro::numerics::pow(-1.0, 4), 1.0);
    CHECK_EQ(corehydro::numerics::pow(2.0, 10), 1024.0);
    CHECK_EQ(corehydro::numerics::pow(2.0, -2), 0.25);
    CHECK_EQ(corehydro::numerics::pow(0.0, -1), std::numeric_limits<double>::infinity());
}

// mean_variance is C#'s literal (Mean(data), Variance(data)) -- an identity test against the
// already-ported, already-oracle-pinned mean()/variance(), not a new oracle.
void test_mean_variance() {
    const std::vector<double> data{4.1, 7.3, 2.9, 8.8, 5.5, 1.2, 9.9, 3.3, 6.6, 0.4};
    auto [m, v] = bfdata::mean_variance(data);
    CHECK_EQ(m, bfdata::mean(data));
    CHECK_EQ(v, bfdata::variance(data));
}

// ranks_in_place(data, ties) -- the tolerance-based tie overload. `data` carries a tie run in
// the middle (three 20s) AND a tie run at the very end (two 50s), so both the recorded-tie and
// the never-recorded-trailing-tie fidelity points (statistics.hpp's numbered note) get exercised.
void test_ranks_in_place_with_ties() {
    std::vector<double> data{10.0, 20.0, 20.0, 20.0, 30.0, 40.0, 50.0, 50.0};
    std::vector<double> ties;
    auto ranks = bfdata::ranks_in_place(data, ties);

    const std::vector<double> expected_ranks{1.0, 3.0, 3.0, 3.0, 5.0, 6.0, 7.5, 7.5};
    CHECK_EQ(ranks.size(), expected_ranks.size());
    for (std::size_t i = 0; i < expected_ranks.size(); ++i) CHECK_NEAR(ranks[i], expected_ranks[i], 0.0);

    // ties is sparse, indexed by the END of each run (i - 1), and allocated at data.size().
    CHECK_EQ(ties.size(), data.size());
    // The middle run [1,4) (three 20s -> 2 ties) closes inside the loop and IS recorded at
    // index i-1 = 3.
    CHECK_NEAR(ties[3], 2.0, 0.0);
    // The trailing run [6,8) (two 50s -> 1 tie) closes ONLY via the post-loop RanksTies call,
    // which upstream never routes through the `ties[i - 1] = t` write. Reproduce the defect:
    // ties[7] (and every other untouched slot) stays 0, not 1.
    for (std::size_t i = 0; i < ties.size(); ++i) {
        if (i == 3) continue;
        CHECK_NEAR(ties[i], 0.0, 0.0);
    }
}

// The vector percentile(data, k) overload agrees element-for-element with seven scalar
// percentile(data, k) calls against the SAME unsorted data (each scalar call re-sorts
// internally; the vector overload sorts once and reuses the sorted copy).
void test_percentile_vector() {
    const std::vector<double> data{12.5, 3.2, 45.6, 7.8, 22.1, 9.9, 31.4, 18.0, 2.5, 40.0};
    const std::vector<double> ks{0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99};

    auto result = bfdata::percentile(data, ks);
    CHECK_EQ(result.size(), ks.size());
    for (std::size_t i = 0; i < ks.size(); ++i) {
        CHECK_NEAR(result[i], bfdata::percentile(data, ks[i]), 0.0);
    }
}

// Addendum (Task 1 review, folded into Task 2): the tie overload's `AlmostEquals` comparison is
// ABSOLUTE (`|a - b| <= kDoubleMachineEpsilon`, 1.11e-16), not a plain `==`. Near a magnitude
// like 20.0 the spacing between adjacent doubles is ~3.55e-15 -- wider than the tolerance -- so
// no representable pair there can ever distinguish "tied under tolerance" from "tied under exact
// equality"; every case in the test above happens to be bit-identical, so it cannot discriminate
// the two comparisons. 1e-17 and 2e-17 can: they are distinct representable doubles (so `==`
// would NOT tie them), but their difference, exactly 1e-17, is within the 1.11e-16 tolerance (so
// `AlmostEquals` DOES tie them). Verified below against the average-rank formula by hand.
void test_ranks_in_place_ties_tolerance_discriminates() {
    std::vector<double> data{1e-17, 2e-17, 100.0};
    std::vector<double> ties;
    auto ranks = bfdata::ranks_in_place(data, ties);

    // Under exact equality, 1e-17 != 2e-17, so no tie would form and ranks would be {1, 2, 3}.
    // Under the tolerance test, |1e-17 - 2e-17| = 1e-17 <= 1.11e-16, so they tie and share the
    // average rank (b + a - 1) / 2 + 1 = (2 + 0 - 1) / 2 + 1 = 1.5; the untied 100.0 gets rank 3.
    const std::vector<double> expected_ranks{1.5, 1.5, 3.0};
    CHECK_EQ(ranks.size(), expected_ranks.size());
    for (std::size_t i = 0; i < expected_ranks.size(); ++i) CHECK_NEAR(ranks[i], expected_ranks[i], 0.0);

    CHECK_EQ(ties.size(), data.size());
    CHECK_NEAR(ties[1], 1.0, 0.0);  // the 2-element tie run [0, 2) closes at index i - 1 = 1
    CHECK_NEAR(ties[0], 0.0, 0.0);
    CHECK_NEAR(ties[2], 0.0, 0.0);
}

}  // namespace

int main() {
    test_pow();
    test_mean_variance();
    test_ranks_in_place_with_ties();
    test_ranks_in_place_ties_tolerance_discriminates();
    test_percentile_vector();
    return chtest::summary("test_statistics_helpers");
}
