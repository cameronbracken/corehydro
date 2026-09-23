// Transcribed from: upstream/Numerics/Numerics/Utilities/Tools.cs (Pow, lines 157-179) and
// upstream/Numerics/Numerics/Data/Statistics/Statistics.cs (MeanVariance line 262,
// Percentile(IList<double>, IList<double>, bool) line 573, RanksInPlace(double[], out double[])
// line 674) @ 2a0357a. P4 Task 1 -- the four small helpers HypothesisTests (P4 Task 2) needs and
// this port had deliberately omitted; see the header notes in tools.hpp / statistics.hpp for the
// full fidelity discussion. P4 Task 4 adds the Correlation.Pearson(double[,])/Spearman(double[,])
// matrix overload tests (upstream/Numerics/Numerics/Data/Statistics/Correlation.cs @ 2a0357a,
// lines 87/169); see correlation.hpp's own file header.
#include <limits>
#include <vector>

#include "corehydro/numerics/data/correlation.hpp"
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

// ranks_in_place(data, ties) records both interior and trailing tie runs.
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
    CHECK_NEAR(ties[7], 1.0, 0.0);
    for (std::size_t i = 0; i < ties.size(); ++i) {
        if (i == 3 || i == 7) continue;
        CHECK_NEAR(ties[i], 0.0, 0.0);
    }
}

void test_v220_statistics_corrections() {
    const std::vector<double> shifted{1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0, 1e12 + 4.0};
    const auto moments = bfdata::product_moments(shifted);
    CHECK_NEAR(moments[0], 1e12 + 2.5, 0.0);
    CHECK_NEAR(moments[1], std::sqrt(5.0 / 3.0), 1e-15);
    CHECK_THROWS(bfdata::percentile({1.0, 2.0}, std::numeric_limits<double>::quiet_NaN()));

    const std::vector<double> data{1.0, 2.0, 4.0};
    const std::vector<double> weights{0.2, 0.3, 0.5};
    CHECK_NEAR(bfdata::mean(data, weights), 2.8, 1e-15);
    CHECK_NEAR(bfdata::variance(data, weights, bfdata::WeightType::Reliability), 1.56 / 0.62,
               1e-14 * (1.56 / 0.62));

    const std::vector<double> percentile_data{2.0, 5.0, 7.0};
    const std::vector<double> percentile_weights{1.0, 3.0, 2.0};
    CHECK_NEAR(bfdata::percentile(percentile_data, 1.0 / 3.0, percentile_weights), 5.0, 0.0);
    CHECK_NEAR(bfdata::percentile(percentile_data, 0.5, percentile_weights), 5.5, 1e-14);
    CHECK_NEAR(corehydro::numerics::expm1(1e-8), 1.0000000050000000167e-8, 1e-24);
    CHECK_EQ(corehydro::numerics::expm1(-745.0), -1.0);

    for (int n : {1292, 1293, 1300}) {
        std::vector<double> linear_data(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            linear_data[static_cast<std::size_t>(i)] = 1.0 + 0.5 * i;
        }
        const auto linear = bfdata::linear_moments(linear_data);
        CHECK_NEAR(linear[2], 0.0, 1e-12);
        CHECK_NEAR(linear[3], 0.0, 1e-12);
    }

    std::vector<double> large(8192);
    for (std::size_t i = 0; i < large.size(); ++i) {
        large[i] = i % 3 == 0 ? 1e16 : (i % 3 == 1 ? 1.0 : -1e16);
    }
    std::vector<double> chunks(64, 0.0);
    for (std::size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        const std::size_t start = chunk * large.size() / chunks.size();
        const std::size_t end = (chunk + 1) * large.size() / chunks.size();
        for (std::size_t i = start; i < end; ++i) chunks[chunk] += large[i];
    }
    double expected_parallel_mean = 0.0;
    for (double chunk : chunks) expected_parallel_mean += chunk;
    expected_parallel_mean /= static_cast<double>(large.size());
    CHECK_EQ(bfdata::parallel_mean(large), expected_parallel_mean);

    const std::vector<double> jack_data{1.0, 2.0, 3.0, 4.0};
    const auto mean_callback = [](const std::vector<double>& values) { return bfdata::mean(values); };
    const auto jack_sample = bfdata::jackknife_sample(jack_data, mean_callback);
    CHECK_EQ(jack_sample, std::vector<double>({3.0, 8.0 / 3.0, 7.0 / 3.0, 2.0}));
    CHECK_NEAR(bfdata::jackknife_standard_error(jack_data, mean_callback),
               std::sqrt(5.0 / 12.0), 1e-15);
    CHECK_EQ(bfdata::jackknife_standard_error({42.0}, mean_callback), 0.0);
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

// pearson_matrix/spearman_matrix on the Test_Correlation.cs "big" dataset (upstream
// Test_Pearson_Big/Test_Spearman_Big), built as a 15-by-2 matrix (XArray column 0, YArray
// column 1). The off-diagonal must equal the already-pinned pairwise oracle at 1e-10
// (0.988054377242161 for Pearson, 1 for Spearman); the diagonal is exactly 1 (cov[j,j] is the
// SAME dx*dx accumulation as ss[j], so corr[j,j] = ss[j] / sqrt(ss[j] * ss[j]) round-trips
// exactly); and the matrix is symmetric bit-for-bit (cov[k,j] is a literal copy of cov[j,k], and
// sqrt(ss[j] * ss[k]) == sqrt(ss[k] * ss[j]) since IEEE-754 multiplication is commutative).
void test_pearson_matrix_big() {
    const std::vector<double> x_array{230408, 288010, 345611, 403213,  460815,  518417, 576019,
                                      633612, 691223, 748825, 806427,  864029,  921631, 1036834,
                                      1152038};
    const std::vector<double> y_array{1519.7, 1520.5, 1520.9, 1521.7, 1523.5, 1525.9, 1528.4,
                                      1530.9, 1533.2, 1534.7, 1535.9, 1538,   1541.3, 1547.7,
                                      1552.7};
    auto m = bfdata::pearson_matrix({x_array, y_array});
    CHECK_EQ(m.size(), std::size_t{2});
    CHECK_EQ(m[0].size(), std::size_t{2});
    CHECK_EQ(m[1].size(), std::size_t{2});

    CHECK_EQ(m[0][0], 1.0);
    CHECK_EQ(m[1][1], 1.0);
    CHECK_NEAR(m[0][1], 0.988054377242161, 1e-10);
    CHECK_EQ(m[0][1], m[1][0]);
}

void test_spearman_matrix_big() {
    const std::vector<double> x_array{230408, 288010, 345611, 403213,  460815,  518417, 576019,
                                      633612, 691223, 748825, 806427,  864029,  921631, 1036834,
                                      1152038};
    const std::vector<double> y_array{1519.7, 1520.5, 1520.9, 1521.7, 1523.5, 1525.9, 1528.4,
                                      1530.9, 1533.2, 1534.7, 1535.9, 1538,   1541.3, 1547.7,
                                      1552.7};
    auto m = bfdata::spearman_matrix({x_array, y_array});
    CHECK_EQ(m.size(), std::size_t{2});
    CHECK_EQ(m[0][0], 1.0);
    CHECK_EQ(m[1][1], 1.0);
    CHECK_NEAR(m[0][1], 1.0, 1e-10);
    CHECK_EQ(m[0][1], m[1][0]);
}

// A three-column case whose three off-diagonals each equal the corresponding pairwise call --
// an identity check against the already-oracle-pinned pairwise pearson()/spearman(), not a new
// oracle (mirrors test_mean_variance's identity-check style above).
void test_pearson_matrix_three_column() {
    const std::vector<double> c0{14, 8, 32, 7, 3, 15};
    const std::vector<double> c1{10, 5, 7, 4, 3, 8};
    const std::vector<double> c2{2, 9, 1, 6, 4, 11};

    auto m = bfdata::pearson_matrix({c0, c1, c2});
    CHECK_EQ(m.size(), std::size_t{3});
    for (const auto& row : m) CHECK_EQ(row.size(), std::size_t{3});

    CHECK_EQ(m[0][0], 1.0);
    CHECK_EQ(m[1][1], 1.0);
    CHECK_EQ(m[2][2], 1.0);
    CHECK_NEAR(m[0][1], bfdata::pearson(c0, c1), 0.0);
    CHECK_NEAR(m[0][2], bfdata::pearson(c0, c2), 0.0);
    CHECK_NEAR(m[1][2], bfdata::pearson(c1, c2), 0.0);
    CHECK_EQ(m[0][1], m[1][0]);
    CHECK_EQ(m[0][2], m[2][0]);
    CHECK_EQ(m[1][2], m[2][1]);
}

void test_spearman_matrix_three_column() {
    const std::vector<double> c0{14, 8, 32, 7, 3, 15};
    const std::vector<double> c1{10, 5, 7, 4, 3, 8};
    const std::vector<double> c2{2, 9, 1, 6, 4, 11};

    auto m = bfdata::spearman_matrix({c0, c1, c2});
    CHECK_EQ(m.size(), std::size_t{3});

    CHECK_EQ(m[0][0], 1.0);
    CHECK_EQ(m[1][1], 1.0);
    CHECK_EQ(m[2][2], 1.0);
    CHECK_NEAR(m[0][1], bfdata::spearman(c0, c1), 0.0);
    CHECK_NEAR(m[0][2], bfdata::spearman(c0, c2), 0.0);
    CHECK_NEAR(m[1][2], bfdata::spearman(c1, c2), 0.0);
    CHECK_EQ(m[0][1], m[1][0]);
    CHECK_EQ(m[0][2], m[2][0]);
    CHECK_EQ(m[1][2], m[2][1]);
}

// Upstream has no guard on an empty column list beyond the explicit "at least one column" throw
// (Correlation.Pearson(double[,])/Spearman(double[,]), both lines ~93/183).
void test_correlation_matrix_empty_columns_throws() {
    bool threw = false;
    try {
        bfdata::pearson_matrix({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_EQ(threw, true);

    threw = false;
    try {
        bfdata::spearman_matrix({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_EQ(threw, true);
}

}  // namespace

int main() {
    test_pow();
    test_mean_variance();
    test_ranks_in_place_with_ties();
    test_ranks_in_place_ties_tolerance_discriminates();
    test_percentile_vector();
    test_v220_statistics_corrections();
    test_pearson_matrix_big();
    test_spearman_matrix_big();
    test_pearson_matrix_three_column();
    test_spearman_matrix_three_column();
    test_correlation_matrix_empty_columns_throws();
    return chtest::summary("test_statistics_helpers");
}
