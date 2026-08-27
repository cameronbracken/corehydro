// P5 Task 2 -- JenksNaturalBreaks and JenksCluster.
//
// Transcribes all three [TestMethod]s of
// upstream/Numerics/Test_Numerics/Machine Learning/Unsupervised/Test_JenksNaturalBreaks.cs
// @ 2a0357a. The expected breaks are R BAMMtools output (upstream's stated reference), so these
// are correctness oracles against the literature, not merely C#-reproduction pins.
//
// The 7,889-value dataset lives in tests/data/jenks_dataset.hpp -- see that header for why it is
// here rather than in fixtures/.
//
// Everything below the three transcribed methods is a COREHYDRO SUPPLEMENT, clearly marked: the
// C# suite asserts only the break values, so it would pass on a port whose clusters, guards or
// goodness-of-fit were wrong.
#include <cmath>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/machine_learning/unsupervised/jenks_natural_breaks.hpp"
#include "data/jenks_dataset.hpp"

namespace ml = corehydro::numerics::machine_learning;
using corehydro::testdata::jenks::kDataset3;

namespace {

// --- Transcribed from Test_JenksNaturalBreaks.cs -----------------------------------------

void test_jenks_5_classes() {
    ml::JenksNaturalBreaks jenks(kDataset3, 5);
    const double true_values[] = {4.141846, 8.523254, 12.64075, 20.13635, 37.00143};
    CHECK_EQ(static_cast<int>(jenks.breaks().size()), 5);
    for (std::size_t i = 0; i < jenks.breaks().size(); i++)
        CHECK_NEAR(jenks.breaks()[i], true_values[i], 1e-5);
}

void test_jenks_7_classes() {
    ml::JenksNaturalBreaks jenks(kDataset3, 7);
    const double true_values[] = {2.769867,  6.317200,  8.810181, 11.378660,
                                  15.062380, 22.131710, 37.001430};
    CHECK_EQ(static_cast<int>(jenks.breaks().size()), 7);
    for (std::size_t i = 0; i < jenks.breaks().size(); i++)
        CHECK_NEAR(jenks.breaks()[i], true_values[i], 1e-5);
}

void test_jenks_9_classes() {
    ml::JenksNaturalBreaks jenks(kDataset3, 9);
    const double true_values[] = {2.426910,  5.687012,  7.817596,  9.682861, 11.828340,
                                  14.905240, 19.065770, 25.039920, 37.001430};
    CHECK_EQ(static_cast<int>(jenks.breaks().size()), 9);
    for (std::size_t i = 0; i < jenks.breaks().size(); i++)
        CHECK_NEAR(jenks.breaks()[i], true_values[i], 1e-5);
}

// --- COREHYDRO SUPPLEMENT (no C# counterpart) --------------------------------------------

void test_constructor_guards() {
    CHECK_THROWS_MSG(ml::JenksNaturalBreaks(std::vector<double>{}, 3), "data array is empty");
    CHECK_THROWS_MSG(ml::JenksNaturalBreaks(std::vector<double>{1, 2, 3}, 0),
                     "greater than zero");
    CHECK_THROWS_MSG(ml::JenksNaturalBreaks(std::vector<double>{1, 2, 3}, -1),
                     "greater than zero");
    CHECK_THROWS_MSG(ml::JenksNaturalBreaks(std::vector<double>{1, 2, 3}, 4),
                     "cannot be greater than the length");
}

void test_clusters_partition_the_data() {
    ml::JenksNaturalBreaks jenks(kDataset3, 5);
    const std::vector<ml::JenksCluster>& c = jenks.clusters();
    CHECK_EQ(static_cast<int>(c.size()), 5);
    CHECK_EQ(c.front().start_index(), 0);
    CHECK_EQ(c.back().end_index(), static_cast<int>(kDataset3.size()) - 1);
    int total = 0;
    for (std::size_t i = 0; i < c.size(); i++) {
        CHECK_TRUE(c[i].end_index() >= c[i].start_index());
        total += c[i].count();
        // Each cluster's break IS its maximum, and the maxima are strictly increasing.
        CHECK_EQ(jenks.breaks()[i], c[i].max_value());
        if (i > 0) {
            // No gap and no overlap.
            CHECK_EQ(c[i].start_index(), c[i - 1].end_index() + 1);
            CHECK_TRUE(c[i].min_value() >= c[i - 1].max_value());
        }
    }
    CHECK_EQ(total, static_cast<int>(kDataset3.size()));

    double gvf = jenks.goodness_of_variance_fit();
    CHECK_TRUE(gvf > 0.0 && gvf < 1.0);
    // More classes cannot fit worse.
    ml::JenksNaturalBreaks jenks9(kDataset3, 9);
    CHECK_TRUE(jenks9.goodness_of_variance_fit() >= gvf);
}

void test_sorting_and_presorted_input() {
    // Unsorted input is sorted internally; passing the same data pre-sorted gives the same fit.
    std::vector<double> raw = {9.0, 1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 0.0};
    std::vector<double> pre = raw;
    std::sort(pre.begin(), pre.end());

    ml::JenksNaturalBreaks a(raw, 3);
    ml::JenksNaturalBreaks b(pre, 3, true);
    CHECK_EQ(static_cast<int>(a.breaks().size()), 3);
    for (std::size_t i = 0; i < a.breaks().size(); i++) CHECK_EQ(a.breaks()[i], b.breaks()[i]);
    CHECK_EQ(a.sorted_data().front(), 0.0);
    CHECK_EQ(a.sorted_data().back(), 9.0);
    // The last break is always the sample maximum.
    CHECK_EQ(a.breaks().back(), 9.0);
}

void test_jenks_cluster_statistics() {
    // Hand-computed Welford statistics over a six-value run, checked against the closed forms.
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    ml::JenksCluster c(data, 0, 7);
    CHECK_EQ(c.count(), 8);
    CHECK_EQ(c.min_value(), 2.0);
    CHECK_EQ(c.max_value(), 9.0);
    CHECK_NEAR(c.sum(), 40.0, 1e-12);
    CHECK_NEAR(c.average(), 5.0, 1e-12);
    // sum((x - 5)^2) = 9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32
    CHECK_NEAR(c.sum_of_squared_deviations(), 32.0, 1e-12);
    // Sample (N-1) variance, NOT the population variance the name might suggest.
    CHECK_NEAR(c.variance(), 32.0 / 7.0, 1e-12);

    // A sub-range is inclusive on both ends.
    ml::JenksCluster mid(data, 4, 6);
    CHECK_EQ(mid.count(), 3);
    CHECK_EQ(mid.min_value(), 5.0);
    CHECK_EQ(mid.max_value(), 7.0);
    CHECK_NEAR(mid.sum(), 17.0, 1e-12);

    // A singleton cluster reports zero variance (the `Count <= 1` branch), not NaN.
    ml::JenksCluster one(data, 3, 3);
    CHECK_EQ(one.count(), 1);
    CHECK_EQ(one.variance(), 0.0);
    CHECK_EQ(one.sum_of_squared_deviations(), 0.0);

    CHECK_THROWS(ml::JenksCluster(data, 3, 2));
    CHECK_THROWS(ml::JenksCluster(data, 0, 8));
}

void test_degenerate_all_identical() {
    // UPSTREAM DEFECT, mirrored and pinned. With every value identical, the dynamic program's
    // `>=` update leaves the class limits at their smallest start index, the walk-back computes
    // kclass[0] = -1, and the first cluster is constructed over [0, -1]. MEASURED against the
    // real library: `new JenksNaturalBreaks(20 copies of 3.5, 3)` throws
    // IndexOutOfRangeException (C# reads data[-1]); the four constructor guards all pass first,
    // so the failure comes from inside the fit. The port range-checks in JenksCluster and throws
    // std::out_of_range instead of reading out of bounds -- same failure, defined behavior.
    // See docs/upstream-csharp-issues.md.
    std::vector<double> flat(20, 3.5);
    CHECK_THROWS_MSG(ml::JenksNaturalBreaks(flat, 3), "index range out of bounds");
    // Heavily tied data is NOT affected -- only the fully degenerate case is. The 7,889-value
    // dataset above contains long runs of exact zeros and fits at k = 5, 7 and 9.
    ml::JenksNaturalBreaks tied(std::vector<double>{0, 0, 0, 0, 0, 1, 1, 1, 9, 9}, 3);
    CHECK_EQ(static_cast<int>(tied.breaks().size()), 3);
    CHECK_EQ(tied.breaks().back(), 9.0);

    // One cluster over the whole sample is the trivial fit.
    ml::JenksNaturalBreaks single(std::vector<double>{5.0, 1.0, 3.0}, 1);
    CHECK_EQ(static_cast<int>(single.breaks().size()), 1);
    CHECK_EQ(single.breaks()[0], 5.0);
    CHECK_EQ(single.clusters()[0].count(), 3);
    CHECK_NEAR(single.goodness_of_variance_fit(), 0.0, 1e-15);
}

}  // namespace

int main() {
    test_jenks_5_classes();
    test_jenks_7_classes();
    test_jenks_9_classes();
    test_constructor_guards();
    test_clusters_partition_the_data();
    test_sorting_and_presorted_input();
    test_jenks_cluster_statistics();
    test_degenerate_all_identical();
    return chtest::summary("test_jenks_natural_breaks");
}
