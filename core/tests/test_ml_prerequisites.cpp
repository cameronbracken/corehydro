// P5 Task 1 -- the machine-learning prerequisites.
//
// Covers the five `Statistics` members, `Tools.Standardize`, the four `Matrix` accessors and
// constructors, the shared .NET introsort (moved out of models/data_frame/data_frame_plotting.hpp
// into numerics/utilities/dotnet_sort.hpp), and the two LINQ-ordering helpers the ported ML
// classes depend on.
//
// Most assertions here are IDENTITIES against already-ported and already-oracle-pinned code
// (population_variance against variance, five_number_summary against percentile), not new oracle
// values -- upstream has no test for any of these members. The one exception is the introsort
// permutation, which is a REGRESSION PIN on a verbatim code move: the literal below was captured
// by running the pre-move `corehydro::models::detail::dotnet_list_sort` on this exact input, so
// it fails if the move was not verbatim. The behavior it pins is itself validated upstream by the
// Hirsch-Stedinger plotting-position fixtures, which the same code path serves.
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/machine_learning/support/linq_order.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/utilities/dotnet_sort.hpp"

namespace nd = corehydro::numerics::data;
namespace nu = corehydro::numerics::utilities;
namespace mls = corehydro::numerics::machine_learning::support;
using corehydro::numerics::math::linalg::Matrix;

namespace {

const std::vector<double> kSample = {12.4, 7.1, 19.8, 3.3, 15.0, 9.9, 22.6, 5.2, 11.1, 17.4};

void test_population_variance() {
    // Statistics.PopulationVariance is the N-normalizer twin of the already-ported N-1
    // `variance()`, so population_variance == variance * (n - 1) / n is an exact identity in
    // exact arithmetic and holds to rounding here.
    double n = static_cast<double>(kSample.size());
    CHECK_NEAR(nd::population_variance(kSample), nd::variance(kSample) * (n - 1.0) / n, 1e-12);
    CHECK_NEAR(nd::population_standard_deviation(kSample),
               std::sqrt(nd::population_variance(kSample)), 1e-15);

    // Empty sequence -> NaN (C# returns double.NaN for data.Count == 0).
    CHECK_TRUE(std::isnan(nd::population_variance(std::vector<double>{})));
    // A single value has zero population variance (the C# loop never runs).
    CHECK_EQ(nd::population_variance(std::vector<double>{4.0}), 0.0);
}

void test_parallel_mean() {
    // The port sums serially where C# uses AsParallel().Sum() -- see the header note. On any
    // input the serial result equals `mean()` exactly, which is what this pins.
    CHECK_EQ(nd::parallel_mean(kSample), nd::mean(kSample));
    CHECK_TRUE(std::isnan(nd::parallel_mean(std::vector<double>{})));
}

void test_five_number_summary() {
    std::vector<double> s = nd::five_number_summary(kSample);
    CHECK_EQ(static_cast<int>(s.size()), 5);
    CHECK_EQ(s[0], nd::minimum(kSample));
    CHECK_NEAR(s[1], nd::percentile(kSample, 0.25), 1e-15);
    CHECK_NEAR(s[2], nd::percentile(kSample, 0.50), 1e-15);
    CHECK_NEAR(s[3], nd::percentile(kSample, 0.75), 1e-15);
    CHECK_EQ(s[4], nd::maximum(kSample));
}

void test_entropy() {
    // Statistics.Entropy(data, pdf) = -sum(p * log(p)) over p = pdf(x), skipping p <= 0.
    std::vector<double> data = {0.25, 0.25, 0.5};
    std::function<double(double)> identity_pdf = [](double x) { return x; };
    double expected = -(0.25 * std::log(0.25) + 0.25 * std::log(0.25) + 0.5 * std::log(0.5));
    CHECK_NEAR(nd::entropy(data, identity_pdf), expected, 1e-15);

    // A zero density contributes nothing rather than -inf (the `if (p > 0)` guard).
    std::function<double(double)> zero_for_first = [](double x) { return x == 0.25 ? 0.0 : x; };
    CHECK_NEAR(nd::entropy(data, zero_for_first), -(0.5 * std::log(0.5)), 1e-15);
    CHECK_TRUE(std::isfinite(nd::entropy(data, zero_for_first)));
}

void test_standardize() {
    std::vector<double> z = nd::standardize(kSample);
    CHECK_EQ(static_cast<int>(z.size()), static_cast<int>(kSample.size()));
    CHECK_NEAR(nd::mean(z), 0.0, 1e-12);
    CHECK_NEAR(nd::standard_deviation(z), 1.0, 1e-12);

    // Degenerate spread returns all zeros, NOT NaNs (the `sd <= 0 || IsNaN(sd)` early return).
    std::vector<double> flat(10, 100.0);
    std::vector<double> zf = nd::standardize(flat);
    for (double v : zf) CHECK_EQ(v, 0.0);
}

void test_matrix_constructors_and_accessors() {
    // C# Matrix(List<double[]>): each entry is a COLUMN. Ported as a named factory because the
    // row-oriented C# ctor Matrix(double[,]) maps to the same C++ type -- see matrix.hpp.
    std::vector<std::vector<double>> columns = {
        {1.0, 2.0, 3.0, 4.0, 5.0}, {10.0, 20.0, 30.0, 40.0, 50.0}, {-1.0, -2.0, -3.0, -4.0, -5.0}};
    Matrix m = Matrix::from_columns(columns);
    CHECK_EQ(m.number_of_rows(), 5);
    CHECK_EQ(m.number_of_columns(), 3);
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 5; ++i)
            CHECK_EQ(m(i, j), columns[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)]);

    std::vector<double> r2 = m.row(2);
    CHECK_EQ(static_cast<int>(r2.size()), 3);
    CHECK_EQ(r2[0], 3.0);
    CHECK_EQ(r2[1], 30.0);
    CHECK_EQ(r2[2], -3.0);

    std::vector<double> c1 = m.column(1);
    CHECK_EQ(static_cast<int>(c1.size()), 5);
    CHECK_EQ(c1[0], 10.0);
    CHECK_EQ(c1[4], 50.0);

    // C# Matrix(double[]): an n-by-1 matrix.
    Matrix single(std::vector<double>{7.0, 8.0, 9.0});
    CHECK_EQ(single.number_of_rows(), 3);
    CHECK_EQ(single.number_of_columns(), 1);
    CHECK_EQ(single(1, 0), 8.0);

    // Ragged columns are rejected; an empty column list is rejected.
    std::vector<std::vector<double>> ragged = {{1.0, 2.0}, {3.0}};
    CHECK_THROWS(Matrix::from_columns(ragged));
    CHECK_THROWS(Matrix::from_columns(std::vector<std::vector<double>>{}));
    CHECK_THROWS(m.row(5));
    CHECK_THROWS(m.column(3));
}

struct SortItem {
    double value;
    int tag;
};

void test_dotnet_sort() {
    // 20 elements (above the 16-element insertion-sort threshold, so the unstable quicksort path
    // runs) with four tie runs. The tag order below is the REGRESSION PIN described in the file
    // header: it was captured from the pre-move code path.
    const std::vector<double> values = {5, 3, 5, 1, 3, 5, 2, 1, 4, 3,
                                        5, 2, 1, 4, 3, 2, 5, 1, 4, 2};
    std::vector<SortItem> items;
    for (int i = 0; i < 20; ++i)
        items.push_back({values[static_cast<std::size_t>(i)], i});

    nu::dotnet_list_sort(items, [](const SortItem& a, const SortItem& b) {
        return nu::compare_double(a.value, b.value);
    });

    const int expected_tags[20] = {17, 3, 12, 7, 19, 15, 6, 11, 14, 9,
                                   4,  1, 8,  13, 18, 10, 5, 2,  16, 0};
    for (int i = 0; i < 20; ++i) CHECK_EQ(items[static_cast<std::size_t>(i)].tag, expected_tags[i]);
    // Ordered by key regardless of the tie permutation.
    for (int i = 1; i < 20; ++i)
        CHECK_TRUE(items[static_cast<std::size_t>(i - 1)].value <=
                   items[static_cast<std::size_t>(i)].value);

    // C# Double.CompareTo: NaN sorts below every number and compares equal to itself.
    CHECK_EQ(nu::compare_double(std::numeric_limits<double>::quiet_NaN(), 1.0), -1);
    CHECK_EQ(nu::compare_double(1.0, std::numeric_limits<double>::quiet_NaN()), 1);
    CHECK_EQ(nu::compare_double(std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()),
             0);
}

void test_linq_order_helpers() {
    // LINQ Distinct() yields first-appearance order.
    std::vector<double> d = mls::distinct_in_first_appearance_order({3.0, 1.0, 3.0, 2.0, 1.0});
    CHECK_EQ(static_cast<int>(d.size()), 3);
    CHECK_EQ(d[0], 3.0);
    CHECK_EQ(d[1], 1.0);
    CHECK_EQ(d[2], 2.0);

    // GroupBy yields groups in first-appearance order and OrderByDescending is stable, so a tie
    // goes to whichever value appeared first. Here 2 and 1 both appear twice; 2 appeared first.
    CHECK_EQ(mls::most_common_first_appearance_wins({2.0, 1.0, 2.0, 1.0, 3.0}), 2.0);
    // Reversing the first two flips the winner, which is the whole point of the ordering rule.
    CHECK_EQ(mls::most_common_first_appearance_wins({1.0, 2.0, 2.0, 1.0, 3.0}), 1.0);
    // A clear majority wins regardless of position.
    CHECK_EQ(mls::most_common_first_appearance_wins({5.0, 7.0, 7.0, 7.0, 5.0}), 7.0);
    CHECK_EQ(mls::most_common_first_appearance_wins({4.0}), 4.0);
}

}  // namespace

int main() {
    test_population_variance();
    test_parallel_mean();
    test_five_number_summary();
    test_entropy();
    test_standardize();
    test_matrix_constructors_and_accessors();
    test_dotnet_sort();
    test_linq_order_helpers();
    return chtest::summary("test_ml_prerequisites");
}
