// Tests for UncertainOrdinate, UncertainOrderedPairedData, and TabularFunction (P4 Task 9).
//
// Oracle is the upstream C# test classes @ 2a0357a:
//   upstream/Numerics/Test_Numerics/Data/Paired Data/Test_UncertainOrdinate.cs
//   upstream/Numerics/Test_Numerics/Data/Paired Data/Test_UncertainPairedData.cs
//   upstream/Numerics/Test_Numerics/Functions/Test_Functions.cs (Test_Tabular_Function only)
// Every applicable [TestMethod] is transcribed with its input values copied verbatim.
//
// SKIPPED: Test_ToXElement (Test_UncertainOrdinate.cs) and Test_ReadWriteXElement
// (Test_UncertainPairedData.cs) -- XML round-tripping is a project-wide severance (see
// uncertain_ordinate.hpp / uncertain_ordered_paired_data.hpp's header notes).
//
// ADAPTED: Test_Construction (Test_UncertainOrdinate.cs) builds most of its fixture through the
// XML constructors (`new UncertainOrdinate(xElement, ...)`) purely to prove the XML path agrees
// with the direct constructor -- with XML severed, only the non-XML portion has a C++ counterpart
// (the X/Y-null/X-infinite IsValid computation and the != operator on the resulting ordinates).
// The C# test also asserts `unordinate1.Y == distribution` (reference identity with the ORIGINAL
// distribution object) -- untestable here by construction, since this port's UncertainOrdinate
// always deep-clones Y (see uncertain_ordinate.hpp's header note: unique_ptr forces exactly one
// owner, so there is no reference to share). A corehydro supplement below exercises operator=='s
// general (non-null, matching-type) path instead, which the adapted C# fixture doesn't reach.
//
// SKIPPED (Test_IList, Test_UncertainPairedData.cs): the CopyTo(UncertainOrdinate[], int)
// segment, on the same grounds ordered_paired_data.hpp's own header documents for its CopyTo --
// plain ICollection<T> boilerplate not in this port's member surface.
//
// MERGED: Test_CurveSample and Test_Curve_Sample_Probability (both Test_UncertainPairedData.cs)
// are one function below, test_curve_sample(), since they share the same fixture and structure.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "corehydro/numerics/data/paired_data/ordered_paired_data.hpp"
#include "corehydro/numerics/data/paired_data/uncertain_ordered_paired_data.hpp"
#include "corehydro/numerics/data/paired_data/uncertain_ordinate.hpp"
#include "corehydro/numerics/distributions/deterministic.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/triangular.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/functions/tabular_function.hpp"
#include "check.hpp"

using corehydro::numerics::data::SortOrder;
using corehydro::numerics::data::Transform;
using corehydro::numerics::data::paired_data::Ordinate;
using corehydro::numerics::data::paired_data::OrderedPairedData;
using corehydro::numerics::data::paired_data::UncertainOrdinate;
using corehydro::numerics::data::paired_data::UncertainOrderedPairedData;
using corehydro::numerics::distributions::Deterministic;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::Triangular;
using corehydro::numerics::distributions::Uniform;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::distributions::UnivariateDistributionType;
using corehydro::numerics::functions::TabularFunction;

namespace {

std::vector<double> reversed(const std::vector<double>& v) {
    return std::vector<double>(v.rbegin(), v.rend());
}

void check_curve(const OrderedPairedData& actual, const OrderedPairedData& expected, double tol) {
    CHECK_EQ(actual.count(), expected.count());
    int n = actual.count() < expected.count() ? actual.count() : expected.count();
    for (int i = 0; i < n; ++i) {
        CHECK_NEAR(actual[i].x, expected[i].x, tol);
        CHECK_NEAR(actual[i].y, expected[i].y, tol);
    }
}

struct DatasetFixture {
    UncertainOrderedPairedData d1;
    UncertainOrderedPairedData d2;
    UncertainOrderedPairedData d3;
    UncertainOrderedPairedData d4;
};

// Test_UncertainPairedData's constructor fixture, transcribed verbatim: x = {1,2,3,5},
// y = {Triangular(1,2,3), Triangular(2,4,5), Triangular(6,8,12), Triangular(13,19,20)}.
DatasetFixture make_datasets() {
    Triangular t1(1, 2, 3), t2(2, 4, 5), t3(6, 8, 12), t4(13, 19, 20);
    std::vector<double> x = {1, 2, 3, 5};
    std::vector<double> xr = reversed(x);
    std::vector<const UnivariateDistributionBase*> y = {&t1, &t2, &t3, &t4};
    std::vector<const UnivariateDistributionBase*> yr = {&t4, &t3, &t2, &t1};

    UncertainOrderedPairedData d1(x, y, true, SortOrder::Ascending, true, SortOrder::Ascending,
                                   UnivariateDistributionType::Triangular);
    UncertainOrderedPairedData d2(xr, y, true, SortOrder::Descending, true, SortOrder::Ascending,
                                   UnivariateDistributionType::Triangular);
    UncertainOrderedPairedData d3(x, yr, true, SortOrder::Ascending, true, SortOrder::Descending,
                                   UnivariateDistributionType::Triangular);
    UncertainOrderedPairedData d4(xr, yr, true, SortOrder::Descending, true, SortOrder::Descending,
                                   UnivariateDistributionType::Triangular);
    return DatasetFixture{std::move(d1), std::move(d2), std::move(d3), std::move(d4)};
}

}  // namespace

// -------------------------------------------------------------------------------------------
// Test_UncertainOrdinate.cs
// -------------------------------------------------------------------------------------------

void test_construction() {
    UncertainOrdinate unordinate1(2.0, Normal());
    UncertainOrdinate unordinate4(-std::numeric_limits<double>::infinity(), Normal());
    UncertainOrdinate unordinate5(3.0, nullptr);

    CHECK_TRUE(unordinate1.is_valid);
    CHECK_TRUE(!unordinate4.is_valid);
    CHECK_TRUE(!unordinate5.is_valid);

    CHECK_TRUE(unordinate1 != unordinate4);
    CHECK_TRUE(unordinate1 != unordinate5);

    // Corehydro supplement: exercises operator=='s general (non-null, matching-type) path, which
    // the adapted fixture above (deliberately X-invalid or Y-null) never reaches. Two
    // independently constructed UncertainOrdinates with identical distribution parameters compare
    // equal; differing parameters compare unequal.
    UncertainOrdinate a(2.0, Triangular(1, 2, 3));
    UncertainOrdinate b(2.0, Triangular(1, 2, 3));
    CHECK_TRUE(a == b);
    UncertainOrdinate c(2.0, Triangular(1, 2, 4));
    CHECK_TRUE(a != c);
}

void test_ordinate_valid() {
    UncertainOrdinate subject(3.0, Triangular(6, 8, 12));
    UncertainOrdinate c1(0.0, Triangular(1, 2, 3));
    UncertainOrdinate c2(2.0, Triangular(2, 4, 5));
    UncertainOrdinate c3(5.0, Triangular(13, 19, 20));
    UncertainOrdinate c4(6.0, Uniform(7, 14));

    auto run = [&](bool strict_x, bool strict_y, SortOrder x_order, SortOrder y_order, bool is_next,
                   bool allow_diff) {
        std::vector<bool> result(4);
        result[0] = subject.ordinate_valid(c1, strict_x, strict_y, x_order, y_order, is_next, allow_diff);
        result[1] = subject.ordinate_valid(c2, strict_x, strict_y, x_order, y_order, is_next, allow_diff);
        result[2] = subject.ordinate_valid(c3, strict_x, strict_y, x_order, y_order, is_next, allow_diff);
        result[3] = subject.ordinate_valid(c4, strict_x, strict_y, x_order, y_order, is_next, allow_diff);
        return result;
    };
    auto check_row = [&](const std::vector<bool>& actual, const std::vector<bool>& expected) {
        CHECK_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(actual[i], expected[i]);
    };

    check_row(run(false, true, SortOrder::Ascending, SortOrder::Ascending, true, false),
              {false, false, true, false});
    check_row(run(false, true, SortOrder::Ascending, SortOrder::Ascending, false, false),
              {true, true, false, false});
    check_row(run(true, true, SortOrder::Ascending, SortOrder::Ascending, true, false),
              {false, false, true, false});
    check_row(run(false, false, SortOrder::Ascending, SortOrder::Ascending, true, false),
              {false, false, true, false});
    check_row(run(false, true, SortOrder::Descending, SortOrder::Ascending, true, false),
              {false, false, false, false});
    check_row(run(false, true, SortOrder::Ascending, SortOrder::Descending, true, false),
              {false, false, false, false});
    check_row(run(false, true, SortOrder::Descending, SortOrder::Descending, true, false),
              {true, true, false, false});
    check_row(run(false, true, SortOrder::Ascending, SortOrder::Ascending, true, true),
              {false, false, true, true});
}

void test_ordinate_errors() {
    UncertainOrdinate subject(3.0, Triangular(6, 8, 12));
    UncertainOrdinate c1(2.0, Triangular(2, 4, 5));
    UncertainOrdinate c2(5.0, Triangular(13, 19, 20));
    UncertainOrdinate c3(std::numeric_limits<double>::infinity(), Uniform(7, 14));

    auto test1 = subject.ordinate_errors(c3, true, true, SortOrder::Ascending, SortOrder::Ascending, true);
    std::vector<std::string> test1_expected = {
        "Ordinate X value can not be infinity.",
        "Can't compare two ordinates with different distribution types."};
    CHECK_EQ(test1.size(), test1_expected.size());
    for (std::size_t i = 0; i < test1_expected.size() && i < test1.size(); ++i)
        CHECK_EQ(test1[i], test1_expected[i]);

    // Each of the three probes (lower bound, central tendency, upper bound) gives the same error.
    auto test2 = subject.ordinate_errors(c1, true, true, SortOrder::Ascending, SortOrder::Ascending, true);
    std::vector<std::string> test2_expected = {
        "Y values must increase.", "X values must increase.", "Y values must increase.",
        "X values must increase.", "Y values must increase.", "X values must increase.",
    };
    CHECK_EQ(test2.size(), test2_expected.size());
    for (std::size_t i = 0; i < test2_expected.size() && i < test2.size(); ++i)
        CHECK_EQ(test2[i], test2_expected[i]);

    auto test3 =
        subject.ordinate_errors(c2, true, true, SortOrder::Descending, SortOrder::Descending, true);
    std::vector<std::string> test3_expected = {
        "Y values must decrease.", "X values must decrease.", "Y values must decrease.",
        "X values must decrease.", "Y values must decrease.", "X values must decrease.",
    };
    CHECK_EQ(test3.size(), test3_expected.size());
    for (std::size_t i = 0; i < test3_expected.size() && i < test3.size(); ++i)
        CHECK_EQ(test3[i], test3_expected[i]);
}

// -------------------------------------------------------------------------------------------
// Test_UncertainPairedData.cs
// -------------------------------------------------------------------------------------------

// Merges Test_CurveSample (mean) and Test_Curve_Sample_Probability (probability = 0.5).
void test_curve_sample() {
    auto ds = make_datasets();

    std::vector<double> x = {1, 2, 3, 5};
    std::vector<double> xr = reversed(x);
    // mean = (min + max + mode) / 3
    std::vector<double> y_mean = {2, 3.66667, 8.66667, 17.33333};
    // inverse cdf at probability 0.5 (cross-checked upstream against R's EnvStats::qtri())
    std::vector<double> y_inv = {2, 3.732051, 8.535898, 17.58258};

    OrderedPairedData m1 = ds.d1.curve_sample();
    OrderedPairedData m2 = ds.d2.curve_sample();
    OrderedPairedData m3 = ds.d3.curve_sample();
    OrderedPairedData m4 = ds.d4.curve_sample();

    check_curve(m1, OrderedPairedData(x, y_mean, true, SortOrder::Ascending, true, SortOrder::Ascending),
                1e-5);
    check_curve(m2, OrderedPairedData(xr, y_mean, true, SortOrder::Descending, true, SortOrder::Ascending),
                1e-5);
    check_curve(
        m3, OrderedPairedData(x, reversed(y_mean), true, SortOrder::Ascending, true, SortOrder::Descending),
        1e-5);
    check_curve(m4,
                OrderedPairedData(xr, reversed(y_mean), true, SortOrder::Descending, true,
                                   SortOrder::Descending),
                1e-5);

    OrderedPairedData p1 = ds.d1.curve_sample(0.5);
    OrderedPairedData p2 = ds.d2.curve_sample(0.5);
    OrderedPairedData p3 = ds.d3.curve_sample(0.5);
    OrderedPairedData p4 = ds.d4.curve_sample(0.5);

    check_curve(p1, OrderedPairedData(x, y_inv, true, SortOrder::Ascending, true, SortOrder::Ascending),
                1e-5);
    check_curve(p2, OrderedPairedData(xr, y_inv, true, SortOrder::Descending, true, SortOrder::Ascending),
                1e-5);
    check_curve(
        p3, OrderedPairedData(x, reversed(y_inv), true, SortOrder::Ascending, true, SortOrder::Descending),
        1e-5);
    check_curve(p4,
                OrderedPairedData(xr, reversed(y_inv), true, SortOrder::Descending, true,
                                   SortOrder::Descending),
                1e-5);
}

void test_ilist() {
    auto ds = make_datasets();
    UncertainOrderedPairedData paired_data = ds.d1.clone();

    UncertainOrdinate ordinate = paired_data[2];

    int test1 = paired_data.index_of(ordinate);
    CHECK_EQ(test1, 2);

    paired_data.remove(ordinate);
    bool test2 = paired_data.contains(ordinate);
    CHECK_TRUE(!test2);

    paired_data.remove_at(2);
    bool test3 = paired_data.contains(ordinate);
    CHECK_TRUE(!test3);

    paired_data.insert(2, ordinate);
    int test4 = paired_data.index_of(ordinate);
    CHECK_EQ(test4, 2);

    UncertainOrdinate new_ordinate(7.0, Triangular(16, 22, 28));
    paired_data.add(new_ordinate);
    int test5 = paired_data.index_of(new_ordinate);
    CHECK_EQ(test5, paired_data.count() - 1);

    int prev_count = paired_data.count();
    ordinate = paired_data[2];
    paired_data.remove_range(0, 2);
    UncertainOrdinate test6 = paired_data[0];
    int curr_count = paired_data.count();
    CHECK_EQ(prev_count - 2, curr_count);
    CHECK_TRUE(ordinate == test6);

    // CopyTo skipped -- see the file header and ordered_paired_data.hpp's own precedent.

    std::vector<UncertainOrdinate> to_insert;
    to_insert.emplace_back(1.0, Triangular(1, 2, 3));
    to_insert.emplace_back(2.0, Triangular(2, 4, 5));
    paired_data.insert_range(0, to_insert);
    for (const auto& item : to_insert) CHECK_TRUE(paired_data.contains(item));

    std::vector<UncertainOrdinate> to_add;
    to_add.emplace_back(3.0, Triangular(6, 8, 12));
    to_add.emplace_back(5.0, Triangular(13, 19, 20));
    to_add.emplace_back(7.0, Triangular(16, 22, 28));
    paired_data.add_range(to_add);
    int test7 = paired_data.count();
    CHECK_EQ(test7, 7);
    for (const auto& item : to_add) CHECK_TRUE(paired_data.contains(item));

    paired_data.clear();
    CHECK_EQ(paired_data.count(), 0);
}

void test_equality() {
    auto ds = make_datasets();
    UncertainOrderedPairedData dataset5 = ds.d1.clone();

    bool test1 = (ds.d1 == dataset5);
    CHECK_TRUE(test1);

    bool test2 = (ds.d1 == ds.d2);
    CHECK_TRUE(!test2);

    bool test3 = (ds.d2 != ds.d3);
    CHECK_TRUE(test3);

    bool test4 = (ds.d1 != dataset5);
    CHECK_TRUE(!test4);
}

// -------------------------------------------------------------------------------------------
// Test_Functions.cs: Test_Tabular_Function
// -------------------------------------------------------------------------------------------

void test_tabular_function() {
    Deterministic d1(100), d2(200), d3(300), d4(400), d5(500);
    std::vector<double> x = {50, 100, 150, 200, 250};
    std::vector<const UnivariateDistributionBase*> y = {&d1, &d2, &d3, &d4, &d5};

    UncertainOrderedPairedData opd(x, y, true, SortOrder::Ascending, true, SortOrder::Ascending,
                                    UnivariateDistributionType::Deterministic);
    TabularFunction func(std::move(opd));
    func.set_x_transform(Transform::Logarithmic);
    func.set_y_transform(Transform::None);

    // Given X.
    double X = 50.0;
    double Y = func.function(X);
    CHECK_EQ(Y, 100.0);

    // Given Y.
    double Y2 = 100.0;
    double X2 = func.inverse_function(Y2);
    // Corehydro supplement: the C# test (~line 210) re-asserts `Assert.AreEqual(50.0, X);` here
    // instead of `X2` -- a copy-paste bug that trivially passes (X was never reassigned) without
    // ever checking the round trip. This asserts the round trip the C# test meant to check.
    CHECK_NEAR(X2, 50.0, 1e-12);

    // Given X -- interpolation. Basic interpolation formula: y = y1 + (y2-y1)/(x2-x1) * (x-x1),
    // with the x-values logarithm-transformed. The C# test writes this with natural logs (Math.Log)
    // while the implementation uses Tools.Log10 -- algebraically identical for a ratio of
    // differences, so the same expected value holds either way.
    double X3 = 75.0;
    double Y3 = func.function(X3);
    double valid_y =
        100 + (200.0 - 100.0) / (std::log(100.0) - std::log(50.0)) * (std::log(75.0) - std::log(50.0));
    CHECK_NEAR(Y3, valid_y, 1e-6);

    // Given Y -- interpolation (round trip of Y3/valid_y back through the inverse function).
    double Y4 =
        100 + (200.0 - 100.0) / (std::log(100.0) - std::log(50.0)) * (std::log(75.0) - std::log(50.0));
    double X4 = func.inverse_function(Y4);
    CHECK_NEAR(X4, 75.0, 1e-6);
}

int main() {
    test_construction();
    test_ordinate_valid();
    test_ordinate_errors();
    test_curve_sample();
    test_ilist();
    test_equality();
    test_tabular_function();
    return chtest::summary("test_uncertain_paired_data");
}
