// Standalone tests for the CubicSpline and Polynomial interpolation classes.
//
// Transcribed 1:1 (structure, inputs and oracle values unaltered) from
//   upstream/Numerics/Test_Numerics/Data/Interpolation/Test_CubicSpline.cs @ 2a0357a
//   upstream/Numerics/Test_Numerics/Data/Interpolation/Test_Polynomial.cs @ 2a0357a
// -- all 12 [TestMethod]s (6 + 6) in C# file order, matching the style of the other
// standalone ctest suites in this directory (see test_root_finding_extras.cpp).
#include <cmath>
#include <vector>

#include "corehydro/numerics/data/interpolation/cubic_spline.hpp"
#include "corehydro/numerics/data/interpolation/polynomial.hpp"
#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "check.hpp"

using corehydro::numerics::data::CubicSpline;
using corehydro::numerics::data::Polynomial;
using corehydro::numerics::data::SortOrder;

namespace {

// ---- Test_CubicSpline.cs -----------------------------------------------------------------

void test_cubic_spline_sequential() {
    std::vector<double> values(1000);
    for (int i = 1; i <= 1000; ++i) values[static_cast<std::size_t>(i - 1)] = i;
    CubicSpline spline(values, values);
    int lo = spline.sequential_search(872.5);
    CHECK_EQ(lo, 871);

    std::vector<double> reversed(values.rbegin(), values.rend());
    CubicSpline spline2(reversed, reversed, SortOrder::Descending);
    lo = spline2.sequential_search(872.5);
    CHECK_EQ(lo, 127);
}

void test_cubic_spline_bisection() {
    std::vector<double> values(1000);
    for (int i = 1; i <= 1000; ++i) values[static_cast<std::size_t>(i - 1)] = i;
    CubicSpline spline(values, values);
    int lo = spline.bisection_search(872.5);
    CHECK_EQ(lo, 871);

    std::vector<double> reversed(values.rbegin(), values.rend());
    CubicSpline spline2(reversed, reversed, SortOrder::Descending);
    lo = spline2.bisection_search(872.5);
    CHECK_EQ(lo, 127);
}

void test_cubic_spline_hunt() {
    std::vector<double> values(1000);
    for (int i = 1; i <= 1000; ++i) values[static_cast<std::size_t>(i - 1)] = i;
    CubicSpline spline(values, values);
    int lo = spline.hunt_search(872.5);
    CHECK_EQ(lo, 871);

    std::vector<double> reversed(values.rbegin(), values.rend());
    CubicSpline spline2(reversed, reversed, SortOrder::Descending);
    lo = spline2.hunt_search(872.5);
    CHECK_EQ(lo, 127);
}

// Test the interpolate function with one value.
void test_cubic_spline_interpolate() {
    std::vector<double> x{6, 24, 48, 72};
    std::vector<double> y{9.96, 22.13, 32.27, 37.60};
    CubicSpline spline(x, y);
    double X = 8;
    double Y = spline.interpolate(X);
    CHECK_NEAR(Y, 11.4049889205445, 1e-6);
}

// Test the interpolate function with a list of values.
void test_cubic_spline_interpolate_r() {
    std::vector<double> x{6, 24, 48, 72};
    std::vector<double> y{9.96, 22.13, 32.27, 37.60};
    CubicSpline spline(x, y);

    std::vector<double> xout{8, 20, 30, 56};
    std::vector<double> true_yout{11.40499, 19.68530, 25.35189, 34.35289};
    for (std::size_t i = 0; i < xout.size(); ++i) {
        double Y = spline.interpolate(xout[i]);
        CHECK_NEAR(Y, true_yout[i], 1e-4);
    }
}

// Test the interpolate function with another list of values (R-validated, 34 elements).
void test_cubic_spline_interpolate_list() {
    std::vector<double> x{6, 24, 48, 72};
    std::vector<double> y{9.96, 22.13, 32.27, 37.60};
    CubicSpline spline(x, y);
    std::vector<double> X(34);
    X[0] = 6;
    for (std::size_t i = 1; i < X.size(); ++i) X[i] = X[i - 1] + 2;

    std::vector<double> Y = spline.interpolate(X);
    std::vector<double> true_Y{
        9.96,
        11.4049889205445,
        12.8430203387148,
        14.2671367521368,
        15.6703806584362,
        17.045794555239,
        18.3864209401709,
        19.6853023108579,
        20.9354811649256,
        22.13,
        23.2634521159782,
        24.3366340218424,
        25.3518930288462,
        26.3115764482431,
        27.2180315912868,
        28.0736057692308,
        28.8806462933286,
        29.6415004748338,
        30.358515625,
        31.0340390550807,
        31.6704180763295,
        32.27,
        32.8352193880579,
        33.3688598053181,
        33.8737920673077,
        34.3528869895537,
        34.8090153875831,
        35.2450480769231,
        35.6638558731007,
        36.0683095916429,
        36.4612800480769,
        36.8456380579297,
        37.2242544367284,
        37.6,
    };
    for (std::size_t i = 0; i < X.size(); ++i) CHECK_NEAR(Y[i], true_Y[i], 1e-6);
}

// ---- Test_Polynomial.cs ------------------------------------------------------------------

void test_polynomial_sequential() {
    std::vector<double> values(1000);
    for (int i = 1; i <= 1000; ++i) values[static_cast<std::size_t>(i - 1)] = i;
    Polynomial poly(3, values, values);
    int lo = poly.sequential_search(872.5);
    CHECK_EQ(lo, 871);

    std::vector<double> reversed(values.rbegin(), values.rend());
    Polynomial poly2(3, reversed, reversed, SortOrder::Descending);
    lo = poly2.sequential_search(872.5);
    CHECK_EQ(lo, 127);
}

void test_polynomial_bisection() {
    std::vector<double> values(1000);
    for (int i = 1; i <= 1000; ++i) values[static_cast<std::size_t>(i - 1)] = i;
    Polynomial poly(3, values, values);
    int lo = poly.bisection_search(872.5);
    CHECK_EQ(lo, 871);

    std::vector<double> reversed(values.rbegin(), values.rend());
    Polynomial poly2(3, reversed, reversed, SortOrder::Descending);
    lo = poly2.bisection_search(872.5);
    CHECK_EQ(lo, 127);
}

void test_polynomial_hunt() {
    std::vector<double> values(1000);
    for (int i = 1; i <= 1000; ++i) values[static_cast<std::size_t>(i - 1)] = i;
    Polynomial poly(3, values, values);
    int lo = poly.hunt_search(872.5);
    CHECK_EQ(lo, 871);

    std::vector<double> reversed(values.rbegin(), values.rend());
    Polynomial poly2(3, reversed, reversed, SortOrder::Descending);
    lo = poly2.hunt_search(872.5);
    CHECK_EQ(lo, 127);
}

// Test the polynomial class and interpolation function with an order 3 polynomial.
void test_polynomial_interpolate_order3() {
    std::vector<double> x{6, 24, 48, 72};
    std::vector<double> y{9.96, 22.13, 32.27, 37.60};
    Polynomial poly(3, x, y);
    double X = 8;
    double Y = poly.interpolate(X);
    CHECK_NEAR(Y, 11.5415808882467, 1e-6);
}

// Test with an order 3 polynomial interpolation with a list of values.
void test_polynomial_interpolate_order3_r() {
    std::vector<double> x{6, 24, 48, 72};
    std::vector<double> y{9.96, 22.13, 32.27, 37.60};
    Polynomial poly(3, x, y);

    std::vector<double> xout{8, 20, 30, 56};
    std::vector<double> true_yout{11.54158, 19.80796, 25.24398, 34.46549};
    for (std::size_t i = 0; i < xout.size(); ++i) {
        double Y = poly.interpolate(xout[i]);
        CHECK_NEAR(Y, true_yout[i], 1e-5);
    }
}

// Test with an order 3 polynomial interpolation with a large list of values. The C# true_Y
// literal has 40 entries but the C# test loop only checks the first X.Length (34) of them --
// transcribed verbatim including the unused tail, matching C# exactly.
void test_polynomial_interpolate_list() {
    std::vector<double> x{6, 24, 48, 72};
    std::vector<double> y{9.96, 22.13, 32.27, 37.60};
    Polynomial poly(3, x, y);
    std::vector<double> X(34);
    X[0] = 6;
    for (std::size_t i = 1; i < X.size(); ++i) X[i] = X[i - 1] + 2;

    std::vector<double> Y = poly.interpolate(X);
    std::vector<double> true_Y{
        9.95999999999906,  11.5415808882467,  13.0626606341182,  14.5245941558435,
        15.9287363716526,  17.2764421997752,  18.5690665584413,  19.807964365881,
        20.994490540324,   22.1300000000003,  23.2158476631398,  24.2533884479724,
        25.2439772727281,  26.1889690556368,  27.0897187149283,  27.9475811688326,
        28.7639113355797,  29.5400641333994,  30.2773944805217,  30.9772572951765,
        31.6410074955936,  32.2700000000031,  32.8655897266348,  33.4291315937186,
        33.9619805194845,  34.4654914221624,  34.9410192199823,  35.3899188311739,
        35.8135451739673,  36.2132531665923,  36.5903977272789,  36.9463337742571,
        37.2824162257566,  37.6000000000075,  37.9004400152396,  38.1850911896829,
        38.4553084415673,  38.7124466891227,  38.9578608505791,  39.1929058441663,
    };
    for (std::size_t i = 0; i < X.size(); ++i) CHECK_NEAR(Y[i], true_Y[i], 1e-6);
}

}  // namespace

int main() {
    test_cubic_spline_sequential();
    test_cubic_spline_bisection();
    test_cubic_spline_hunt();
    test_cubic_spline_interpolate();
    test_cubic_spline_interpolate_r();
    test_cubic_spline_interpolate_list();

    test_polynomial_sequential();
    test_polynomial_bisection();
    test_polynomial_hunt();
    test_polynomial_interpolate_order3();
    test_polynomial_interpolate_order3_r();
    test_polynomial_interpolate_list();

    return chtest::summary("interpolation_extras");
}
