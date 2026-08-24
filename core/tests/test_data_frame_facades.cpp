// Tests for DataFrame's hypothesis-test and summary-statistics facades (P4 Task 5), the
// un-gating of the two `Deliberately NOT ported` regions data_frame.hpp used to carry.
//
// Oracle is the upstream C# test classes @ c2e6192:
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/DataFrame/ExactDataHypothesisTests.cs
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/DataFrame/NonparametricEmpiricalTests.cs
// Every applicable [TestMethod] is transcribed with its input arrays copied verbatim.
// Test_UnimodalityTest is SKIPPED (UnimodalityTest is a locked P5 scope decision -- see
// data_frame.hpp's header). Of NonparametricEmpiricalTests.cs, the two duplicate-value
// GetNonparametricMoments*/ROS cases are already covered by test_nonparametric_empirical.cpp
// (ported in B9); only the two summary-statistics/standardized-value cases are new here.
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::models::DataFrame;
using corehydro::models::ExactSeries;

namespace {

double find_value(const std::vector<std::pair<std::string, double>>& summary,
                   const std::string& label) {
    for (const auto& kv : summary) {
        if (kv.first == label) return kv.second;
    }
    CHECK_TRUE(false && "label not found in summary statistics result");
    return std::numeric_limits<double>::quiet_NaN();
}

// C# Test_EqualVarianceTtest.
void test_equal_variance_t_test() {
    std::vector<double> data1 = {
        8.782932,    10.64199,    -1.63955,    -6.802458,   9.088312,   -27.26934,
        9.451478,    -4.142762,   -4.262396,   -13.78983,   -1.743717,  27.259681,
        5.559418,    7.803247,    -11.25798,   12.253498,   -13.295363, -4.973664,
        16.81069,    4.480855,    11.694329,   21.836776,   -9.664926,  -23.297061,
        -23.965643,  27.076463,   -7.22471,    9.305697,    9.181852,   -2.434665};
    std::vector<double> data2 = {
        34.3561954,  75.9050064,  71.4757101,  58.9733692,  17.6281358, 24.7356484,
        0.2774026,   -39.8615073, 63.0320155,  10.7740315,  43.855325,  -61.4107418,
        -21.8079666, 38.1142162,  35.6335516,  53.8218821,  -32.3929633, 27.0220976,
        27.4956296,  -29.203965,  -6.2115822,  68.4307799,  11.6077081, 20.5498852,
        -10.1292962, 18.3386108,  30.7351382,  34.7138599,  74.3519506, -60.4083194};
    std::vector<double> data = data1;
    data.insert(data.end(), data2.begin(), data2.end());

    DataFrame df;
    df.set_exact_series(ExactSeries(data));

    double p = df.equal_variance_t_test(static_cast<int>(data1.size()));
    CHECK_NEAR(p, 0.0185, 1e-4);
}

// C# Test_UnequalVarianceTtest.
void test_unequal_variance_t_test() {
    std::vector<double> data1 = {
        8.782932,    10.64199,    -1.63955,    -6.802458,   9.088312,   -27.26934,
        9.451478,    -4.142762,   -4.262396,   -13.78983,   -1.743717,  27.259681,
        5.559418,    7.803247,    -11.25798,   12.253498,   -13.295363, -4.973664,
        16.81069,    4.480855,    11.694329,   21.836776,   -9.664926,  -23.297061,
        -23.965643,  27.076463,   -7.22471,    9.305697,    9.181852,   -2.434665};
    std::vector<double> data2 = {
        34.3561954,  75.9050064,  71.4757101,  58.9733692,  17.6281358, 24.7356484,
        0.2774026,   -39.8615073, 63.0320155,  10.7740315,  43.855325,  -61.4107418,
        -21.8079666, 38.1142162,  35.6335516,  53.8218821,  -32.3929633, 27.0220976,
        27.4956296,  -29.203965,  -6.2115822,  68.4307799,  11.6077081, 20.5498852,
        -10.1292962, 18.3386108,  30.7351382,  34.7138599,  74.3519506, -60.4083194};
    std::vector<double> data = data1;
    data.insert(data.end(), data2.begin(), data2.end());

    DataFrame df;
    df.set_exact_series(ExactSeries(data));

    double p = df.unequal_variance_t_test(static_cast<int>(data1.size()));
    CHECK_NEAR(p, 0.02043, 1e-4);
}

// C# Test_Ftest.
void test_f_test() {
    std::vector<double> data1 = {200.1, 190.9, 192.7, 213, 241.4, 196.9, 172.2, 185.5, 205.2, 193.7};
    std::vector<double> data2 = {392.9, 393.2, 345.1, 393, 434, 427.9, 422, 383.9, 392.3, 352.2};
    std::vector<double> data = data1;
    data.insert(data.end(), data2.begin(), data2.end());

    DataFrame df;
    df.set_exact_series(ExactSeries(data));

    double p = df.f_test(static_cast<int>(data1.size()));
    CHECK_NEAR(p, 0.1825, 1e-4);
}

// C# Test_JarqueBera.
void test_jarque_bera_test() {
    std::vector<double> data = {
        -17.82175266, -2.33394663,  4.66366786,   6.77181741,   48.09893105,  -28.26940615,
        -9.98265593,  -0.87518792,  14.97758789,  -0.54200675,  8.80374205,   -3.40846222,
        -3.35109891,  -12.98362149, -1.42481547,  18.5800533,   10.86238267,  -13.65904345,
        1.76995771,   13.91485418,  10.8528196,   -11.69442361, -11.6048953,  5.89082943,
        -13.20258835, 3.93329214,   2.62990935,   -4.00680666,  18.4215721,   0.14773234,
        10.20778973,  -7.41284797,  8.42081407,   35.41116192,  59.37166512,  9.36721778,
        22.37395361,  22.9971476,   13.47067667,  -18.98066759, -22.84094314, 16.7108515,
        18.72618308,  29.97227498,  -16.078326,   -0.26901107,  -0.05773469,  14.44571902,
        -7.23727541,  18.87940528,  -10.55665291, -0.40463948,  1.1599797,    -9.47746043,
        -8.83651712,  -0.1277879,   -7.43500345,  18.02267959,  10.38996171,  6.73008507,
        -0.78965999,  -6.63662283,  -0.26534812,  -18.26597299, 10.68284417,  4.14715065,
        -22.73605154, 0.38107214,   17.99480125,  4.67217999,   -7.55979566,  -5.02964486,
        10.07853161,  -10.20580542, -3.83664015,  -3.13645528,  -4.30412819,  12.06651361,
        16.46249676,  -0.77303738,  10.72787315,  12.09162065,  8.22959713,   5.86544228,
        -11.14598952, 9.55434186,   -4.24740884,  -2.84574008,  -7.08625811,  0.80619592,
        12.92545548,  -3.22668772,  -25.39204102, 9.92546076,   -3.16982112,  18.60432604,
        -14.00214643, 1.17374306,   -13.04390662, 24.21704845,  3.82716675,   -5.17619789,
        -8.06288031,  4.1033081,    -13.36564786, -15.91238602, -25.39452748, 18.80121063,
        7.80923857,   -6.8516946,   6.54494797,   26.80612853,  4.65921504,   23.73597901,
        44.15782916,  2.64243694,   -24.27815428, 40.02096079,  6.5730404,    25.71086816,
        28.14206721,  55.47192889,  -14.9762203,  -10.03718017, 38.03084527,  7.20256442};

    DataFrame df;
    df.set_exact_series(ExactSeries(data));

    double p = df.jarque_bera_test();
    CHECK_NEAR(p, 3.444e-05 / 2.0, 1e-5);

    std::vector<double> data2 = {4, 5, 5, 6, 9, 12, 13, 14, 14, 19, 22, 24, 25};
    df.set_exact_series(ExactSeries(data2));
    p = df.jarque_bera_test();
    CHECK_NEAR(p, 0.592128, 1e-6);
}

// C# Test_LjungBox.
void test_ljung_box_test() {
    std::vector<double> data = {
        -17.82175266, -2.33394663,  4.66366786,   6.77181741,   48.09893105,  -28.26940615,
        -9.98265593,  -0.87518792,  14.97758789,  -0.54200675,  8.80374205,   -3.40846222,
        -3.35109891,  -12.98362149, -1.42481547,  18.5800533,   10.86238267,  -13.65904345,
        1.76995771,   13.91485418,  10.8528196,   -11.69442361, -11.6048953,  5.89082943,
        -13.20258835, 3.93329214,   2.62990935,   -4.00680666,  18.4215721,   0.14773234,
        10.20778973,  -7.41284797,  8.42081407,   35.41116192,  59.37166512,  9.36721778,
        22.37395361,  22.9971476,   13.47067667,  -18.98066759, -22.84094314, 16.7108515,
        18.72618308,  29.97227498,  -16.078326,   -0.26901107,  -0.05773469,  14.44571902,
        -7.23727541,  18.87940528,  -10.55665291, -0.40463948,  1.1599797,    -9.47746043,
        -8.83651712,  -0.1277879,   -7.43500345,  18.02267959,  10.38996171,  6.73008507,
        -0.78965999,  -6.63662283,  -0.26534812,  -18.26597299, 10.68284417,  4.14715065,
        -22.73605154, 0.38107214,   17.99480125,  4.67217999,   -7.55979566,  -5.02964486,
        10.07853161,  -10.20580542, -3.83664015,  -3.13645528,  -4.30412819,  12.06651361,
        16.46249676,  -0.77303738,  10.72787315,  12.09162065,  8.22959713,   5.86544228,
        -11.14598952, 9.55434186,   -4.24740884,  -2.84574008,  -7.08625811,  0.80619592,
        12.92545548,  -3.22668772,  -25.39204102, 9.92546076,   -3.16982112,  18.60432604,
        -14.00214643, 1.17374306,   -13.04390662, 24.21704845,  3.82716675,   -5.17619789,
        -8.06288031,  4.1033081,    -13.36564786, -15.91238602, -25.39452748, 18.80121063,
        7.80923857,   -6.8516946,   6.54494797,   26.80612853,  4.65921504,   23.73597901,
        44.15782916,  2.64243694,   -24.27815428, 40.02096079,  6.5730404,    25.71086816,
        28.14206721,  55.47192889,  -14.9762203,  -10.03718017, 38.03084527,  7.20256442};

    DataFrame df;
    df.set_exact_series(ExactSeries(data));

    double p1 = df.ljung_box_test(5);
    CHECK_NEAR(p1, 0.2314, 1e-4);

    double p2 = df.ljung_box_test(30);
    CHECK_NEAR(p2, 0.7548, 1e-4);
}

// C# Test_WaldWolfowitz -- Harricana River data (Bobee & Ashkar 1991, Table 1.2, page 5).
void test_wald_wolfowitz_test() {
    std::vector<double> data = {
        122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225,
        174, 98.8, 149, 238, 262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172,
        153, 142, 317, 161, 201, 204, 194, 164, 183, 161, 167, 179, 185, 117, 192,
        337, 125, 166, 99.1, 202, 230, 158, 262, 154, 164, 182, 164, 183, 171, 250,
        184, 205, 237, 177, 239, 187, 180, 173, 174};

    DataFrame df;
    df.set_exact_series(ExactSeries(data));

    double p = df.wald_wolfowitz_test();
    double true_p =
        (1.0 - corehydro::numerics::distributions::Normal::standard_cdf(1.167)) * 2.0;
    CHECK_NEAR(p, true_p, 1e-3);
}

// C# Test_MannWhitney -- same Harricana River data, page 7.
void test_mann_whitney_test() {
    std::vector<double> data = {
        122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225,
        174, 98.8, 149, 238, 262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172,
        153, 142, 317, 161, 201, 204, 194, 164, 183, 161, 167, 179, 185, 117, 192,
        337, 125, 166, 99.1, 202, 230, 158, 262, 154, 164, 182, 164, 183, 171, 250,
        184, 205, 237, 177, 239, 187, 180, 173, 174};

    DataFrame df;
    df.set_exact_series(ExactSeries(data));

    double p = df.mann_whitney_test(50);
    double true_p =
        (1.0 - corehydro::numerics::distributions::Normal::standard_cdf(0.54)) * 2.0;
    CHECK_NEAR(p, true_p, 1e-2);
}

// C# Test_MannKendall -- same Harricana River data.
void test_mann_kendall_test() {
    std::vector<double> data = {
        122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225,
        174, 98.8, 149, 238, 262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172,
        153, 142, 317, 161, 201, 204, 194, 164, 183, 161, 167, 179, 185, 117, 192,
        337, 125, 166, 99.1, 202, 230, 158, 262, 154, 164, 182, 164, 183, 171, 250,
        184, 205, 237, 177, 239, 187, 180, 173, 174};

    DataFrame df;
    df.set_exact_series(ExactSeries(data));

    double p = df.mann_kendall_test();
    CHECK_NEAR(p, 0.7757, 1e-4);
}

// C# Test_LinearTrendTest.
void test_linear_trend_test() {
    std::vector<double> data(100);
    for (int i = 0; i < 100; i++) data[static_cast<std::size_t>(i)] = std::cos(i + 1);

    DataFrame df;
    df.set_exact_series(ExactSeries(data));

    double p = df.linear_trend_test();
    CHECK_NEAR(p, 0.9092, 1e-4);
}

// C# SummaryAndStandardizedValues_DuplicateValues_ReturnFiniteResults.
void test_summary_and_standardized_values_duplicate_values_return_finite_results() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{
        100, 100, 125, 125, 150, 150, 175, 175, 200, 200, 250, 300}));
    df.calculate_plotting_positions();

    std::vector<std::pair<std::string, double>> summary = df.summary_statistics_all_data();
    df.set_standardized_values();

    CHECK_TRUE(std::isfinite(find_value(summary, "Mean")));
    CHECK_TRUE(std::isfinite(find_value(summary, "Std Dev")));
    CHECK_TRUE(std::isfinite(find_value(summary, "Mean (of log)")));
    CHECK_TRUE(std::isfinite(find_value(summary, "50%")));

    bool all_finite = true;
    for (std::size_t i = 0; i < df.exact_series().count(); i++) {
        if (!std::isfinite(df.exact_series()[i].standardized_value()) ||
            !std::isfinite(df.exact_series()[i].standardized_log10_value())) {
            all_finite = false;
        }
    }
    CHECK_TRUE(all_finite);
}

// C# EmpiricalConsumers_AllValuesIdentical_ReportUnavailableResults.
void test_empirical_consumers_all_values_identical_report_unavailable_results() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>(10, 100.0)));
    df.calculate_plotting_positions();

    std::optional<std::vector<double>> moments = df.get_nonparametric_moments();
    std::vector<std::pair<std::string, double>> summary = df.summary_statistics_all_data();
    df.set_standardized_values();

    CHECK_TRUE(!moments.has_value());
    CHECK_EQ(find_value(summary, "Minimum"), 100.0);
    CHECK_EQ(find_value(summary, "Maximum"), 100.0);
    CHECK_TRUE(std::isnan(find_value(summary, "Mean")));
    CHECK_TRUE(std::isnan(find_value(summary, "50%")));

    bool all_nan = true;
    for (std::size_t i = 0; i < df.exact_series().count(); i++) {
        if (!std::isnan(df.exact_series()[i].standardized_value()) ||
            !std::isnan(df.exact_series()[i].standardized_log10_value())) {
            all_nan = false;
        }
    }
    CHECK_TRUE(all_nan);
}

}  // namespace

int main() {
    test_equal_variance_t_test();
    test_unequal_variance_t_test();
    test_f_test();
    test_jarque_bera_test();
    test_ljung_box_test();
    test_wald_wolfowitz_test();
    test_mann_whitney_test();
    test_mann_kendall_test();
    test_linear_trend_test();
    test_summary_and_standardized_values_duplicate_values_return_finite_results();
    test_empirical_consumers_all_values_identical_report_unavailable_results();
    return chtest::summary("test_data_frame_facades");
}
