// Transcribed from: upstream/Numerics/Test_Numerics/Data/Statistics/Test_HypothesisTests.cs
// @ 2a0357a. Every [TestMethod] in that file EXCEPT Test_UnimodalityTest, Test_GrubbsBeck, and
// Test_MultipleGrubbsBeck (the last two exercise the already-ported MultipleGrubbsBeckTest, not
// this class; UnimodalityTest is a documented P4 severance -- see hypothesis_tests.hpp's file
// header). All input arrays are copied verbatim from the C# test file (byte-for-byte, verified
// by scripted extraction against the checked-out submodule, not retyped from any summary).
#include <cmath>
#include <vector>

#include "corehydro/numerics/data/hypothesis_tests.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

namespace ht = corehydro::numerics::data::hypothesis_tests;
namespace dist = corehydro::numerics::distributions;

namespace {

// Test_OneSampleTtest
void test_one_sample_t_test() {
    std::vector<double> data{8.782932,   10.64199,   -1.63955,   -6.802458,  9.088312,
                              -27.26934,  9.451478,   -4.142762,  -4.262396,  -13.78983,
                              -1.743717,  27.259681,  5.559418,   7.803247,   -11.25798,
                              12.253498,  -13.295363, -4.973664,  16.81069,   4.480855,
                              11.694329,  21.836776,  -9.664926,  -23.297061, -23.965643,
                              27.076463,  -7.22471,   9.305697,   9.181852,   -2.434665};

    double p = ht::one_sample_t_test(data);
    CHECK_NEAR(p, 0.6489, 1e-4);

    p = ht::one_sample_t_test(data, 10.0);
    CHECK_NEAR(p, 0.001823, 1e-4);

    double t = ht::one_sample_t_test(
        std::vector<double>{23, 15, -5, 7, 1, -10, 12, -8, 20, 8, -2, -5});
    CHECK_NEAR(t, 0.087585 * 2, 1e-6);
}

// Test_EqualVarianceTtest
void test_equal_variance_t_test() {
    std::vector<double> data1{8.782932,   10.64199,   -1.63955,   -6.802458,  9.088312,
                               -27.26934,  9.451478,   -4.142762,  -4.262396,  -13.78983,
                               -1.743717,  27.259681,  5.559418,   7.803247,   -11.25798,
                               12.253498,  -13.295363, -4.973664,  16.81069,   4.480855,
                               11.694329,  21.836776,  -9.664926,  -23.297061, -23.965643,
                               27.076463,  -7.22471,   9.305697,   9.181852,   -2.434665};
    std::vector<double> data2{34.3561954,  75.9050064,  71.4757101,  58.9733692,  17.6281358,
                                24.7356484,  0.2774026,   -39.8615073, 63.0320155,  10.7740315,
                                43.855325,   -61.4107418, -21.8079666, 38.1142162,  35.6335516,
                                53.8218821,  -32.3929633, 27.0220976,  27.4956296,  -29.203965,
                                -6.2115822,  68.4307799,  11.6077081,  20.5498852,  -10.1292962,
                                18.3386108,  30.7351382,  34.7138599,  74.3519506,  -60.4083194};

    double p = ht::equal_variance_t_test(data1, data2);
    CHECK_NEAR(p, 0.0185, 1e-4);
}

// Test_UnequalVarianceTtest
void test_unequal_variance_t_test() {
    std::vector<double> data1{8.782932,   10.64199,   -1.63955,   -6.802458,  9.088312,
                               -27.26934,  9.451478,   -4.142762,  -4.262396,  -13.78983,
                               -1.743717,  27.259681,  5.559418,   7.803247,   -11.25798,
                               12.253498,  -13.295363, -4.973664,  16.81069,   4.480855,
                               11.694329,  21.836776,  -9.664926,  -23.297061, -23.965643,
                               27.076463,  -7.22471,   9.305697,   9.181852,   -2.434665};
    std::vector<double> data2{34.3561954,  75.9050064,  71.4757101,  58.9733692,  17.6281358,
                                24.7356484,  0.2774026,   -39.8615073, 63.0320155,  10.7740315,
                                43.855325,   -61.4107418, -21.8079666, 38.1142162,  35.6335516,
                                53.8218821,  -32.3929633, 27.0220976,  27.4956296,  -29.203965,
                                -6.2115822,  68.4307799,  11.6077081,  20.5498852,  -10.1292962,
                                18.3386108,  30.7351382,  34.7138599,  74.3519506,  -60.4083194};

    double p = ht::unequal_variance_t_test(data1, data2);
    CHECK_NEAR(p, 0.02043, 1e-4);
}

// Test_PairedTtest
void test_paired_t_test() {
    std::vector<double> data1{200.1, 190.9, 192.7, 213, 241.4, 196.9, 172.2, 185.5, 205.2, 193.7};
    std::vector<double> data2{392.9, 393.2, 345.1, 393, 434, 427.9, 422, 383.9, 392.3, 352.2};

    double p = ht::paired_t_test(data1, data2);
    CHECK_NEAR(p, 6.2e-9, 1e-4);
}

// Test_Ftest
void test_f_test() {
    std::vector<double> data1{200.1, 190.9, 192.7, 213, 241.4, 196.9, 172.2, 185.5, 205.2, 193.7};
    std::vector<double> data2{392.9, 393.2, 345.1, 393, 434, 427.9, 422, 383.9, 392.3, 352.2};

    double p = ht::f_test(data1, data2);
    CHECK_NEAR(p, 0.1825, 1e-4);
}

// Test_FtestModels
void test_f_test_models() {
    double f_stat = 0, p_value = 0;
    ht::f_test_models(1224.32, 720.27, 49, 48, f_stat, p_value);
    CHECK_NEAR(f_stat, 33.5899, 1e-3);
    CHECK_NEAR(p_value, 0.0, 1e-6);
}

// Test_JarqueBera
void test_jarque_bera_test() {
    std::vector<double> data{
        -17.82175266, -2.33394663, 4.66366786, 6.77181741, 48.09893105, -28.26940615, -9.98265593,
        -0.87518792, 14.97758789, -0.54200675, 8.80374205, -3.40846222, -3.35109891, -12.98362149,
        -1.42481547, 18.5800533, 10.86238267, -13.65904345, 1.76995771, 13.91485418, 10.8528196,
        -11.69442361, -11.6048953, 5.89082943, -13.20258835, 3.93329214, 2.62990935, -4.00680666,
        18.4215721, 0.14773234, 10.20778973, -7.41284797, 8.42081407, 35.41116192, 59.37166512,
        9.36721778, 22.37395361, 22.9971476, 13.47067667, -18.98066759, -22.84094314, 16.7108515,
        18.72618308, 29.97227498, -16.078326, -0.26901107, -0.05773469, 14.44571902, -7.23727541,
        18.87940528, -10.55665291, -0.40463948, 1.1599797, -9.47746043, -8.83651712, -0.1277879,
        -7.43500345, 18.02267959, 10.38996171, 6.73008507, -0.78965999, -6.63662283, -0.26534812,
        -18.26597299, 10.68284417, 4.14715065, -22.73605154, 0.38107214, 17.99480125, 4.67217999,
        -7.55979566, -5.02964486, 10.07853161, -10.20580542, -3.83664015, -3.13645528, -4.30412819,
        12.06651361, 16.46249676, -0.77303738, 10.72787315, 12.09162065, 8.22959713, 5.86544228,
        -11.14598952, 9.55434186, -4.24740884, -2.84574008, -7.08625811, 0.80619592, 12.92545548,
        -3.22668772, -25.39204102, 9.92546076, -3.16982112, 18.60432604, -14.00214643, 1.17374306,
        -13.04390662, 24.21704845, 3.82716675, -5.17619789, -8.06288031, 4.1033081, -13.36564786,
        -15.91238602, -25.39452748, 18.80121063, 7.80923857, -6.8516946, 6.54494797, 26.80612853,
        4.65921504, 23.73597901, 44.15782916, 2.64243694, -24.27815428, 40.02096079, 6.5730404,
        25.71086816, 28.14206721, 55.47192889, -14.9762203, -10.03718017, 38.03084527, 7.20256442};

    double p = ht::jarque_bera_test(data);
    CHECK_NEAR(p, 3.444e-05 / 2.0, 1e-5);

    double jb = ht::jarque_bera_test(
        std::vector<double>{4, 5, 5, 6, 9, 12, 13, 14, 14, 19, 22, 24, 25});
    CHECK_NEAR(jb, 0.592128, 1e-6);
}

// Table 1.2 Maximum annual peak discharge values in cms, observed at the Harricana River at
// Amos (Quebec, Canada). Shared verbatim by Test_WaldWolfowitz and Test_MannKendall in the C#
// source (confirmed byte-identical by scripted extraction).
std::vector<double> harricana69() {
    return {122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225, 174, 98.8,
             149, 238, 262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172, 153, 142, 317, 161,
             201, 204, 194, 164, 183, 161, 167, 179, 185, 117, 192, 337, 125, 166, 99.1, 202, 230,
             158, 262, 154, 164, 182, 164, 183, 171, 250, 184, 205, 237, 177, 239, 187, 180, 173,
             174};
}

// Test_WaldWolfowitz
void test_wald_wolfowitz_test() {
    auto data = harricana69();
    double p = ht::wald_wolfowitz_test(data);
    double true_p = (1.0 - dist::Normal::standard_cdf(1.167)) * 2.0;  // see page 5 of reference
    CHECK_NEAR(p, true_p, 1e-3);
}

// Test_LjungBox (reuses the same 126-value series as Test_JarqueBera)
void test_ljung_box_test() {
    std::vector<double> data{
        -17.82175266, -2.33394663, 4.66366786, 6.77181741, 48.09893105, -28.26940615, -9.98265593,
        -0.87518792, 14.97758789, -0.54200675, 8.80374205, -3.40846222, -3.35109891, -12.98362149,
        -1.42481547, 18.5800533, 10.86238267, -13.65904345, 1.76995771, 13.91485418, 10.8528196,
        -11.69442361, -11.6048953, 5.89082943, -13.20258835, 3.93329214, 2.62990935, -4.00680666,
        18.4215721, 0.14773234, 10.20778973, -7.41284797, 8.42081407, 35.41116192, 59.37166512,
        9.36721778, 22.37395361, 22.9971476, 13.47067667, -18.98066759, -22.84094314, 16.7108515,
        18.72618308, 29.97227498, -16.078326, -0.26901107, -0.05773469, 14.44571902, -7.23727541,
        18.87940528, -10.55665291, -0.40463948, 1.1599797, -9.47746043, -8.83651712, -0.1277879,
        -7.43500345, 18.02267959, 10.38996171, 6.73008507, -0.78965999, -6.63662283, -0.26534812,
        -18.26597299, 10.68284417, 4.14715065, -22.73605154, 0.38107214, 17.99480125, 4.67217999,
        -7.55979566, -5.02964486, 10.07853161, -10.20580542, -3.83664015, -3.13645528, -4.30412819,
        12.06651361, 16.46249676, -0.77303738, 10.72787315, 12.09162065, 8.22959713, 5.86544228,
        -11.14598952, 9.55434186, -4.24740884, -2.84574008, -7.08625811, 0.80619592, 12.92545548,
        -3.22668772, -25.39204102, 9.92546076, -3.16982112, 18.60432604, -14.00214643, 1.17374306,
        -13.04390662, 24.21704845, 3.82716675, -5.17619789, -8.06288031, 4.1033081, -13.36564786,
        -15.91238602, -25.39452748, 18.80121063, 7.80923857, -6.8516946, 6.54494797, 26.80612853,
        4.65921504, 23.73597901, 44.15782916, 2.64243694, -24.27815428, 40.02096079, 6.5730404,
        25.71086816, 28.14206721, 55.47192889, -14.9762203, -10.03718017, 38.03084527, 7.20256442};

    double p = ht::ljung_box_test(data, 5);
    CHECK_NEAR(p, 0.2314, 1e-4);

    double p2 = ht::ljung_box_test(data, 30);
    CHECK_NEAR(p2, 0.7548, 1e-4);
}

// Test_MannWhitney (note the SHORTER sample first -- the method throws when n1 > n2)
void test_mann_whitney_test() {
    std::vector<double> data1{122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331,
                                225, 174, 98.8, 149, 238, 262, 132, 235, 216, 240, 230, 192, 195,
                                172, 173, 172, 153, 142, 317, 161, 201, 204, 194, 164, 183, 161, 167,
                                179, 185, 117, 192, 337, 125, 166, 99.1, 202};
    std::vector<double> data2{230, 158, 262, 154, 164, 182, 164, 183, 171, 250, 184, 205, 237, 177,
                                239, 187, 180, 173, 174};

    double p = ht::mann_whitney_test(data2, data1);
    double true_p = (1.0 - dist::Normal::standard_cdf(0.54)) * 2.0;  // see page 7 of reference
    CHECK_NEAR(p, true_p, 1e-2);
}

// Test_MannKendall
void test_mann_kendall_test() {
    auto data = harricana69();
    double p = ht::mann_kendall_test(data);
    CHECK_NEAR(p, 0.7757, 1e-4);
}

// Test_LinearTrendTest
void test_linear_trend_test() {
    std::vector<double> time{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                              18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                              35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                              52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                              69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                              86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100};
    std::vector<double> data(time.size());
    for (std::size_t i = 0; i < time.size(); ++i) data[i] = std::cos(time[i]);

    double p_val = ht::linear_trend_test(time, data);
    CHECK_NEAR(p_val, 0.9092, 1e-4);
}

// Supplement (corehydro addition beyond the C# assertions): every guard actually throws.
void test_guards() {
    CHECK_THROWS(ht::one_sample_t_test(std::vector<double>{1.0}));
    CHECK_THROWS(ht::equal_variance_t_test(std::vector<double>{1.0}, std::vector<double>{1.0}));

    {
        std::vector<double> a{1.0, 2.0, 3.0};
        std::vector<double> b{1.0, 2.0};
        CHECK_THROWS(ht::paired_t_test(a, b));
    }

    CHECK_THROWS(ht::f_test(std::vector<double>{1.0}, std::vector<double>{1.0, 2.0}));

    {
        double f_stat = 0, p_value = 0;
        CHECK_THROWS(ht::f_test_models(1224.32, 720.27, 48, 48, f_stat, p_value));
    }

    {
        // n1 > n2: the first sample must be <= the second in length.
        std::vector<double> longer{1, 2, 3, 4, 5};
        std::vector<double> shorter{1, 2, 3, 4};
        CHECK_THROWS(ht::mann_whitney_test(longer, shorter));
    }
    {
        // combined length exactly 20 -- must exceed 20, not merely reach it.
        std::vector<double> s1{1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<double> s2{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        CHECK_EQ(s1.size() + s2.size(), static_cast<std::size_t>(20));
        CHECK_THROWS(ht::mann_whitney_test(s1, s2));
    }

    {
        std::vector<double> nine{1, 2, 3, 4, 5, 6, 7, 8, 9};
        CHECK_THROWS(ht::mann_kendall_test(nine));
    }
}

}  // namespace

int main() {
    test_one_sample_t_test();
    test_equal_variance_t_test();
    test_unequal_variance_t_test();
    test_paired_t_test();
    test_f_test();
    test_f_test_models();
    test_jarque_bera_test();
    test_wald_wolfowitz_test();
    test_ljung_box_test();
    test_mann_whitney_test();
    test_mann_kendall_test();
    test_linear_trend_test();
    test_guards();
    return chtest::summary("test_hypothesis_tests");
}
