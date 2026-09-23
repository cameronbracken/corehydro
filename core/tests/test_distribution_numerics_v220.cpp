#include <cmath>
#include <limits>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/distributions/base/distribution_numerics.hpp"
#include "corehydro/numerics/distributions/base/gamma_distribution_numerics.hpp"
#include "corehydro/numerics/distributions/base/distribution_snapshot.hpp"
#include "corehydro/numerics/distributions/base/distribution_moment_integration.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/exponential.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"
#include "corehydro/numerics/distributions/generalized_logistic.hpp"
#include "corehydro/numerics/distributions/generalized_normal.hpp"
#include "corehydro/numerics/distributions/generalized_pareto.hpp"
#include "corehydro/numerics/distributions/gumbel.hpp"
#include "corehydro/numerics/distributions/kappa_four.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/log_pearson_type_iii.hpp"
#include "corehydro/numerics/distributions/logistic.hpp"
#include "corehydro/numerics/distributions/pearson_type_iii.hpp"
#include "corehydro/numerics/distributions/weibull.hpp"

namespace dn = corehydro::numerics::distributions::distribution_numerics;

namespace {

void test_gamma_tiny_shape() {
    CHECK_NEAR(dn::gamma_log_cdf(1e-20, 0.5), -5.597735947761608e-21, 1e-34);
    CHECK_NEAR(dn::gamma_log_cdf(1e-100, 0.5), -5.597735947761608e-101, 1e-113);
    CHECK_NEAR(dn::gamma_quantile_shape_derivative(0.9, std::numeric_limits<double>::denorm_min()),
               4.090863547565521e-321, 2.0 * std::numeric_limits<double>::denorm_min());
}

void test_gamma_log_tails() {
    const double cases[][4] = {
        {0.001, 1e-10, -0.022449457331756989, -3.8076925670559727},
        {0.001, 1, -0.0002196324750319099, -8.4236647908310722},
        {0.001, 1000, 0, -1013.8090239153952},
        {0.5, 1e-10, -11.392143227368317, -1.1283855333035131e-5},
        {0.5, 1, -0.17114331524104021, -1.8496055099332522},
        {0.5, 1000, 0, -1004.0267419589519},
        {1, 1e-10, -23.025850929990458, -1e-10},
        {1, 1, -0.45867514538708193, -1},
        {100, 80, -4.0681907911970132, -0.017256351106458189},
        {100, 100, -0.66689715058528931, -0.72010489302547409},
        {100, 120, -0.028259299048920376, -3.5804290809275314},
        {10000, 9800, -3.8073232363866576, -0.022457843956297584},
        {10000, 10000, -0.69049109440197243, -0.69581034030382005},
        {10000, 10200, -0.023562756305381644, -3.7598461809322816},
        {1e8, 99980000, -3.7834216952685238, -0.023007384281740352},
        {1e8, 1e8, -0.69312058476158833, -0.69317377706565741},
        {1e8, 100020000, -0.023018433854231277, -3.7829470521492174},
    };
    for (const auto& row : cases) {
        CHECK_NEAR(dn::gamma_log_cdf(row[0], row[1]), row[2], 3e-12);
        CHECK_NEAR(dn::gamma_log_survival(row[0], row[1]), row[3], 3e-12);
    }
}

void test_gamma_inverse() {
    CHECK_NEAR(dn::gamma_inverse_cdf(1, 1e-20, true), 46.051701859880914, 1e-13);
    CHECK_NEAR(dn::gamma_inverse_cdf(1, 1e-20), 1e-20, 1e-35);
    CHECK_NEAR(dn::gamma_inverse_cdf(0.001, 0.5), 5.244206408274966e-302, 2e-313);
    for (double a : {0.1, 0.5, 2.0, 100.0, 10000.0, 1e8}) {
        for (double p : {1e-10, 0.1, 0.5}) {
            for (bool upper : {false, true}) {
                double value = dn::gamma_inverse_cdf(a, p, upper);
                double actual = upper ? dn::gamma_log_survival(a, value)
                                      : dn::gamma_log_cdf(a, value);
                CHECK_NEAR(actual, std::log(p), a >= 1e8 ? 2e-11 : 2e-12);
            }
        }
    }
}

void test_log_primitives_and_determinant() {
    CHECK_NEAR(dn::normal_log_cdf(-40), -804.6084420137538, 2e-13);
    CHECK_NEAR(dn::normal_log_survival(9), -43.62814911333212, 2e-13);
    CHECK_TRUE(std::isinf(dn::log1m_exp(0)) && dn::log1m_exp(0) < 0);
    CHECK_EQ(dn::log1m_exp(-std::numeric_limits<double>::infinity()), 0.0);
    CHECK_EQ(dn::exprel(0), 1.0);
    CHECK_EQ(dn::exprel_derivative(0), 0.5);
    CHECK_EQ(dn::standardize(1e308, -1e308, 1e308), 2.0);
    CHECK_EQ(dn::standardize(5.0, 1.0, 2.0), 2.0);

    int sign = 0;
    CHECK_NEAR(dn::log_abs_determinant({{1e200, 1e200}, {1e200, -1e200}}, sign),
               400 * std::log(10.0) + std::log(2.0), 2e-13);
    CHECK_EQ(sign, -1);
    CHECK_TRUE(std::isinf(dn::log_abs_determinant({{1, 2}, {1, 2}}, sign)));
    CHECK_EQ(sign, 0);
    CHECK_NEAR(dn::log_abs_determinant({{1e308, 1e-308}, {1e308, 2e-308}}, sign), 0, 1e-12);
    CHECK_EQ(sign, 1);
    CHECK_TRUE(std::isinf(
        dn::log_abs_determinant({{1, 2, 3}, {4, 5, 6}, {5, 7, 9}}, sign)));
    CHECK_EQ(sign, 0);
}

void test_snapshot_round_trip() {
    corehydro::numerics::distributions::Normal distribution(-0.0, 2.0);
    auto snapshot = corehydro::numerics::distributions::DistributionSnapshot::try_capture(
        &distribution);
    CHECK_TRUE(snapshot.has_value());
    CHECK_TRUE(snapshot->matches(&distribution));
    distribution.set_parameters(+0.0, 2.0);
    CHECK_TRUE(!snapshot->matches(&distribution));
    distribution.set_parameters(1.0, 3.0);
    CHECK_TRUE(!snapshot->matches(&distribution));
    CHECK_TRUE(!corehydro::numerics::distributions::DistributionSnapshot::try_capture(nullptr)
                    .has_value());
}

void test_full_support_moment_integration() {
    auto moments =
        corehydro::numerics::distributions::distribution_moment_integration::compute(
            [](double x) { return -0.5 * x * x - corehydro::numerics::kLogSqrt2PI; },
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(), 0.0, 1.0);
    CHECK_NEAR(moments[0], 0.0, 1e-10);
    CHECK_NEAR(moments[1], 1.0, 1e-8);
    CHECK_NEAR(moments[2], 0.0, 1e-8);
    CHECK_NEAR(moments[3], 3.0, 1e-7);
    CHECK_THROWS(corehydro::numerics::distributions::distribution_moment_integration::compute(
        [](double) { return 0.0; }, 0.0, 1.0, 0.0, 0.0));
}

void test_workhorse_family_log_tails() {
    using namespace corehydro::numerics::distributions;
    Normal normal(0.0, 1.0);
    CHECK_NEAR(normal.log_cdf(-40.0), -804.6084420137538, 2e-13);
    CHECK_NEAR(normal.log_ccdf(9.0), -43.62814911333212, 2e-13);
    CHECK_TRUE(std::isfinite(normal.log_pdf(40.0)));

    GammaDistribution gamma(2.0, 0.5);
    CHECK_NEAR(gamma.log_ccdf(2000.0), -1004.0267419589519, 3e-12);
    CHECK_TRUE(std::isinf(GammaDistribution(2.0, 0.5).log_pdf(0.0)));

    Exponential exponential(3.0, 2.0);
    CHECK_NEAR(exponential.log_ccdf(2003.0), -1000.0, 0.0);
    CHECK_NEAR(exponential.inverse_cdf(exponential.cdf(5.0)), 5.0, 1e-14);

    Gumbel gumbel(3.0, 2.0);
    CHECK_NEAR(gumbel.log_ccdf(2003.0), -1000.0, 1e-12);
    Logistic logistic(3.0, 2.0);
    CHECK_NEAR(logistic.log_ccdf(2003.0), -1000.0, 1e-12);

    GeneralizedExtremeValue gev(3.0, 2.0, 1e-8);
    const double p = 0.9;
    CHECK_NEAR(gev.cdf(gev.inverse_cdf(p)), p, 2e-15);

    GeneralizedLogistic glo(3.0, 2.0, 1e-8);
    CHECK_NEAR(glo.cdf(glo.inverse_cdf(p)), p, 2e-15);
    GeneralizedNormal gno(3.0, 2.0, 1e-8);
    CHECK_NEAR(gno.cdf(gno.inverse_cdf(p)), p, 2e-15);
    GeneralizedPareto gpa(3.0, 2.0, 1e-8);
    CHECK_NEAR(gpa.cdf(gpa.inverse_cdf(p)), p, 2e-15);

    Weibull weibull(2.0, 0.5);
    CHECK_TRUE(std::isfinite(weibull.log_pdf(1e-300)));
    CHECK_NEAR(weibull.cdf(weibull.inverse_cdf(p)), p, 2e-15);

    LogNormal lognormal(-2.0, 0.5);
    CHECK_TRUE(std::isfinite(lognormal.log_pdf(1e-300)));
    CHECK_NEAR(lognormal.cdf(lognormal.inverse_cdf(p)), p, 2e-15);
    CHECK_TRUE(lognormal.mode() < lognormal.median());
    CHECK_TRUE(lognormal.quantile_variance(
                   p, 100, ParameterEstimationMethod::MaximumLikelihood) > 0.0);

    LnNormal lnnormal(1e-200, 2e-200);
    CHECK_EQ(lnnormal.get_parameters()[0], 1e-200);
    CHECK_EQ(lnnormal.get_parameters()[1], 2e-200);
    CHECK_TRUE(std::isfinite(lnnormal.log_pdf(1e-300)));

    PearsonTypeIII pearson(10.0, 2.0, 1e-8);
    CHECK_TRUE(pearson.minimum() > -std::numeric_limits<double>::infinity());
    CHECK_TRUE(pearson.inverse_cdf(0.9) != 10.0 + 2.0 * Normal::standard_z(0.9));
    CHECK_NEAR(pearson.cdf(pearson.inverse_cdf(0.9)), 0.9, 2e-15);
    CHECK_TRUE(std::isfinite(pearson.log_ccdf(100.0)));

    PearsonTypeIII reverse_pearson(10.0, 2.0, -0.5);
    CHECK_NEAR(reverse_pearson.cdf(reverse_pearson.inverse_cdf(1e-10)), 1e-10, 2e-22);
    CHECK_TRUE(std::isinf(PearsonTypeIII(0.0, 1.0, 3.0).log_pdf(-2.0 / 3.0)));

    LogPearsonTypeIII log_pearson(2.0, 0.3, 1e-8);
    CHECK_NEAR(log_pearson.cdf(log_pearson.inverse_cdf(0.9)), 0.9, 2e-15);
    CHECK_TRUE(std::isfinite(log_pearson.log_ccdf(1e100)));
    CHECK_TRUE(std::isinf(LogPearsonTypeIII(1.0, 1.5, 3.0).log_pdf(1.0)));
    CHECK_TRUE(std::isinf(LogPearsonTypeIII(-1.0, 1.5, -3.0).log_pdf(1.0)));
    LogPearsonTypeIII small_scale(0.0, 1e-4, 0.25);
    CHECK_TRUE(std::isfinite(small_scale.skewness()));
    CHECK_TRUE(std::isfinite(small_scale.kurtosis()));
    LogPearsonTypeIII divergent(0.0, 1.0, 0.5);
    CHECK_TRUE(std::isfinite(divergent.mean()));
    CHECK_TRUE(std::isinf(divergent.standard_deviation()));
    CHECK_TRUE(std::isnan(divergent.skewness()));

    std::vector<double> initials;
    std::vector<double> lowers;
    std::vector<double> uppers;
    Normal().get_parameter_constraints({-1e300, -5e299, 5e299, 1e300}, initials, lowers,
                                       uppers);
    CHECK_TRUE(std::isfinite(initials[0]) && std::isfinite(initials[1]));
    CHECK_TRUE(lowers[0] < initials[0] && initials[0] < uppers[0]);
    PearsonTypeIII().get_parameter_constraints({-12.0, -8.0, -3.0, -1.0}, initials, lowers,
                                               uppers);
    CHECK_TRUE(lowers[0] < initials[0] && initials[0] < uppers[0]);
    LogPearsonTypeIII().get_parameter_constraints(
        {0.08, 0.12, 0.25, 0.31, 0.45, 0.6, 0.75, 0.9, 1.4}, initials, lowers, uppers);
    CHECK_TRUE(initials[0] < 0.0);
    CHECK_TRUE(lowers[0] <= initials[0] && initials[0] <= uppers[0]);
    CHECK_THROWS(LogPearsonTypeIII().estimate(
        {0.1, 0.2, 0.3, 0.0}, ParameterEstimationMethod::MethodOfMoments));

    LogNormal().get_parameter_constraints(
        {0.08, 0.12, 0.25, 0.31, 0.45, 0.6}, initials, lowers, uppers);
    CHECK_TRUE(initials[0] < 0.0);
    CHECK_TRUE(lowers[0] <= initials[0] && initials[0] <= uppers[0]);
    Exponential().get_parameter_constraints(
        {-1e300, -8e299, -4e299, -1e299}, initials, lowers, uppers);
    CHECK_TRUE(std::isfinite(initials[0]) && std::isfinite(initials[1]));
    CHECK_TRUE(lowers[0] <= initials[0] && initials[0] <= uppers[0]);
    GammaDistribution().get_parameter_constraints(
        {1e-300, 2e-300, 4e-300, 8e-300}, initials, lowers, uppers);
    CHECK_TRUE(initials[0] > 0.0 && initials[1] > 0.0);
    Gumbel().get_parameter_constraints(
        {-1e300, -8e299, -4e299, -1e299}, initials, lowers, uppers);
    CHECK_TRUE(std::isfinite(initials[0]) && std::isfinite(initials[1]));
    Logistic().get_parameter_constraints(
        {-1e300, -8e299, -4e299, -1e299}, initials, lowers, uppers);
    CHECK_TRUE(std::isfinite(initials[0]) && std::isfinite(initials[1]));
    LnNormal().get_parameter_constraints(
        {1e-300, 2e-300, 4e-300, 8e-300}, initials, lowers, uppers);
    CHECK_TRUE(initials[0] > 0.0 && initials[1] > 0.0);
    Weibull().get_parameter_constraints(
        {1e-300, 2e-300, 4e-300, 8e-300}, initials, lowers, uppers);
    CHECK_TRUE(initials[0] > 0.0 && initials[1] > 0.0);

    for (const auto* distribution : std::vector<const IStandardError*>{
             static_cast<const IStandardError*>(&gev),
             static_cast<const IStandardError*>(&glo),
             static_cast<const IStandardError*>(&gpa)}) {
        const auto covariance = distribution->parameter_covariance(
            100, ParameterEstimationMethod::MaximumLikelihood);
        CHECK_EQ(covariance.size(), 3U);
        for (std::size_t i = 0; i < covariance.size(); ++i) {
            CHECK_TRUE(covariance[i][i] > 0.0 && std::isfinite(covariance[i][i]));
            for (std::size_t j = 0; j < covariance.size(); ++j)
                CHECK_NEAR(covariance[i][j], covariance[j][i], 0.0);
        }
        CHECK_TRUE(distribution->quantile_variance(
                       p, 100, ParameterEstimationMethod::MaximumLikelihood) > 0.0);
        const auto gradient = distribution->quantile_gradient(p);
        CHECK_EQ(gradient.size(), 3U);
        for (double value : gradient) CHECK_TRUE(std::isfinite(value));
    }

    KappaFour kappa_gumbel(0.0, 1.0, 0.0, 0.0);
    CHECK_NEAR(kappa_gumbel.inverse_cdf(0.5), 0.366512920581664327, 2e-14);
    CHECK_NEAR(kappa_gumbel.cdf(0.366512920581664327), 0.5, 2e-14);
    CHECK_NEAR(kappa_gumbel.pdf(0.366512920581664327), 0.346573590279972655, 2e-14);
    CHECK_NEAR(kappa_gumbel.log_pdf(-7.0), -1089.6331584284585, 2e-12);
    CHECK_NEAR(kappa_gumbel.log_ccdf(40.0), -40.0, 2e-14);
    CHECK_NEAR(kappa_gumbel.ccdf(40.0), 4.248354255291589e-18, 1e-31);
    for (double kappa : {-1e-16, 0.0, 1e-16})
        for (double hondo : {-1e-16, 0.0, 1e-16}) {
            KappaFour near_zero(0.0, 1.0, kappa, hondo);
            CHECK_NEAR(near_zero.inverse_cdf(0.5), 0.366512920581664327, 2e-14);
            CHECK_NEAR(near_zero.cdf(0.366512920581664327), 0.5, 2e-14);
        }
    KappaFour exponential_kappa(0.0, 1.0, 0.0, 1.0);
    CHECK_NEAR(exponential_kappa.mean(), 1.0, 1e-8);
    CHECK_NEAR(exponential_kappa.standard_deviation(), 1.0, 1e-8);
    CHECK_NEAR(exponential_kappa.skewness(), 2.0, 1e-8);
    CHECK_NEAR(exponential_kappa.kurtosis(), 9.0, 1e-8);
    KappaFour heavy_kappa(0.0, 1.0, -0.5, 0.0);
    CHECK_NEAR(heavy_kappa.mean(), 1.544907701811032, 1e-8);
    CHECK_TRUE(std::isnan(heavy_kappa.standard_deviation()));
    CHECK_TRUE(std::isinf(KappaFour(0.0, 1.0, 0.0, 1.5).pdf(
        KappaFour(0.0, 1.0, 0.0, 1.5).minimum())));
    CHECK_NEAR(KappaFour(0.0, 1.0, 2.0, 0.5).log_cdf(-1.4999999999999998),
               -74.85989550047409, 5e-12);
    const auto kappa_lmom = kappa_gumbel.linear_moments_from_parameters({0.0, 1.0, 0.0, 0.0});
    CHECK_NEAR(kappa_lmom[0], 0.5772156649015329, 1e-12);
    CHECK_NEAR(kappa_lmom[1], 0.6931471805599453, 1e-12);
    CHECK_NEAR(kappa_lmom[2], 0.16992500144231236, 1e-12);
    CHECK_NEAR(kappa_lmom[3], 0.15037499278843736, 1e-12);
    const auto kappa_small =
        kappa_gumbel.linear_moments_from_parameters({0.0, 1.0, 1e-12, 1e-12});
    CHECK_NEAR(kappa_small[0], kappa_lmom[0], 1e-8);
    CHECK_NEAR(kappa_small[1], kappa_lmom[1], 1e-8);
    const auto kappa_recovered = kappa_gumbel.parameters_from_linear_moments(kappa_lmom);
    const auto kappa_round_trip =
        kappa_gumbel.linear_moments_from_parameters(kappa_recovered);
    for (std::size_t i = 0; i < kappa_lmom.size(); ++i)
        CHECK_NEAR(kappa_round_trip[i], kappa_lmom[i], 1e-6);
    const auto kappa_gradient = kappa_gumbel.quantile_gradient(0.5);
    CHECK_NEAR(kappa_gradient[0], 1.0, 0.0);
    CHECK_NEAR(kappa_gradient[1], 0.366512920581664327, 2e-14);
    CHECK_NEAR(kappa_gradient[2], -0.5 * kappa_gradient[1] * kappa_gradient[1], 1e-14);
    CHECK_NEAR(kappa_gradient[3], 0.346573590279972655, 2e-14);
    const auto kappa_covariance = kappa_gumbel.parameter_covariance(
        100, ParameterEstimationMethod::MaximumLikelihood);
    for (std::size_t i = 0; i < kappa_covariance.size(); ++i) {
        CHECK_TRUE(kappa_covariance[i][i] > 0.0 && std::isfinite(kappa_covariance[i][i]));
        for (std::size_t j = 0; j < kappa_covariance.size(); ++j)
            CHECK_NEAR(kappa_covariance[i][j], kappa_covariance[j][i], 0.0);
    }
    CHECK_TRUE(kappa_gumbel.quantile_variance(
                   0.9, 100, ParameterEstimationMethod::MaximumLikelihood) > 0.0);
    CHECK_THROWS(KappaFour(0.0, 1.0, 0.5, 0.0).parameter_covariance(
        100, ParameterEstimationMethod::MaximumLikelihood));

    const std::vector<double> fitting_sample = {
        1.354784607887268, 0.41693252325057983, 0.8899999856948853,
        0.8853314518928528, 2.170344829559326, 1.334205150604248,
        0.5150537490844727, 0.8244029879570007, 0.5099999904632568,
        0.44740423560142517, 0.739365816116333, 1.8200000524520874,
        1.9778419733047485, 0.6553186774253845, 0.8488552570343018,
        2.2905187606811523, 0.9865325689315796, 0.7076110243797302,
        0.2929774820804596, 1.5241589546203613, 0.5311949253082275,
        0.5221586227416992, 0.907939612865448, 0.2859921157360077,
        0.6391091346740723, 0.6330636739730835, 0.520042359828949,
        2.0499587059020996, 1.83543860912323, 2.450000047683716,
        0.6613662838935852, 1.1201441287994385, 1.020573377609253,
        0.5620101094245911, 0.5419405102729797, 2.2691876888275146,
        1.5167182683944702, 3.119999885559082};
    kappa_gumbel.get_parameter_constraints(fitting_sample, initials, lowers, uppers);
    CHECK_EQ(lowers[0], -10.0);
    CHECK_EQ(lowers[1], corehydro::numerics::kDoubleMachineEpsilon);
    CHECK_EQ(lowers[2], -10.0);
    CHECK_EQ(lowers[3], -2.0);
    CHECK_EQ(uppers[0], 10.0);
    CHECK_EQ(uppers[1], 100.0);
    CHECK_EQ(uppers[2], 10.0);
    CHECK_EQ(uppers[3], 2.0);
    KappaFour fitting_initial(initials[0], initials[1], initials[2], initials[3]);
    CHECK_TRUE(std::isfinite(fitting_initial.log_likelihood(fitting_sample)));
    for (double observation : fitting_sample)
        CHECK_TRUE(observation > fitting_initial.minimum() &&
                   observation < fitting_initial.maximum());
    auto invalid_sample = fitting_sample;
    invalid_sample[17] = std::numeric_limits<double>::quiet_NaN();
    const auto before_failed_fit = kappa_gumbel.get_parameters();
    CHECK_THROWS(kappa_gumbel.estimate(
        invalid_sample, ParameterEstimationMethod::MaximumLikelihood));
    CHECK_TRUE(kappa_gumbel.get_parameters() == before_failed_fit);
    CHECK_THROWS(kappa_gumbel.get_parameter_constraints(
        {1.0, 1.0, 1.0, 1.0}, initials, lowers, uppers));
}

}  // namespace

int main() {
    test_gamma_tiny_shape();
    test_gamma_log_tails();
    test_gamma_inverse();
    test_log_primitives_and_determinant();
    test_snapshot_round_trip();
    test_full_support_moment_integration();
    test_workhorse_family_log_tails();
    return chtest::summary("test_distribution_numerics_v220");
}
