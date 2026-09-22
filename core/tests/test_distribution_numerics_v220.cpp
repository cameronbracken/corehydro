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
