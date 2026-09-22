#include <cmath>
#include <string>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/data/correlation.hpp"
#include "corehydro/numerics/distributions/copulas/base/bivariate_copula_estimation.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_factory.hpp"
#include "corehydro/numerics/distributions/copulas/frank_copula.hpp"
#include "corehydro/numerics/distributions/copulas/joe_copula.hpp"

namespace cop = corehydro::numerics::distributions::copulas;

namespace {

void test_independence() {
    auto c = cop::create_copula("Independence");
    CHECK_EQ(c->number_of_copula_parameters(), 0);
    CHECK_TRUE(c->get_copula_parameters().empty());
    CHECK_NEAR(c->pdf(0.2, 0.8), 1.0, 0.0);
    CHECK_NEAR(c->cdf(0.2, 0.8), 0.16, 1e-15);
    CHECK_NEAR(c->conditional_cdf(0.2, 0.8), 0.8, 0.0);
    CHECK_NEAR(c->inverse_conditional_cdf(0.2, 0.8), 0.8, 0.0);
    CHECK_NEAR(c->inverse_cdf(0.2, 0.8)[1], 0.8, 0.0);
    CHECK_NEAR(c->lower_tail_dependence(), 0.0, 0.0);
    CHECK_NEAR(c->upper_tail_dependence(), 0.0, 0.0);
    cop::estimate(*c, {1.0, 2.0}, {2.0, 1.0}, cop::CopulaEstimationMethod::PseudoLikelihood);
    CHECK_TRUE(c->get_copula_parameters().empty());
    CHECK_TRUE(c->clone()->type() == cop::CopulaType::Independence);
}

void test_conditional_round_trips() {
    const std::vector<std::string> families = {"AliMikhailHaq", "Clayton", "Frank", "Gumbel",
                                               "Joe", "Normal", "StudentT", "Independence"};
    for (const auto& family : families) {
        auto c = cop::create_copula(family);
        if (family == "AliMikhailHaq" || family == "Normal") c->set_copula_parameters({0.5});
        if (family == "StudentT") c->set_copula_parameters({0.5, 5.0});
        double v = c->inverse_conditional_cdf(0.37, 0.63);
        CHECK_NEAR(c->conditional_cdf(0.37, v), 0.63, 2e-8);
        CHECK_NEAR(c->inverse_cdf(0.37, 0.63)[1], v, 0.0);
        if (family != "StudentT") {
            CHECK_NEAR(c->conditional_cdf(0.37, 0.0), 0.0, 1e-14);
            CHECK_NEAR(c->conditional_cdf(0.37, 1.0), 1.0, 1e-14);
        }
    }
    auto gumbel = cop::create_copula("Gumbel");
    auto joe = cop::create_copula("Joe");
    CHECK_NEAR(gumbel->inverse_conditional_cdf(0.37, 1.0), 1.0, 0.0);
    CHECK_NEAR(joe->inverse_conditional_cdf(0.37, 1.0), 1.0, 0.0);
}

void test_upstream_conditional_pins() {
    auto check = [](const std::string& family, const std::vector<double>& parameters,
                    double expected, double tolerance) {
        auto c = cop::create_copula(family);
        c->set_copula_parameters(parameters);
        CHECK_NEAR(c->conditional_cdf(0.3, 0.7), expected, tolerance);
    };
    check("AliMikhailHaq", {0.5}, 0.742798289691333, 1e-12);
    check("Clayton", {2.0}, 0.8743161176077272, 1e-10);
    check("Frank", {4.0}, 0.8693978167836881, 1e-10);
    check("Gumbel", {2.0}, 0.9104803864754554, 1e-10);
    check("Joe", {2.5}, 0.9123991248836704, 1e-10);
    check("Normal", {0.5}, 0.8181370471246912, 1e-10);
    check("StudentT", {0.5, 5.0}, 0.8285717360290429, 1e-8);
    check("Independence", {}, 0.7, 0.0);
}

void test_tau_fits() {
    const std::vector<double> x = {1, 2, 3, 4, 5, 6};
    const std::vector<double> y = {1, 3, 2, 5, 4, 6};
    double tau = corehydro::numerics::data::kendalls_tau(x, y);

    cop::FrankCopula frank;
    frank.set_theta_from_tau(x, y);
    CHECK_NEAR(cop::FrankCopula::kendalls_tau_from_theta(frank.theta()), tau, 2e-10);

    cop::JoeCopula joe;
    joe.set_theta_from_tau(x, y);
    CHECK_NEAR(cop::JoeCopula::kendalls_tau_from_theta(joe.theta()), tau, 2e-10);
    CHECK_THROWS(joe.set_theta_from_tau(x, {6, 5, 4, 3, 2, 1}));
}

}  // namespace

int main() {
    test_independence();
    test_conditional_round_trips();
    test_upstream_conditional_pins();
    test_tau_fits();
    return chtest::summary("test_copulas_v220");
}
