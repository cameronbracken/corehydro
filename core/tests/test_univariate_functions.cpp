// Transcribed C# oracle tests for the Numerics univariate Functions layer:
//   upstream/Numerics/Test_Numerics/Functions/*.cs @ 7e8e8d1
//
// Six of the seven Test_Functions.cs methods are transcribed here: Test_Linear_Function,
// Test_Linear_Function_Inverse, Test_Power_Function, Test_Power_Function_Inverse,
// Test_InversePower_Function, Test_InversePower_Function_Inverse.
//
// TabularFunction remains tested with the paired-data subsystem. This file adds the v2.2.0
// segmented, composite, ensemble, and JSON factory cases beside the original linear and power
// checks.
//
// The C# test file mixes computed-expected literals (e.g. `double valid1 = (5*6) + -2;`) and
// hand-transcribed decimal literals found via R's norminv() (e.g. `30.0234692505882`); both
// forms are transcribed verbatim below, matching each source line.
#include <cmath>
#include <memory>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/functions/composite_function.hpp"
#include "corehydro/numerics/functions/ensemble_function.hpp"
#include "corehydro/numerics/functions/linear_function.hpp"
#include "corehydro/numerics/functions/power_function.hpp"
#include "corehydro/numerics/functions/segmented_power_function.hpp"
#include "corehydro/numerics/functions/univariate_function_factory.hpp"
#include "check.hpp"

using corehydro::numerics::functions::LinearFunction;
using corehydro::numerics::functions::PowerFunction;
using corehydro::numerics::functions::CompositeFunction;
using corehydro::numerics::functions::CompositeFunctionMode;
using corehydro::numerics::functions::EnsembleFunction;
using corehydro::numerics::functions::IUnivariateFunction;
using corehydro::numerics::functions::SegmentedPowerFunction;

namespace {

void test_linear_function() {
    // Default constructor with alpha = 0 and beta = 1.
    LinearFunction func0;
    double y0 = func0.function(6);
    double valid0 = 6;
    CHECK_NEAR(y0, valid0, 1e-6);

    double alpha = -2;
    double beta = 5;
    double sigma = 3;

    LinearFunction func1(alpha, beta);
    double y1 = func1.function(6);
    double valid1 = (5 * 6) + -2;
    CHECK_NEAR(y1, valid1, 1e-6);

    LinearFunction func2(alpha, beta, sigma);
    double y2 = func2.function(6);
    CHECK_NEAR(y2, valid1, 1e-6);

    func2.set_confidence_level(0.75);
    double y3 = func2.function(6);
    // Found using R's norminv() function for epsilon from the "drcarlate" package.
    double valid3 = 30.0234692505882;
    CHECK_NEAR(y3, valid3, 1e-6);
}

void test_linear_function_inverse() {
    LinearFunction func(10, 0.5, 20);
    double y = func.function(400);
    double x = func.inverse_function(y);
    CHECK_NEAR(x, 400, 1e-6);

    func.set_confidence_level(0.75);
    double yy = func.function(400);
    double xx = func.inverse_function(yy);
    CHECK_NEAR(xx, 400, 1e-6);
}

void test_power_function() {
    // Default constructor with alpha = 1, beta = 1.5, and xi = 0.
    PowerFunction func0;
    double y0 = func0.function(6);
    double valid0 = 1 * std::pow(6 - 0, 1.5);
    CHECK_NEAR(y0, valid0, 1e-6);

    double alpha = 5;
    double beta = 2;
    double sigma = 3;
    double xi = 0;

    PowerFunction func1(alpha, beta, xi);
    double y1 = func1.function(6);
    double valid1 = alpha * std::pow(6 - xi, beta);
    CHECK_NEAR(y1, valid1, 1e-6);

    PowerFunction func2(alpha, beta, xi, sigma);
    double y2 = func2.function(6);
    CHECK_NEAR(y2, valid1, 1e-6);

    func2.set_confidence_level(0.75);
    double y3 = func2.function(6);
    // Found using R's norminv() function for epsilon from the "drcarlate" package.
    double valid3 = 1361.61408399941;
    CHECK_NEAR(y3, valid3, 1e-6);
}

void test_power_function_inverse() {
    PowerFunction func(10, 2, 0, 0.1);
    double y = func.function(400);
    double x = func.inverse_function(y);
    CHECK_NEAR(x, 400, 1e-6);

    func.set_confidence_level(0.75);
    double yy = func.function(400);
    double xx = func.inverse_function(yy);
    CHECK_NEAR(xx, 400, 1e-6);
}

void test_inverse_power_function() {
    double alpha = 5;
    double beta = 2;
    double sigma = 3;
    double xi = 0;

    PowerFunction func(alpha, beta, xi);
    func.set_is_inverse(true);
    double y = func.function(6);
    double valid = std::sqrt(6.0 / alpha) + xi;
    CHECK_NEAR(y, valid, 1e-6);

    PowerFunction func2(alpha, beta, xi, sigma);
    func2.set_is_inverse(true);
    double y2 = func2.function(6);
    CHECK_NEAR(y2, valid, 1e-6);

    func2.set_confidence_level(0.75);
    double y3 = func2.function(6);
    // Found using R's norminv() function for epsilon from the "drcarlate" package.
    double valid3 = 0.398290417772997;
    CHECK_NEAR(y3, valid3, 1e-6);
}

void test_inverse_power_function_inverse() {
    PowerFunction func(10, 2, 0, 0.1);
    func.set_is_inverse(true);
    double y = func.function(6);
    double x = func.inverse_function(y);
    CHECK_NEAR(x, 6, 1e-6);

    func.set_confidence_level(0.75);
    double yy = func.function(6);
    double xx = func.inverse_function(yy);
    CHECK_NEAR(xx, 6, 1e-6);
}

void test_segmented_power_function() {
    SegmentedPowerFunction one({1.0, 1.5, 2.0, 0.1});
    one.set_is_deterministic(true);
    CHECK_NEAR(one.function(0.5), 0.0, 0.0);
    CHECK_NEAR(one.function(5.0), 505.9644256269407, 1E-10);

    SegmentedPowerFunction two({1.0, 1.5, 2.0, 3.0, 1.2, 1.5, 0.1});
    two.set_is_deterministic(true);
    CHECK_NEAR(two.function(2.5), 71.15124735378853, 1E-10);
    CHECK_NEAR(two.function(5.0), 550.7919745807667, 1E-10);
    CHECK_NEAR(two.inverse_function(two.function(5.0)), 5.0, 1E-6);
    CHECK_THROWS(SegmentedPowerFunction(0));
}

std::vector<std::unique_ptr<IUnivariateFunction>> two_linear_children() {
    std::vector<std::unique_ptr<IUnivariateFunction>> children;
    children.push_back(std::make_unique<LinearFunction>(0.0, 2.0));
    children.push_back(std::make_unique<LinearFunction>(10.0, 4.0));
    return children;
}

void test_composite_function() {
    CompositeFunction average(two_linear_children(), {0.25, 0.75});
    CHECK_NEAR(average.function(4.0), 21.5, 1E-12);
    CHECK_NEAR(average.inverse_function(average.function(5.0)), 5.0, 1E-8);
    CHECK_THROWS(CompositeFunction(two_linear_children(), {0.3, 0.3}));

    std::vector<std::unique_ptr<IUnivariateFunction>> mixture_children;
    mixture_children.push_back(std::make_unique<LinearFunction>(0.0, 1.0));
    mixture_children.push_back(std::make_unique<LinearFunction>(100.0, 1.0));
    CompositeFunction mixture(std::move(mixture_children), {0.25, 0.75});
    mixture.set_mode(CompositeFunctionMode::Mixture);
    mixture.set_confidence_level(0.25);
    CHECK_NEAR(mixture.function(5.0), 5.0, 1E-12);
    mixture.set_confidence_level(0.75);
    CHECK_NEAR(mixture.function(5.0), 105.0, 1E-12);
}

void test_ensemble_and_json_factory() {
    auto function_template =
        std::make_unique<SegmentedPowerFunction>(std::vector<double>{1.0, 1.5, 2.0, 0.1});
    EnsembleFunction ensemble(std::move(function_template),
                              {{1.0, 1.5, 2.0, 0.1}, {0.9, 1.6, 1.9, 0.12},
                               {1.1, 1.4, 2.1, 0.08}});
    auto first = ensemble.sample_at(1);
    first->set_parameters({8.0, 1.6, 1.9, 0.12});
    auto second = ensemble.sample_at(1);
    CHECK_NEAR(dynamic_cast<SegmentedPowerFunction&>(*second).breakpoint(1), 0.9, 0.0);
    auto last = ensemble.sample(1.0);
    CHECK_NEAR(dynamic_cast<SegmentedPowerFunction&>(*last).breakpoint(1), 1.1, 0.0);
    CHECK_THROWS(ensemble.sample_at(3));

    auto spec = corehydro::models::spec::JsonParser::parse(
        R"({"type":"composite","mode":"weighted_average","weights":[0.25,0.75],"functions":[{"type":"linear","parameters":[0,2,0]},{"type":"linear","parameters":[10,4,0]}]})");
    auto built = corehydro::numerics::functions::build_function_spec(spec);
    CHECK_NEAR(built->function(4.0), 21.5, 1E-12);
}

}  // namespace

int main() {
    test_linear_function();
    test_linear_function_inverse();
    test_power_function();
    test_power_function_inverse();
    test_inverse_power_function();
    test_inverse_power_function_inverse();
    test_segmented_power_function();
    test_composite_function();
    test_ensemble_and_json_factory();
    return chtest::summary("test_univariate_functions");
}
