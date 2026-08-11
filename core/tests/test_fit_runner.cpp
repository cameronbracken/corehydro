// ctest coverage of the shared fit runner. Oracle VALUES live in fixtures/*.json and are
// asserted by test_fixtures.cpp; this file asserts SHAPE, GUARDS and ERROR PATHS, which are
// corehydro additions with no C# counterpart and therefore have no oracle.
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "corehydro/estimation/maximum_likelihood.hpp"
#include "corehydro/estimation/optimization_method.hpp"
#include "corehydro/estimation/support/fit_runner.hpp"
#include "corehydro/models/json_lite.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "check.hpp"

namespace support = corehydro::estimation::support;
namespace spec = corehydro::models::spec;

static const std::vector<double> kPeaks = {12500, 15300, 9870, 21000, 18400,
                                           11200, 26800, 14100, 19500, 11600};

namespace {

// A minimal one-parameter ModelBase test double (the StubNormalModel precedent from
// test_model_base.cpp), needed only for the n < 2 covariance guard below. run_fit builds
// models exclusively through the univariate_distribution factory, and every factory family
// that implements IMaximumLikelihoodEstimation -- the interface UnivariateDistributionModel's
// mandatory default-parameter step requires -- has two or more parameters in this catalog;
// there is no single-parameter family reachable through the JSON model spec (Exponential is a
// 2-parameter location/scale form in this port, and single-parameter distributions such as
// Rayleigh don't implement IMaximumLikelihoodEstimation, so the vector-dataset construction
// path throws before this guard is ever reached). fill_common is the exact function run_fit
// calls internally, so driving it directly against a genuine 1-parameter model still proves
// the real guard.
class OneParamExponentialModel : public corehydro::models::ModelBase {
   public:
    explicit OneParamExponentialModel(std::vector<double> data) : data_(std::move(data)) {
        parameters().push_back(corehydro::models::ModelParameter(
            "OneParamExponentialModel", "rate", 1.0 / mean(data_), 1e-8, 10.0,
            std::make_unique<corehydro::numerics::distributions::Uniform>(1e-8, 10.0)));
    }

    void set_default_parameters() override {}
    corehydro::models::ValidationResult validate() const override { return {}; }

    double data_log_likelihood(std::vector<double>& p) const override {
        double rate = p[0];
        if (rate <= 0.0) return -std::numeric_limits<double>::infinity();
        double sum = 0.0;
        for (double x : data_) sum += x;
        return static_cast<double>(data_.size()) * std::log(rate) - rate * sum;
    }

    std::vector<double> pointwise_data_log_likelihood(const std::vector<double>& p) const override {
        double rate = p[0];
        std::vector<double> out;
        out.reserve(data_.size());
        for (double x : data_) out.push_back(std::log(rate) - rate * x);
        return out;
    }

    std::vector<corehydro::models::DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& p) const override {
        std::vector<double> pw = pointwise_data_log_likelihood(p);
        std::vector<corehydro::models::DataComponent> out;
        out.reserve(pw.size());
        for (std::size_t i = 0; i < pw.size(); ++i)
            out.emplace_back(static_cast<int>(i), pw[i], data_[i]);
        return out;
    }

   private:
    static double mean(const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x;
        return s / static_cast<double>(v.size());
    }
    std::vector<double> data_;
};

}  // namespace

static void test_mle_shape() {
    std::string construct =
        R"({"model":{"family":"Normal","dataset":"peaks"},"optimizer":"NelderMead"})";
    support::FitResult r = support::run_fit("MaximumLikelihood", construct, kPeaks);

    CHECK_TRUE(r.method == "MaximumLikelihood");
    CHECK_TRUE(r.parameters.size() == 2);
    CHECK_TRUE(r.parameter_names.size() == 2);
    CHECK_TRUE(!r.parameter_names[0].empty());
    CHECK_TRUE(r.converged);
    CHECK_TRUE(r.status == "Success");
    CHECK_TRUE(r.function_evaluations > 0);
    CHECK_TRUE(r.nobs == 10);
    CHECK_TRUE(std::isfinite(r.log_likelihood));
    CHECK_TRUE(std::isfinite(r.aic));
    CHECK_TRUE(std::isfinite(r.bic));
    // Hessian stack present and square for a 2-parameter model.
    CHECK_TRUE(r.covariance.size() == 4);
    CHECK_TRUE(r.standard_errors.size() == 2);
    CHECK_TRUE(r.correlation.size() == 4);
    // The fitted spec round-trips: it carries the fitted values.
    CHECK_TRUE(r.model_spec.find("parameter_values") != std::string::npos);
}

static void test_hessian_can_be_disabled() {
    std::string construct = R"({"model":{"family":"Normal","dataset":"peaks"},)"
                            R"("optimizer":"NelderMead","hessian":false})";
    support::FitResult r = support::run_fit("MaximumLikelihood", construct, kPeaks);
    CHECK_TRUE(r.covariance.empty());
    CHECK_TRUE(r.standard_errors.empty());
    CHECK_TRUE(r.correlation.empty());
    CHECK_TRUE(std::isfinite(r.log_likelihood));  // the fit itself still happened
}

static void test_single_parameter_covariance_is_nan_not_zero() {
    // fill_common's n < 2 guard: C# GetCovarianceMatrix throws below two parameters, so the
    // honest report is NaN, NOT the silent zeros the fixture glue used to return. See
    // OneParamExponentialModel's header note for why this drives fill_common directly instead
    // of going through run_fit's JSON model spec.
    OneParamExponentialModel model(kPeaks);
    corehydro::estimation::MaximumLikelihood e(model,
                                               corehydro::estimation::OptimizationMethod::Brent);
    e.set_compute_hessian(true);
    CHECK_TRUE(e.estimate());

    support::FitResult r;
    r.method = "MaximumLikelihood";
    support::fill_common(r, e, model, /*want_hessian=*/true);
    CHECK_TRUE(r.parameters.size() == 1);
    CHECK_TRUE(r.covariance.size() == 1);
    CHECK_TRUE(std::isnan(r.covariance[0]));
    CHECK_TRUE(std::isnan(r.standard_errors[0]));
    CHECK_TRUE(std::isnan(r.correlation[0]));
}

static void test_map_reports_status() {
    std::string construct =
        R"({"model":{"family":"Normal","dataset":"peaks"},"optimizer":"NelderMead"})";
    support::FitResult r = support::run_fit("MaximumAPosteriori", construct, kPeaks);
    CHECK_TRUE(r.method == "MaximumAPosteriori");
    CHECK_TRUE(r.status == "Success");
    CHECK_TRUE(std::isfinite(r.prior_log_likelihood));
}

static void test_unknown_target_throws() {
    bool threw = false;
    try {
        support::run_fit("NotAnEstimator", R"({"model":{"family":"Normal"}})", kPeaks);
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("NotAnEstimator") != std::string::npos;
    }
    CHECK_TRUE(threw);
}

static void test_unknown_optimizer_throws_naming_it() {
    bool threw = false;
    try {
        support::run_fit("MaximumLikelihood",
                         R"({"model":{"family":"Normal","dataset":"peaks"},)"
                         R"("optimizer":"Simplexx"})",
                         kPeaks);
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("Simplexx") != std::string::npos;
    }
    CHECK_TRUE(threw);
}

// fitted_spec must not duplicate `parameter_values` when the caller's construct already
// carries one (e.g. as a starting/fixed value): the fitted values must win, and there must be
// exactly one such key so a caller re-parsing model_spec reads the FITTED values back, not the
// stale pre-fit input. See the review finding this covers: JsonValue::at() returns the first
// match by insertion order, so a duplicated key would silently resurrect the input values.
static void test_fitted_spec_does_not_duplicate_parameter_values() {
    std::string construct = R"({"model":{"family":"Normal","dataset":"peaks",)"
                            R"("parameter_values":[10000,1000]},"optimizer":"NelderMead"})";
    support::FitResult r = support::run_fit("MaximumLikelihood", construct, kPeaks);

    // Exactly one parameter_values key in the returned spec.
    std::size_t first = r.model_spec.find("\"parameter_values\"");
    CHECK_TRUE(first != std::string::npos);
    std::size_t second = r.model_spec.find("\"parameter_values\"", first + 1);
    CHECK_TRUE(second == std::string::npos);

    // It reads back as the FITTED values, not the input [10, 2].
    spec::JsonValue parsed = spec::parse_json(r.model_spec);
    std::vector<double> readback = parsed.at("parameter_values").as_double_vector();
    CHECK_TRUE(readback.size() == r.parameters.size());
    for (std::size_t i = 0; i < readback.size(); ++i) CHECK_NEAR(readback[i], r.parameters[i], 1e-12);
    // Sanity: the input values were a poor starting guess far from the fit, so a bug that
    // resurrects them would fail the readback-equals-fitted check above.
    CHECK_TRUE(std::abs(readback[0] - 10000.0) > 1.0 || std::abs(readback[1] - 1000.0) > 1.0);
}

// json_lite.hpp's to_json_string is new surface added by this task; prove a double round-trips
// bit-exactly through parse_json(to_json_string(v)) at %.17g precision.
static void test_json_round_trip() {
    std::string original = R"({"family":"Normal","parameter_values":[12345.6789012345,-0.1]})";
    spec::JsonValue v = spec::parse_json(original);
    std::string reserialized = spec::to_json_string(v);
    spec::JsonValue v2 = spec::parse_json(reserialized);
    CHECK_TRUE(v2.at("parameter_values").items()[0].as_double() ==
              v.at("parameter_values").items()[0].as_double());
    CHECK_TRUE(v2.at("parameter_values").items()[1].as_double() ==
              v.at("parameter_values").items()[1].as_double());
    CHECK_TRUE(v2.at("family").as_string() == "Normal");
}

int main() {
    test_mle_shape();
    test_hessian_can_be_disabled();
    test_single_parameter_covariance_is_nan_not_zero();
    test_map_reports_status();
    test_fitted_spec_does_not_duplicate_parameter_values();
    test_unknown_target_throws();
    test_unknown_optimizer_throws_naming_it();
    test_json_round_trip();
    return chtest::summary("fit_runner");
}
