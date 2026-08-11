// Structural tests for the model-spec `parameters` override block and the
// `use_default_flat_priors` key (corehydro/models/model_spec.hpp).
//
// The block is a corehydro ADDITION to the shared fixture spec builder, not a port of a C#
// class: it just writes onto the ModelParameter elements a model already built (bounds, fixed
// flag, prior distribution, starting value), so what needs proving here is the wiring and the
// ordering contract, not a numeric oracle. The numeric consequences (a changed
// prior_log_likelihood under a caller-supplied prior) are pinned in fixtures/ and reproduced by
// the dotnet gate.
#include <string>
#include <vector>

#include "corehydro/models/model_spec.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::models::ModelBase;
using corehydro::models::spec::build_model_from_json;
using corehydro::numerics::distributions::Normal;

namespace {

const std::vector<double> kSample = {12.0, 15.0, 9.0, 22.0, 18.0, 14.0, 10.0, 28.0, 17.0, 11.0};

// Bounds, the fixed flag, the name label, and a starting value all land on the element.
void test_overrides_land_on_parameters() {
    std::unique_ptr<ModelBase> m = build_model_from_json(
        R"({"family": "Normal", "dataset": "d",
            "parameters": [{"index": 1, "lower": 0.5, "upper": 40.0, "is_fixed": true,
                            "name": "sigma", "value": 7.25}]})",
        kSample);

    const auto& p = m->parameters();
    CHECK_EQ(static_cast<int>(p.size()), 2);
    CHECK_NEAR(p[1].lower_bound(), 0.5, 1e-12);
    CHECK_NEAR(p[1].upper_bound(), 40.0, 1e-12);
    CHECK_TRUE(p[1].is_fixed());
    CHECK_EQ(p[1].name(), std::string("sigma"));
    CHECK_NEAR(p[1].value(), 7.25, 1e-12);
}

// A caller-supplied prior replaces the model's default and changes prior_log_likelihood by
// exactly the difference in log-densities.
void test_prior_override_changes_prior_log_likelihood() {
    const std::string base = R"({"family": "Normal", "dataset": "d")";
    std::unique_ptr<ModelBase> plain = build_model_from_json(base + "}", kSample);

    std::unique_ptr<ModelBase> primed = build_model_from_json(
        base + R"(, "use_default_flat_priors": false,
                   "parameters": [{"index": 0,
                                   "prior": {"family": "Normal", "parameters": [15.0, 3.0]}}]})",
        kSample);

    CHECK_TRUE(plain->use_default_flat_priors());
    CHECK_TRUE(!primed->use_default_flat_priors());

    std::vector<double> at = {15.6, 5.9};
    std::vector<double> at_copy = at;
    double plain_ll = plain->prior_log_likelihood(at);
    double primed_ll = primed->prior_log_likelihood(at_copy);

    // Only parameter 0's prior moved, so the shift is that one term's log-density difference.
    double replaced = Normal(15.0, 3.0).log_pdf(at[0]);
    double original = plain->parameters()[0].prior_distribution().log_pdf(at[0]);
    CHECK_NEAR(primed_ll - plain_ll, replaced - original, 1e-12);
}

// `parameter_values` is the sync-safe whole-vector setter and must win over a per-parameter
// `value` in the same spec.
void test_parameter_values_wins_over_per_parameter_value() {
    std::unique_ptr<ModelBase> m = build_model_from_json(
        R"({"family": "Normal", "dataset": "d",
            "parameters": [{"index": 0, "value": 1.0}],
            "parameter_values": [16.0, 6.0]})",
        kSample);

    CHECK_NEAR(m->parameters()[0].value(), 16.0, 1e-12);
    CHECK_NEAR(m->parameters()[1].value(), 6.0, 1e-12);
}

// Indexes are resolved against the model's own parameter vector, so a trend that widens the
// layout shifts what a given index means; an index past the end is an error, not a silent skip.
void test_index_is_validated_against_the_built_layout() {
    // Stationary Normal: 2 parameters, so index 2 is out of range.
    CHECK_THROWS(build_model_from_json(
        R"({"family": "Normal", "dataset": "d", "parameters": [{"index": 2, "value": 1.0}]})",
        kSample));
    CHECK_THROWS(build_model_from_json(
        R"({"family": "Normal", "dataset": "d", "parameters": [{"index": -1, "value": 1.0}]})",
        kSample));

    // A linear trend on the location parameter widens the vector, so index 2 becomes valid and
    // the override applies after the trend layout is final.
    std::unique_ptr<ModelBase> trended = build_model_from_json(
        R"({"family": "Normal", "dataset": "d",
            "trends": [{"parameter": 0, "type": "Linear"}],
            "parameters": [{"index": 2, "value": 3.5}]})",
        kSample);
    CHECK_TRUE(trended->number_of_parameters() > 2);
    CHECK_NEAR(trended->parameters()[2].value(), 3.5, 1e-12);
}

// The block is optional everywhere: a spec without it builds exactly as before.
void test_block_is_optional() {
    std::unique_ptr<ModelBase> m =
        build_model_from_json(R"({"family": "Normal", "dataset": "d"})", kSample);
    CHECK_EQ(static_cast<int>(m->parameters().size()), 2);
    CHECK_TRUE(m->use_default_flat_priors());
}

}  // namespace

int main() {
    test_overrides_land_on_parameters();
    test_prior_override_changes_prior_log_likelihood();
    test_parameter_values_wins_over_per_parameter_value();
    test_index_is_validated_against_the_built_layout();
    test_block_is_optional();
    return chtest::summary("model_spec_parameters");
}
