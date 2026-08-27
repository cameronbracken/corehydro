// P5 Task 11 -- GeneralizedLinearModel.
//
// Transcribes all six [TestMethod]s of
// upstream/Numerics/Test_Numerics/Machine Learning/Supervised/Test_GeneralizedLinearModel.cs
// @ 2a0357a. Every expected value is R `glm` output, so these are correctness oracles against a
// reference implementation, not merely C#-reproduction pins.
//
// Note the tolerances are taken exactly as written, including Test_Logistic's asymmetric pair
// (parameters at 1E-2 but standard errors loosened to 1E-1) -- reproduced rather than flattened.
//
// `Summary()` is a documented severance (presentation-only text); `parameter_names()`, the one
// real member it reads, is tested below.
#include <cmath>
#include <string>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/machine_learning/supervised/generalized_linear_model.hpp"
#include "data/fpp3_dataset.hpp"
#include "data/glm_dataset.hpp"

namespace ml = corehydro::numerics::machine_learning;
namespace la = corehydro::numerics::math::linalg;
namespace fn = corehydro::numerics::functions;
namespace opt = corehydro::numerics::math::optimization;
namespace fpp3 = corehydro::testdata::fpp3;
namespace glm = corehydro::testdata::glm;

namespace {

// --- Transcribed from Test_GeneralizedLinearModel.cs --------------------------------------

void test_simple_linear_regression() {
    la::Vector y(fpp3::kConsumption);
    la::Matrix x = la::Matrix::from_columns({fpp3::kIncome});
    ml::GeneralizedLinearModel model(x, y);
    model.train();

    CHECK_NEAR(model.parameters()[0], 0.54510, 1e-3);
    CHECK_NEAR(model.parameters()[1], 0.28060, 1e-3);
    CHECK_NEAR(model.parameter_standard_errors()[0], 0.05569, 1e-3);
    CHECK_NEAR(model.parameter_standard_errors()[1], 0.04744, 1e-3);
    CHECK_NEAR(model.standard_error(), 0.6026, 1e-3);
    CHECK_EQ(model.degrees_of_freedom(), 185);
}

void test_multiple_linear_regression() {
    la::Vector y(fpp3::kConsumption);
    la::Matrix x = la::Matrix::from_columns(
        {fpp3::kIncome, fpp3::kProduction, fpp3::kSavings, fpp3::kUnemployment});

    ml::GeneralizedLinearModel model(x, y);
    model.train();

    const double true_par[5] = {0.26729, 0.71449, 0.04589, -0.04527, -0.20477};
    const double true_sig[5] = {0.03721, 0.04219, 0.02588, 0.00278, 0.10550};
    CHECK_EQ(static_cast<int>(model.parameters().size()), 5);
    for (int i = 0; i < 5; i++) {
        CHECK_NEAR(model.parameters()[static_cast<std::size_t>(i)], true_par[i], 1e-3);
        CHECK_NEAR(model.parameter_standard_errors()[static_cast<std::size_t>(i)], true_sig[i],
                   1e-3);
    }
    CHECK_NEAR(model.standard_error(), 0.3286, 1e-3);
    CHECK_EQ(model.degrees_of_freedom(), 182);
}

void test_log() {
    la::Vector y(glm::kDeaths);
    la::Matrix x = la::Matrix::from_columns({glm::kDrivers, glm::kPopden});

    ml::GeneralizedLinearModel model(x, y, true, fn::LinkFunctionType::Log);
    model.train();

    const double true_par[3] = {6.322e+00, 2.405e-03, -1.480e-04};
    const double true_sig[3] = {9.382e-03, 1.885e-05, 8.163e-06};
    for (int i = 0; i < 3; i++) {
        CHECK_NEAR(model.parameters()[static_cast<std::size_t>(i)], true_par[i], 1e-2);
        CHECK_NEAR(model.parameter_standard_errors()[static_cast<std::size_t>(i)], true_sig[i],
                   1e-2);
    }
    CHECK_NEAR(model.aic(), 4069.4, 1e-2);
    CHECK_EQ(model.degrees_of_freedom(), 23);
}

void test_logistic() {
    la::Vector y(glm::kAdmits);
    la::Matrix x = la::Matrix::from_columns({glm::kGre, glm::kGpa, glm::kRank});

    ml::GeneralizedLinearModel model(x, y, true, fn::LinkFunctionType::Logit);
    model.train();

    const double true_par[4] = {-3.449548, 0.002294, 0.777014, -0.560031};
    const double true_sig[4] = {1.132846, 0.001092, 0.327484, 0.127137};
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(model.parameters()[static_cast<std::size_t>(i)], true_par[i], 1e-2);
        // The C# source loosens the standard errors to 1E-1 for this test only.
        CHECK_NEAR(model.parameter_standard_errors()[static_cast<std::size_t>(i)], true_sig[i],
                   1e-1);
    }
    CHECK_NEAR(model.aic(), 467.44, 1e-2);
    CHECK_EQ(model.degrees_of_freedom(), 396);
}

void test_probit() {
    la::Vector y(glm::kAdmits);
    la::Matrix x = la::Matrix::from_columns({glm::kGre, glm::kGpa, glm::kRank});

    ml::GeneralizedLinearModel model(x, y, true, fn::LinkFunctionType::Probit);
    model.train();

    const double true_par[4] = {-2.0915037, 0.0013982, 0.4643598, -0.3317117};
    const double true_sig[4] = {0.6718360, 0.0006487, 0.1950263, 0.0745524};
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(model.parameters()[static_cast<std::size_t>(i)], true_par[i], 1e-2);
        CHECK_NEAR(model.parameter_standard_errors()[static_cast<std::size_t>(i)], true_sig[i],
                   1e-2);
    }
    CHECK_NEAR(model.aic(), 467.48, 1e-2);
    CHECK_EQ(model.degrees_of_freedom(), 396);
}

void test_log_log() {
    la::Vector y(glm::kAdmits);
    la::Matrix x = la::Matrix::from_columns({glm::kGre, glm::kGpa, glm::kRank});

    ml::GeneralizedLinearModel model(x, y, true, fn::LinkFunctionType::ComplementaryLogLog);
    model.train();

    const double true_par[4] = {-3.0603413, 0.0017513, 0.6213458, -0.4587797};
    const double true_sig[4] = {0.9223750, 0.0008744, 0.2626932, 0.1014823};
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(model.parameters()[static_cast<std::size_t>(i)], true_par[i], 1e-2);
        CHECK_NEAR(model.parameter_standard_errors()[static_cast<std::size_t>(i)], true_sig[i],
                   1e-2);
    }
    CHECK_NEAR(model.aic(), 467.5, 1e-2);
    CHECK_EQ(model.degrees_of_freedom(), 396);
}

// --- COREHYDRO SUPPLEMENT (no C# counterpart) ---------------------------------------------

void test_parameter_names_and_shape() {
    la::Vector y(fpp3::kConsumption);
    la::Matrix x = la::Matrix::from_columns({fpp3::kIncome, fpp3::kProduction});
    ml::GeneralizedLinearModel model(x, y);

    // The C# default names, including the UTF-8 beta. (`Summary()` is severed; these are not.)
    CHECK_EQ(static_cast<int>(model.parameter_names().size()), 3);
    CHECK_TRUE(model.parameter_names()[0] == "Intercept");
    CHECK_TRUE(model.parameter_names()[1] == "\xce\xb2" "1");
    CHECK_TRUE(model.parameter_names()[2] == "\xce\xb2" "2");

    // Without an intercept there is no "Intercept" name and one fewer parameter.
    ml::GeneralizedLinearModel no_int(x, y, false);
    CHECK_EQ(static_cast<int>(no_int.parameter_names().size()), 2);
    CHECK_TRUE(no_int.parameter_names()[0] == "\xce\xb2" "1");
    CHECK_EQ(no_int.x().number_of_columns(), 2);
    CHECK_EQ(model.x().number_of_columns(), 3);
    CHECK_EQ(no_int.degrees_of_freedom(), 185);
    CHECK_EQ(model.degrees_of_freedom(), 184);

    CHECK_EQ(model.sample_size(), 187);
    CHECK_TRUE(model.has_intercept());
    CHECK_TRUE(!no_int.has_intercept());
    CHECK_TRUE(!model.use_robust_se());
}

void test_predict_and_intervals() {
    la::Vector y(fpp3::kConsumption);
    la::Matrix x = la::Matrix::from_columns({fpp3::kIncome});
    ml::GeneralizedLinearModel model(x, y);
    model.train();

    // predict() accepts BOTH the p-column design matrix and the (p-1)-column raw matrix when
    // there is an intercept -- the PrepareDesignMatrix contract.
    la::Matrix raw(3, 1, {0.0, 1.0, 2.0});
    std::vector<double> p_raw = model.predict(raw);
    la::Matrix designed(3, 2, {1.0, 0.0, 1.0, 1.0, 1.0, 2.0});
    std::vector<double> p_designed = model.predict(designed);
    CHECK_EQ(static_cast<int>(p_raw.size()), 3);
    for (std::size_t i = 0; i < 3; i++) CHECK_EQ(p_raw[i], p_designed[i]);
    // Identity link: the prediction is the linear predictor.
    for (std::size_t i = 0; i < 3; i++)
        CHECK_NEAR(p_raw[i], model.parameters()[0] + model.parameters()[1] * static_cast<double>(i),
                   1e-12);

    // Any other column count is rejected with the C# message.
    CHECK_THROWS_MSG(model.predict(la::Matrix(3, 5, std::vector<double>(15, 1.0))),
                     "Expected 2 columns");

    // The interval table is n-by-3, ordered lower/mean/upper, and symmetric about the mean.
    la::Matrix pi = model.predict_intervals(raw, 0.1);
    CHECK_EQ(pi.number_of_rows(), 3);
    CHECK_EQ(pi.number_of_columns(), 3);
    for (int i = 0; i < 3; i++) {
        CHECK_TRUE(pi(i, 0) < pi(i, 1));
        CHECK_TRUE(pi(i, 1) < pi(i, 2));
        CHECK_EQ(pi(i, 1), p_raw[static_cast<std::size_t>(i)]);
        CHECK_NEAR(pi(i, 1) - pi(i, 0), pi(i, 2) - pi(i, 1), 1e-12);
    }
    // A wider alpha gives a narrower band.
    la::Matrix pi50 = model.predict_intervals(raw, 0.5);
    for (int i = 0; i < 3; i++) CHECK_TRUE(pi50(i, 2) - pi50(i, 0) < pi(i, 2) - pi(i, 0));
}

void test_residuals_and_information_criteria() {
    la::Vector y(fpp3::kConsumption);
    la::Matrix x = la::Matrix::from_columns({fpp3::kIncome});
    ml::GeneralizedLinearModel model(x, y);
    model.train();

    // The residuals are y - fitted, and the model standard error is sqrt(SSE / (n - p)).
    std::vector<double> fitted = model.predict(model.x());
    CHECK_EQ(static_cast<int>(model.residuals().size()), 187);
    double sse = 0.0;
    for (std::size_t i = 0; i < model.residuals().size(); i++) {
        CHECK_NEAR(model.residuals()[i], y[static_cast<int>(i)] - fitted[i], 1e-12);
        sse += model.residuals()[i] * model.residuals()[i];
    }
    CHECK_NEAR(model.standard_error(), std::sqrt(sse / (187 - 2)), 1e-12);

    // AIC < AICc, and BIC exceeds AIC at this sample size (log(187) > 2).
    CHECK_TRUE(model.aic() < model.aicc());
    CHECK_TRUE(model.bic() > model.aic());

    // The three information criteria, MEASURED against the real library for this exact fit.
    //
    // DO NOT take these from the commented-out `Summary()` transcript in the C# test file: that
    // block reads "AIC: 71.1801  AICc: 71.2453  BIC: 77.6423", which the shipped library does NOT
    // produce -- it returns the values below, 272.08 higher. The transcript is stale commentary
    // (it is inside a `/* ... */` after a `Debug.WriteLine` loop, never asserted), so nothing in
    // the C# suite fails when it drifts. Only the ASSERTED values in that file are oracles.
    CHECK_NEAR(model.aic(), 343.25605266374168, 1e-9);
    CHECK_NEAR(model.aicc(), 343.32127005504606, 1e-9);
    CHECK_NEAR(model.bic(), 349.71826989745085, 1e-9);

    // While measuring those, the same probe pinned the fit itself far tighter than the R-rounded
    // C# assertions do. These are the library's own values to full precision.
    CHECK_NEAR(model.parameters()[0], 0.5450732441211612, 1e-12);
    CHECK_NEAR(model.parameters()[1], 0.28060053826938081, 1e-12);
    CHECK_NEAR(model.parameter_standard_errors()[0], 0.055686680735540617, 1e-12);
    CHECK_NEAR(model.parameter_standard_errors()[1], 0.047441944810231375, 1e-12);
    CHECK_NEAR(model.standard_error(), 0.6026075070895438, 1e-12);

    // z = beta / se, and the p-value is the two-sided standard normal tail.
    for (std::size_t i = 0; i < model.parameters().size(); i++) {
        CHECK_NEAR(model.parameter_z_scores()[i],
                   model.parameters()[i] / model.parameter_standard_errors()[i], 1e-12);
        CHECK_TRUE(model.parameter_p_values()[i] >= 0.0 && model.parameter_p_values()[i] <= 1.0);
    }
}

void test_robust_standard_errors_and_alternate_optimizers() {
    la::Vector y(fpp3::kConsumption);
    la::Matrix x = la::Matrix::from_columns({fpp3::kIncome});

    // Transcription note 5: the robust switch changes the covariance to the sandwich form, so
    // the standard errors move but the point estimates do not.
    ml::GeneralizedLinearModel plain(x, y);
    plain.train();
    ml::GeneralizedLinearModel robust(x, y);
    robust.set_use_robust_se(true);
    robust.train();
    for (std::size_t i = 0; i < plain.parameters().size(); i++) {
        CHECK_EQ(plain.parameters()[i], robust.parameters()[i]);
        CHECK_TRUE(plain.parameter_standard_errors()[i] != robust.parameter_standard_errors()[i]);
        CHECK_TRUE(std::isfinite(robust.parameter_standard_errors()[i]));
    }

    // A different local optimizer reaches the same identity-link answer. The C# suite never
    // exercises set_optimizer, so this is the only check that the other four branches work.
    for (opt::LocalMethod method : {opt::LocalMethod::BFGS, opt::LocalMethod::Powell}) {
        ml::GeneralizedLinearModel alt(x, y);
        alt.set_optimizer(method);
        alt.train();
        CHECK_NEAR(alt.parameters()[0], 0.54510, 1e-3);
        CHECK_NEAR(alt.parameters()[1], 0.28060, 1e-3);
    }
}

void test_constructor_guards() {
    la::Matrix x = la::Matrix::from_columns({fpp3::kIncome});
    CHECK_THROWS_MSG(
        ml::GeneralizedLinearModel(x, la::Vector(std::vector<double>(5, 1.0))),
        "same number of rows");
    CHECK_THROWS_MSG(ml::GeneralizedLinearModel(la::Matrix(2, 1, {1.0, 2.0}),
                                                 la::Vector(std::vector<double>{1.0, 2.0})),
                     "at least three data points");
    // More columns than observations.
    CHECK_THROWS_MSG(ml::GeneralizedLinearModel(la::Matrix(3, 5, std::vector<double>(15, 1.0)),
                                                 la::Vector(std::vector<double>{1.0, 2.0, 3.0})),
                     "requires at least 5 data points");
}

}  // namespace

int main() {
    test_simple_linear_regression();
    test_multiple_linear_regression();
    test_log();
    test_logistic();
    test_probit();
    test_log_log();
    test_parameter_names_and_shape();
    test_predict_and_intervals();
    test_residuals_and_information_criteria();
    test_robust_standard_errors_and_alternate_optimizers();
    test_constructor_guards();
    return chtest::summary("test_generalized_linear_model");
}
