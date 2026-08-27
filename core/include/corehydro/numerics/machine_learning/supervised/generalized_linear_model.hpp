// ported from: Numerics/Machine Learning/Supervised/GeneralizedLinearModel.cs @ 2a0357a
//
// Generalized linear regression: a linear predictor `eta = X * beta` mapped to the response scale
// by a link function, fitted by maximizing a family-specific log-likelihood with a local
// optimizer.
//
// SEVERED: `Summary()`. It builds roughly 35 lines of formatted `StringBuilder` text using
// .NET-specific numeric formats (`N5`, `E2`, `N4`), significance stars, and the `Y.Header` model
// name. It is presentation only, with no numeric surface, and joins the Bulletin17C GMM report
// text as a documented severance (see upstream/CLAUDE.md). `parameter_names()` is NOT severed --
// it is a real public member the summary merely reads, and it ships, including the C# default
// "beta-subscript" names.
//
// Seven transcription notes, each on something a "cleanup" would silently change:
//
// 1. The optimizer objective returns `double.MaxValue` -- NOT infinity -- for a non-finite
//    log-likelihood, and returns the NEGATED log-likelihood because the optimizers minimize.
// 2. The per-observation identity-family term is `-0.5 * resid^2`: no sigma, no constant. So the
//    quantity `SetOptimizer`'s objective maximizes is NOT the log-likelihood `ComputeDiagnostics`
//    later reports. `ComputeDiagnostics` recomputes a SECOND, family-specific log-likelihood WITH
//    its constant, and only that one feeds AIC/AICc/BIC. The two do not agree, and both are
//    transcribed as written.
// 3. The parameter bounds are family-specific and asymmetric. Identity and Log build them from
//    the response range (`slope = range / max(n, 1)`, bounds `+/- slope * 100`) and bound the
//    intercept by `min(init/100, init*100)` / `max(init/100, init*100)` -- which for a NEGATIVE
//    initial value gives `[init*100, init/100]`, a correctly-ordered but very wide box. Logit and
//    ComplementaryLogLog use `+/- 10`; Probit uses `+/- 6`. The expressions are transcribed, not
//    their intent.
// 4. `ParameterStandardErrors` multiplies the covariance diagonal's square root by
//    `StandardError` for EVERY link EXCEPT `Log`. The asymmetry is upstream's, and it is what
//    makes the Poisson standard errors reproduce.
// 5. `Covariance` is `(J'J)^-1` from the delta-method Jacobian -- NOT the observed information.
//    `UseRobustSE` switches it to the sandwich `(J'J)^-1 J' diag(resid^2) J (J'J)^-1`.
// 6. The Poisson branch of `ComputeDiagnostics` calls `Factorial.LogFactorial((int)Math.Round(y))`,
//    which silently ROUNDS a non-integer response and throws for a negative one.
// 7. `SetOptimizer` is public and is called by the constructor, so changing `UseRobustSE` or the
//    optimizer method after construction is supported -- but `Train()` must be re-run for it to
//    take effect.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/functions/i_link_function.hpp"
#include "corehydro/numerics/functions/link_function_factory.hpp"
#include "corehydro/numerics/functions/link_function_type.hpp"
#include "corehydro/numerics/math/linalg/lu_decomposition.hpp"  // Matrix::inverse()
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/optimization/adam.hpp"
#include "corehydro/numerics/math/optimization/bfgs.hpp"
#include "corehydro/numerics/math/optimization/gradient_descent.hpp"
#include "corehydro/numerics/math/optimization/powell.hpp"
#include "corehydro/numerics/math/optimization/support/local_method.hpp"
#include "corehydro/numerics/math/optimization/support/nelder_mead_solver.hpp"
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::machine_learning {

class GeneralizedLinearModel {
   public:
    // Constructs a generalized linear model with a link chosen by type.
    GeneralizedLinearModel(const math::linalg::Matrix& x, const math::linalg::Vector& y,
                           bool has_intercept = true,
                           functions::LinkFunctionType link_type =
                               functions::LinkFunctionType::Identity)
        : GeneralizedLinearModel(x, y, functions::LinkFunctionFactory::create(link_type),
                                 has_intercept, link_type) {}

    // Constructs a generalized linear model with a custom link function.
    GeneralizedLinearModel(const math::linalg::Matrix& x, const math::linalg::Vector& y,
                           std::unique_ptr<functions::ILinkFunction> link_function,
                           bool has_intercept = true,
                           functions::LinkFunctionType link_type =
                               functions::LinkFunctionType::Identity)
        : y_(y),
          x_(has_intercept ? add_intercept_column(x) : x),
          has_intercept_(has_intercept),
          link_type_(link_type),
          link_function_(std::move(link_function)) {
        if (y.length() != x.number_of_rows())
            throw std::invalid_argument("X and Y must have the same number of rows.");
        if (y.length() <= 2)
            throw std::runtime_error("There must be at least three data points.");
        if (x.number_of_columns() > y.length())
            throw std::runtime_error(
                "A regression of the requested order requires at least " +
                std::to_string(x.number_of_columns()) + " data points. Only " +
                std::to_string(y.length()) + " data points have been provided.");

        degrees_of_freedom_ = x_.number_of_rows() - x_.number_of_columns();

        // Set the parameter names for the summary report. (The C# `X.Header`/`Y.Header` string
        // columns are not ported -- this port's Matrix/Vector carry no header -- so the default
        // beta-subscript names are always used. The UTF-8 beta below is deliberate and matches
        // the C# literal.)
        parameter_names_.clear();
        if (has_intercept) parameter_names_.push_back("Intercept");
        for (int i = 1; i <= x.number_of_columns(); i++)
            parameter_names_.push_back("\xce\xb2" + std::to_string(i));  // "β" + i

        set_optimizer();
    }

    // Determines if the linear model has an intercept.
    bool has_intercept() const { return has_intercept_; }

    // The vector of response values.
    const math::linalg::Vector& y() const { return y_; }

    // The matrix of predictor values (with the intercept column, if any).
    const math::linalg::Matrix& x() const { return x_; }

    // The estimated parameter values.
    const std::vector<double>& parameters() const { return parameters_; }

    // The estimated parameter names.
    const std::vector<std::string>& parameter_names() const { return parameter_names_; }

    // The estimated parameter standard errors.
    const std::vector<double>& parameter_standard_errors() const {
        return parameter_standard_errors_;
    }

    // The estimated parameter z-scores.
    const std::vector<double>& parameter_z_scores() const { return parameter_z_scores_; }

    // The estimated parameter p-values.
    const std::vector<double>& parameter_p_values() const { return parameter_p_values_; }

    // The estimated parameter covariance matrix.
    const math::linalg::Matrix& covariance() const { return covariance_; }

    // The residuals of the fitted model.
    const std::vector<double>& residuals() const { return residuals_; }

    // The model standard error.
    double standard_error() const { return standard_error_; }

    // The data sample size.
    int sample_size() const { return y_.length(); }

    // The model degrees of freedom.
    int degrees_of_freedom() const { return degrees_of_freedom_; }

    // The Akaike information criterion.
    double aic() const { return aic_; }

    // The Akaike information criterion corrected for small sample sizes.
    double aicc() const { return aicc_; }

    // The Bayesian information criterion.
    double bic() const { return bic_; }

    // Determines whether to estimate robust (sandwich) standard errors.
    bool use_robust_se() const { return use_robust_se_; }
    void set_use_robust_se(bool value) { use_robust_se_ = value; }

    // The optimizer used to train the model. Default = Nelder-Mead.
    math::optimization::Optimizer& optimizer() { return *optimizer_; }
    const math::optimization::Optimizer& optimizer() const { return *optimizer_; }

    // The link function type.
    functions::LinkFunctionType link_type() const { return link_type_; }

    // The link function instance used by this model.
    const functions::ILinkFunction& link_function() const { return *link_function_; }

    // Sets up the local optimizer. Called by the constructor with the default method.
    void set_optimizer(math::optimization::LocalMethod method =
                           math::optimization::LocalMethod::NelderMead) {
        int n = x_.number_of_rows();
        int p = x_.number_of_columns();

        // Transcription note 1: the objective is the NEGATED log-likelihood, and a non-finite
        // value becomes double.MaxValue rather than infinity.
        auto log_likelihood = [this](std::vector<double>& beta) {
            int rows = x_.number_of_rows();
            double log_lh = 0.0;
            for (int i = 0; i < rows; i++) {
                double mu = link_function_->inverse_link(dot(x_.row(i), beta));
                log_lh += log_likelihood_term(mu, y_[i]);
            }
            if (std::isnan(log_lh) || std::isinf(log_lh))
                return std::numeric_limits<double>::max();
            return -log_lh;
        };
        // (The objective takes a MUTABLE reference -- the ported Optimizer's `Objective` mirrors
        // C#'s `Func<double[], double>`, whose array argument an objective can write through --
        // while `GradientFunction` takes a const one. Two different signatures, both as declared.)
        auto gradient = [this](const std::vector<double>& beta) {
            int rows = x_.number_of_rows();
            int cols = x_.number_of_columns();
            std::vector<double> g(static_cast<std::size_t>(cols), 0.0);
            for (int i = 0; i < rows; i++) {
                std::vector<double> xi = x_.row(i);
                double eta = dot(xi, beta);
                double mu = link_function_->inverse_link(eta);
                double d_mu_d_eta = inverse_link_derivative(eta);
                double grad_term = (y_[i] - mu) * d_mu_d_eta;
                for (int j = 0; j < cols; j++)
                    g[static_cast<std::size_t>(j)] -= xi[static_cast<std::size_t>(j)] * grad_term;
            }
            return g;
        };

        // Set the parameter constraints (transcription note 3).
        std::vector<double> initial(static_cast<std::size_t>(p), 0.0);
        std::vector<double> lower(static_cast<std::size_t>(p), 0.0);
        std::vector<double> upper(static_cast<std::size_t>(p), 0.0);

        if (link_type_ == functions::LinkFunctionType::Identity ||
            link_type_ == functions::LinkFunctionType::Log) {
            std::vector<double> scale = y_.to_array();
            if (link_type_ == functions::LinkFunctionType::Log)
                for (double& v : scale) v = std::log(std::max(v, 1e-6));
            double min = data::minimum(scale);
            double max = data::maximum(scale);
            double range = std::max(max - min, 1e-6);
            double slope = range / std::max(n, 1);
            for (int i = 0; i < p; i++) {
                initial[static_cast<std::size_t>(i)] = 0.0;
                lower[static_cast<std::size_t>(i)] = -slope * 100;
                upper[static_cast<std::size_t>(i)] = slope * 100;
            }
            if (has_intercept_) {
                initial[0] = (min + max) / 2.0;
                lower[0] = std::min(initial[0] / 100, initial[0] * 100);
                upper[0] = std::max(initial[0] / 100, initial[0] * 100);
            }
        } else if (link_type_ == functions::LinkFunctionType::Logit) {
            std::fill(lower.begin(), lower.end(), -10.0);
            std::fill(upper.begin(), upper.end(), 10.0);
            if (has_intercept_) {
                double sum = 0.0;
                for (int i = 0; i < y_.length(); i++) sum += y_[i];
                // Log-odds of the observed ratio.
                initial[0] = std::log((sum + 0.5) / (y_.length() - sum + 0.5));
            }
        } else if (link_type_ == functions::LinkFunctionType::Probit) {
            std::fill(lower.begin(), lower.end(), -6.0);
            std::fill(upper.begin(), upper.end(), 6.0);
            if (has_intercept_) {
                double sum = 0.0;
                for (int i = 0; i < y_.length(); i++) sum += y_[i];
                double rate = (sum + 0.5) / (y_.length() + 1.0);  // Additive smoothing
                initial[0] = distributions::Normal::standard_z(rate);
            }
        } else if (link_type_ == functions::LinkFunctionType::ComplementaryLogLog) {
            std::fill(lower.begin(), lower.end(), -10.0);
            std::fill(upper.begin(), upper.end(), 10.0);
            if (has_intercept_) {
                double prob = data::mean(y_.to_array());
                initial[0] = std::log(-std::log(1 - prob));
            }
        }

        // Set the optimizer.
        namespace opt = math::optimization;
        switch (method) {
            case opt::LocalMethod::ADAM:
                optimizer_ = std::make_unique<opt::ADAM>(log_likelihood, p, initial, lower, upper,
                                                          0.001, gradient);
                break;
            case opt::LocalMethod::BFGS:
                optimizer_ =
                    std::make_unique<opt::BFGS>(log_likelihood, p, initial, lower, upper, gradient);
                break;
            case opt::LocalMethod::GradientDescent:
                optimizer_ = std::make_unique<opt::GradientDescent>(log_likelihood, p, initial,
                                                                     lower, upper, 0.001, gradient);
                break;
            case opt::LocalMethod::NelderMead:
                // See support/nelder_mead_solver.hpp: NelderMead is a standalone class in this
                // port, so it reaches an `Optimizer` reference through the shared adapter.
                optimizer_ = std::make_unique<opt::NelderMeadSolver>(log_likelihood, p, initial,
                                                                      lower, upper);
                break;
            case opt::LocalMethod::Powell:
                optimizer_ = std::make_unique<opt::Powell>(log_likelihood, p, initial, lower, upper);
                break;
            default:
                throw std::out_of_range("Unknown local method.");
        }
    }

    // Trains the generalized linear model.
    void train() {
        if (optimizer_ == nullptr) throw std::invalid_argument("The optimizer is null.");
        optimizer_->minimize();
        parameters_ = optimizer_->best_parameter_set().values;
        compute_diagnostics();
    }

    // Returns the mean prediction for each row of `x`.
    std::vector<double> predict(const math::linalg::Matrix& x) const {
        math::linalg::Matrix xp = prepare_design_matrix(x);
        int n = xp.number_of_rows();
        std::vector<double> result(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; i++)
            result[static_cast<std::size_t>(i)] =
                link_function_->inverse_link(dot(xp.row(i), parameters_));
        return result;
    }

    // Returns the prediction with confidence intervals as an n-by-3 matrix with columns lower,
    // mean, upper. `alpha` is the confidence level; the default 0.1 gives 90% intervals.
    math::linalg::Matrix predict_intervals(const math::linalg::Matrix& x,
                                            double alpha = 0.1) const {
        math::linalg::Matrix xp = prepare_design_matrix(x);
        double z = distributions::Normal::standard_z(1 - alpha / 2);
        math::linalg::Matrix result(xp.number_of_rows(), 3);
        for (int i = 0; i < xp.number_of_rows(); i++) {
            std::vector<double> xi = xp.row(i);
            double mu = link_function_->inverse_link(dot(xi, parameters_));
            double se = std::sqrt(dot(xi, covariance_.multiply(math::linalg::Vector(xi)).to_array()));
            result(i, 0) = mu - z * se;
            result(i, 1) = mu;
            result(i, 2) = mu + z * se;
        }
        return result;
    }

   private:
    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
        return s;
    }

    // Adds an intercept column to the covariate matrix.
    static math::linalg::Matrix add_intercept_column(const math::linalg::Matrix& x) {
        math::linalg::Matrix result(x.number_of_rows(), x.number_of_columns() + 1);
        for (int i = 0; i < x.number_of_rows(); i++) {
            result(i, 0) = 1.0;
            for (int j = 0; j < x.number_of_columns(); j++) result(i, j + 1) = x(i, j);
        }
        return result;
    }

    // Prepares the design matrix for prediction, adding an intercept column if needed.
    math::linalg::Matrix prepare_design_matrix(const math::linalg::Matrix& x) const {
        int expected = static_cast<int>(parameters_.size());
        if (x.number_of_columns() == expected) return x;
        if (has_intercept_ && x.number_of_columns() == expected - 1)
            return add_intercept_column(x);
        throw std::invalid_argument(
            "Expected " + std::to_string(expected) + " columns" +
            (has_intercept_ ? " (or " + std::to_string(expected - 1) + " without intercept)" : "") +
            ", but got " + std::to_string(x.number_of_columns()) + ".");
    }

    // The distribution-family-specific per-observation log-likelihood (transcription note 2).
    double log_likelihood_term(double mu, double y) const {
        switch (link_type_) {
            // Normal family (identity link).
            case functions::LinkFunctionType::Identity: {
                double resid = y - mu;
                return -0.5 * resid * resid;
            }
            // Poisson family (log link).
            case functions::LinkFunctionType::Log:
                return y * std::log(mu) - mu;
            // Binomial family (logit, probit, or complementary log-log link).
            case functions::LinkFunctionType::Logit:
            case functions::LinkFunctionType::Probit:
            case functions::LinkFunctionType::ComplementaryLogLog:
                return y * std::log(mu) + (1 - y) * std::log(1 - mu);
            default:
                throw std::out_of_range("Unsupported link function type.");
        }
    }

    // Computes the inverse-link derivative dmu/deta at the given link-space value.
    double inverse_link_derivative(double eta) const {
        double mu = link_function_->inverse_link(eta);
        double d_eta_d_mu = link_function_->d_link(mu);
        return 1.0 / d_eta_d_mu;
    }

    // Computes the model diagnostics.
    void compute_diagnostics() {
        int n = x_.number_of_rows();
        int p = x_.number_of_columns();

        residuals_.assign(static_cast<std::size_t>(n), 0.0);
        std::vector<double> mu = predict(x_);
        math::linalg::Vector diag(n);
        double sse = 0.0;
        for (int i = 0; i < n; i++) {
            residuals_[static_cast<std::size_t>(i)] = y_[i] - mu[static_cast<std::size_t>(i)];
            diag[i] = sqr(residuals_[static_cast<std::size_t>(i)]);
            sse += diag[i];
        }
        standard_error_ = std::sqrt(sse / (n - p));

        // Compute the log-likelihood WITH its family constant (transcription note 2: this is a
        // second, different log-likelihood from the one the optimizer maximized, and only this
        // one feeds the information criteria).
        double log_lik = 0.0;
        switch (link_type_) {
            case functions::LinkFunctionType::Identity: {
                // Assume the residual variance is SE^2.
                double sigma = standard_error_;
                log_lik = -n * 0.5 * std::log(2 * kPi) - n * std::log(sigma) -
                          sse / (2 * sigma * sigma);
                break;
            }
            case functions::LinkFunctionType::Log:
                for (int i = 0; i < n; i++) {
                    double y = y_[i];
                    double m = mu[static_cast<std::size_t>(i)];
                    // Transcription note 6: a non-integer response is silently rounded.
                    log_lik += y * std::log(m) - m -
                               math::special::factorial::log_factorial(
                                   static_cast<int>(std::round(y)));
                }
                break;
            case functions::LinkFunctionType::Logit:
            case functions::LinkFunctionType::Probit:
            case functions::LinkFunctionType::ComplementaryLogLog:
                for (int i = 0; i < n; i++) {
                    double y = y_[i];
                    double m = mu[static_cast<std::size_t>(i)];
                    log_lik += y * std::log(m) + (1 - y) * std::log(1 - m);
                }
                break;
            default:
                throw std::runtime_error("Unknown link function.");
        }

        aic_ = data::GoodnessOfFit::aic(p, log_lik);
        aicc_ = data::GoodnessOfFit::aicc(n, p, log_lik);
        bic_ = data::GoodnessOfFit::bic(n, p, log_lik);

        // The Jacobian matrix for the delta method (transcription note 5).
        math::linalg::Matrix j(n, p);
        for (int i = 0; i < n; i++) {
            std::vector<double> xi = x_.row(i);
            double eta = dot(xi, parameters_);
            double d_mu = inverse_link_derivative(eta);
            for (int k = 0; k < p; k++) j(i, k) = d_mu * xi[static_cast<std::size_t>(k)];
        }

        math::linalg::Matrix jtj = j.transpose().multiply(j);
        math::linalg::Matrix jtj_inv = jtj.inverse();

        if (!use_robust_se_) {
            covariance_ = jtj_inv;
        } else {
            math::linalg::Matrix omega = math::linalg::Matrix::diagonal(diag);
            math::linalg::Matrix meat = j.transpose().multiply(omega).multiply(j);
            covariance_ = jtj_inv.multiply(meat).multiply(jtj_inv);
        }

        // Transcription note 4: every link EXCEPT Log scales by the model standard error.
        std::vector<double> cov_diag = covariance_.diagonal();
        parameter_standard_errors_.assign(static_cast<std::size_t>(p), 0.0);
        for (int i = 0; i < p; i++) {
            double sd = std::sqrt(cov_diag[static_cast<std::size_t>(i)]);
            parameter_standard_errors_[static_cast<std::size_t>(i)] =
                link_type_ != functions::LinkFunctionType::Log ? sd * standard_error_ : sd;
        }
        parameter_z_scores_.assign(static_cast<std::size_t>(p), 0.0);
        parameter_p_values_.assign(static_cast<std::size_t>(p), 0.0);
        for (int i = 0; i < p; i++) {
            double z = parameters_[static_cast<std::size_t>(i)] /
                       parameter_standard_errors_[static_cast<std::size_t>(i)];
            parameter_z_scores_[static_cast<std::size_t>(i)] = z;
            parameter_p_values_[static_cast<std::size_t>(i)] =
                2 * (1 - distributions::Normal::standard_cdf(std::fabs(z)));
        }
    }

    math::linalg::Vector y_;
    math::linalg::Matrix x_;
    bool has_intercept_ = true;
    functions::LinkFunctionType link_type_ = functions::LinkFunctionType::Identity;
    std::unique_ptr<functions::ILinkFunction> link_function_;
    std::vector<double> parameters_;
    std::vector<std::string> parameter_names_;
    std::vector<double> parameter_standard_errors_;
    std::vector<double> parameter_z_scores_;
    std::vector<double> parameter_p_values_;
    math::linalg::Matrix covariance_{0, 0};
    std::vector<double> residuals_;
    double standard_error_ = 0.0;
    int degrees_of_freedom_ = 0;
    double aic_ = 0.0;
    double aicc_ = 0.0;
    double bic_ = 0.0;
    bool use_robust_se_ = false;
    std::unique_ptr<math::optimization::Optimizer> optimizer_;
};

}  // namespace corehydro::numerics::machine_learning
