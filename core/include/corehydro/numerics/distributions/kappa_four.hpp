// ported from: Numerics/Distributions/Univariate/KappaFour.cs @ 7e8e8d1
//
// Kappa-4 distribution parameterized by location ξ, scale α, shape κ, shape h.
// InverseCDF has closed form with limit branches for h→0 and κ→0 that mirror the C#
// exactly. Moments use numerical integration (stratified bins) matching C# CentralMoments.
// Special cases: h=-1 → Generalized Logistic; h=0 → GEV; h=1 → Generalized Pareto.
#pragma once
#include <algorithm>
#include <string>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_standard_error.hpp"
#include "corehydro/numerics/distributions/base/distribution_numerics.hpp"
#include "corehydro/numerics/distributions/base/kappa_expected_information.hpp"
#include "corehydro/numerics/distributions/base/kappa_four_boundary.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/integration/adaptive_gauss_kronrod.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf = corehydro::numerics::math::special;

class KappaFour : public UnivariateDistributionBase,
                  public IEstimation,
                  public ILinearMomentEstimation,
                  public IMaximumLikelihoodEstimation,
                  public IStandardError {
   public:
    // Constructs a Kappa-4 distribution with ξ=100, α=10, κ=0, h=0.
    KappaFour() { set_parameters({100.0, 10.0, 0.0, 0.0}); }

    // Constructs a Kappa-4 distribution with given parameters.
    KappaFour(double location, double scale, double shape, double shape2) {
        set_parameters({location, scale, shape, shape2});
    }

    double xi()    const { return xi_; }
    double alpha() const { return alpha_; }
    double kappa() const { return kappa_; }
    double hondo() const { return hondo_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::KappaFour;
    }
    int number_of_parameters() const override { return 4; }
    std::vector<double> get_parameters() const override {
        return {xi_, alpha_, kappa_, hondo_};
    }

    void set_parameters(const std::vector<double>& p) override {
        parameters_valid_ = validate(p);
        xi_    = p[0];
        alpha_ = p[1];
        kappa_ = p[2];
        hondo_ = p[3];
        moments_computed_ = false;
        compensated_minimum_computed_ = false;
    }

    // --- Moments / support (numerical integration, mirrors C# CentralMoments(1000)) ---
    double mean() const override {
        ensure_moments();
        return u_[0];
    }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override {
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        const double lower = minimum();
        const double upper = maximum();
        const double lower_density = std::isfinite(lower) ? lower_endpoint_density() : 0.0;
        const double upper_density = std::isfinite(upper) ? pdf(upper) : 0.0;
        if (std::isinf(lower_density) && std::isinf(upper_density)) return kNaN;
        if (std::isinf(lower_density)) return lower;
        if (std::isinf(upper_density)) return upper;
        const double denominator = 1.0 - kappa_ * hondo_;
        const double t = (1.0 - kappa_) / denominator;
        if (denominator > 0.0 && t > 0.0 && (hondo_ <= 0.0 || hondo_ * t < 1.0)) {
            const double log_t = std::log1p(-kappa_) - std::log1p(-kappa_ * hondo_);
            return affine_quantile(xi_, alpha_, quantile_from_log_t(log_t, kappa_));
        }
        if (lower_density > upper_density) return lower;
        if (upper_density > lower_density) return upper;
        return kNaN;
    }

    double standard_deviation() const override {
        ensure_moments();
        return u_[1];
    }

    double skewness() const override {
        ensure_moments();
        return u_[2];
    }

    double kurtosis() const override {
        ensure_moments();
        return u_[3];
    }

    // Minimum. Mirrors C# Minimum property.
    double minimum() const override {
        if (hondo_ > 0.0 && !compensated_minimum_computed_)
            compensated_minimum_computed_ =
                distribution_numerics::KappaFourBoundary::try_lower_endpoint(
                    xi_, alpha_, kappa_, hondo_, compensated_minimum_);
        if (hondo_ > 0.0 && compensated_minimum_computed_) return compensated_minimum_;
        if (hondo_ <= 0.0 && kappa_ < 0.0) {
            return location_plus_scale_over_shape();
        } else if (hondo_ > 0.0 && kappa_ != 0.0) {
            return affine_quantile(
                xi_, alpha_, quantile_from_log_t(-std::log(hondo_), kappa_));
        } else if (hondo_ > 0.0 && kappa_ == 0.0) {
            return affine_quantile(xi_, alpha_, std::log(hondo_));
        } else if (hondo_ <= 0.0 && kappa_ >= 0.0) {
            return -kInf;
        }
        return kNaN;
    }

    // Maximum. Mirrors C# Maximum property.
    double maximum() const override {
        if (kappa_ <= 0.0) {
            return kInf;
        } else {
            return location_plus_scale_over_shape();
        }
    }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        if (std::isnan(x)) return kNaN;
        if (std::isinf(x) || x < minimum() || x > maximum()) return 0.0;
        if (x == minimum()) return lower_endpoint_density();
        if (x == maximum())
            return kappa_ < 1.0 ? 0.0 : kappa_ == 1.0 ? 1.0 / alpha_ : kInf;
        return std::exp(interior_log_density(x));
    }

    double cdf(double x) const override {
        return std::exp(log_cdf(x));
    }

    double log_pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        if (std::isnan(x) || std::isinf(x) || x < minimum() || x > maximum()) return -kInf;
        if (x == minimum() || x == maximum()) {
            const double density = pdf(x);
            return density > 0.0 ? std::log(density) : -kInf;
        }
        return interior_log_density(x);
    }

    double log_cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        if (x <= minimum()) return -kInf;
        if (x >= maximum()) return 0.0;
        return interior_log_probability(x, log_t(x));
    }

    double ccdf(double x) const override { return -std::expm1(log_cdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return -kInf;
        const double w = log_t(x);
        const double log_f = interior_log_probability(x, w);
        if (log_f == 0.0) return w;
        return log_f < -std::log(2.0) ? std::log1p(-std::exp(log_f))
                                     : std::log(-std::expm1(log_f));
    }

    // InverseCDF with limit branches for h→0 and κ→0. Mirrors C# InverseCDF.
    double inverse_cdf(double probability) const override {
        if (std::isnan(probability) || probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return affine_quantile(
            xi_, alpha_, standard_quantile(std::log(probability), kappa_, hondo_));
    }

    // --- Parameter display names (X1; C# KappaFour.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xCE\xBE)", "Scale (\xCE\xB1)", "Shape (\xCE\xBA)", "Shape (h)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBE", "\xCE\xB1", "\xCE\xBA", "h"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<KappaFour>(xi_, alpha_, kappa_, hondo_);
    }

    math::linalg::Matrix2D parameter_covariance(
        int sample_size, ParameterEstimationMethod method) const override {
        distribution_numerics::validate_sample_size(sample_size);
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        if (method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::runtime_error(
                "Kappa Four covariance is implemented only for maximum likelihood");
        return distribution_numerics::KappaExpectedInformation::parameter_covariance(
            alpha_, kappa_, hondo_, sample_size, 4);
    }

    double quantile_variance(double probability, int sample_size,
                             ParameterEstimationMethod method) const override {
        distribution_numerics::validate_probability(probability);
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        const auto covariance = KappaFour(0.0, 1.0, kappa_, hondo_)
                                    .parameter_covariance(sample_size, method);
        const double log_probability = std::log(probability);
        const double w = quantile_log_t(log_probability, hondo_);
        const double argument = kappa_ * w;
        const double s = hondo_ * log_probability;
        const double dh = hondo_ > 0.0 && s < -0.5
                              ? log_probability * (std::exp(s) / std::expm1(s)) - 1.0 / hondo_
                              : log_probability * log_exponential_relative_derivative(s);
        const double scale_gradient =
            distribution_numerics::scaled_exprel_product(alpha_, -w, argument);
        const double shape_gradient =
            -distribution_numerics::scaled_exprel_derivative_product(alpha_, w, argument);
        const double hondo_gradient =
            dh == 0.0 ? 0.0
                      : -std::copysign(
                            std::exp(std::log(alpha_) + argument + std::log(std::fabs(dh))), dh);
        return distribution_numerics::scaled_quantile_variance(
            covariance, {alpha_, scale_gradient, shape_gradient, hondo_gradient});
    }

    std::vector<double> quantile_gradient(double probability) const override {
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        distribution_numerics::validate_probability(probability);
        auto gradient = standard_quantile_gradient(std::log(probability), kappa_, hondo_);
        gradient[2] *= alpha_;
        gradient[3] *= alpha_;
        return gradient;
    }

    math::linalg::Matrix2D quantile_jacobian(
        const std::vector<double>& probabilities, double& determinant) const override {
        return distribution_numerics::quantile_jacobian(
            *this, number_of_parameters(), probabilities, determinant);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        validate_fitting_sample(sample);
        if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("KappaFour: unsupported estimation method");
        }
    }

    // ParametersFromLinearMoments: Newton-Raphson iteration, ported directly from Fortran
    // (Hosking, IBM Research Report RC20525, Version 3, August 1996). Mirrors C# exactly.
    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        if (moments.size() != 4)
            throw std::invalid_argument("exactly four L-moments are required");
        for (double moment : moments)
            if (!std::isfinite(moment)) throw std::out_of_range("L-moments must be finite");
        double L1 = moments[0];
        double L2 = moments[1];
        double T3 = moments[2];
        double T4 = moments[3];

        constexpr double eps = 1e-6;
        constexpr int maxit = 20, maxsr = 10;

        if (L2 <= 0.0)
            throw std::invalid_argument("L-moments invalid.");
        if (std::fabs(T3) >= 1.0 || std::fabs(T4) >= 1.0)
            throw std::invalid_argument("L-moments invalid.");
        if (T4 <= (5.0 * T3 * T3 - 1.0) / 4.0)
            throw std::invalid_argument("L-moments invalid.");
        if (T4 >= (5.0 * T3 * T3 + 1.0) / 6.0)
            throw std::invalid_argument(
                "(TAU-3, TAU-4) lies above the Generalized Logistic.");

        // Initial values
        double G = (1.0 - 3.0 * T3) / (1.0 + T3);
        double H = 1.001;
        double Z = G + H * 0.725;
        double XDIST = 10.0, DIST = 0.0;
        double U1 = 0, U2 = 0, U3 = 0, U4 = 0;
        double ALAM2 = 0, ALAM3 = 0, ALAM4 = 0;
        double TAU3 = 0, TAU4 = 0;
        double E1 = 0, E2 = 0;
        double DEL1 = 0, DEL2 = 0;
        double XG = 0, XH = 0, XZ, RHH;
        double U1G, U2G, U3G, U4G, U1H, U2H, U3H, U4H;
        double DL2G, DL2H, DL3G, DL3H, DL4G, DL4H;
        double D11, D12, D21, D22, DET, H11, H12, H21, H22;
        double FACTOR;

        bool converged = false;
        for (int i = 1; i <= maxit; i++) {
            for (int j = 1; j <= maxsr; j++) {
                if (!std::isfinite(G) || !std::isfinite(H) || G > 53.0)
                    throw std::runtime_error(
                        "L-moment iteration encountered a nonfinite shape");
                if (kappa_linear_moments_need_integration(G, H)) {
                    const auto linear = kappa_standard_linear_moments(G, H);
                    ALAM2 = linear[1];
                    ALAM3 = linear[2];
                    ALAM4 = linear[3];
                } else if (H < 0.0) {
                    U1 = std::exp(sf::log_gamma(-1.0 / H - G) - sf::log_gamma(-1.0 / H + 1.0));
                    U2 = std::exp(sf::log_gamma(-2.0 / H - G) - sf::log_gamma(-2.0 / H + 1.0));
                    U3 = std::exp(sf::log_gamma(-3.0 / H - G) - sf::log_gamma(-3.0 / H + 1.0));
                    U4 = std::exp(sf::log_gamma(-4.0 / H - G) - sf::log_gamma(-4.0 / H + 1.0));
                } else {
                    U1 = std::exp(sf::log_gamma(1.0 / H) - sf::log_gamma(1.0 / H + 1.0 + G));
                    U2 = std::exp(sf::log_gamma(2.0 / H) - sf::log_gamma(2.0 / H + 1.0 + G));
                    U3 = std::exp(sf::log_gamma(3.0 / H) - sf::log_gamma(3.0 / H + 1.0 + G));
                    U4 = std::exp(sf::log_gamma(4.0 / H) - sf::log_gamma(4.0 / H + 1.0 + G));
                }
                if (!kappa_linear_moments_need_integration(G, H)) {
                    ALAM2 = U1 - 2.0 * U2;
                    ALAM3 = -U1 + 6.0 * U2 - 6.0 * U3;
                    ALAM4 = U1 - 12.0 * U2 + 30.0 * U3 - 20.0 * U4;
                }
                if (ALAM2 == 0.0 || !std::isfinite(ALAM2) || !std::isfinite(ALAM3) ||
                    !std::isfinite(ALAM4))
                    throw std::runtime_error(
                        "L-moment iteration could not evaluate finite ratios");
                TAU3 = ALAM3 / ALAM2;
                TAU4 = ALAM4 / ALAM2;
                E1 = TAU3 - T3;
                E2 = TAU4 - T4;

                DIST = std::max(std::fabs(E1), std::fabs(E2));
                if (DIST < XDIST) break;

                DEL1 *= 0.5;
                DEL2 *= 0.5;
                G = XG - DEL1;
                H = XH - DEL2;
                Z = G + H * 0.725;

                if (j == maxsr)
                    throw std::runtime_error(
                        "Iteration encountered numerical difficulties - overflow.");
            }

            if (DIST < eps) {
                converged = true;
                break;
            }
            if (i == maxit) break;

            XG = G; XH = H; XZ = Z; XDIST = DIST;
            if (kappa_linear_moments_need_integration(G, H)) {
                DL2G = kappa_linear_moment_derivative(G, H, 1, 2);
                DL2H = kappa_linear_moment_derivative(G, H, 1, 3);
                DL3G = kappa_linear_moment_derivative(G, H, 2, 2);
                DL3H = kappa_linear_moment_derivative(G, H, 2, 3);
                DL4G = kappa_linear_moment_derivative(G, H, 3, 2);
                DL4H = kappa_linear_moment_derivative(G, H, 3, 3);
            } else if (H > 0.0) {
                RHH = 1.0 / (H * H);
                U1G = -U1 * sf::digamma(1.0 / H + 1.0 + G);
                U2G = -U2 * sf::digamma(2.0 / H + 1.0 + G);
                U3G = -U3 * sf::digamma(3.0 / H + 1.0 + G);
                U4G = -U4 * sf::digamma(4.0 / H + 1.0 + G);
                U1H = RHH * (-U1G - U1 * sf::digamma(1.0 / H));
                U2H = 2.0 * RHH * (-U2G - U2 * sf::digamma(2.0 / H));
                U3H = 3.0 * RHH * (-U3G - U3 * sf::digamma(3.0 / H));
                U4H = 4.0 * RHH * (-U4G - U4 * sf::digamma(4.0 / H));
            } else {
                RHH = 1.0 / (H * H);
                U1G = -U1 * sf::digamma(-1.0 / H - G);
                U2G = -U2 * sf::digamma(-2.0 / H - G);
                U3G = -U3 * sf::digamma(-3.0 / H - G);
                U4G = -U4 * sf::digamma(-4.0 / H - G);
                U1H = RHH * (-U1G - U1 * sf::digamma(-1.0 / H + 1.0));
                U2H = 2.0 * RHH * (-U2G - U2 * sf::digamma(-2.0 / H + 1.0));
                U3H = 3.0 * RHH * (-U3G - U3 * sf::digamma(-3.0 / H + 1.0));
                U4H = 4.0 * RHH * (-U4G - U4 * sf::digamma(-4.0 / H + 1.0));
            }

            if (!kappa_linear_moments_need_integration(G, H)) {
                DL2G = U1G - 2.0 * U2G;
                DL2H = U1H - 2.0 * U2H;
                DL3G = -U1G + 6.0 * U2G - 6.0 * U3G;
                DL3H = -U1H + 6.0 * U2H - 6.0 * U3H;
                DL4G = U1G - 12.0 * U2G + 30.0 * U3G - 20.0 * U4G;
                DL4H = U1H - 12.0 * U2H + 30.0 * U3H - 20.0 * U4H;
            }
            D11 = (DL3G - TAU3 * DL2G) / ALAM2;
            D12 = (DL3H - TAU3 * DL2H) / ALAM2;
            D21 = (DL4G - TAU4 * DL2G) / ALAM2;
            D22 = (DL4H - TAU4 * DL2H) / ALAM2;
            DET = D11 * D22 - D12 * D21;
            if (DET == 0.0 || !std::isfinite(DET))
                throw std::runtime_error("L-moment derivative matrix is singular or nonfinite");
            H11 = D22 / DET;
            H12 = -D12 / DET;
            H21 = -D21 / DET;
            H22 = D11 / DET;
            DEL1 = E1 * H11 + E2 * H12;
            DEL2 = E1 * H21 + E2 * H22;

            G = XG - DEL1;
            H = XH - DEL2;
            Z = G + H * 0.725;

            FACTOR = 1.0;
            if (G <= -1.0) FACTOR = 0.8 * (XG + 1.0) / DEL1;
            if (H <= -1.0) FACTOR = std::min(FACTOR, 0.8 * (XH + 1.0) / DEL2);
            if (Z <= -1.0) FACTOR = std::min(FACTOR, 0.8 * (XZ + 1.0) / (XZ - Z));
            if (H <= 0.0 && G * H <= -1.0)
                FACTOR = std::min(FACTOR, 0.8 * (XG * XH + 1.0) / (XG * XH - G * H));
            if (FACTOR != 1.0) {
                DEL1 *= FACTOR;
                DEL2 *= FACTOR;
                G = XG - DEL1;
                H = XH - DEL2;
                Z = G + H * 0.725;
            }

        }
        if (!converged)
            throw std::runtime_error("L-moment iterations failed to converge after 20 iterations");
        const auto standardized = kappa_standard_linear_moments(G, H);
        const double alpha_r = L2 / standardized[1];
        const double xi_r = L1 - alpha_r * standardized[0];
        if (!std::isfinite(xi_r) || !std::isfinite(alpha_r) || alpha_r <= 0.0)
            throw std::runtime_error(
                "L-moment shapes converged without finite valid location and scale");
        return {xi_r, alpha_r, G, H};
    }

    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        if (parameters.size() != 4 || !validate(parameters))
            throw std::invalid_argument("exactly four valid Kappa Four parameters are required");
        const double kappa = parameters[2];
        const double hondo = parameters[3];
        if (kappa <= -1.0 || (hondo < 0.0 && kappa >= -1.0 / hondo))
            throw std::out_of_range(
                "L-moments require kappa > -1 and, for hondo < 0, kappa*hondo > -1");
        const auto standardized = kappa_standard_linear_moments(kappa, hondo);
        if (!std::isfinite(standardized[0]) || !std::isfinite(standardized[1]) ||
            standardized[1] <= 0.0 || !std::isfinite(standardized[2]) ||
            !std::isfinite(standardized[3]))
            throw std::runtime_error("finite standardized L-moments could not be evaluated");
        const double l1 = parameters[0] + parameters[1] * standardized[0];
        const double l2 = parameters[1] * standardized[1];
        if (!std::isfinite(l1) || !std::isfinite(l2) || l2 <= 0.0)
            throw std::runtime_error("L-moments are outside the finite numerical range");
        return {l1, l2, standardized[2] / standardized[1],
                standardized[3] / standardized[1]};
    }

    // GetParameterConstraints for MLE.
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        validate_fitting_sample(sample);
        initials.resize(4, 0.0);
        lowers.resize(4, 0.0);
        uppers.resize(4, 0.0);
        bool have_bounds = false;
        try {
            auto lm = data::linear_moments(sample);
            initials = parameters_from_linear_moments(lm);

            if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
            lowers[0] = -std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
            uppers[0] =  std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
            lowers[1] = kDoubleMachineEpsilon;
            uppers[1] =  std::pow(10.0, std::ceil(std::log10(std::fabs(initials[1])) + 1.0));
            lowers[2] = -10.0; uppers[2] = 10.0;
            lowers[3] = -2.0;  uppers[3] = 2.0;
            have_bounds = true;

            if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) initials[2] = 0.0;
            if (initials[3] <= lowers[3] || initials[3] >= uppers[3]) initials[3] = 0.0;
            if (is_usable_initializer(initials, lowers, uppers, sample)) return;
        } catch (...) {
            // Fall through to the shipped GEV initializer.
        }

        try {
            GeneralizedExtremeValue gev;
            std::vector<double> gev_initials, gev_lowers, gev_uppers;
            gev.get_parameter_constraints(sample, gev_initials, gev_lowers, gev_uppers);
            for (int i = 0; i < 3; ++i) {
                initials[static_cast<std::size_t>(i)] =
                    gev_initials[static_cast<std::size_t>(i)];
                if (!have_bounds) {
                    lowers[static_cast<std::size_t>(i)] =
                        gev_lowers[static_cast<std::size_t>(i)];
                    uppers[static_cast<std::size_t>(i)] =
                        gev_uppers[static_cast<std::size_t>(i)];
                }
            }
            initials[3] = 0.0;
            lowers[3] = -2.0;
            uppers[3] = 2.0;
            if (is_usable_initializer(initials, lowers, uppers, sample)) return;
        } catch (...) {
            throw std::runtime_error(
                "Neither Kappa nor GEV initialization produced a valid Kappa Four fitting "
                "candidate");
        }
        throw std::runtime_error(
            "Neither Kappa nor GEV initialization has finite sample likelihood within the "
            "fitting bounds");
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            KappaFour k4;
            k4.set_parameters(x);
            return k4.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 4, initials, lowers, uppers);
        solver.maximize();
        if (solver.status() != math::optimization::OptimizationStatus::Success)
            throw std::runtime_error(
                "Kappa Four maximum likelihood estimation failed to converge");
        const auto result = solver.best_parameters();
        if (!is_usable_initializer(result, lowers, uppers, sample))
            throw std::runtime_error(
                "Kappa Four maximum likelihood estimation returned invalid parameters or "
                "nonfinite sample likelihood");
        return result;
    }

   private:
    static void validate_fitting_sample(const std::vector<double>& sample) {
        if (sample.size() < 4)
            throw std::out_of_range(
                "at least four observations are required for Kappa Four estimation");
        for (double observation : sample)
            if (!std::isfinite(observation))
                throw std::out_of_range("Kappa Four observations must be finite");
    }

    static bool is_usable_initializer(const std::vector<double>& parameters,
                                      const std::vector<double>& lower,
                                      const std::vector<double>& upper,
                                      const std::vector<double>& sample) {
        if (!validate(parameters) || lower.size() != 4 || upper.size() != 4) return false;
        for (std::size_t i = 0; i < 4; ++i)
            if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i] >= upper[i] ||
                parameters[i] < lower[i] || parameters[i] > upper[i])
                return false;
        KappaFour candidate(parameters[0], parameters[1], parameters[2], parameters[3]);
        return std::isfinite(candidate.log_likelihood(sample));
    }

    static double exponential_relative(double x) {
        return x == 0.0 ? 1.0 : std::expm1(x) / x;
    }

    static double log_exponential_relative(double x) {
        if (std::fabs(x) < 1e-4)
            return std::log1p(
                x * (0.5 + x * (1.0 / 6.0 +
                                x * (1.0 / 24.0 + x * (1.0 / 120.0 + x / 720.0)))));
        return x > 0.0 ? x + std::log(-std::expm1(-x)) - std::log(x)
                       : std::log(-std::expm1(x)) - std::log(-x);
    }

    static double quantile_log_t(double log_probability, double hondo) {
        const double s = hondo * log_probability;
        if (std::isinf(s) && s > 0.0) return kInf;
        if (hondo > 0.0 && s < -0.5)
            return std::log(-std::expm1(s)) - std::log(hondo);
        return std::log(-log_probability) + log_exponential_relative(s);
    }

    static double quantile_from_log_t(double w, double kappa) {
        if (kappa == 0.0) return -w;
        const double v = kappa * w;
        return std::fabs(v) < 0.5 ? -w * exponential_relative(v) : -std::expm1(v) / kappa;
    }

    static double standard_quantile(double log_probability, double kappa, double hondo) {
        return quantile_from_log_t(quantile_log_t(log_probability, hondo), kappa);
    }

    static double affine_quantile(double location, double scale, double value) {
        const double product = scale * value;
        if (std::isinf(product) && std::isfinite(value) &&
            std::signbit(location) != std::signbit(product))
            return scale * (value + location / scale);
        return location + product;
    }

    double location_plus_scale_over_shape() const {
        const double shift = alpha_ / kappa_;
        if (std::isinf(shift) && std::signbit(xi_) != std::signbit(shift))
            return (xi_ * kappa_ + alpha_) / kappa_;
        return xi_ + shift;
    }

    double log_t(double x) const {
        double y = (x - xi_) / alpha_;
        if (std::isinf(y) && std::isfinite(x)) y = x / alpha_ - xi_ / alpha_;
        if (kappa_ == 0.0) return -y;
        const double product = -kappa_ * y;
        if (product < -0.9)
            return distribution_numerics::KappaFourBoundary::log_t(x, xi_, alpha_, kappa_);
        const double log_base = std::isinf(product) && product > 0.0
                                    ? std::log(std::fabs(kappa_)) + std::log(std::fabs(y))
                                    : std::log1p(product);
        return log_base / kappa_;
    }

    double log_probability_from_log_t(double w) const {
        if (hondo_ == 0.0) return -std::exp(w);
        const double z = std::log(std::fabs(hondo_)) + w;
        if (z < -36.0) return -std::exp(w);
        if (hondo_ < 0.0)
            return (z > 0.0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z))) /
                   hondo_;
        return (z < -std::log(2.0) ? std::log1p(-std::exp(z))
                                   : std::log(-std::expm1(z))) /
               hondo_;
    }

    double interior_log_probability(double x, double w) const {
        if (hondo_ > 0.0 && std::fabs(std::log(hondo_) + w) < 1e-5)
            return distribution_numerics::KappaFourBoundary::log_probability(
                x, xi_, alpha_, kappa_, hondo_);
        return log_probability_from_log_t(w);
    }

    double interior_log_density(double x) const {
        const double w = log_t(x);
        if (hondo_ < 0.0 && std::log(-hondo_) + w > 0.0) {
            const double z = std::log(-hondo_) + w;
            const double correction = std::log(-hondo_) + std::log1p(std::exp(-z));
            const double result = (1.0 / hondo_ - kappa_) * w +
                                  (1.0 / hondo_ - 1.0) * correction - std::log(alpha_);
            return std::isnan(result)
                       ? (w + correction) / hondo_ - kappa_ * w - correction -
                             std::log(alpha_)
                       : result;
        }
        return (1.0 - kappa_) * w +
               (1.0 - hondo_) * interior_log_probability(x, w) - std::log(alpha_);
    }

    double lower_endpoint_density() const {
        if (hondo_ > 0.0)
            return hondo_ < 1.0 ? 0.0 : hondo_ == 1.0 ? 1.0 / alpha_ : kInf;
        if (hondo_ == 0.0) return 0.0;
        const double exponent = 1.0 / hondo_ - kappa_;
        return exponent < 0.0
                   ? 0.0
                   : exponent > 0.0
                         ? kInf
                         : std::exp((1.0 / hondo_ - 1.0) * std::log(-hondo_) -
                                    std::log(alpha_));
    }

    static double integrate_probability(const std::function<double(double)>& integrand) {
        math::integration::AdaptiveGaussKronrod integration(
            [&](double u) {
                const double log_u = std::log(u);
                const double log_tail = 8.0 * log_u - std::log(2.0);
                const double jacobian = std::exp(std::log(4.0) + 7.0 * log_u);
                const double value =
                    integrand(log_tail) * jacobian +
                    integrand(std::log1p(-std::exp(log_tail))) * jacobian;
                if (!std::isfinite(value))
                    throw std::runtime_error("Kappa Four quantile integrand is nonfinite");
                return value;
            },
            0.0, 1.0);
        integration.relative_tolerance = 1e-10;
        integration.absolute_tolerance = 1e-12;
        integration.report_failure = false;
        integration.integrate();
        if (integration.status() != math::integration::IntegrationStatus::Success ||
            !std::isfinite(integration.result()) ||
            !std::isfinite(integration.standard_error()))
            return kNaN;
        return integration.standard_error() <=
                       std::max(1e-12, 1e-10 * std::fabs(integration.result()))
                   ? integration.result()
                   : kNaN;
    }

    static double kappa_linear_moment_polynomial(double log_probability, int order) {
        const double probability = std::exp(log_probability);
        if (order == 0) return 1.0;
        if (order == 1) return 2.0 * probability - 1.0;
        if (order == 2) return (6.0 * probability - 6.0) * probability + 1.0;
        return ((20.0 * probability - 30.0) * probability + 12.0) * probability - 1.0;
    }

    static double exponential_relative_derivative(double x) {
        if (std::fabs(x) < 1e-3)
            return 0.5 + x * (1.0 / 3.0 +
                              x * (1.0 / 8.0 +
                                   x * (1.0 / 30.0 +
                                        x * (1.0 / 144.0 + x / 840.0))));
        if (x > 50.0) return std::exp(x + std::log(x - 1.0) - 2.0 * std::log(x));
        return (x * std::exp(x) - std::expm1(x)) / (x * x);
    }

    static double log_exponential_relative_derivative(double x) {
        if (std::fabs(x) < 1e-3)
            return 0.5 + x * (1.0 / 12.0 +
                              x * x * (-1.0 / 720.0 + x * x / 30240.0));
        return x > 0.0 ? 1.0 / -std::expm1(-x) - 1.0 / x
                       : std::exp(x) / std::expm1(x) - 1.0 / x;
    }

    static std::vector<double> standard_quantile_gradient(double log_probability, double kappa,
                                                          double hondo) {
        const double s = hondo * log_probability;
        const double w = quantile_log_t(log_probability, hondo);
        const double v = kappa * w;
        const double dh = hondo > 0.0 && s < -0.5
                              ? log_probability * (std::exp(s) / std::expm1(s)) - 1.0 / hondo
                              : log_probability * log_exponential_relative_derivative(s);
        const double dk = v < -50.0
                              ? -1.0 / (kappa * kappa)
                              : v > 50.0
                                    ? -std::exp(v + std::log(v - 1.0) -
                                                2.0 * std::log(std::fabs(kappa)))
                                    : -w * w * exponential_relative_derivative(v);
        return {1.0, quantile_from_log_t(w, kappa), dk, -std::exp(v) * dh};
    }

    static bool kappa_linear_moments_need_integration(double kappa, double hondo) {
        return std::fabs(kappa) < 0.001 || std::fabs(hondo) < 0.001;
    }

    static double kappa_linear_moment_derivative(double kappa, double hondo, int order,
                                                 int component) {
        return integrate_probability([=](double log_probability) {
            return standard_quantile_gradient(log_probability, kappa, hondo)
                       [static_cast<std::size_t>(component)] *
                   kappa_linear_moment_polynomial(log_probability, order);
        });
    }

    static std::vector<double> kappa_standard_linear_moments(double kappa, double hondo) {
        if (kappa == 0.0 && hondo == 0.0) {
            const double log2 = std::log(2.0);
            const double log3 = std::log(3.0);
            return {0.577215664901532860606512090082402431, log2,
                    2.0 * log3 - 3.0 * log2, 16.0 * log2 - 10.0 * log3};
        }
        if (std::fabs(kappa) < 0.001 || (hondo != 0.0 && std::fabs(hondo) < 0.001)) {
            std::vector<double> integrated(4);
            for (int order = 0; order < 4; ++order)
                integrated[static_cast<std::size_t>(order)] = integrate_probability(
                    [=](double log_probability) {
                        return standard_quantile(log_probability, kappa, hondo) *
                               kappa_linear_moment_polynomial(log_probability, order);
                    });
            return integrated;
        }
        double beta[4];
        const double log_gamma = sf::log_gamma(1.0 + kappa);
        for (int r = 1; r <= 4; ++r) {
            double log_g;
            if (hondo == 0.0) {
                log_g = log_gamma - kappa * std::log(static_cast<double>(r));
            } else if (hondo > 0.0) {
                log_g = std::log(static_cast<double>(r)) + log_gamma +
                        sf::log_gamma(r / hondo) - (1.0 + kappa) * std::log(hondo) -
                        sf::log_gamma(1.0 + kappa + r / hondo);
            } else {
                log_g = std::log(static_cast<double>(r)) + log_gamma +
                        sf::log_gamma(-kappa - r / hondo) -
                        (1.0 + kappa) * std::log(-hondo) -
                        sf::log_gamma(1.0 - r / hondo);
            }
            beta[r - 1] = -std::expm1(log_g) / (kappa * r);
        }
        return {beta[0], 2.0 * beta[1] - beta[0],
                6.0 * beta[2] - 6.0 * beta[1] + beta[0],
                20.0 * beta[3] - 30.0 * beta[2] + 12.0 * beta[1] - beta[0]};
    }

    bool moment_exists(int order) const {
        return order * kappa_ > -1.0 &&
               (hondo_ >= 0.0 || order * kappa_ * hondo_ > -1.0);
    }

    void compute_moments() const {
        u_[0] = u_[1] = u_[2] = u_[3] = kNaN;
        if (!moment_exists(1)) {
            moments_computed_ = true;
            return;
        }
        const double standardized_mean = integrate_probability(
            [&](double log_p) { return standard_quantile(log_p, kappa_, hondo_); });
        if (!std::isfinite(standardized_mean)) {
            moments_computed_ = true;
            return;
        }
        u_[0] = affine_quantile(xi_, alpha_, standardized_mean);
        if (!moment_exists(2)) {
            moments_computed_ = true;
            return;
        }
        const double variance = integrate_probability([&](double log_p) {
            const double delta = standard_quantile(log_p, kappa_, hondo_) - standardized_mean;
            return delta * delta;
        });
        if (!(variance > 0.0) || !std::isfinite(variance)) {
            moments_computed_ = true;
            return;
        }
        const double sd = std::sqrt(variance);
        u_[1] = alpha_ * sd;
        if (moment_exists(3))
            u_[2] = integrate_probability([&](double log_p) {
                return std::pow((standard_quantile(log_p, kappa_, hondo_) -
                                 standardized_mean) /
                                    sd,
                                3.0);
            });
        if (moment_exists(4))
            u_[3] = integrate_probability([&](double log_p) {
                return std::pow((standard_quantile(log_p, kappa_, hondo_) -
                                 standardized_mean) /
                                    sd,
                                4.0);
            });
        moments_computed_ = true;
    }

    void ensure_moments() const {
        if (!moments_computed_) compute_moments();
    }

    static bool validate(const std::vector<double>& p) {
        if (p.size() < 4) return false;
        if (std::isnan(p[0]) || std::isinf(p[0])) return false;
        if (std::isnan(p[1]) || std::isinf(p[1]) || p[1] <= 0.0) return false;
        if (std::isnan(p[2]) || std::isinf(p[2])) return false;
        if (std::isnan(p[3]) || std::isinf(p[3])) return false;
        return true;
    }

    double xi_    = 100.0;
    double alpha_ = 10.0;
    double kappa_ = 0.0;
    double hondo_ = 0.0;

    mutable bool moments_computed_ = false;
    mutable double u_[4] = {kNaN, kNaN, kNaN, kNaN};
    mutable bool compensated_minimum_computed_ = false;
    mutable double compensated_minimum_ = kNaN;
};

}  // namespace corehydro::numerics::distributions
