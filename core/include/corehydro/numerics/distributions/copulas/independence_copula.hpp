// ported from: Numerics/Distributions/Bivariate Copulas/IndependenceCopula.cs @ 7e8e8d1
#pragma once
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/copulas/base/bivariate_copula.hpp"

namespace corehydro::numerics::distributions::copulas {

class IndependenceCopula : public BivariateCopula {
   public:
    IndependenceCopula() = default;
    IndependenceCopula(std::shared_ptr<UnivariateDistributionBase> margin_x,
                       std::shared_ptr<UnivariateDistributionBase> margin_y) {
        marginal_distribution_x = std::move(margin_x);
        marginal_distribution_y = std::move(margin_y);
    }

    CopulaType type() const override { return CopulaType::Independence; }
    double theta_minimum() const override { return 0.0; }
    double theta_maximum() const override { return 0.0; }
    int number_of_copula_parameters() const override { return 0; }
    std::vector<double> get_copula_parameters() const override { return {}; }
    void set_copula_parameters(const std::vector<double>&) override {}
    math::linalg::Matrix2D parameter_constraints(const std::vector<double>&,
                                                  const std::vector<double>&) const override {
        return {};
    }
    std::optional<std::string> validate_parameter(double, bool) const override {
        return std::nullopt;
    }
    double pdf(double, double) const override { return 1.0; }
    double cdf(double u, double v) const override { return u * v; }
    double conditional_cdf(double, double v) const override { return v; }
    double inverse_conditional_cdf(double, double t) const override { return t; }
    std::array<double, 2> inverse_cdf(double u, double v) const override { return {u, v}; }
    double upper_tail_dependence() const override { return 0.0; }
    double lower_tail_dependence() const override { return 0.0; }
    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<IndependenceCopula>(clone_marginal(marginal_distribution_x),
                                                     clone_marginal(marginal_distribution_y));
    }
};

}  // namespace corehydro::numerics::distributions::copulas
