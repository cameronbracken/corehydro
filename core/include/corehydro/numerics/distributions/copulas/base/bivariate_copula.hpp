// ported from: Numerics/Distributions/Bivariate Copulas/Base/BivariateCopula.cs @ 7e8e8d1
//
// Abstract base for every bivariate copula. Folds IBivariateCopula's members directly into
// the base -- the Phase 1/2 pattern (interfaces fold into the abstract base rather than
// being ported as a separate file) already used by UnivariateDistributionBase and
// MultivariateDistribution, applicable here because every C# copula class derives from
// `BivariateCopula : IBivariateCopula` and nothing implements only the interface.
//
// Declares the copula-core surface (theta + bounds + validated setter, PDF/CDF/InverseCDF,
// tail dependence, parameter accessors) as pure/concrete virtuals and promotes the shared
// concrete helpers (LogPDF, joint-exceedance probabilities, GenerateRandomValues, the three
// log-likelihood objectives) with the exact C# NaN/Inf guard and exception behavior.
//
// Marginal distributions: `marginal_distribution_x`/`marginal_distribution_y` are
// std::shared_ptr<UnivariateDistributionBase>, a DELIBERATE DEVIATION from the rest of the
// C++ core (UnivariateDistributionBase is normally owned via unique_ptr, e.g. the
// univariate factory).
//
// Clone() deep-copies marginals (Task 8 / Numerics v2.1.4 2a0357a): every concrete copula's
// `Clone()` (see clayton_copula.hpp) now routes MarginalDistributionX/Y through the new
// protected static `clone_marginal` below before assigning them to the new instance, mirroring
// the C# `BivariateCopula.CloneMarginal`. Distributions memoize internal state lazily, so two
// clones sharing a marginal instance are not safe to use concurrently (e.g. one clone per
// thread in a Monte Carlo framework or a parallel likelihood evaluation) -- deep-copying closes
// that hazard. This does NOT change BivariateCopulaEstimation's MPL/IFM fits (see
// bivariate_copula_estimation.hpp): their per-evaluation `copula.clone()` only ever READS the
// clone's marginals (`pseudo_log_likelihood` doesn't touch marginals at all; `ifm_log_likelihood`
// only calls `->cdf(...)` on them) before the clone is discarded at the end of that one
// evaluation, so a deep copy and a shared alias produce identical results there -- the extra
// clone allocation is the only cost. MLE's own objective explicitly clones a *fresh*
// UnivariateDistributionBase per evaluation regardless (see `detail::mle`), unaffected either
// way. A unique_ptr member could not reproduce this without always deep-copying on every
// BivariateCopula::clone() call, which is now exactly the desired behavior anyway -- shared_ptr
// is kept for its lower churn on the existing binding/glue call sites, not because clone()
// still needs aliasing.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/sampling/latin_hypercube.hpp"

namespace corehydro::numerics::distributions::copulas {

using corehydro::numerics::distributions::UnivariateDistributionBase;

class BivariateCopula {
   public:
    virtual ~BivariateCopula() = default;

    // --- Identity / parameters ---
    virtual CopulaType type() const = 0;

    double theta() const { return theta_; }
    void set_theta(double value) {
        parameters_valid_ = !validate_parameter(value, false).has_value();
        theta_ = value;
    }

    virtual double theta_minimum() const = 0;
    virtual double theta_maximum() const = 0;

    bool parameters_valid() const { return parameters_valid_; }

    // The X/Y marginal distributions for the copula (see the shared_ptr rationale above).
    // Public and mutable, mirroring the C# `IUnivariateDistribution? MarginalDistributionX
    // { get; set; }` auto-property -- the global "never mutate" rule is relaxed for these
    // binding/model objects (see .claude/CLAUDE.md's Mutation note), matching the upstream
    // design and the existing CompetingRisks::minimum_of_random_variables precedent.
    std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x;
    std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y;

    virtual int number_of_copula_parameters() const = 0;
    virtual std::vector<double> get_copula_parameters() const = 0;
    virtual void set_copula_parameters(const std::vector<double>& parameters) = 0;

    // Shape [number_of_copula_parameters, 2]; column 0 = lower bound, column 1 = upper
    // bound. Matrix2D (std::vector<std::vector<double>>) mirrors the C# `double[,]`, the
    // same convention LatinHypercube::random/median already use for a 2D fixture-facing
    // return value.
    virtual math::linalg::Matrix2D parameter_constraints(
        const std::vector<double>& sample_data_x, const std::vector<double>& sample_data_y) const = 0;

    // Mirrors the C# `ArgumentOutOfRangeException? ValidateParameter(double, bool)`:
    // nullopt means "no exception" (parameter accepted); a value is the exception message,
    // thrown (std::out_of_range) when throw_exception is true, else just returned. NOTE:
    // ArchimedeanCopula's override does not honor "non-nullopt only when invalid" -- see
    // archimedean_copula.hpp and docs/upstream-csharp-issues.md for the upstream bug this
    // reproduces verbatim.
    virtual std::optional<std::string> validate_parameter(double parameter,
                                                            bool throw_exception) const = 0;

    // --- Distribution functions ---
    virtual double pdf(double u, double v) const = 0;

    double log_pdf(double u, double v) const {
        double f = pdf(u, v);
        if (std::isnan(f) || std::isinf(f) || f <= 0.0) return -kInf;
        return std::log(f);
    }

    virtual double cdf(double u, double v) const = 0;
    virtual double conditional_cdf(double u, double v) const {
        double u_plus = std::min(u + 1.0e-6, 1.0);
        double u_minus = std::max(u - 1.0e-6, 0.0);
        return (cdf(u_plus, v) - cdf(u_minus, v)) / (u_plus - u_minus);
    }
    virtual double inverse_conditional_cdf(double u, double t) const {
        return inverse_cdf(u, t)[1];
    }
    virtual std::array<double, 2> inverse_cdf(double u, double v) const = 0;

    virtual double upper_tail_dependence() const = 0;
    virtual double lower_tail_dependence() const = 0;

    virtual std::unique_ptr<BivariateCopula> clone() const = 0;

    // Returns the OR joint exceedance probability: when either variable exceeds a
    // particular threshold.
    double or_joint_exceedance_probability(double u, double v) const { return 1.0 - cdf(u, v); }

    // Returns the AND joint exceedance probability: when both variables exceed a
    // particular threshold simultaneously.
    double and_joint_exceedance_probability(double u, double v) const {
        return 1.0 - u - v + cdf(u, v);
    }

    // Generate random values via Latin Hypercube sampling of the reduced (u, v) plane,
    // back-transformed through the marginals' InverseCDF when both are set. Matrix2D row i
    // = sample i's (x, y) pair, mirroring the C# `double[,] GenerateRandomValues`.
    math::linalg::Matrix2D generate_random_values(int sample_size, int seed = -1) const {
        auto rand = corehydro::numerics::sampling::LatinHypercube::random(sample_size, 2, seed);
        math::linalg::Matrix2D sample(static_cast<std::size_t>(sample_size),
                                       std::vector<double>(2));
        for (int i = 0; i < sample_size; ++i) {
            auto vals = inverse_cdf(rand[static_cast<std::size_t>(i)][0],
                                     rand[static_cast<std::size_t>(i)][1]);
            if (marginal_distribution_x && marginal_distribution_y) {
                sample[static_cast<std::size_t>(i)][0] = marginal_distribution_x->inverse_cdf(vals[0]);
                sample[static_cast<std::size_t>(i)][1] = marginal_distribution_y->inverse_cdf(vals[1]);
            } else {
                sample[static_cast<std::size_t>(i)][0] = vals[0];
                sample[static_cast<std::size_t>(i)][1] = vals[1];
            }
        }
        return sample;
    }

    // The pseudo log-likelihood function. sample_data_x/y should be the plotting positions
    // of the data when used for estimation.
    double pseudo_log_likelihood(const std::vector<double>& sample_data_x,
                                  const std::vector<double>& sample_data_y) const {
        check_sample_lengths(sample_data_x, sample_data_y);
        double log_lh = 0.0;
        for (std::size_t i = 0; i < sample_data_x.size(); ++i)
            log_lh += log_pdf(sample_data_x[i], sample_data_y[i]);
        if (std::isnan(log_lh) || std::isinf(log_lh)) return -kInf;
        return log_lh;
    }

    // The inference-from-margins (IFM) log-likelihood function. The marginal distributions
    // are assumed to have already been estimated independently.
    double ifm_log_likelihood(const std::vector<double>& sample_data_x,
                               const std::vector<double>& sample_data_y) const {
        check_sample_lengths(sample_data_x, sample_data_y);
        if (!marginal_distribution_x || !marginal_distribution_y)
            throw std::out_of_range("There must be 2 marginal distributions to evaluate.");
        double log_lh = 0.0;
        for (std::size_t i = 0; i < sample_data_x.size(); ++i)
            log_lh += log_pdf(marginal_distribution_x->cdf(sample_data_x[i]),
                               marginal_distribution_y->cdf(sample_data_y[i]));
        if (std::isnan(log_lh) || std::isinf(log_lh)) return -kInf;
        return log_lh;
    }

    // The full log-likelihood function. The marginal distributions are estimated
    // simultaneously with the copula.
    double log_likelihood(const std::vector<double>& sample_data_x,
                           const std::vector<double>& sample_data_y) const {
        check_sample_lengths(sample_data_x, sample_data_y);
        if (!marginal_distribution_x || !marginal_distribution_y)
            throw std::out_of_range("There must be 2 marginal distributions to evaluate.");
        double log_lh = 0.0;
        for (std::size_t i = 0; i < sample_data_x.size(); ++i) {
            log_lh += log_pdf(marginal_distribution_x->cdf(sample_data_x[i]),
                               marginal_distribution_y->cdf(sample_data_y[i])) +
                      marginal_distribution_x->log_pdf(sample_data_x[i]) +
                      marginal_distribution_y->log_pdf(sample_data_y[i]);
        }
        if (std::isnan(log_lh) || std::isinf(log_lh)) return -kInf;
        return log_lh;
    }

   protected:
    static constexpr double kInf = std::numeric_limits<double>::infinity();

    double theta_ = 0.0;
    bool parameters_valid_ = true;

    // Deep-copies an attached marginal for Clone() (ported from the new C# protected static
    // `BivariateCopula.CloneMarginal`, Numerics v2.1.4 / 2a0357a). C#'s version only clones
    // when the marginal `is UnivariateDistributionBase` (its field is the wider
    // `IUnivariateDistribution` interface) and returns any other implementation by reference.
    // Every marginal in THIS port is already declared std::shared_ptr<UnivariateDistributionBase>
    // (see the file header), so that runtime type test is unconditionally true here -- every
    // non-null marginal is always cloned. A null (unattached) marginal passes through
    // unchanged, mirroring the C# null branch.
    static std::shared_ptr<UnivariateDistributionBase> clone_marginal(
        const std::shared_ptr<UnivariateDistributionBase>& marginal) {
        if (!marginal) return marginal;
        return std::shared_ptr<UnivariateDistributionBase>(marginal->clone());
    }

   private:
    static void check_sample_lengths(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() < 2 || y.size() < 2)
            throw std::out_of_range("There must be at least two items in each sample.");
        if (x.size() != y.size())
            throw std::out_of_range("The sample data arrays must be the same length.");
    }
};

}  // namespace corehydro::numerics::distributions::copulas
