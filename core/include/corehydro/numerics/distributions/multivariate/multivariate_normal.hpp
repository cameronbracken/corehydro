// ported from: Numerics/Distributions/Multivariate/MultivariateNormal.cs @ 7e8e8d1
//
// The Multivariate Normal (Gaussian) distribution, dimensions >= 1, mean vector mu and
// covariance matrix Sigma: the deterministic core (ctors, PDF/LogPDF/CDF for dim 1-2,
// Mahalanobis distance, sampling, the closed-form bivariate CDF machinery), the
// Genz-Bretz MVNDST quasi-Monte-Carlo integrator backing the CDF for dim >= 3, and (this
// task, v2.1.4) the Try/Marginal/Conditional API. Member order mirrors the C# source
// throughout.
//
// v2.1.4 additions (this task):
//   - `validate_parameters` gained finite-mean/finite-covariance checks (NaN/Inf rejected
//     BEFORE the square/dimension/positive-definite checks, exact C# order).
//   - `try_set_parameters`/`try_set_covariance`: non-throwing mutable-covariance setters
//     for samplers/likelihood loops that swap covariances per evaluation -- an infeasible
//     proposal marks `is_density_valid() == false` (PDF -> 0, LogPDF -> -inf) instead of
//     throwing. Mirrors C# `TrySetParameters`/`TrySetCovariance`/`IsDensityValid`.
//   - `marginal(indices)`: the sub-mean/sub-covariance at the given dimension indices (the
//     marginal of a joint Gaussian is the corresponding sub-Gaussian).
//   - `conditional(observed_indices, observed_values)`: the Schur-complement conditional
//     distribution of the unobserved dimensions given fixed values at the observed ones.
//   - COVSRT's degenerate-diagonal "permute limits and/or rows" loop was FIXED upstream
//     (loop bounds `j >= 0`/`l <= j`/`l <= i-1`, packed-index `l*(l+1)/2`, the swap-offset
//     `-1`, and the `IJ` decrement `- k - 1`) -- transcribed verbatim from the fixed C#;
//     see the `covsrt` method comment (the old bounds-guard divergence note is retired,
//     since the fixed loop no longer produces an out-of-range access on this port's
//     fixture-exercised singular/rank-deficient inputs).
//   - `MVNDNT` changed from `private double` (always returning the dead constant `0.0`) to
//     `private void`, with `MVNDST` now setting `INFORM = 0` explicitly before calling it --
//     a pure shape mirror with no behavior change (see the `mvndst`/`mvndnt` comments).
//
// Divergence notes (see also docs/upstream-csharp-issues.md for the running log):
//   - `Variance`/`StandardDeviation` are computed on every call rather than lazily cached
//     in nullable backing fields (the C# `_variance`/`_standardDeviation` pattern) --
//     same simplification and rationale as the Dirichlet port (dirichlet.hpp): the C#
//     caching is a micro-optimization orthogonal to the math, and this port's "never
//     mutate a value type without reason" convention prefers recomputation. Results are
//     identical.
//   - `CreateCorrelationMatrix`'s `invSqrtD = !D` (C# `Matrix.operator!` = a full
//     LU-decomposition-based inverse, not yet ported -- see matrix.hpp's header note) is
//     implemented here as an elementwise reciprocal of the diagonal instead. `D` is
//     provably diagonal at that call site (built via `Matrix::diagonal` + elementwise
//     `sqrt`), and the LU inverse of a diagonal matrix reduces exactly to elementwise
//     reciprocal (L=I, U=D, so Ainv=diag(1/d_ii) with no extra rounding) -- bit-identical
//     to the general algorithm for this diagonal input, without porting the general
//     LU-based `Matrix::inverse()` this port doesn't otherwise need.
//   - `clone()`'s `MVNUNI` copy is a VALUE copy (an independent RNG stream) rather than
//     the C# `_MVNUNI = this._MVNUNI` reference-sharing semantics (original and clone
//     alias the SAME `Random`/stream in C#, so later draws on either object consume from
//     one shared sequence). `Clone()` is not fixture-observable (mirrors the precedent
//     set for Dirichlet/Multinomial/BivariateEmpirical's Clone in fixtures/README.md),
//     and an independent stream is the safer, more intuitive choice for a value-oriented
//     C++ port.
//   - `ParametersValid`/`_parametersValid` and `_covSRTed` are ported verbatim as fields
//     but are DEAD STATE in the C# source: `_parametersValid` is initialized `true` and
//     never reassigned anywhere in the class (`SetParameters` throws on invalid input
//     rather than flipping the flag, so any live instance reports valid), and
//     `_covSRTed` is declared, copied by `Clone()`, and never read or written anywhere
//     else. Kept for structural fidelity; both are effectively constant for this class.
//   - The C# null-checks on the `mean`/`covariance` constructor arguments
//     (`ArgumentOutOfRangeException` on `null`) have no C++ equivalent (references/
//     vectors cannot be null) and are omitted from `validate_parameters`.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/linalg/cholesky_decomposition.hpp"
#include "corehydro/numerics/math/linalg/decomposition_method.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/singular_value_decomposition.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/special/erf.hpp"
#include "corehydro/numerics/sampling/latin_hypercube.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/sampling/stratification_bin.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace la = corehydro::numerics::math::linalg;
namespace sf = corehydro::numerics::math::special;

class MultivariateNormal : public MultivariateDistribution {
   public:
    // Constructs a multivariate Gaussian with zero mean vector and identity covariance.
    explicit MultivariateNormal(int dimension) {
        std::vector<double> mean(static_cast<std::size_t>(dimension), 0.0);
        set_parameters(std::move(mean), la::Matrix::identity(dimension).to_array());
    }

    MultivariateNormal(int dimension, la::DecompositionMethod decomposition)
        : decomposition_(decomposition) {
        std::vector<double> mean(static_cast<std::size_t>(dimension), 0.0);
        set_parameters(std::move(mean), la::Matrix::identity(dimension).to_array());
    }

    // Constructs a new Multivariate Normal distribution with an identity covariance matrix.
    explicit MultivariateNormal(std::vector<double> mean) {
        int dim = static_cast<int>(mean.size());
        set_parameters(std::move(mean), la::Matrix::identity(dim).to_array());
    }


    MultivariateNormal(std::vector<double> mean, la::DecompositionMethod decomposition)
        : decomposition_(decomposition) {
        int dim = static_cast<int>(mean.size());
        set_parameters(std::move(mean), la::Matrix::identity(dim).to_array());
    }

    // Constructs a new Multivariate Normal distribution.
    MultivariateNormal(std::vector<double> mean, std::vector<std::vector<double>> covariance) {
        set_parameters(std::move(mean), std::move(covariance));
    }


    MultivariateNormal(std::vector<double> mean, std::vector<std::vector<double>> covariance,
                       la::DecompositionMethod decomposition)
        : decomposition_(decomposition) {
        set_parameters(std::move(mean), std::move(covariance));
    }

    // --- MVNUNI: the uniform(0,1) generator required to compute the multivariate CDF for
    // dimensions greater than 2 (C# `public Random MVNUNI { get; set; }`). This port has
    // no generic RNG interface (every ported sampling routine takes a concrete
    // MersenneTwister -- see multinomial.hpp/dirichlet.hpp for the same choice), so the
    // property is exposed as a seed-setter mirroring how the C# tests assign it
    // (`MVNUNI = new MersenneTwister(12345)`), plus direct access for MVNDST's internals.
    void set_mvnuni_seed(int seed) { mvnuni_ = sampling::MersenneTwister(static_cast<std::uint32_t>(seed)); }
    sampling::MersenneTwister& mvnuni() const { return mvnuni_; }

    // The maximum number of function evaluations allowed when computing the multivariate
    // CDF. Default = 1,000 x D.
    int max_evaluations() const { return max_evaluations_; }
    void set_max_evaluations(int value) { max_evaluations_ = value; }

    // The absolute error tolerance when computing the multivariate CDF.
    double absolute_error() const { return absolute_error_; }
    void set_absolute_error(double value) { absolute_error_ = value; }

    // The relative error tolerance when computing the multivariate CDF.
    double relative_error() const { return relative_error_; }
    void set_relative_error(double value) { relative_error_ = value; }

    // --- Identity / parameters ---
    int dimension() const override { return dimension_; }
    MultivariateDistributionType type() const override {
        return MultivariateDistributionType::MultivariateNormal;
    }
    std::string display_name() const override { return "Multivariate Normal"; }
    std::string short_display_name() const override { return "Multi-N"; }
    bool parameters_valid() const override { return parameters_valid_; }

    // Gets the mean vector of the distribution.
    const std::vector<double>& mean() const { return mean_; }

    // Gets the median vector of the distribution (== mean).
    const std::vector<double>& median() const { return mean_; }

    // Gets the mode vector of the distribution (== mean).
    const std::vector<double>& mode() const { return mean_; }

    // Gets the variance vector of the distribution (computed each call -- see file header).
    std::vector<double> variance() const { return covariance_.diagonal(); }

    // Gets the standard deviation vector of the distribution (computed each call).
    std::vector<double> standard_deviation() const {
        std::vector<double> v = variance();
        for (double& x : v) x = std::sqrt(x);
        return v;
    }

    // Gets the Variance-Covariance matrix for the distribution.
    la::Matrix2D covariance() const { return covariance_.to_array(); }

    la::DecompositionMethod decomposition() const { return decomposition_; }

    // Element accessor (not in the C# source -- added for fixture/dispatch convenience,
    // mirroring Dirichlet's/Multinomial's `covariance(i, j)` method).
    double covariance(int i, int j) const { return covariance_(i, j); }

    // Determines if the covariance matrix is positive definite.
    bool is_positive_definite() const {
        return decomposition_ == la::DecompositionMethod::Cholesky
                   ? cholesky_->is_positive_definite()
                   : rank_ == dimension_;
    }

    // --- Parameter setting / validation ---

    // Set the distribution parameters.
    void set_parameters(std::vector<double> mean, std::vector<std::vector<double>> covariance) {
        std::optional<la::SingularValueDecomposition> singular_values;
        validate_parameters_impl(mean, covariance, true, singular_values);

        set_parameters_core(std::move(mean), std::move(covariance), std::move(singular_values));
    }

    void set_mean(std::vector<double> mean) {
        if (mean.size() != static_cast<std::size_t>(dimension_))
            throw std::out_of_range("Mean length must match covariance dimension.");
        for (double value : mean)
            if (!std::isfinite(value)) throw std::out_of_range("Mean values must be finite.");
        mean_ = std::move(mean);
    }

   private:
    void set_parameters_core(std::vector<double> mean, std::vector<std::vector<double>> covariance,
                             std::optional<la::SingularValueDecomposition> singular_values) {

        dimension_ = static_cast<int>(mean.size());
        mean_ = std::move(mean);
        covariance_ = la::Matrix(std::move(covariance));
        if (decomposition_ == la::DecompositionMethod::Cholesky) {
            cholesky_.emplace(covariance_);
            double lndet = cholesky_->log_determinant();
            lnconstant_ = -(std::log(2.0 * kPi) * static_cast<double>(mean_.size()) + lndet) * 0.5;
            factor_ = cholesky_->l();
            svd_.reset();
            nullspace_ = la::Matrix(0, 0);
            rank_ = dimension_;
        } else {
            factorize_with_singular_values(std::move(*singular_values));
        }

        // Set up parameters for MVN CDF
        correlation_matrix_created_ = false;
        max_evaluations_ = 1000 * dimension_;
        lower_.assign(static_cast<std::size_t>(dimension_), -kInf);
        upper_.assign(static_cast<std::size_t>(dimension_), 0.0);
        infin_.assign(static_cast<std::size_t>(dimension_), 0);
        // NL reassignment quirk (MultivariateNormal.cs:254): NL defaults to 500 (see the
        // field declaration below, sized to match the C# Fortran-derived MVNDFN_A/B/INFI
        // working-storage arrays added in the second commit) but is REASSIGNED here, per
        // instance, to the number of below-diagonal correlation pairs. Only `_correl`'s
        // allocation and (in the second commit) `MVNDFN`'s local `Y` scratch array use
        // the reassigned value -- the fixed-500-sized working arrays are untouched by it.
        nl_ = static_cast<int>(dimension_ * (dimension_ - 1) / 2.0);
        // Create empty arrays
        correl_.assign(static_cast<std::size_t>(nl_), 0.0);
        correlation_ = la::Matrix(dimension_, dimension_);
    }

   public:

    // Validate the parameters. Returns nullopt if valid; if `throw_exception` is false and
    // invalid, returns the error message instead of throwing. v2.1.4: the mean/covariance
    // finiteness checks (below) respect `throw_exception` faithfully -- unlike the
    // Cholesky-based check further down, which always throws regardless (see that check's
    // own comment).
    std::optional<std::string> validate_parameters(const std::vector<double>& mean,
                                                     const std::vector<std::vector<double>>& covariance,
                                                     bool throw_exception) const {
        std::optional<la::SingularValueDecomposition> singular_values;
        return validate_parameters_impl(mean, covariance, throw_exception, singular_values);
    }

   private:
    std::optional<std::string> validate_parameters_impl(
        const std::vector<double>& mean, const std::vector<std::vector<double>>& covariance,
        bool throw_exception,
        std::optional<la::SingularValueDecomposition>& singular_values) const {
        for (double v : mean) {
            if (!std::isfinite(v)) {
                if (throw_exception) throw std::out_of_range("Mean values must be finite.");
                return "Mean values must be finite.";
            }
        }
        la::Matrix m(covariance);
        if (!m.is_square()) {
            if (throw_exception) throw std::out_of_range("Covariance matrix must be square.");
            return "Covariance matrix must be square.";
        }
        if (m.number_of_rows() != static_cast<int>(mean.size())) {
            if (throw_exception) throw std::out_of_range("Mean length must match covariance dimension.");
            return "Mean length must match covariance dimension.";
        }
        for (int i = 0; i < m.number_of_rows(); ++i) {
            for (int j = 0; j < m.number_of_columns(); ++j) {
                if (!std::isfinite(m(i, j))) {
                    if (throw_exception) throw std::out_of_range("Covariance values must be finite.");
                    return "Covariance values must be finite.";
                }
            }
        }
        // Cholesky try/propagate quirk: the C# source constructs `new
        // CholeskyDecomposition(m)` directly here, with no try/catch. This port's
        // CholeskyDecomposition ctor (like the C# one) unconditionally THROWS on a
        // non-positive-definite or NaN-diagonal input rather than returning a status (see
        // cholesky_decomposition.hpp) -- so that exception always propagates out of
        // validate_parameters regardless of `throw_exception`, and the
        // `!chol.is_positive_definite()` check below never actually fires (construction
        // already succeeded by the time it runs). Ported verbatim anyway for structural
        // fidelity with the C# source. `try_set_parameters` below absorbs this always-throw
        // behavior in its own try/catch, exactly mirroring C#'s `TrySetParameters`.
        if (decomposition_ == la::DecompositionMethod::Cholesky) {
            try {
                la::CholeskyDecomposition chol(m);
                (void)chol;
            } catch (...) {
                if (throw_exception)
                    throw std::out_of_range("Covariance matrix is not positive-definite.");
                return "Covariance matrix is not positive-definite.";
            }
        } else {
            try {
                la::SingularValueDecomposition candidate(m);
                if (!is_symmetric_positive_semidefinite(candidate, m)) {
                    if (throw_exception)
                        throw std::out_of_range(
                            "Covariance matrix is not symmetric positive-semi-definite.");
                    return "Covariance matrix is not symmetric positive-semi-definite.";
                }
                singular_values.emplace(std::move(candidate));
            } catch (const std::out_of_range&) {
                throw;
            } catch (...) {
                if (throw_exception)
                    throw std::out_of_range("The covariance matrix decomposition did not converge.");
                return "The covariance matrix decomposition did not converge.";
            }
        }
        return std::nullopt;
    }

   public:

    // Attempts to set mean+covariance without throwing: the covariance is factorized
    // eagerly, and when it is not positive-definite (or otherwise invalid) the density is
    // marked invalid -- `log_pdf()` returns -inf and `pdf()` returns 0 until a valid
    // covariance is set. Mirrors C# `TrySetParameters` (the non-throwing mutable path for
    // samplers/likelihood loops that swap covariances per evaluation, e.g. MCMC proposals
    // of correlation parameters: an infeasible proposal scores -inf and is rejected instead
    // of raising an exception). `set_parameters()` keeps its throwing contract.
    bool try_set_parameters(std::vector<double> mean, std::vector<std::vector<double>> covariance) {
        try {
            // validate_parameters(..., false) still THROWS on a non-positive-definite
            // covariance (the Cholesky-unconditional-throw quirk noted above) -- this
            // try/catch absorbs that, mirroring C#'s own `try { if (ValidateParameters(...)
            // is null) {...} } catch (Exception) {...}`.
            std::optional<la::SingularValueDecomposition> singular_values;
            if (!validate_parameters_impl(mean, covariance, false, singular_values).has_value()) {
                set_parameters_core(std::move(mean), std::move(covariance),
                                    std::move(singular_values));
                density_valid_ = true;
                return true;
            }
        } catch (...) {
            // Swallowed exactly like C#'s catch(Exception) -- falls through to the
            // invalid-density return below.
        }
        density_valid_ = false;
        return false;
    }

    // Attempts to swap the covariance matrix in place (the mean is kept) without throwing;
    // see `try_set_parameters` for the invalid-density contract. Mirrors C#
    // `TrySetCovariance`.
    bool try_set_covariance(std::vector<std::vector<double>> covariance) {
        return try_set_parameters(mean_, std::move(covariance));
    }

    // Whether the current covariance factorization supports density evaluation. False
    // after a failed try_set_parameters()/try_set_covariance() until a valid covariance is
    // set. Mirrors C# `IsDensityValid`.
    bool is_density_valid() const { return density_valid_; }

    // Returns the marginal distribution over a subset of the dimensions -- the sub-mean
    // and sub-covariance at the given indices (the marginal of a joint Gaussian is the
    // corresponding sub-Gaussian). `indices` must be non-empty, in range, and without
    // repeats (see validate_indices()). Mirrors C# `Marginal(params int[] indices)`.
    MultivariateNormal marginal(const std::vector<int>& indices) const {
        validate_indices(indices, dimension_);
        std::vector<double> sub_mean(indices.size());
        std::vector<std::vector<double>> sub_covariance(indices.size(),
                                                          std::vector<double>(indices.size()));
        for (std::size_t i = 0; i < indices.size(); ++i) {
            sub_mean[i] = mean_[static_cast<std::size_t>(indices[i])];
            for (std::size_t j = 0; j < indices.size(); ++j)
                sub_covariance[i][j] = covariance_(indices[i], indices[j]);
        }
        return MultivariateNormal(std::move(sub_mean), std::move(sub_covariance));
    }

    // Returns the conditional distribution of the remaining dimensions given observed
    // values at a subset of the dimensions: mu_c = mu1 + Sigma12*Sigma22^-1*(x2 - mu2) and
    // Sigma_c = Sigma11 - Sigma12*Sigma22^-1*Sigma21 (the Schur complement), with Sigma22
    // solved through its Cholesky factorization. `observed_indices` must be non-empty, in
    // range, without repeats, and leave at least one dimension unobserved;
    // `observed_values` must parallel `observed_indices`. Mirrors C#
    // `Conditional(int[] observedIndices, double[] observedValues)`.
    // Reference: Eaton, M.L. (1983). Multivariate Statistics: A Vector Space Approach.
    // Wiley. (The Gaussian conditioning identities.)
    MultivariateNormal conditional(const std::vector<int>& observed_indices,
                                    const std::vector<double>& observed_values) const {
        validate_indices(observed_indices, dimension_);
        if (observed_values.size() != observed_indices.size())
            throw std::out_of_range("The observed values must parallel the observed indices.");
        if (static_cast<int>(observed_indices.size()) >= dimension_)
            throw std::out_of_range("At least one dimension must remain unobserved.");

        // The remaining (free) dimensions in ascending order.
        std::vector<bool> is_observed(static_cast<std::size_t>(dimension_), false);
        for (int index : observed_indices) is_observed[static_cast<std::size_t>(index)] = true;
        std::vector<int> free;
        free.reserve(static_cast<std::size_t>(dimension_) - observed_indices.size());
        for (int i = 0; i < dimension_; ++i)
            if (!is_observed[static_cast<std::size_t>(i)]) free.push_back(i);

        // Partition: Sigma22 (observed), Sigma12 (free x observed), and the innovation
        // z = x2 - mu2.
        int n_observed = static_cast<int>(observed_indices.size());
        int n_free = static_cast<int>(free.size());
        la::Matrix sigma22(n_observed, n_observed);
        for (int i = 0; i < n_observed; ++i)
            for (int j = 0; j < n_observed; ++j)
                sigma22(i, j) = covariance_(observed_indices[static_cast<std::size_t>(i)],
                                            observed_indices[static_cast<std::size_t>(j)]);
        // Cholesky-throws-unconditionally quirk (see validate_parameters' comment above):
        // a non-positive-definite Sigma22 throws out of the CholeskyDecomposition ctor
        // itself, so `!cholesky22.is_positive_definite()` below is dead code (construction
        // would already have thrown) -- ported verbatim for structural fidelity with C#,
        // whose own `!cholesky22.IsPositiveDefinite` check is equally unreachable there.
        la::CholeskyDecomposition cholesky22(sigma22);
        if (!cholesky22.is_positive_definite())
            throw std::out_of_range("The observed sub-covariance is not positive-definite.");

        std::vector<double> z(static_cast<std::size_t>(n_observed));
        for (int i = 0; i < n_observed; ++i)
            z[static_cast<std::size_t>(i)] =
                observed_values[static_cast<std::size_t>(i)] -
                mean_[static_cast<std::size_t>(observed_indices[static_cast<std::size_t>(i)])];
        la::Vector solved_z = cholesky22.solve(la::Vector(z));

        // Solve Sigma22^-1*Sigma21 column-by-column (Sigma21 columns are the free rows of
        // Sigma12^T).
        std::vector<la::Vector> solved_columns;
        solved_columns.reserve(static_cast<std::size_t>(n_free));
        for (int i = 0; i < n_free; ++i) {
            std::vector<double> column(static_cast<std::size_t>(n_observed));
            for (int j = 0; j < n_observed; ++j)
                column[static_cast<std::size_t>(j)] =
                    covariance_(observed_indices[static_cast<std::size_t>(j)], free[static_cast<std::size_t>(i)]);
            solved_columns.push_back(cholesky22.solve(la::Vector(column)));
        }

        std::vector<double> conditional_mean(static_cast<std::size_t>(n_free));
        std::vector<std::vector<double>> conditional_covariance(
            static_cast<std::size_t>(n_free), std::vector<double>(static_cast<std::size_t>(n_free)));
        for (int i = 0; i < n_free; ++i) {
            double mean_adjustment = 0.0;
            for (int j = 0; j < n_observed; ++j)
                mean_adjustment += covariance_(free[static_cast<std::size_t>(i)],
                                                observed_indices[static_cast<std::size_t>(j)]) *
                                    solved_z[j];
            conditional_mean[static_cast<std::size_t>(i)] =
                mean_[static_cast<std::size_t>(free[static_cast<std::size_t>(i)])] + mean_adjustment;
            for (int k = 0; k < n_free; ++k) {
                double schur = 0.0;
                for (int j = 0; j < n_observed; ++j)
                    schur += covariance_(free[static_cast<std::size_t>(i)],
                                          observed_indices[static_cast<std::size_t>(j)]) *
                             solved_columns[static_cast<std::size_t>(k)][j];
                conditional_covariance[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] =
                    covariance_(free[static_cast<std::size_t>(i)], free[static_cast<std::size_t>(k)]) - schur;
            }
        }
        return MultivariateNormal(std::move(conditional_mean), std::move(conditional_covariance));
    }

    // --- Distribution functions ---

    // The Probability Density Function (PDF) of the distribution evaluated at a point X.
    double pdf(const std::vector<double>& x) const override {
        if (!density_valid_) return 0.0;
        if (decomposition_ == la::DecompositionMethod::SingularValue && !is_on_support(x))
            return 0.0;
        return std::exp(-0.5 * mahalanobis(x) + lnconstant_);
    }

    // Returns the natural log of the PDF.
    double log_pdf(const std::vector<double>& x) const override {
        if (!density_valid_) return -kInf;
        if (decomposition_ == la::DecompositionMethod::SingularValue && !is_on_support(x))
            return -kInf;
        double f = -0.5 * mahalanobis(x) + lnconstant_;
        if (std::isnan(f) || std::isinf(f)) return -kInf;
        return f;
    }

    // Gets the Mahalanobis distance between a sample and this distribution.
    double mahalanobis(const std::vector<double>& x) const {
        if (x.size() != mean_.size())
            throw std::out_of_range("The vector must be the same dimension as the distribution.");
        std::vector<double> z(mean_.size());
        for (std::size_t i = 0; i < x.size(); ++i) z[i] = x[i] - mean_[i];
        la::Vector a = decomposition_ == la::DecompositionMethod::Cholesky
                           ? cholesky_->solve(la::Vector(z))
                           : svd_->solve(la::Vector(z), svd_threshold_);
        double b = 0.0;
        for (std::size_t i = 0; i < z.size(); ++i) b += a[static_cast<int>(i)] * z[i];
        return b;
    }

    // The Cumulative Distribution Function (CDF) for the distribution evaluated at a point X.
    double cdf(const std::vector<double>& x) const override {
        if (dimension_ == 1) {
            double std_dev = std::sqrt(covariance_(0, 0));
            if (std_dev == 0.0) return x[0] >= mean_[0] ? 1.0 : 0.0;
            double z = (x[0] - mean_[0]) / std_dev;
            return Normal::standard_cdf(z);
        } else if (dimension_ == 2) {
            double sigma1 = std::sqrt(covariance_(0, 0));
            double sigma2 = std::sqrt(covariance_(1, 1));
            double rho = covariance_(0, 1) / (sigma1 * sigma2);
            if (std::isnan(rho)) {
                bool at_mean = x.size() == 2 && x[0] == mean_[0] && x[1] == mean_[1];
                return at_mean ? 1.0 : 0.0;
            }
            double z = (x[0] - mean_[0]) / sigma1;
            double w = (x[1] - mean_[1]) / sigma2;
            return bivariate_cdf(-z, -w, rho);
        } else {
            // Construct inputs for the GenzBretz method
            for (std::size_t i = 0; i < x.size(); ++i)
                upper_[i] = (x[i] - mean_[i]) / std::sqrt(covariance_(static_cast<int>(i), static_cast<int>(i)));

            double error = 0, val = 0;
            int inform = 0;

            if (!correlation_matrix_created_) create_correlation_matrix();
            mvndst(dimension_, lower_, upper_, infin_, correl_, max_evaluations_, absolute_error_,
                   relative_error_, error, val, inform);
            return val;
        }
    }

    // Computes the probability of a value occurring in the interval.
    double interval(std::vector<double> lower, std::vector<double> upper) const {
        std::vector<int> infin(upper.size(), 2);
        for (std::size_t i = 0; i < upper.size(); ++i) {
            lower[i] = (lower[i] - mean_[i]) / std::sqrt(covariance_(static_cast<int>(i), static_cast<int>(i)));
            upper[i] = (upper[i] - mean_[i]) / std::sqrt(covariance_(static_cast<int>(i), static_cast<int>(i)));
        }

        double error = 0, val = 0;
        int inform = 0;

        if (!correlation_matrix_created_) create_correlation_matrix();
        mvndst(dimension_, lower, upper, infin, correl_, max_evaluations_, absolute_error_, relative_error_,
               error, val, inform);
        return val;
    }

    // The inverse cumulative distribution function (InverseCDF).
    std::vector<double> inverse_cdf(const std::vector<double>& probabilities) const {
        std::vector<double> sample(static_cast<std::size_t>(dimension_));
        // Create vector z of standard normal variates for each dimension
        std::vector<double> z(static_cast<std::size_t>(dimension_));
        for (int j = 0; j < dimension_; ++j) z[static_cast<std::size_t>(j)] = Normal::standard_z(probabilities[static_cast<std::size_t>(j)]);
        // x = A*z + mu
        la::Vector Az = factor_ * la::Vector(z);
        for (int j = 0; j < dimension_; ++j) sample[static_cast<std::size_t>(j)] = Az[j] + mean_[static_cast<std::size_t>(j)];
        return sample;
    }

    // Returns an independent univariate Normal distribution for the given index.
    Normal independent_normal(int index) const {
        return Normal(mean_[static_cast<std::size_t>(index)], standard_deviation()[static_cast<std::size_t>(index)]);
    }

    // Returns a new univariate Normal distribution.
    static MultivariateNormal univariate(double mu, double sigma) {
        return MultivariateNormal(std::vector<double>{mu}, std::vector<std::vector<double>>{{sigma * sigma}});
    }

    // Returns a new bivariate Normal distribution.
    static MultivariateNormal bivariate(double mu1, double mu2, double sigma1, double sigma2, double rho) {
        std::vector<double> mu = {mu1, mu2};
        std::vector<std::vector<double>> covariance = {{sigma1 * sigma1, sigma1 * sigma2 * rho},
                                                         {sigma1 * sigma2 * rho, sigma2 * sigma2}};
        return MultivariateNormal(std::move(mu), std::move(covariance));
    }

    // Generate random values of a distribution given a sample size. seed<=0 seeds from a
    // clock (no reproducible draw; see make_master()).
    std::vector<std::vector<double>> generate_random_values(int sample_size, int seed = -1) const {
        sampling::MersenneTwister rng = make_master(seed);
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                  std::vector<double>(static_cast<std::size_t>(dimension_)));
        for (int i = 0; i < sample_size; ++i) {
            std::vector<double> z(static_cast<std::size_t>(dimension_));
            for (int j = 0; j < dimension_; ++j) z[static_cast<std::size_t>(j)] = Normal::standard_z(rng.next_double());
            la::Vector Az = factor_ * la::Vector(z);
            for (int j = 0; j < dimension_; ++j)
                sample[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Az[j] + mean_[static_cast<std::size_t>(j)];
        }
        return sample;
    }

    // Use Latin hypercube method to generate random values of a distribution given a
    // sample size and a user-defined seed.
    std::vector<std::vector<double>> latin_hypercube_random_values(int sample_size, int seed) const {
        la::Matrix2D r = sampling::LatinHypercube::random(sample_size, dimension_, seed);
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                  std::vector<double>(static_cast<std::size_t>(dimension_)));
        for (int i = 0; i < sample_size; ++i) {
            std::vector<double> z(static_cast<std::size_t>(dimension_));
            for (int j = 0; j < dimension_; ++j)
                z[static_cast<std::size_t>(j)] = Normal::standard_z(r[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            la::Vector Az = factor_ * la::Vector(z);
            for (int j = 0; j < dimension_; ++j)
                sample[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Az[j] + mean_[static_cast<std::size_t>(j)];
        }
        return sample;
    }

    // Returns a 2D array of stratified z-variates. The first dimension is stratified, and
    // the remaining are sampled randomly.
    std::vector<std::vector<double>> stratified_random_values(
        const std::vector<sampling::StratificationBin>& stratification_bins, int seed) const {
        int sample_size = static_cast<int>(stratification_bins.size());
        // Matches the C# source exactly: always seeds directly from `seed` (no clock
        // fallback for seed<=0, unlike generate_random_values/latin_hypercube_random_values).
        sampling::MersenneTwister r(static_cast<std::uint32_t>(seed));
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                  std::vector<double>(static_cast<std::size_t>(dimension_)));
        for (int i = 0; i < sample_size; ++i) {
            std::vector<double> z(static_cast<std::size_t>(dimension_));
            for (int j = 0; j < dimension_; ++j) {
                if (j == 0) {
                    z[0] = Normal::standard_z(stratification_bins[static_cast<std::size_t>(i)].midpoint());
                } else {
                    z[static_cast<std::size_t>(j)] = Normal::standard_z(r.next_double());
                }
            }
            la::Vector Az = factor_ * la::Vector(z);
            for (int j = 0; j < dimension_; ++j)
                sample[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Az[j] + mean_[static_cast<std::size_t>(j)];
        }
        return sample;
    }

    // ========================================================================================
    // Cumulative Distribution Support
    // ========================================================================================

    // Computes the bivariate normal CDF: the probability for two normal variates X and Y
    // whose correlation is R, that AH <= X and AK <= Y.
    //
    // Original FORTRAN77 by Thomas Donnelly; C++ (Numerics' C#, ported here) by John
    // Burkardt. NOTE: this method is declared `public static` in the C# source but is not
    // called from anywhere else in the class (or exercised by any upstream unit test) --
    // `bivariate_cdf`/`bvu`/`mvnphi` are what CDF()/MVNDST actually use. Ported verbatim
    // for API completeness per the task brief; not exercised by any fixture (no oracle
    // value exists to scrape).
    static double bivnor(double ah, double ak, double r) {
        double a2;
        double ap;
        double b;
        double cn;
        double con;
        double conex;
        double ex;
        double g2;
        double gh;
        double gk;
        double gw = 0;
        double h2;
        double h4;
        int i;
        int idig = 15;
        int i5 = 0;
        double rr;
        double s1;
        double s2;
        double sgn;
        double sn;
        double sp;
        double sqr;
        double t;
        double twopi = 6.283185307179587;
        double w2 = 0;
        double wh = 0;
        double wk = 0;

        b = 0.0;

        gh = gauss(-ah) / 2.0;
        gk = gauss(-ak) / 2.0;

        if (r == 0.0) {
            b = 4.00 * gh * gk;
            b = std::max(b, 0.0);
            b = std::min(b, 1.0);
            return b;
        }

        rr = (1.0 + r) * (1.0 - r);

        // (rr < 0.0 in the C# source is a fatal-error branch whose body is entirely
        // commented out -- nothing to port.)

        if (rr == 0.0) {
            if (r < 0.0) {
                if (ah + ak < 0.0) {
                    b = 2.0 * (gh + gk) - 1.0;
                }
            } else {
                if (ah - ak < 0.0) {
                    b = 2.0 * gk;
                } else {
                    b = 2.0 * gh;
                }
            }
            b = std::max(b, 0.0);
            b = std::min(b, 1.0);
            return b;
        }

        sqr = std::sqrt(rr);

        if (idig == 15) {
            con = twopi * 1.0E-15 / 2.0;
        } else {
            con = twopi / 2.0;
            for (i = 1; i <= idig; i++) {
                con = con / 10.0;
            }
        }
        // (0,0)
        if (ah == 0.0 && ak == 0.0) {
            b = 0.25 + std::asin(r) / twopi;
            b = std::max(b, 0.0);
            b = std::min(b, 1.0);
            return b;
        }
        // (0,nonzero)
        if (ah == 0.0 && ak != 0.0) {
            b = gk;
            wh = -ak;
            wk = (ah / ak - r) / sqr;
            gw = 2.0 * gk;
            i5 = 1;
        }
        // (nonzero,0)
        else if (ah != 0.0 && ak == 0.0) {
            b = gh;
            wh = -ah;
            wk = (ak / ah - r) / sqr;
            gw = 2.0 * gh;
            i5 = -1;
        }
        // (nonzero,nonzero)
        else if (ah != 0.0 && ak != 0.0) {
            b = gh + gk;
            if (ah * ak < 0.0) {
                b = b - 0.5;
            }
            wh = -ah;
            wk = (ak / ah - r) / sqr;
            gw = 2.0 * gh;
            i5 = -1;
        }

        for (;;) {
            sgn = -1.0;
            t = 0.0;

            if (wk != 0.0) {
                if (std::fabs(wk) == 1.0) {
                    t = wk * gw * (1.0 - gw) / 2.0;
                    b = b + sgn * t;
                } else {
                    if (1.0 < std::fabs(wk)) {
                        sgn = -sgn;
                        wh = wh * wk;
                        g2 = gauss(wh);
                        wk = 1.0 / wk;

                        if (wk < 0.0) {
                            b = b + 0.5;
                        }
                        b = b - (gw + g2) / 2.0 + gw * g2;
                    }
                    h2 = wh * wh;
                    a2 = wk * wk;
                    h4 = h2 / 2.0;
                    ex = std::exp(-h4);
                    w2 = h4 * ex;
                    ap = 1.0;
                    s2 = ap - ex;
                    sp = ap;
                    s1 = 0.0;
                    sn = s1;
                    conex = std::fabs(con / wk);

                    for (;;) {
                        cn = ap * s2 / (sn + sp);
                        s1 = s1 + cn;

                        if (std::fabs(cn) <= conex) {
                            break;
                        }
                        sn = sp;
                        sp = sp + 1.0;
                        s2 = s2 - w2;
                        w2 = w2 * h4 / sp;
                        ap = -ap * a2;
                    }
                    t = (std::atan(wk) - wk * s1) / twopi;
                    b = b + sgn * t;
                }
            }
            if (0 <= i5) {
                break;
            }
            if (ak == 0.0) {
                break;
            }
            wh = -ak;
            wk = (ah / ak - r) / sqr;
            gw = 2.0 * gk;
            i5 = 1;
        }

        b = std::max(b, 0.0);
        b = std::min(b, 1.0);

        return b;
    }

    // A function for computing the bivariate normal CDF.
    //
    // This code was copied from the Accord Math Library, http://accord-framework.net,
    // based on work by Alan Genz, Dept. of Mathematics, Washington State University
    // (3-clause BSD license; see the upstream C# source for the full license text). Based
    // on Drezner & Wesolowsky (1989), "On the computation of the bivariate normal
    // integral", with modifications for double precision and for |R| close to 1.
    static double bivariate_cdf(double z1, double z2, double r) {
        const double TWOPI = 2.0 * kPi;
        const double* x;
        const double* w;
        int n;
        if (std::fabs(r) < 0.3) {
            x = kBvndXn6;
            w = kBvndWn6;
            n = 3;
        } else if (std::fabs(r) < 0.75) {
            x = kBvndXn12;
            w = kBvndWn12;
            n = 6;
        } else {
            x = kBvndXn20;
            w = kBvndWn20;
            n = 10;
        }

        double h = z1;
        double k = z2;
        double hk = h * k;
        double bvn = 0.0;
        if (std::fabs(r) < 0.925) {
            if (std::fabs(r) > 0.0) {
                double sh = (h * h + k * k) / 2.0;
                double asr = std::asin(r);
                for (int i = 0; i < n; i++) {
                    for (int j = -1; j <= 1; j += 2) {
                        double sn = std::sin(asr * (j * x[i] + 1.0) / 2.0);
                        bvn = bvn + w[i] * std::exp((sn * hk - sh) / (1.0 - sn * sn));
                    }
                }

                bvn = bvn * asr / (2.0 * TWOPI);
            }

            return bvn + mvnphi(-h) * mvnphi(-k);
        }

        if (r < 0.0) {
            k = -k;
            hk = -hk;
        }

        if (std::fabs(r) < 1.0) {
            double sa = (1.0 - r) * (1.0 + r);
            double A = std::sqrt(sa);
            double sb = h - k;
            sb = sb * sb;
            double c = (4.0 - hk) / 8.0;
            double d = (12.0 - hk) / 16.0;
            double asr = -(sb / sa + hk) / 2.0;
            if (asr > -100)
                bvn = A * std::exp(asr) * (1.0 - c * (sb - sa) * (1.0 - d * sb / 5.0) / 3.0 + c * d * sa * sa / 5.0);
            if (-hk < 100.0) {
                double B = std::sqrt(sb);
                bvn = bvn - std::exp(-hk / 2.0) * std::sqrt(TWOPI) * mvnphi(-B / A) * B *
                                (1.0 - c * sb * (1.0 - d * sb / 5.0) / 3.0);
            }

            A = A / 2.0;
            for (int i = 0; i < n; i++) {
                for (int j = -1; j <= 1; j += 2) {
                    double xs = A * (j * x[i] + 1.0);
                    xs = xs * xs;
                    double rs = std::sqrt(1.0 - xs);
                    asr = -(sb / xs + hk) / 2.0;
                    if (asr > -100) {
                        bvn = bvn + A * w[i] * std::exp(asr) *
                                        (std::exp(-hk * xs / (2.0 * (1.0 + rs) * (1.0 + rs))) / rs -
                                         (1.0 + c * xs * (1.0 + d * xs)));
                    }
                }
            }

            bvn = -bvn / TWOPI;
        }

        if (r > 0.0) return bvn + mvnphi(-std::max(h, k));
        bvn = -bvn;
        if (k <= h) return bvn;
        if (h < 0.0) return bvn + mvnphi(k) - mvnphi(h);
        return bvn + mvnphi(-h) - mvnphi(-k);
    }

    // Computes the standard normal CDF, accurate to 1E-15.
    // Reference: J.L. Schonfelder, Math Comp 32(1978), pp 1232-1240.
    static double mvnphi(double Z) {
        static constexpr double kMvnphiA[44] = {
            6.10143081923200417926465815756E-1,   -4.34841272712577471828182820888E-1,
            1.76351193643605501125840298123E-1,   -6.0710795609249414860051215825E-2,
            1.7712068995694114486147141191E-2,    -4.321119385567293818599864968E-3,
            8.54216676887098678819832055E-4,      -1.27155090609162742628893940E-4,
            1.1248167243671189468847072E-5,       3.13063885421820972630152E-7,
            -2.70988068537762022009086E-7,        3.0737622701407688440959E-8,
            2.515620384817622937314E-9,           -1.028929921320319127590E-9,
            2.9944052119949939363E-11,            2.6051789687266936290E-11,
            -2.634839924171969386E-12,            -6.43404509890636443E-13,
            1.12457401801663447E-13,              1.7281533389986098E-14,
            -4.264101694942375E-15,               -5.45371977880191E-16,
            1.58697607761671E-16,                 2.0899837844334E-17,
            -5.900526869409E-18,                  -9.41893387554E-19,
            2.14977356470E-19,                    4.6660985008E-20,
            -7.243011862E-21,                     -2.387966824E-21,
            1.91177535E-22,                       1.20482568E-22,
            -6.72377E-25,                         -5.747997E-24,
            -4.28493E-25,                         2.44856E-25,
            4.3793E-26,                            -8.151E-27,
            -3.089E-27,                            9.3E-29,
            1.74E-28,                              1.6E-29,
            -8.0E-30,                              -2.0E-30,
        };

        double RTWO = 1.414213562373095048801688724209;
        int IM = 24;
        double BM, B, BP = 0, P, T, XA;

        XA = std::fabs(Z) / RTWO;
        if (XA > 100) {
            P = 0;
        } else {
            T = (8 * XA - 30) / (4 * XA + 15);
            BM = 0;
            B = 0;
            for (int i = IM; i >= 0; i--) {
                BP = B;
                B = BM;
                BM = T * B - BP + kMvnphiA[i];
            }

            P = std::exp(-XA * XA) * (BM - BP) / 4;
        }

        if (Z > 0) P = 1 - P;

        return P;
    }

    // ALGORITHM AS241  APPL.STATIST. (1988) VOL. 37, NO. 3
    //
    // Produces the normal deviate Z corresponding to a given lower tail area of p.
    static double phinvs(double P) {
        // Coefficients for P close to 0.5 (HASH SUM AB 55.88319 28806 14901 4439)
        static constexpr double A0 = 3.3871328727963666080;
        static constexpr double A1 = 1.3314166789178437745E+2;
        static constexpr double A2 = 1.9715909503065514427E+3;
        static constexpr double A3 = 1.3731693765509461125E+4;
        static constexpr double A4 = 4.5921953931549871457E+4;
        static constexpr double A5 = 6.7265770927008700853E+4;
        static constexpr double A6 = 3.3430575583588128105E+4;
        static constexpr double A7 = 2.5090809287301226727E+3;
        static constexpr double B1 = 4.2313330701600911252E+1;
        static constexpr double B2 = 6.8718700749205790830E+2;
        static constexpr double B3 = 5.3941960214247511077E+3;
        static constexpr double B4 = 2.1213794301586595867E+4;
        static constexpr double B5 = 3.9307895800092710610E+4;
        static constexpr double B6 = 2.8729085735721942674E+4;
        static constexpr double B7 = 5.2264952788528545610E+3;
        // Coefficients for P not close to 0, 0.5, or 1 (HASH SUM CD 49.33206 50330 16102 89036)
        static constexpr double C0 = 1.42343711074968357734;
        static constexpr double C1 = 4.63033784615654529590;
        static constexpr double C2 = 5.76949722146069140550;
        static constexpr double C3 = 3.64784832476320460504;
        static constexpr double C4 = 1.27045825245236838258;
        static constexpr double C5 = 2.41780725177450611770E-1;
        static constexpr double C6 = 2.27238449892691845833E-2;
        static constexpr double C7 = 7.74545014278341407640E-4;
        static constexpr double D1 = 2.05319162663775882187;
        static constexpr double D2 = 1.67638483018380384940;
        static constexpr double D3 = 6.89767334985100004550E-1;
        static constexpr double D4 = 1.48103976427480074590E-1;
        static constexpr double D5 = 1.51986665636164571966E-2;
        static constexpr double D6 = 5.47593808499534494600E-4;
        static constexpr double D7 = 1.05075007164441684324E-9;
        // Coefficients for P near 0 or 1 (HASH SUM EF 47.52583 31754 92896 71629)
        static constexpr double E0 = 6.65790464350110377720;
        static constexpr double E1 = 5.46378491116411436990;
        static constexpr double E2 = 1.78482653991729133580;
        static constexpr double E3 = 2.96560571828504891230E-1;
        static constexpr double E4 = 2.65321895265761230930E-2;
        static constexpr double E5 = 1.24266094738807843860E-3;
        static constexpr double E6 = 2.71155556874348757815E-5;
        static constexpr double E7 = 2.01033439929228813265E-7;
        static constexpr double F1 = 5.99832206555887937690E-1;
        static constexpr double F2 = 1.36929880922735805310E-1;
        static constexpr double F3 = 1.48753612908506148525E-2;
        static constexpr double F4 = 7.86869131145613259100E-4;
        static constexpr double F5 = 1.84631831751005468180E-5;
        static constexpr double F6 = 1.42151175831644588870E-7;
        static constexpr double F7 = 2.04426310338993978564E-15;

        double SPLIT1 = 0.425, SPLIT2 = 5, CONST1 = 0.180625, CONST2 = 1.6;

        double Q = (2 * P - 1) / 2;
        double R;
        if (std::fabs(Q) < SPLIT1) {
            R = CONST1 - Q * Q;
            return Q *
                   (((((((A7 * R + A6) * R + A5) * R + A4) * R + A3) * R + A2) * R + A1) * R + A0) /
                   (((((((B7 * R + B6) * R + B5) * R + B4) * R + B3) * R + B2) * R + B1) * R + 1);
        } else {
            double result;
            R = std::min(P, 1 - P);
            if (R > 0) {
                R = std::sqrt(-std::log(R));
                if (R < SPLIT2) {
                    R -= CONST2;
                    result = (((((((C7 * R + C6) * R + C5) * R + C4) * R + C3) * R + C2) * R + C1) * R + C0) /
                             (((((((D7 * R + D6) * R + D5) * R + D4) * R + D3) * R + D2) * R + D1) * R + 1);
                } else {
                    R -= SPLIT2;
                    result = (((((((E7 * R + E6) * R + E5) * R + E4) * R + E3) * R + E2) * R + E1) * R + E0) /
                             (((((((F7 * R + F6) * R + F5) * R + F4) * R + F3) * R + F2) * R + F1) * R + 1);
                }
            } else {
                result = 9;
            }
            if (Q < 0) {
                result = -result;
            }
            return result;
        }
    }

    // Adapted from a function for computing bivariate normal probabilities by Yihong Ge
    // and Alan Genz, Washington State University.
    //
    // BVU - calculate the probability that X is larger than SH and Y is larger than SK.
    static double bvu(double SH, double SK, double R) {
        // Gauss Legendre Points and Weights, N = 6, 12, and 20 (a SEPARATE literal copy of
        // the same underlying constants as kBvndX*/kBvndW* above -- the C# source keeps
        // two independently-rounded tables for BivariateCDF vs BVU; both are transcribed
        // verbatim rather than consolidated, to avoid changing either algorithm's exact
        // floating-point results).
        static constexpr double kX[10][3] = {
            {-0.932469514203152, -0.9815606342467192, -0.9931285991850949},
            {-0.6612093864662645, -0.9041172563704749, -0.9639719272779138},
            {-0.2386191860831969, -0.7699026741943047, -0.912234428251326},
            {0, -0.5873179542866175, -0.8391169718222188},
            {0, -0.3678314989981802, -0.7463319064601508},
            {0, -0.1252334085114689, -0.636053680726515},
            {0, 0, -0.5108670019508271},
            {0, 0, -0.37370608871541955},
            {0, 0, -0.22778585114164507},
            {0, 0, -0.07652652113349734},
        };
        static constexpr double kW[10][3] = {
            {0.17132449237917036, 0.04717533638651183, 0.0176140071391521},
            {0.3607615730481386, 0.10693932599531843, 0.0406014298003869},
            {0.46791393457269104, 0.16007832854334622, 0.062672048334109},
            {0, 0.20316742672306592, 0.0832767415767047},
            {0, 0.2334925365383548, 0.10193011981724},
            {0, 0.24914704581340277, 0.118194531961518},
            {0, 0, 0.131688638449176},
            {0, 0, 0.142096109318382},
            {0, 0, 0.149172986472603},
            {0, 0, 0.152753387130725},
        };

        double BVN;
        int LG, NG;
        double TWOPI = 6.283185307179586;
        double AS, A, B, C, D, RS, XS;
        double SN, ASR, H, K, BS, HS, HK;

        if (std::fabs(R) < .3) {
            NG = 0;
            LG = 3;
        } else if (std::fabs(R) < .75) {
            NG = 1;
            LG = 6;
        } else {
            NG = 2;
            LG = 10;
        }

        H = SH;
        K = SK;
        HK = H * K;
        BVN = 0;
        if (std::fabs(R) < .925) {
            HS = (H * H + K * K) / 2;
            ASR = std::asin(R);
            for (int i = 0; i < LG; i++) {
                SN = std::sin(ASR * (kX[i][NG] + 1) / 2);
                BVN += kW[i][NG] * std::exp((SN * HK - HS) / (1 - SN * SN));
                SN = std::sin(ASR * (-kX[i][NG] + 1) / 2);
                BVN += kW[i][NG] * std::exp((SN * HK - HS) / (1 - SN * SN));
            }

            BVN = BVN * ASR / (2 * TWOPI) + mvnphi(-H) * mvnphi(-K);
        } else {
            if (R < 0) {
                K = -K;
                HK = -HK;
            }
            if (std::fabs(R) < 1) {
                AS = (1 - R) * (1 + R);
                A = std::sqrt(AS);
                BS = (H - K) * (H - K);
                C = (4 - HK) / 8;
                D = (12 - HK) / 16;
                BVN = A * std::exp(-(BS / AS + HK) / 2) *
                      (1 - C * (BS - AS) * (1 - D * BS / 5) / 3 + C * D * AS * AS / 5);
                if (HK > -160) {
                    B = std::sqrt(BS);
                    BVN = BVN - std::exp(-HK / 2) * std::sqrt(TWOPI) * mvnphi(-B / A) * B *
                                    (1 - C * BS * (1 - D * BS / 5) / 3);
                }
                A = A / 2;
                for (int i = 0; i < LG; i++) {
                    XS = (A * (kX[i][NG] + 1)) * (A * (kX[i][NG] + 1));
                    RS = std::sqrt(1 - XS);
                    BVN = BVN + A * kW[i][NG] *
                                    (std::exp(-BS / (2 * XS) - HK / (1 + RS)) / RS -
                                     std::exp(-(BS / XS + HK) / 2) * (1 + C * XS * (1 + D * XS)));
                    XS = AS * (-kX[i][NG] + 1) * (-kX[i][NG] + 1) / 4;
                    RS = std::sqrt(1 - XS);
                    BVN = BVN + A * kW[i][NG] * std::exp(-(BS / XS + HK) / 2) *
                                    (std::exp(-HK * XS / (2 * (1 + RS) * (1 + RS))) / RS -
                                     (1 + C * XS * (1 + D * XS)));
                }

                BVN = -BVN / TWOPI;
            }

            if (R > 0) {
                BVN = BVN + mvnphi(-std::max(H, K));
            } else {
                BVN = -BVN;
                if (K > H) {
                    if (H < 0) {
                        BVN = BVN + mvnphi(K) - mvnphi(H);
                    } else {
                        BVN = BVN + mvnphi(-H) - mvnphi(-K);
                    }
                }
            }
        }

        // This can sometimes produce very small, but negative values (approx. -5E-73)
        if (BVN < 0.0) BVN = 0.0;
        return BVN;
    }

    // A subroutine for computing multivariate normal probabilities, using the algorithm
    // in Alan Genz, "Numerical Computation of Multivariate Normal Probabilities", J. of
    // Computational and Graphical Stat., 1(1992), pp. 141-149.
    //
    // v2.1.4 shape note: `MVNDNT` is `void` (no return value at all -- see the comment on
    // `mvndnt` itself); `INFORM` is set to 0 explicitly here immediately before calling it,
    // mirroring the C# source exactly (`INFORM = 0; MVNDNT(...);` in place of the old
    // `INFORM = (int)MVNDNT(...)`). `INFORM` is only reassigned afterward by `dkbvrc` in
    // the N-INFIS>=2 branch below -- purely a shape mirror, no behavior change (the old
    // cast always evaluated to 0 too, since `MVNDNT`'s local `result` was never
    // reassigned).
    void mvndst(int N, const std::vector<double>& LOWER, const std::vector<double>& UPPER,
                const std::vector<int>& INFIN, const std::vector<double>& CORREL, int MAXPTS,
                double ABSEPS, double RELEPS, double& ERROR, double& VALUE, int& INFORM) const {
        int INFIS = 0, IVLS;
        double D = 0, E = 0;
        std::vector<double> Y(500, 0.0);

        if (N > 500 || N < 1) {
            INFORM = 2;
            VALUE = 0;
            ERROR = 1;
        } else {
            INFORM = 0;
            mvndnt(N, CORREL, LOWER, UPPER, INFIN, INFIS, D, E, Y);
            if (N - INFIS == 0) {
                VALUE = 1;
                ERROR = 0;
            } else if (N - INFIS == 1) {
                VALUE = E - D;
                // This can sometimes produce very small, but negative values (approx. -5E-73)
                if (VALUE < 0) VALUE = 0;
                ERROR = 2E-16;
            } else {
                // Call the lattice rule integration subroutine.
                IVLS = 0;
                dkbvrc(
                    N - INFIS - 1, IVLS, MAXPTS,
                    [this](int n, const std::vector<double>& w) { return mvndfn(n, w); }, ABSEPS,
                    RELEPS, ERROR, VALUE, INFORM);
            }
        }
    }

    // Creates a copy of the distribution.
    std::unique_ptr<MultivariateDistribution> clone() const override {
        std::unique_ptr<MultivariateNormal> mvn(new MultivariateNormal());
        mvn->parameters_valid_ = parameters_valid_;
        mvn->dimension_ = dimension_;
        mvn->mean_ = mean_;
        mvn->covariance_ = covariance_.clone();
        mvn->decomposition_ = decomposition_;
        mvn->cholesky_ = cholesky_;
        mvn->svd_ = svd_;
        mvn->factor_ = factor_.clone();
        mvn->nullspace_ = nullspace_.clone();
        mvn->svd_threshold_ = svd_threshold_;
        mvn->rank_ = rank_;
        mvn->lnconstant_ = lnconstant_;
        mvn->correlation_ = correlation_.clone();
        mvn->correl_ = correl_;
        mvn->mvnuni_ = mvnuni_;  // value copy -- see file header divergence note
        mvn->max_evaluations_ = max_evaluations_;
        mvn->absolute_error_ = absolute_error_;
        mvn->relative_error_ = relative_error_;
        mvn->lower_ = lower_;
        mvn->upper_ = upper_;
        mvn->infin_ = infin_;
        mvn->correlation_matrix_created_ = correlation_matrix_created_;
        mvn->cov_srted_ = cov_srted_;
        mvn->nl_ = nl_;
        return mvn;
    }

   private:
    // Private parameterless ctor for clone() (matches the C# `private MultivariateNormal()`).
    MultivariateNormal() = default;

    // Validates a dimension-index subset: non-null, non-empty, in range, and without
    // repeats. Mirrors C# `ValidateIndices` (a private static helper on this class).
    static void validate_indices(const std::vector<int>& indices, int dimension) {
        if (indices.empty())
            throw std::out_of_range("At least one dimension index is required.");
        std::vector<bool> seen(static_cast<std::size_t>(dimension), false);
        for (int index : indices) {
            if (index < 0 || index >= dimension)
                throw std::out_of_range("A dimension index is out of range.");
            if (seen[static_cast<std::size_t>(index)])
                throw std::out_of_range("Dimension indices must not repeat.");
            seen[static_cast<std::size_t>(index)] = true;
        }
    }

    static double gauss(double t) {
        // GAUSS returns the area of the lower tail of the normal curve.
        return (1.0 + sf::erf::function(t / std::sqrt(2.0))) / 2.0;
    }

    static constexpr double kZeroToleranceFactor = 1E6;
    static constexpr double kRelativeMachineEpsilon = 2.220446049250313E-16;
    static constexpr int kDefaultMvnuniSeed = 12345;

    static double singular_value_threshold(const la::SingularValueDecomposition& singular_values) {
        return kZeroToleranceFactor * kRelativeMachineEpsilon * singular_values.w()[0];
    }

    static bool is_symmetric_positive_semidefinite(
        la::SingularValueDecomposition& singular_values, const la::Matrix& covariance) {
        int n = covariance.number_of_rows();
        double threshold = singular_value_threshold(singular_values);
        std::vector<double> eigenvalues(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) {
            double quotient = 0.0;
            for (int i = 0; i < n; ++i) {
                double row = 0.0;
                for (int k = 0; k < n; ++k) row += covariance(i, k) * singular_values.u()(k, j);
                quotient += singular_values.u()(i, j) * row;
            }
            eigenvalues[static_cast<std::size_t>(j)] = quotient;
            if (!(quotient >= -threshold)) return false;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                double reconstructed = 0.0;
                for (int k = 0; k < n; ++k)
                    reconstructed += eigenvalues[static_cast<std::size_t>(k)] *
                                     singular_values.u()(i, k) * singular_values.u()(j, k);
                if (!(std::abs(reconstructed - covariance(i, j)) <= threshold) ||
                    !(std::abs(reconstructed - covariance(j, i)) <= threshold))
                    return false;
            }
        }
        return true;
    }

    void factorize_with_singular_values(la::SingularValueDecomposition singular_values) {
        cholesky_.reset();
        svd_.emplace(std::move(singular_values));
        svd_threshold_ = singular_value_threshold(*svd_);
        rank_ = svd_->rank(svd_threshold_);
        nullspace_ = svd_->nullspace(svd_threshold_);
        double lndet = svd_->log_pseudo_determinant(svd_threshold_);
        lnconstant_ = -(std::log(2.0 * kPi) * rank_ + lndet) * 0.5;
        factor_ = la::Matrix(dimension_, dimension_);
        for (int j = 0; j < dimension_; ++j) {
            double scale = svd_->w()[j] > svd_threshold_ ? std::sqrt(svd_->w()[j]) : 0.0;
            for (int i = 0; i < dimension_; ++i) factor_(i, j) = svd_->u()(i, j) * scale;
        }
    }

    bool is_on_support(const std::vector<double>& x) const {
        if (nullspace_.number_of_columns() == 0 || x.size() != mean_.size()) return true;
        std::vector<double> z(x.size());
        double norm = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            z[i] = x[i] - mean_[i];
            norm += z[i] * z[i];
        }
        double tolerance = kZeroToleranceFactor * kRelativeMachineEpsilon *
                           std::max(1.0, std::sqrt(norm));
        for (int j = 0; j < nullspace_.number_of_columns(); ++j) {
            double projection = 0.0;
            for (int i = 0; i < dimension_; ++i)
                projection += nullspace_(i, j) * z[static_cast<std::size_t>(i)];
            if (!(std::abs(projection) <= tolerance)) return false;
        }
        return true;
    }

    // Create the correlation arrays required for computing the CDF (see the file header's
    // divergence note on the `invSqrtD` elementwise-reciprocal simplification).
    void create_correlation_matrix() const {
        la::Matrix D = la::Matrix::diagonal(covariance_);
        D.sqrt();
        la::Matrix inv_sqrt_d(dimension_);
        for (int i = 0; i < dimension_; ++i) inv_sqrt_d(i, i) = 1.0 / D(i, i);
        correlation_ = (inv_sqrt_d * covariance_) * inv_sqrt_d;
        // Create collapsed matrix
        correl_.assign(static_cast<std::size_t>(nl_), 0.0);
        int t = 0;
        for (int i = 1; i < dimension_; ++i) {
            for (int j = 0; j < i; ++j) {
                correl_[static_cast<std::size_t>(t)] = correlation_(i, j);
                ++t;
            }
        }
        correlation_matrix_created_ = true;
    }

    // ==== Genz-Bretz MVNDST machinery (MVNDFN/MVNDNT/MVNLMS/COVSRT/RCSWP/DKBVRC/DKSMRC) ====
    //
    // Ported from Alan Genz's MVNDST (Numerical Computation of Multivariate Normal
    // Probabilities, J. Comp. Graph. Stat. 1(1992)), via the C# translation in
    // MultivariateNormal.cs. C# `goto`s are kept as C++ `goto` (legal; minimizes
    // transcription risk on this densely-indexed lattice-rule code) and every
    // integer/double modulo/index expression is preserved exactly, including several
    // upstream oddities noted at their call sites below. `mvnlms`/`covsrt`/`rcswp`/
    // `bvnmvn` don't reference instance state (the C# versions don't either -- they only
    // read/write the arrays passed to them by reference) but are kept as ordinary member
    // functions rather than hoisted to `static`, matching the C# access-modifier
    // structure as closely as C++ allows.

    // Integrand subroutine: evaluates the transformed integrand at one quasi-random point
    // W, using and advancing the COVSRT-sorted instance working arrays.
    //
    // Note: this method's local `Y` (sized `nl_`) is UNRELATED to `mvndst`'s own local
    // `Y` (sized 500) or `covsrt`'s `Y` OUT-parameter -- the C# source reuses the name
    // `Y` for three distinct local/parameter variables across MVNDST/MVNDFN/COVSRT; kept
    // as `Y` in each to mirror the source, since they're in disjoint C++ scopes.
    double mvndfn(int N, const std::vector<double>& W) const {
        double SUM, AI = 0, BI = 0, DI = 0, EI = 0;

        std::vector<double> Y(static_cast<std::size_t>(nl_), 0.0);

        double result = 1;
        int INFA = 0;
        int INFB = 0;
        int IK = 0;
        int IJ = 0;
        for (int i = 0; i < N + 1; i++) {
            SUM = 0;
            for (int j = 0; j <= i - 1; j++) {
                if (j < IK) {
                    SUM += mvndfn_cov_[static_cast<std::size_t>(IJ)] * Y[static_cast<std::size_t>(j)];
                }
                IJ++;
            }

            if (mvndfn_infi_[static_cast<std::size_t>(i)] != 0) {
                if (INFA == 1) {
                    AI = std::max(AI, mvndfn_a_[static_cast<std::size_t>(i)] - SUM);
                } else {
                    AI = mvndfn_a_[static_cast<std::size_t>(i)] - SUM;
                    INFA = 1;
                }
            }
            if (mvndfn_infi_[static_cast<std::size_t>(i)] != 1) {
                if (INFB == 1) {
                    BI = std::min(BI, mvndfn_b_[static_cast<std::size_t>(i)] - SUM);
                } else {
                    BI = mvndfn_b_[static_cast<std::size_t>(i)] - SUM;
                    INFB = 1;
                }
            }
            IJ++;

            if (i == (N) || mvndfn_cov_[static_cast<std::size_t>(IJ + IK + 1)] > 0) {
                mvnlms(AI, BI, 2 * INFA + INFB - 1, DI, EI);

                if (DI >= EI) {
                    result = 0;
                    break;
                } else {
                    result = result * (EI - DI);
                    if (i <= N) {
                        Y[static_cast<std::size_t>(IK)] =
                            phinvs(DI + W[static_cast<std::size_t>(IK)] * (EI - DI));
                    }
                    IK = IK + 1;
                    INFA = 0;
                    INFB = 0;
                }
            }
        }
        return result;
    }

    // v2.1.4 shape note: changed from `private double MVNDNT(...)` (a local `result`
    // initialized to 0 and never reassigned -- i.e. always returning the dead constant
    // 0.0) to `private void MVNDNT(...)`; the caller (`mvndst` above) now sets
    // `INFORM = 0` explicitly instead of casting this method's old always-0 return value.
    // No behavior change -- see the comment on `mvndst`'s call site.
    void mvndnt(int N, const std::vector<double>& CORREL, const std::vector<double>& LOWER,
                const std::vector<double>& UPPER, const std::vector<int>& INFIN, int& INFIS,
                double& D, double& E, std::vector<double>& Y) const {
        covsrt(N, LOWER, UPPER, CORREL, INFIN, Y, INFIS, mvndfn_a_, mvndfn_b_, mvndfn_cov_, mvndfn_infi_);

        if (N - INFIS == 1) {
            mvnlms(mvndfn_a_[0], mvndfn_b_[0], mvndfn_infi_[0], D, E);
        } else if (N - INFIS == 2) {
            if (std::fabs(mvndfn_cov_[2]) > 0) {
                D = std::sqrt(1 + mvndfn_cov_[1] * mvndfn_cov_[1]);
                if (mvndfn_infi_[1] != 0) {
                    mvndfn_a_[1] = mvndfn_a_[1] / D;
                }
                if (mvndfn_infi_[1] != 1) {
                    mvndfn_b_[1] = mvndfn_b_[1] / D;
                }
                E = bvnmvn(mvndfn_a_, mvndfn_b_, mvndfn_infi_, mvndfn_cov_[1] / D);
                D = 0;
            } else {
                if (mvndfn_infi_[0] != 0) {
                    if (mvndfn_infi_[1] != 0) {
                        mvndfn_a_[0] = std::max(mvndfn_a_[0], mvndfn_a_[1]);
                    }
                } else {
                    if (mvndfn_infi_[1] != 0) {
                        mvndfn_a_[0] = mvndfn_a_[1];
                    }
                }

                if (mvndfn_infi_[0] != 1) {
                    if (mvndfn_infi_[1] != 1) {
                        mvndfn_b_[0] = std::min(mvndfn_b_[0], mvndfn_b_[1]);
                    }
                } else {
                    if (mvndfn_infi_[1] != 1) {
                        mvndfn_b_[0] = mvndfn_b_[1];
                    }
                }

                if (mvndfn_infi_[0] != mvndfn_infi_[1]) {
                    mvndfn_infi_[0] = 2;
                }
                mvnlms(mvndfn_a_[0], mvndfn_b_[0], mvndfn_infi_[0], D, E);
            }
            INFIS++;
        }
    }

    void mvnlms(double A, double B, int INFIN, double& LOWER, double& UPPER) const {
        LOWER = 0;
        UPPER = 1;
        if (INFIN >= 0) {
            if (INFIN != 0) {
                LOWER = mvnphi(A);
            }
            if (INFIN != 1) {
                UPPER = mvnphi(B);
            }
        }
        UPPER = std::max(UPPER, LOWER);
    }

    // Subroutine to sort integration limits and determine Cholesky factor.
    //
    // v2.1.4 COVSRT repair (RESOLVED -- see docs/upstream-csharp-issues.md for the
    // pre-fix history): the "permute limits and/or rows" degenerate-diagonal branch below
    // (reached when a covariance diagonal entry is zero after RCSWP) previously had
    // multiple inverted/off-by-one loop bounds -- `for (int j = i - 1; j < 0; j--)` (never
    // iterated for `i >= 1`; at `i == 0` it indexed `COV[-1]`, UB in C++ / a caught
    // `IndexOutOfRangeException` in C#, which this port worked around with a bounds
    // guard), `for (int l = 0; l < j; l++)`, `for (int l = j + 1; l < i - 1; l++)` with the
    // packed index `(l - 1) * l / 2 + j + 1`, `for (int k = i - 1; k < l; k--)`, and
    // `for (int m = 0; m < k; m++)` with the swap offset `IJ - k + m` / decrement
    // `IJ = IJ - k`. Upstream fixed ALL of these in the same commit (bounds `j >= 0`,
    // `l <= j`, `l <= i - 1`, `k >= l`, `m <= k`; packed index `l * (l + 1) / 2 + j + 1`;
    // swap offset `IJ - k + m - 1`; decrement `IJ - k - 1`) -- transcribed verbatim below
    // from the fixed C#, replacing the whole region wholesale rather than patching the old
    // loops incrementally. The old bounds-guard divergence note is retired: the fixed loop
    // does not produce an out-of-range access on this port's fixture-exercised
    // singular/rank-deficient inputs.
    void covsrt(int N, const std::vector<double>& LOWER, const std::vector<double>& UPPER,
                const std::vector<double>& CORREL, const std::vector<int>& INFIN, std::vector<double>& Y,
                int& INFIS, std::vector<double>& A, std::vector<double>& B, std::vector<double>& COV,
                std::vector<int>& INFI) const {
        double SQTWPI = 2.506628274631001;
        double EPS = 1E-10;

        int IJ = 0;
        int II = 0;
        INFIS = 0;

        for (int i = 0; i < N; i++) {
            A[static_cast<std::size_t>(i)] = 0;
            B[static_cast<std::size_t>(i)] = 0;
            INFI[static_cast<std::size_t>(i)] = INFIN[static_cast<std::size_t>(i)];
            if (INFI[static_cast<std::size_t>(i)] < 0) {
                INFIS++;
            } else {
                if (INFI[static_cast<std::size_t>(i)] != 0) {
                    A[static_cast<std::size_t>(i)] = LOWER[static_cast<std::size_t>(i)];
                }
                if (INFI[static_cast<std::size_t>(i)] != 1) {
                    B[static_cast<std::size_t>(i)] = UPPER[static_cast<std::size_t>(i)];
                }
            }

            for (int j = 0; j < i; j++) {
                COV[static_cast<std::size_t>(IJ)] = CORREL[static_cast<std::size_t>(II)];
                II++;
                IJ++;
            }
            COV[static_cast<std::size_t>(IJ)] = 1;
            IJ++;
        }

        // First move any doubly infinite limits to innermost positions.
        if (INFIS < N) {
            for (int i = N - 1; i >= N - INFIS; i--) {
                if (INFI[static_cast<std::size_t>(i)] >= 0) {
                    for (int j = 0; j < i - 1; j++) {
                        if (INFI[static_cast<std::size_t>(j)] < 0) {
                            rcswp(j, i, A, B, INFI, N, COV);
                            goto ten;
                        }
                    }
                }
            }

        ten:

            // Sort remaining limits and determine Cholesky factor.
            II = 0;
            double AMIN = 0, BMIN = 0, DMIN, EMIN, CVDIAG, SUMSQ = 0, SUM, AJ = 0, BJ = 0, D = 0, E = 0, YL, YU;
            int JMIN, IL, M;
            for (int i = 0; i < N - INFIS; i++) {
                // Determine the integration limits for variable with minimum expected
                // probability and interchange that variable with Ith.

                DMIN = 0;
                EMIN = 1;
                JMIN = i;
                CVDIAG = 0;
                IJ = II;

                for (int j = i; j < N - INFIS; j++) {
                    if (COV[static_cast<std::size_t>(IJ + j)] > EPS) {
                        SUMSQ = std::sqrt(COV[static_cast<std::size_t>(IJ + j)]);
                        SUM = 0;
                        for (int k = 0; k < i; k++) {
                            SUM += COV[static_cast<std::size_t>(IJ + k)] * Y[static_cast<std::size_t>(k)];
                        }

                        AJ = (A[static_cast<std::size_t>(j)] - SUM) / SUMSQ;
                        BJ = (B[static_cast<std::size_t>(j)] - SUM) / SUMSQ;

                        mvnlms(AJ, BJ, INFI[static_cast<std::size_t>(j)], D, E);

                        if (EMIN + D >= E + DMIN) {
                            JMIN = j;
                            AMIN = AJ;
                            BMIN = BJ;
                            DMIN = D;
                            EMIN = E;
                            CVDIAG = SUMSQ;
                        }
                    }
                    IJ += (j + 1);
                }

                if (JMIN > i) {
                    rcswp(i, JMIN, A, B, INFI, N, COV);
                }
                COV[static_cast<std::size_t>(II + i)] = CVDIAG;

                // Compute Ith column of Cholesky factor.
                // Compute expected value for Ith integration variable and
                // scale Ith covariance matrix row and limits.

                if (CVDIAG > 0) {
                    IL = II + (i + 1);
                    for (int l = i + 2; l <= N - INFIS; l++) {
                        COV[static_cast<std::size_t>(IL + i)] = COV[static_cast<std::size_t>(IL + i)] / CVDIAG;
                        IJ = II + i;

                        for (int j = i + 1; j < l; j++) {
                            COV[static_cast<std::size_t>(IL + j)] =
                                COV[static_cast<std::size_t>(IL + j)] -
                                COV[static_cast<std::size_t>(IL + i)] * COV[static_cast<std::size_t>(IJ + i + 1)];
                            IJ += j + 1;
                        }
                        IL += l;
                    }

                    if (EMIN > DMIN + EPS) {
                        YL = 0;
                        YU = 0;
                        if (INFI[static_cast<std::size_t>(i)] != 0) {
                            YL = -std::exp(-(AMIN * AMIN) / 2) / SQTWPI;
                        }
                        if (INFI[static_cast<std::size_t>(i)] != 1) {
                            YU = -std::exp(-(BMIN * BMIN) / 2) / SQTWPI;
                        }
                        Y[static_cast<std::size_t>(i)] = (YU - YL) / (EMIN - DMIN);
                    } else {
                        if (INFI[static_cast<std::size_t>(i)] == 0) Y[static_cast<std::size_t>(i)] = BMIN;
                        if (INFI[static_cast<std::size_t>(i)] == 1) Y[static_cast<std::size_t>(i)] = AMIN;
                        if (INFI[static_cast<std::size_t>(i)] == 2) Y[static_cast<std::size_t>(i)] = (AMIN + BMIN) / 2;
                    }
                    for (int j = 0; j <= i; j++) {
                        COV[static_cast<std::size_t>(II)] = COV[static_cast<std::size_t>(II)] / CVDIAG;
                        II += 1;
                    }
                    A[static_cast<std::size_t>(i)] = A[static_cast<std::size_t>(i)] / CVDIAG;
                    B[static_cast<std::size_t>(i)] = B[static_cast<std::size_t>(i)] / CVDIAG;
                } else {
                    IL = II + i;
                    for (int l = i + 1; l < N - INFIS; l++) {
                        COV[static_cast<std::size_t>(IL + i)] = 0;
                        IL += l;
                    }

                    // If the covariance matrix diagonal entry is zero, permute limits
                    // and/or rows, if necessary. (FIXED loop bounds -- see the v2.1.4
                    // COVSRT repair note above this method.)

                    for (int j = i - 1; j >= 0; j--) {
                        if (std::fabs(COV[static_cast<std::size_t>(II + j)]) > EPS) {
                            A[static_cast<std::size_t>(i)] = A[static_cast<std::size_t>(i)] / COV[static_cast<std::size_t>(II + j)];
                            B[static_cast<std::size_t>(i)] = B[static_cast<std::size_t>(i)] / COV[static_cast<std::size_t>(II + j)];
                            if (COV[static_cast<std::size_t>(II + j)] < 0) {
                                std::swap(A[static_cast<std::size_t>(i)], B[static_cast<std::size_t>(i)]);
                                if (INFI[static_cast<std::size_t>(i)] != 2) INFI[static_cast<std::size_t>(i)] = 1 - INFI[static_cast<std::size_t>(i)];
                            }
                            for (int l = 0; l <= j; l++) {
                                COV[static_cast<std::size_t>(II + l)] =
                                    COV[static_cast<std::size_t>(II + l)] / COV[static_cast<std::size_t>(II + j)];
                            }
                            for (int l = j + 1; l <= i - 1; l++) {
                                if (COV[static_cast<std::size_t>(l * (l + 1) / 2 + j + 1)] > 0) {
                                    IJ = II;
                                    for (int k = i - 1; k >= l; k--) {
                                        for (int m = 0; m <= k; m++) {
                                            std::swap(COV[static_cast<std::size_t>(IJ - k + m - 1)],
                                                      COV[static_cast<std::size_t>(IJ + m)]);
                                        }
                                        std::swap(A[static_cast<std::size_t>(k)], A[static_cast<std::size_t>(k + 1)]);
                                        std::swap(B[static_cast<std::size_t>(k)], B[static_cast<std::size_t>(k + 1)]);
                                        M = INFI[static_cast<std::size_t>(k)];
                                        INFI[static_cast<std::size_t>(k)] = INFI[static_cast<std::size_t>(k + 1)];
                                        INFI[static_cast<std::size_t>(k + 1)] = M;
                                        IJ = IJ - k - 1;
                                    }
                                    goto twenty;
                                }
                            }
                            goto twenty;
                        }
                        COV[static_cast<std::size_t>(II + j)] = 0;
                    }
                twenty:
                    II = II + i;
                    Y[static_cast<std::size_t>(i)] = 0;
                }
            }
        }
    }

    // Swaps rows and columns P and Q in situ, with P <= Q.
    void rcswp(int P, int Q, std::vector<double>& A, std::vector<double>& B, std::vector<int>& INFIN,
               int N, std::vector<double>& C) const {
        int II, JJ;
        std::swap(A[static_cast<std::size_t>(P)], A[static_cast<std::size_t>(Q)]);
        std::swap(B[static_cast<std::size_t>(P)], B[static_cast<std::size_t>(Q)]);
        int J = INFIN[static_cast<std::size_t>(P)];
        INFIN[static_cast<std::size_t>(P)] = INFIN[static_cast<std::size_t>(Q)];
        INFIN[static_cast<std::size_t>(Q)] = J;
        JJ = (P * (P + 1)) / 2;
        II = (Q * (Q + 1)) / 2;

        std::swap(C[static_cast<std::size_t>(JJ + P)], C[static_cast<std::size_t>(II + Q)]);

        for (int i = 0; i < P; i++) {
            std::swap(C[static_cast<std::size_t>(JJ + i)], C[static_cast<std::size_t>(II + i)]);
        }

        JJ += (P + 1);
        for (int i = P + 1; i < Q; i++) {
            std::swap(C[static_cast<std::size_t>(JJ + P)], C[static_cast<std::size_t>(II + i)]);
            JJ += (i + 1);
        }

        II += (Q + 1);
        for (int i = Q + 1; i < N; i++) {
            std::swap(C[static_cast<std::size_t>(II + P)], C[static_cast<std::size_t>(II + Q)]);
            II += (i + 1);
        }
    }

    // Automatic Multidimensional Integration Subroutine (Alan Genz). DKBVRC computes an
    // approximation to the NDIM-dimensional integral of FUNCTN over the unit hypercube
    // using randomized Korobov rules for the first 100 variables (Cranley & Patterson;
    // Keast's optimal lattice parameters), falling back to Niederreiter's method beyond
    // that. RNG draw order is sacred: `dksmrc` below draws (NK-1) shuffle values then
    // NDIM shift values from `mvnuni_`, per sample.
    //
    // Divergence note: the C# `MINVLS` parameter is NOT `ref`/`out` (unlike the
    // corresponding Fortran INOUT argument), so the `MINVLS = INTVLS;` reassignment at
    // the very end of the C# method (and here) only mutates the local copy and is
    // discarded on return -- the "continue a previous calculation" (negative MINVLS)
    // re-entry protocol this exists to support is therefore unreachable through this
    // class's only call site (`mvndst` always passes `IVLS = 0`). Ported verbatim
    // (`MINVLS` taken by value, final reassignment kept as dead code).
    void dkbvrc(int NDIM, int MINVLS, int MAXVLS,
                const std::function<double(int, const std::vector<double>&)>& FUNCTN, double ABSEPS,
                double RELEPS, double& ABSERR, double& FINEST, int& INFORM) const {
        int PLIM = 28, NLIM = 1000, KLIM = 100, KLIMI, K, INTVLS, MINSMP = 8;

        double DIFINT, FINVAL, VARSQR, VARPRD, VALUE = 0;
        std::vector<double> X(static_cast<std::size_t>(2 * NLIM), 0.0);
        std::vector<double> VK(static_cast<std::size_t>(NLIM), 0.0);

        // DKBVRC SAVE P, C, SAMPLS, NP, VAREST (P/C are the static kDkbvrcP/kDkbvrcC
        // tables below; SAMPLS/NP/VAREST are the mutable dkbvrc_sampls_/np_/varest_
        // instance fields).
        INFORM = 1;
        INTVLS = 0;
        KLIMI = KLIM;
        if (MINVLS >= 0) {
            FINEST = 0;
            dkbvrc_varest_ = 0;
            dkbvrc_sampls_ = MINSMP;
            for (int i = std::min(NDIM, 10) - 1; i < PLIM; i++) {
                dkbvrc_np_ = i;
                if (MINVLS < 2 * dkbvrc_sampls_ * kDkbvrcP[i]) {
                    goto ten;
                }
            }
            dkbvrc_sampls_ = std::max(MINSMP, MINVLS / (2 * kDkbvrcP[dkbvrc_np_]));
        }

    ten:

        VK[0] = 1.0 / kDkbvrcP[dkbvrc_np_];
        K = 1;
        for (int i = 1; i < NDIM; i++) {
            if (i < KLIM) {
                // Preserve C#'s double `%` (== std::fmod) exactly, per the task brief.
                K = static_cast<int>(std::fmod(
                    kDkbvrcC[dkbvrc_np_][std::min(NDIM - 1, KLIM - 1) - 1] * static_cast<double>(K),
                    static_cast<double>(kDkbvrcP[dkbvrc_np_])));
                VK[static_cast<std::size_t>(i)] = K * VK[0];
            } else {
                VK[static_cast<std::size_t>(i)] =
                    static_cast<int>(std::pow(kDkbvrcP[dkbvrc_np_] * 2,
                                               (static_cast<double>(i - KLIM) / (NDIM - KLIM + 1))));
                VK[static_cast<std::size_t>(i)] = std::fmod(VK[static_cast<std::size_t>(i)] / kDkbvrcP[dkbvrc_np_], 1.0);
            }
        }

        FINVAL = 0;
        VARSQR = 0;

        for (int i = 1; i <= dkbvrc_sampls_; i++) {
            VALUE = 0;
            dksmrc(NDIM, KLIMI, VALUE, kDkbvrcP[dkbvrc_np_], VK, FUNCTN, X);
            DIFINT = (VALUE - FINVAL) / i;
            FINVAL += DIFINT;
            VARSQR = (i - 2) * VARSQR / i + DIFINT * DIFINT;
        }

        INTVLS = INTVLS + 2 * dkbvrc_sampls_ * kDkbvrcP[dkbvrc_np_];
        VARPRD = dkbvrc_varest_ * VARSQR;
        FINEST = FINEST + (FINVAL - FINEST) / (1 + VARPRD);
        if (VARSQR > 0) dkbvrc_varest_ = (1 + VARPRD) / VARSQR;
        ABSERR = 7 * std::sqrt(VARSQR / (1 + VARPRD)) / 2;

        if (ABSERR > std::max(ABSEPS, std::fabs(FINEST) * RELEPS)) {
            if (dkbvrc_np_ < PLIM - 1) {
                dkbvrc_np_ = dkbvrc_np_ + 1;
            } else {
                dkbvrc_sampls_ = std::min(3 * dkbvrc_sampls_ / 2, (MAXVLS - INTVLS) / (2 * kDkbvrcP[dkbvrc_np_]));
                dkbvrc_sampls_ = std::max(MINSMP, dkbvrc_sampls_);
            }
            if (INTVLS + 2 * dkbvrc_sampls_ * kDkbvrcP[dkbvrc_np_] <= MAXVLS) {
                goto ten;
            }
        } else {
            INFORM = 0;
        }

        MINVLS = INTVLS;  // dead store -- see the divergence note above (MINVLS is by-value)
        (void)MINVLS;
    }

    // RNG draw order is sacred: (NK-1) shuffle draws then NDIM shift draws, both from
    // `mvnuni_`, per sample -- any reordering here silently diverges the MersenneTwister
    // stream from the C# reference and fails the seeded fixtures (that's the point of
    // those fixtures: catching exactly this class of transcription slip).
    void dksmrc(int NDIM, int KLIM, double& SUMKRO, int PRIME, std::vector<double>& VK,
                const std::function<double(int, const std::vector<double>&)>& FUNCTN,
                std::vector<double>& X) const {
        double sampleValue;
        double XT;
        int JP;
        SUMKRO = 0;
        int NK = std::min(NDIM, KLIM);
        for (int i = 0; i < NK - 1; i++) {
            sampleValue = mvnuni_.next_double();
            JP = static_cast<int>(i + sampleValue * (NK - i));
            XT = VK[static_cast<std::size_t>(i)];
            VK[static_cast<std::size_t>(i)] = VK[static_cast<std::size_t>(JP)];
            VK[static_cast<std::size_t>(JP)] = XT;
        }

        for (int i = 0; i < NDIM; i++) {
            X[static_cast<std::size_t>(NDIM + i)] = mvnuni_.next_double();
        }

        for (int i = 1; i <= PRIME; i++) {
            for (int j = 0; j < NDIM; j++) {
                X[static_cast<std::size_t>(j)] =
                    std::fabs(2.0 * std::fmod(i * VK[static_cast<std::size_t>(j)] + X[static_cast<std::size_t>(NDIM + j)], 1.0) - 1);
            }

            SUMKRO += (FUNCTN(NDIM, X) - SUMKRO) / (2 * i - 1);

            for (int j = 0; j < NDIM; j++) {
                X[static_cast<std::size_t>(j)] = 1 - X[static_cast<std::size_t>(j)];
            }
            SUMKRO += (FUNCTN(NDIM, X) - SUMKRO) / (2 * i);
        }
    }

    // A function for computing bivariate normal probabilities (used by mvndnt's
    // N-INFIS==2 branch). See BVU for the general algorithm this specializes the
    // integration-limits handling around.
    double bvnmvn(const std::vector<double>& LOWER, const std::vector<double>& UPPER,
                  const std::vector<int>& INFIN, double CORREL) const {
        double result = 0;
        if (INFIN[0] == 2 && INFIN[1] == 2) {
            result = bvu(LOWER[0], LOWER[1], CORREL) - bvu(UPPER[0], LOWER[1], CORREL) -
                     bvu(LOWER[0], UPPER[1], CORREL) + bvu(UPPER[0], UPPER[1], CORREL);
        } else if (INFIN[0] == 2 && INFIN[1] == 1) {
            result = bvu(LOWER[0], LOWER[1], CORREL) - bvu(UPPER[0], LOWER[1], CORREL);
        } else if (INFIN[0] == 1 && INFIN[1] == 2) {
            result = bvu(LOWER[0], LOWER[1], CORREL) - bvu(LOWER[0], UPPER[1], CORREL);
        } else if (INFIN[0] == 2 && INFIN[1] == 0) {
            result = bvu(-UPPER[0], -UPPER[1], CORREL) - bvu(-LOWER[0], -UPPER[1], CORREL);
        } else if (INFIN[0] == 0 && INFIN[1] == 2) {
            result = bvu(-UPPER[0], -UPPER[1], CORREL) - bvu(-UPPER[0], -LOWER[1], CORREL);
        } else if (INFIN[0] == 1 && INFIN[1] == 0) {
            result = bvu(LOWER[0], -UPPER[1], -CORREL);
        } else if (INFIN[0] == 0 && INFIN[1] == 1) {
            result = bvu(-UPPER[0], LOWER[1], -CORREL);
        } else if (INFIN[0] == 1 && INFIN[1] == 1) {
            result = bvu(LOWER[0], LOWER[1], CORREL);
        } else if (INFIN[0] == 0 && INFIN[1] == 0) {
            result = bvu(-UPPER[0], -UPPER[1], CORREL);
        }

        // This can sometimes produce very small, but negative values (approx. -5E-73)
        if (result < 0.0) result = 0.0;
        return result;
    }

    static sampling::MersenneTwister make_clock_seeded() {
        auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return sampling::MersenneTwister(static_cast<std::uint32_t>(ticks));
    }

    static sampling::MersenneTwister make_master(int seed) {
        if (seed > 0) return sampling::MersenneTwister(static_cast<std::uint32_t>(seed));
        return make_clock_seeded();
    }

    // (`kInf` is inherited from MultivariateDistribution -- protected static constexpr,
    // usable unqualified here; not redeclared, matching the Dirichlet/Multinomial precedent.)

    static constexpr double kBvndWn20[10] = {
        0.017614007139152121, 0.040601429800386939, 0.062672048334109054, 0.083276741576704755,
        0.1019301198172404,   0.1181945319615184,   0.13168863844917661,  0.1420961093183821,
        0.14917298647260371,  0.15275338713072589,
    };
    static constexpr double kBvndXn20[10] = {
        -0.99312859918509488, -0.96397192727791381, -0.912234428251326,   -0.83911697182221878,
        -0.7463319064601508,  -0.636053680726515,   -0.51086700195082713, -0.3737060887154196,
        -0.2277858511416451,  -0.076526521133497324,
    };
    static constexpr double kBvndWn12[6] = {
        0.047175336386511772, 0.1069393259953183, 0.16007832854334639,
        0.2031674267230659,   0.2334925365383547, 0.24914704581340291,
    };
    static constexpr double kBvndXn12[6] = {
        -0.98156063424671913, -0.904117256370475, -0.769902674194305,
        -0.58731795428661715, -0.36783149899818018, -0.12523340851146919,
    };
    static constexpr double kBvndWn6[3] = {0.1713244923791705, 0.36076157304813838, 0.46791393457269043};
    static constexpr double kBvndXn6[3] = {-0.93246951420315216, -0.6612093864662647, -0.238619186083197};

    // Optimal Parameters for Lattice Rules (Cranley-Patterson / Keast).
    static constexpr int kDkbvrcP[28] = {31, 47, 73, 113, 173, 263, 397, 593, 907, 1361, 2053, 3079, 4621, 6947, 10427, 15641, 23473, 35221, 52837, 79259, 118891, 178349, 267523, 401287, 601943, 902933, 1354471, 2031713};
    static constexpr int kDkbvrcC[28][99] = {
    {12, 9, 9, 13, 12, 12, 12, 12, 12, 12, 12, 12, 3, 3, 3, 12, 7, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12, 3, 3, 3, 12, 7, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12, 3, 3, 3, 12, 7, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12, 3, 3, 3, 12, 7, 7, 12, 12, 12, 12, 12, 12, 12, 12, 7, 3, 3, 3, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
    {13, 11, 17, 10, 15, 15, 15, 15, 15, 15, 22, 15, 15, 6, 6, 6, 15, 15, 9, 13, 2, 2, 2, 13, 11, 11, 10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 6, 6, 6, 15, 15, 9, 13, 2, 2, 2, 13, 11, 11, 10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 6, 6, 6, 15, 15, 9, 13, 2, 2, 2, 13, 11, 11, 10, 10, 15, 15, 15, 15, 15, 15, 15, 15, 6, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    {27, 28, 10, 11, 11, 20, 11, 11, 28, 13, 13, 28, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 31, 31, 5, 5, 5, 31, 13, 11, 11, 11, 11, 11, 11, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 31, 31, 5, 5, 5, 11, 13, 11, 11, 11, 11, 11, 11, 11, 13, 13, 11, 13, 5, 5, 5, 5, 14, 13, 5, 5, 5, 5, 5, 5, 5, 5},
    {35, 27, 27, 36, 22, 29, 29, 20, 45, 5, 5, 5, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 29, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 21, 27, 3, 3, 3, 24, 27, 27, 17, 29, 29, 29, 17, 5, 5, 5, 5, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 17, 17, 17, 6, 17, 17, 6, 3, 6, 6, 3, 3, 3, 3, 3},
    {64, 66, 28, 28, 44, 44, 55, 67, 10, 10, 10, 10, 10, 10, 38, 38, 10, 10, 10, 10, 10, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 38, 38, 31, 4, 4, 31, 64, 4, 4, 4, 64, 45, 45, 45, 45, 45, 45, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 11, 66, 66, 66, 66, 66, 66, 66, 66, 66, 45, 11, 7, 3, 2, 2, 2, 27, 5, 3, 3, 5, 5, 2, 2, 2, 2, 2, 2, 2},
    {111, 42, 54, 118, 20, 31, 31, 72, 17, 94, 14, 14, 11, 14, 14, 14, 94, 10, 10, 10, 10, 14, 14, 14, 14, 14, 14, 14, 11, 11, 11, 8, 8, 8, 8, 8, 8, 8, 18, 18, 18, 18, 18, 113, 62, 62, 45, 45, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 63, 63, 53, 63, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 51, 51, 51, 51, 51, 12, 51, 12, 51, 5, 3, 3, 2, 2, 5},
    {163, 154, 83, 43, 82, 92, 150, 59, 76, 76, 47, 11, 11, 100, 131, 116, 116, 116, 116, 116, 116, 138, 138, 138, 138, 138, 138, 138, 138, 138, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 116, 116, 116, 116, 116, 116, 100, 100, 100, 100, 100, 138, 138, 138, 138, 138, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 38, 38, 38, 38, 38, 38, 38, 38, 3, 3, 3, 3, 3},
    {246, 189, 242, 102, 250, 250, 102, 250, 280, 118, 196, 118, 191, 215, 121, 121, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 161, 161, 161, 161, 161, 161, 161, 161, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 10, 10, 10, 10, 10, 10, 103, 10, 10, 10, 10, 5},
    {347, 402, 322, 418, 215, 220, 339, 339, 339, 337, 218, 315, 315, 315, 315, 167, 167, 167, 167, 361, 201, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 231, 231, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 48, 48, 48, 48, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 283, 283, 283, 283, 283, 283, 283, 283, 283, 16, 283, 16, 283, 283},
    {505, 220, 601, 644, 612, 160, 206, 206, 206, 422, 134, 518, 134, 134, 518, 652, 382, 206, 158, 441, 179, 441, 56, 559, 559, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 101, 101, 56, 101, 101, 101, 101, 101, 101, 101, 101, 193, 193, 193, 193, 193, 193, 193, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 101, 101, 101, 101},
    {794, 325, 960, 528, 247, 247, 338, 366, 847, 753, 753, 236, 334, 334, 461, 711, 652, 381, 381, 381, 652, 381, 381, 381, 381, 381, 381, 381, 226, 326, 326, 326, 326, 326, 326, 326, 126, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 195, 195, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 195, 195, 195, 195, 195, 195, 195, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 387, 387, 387, 387, 387, 387, 387, 387, 387, 387, 387, 387, 387},
    {1189, 888, 259, 1082, 725, 811, 636, 965, 497, 497, 1490, 1490, 392, 1291, 508, 508, 1291, 1291, 508, 1291, 508, 508, 867, 867, 867, 867, 934, 867, 867, 867, 867, 867, 867, 867, 1284, 1284, 1284, 1284, 1284, 1284, 1284, 1284, 1284, 563, 563, 563, 563, 1010, 1010, 1010, 208, 838, 563, 563, 563, 759, 759, 564, 759, 759, 801, 801, 801, 801, 759, 759, 759, 759, 759, 563, 563, 563, 563, 563, 563, 563, 563, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226},
    {1763, 1018, 1500, 432, 1332, 2203, 126, 2240, 1719, 1284, 878, 1983, 266, 266, 266, 266, 747, 747, 127, 127, 2074, 127, 2074, 1400, 1383, 1383, 1383, 1383, 1383, 1383, 1383, 1383, 1383, 1383, 1400, 1383, 1383, 1383, 1383, 1383, 1383, 1383, 507, 1073, 1073, 1073, 1073, 1990, 1990, 1990, 1990, 1990, 507, 507, 507, 507, 507, 507, 507, 507, 507, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 1073, 22, 22, 22, 22, 22, 22, 1073, 452, 452, 452, 452, 452, 452, 318, 301, 301, 301, 301, 86, 86, 15},
    {2872, 3233, 1534, 2941, 2910, 393, 1796, 919, 446, 919, 919, 1117, 103, 103, 103, 103, 103, 103, 103, 2311, 3117, 1101, 3117, 3117, 1101, 1101, 1101, 1101, 1101, 2503, 2503, 2503, 2503, 2503, 2503, 2503, 2503, 429, 429, 429, 429, 429, 429, 429, 1702, 1702, 1702, 184, 184, 184, 184, 184, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 784, 784, 784, 784, 784, 784, 784, 784, 784, 784, 784, 784, 784},
    {4309, 3758, 4034, 1963, 730, 642, 1502, 2246, 3834, 1511, 1102, 1102, 1522, 1522, 3427, 3427, 3928, 915, 915, 3818, 3818, 3818, 3818, 4782, 4782, 4782, 3818, 4782, 3818, 3818, 1327, 1327, 1327, 1327, 1327, 1327, 1327, 1387, 1387, 1387, 1387, 1387, 1387, 1387, 1387, 1387, 2339, 2339, 2339, 2339, 2339, 2339, 2339, 2339, 2339, 2339, 2339, 2339, 2339, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 3148, 1776, 1776, 1776, 3354, 3354, 3354, 925, 3354, 3354, 925, 925, 925, 925, 925, 2133, 2133, 2133, 2133, 2133, 2133, 2133, 2133},
    {6610, 6977, 1686, 3819, 2314, 5647, 3953, 3614, 5115, 423, 423, 5408, 7426, 423, 423, 487, 6227, 2660, 6227, 1221, 3811, 197, 4367, 351, 1281, 1221, 351, 351, 351, 7245, 1984, 2999, 2999, 2999, 2999, 2999, 2999, 3995, 2063, 2063, 2063, 2063, 1644, 2063, 2077, 2512, 2512, 2512, 2077, 2077, 2077, 2077, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 754, 1097, 1097, 754, 754, 754, 754, 248, 754, 1097, 1097, 1097, 1097, 222, 222, 222, 222, 754, 1982, 1982, 1982, 1982, 1982, 1982, 1982, 1982, 1982, 1982, 1982},
    {9861, 3647, 4073, 2535, 3430, 9865, 2830, 9328, 4320, 5913, 10365, 8272, 3706, 6186, 7806, 7806, 7806, 8610, 2563, 11558, 11558, 9421, 1181, 9421, 1181, 1181, 1181, 9421, 1181, 1181, 10574, 10574, 3534, 3534, 3534, 3534, 3534, 2898, 2898, 2898, 3450, 2141, 2141, 2141, 2141, 2141, 2141, 2141, 7055, 7055, 7055, 7055, 7055, 7055, 7055, 7055, 7055, 7055, 7055, 7055, 7055, 7055, 7055, 2831, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 8204, 4688, 4688, 4688, 2831, 2831, 2831, 2831, 2831, 2831, 2831, 2831},
    {10327, 7582, 7124, 8214, 9600, 10271, 10193, 10800, 9086, 2365, 4409, 13812, 5661, 9344, 9344, 10362, 9344, 9344, 8585, 11114, 13080, 13080, 13080, 6949, 3436, 3436, 3436, 13213, 6130, 6130, 8159, 8159, 11595, 8159, 3436, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 7096, 4377, 7096, 4377, 4377, 4377, 4377, 4377, 5410, 5410, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377, 440, 440, 1199, 1199, 1199},
    {19540, 19926, 11582, 11113, 24585, 8726, 17218, 419, 4918, 4918, 4918, 15701, 17710, 4037, 4037, 15808, 11401, 19398, 25950, 25950, 4454, 24987, 11719, 8697, 1452, 1452, 1452, 1452, 1452, 8697, 8697, 6436, 21475, 6436, 22913, 6434, 18497, 11089, 11089, 11089, 11089, 3036, 3036, 14208, 14208, 14208, 14208, 12906, 12906, 12906, 12906, 12906, 12906, 12906, 12906, 7614, 7614, 7614, 7614, 5021, 5021, 5021, 5021, 5021, 5021, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 10145, 4544, 4544, 4544, 4544, 4544, 4544, 8394, 8394, 8394, 8394},
    {34566, 9579, 12654, 26856, 37873, 38806, 29501, 17271, 3663, 10763, 18955, 1298, 26560, 17132, 17132, 4753, 4753, 8713, 18624, 13082, 6791, 1122, 19363, 34695, 18770, 18770, 18770, 18770, 15628, 18770, 18770, 18770, 18770, 33766, 20837, 20837, 20837, 20837, 20837, 20837, 6545, 6545, 6545, 6545, 6545, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 30483, 30483, 30483, 30483, 30483, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 12138, 9305, 11107, 11107, 11107, 11107, 11107, 11107, 11107, 11107, 11107, 11107, 11107, 11107, 11107, 9305, 9305},
    {31929, 49367, 10982, 3527, 27066, 13226, 56010, 18911, 40574, 20767, 20767, 9686, 47603, 47603, 11736, 11736, 41601, 12888, 32948, 30801, 44243, 53351, 53351, 16016, 35086, 35086, 32581, 2464, 2464, 49554, 2464, 2464, 49554, 49554, 2464, 81, 27260, 10681, 2185, 2185, 2185, 2185, 2185, 2185, 2185, 18086, 18086, 18086, 18086, 18086, 17631, 17631, 18086, 18086, 18086, 37335, 37774, 37774, 37774, 26401, 26401, 26401, 26401, 26401, 26401, 26401, 26401, 26401, 26401, 26401, 26401, 26401, 12982, 40398, 40398, 40398, 40398, 40398, 40398, 3518, 3518, 3518, 37799, 37799, 37799, 37799, 37799, 37799, 37799, 37799, 37799, 4721, 4721, 4721, 4721, 7067, 7067, 7067, 7067},
    {40701, 69087, 77576, 64590, 39397, 33179, 10858, 38935, 43129, 35468, 35468, 5279, 61518, 61518, 27945, 70975, 70975, 86478, 86478, 20514, 20514, 73178, 73178, 43098, 43098, 4701, 59979, 59979, 58556, 69916, 15170, 15170, 4832, 4832, 43064, 71685, 4832, 15170, 15170, 15170, 27679, 27679, 27679, 60826, 60826, 6187, 6187, 4264, 4264, 4264, 4264, 4264, 45567, 32269, 32269, 32269, 32269, 62060, 62060, 62060, 62060, 62060, 62060, 62060, 62060, 62060, 1803, 1803, 1803, 1803, 1803, 1803, 1803, 1803, 1803, 1803, 1803, 1803, 1803, 51108, 51108, 51108, 51108, 51108, 51108, 51108, 51108, 51108, 51108, 51108, 51108, 55315, 55315, 54140, 54140, 54140, 54140, 54140, 13134},
    {103650, 125480, 59978, 46875, 77172, 83021, 126904, 14541, 56299, 43636, 11655, 52680, 88549, 29804, 101894, 113675, 48040, 113675, 34987, 48308, 97926, 5475, 49449, 6850, 62545, 62545, 9440, 33242, 9440, 33242, 9440, 33242, 9440, 62850, 9440, 9440, 9440, 90308, 90308, 90308, 47904, 47904, 47904, 47904, 47904, 47904, 47904, 47904, 47904, 41143, 41143, 41143, 41143, 41143, 41143, 41143, 36114, 36114, 36114, 36114, 36114, 24997, 65162, 65162, 65162, 65162, 65162, 65162, 65162, 65162, 65162, 65162, 65162, 65162, 65162, 65162, 47650, 47650, 47650, 47650, 47650, 47650, 47650, 40586, 40586, 40586, 40586, 40586, 40586, 40586, 38725, 38725, 38725, 38725, 88329, 88329, 88329, 88329, 88329},
    {165843, 90647, 59925, 189541, 67647, 74795, 68365, 167485, 143918, 74912, 167289, 75517, 8148, 172106, 126159, 35867, 35867, 35867, 121694, 52171, 95354, 113969, 113969, 76304, 123709, 123709, 144615, 123709, 64958, 64958, 32377, 193002, 193002, 25023, 40017, 141605, 189165, 189165, 141605, 189165, 189165, 141605, 141605, 141605, 189165, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127047, 127785, 127785, 127785, 127785, 127785, 127785, 127785, 127785, 127785, 127785, 80822, 80822, 80822, 80822, 80822, 80822, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 131661, 7114, 131661},
    {130365, 236711, 110235, 125699, 56483, 93735, 234469, 60549, 1291, 93937, 245291, 196061, 258647, 162489, 176631, 204895, 73353, 172319, 28881, 136787, 122081, 122081, 275993, 64673, 211587, 211587, 211587, 282859, 282859, 211587, 242821, 256865, 256865, 256865, 122203, 291915, 122203, 291915, 291915, 122203, 25639, 25639, 291803, 245397, 284047, 245397, 245397, 245397, 245397, 245397, 245397, 245397, 94241, 66575, 66575, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 217673, 210249, 210249, 210249, 210249, 210249, 210249, 210249, 210249, 210249, 210249, 94453, 94453, 94453, 94453, 94453, 94453, 94453, 94453, 94453, 94453, 94453, 94453, 94453, 94453, 94453},
    {333459, 375354, 102417, 383544, 292630, 41147, 374614, 48032, 435453, 281493, 358168, 114121, 346892, 238990, 317313, 164158, 35497, 70530, 70530, 434839, 24754, 24754, 24754, 393656, 118711, 118711, 148227, 271087, 355831, 91034, 417029, 417029, 91034, 91034, 417029, 91034, 299843, 299843, 413548, 413548, 308300, 413548, 413548, 413548, 308300, 308300, 308300, 413548, 308300, 308300, 308300, 308300, 308300, 15311, 15311, 15311, 15311, 176255, 176255, 23613, 23613, 23613, 23613, 23613, 23613, 172210, 204328, 204328, 204328, 204328, 121626, 121626, 121626, 121626, 121626, 200187, 200187, 200187, 200187, 200187, 121551, 121551, 248492, 248492, 248492, 248492, 248492, 248492, 248492, 248492, 248492, 248492, 248492, 248492, 13942, 13942, 13942, 13942, 13942},
    {500884, 566009, 399251, 652979, 355008, 430235, 328722, 670680, 405585, 405585, 424646, 670180, 670180, 641587, 215580, 59048, 633320, 81010, 20789, 389250, 389250, 638764, 638764, 389250, 389250, 398094, 80846, 147776, 147776, 296177, 398094, 398094, 147776, 147776, 396313, 578233, 578233, 578233, 19482, 620706, 187095, 620706, 187095, 126467, 241663, 241663, 241663, 241663, 241663, 241663, 241663, 241663, 241663, 241663, 241663, 241663, 321632, 23210, 23210, 394484, 394484, 394484, 78101, 78101, 78101, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 542095, 277743, 277743, 277743, 457259, 457259, 457259, 457259, 457259, 457259, 457259, 457259, 457259, 457259, 457259, 457259},
    {858339, 918142, 501970, 234813, 460565, 31996, 753018, 256150, 199809, 993599, 245149, 794183, 121349, 150619, 376952, 809123, 809123, 804319, 67352, 969594, 434796, 969594, 804319, 391368, 761041, 754049, 466264, 754049, 754049, 466264, 754049, 754049, 282852, 429907, 390017, 276645, 994856, 250142, 144595, 907454, 689648, 687580, 687580, 687580, 687580, 978368, 687580, 552742, 105195, 942843, 768249, 307142, 307142, 307142, 307142, 880619, 880619, 880619, 880619, 880619, 880619, 880619, 117185, 117185, 117185, 117185, 117185, 117185, 117185, 117185, 117185, 117185, 117185, 60731, 60731, 60731, 60731, 60731, 60731, 60731, 60731, 60731, 60731, 60731, 178309, 178309, 178309, 178309, 74373, 74373, 74373, 74373, 74373, 74373, 74373, 74373, 214965, 214965, 214965},
    };

    // --- Fields (mirrors the C# field declaration order) ---

    bool parameters_valid_ = true;  // dead state -- see file header divergence note
    int dimension_ = 0;
    std::vector<double> mean_;
    la::Matrix covariance_ = la::Matrix(0, 0);
    la::DecompositionMethod decomposition_ = la::DecompositionMethod::Cholesky;
    std::optional<la::CholeskyDecomposition> cholesky_;
    mutable std::optional<la::SingularValueDecomposition> svd_;
    la::Matrix factor_ = la::Matrix(0, 0);
    la::Matrix nullspace_ = la::Matrix(0, 0);
    double svd_threshold_ = 0.0;
    int rank_ = 0;
    double lnconstant_ = 0.0;
    // (`_variance`/`_standardDeviation` lazy-cache fields omitted -- see file header note)

    // Tracks the non-throwing mutable-covariance state (v2.1.4). NOT copied by clone() --
    // matches C#'s Clone(), which likewise omits `_densityValid` (a fresh clone always
    // starts density-valid, regardless of the original's state).
    bool density_valid_ = true;

    // Variables required for the multivariate CDF. `mutable`: MultivariateDistribution's
    // `cdf()` is `const`, but MVNDST-backed evaluation genuinely mutates the correlation
    // cache, the per-call `upper_` scratch, and the MVNUNI RNG stream -- mirroring the C#
    // source's mutable instance state (this port's documented relaxed-mutation rule for
    // stateful binding/model objects; see corehydro/CLAUDE.md).
    mutable la::Matrix correlation_ = la::Matrix(0, 0);
    mutable std::vector<double> correl_;
    mutable sampling::MersenneTwister mvnuni_ =
        sampling::MersenneTwister(static_cast<std::uint32_t>(kDefaultMvnuniSeed));
    int max_evaluations_ = 100000;
    double absolute_error_ = 1E-4;
    double relative_error_ = 1E-4;
    std::vector<double> lower_;
    mutable std::vector<double> upper_;
    std::vector<int> infin_;
    mutable bool correlation_matrix_created_ = false;
    bool cov_srted_ = false;  // dead field -- see file header divergence note

    // SAVE A, B, INFI, COV / NL = 500 default, reassigned per-instance in set_parameters
    // (see the NL-quirk comment there).
    int nl_ = 500;

    // MVNDST working storage (C# `//SAVE A, B, INFI, COV`; NL = 500 default). Sized fixed
    // at 500 / 500*501/2 = 125250 regardless of `dimension_`/`nl_` -- `mvndfn` only reads
    // the first N+1 / NL entries actually populated by `covsrt` for the current call, so
    // the surplus capacity is simply unused headroom (matches the C# fixed-size arrays).
    // Not copied by clone() -- matches the C# `Clone()`, which also omits them (they are
    // reentrant scratch space, rebuilt by `covsrt` on every `mvndst` call).
    mutable std::vector<double> mvndfn_a_ = std::vector<double>(500, 0.0);
    mutable std::vector<double> mvndfn_b_ = std::vector<double>(500, 0.0);
    mutable std::vector<int> mvndfn_infi_ = std::vector<int>(500, 0);
    mutable std::vector<double> mvndfn_cov_ = std::vector<double>(125250, 0.0);  // NL*(NL+1)/2 at NL=500

    mutable int dkbvrc_sampls_ = 0;
    mutable int dkbvrc_np_ = 0;
    mutable double dkbvrc_varest_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
