// ported from: Numerics/Distributions/Multivariate/MultivariateStudentT.cs @ 7e8e8d1
//
// Re-audited against v2.1.4's "Harden distribution parameter validation" wave:
// ValidateParameters gained three NaN/Infinity checks -- degrees_of_freedom (previously only
// `<= 0`, which is false for both NaN and +Infinity), every location element (previously
// unchecked), and every scale-matrix element (previously unchecked; a non-finite entry would
// otherwise propagate into the Cholesky decomposition and surface as a confusing
// not-positive-definite failure rather than a direct "must be finite" one). SetParameters
// itself is unchanged: it already called validate_parameters(..., /*throw=*/true) BEFORE
// mutating any field (transactional -- an invalid call throws and leaves the previous state
// untouched), matching the new C# exactly; this ordering predates this wave for MVT (unlike
// the univariate distributions in this same wave, which had the OLD buggy
// validate-via-stale-sibling-property-setter pattern). Because SetParameters always throws
// rather than leaving the object alive in an invalid state, `parameters_valid_` stays dead
// state here (see the divergence note below) and there is no "construct valid -> mutate
// invalid -> read parameters_valid()" sequence to fixture -- this exact class of throw-only
// validation is already documented as out of fixture scope in
// fixtures/distributions/multivariate/multivariate_student_t.json's file-level "note" (the
// Test_ParameterValidation_*/Test_ValidateParameters_ReturnException skips), so no new
// fixture case is added here; the three new checks are covered by mirroring the C# source
// line-for-line and by the existing valid-construction fixtures continuing to pass.
//
// The Multivariate Student's t-distribution, dimensions >= 1, degrees of freedom v (nu),
// location vector mu, and (not-covariance) scale matrix Sigma. PDF/LogPDF/Mahalanobis are
// closed-form; CDF for dim 1 delegates to the univariate StudentT CDF, and for dim >= 2
// uses a deterministic K=200 stratified-quantile chi-square(v) mixture over
// MultivariateNormal's CDF (see cdf() below): every quantile is a midpoint of an
// equal-probability stratum, so the mixture construction itself has no seeded/statistical
// component. For dim <= 2 this makes the result fully bit-reproducible for a given (v, mu,
// Sigma, x), because the inner MultivariateNormal::cdf() calls it drives are themselves
// closed-form/deterministic at dim 1-2. For dim >= 3, though, each of the 200 inner
// MultivariateNormal::cdf() calls runs the seeded Genz-Bretz quasi-Monte-Carlo integrator
// (see multivariate_normal.hpp), and the MultivariateNormal instance this constructor
// builds is never given an explicit seed, so its `mvnuni_` stream defaults to
// make_clock_seeded() -- meaning MVT's own cdf() is NOT bit-reproducible across runs at
// dim >= 3 (a faithful mirror of the upstream C# `_MVNUNI = new MersenneTwister()` default
// in MultivariateNormal.cs, not a divergence introduced by this port; no fixture exercises
// MVT CDF at dim >= 3). Member order mirrors the C# source throughout.
//
// Divergence notes (see also docs/upstream-csharp-issues.md for the running log):
//   - `Variance`/`StandardDeviation` are computed on every call rather than lazily cached
//     in nullable backing fields (the C# `_variance`/`_standardDeviation` pattern) -- same
//     simplification and rationale as MultivariateNormal's port (see that file's header):
//     the C# caching is a micro-optimization orthogonal to the math, and this port's
//     "never mutate a value type without reason" convention prefers recomputation.
//     Results are identical.
//   - The C# null-checks on the `location`/`scaleMatrix` constructor arguments
//     (`ArgumentOutOfRangeException` on `null`) have no C++ equivalent (references/vectors
//     cannot be null) and are omitted from `validate_parameters`, matching
//     MultivariateNormal's precedent.
//   - `ParametersValid`/`_parametersValid` is ported verbatim as a field but is DEAD STATE
//     in the C# source (initialized `true`, never reassigned -- `SetParameters` throws on
//     invalid input rather than flipping the flag), matching MultivariateNormal's
//     `_parametersValid`. `IsPositiveDefinite` is likewise always true for any
//     successfully-constructed instance and is not independently exercised by the fixture
//     suite (same precedent as MultivariateNormal).
//   - Fidelity note (not a divergence): unlike `validate_parameters`, `set_parameters`
//     itself does not wrap its direct `CholeskyDecomposition` construction in a try/catch
//     -- this matches the C# source exactly (`SetParameters`'s `_cholesky = new
//     CholeskyDecomposition(_scaleMatrix);` has no try/catch either; only
//     `ValidateParameters`'s internal probe construction does). See `validate_parameters`'s
//     own comment for why its try/catch is otherwise a no-op given this port's Cholesky
//     exception vocabulary.
//   - Fidelity note (not a divergence): `validate_parameters` only requires `v > 0`, unlike
//     the univariate `StudentT` port's `v >= 1` floor (see student_t.hpp). This is
//     transcribed verbatim from the C# source (`ValidateParameters` here has no lower bound
//     beyond positivity). A dimension-1 instance with `0 < v < 1` therefore constructs
//     successfully but its `cdf()` -- which delegates to a univariate `StudentT` -- will
//     throw when called, since that inner distribution's own stricter validation marks
//     itself invalid. No fixture exercises this corner; noted for completeness.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/student_t.hpp"
#include "corehydro/numerics/math/linalg/cholesky_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/sampling/latin_hypercube.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/sampling/stratification_bin.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace la = corehydro::numerics::math::linalg;
namespace sf = corehydro::numerics::math::special;

class MultivariateStudentT : public MultivariateDistribution {
   public:
    void set_mvnuni_seed(int seed) {
        mvnuni_ = sampling::MersenneTwister(static_cast<std::uint32_t>(seed));
    }
    // Constructs a standard multivariate Student's t-distribution with zero location vector,
    // identity scale matrix, and the specified degrees of freedom.
    MultivariateStudentT(int dimension, double degrees_of_freedom) {
        std::vector<double> location(static_cast<std::size_t>(dimension), 0.0);
        set_parameters(degrees_of_freedom, std::move(location), la::Matrix::identity(dimension).to_array());
    }

    // Constructs a multivariate Student's t-distribution with the specified location vector,
    // identity scale matrix, and degrees of freedom.
    MultivariateStudentT(double degrees_of_freedom, std::vector<double> location) {
        int dim = static_cast<int>(location.size());
        set_parameters(degrees_of_freedom, std::move(location), la::Matrix::identity(dim).to_array());
    }

    // Constructs a multivariate Student's t-distribution with the specified degrees of
    // freedom, location vector, and scale matrix. Note: `scale_matrix` is the scale matrix,
    // not the covariance matrix -- the covariance is v/(v-2)*Sigma for v > 2.
    MultivariateStudentT(double degrees_of_freedom, std::vector<double> location,
                         std::vector<std::vector<double>> scale_matrix) {
        set_parameters(degrees_of_freedom, std::move(location), std::move(scale_matrix));
    }

    // --- Identity / parameters ---
    int dimension() const override { return dimension_; }
    MultivariateDistributionType type() const override {
        return MultivariateDistributionType::MultivariateStudentT;
    }
    std::string display_name() const override { return "Multivariate Student's t"; }
    std::string short_display_name() const override { return "Multi-T"; }
    bool parameters_valid() const override { return parameters_valid_; }

    // Gets the degrees of freedom v (nu) of the distribution. Validity floor is v > 0 (see
    // the file header's fidelity note -- looser than univariate StudentT's v >= 1).
    double degrees_of_freedom() const { return degrees_of_freedom_; }

    // Gets the location vector mu of the distribution.
    const std::vector<double>& location() const { return location_; }

    // Gets the mean vector of the distribution. Equal to the location vector mu when v > 1.
    const std::vector<double>& mean() const {
        if (degrees_of_freedom_ <= 1.0)
            throw std::domain_error("The mean is undefined for degrees of freedom v <= 1.");
        return location_;
    }

    // Gets the median vector of the distribution. Equal to the location vector mu.
    const std::vector<double>& median() const { return location_; }

    // Gets the mode vector of the distribution. Equal to the location vector mu.
    const std::vector<double>& mode() const { return location_; }

    // Gets the marginal variance vector of the distribution: v/(v-2) * diag(Sigma).
    std::vector<double> variance() const {
        if (degrees_of_freedom_ <= 2.0)
            throw std::domain_error("The variance is undefined for degrees of freedom v <= 2.");
        double scale = degrees_of_freedom_ / (degrees_of_freedom_ - 2.0);
        std::vector<double> diag = scale_matrix_.diagonal();
        std::vector<double> v(static_cast<std::size_t>(dimension_));
        for (int i = 0; i < dimension_; ++i) v[static_cast<std::size_t>(i)] = scale * diag[static_cast<std::size_t>(i)];
        return v;
    }

    // Gets the marginal standard deviation vector of the distribution.
    std::vector<double> standard_deviation() const {
        std::vector<double> v = variance();
        for (double& x : v) x = std::sqrt(x);
        return v;
    }

    // Gets the scale matrix Sigma of the distribution. Not the covariance matrix -- the
    // covariance matrix is v/(v-2)*Sigma for v > 2.
    la::Matrix2D scale_matrix() const { return scale_matrix_.to_array(); }

    // Gets the covariance matrix of the distribution: v/(v-2) * Sigma.
    la::Matrix2D covariance() const {
        if (degrees_of_freedom_ <= 2.0)
            throw std::domain_error("The covariance is undefined for degrees of freedom v <= 2.");
        double scale = degrees_of_freedom_ / (degrees_of_freedom_ - 2.0);
        la::Matrix cov = scale_matrix_.clone();
        for (int i = 0; i < dimension_; ++i)
            for (int j = 0; j < dimension_; ++j) cov(i, j) *= scale;
        return cov.to_array();
    }

    // Element accessor (not in the C# source -- added for fixture/dispatch convenience,
    // mirroring MultivariateNormal's `covariance(i, j)`).
    double covariance(int i, int j) const {
        if (degrees_of_freedom_ <= 2.0)
            throw std::domain_error("The covariance is undefined for degrees of freedom v <= 2.");
        double scale = degrees_of_freedom_ / (degrees_of_freedom_ - 2.0);
        return scale * scale_matrix_(i, j);
    }

    // Determines if the scale matrix is positive definite.
    bool is_positive_definite() const { return cholesky_->is_positive_definite(); }

    // --- Parameter setting / validation ---

    // Set the distribution parameters.
    void set_parameters(double degrees_of_freedom, std::vector<double> location,
                        std::vector<std::vector<double>> scale_matrix) {
        // Validate parameters
        validate_parameters(degrees_of_freedom, location, scale_matrix, true);

        degrees_of_freedom_ = degrees_of_freedom;
        dimension_ = static_cast<int>(location.size());
        location_ = std::move(location);
        scale_matrix_ = la::Matrix(std::move(scale_matrix));
        cholesky_.emplace(scale_matrix_);

        // Precompute the log of the normalizing constant for the PDF:
        // ln C = LogGamma((v+p)/2) - LogGamma(v/2) - (p/2)*ln(v*pi) - (1/2)*ln|Sigma|
        double lndet = cholesky_->log_determinant();
        lnconstant_ = sf::log_gamma((degrees_of_freedom_ + dimension_) / 2.0) -
                     sf::log_gamma(degrees_of_freedom_ / 2.0) -
                     (dimension_ / 2.0) * std::log(degrees_of_freedom_ * kPi) - 0.5 * lndet;

        // (`_variance`/`_standardDeviation` lazy-cache fields omitted -- see file header note)
    }

    // Validate the distribution parameters. Returns nullopt if valid; if `throw_exception`
    // is false and invalid, returns the error message instead of throwing.
    std::optional<std::string> validate_parameters(double degrees_of_freedom,
                                                     const std::vector<double>& location,
                                                     const std::vector<std::vector<double>>& scale_matrix,
                                                     bool throw_exception) const {
        if (std::isnan(degrees_of_freedom) || std::isinf(degrees_of_freedom) || degrees_of_freedom <= 0.0) {
            if (throw_exception) throw std::out_of_range("Degrees of freedom must be greater than zero.");
            return "Degrees of freedom must be greater than zero.";
        }
        // location/scale_matrix null-checks omitted -- see file header divergence note.
        for (double loc : location) {
            if (std::isnan(loc) || std::isinf(loc)) {
                if (throw_exception) throw std::out_of_range("Location values must be finite.");
                return "Location values must be finite.";
            }
        }
        la::Matrix m(scale_matrix);
        if (!m.is_square()) {
            if (throw_exception) throw std::out_of_range("Scale matrix must be square.");
            return "Scale matrix must be square.";
        }
        if (m.number_of_rows() != static_cast<int>(location.size())) {
            if (throw_exception) throw std::out_of_range("Location vector length must match scale matrix dimension.");
            return "Location vector length must match scale matrix dimension.";
        }
        for (int i = 0; i < m.number_of_rows(); ++i) {
            for (int j = 0; j < m.number_of_columns(); ++j) {
                if (std::isnan(m(i, j)) || std::isinf(m(i, j))) {
                    if (throw_exception) throw std::out_of_range("Scale-matrix values must be finite.");
                    return "Scale-matrix values must be finite.";
                }
            }
        }
        // Cholesky try/catch quirk: UNLIKE MultivariateNormal::validate_parameters (which
        // lets a non-positive-definite construction propagate unconditionally), the C#
        // source wraps this probe construction in try/catch and converts ANY exception into
        // an ArgumentOutOfRangeException ("...not positive-definite"), honoring
        // `throwException` either way. This port's CholeskyDecomposition ctor throws
        // `std::invalid_argument` for a non-square matrix (unreachable here -- already
        // checked above, mirroring the C# ArgumentOutOfRangeException catch-and-rethrow
        // branch) and `std::runtime_error` for a non-positive-definite/NaN-diagonal one
        // (mirroring the C# generic-Exception branch). The `!chol.is_positive_definite()`
        // check below is dead code -- like MultivariateNormal's analogous check, a
        // construction that returns without throwing always has already set
        // `is_positive_definite_ = true` -- but is ported verbatim for structural fidelity.
        try {
            la::CholeskyDecomposition chol(m);
            if (!chol.is_positive_definite()) {
                if (throw_exception) throw std::out_of_range("Scale matrix is not positive-definite.");
                return "Scale matrix is not positive-definite.";
            }
        } catch (const std::invalid_argument&) {
            throw;
        } catch (const std::exception&) {
            if (throw_exception) throw std::out_of_range("Scale matrix is not positive-definite.");
            return "Scale matrix is not positive-definite.";
        }
        return std::nullopt;
    }

    // --- Distribution functions ---

    // The Probability Density Function (PDF) of the distribution evaluated at a point x.
    //
    // f(x) = C * [1 + (x-mu)'Sigma^-1(x-mu)/v]^(-(v+p)/2)
    // where C = Gamma((v+p)/2) / [Gamma(v/2) * (v*pi)^(p/2) * |Sigma|^(1/2)].
    double pdf(const std::vector<double>& x) const override { return std::exp(log_pdf(x)); }

    // Returns the natural log of the PDF evaluated at a point x, via the numerically stable
    // formulation log f(x) = ln C - ((v+p)/2) * ln(1 + Q/v), where Q is the squared
    // Mahalanobis distance and ln C is the precomputed log normalizing constant.
    double log_pdf(const std::vector<double>& x) const override {
        double Q = mahalanobis(x);
        double f = lnconstant_ - ((degrees_of_freedom_ + dimension_) / 2.0) * std::log(1.0 + Q / degrees_of_freedom_);
        if (std::isnan(f) || std::isinf(f)) return -kInf;
        return f;
    }

    // Gets the squared Mahalanobis distance between a sample point and this distribution:
    // (x-mu)'Sigma^-1(x-mu).
    double mahalanobis(const std::vector<double>& x) const {
        if (x.size() != static_cast<std::size_t>(dimension_))
            throw std::out_of_range("The vector must be the same dimension as the distribution.");
        std::vector<double> z(location_.size());
        for (std::size_t i = 0; i < x.size(); ++i) z[i] = x[i] - location_[i];
        la::Vector a = cholesky_->solve(la::Vector(z));
        double b = 0.0;
        for (std::size_t i = 0; i < z.size(); ++i) b += a[static_cast<int>(i)] * z[i];
        return b;
    }

    // The Cumulative Distribution Function (CDF) for the distribution evaluated at a point x.
    //
    // For dimension 1, this delegates to the univariate Student's t CDF. For dimensions >=
    // 2, the CDF is computed via stratified numerical integration over the mixing
    // representation: P(X <= x) = E_W[ Phi_p((x-mu)*sqrt(W/v); 0, Sigma) ] where W ~
    // chi^2(v), and Phi_p is the multivariate normal CDF. The expectation is computed by
    // evaluating the MVN CDF at K=200 stratified quantiles of the chi^2(v) distribution and
    // averaging (Genz & Bretz (2009), "Computation of Multivariate Normal and t
    // Probabilities").
    double cdf(const std::vector<double>& x) const override {
        if (x.size() != static_cast<std::size_t>(dimension_))
            throw std::out_of_range("The vector must be the same dimension as the distribution.");

        if (dimension_ == 1) {
            // Delegate to univariate Student's t CDF
            double sigma = std::sqrt(scale_matrix_(0, 0));
            StudentT univ_t(location_[0], sigma, degrees_of_freedom_);
            return univ_t.cdf(x[0]);
        }

        // For D >= 2, use stratified quantile integration over the chi^2(v) mixing variable.
        // The multivariate-t CDF can be written as:
        //   P(X <= x) = E_W[ Phi_MVN((x-mu)*sqrt(W/v); 0, Sigma) ]   where W ~ chi^2(v)
        //
        // We evaluate this by computing the MVN CDF at K equally-spaced quantiles of
        // chi^2(v) and averaging. This is deterministic and works for any v.

        const int K = 200;
        GammaDistribution gamma(2.0, degrees_of_freedom_ / 2.0);

        // Centered point for MVN CDF evaluation
        std::vector<double> z_vec(static_cast<std::size_t>(dimension_));
        for (int i = 0; i < dimension_; ++i)
            z_vec[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] - location_[static_cast<std::size_t>(i)];

        // Create MVN with zero mean and the scale matrix Sigma for CDF evaluation
        MultivariateNormal mvn(std::vector<double>(static_cast<std::size_t>(dimension_), 0.0), scale_matrix_.to_array());
        mvn.mvnuni() = mvnuni_;

        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            // Use midpoint of each stratum for stratified integration
            double p = (k + 0.5) / K;
            double w = gamma.inverse_cdf(p);
            double scale_factor = std::sqrt(w / degrees_of_freedom_);

            // Scale the centered vector: (x-mu) * sqrt(W/v)
            std::vector<double> scaled_z(static_cast<std::size_t>(dimension_));
            for (int i = 0; i < dimension_; ++i)
                scaled_z[static_cast<std::size_t>(i)] = z_vec[static_cast<std::size_t>(i)] * scale_factor;

            sum += mvn.cdf(scaled_z);
        }

        mvnuni_ = mvn.mvnuni();
        double result = sum / K;

        // Clamp to [0, 1]
        return std::max(0.0, std::min(1.0, result));
    }

    // Generate a matrix of random values from the multivariate Student's t-distribution,
    // using the representation X = mu + L*z / sqrt(W/v), where L is the Cholesky factor of
    // Sigma, z ~ N(0, I_p), and W ~ chi^2(v) (generated via the equivalent Gamma(v/2, 2)
    // distribution to support non-integer degrees of freedom). seed<=0 seeds from a clock
    // (no reproducible draw; see make_master()).
    std::vector<std::vector<double>> generate_random_values(int sample_size, int seed = -1) const {
        sampling::MersenneTwister rnd = make_master(seed);
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                  std::vector<double>(static_cast<std::size_t>(dimension_)));

        // Use Gamma(v/2, 2) to generate chi^2(v) variates, supporting non-integer v
        GammaDistribution gamma(2.0, degrees_of_freedom_ / 2.0);

        for (int i = 0; i < sample_size; ++i) {
            // z ~ N(0, I_p)
            std::vector<double> z(static_cast<std::size_t>(dimension_));
            for (int j = 0; j < dimension_; ++j) z[static_cast<std::size_t>(j)] = Normal::standard_z(rnd.next_double());

            // W ~ chi^2(v) via Gamma(v/2, 2)
            double w = gamma.inverse_cdf(rnd.next_double());

            // scale = sqrt(v / W)
            double scale = std::sqrt(degrees_of_freedom_ / w);

            // x = mu + L*z * scale
            la::Vector Lz = cholesky_->l() * la::Vector(z);
            for (int j = 0; j < dimension_; ++j)
                sample[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    location_[static_cast<std::size_t>(j)] + Lz[j] * scale;
        }
        return sample;
    }

    // Generate random values using Latin Hypercube Sampling (LHS) for improved space-filling
    // properties. The LHS uniforms are used for the normal component z ~ N(0, I_p), while a
    // separate independent random stream (seed+1) generates the chi^2(v) mixing variates.
    std::vector<std::vector<double>> latin_hypercube_random_values(int sample_size, int seed) const {
        la::Matrix2D r = sampling::LatinHypercube::random(sample_size, dimension_, seed);
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                  std::vector<double>(static_cast<std::size_t>(dimension_)));

        // Separate PRNG stream for chi^2 variates (independent of the LHS grid)
        sampling::MersenneTwister rnd_chi(static_cast<std::uint32_t>(seed + 1));
        GammaDistribution gamma(2.0, degrees_of_freedom_ / 2.0);

        for (int i = 0; i < sample_size; ++i) {
            // z ~ N(0, I_p) via LHS uniforms
            std::vector<double> z(static_cast<std::size_t>(dimension_));
            for (int j = 0; j < dimension_; ++j)
                z[static_cast<std::size_t>(j)] =
                    Normal::standard_z(r[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);

            // W ~ chi^2(v) via independent stream
            double w = gamma.inverse_cdf(rnd_chi.next_double());
            double scale = std::sqrt(degrees_of_freedom_ / w);

            // x = mu + L*z * scale
            la::Vector Lz = cholesky_->l() * la::Vector(z);
            for (int j = 0; j < dimension_; ++j)
                sample[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    location_[static_cast<std::size_t>(j)] + Lz[j] * scale;
        }
        return sample;
    }

    // Returns a 2D array of stratified random variates. The first dimension uses stratified
    // uniform quantiles (bin midpoints), the remaining dimensions use independent random
    // uniforms, and the chi^2(v) mixing variate is drawn from a separate random stream
    // (seed+1).
    std::vector<std::vector<double>> stratified_random_values(
        const std::vector<sampling::StratificationBin>& stratification_bins, int seed) const {
        int sample_size = static_cast<int>(stratification_bins.size());
        sampling::MersenneTwister rnd(static_cast<std::uint32_t>(seed));
        sampling::MersenneTwister rnd_chi(static_cast<std::uint32_t>(seed + 1));
        GammaDistribution gamma(2.0, degrees_of_freedom_ / 2.0);
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                  std::vector<double>(static_cast<std::size_t>(dimension_)));

        for (int i = 0; i < sample_size; ++i) {
            // z ~ N(0, I_p) with first dimension stratified
            std::vector<double> z(static_cast<std::size_t>(dimension_));
            for (int j = 0; j < dimension_; ++j) {
                if (j == 0) {
                    z[0] = Normal::standard_z(stratification_bins[static_cast<std::size_t>(i)].midpoint());
                } else {
                    z[static_cast<std::size_t>(j)] = Normal::standard_z(rnd.next_double());
                }
            }

            // W ~ chi^2(v) via independent stream
            double w = gamma.inverse_cdf(rnd_chi.next_double());
            double scale = std::sqrt(degrees_of_freedom_ / w);

            // x = mu + L*z * scale
            la::Vector Lz = cholesky_->l() * la::Vector(z);
            for (int j = 0; j < dimension_; ++j)
                sample[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    location_[static_cast<std::size_t>(j)] + Lz[j] * scale;
        }
        return sample;
    }

    // The inverse cumulative distribution function (InverseCDF).
    //
    // `probabilities` must have length Dimension + 1: the first `Dimension` entries
    // correspond to the correlated normal dimensions (z_j = Phi^-1(p_j)), and the last
    // entry is the probability for the chi^2(v) mixing variable W = F_chi2(v)^-1(p_last)
    // that controls the tail thickness. Uses the representation
    // X = mu + L*z * sqrt(v/W), where L is the Cholesky factor of the scale matrix Sigma.
    std::vector<double> inverse_cdf(const std::vector<double>& probabilities) const {
        if (probabilities.size() != static_cast<std::size_t>(dimension_ + 1))
            throw std::out_of_range(
                "The probabilities array must have length Dimension + 1 (Dimension + 1 for "
                "the chi-squared mixing variable).");

        // Convert first p probabilities to standard normal variates
        std::vector<double> z(static_cast<std::size_t>(dimension_));
        for (int j = 0; j < dimension_; ++j)
            z[static_cast<std::size_t>(j)] = Normal::standard_z(probabilities[static_cast<std::size_t>(j)]);

        // Convert last probability to chi^2(v) variate via Gamma(v/2, 2)
        GammaDistribution gamma(2.0, degrees_of_freedom_ / 2.0);
        double w = gamma.inverse_cdf(probabilities[static_cast<std::size_t>(dimension_)]);
        double scale = std::sqrt(degrees_of_freedom_ / w);

        // x = mu + L*z * scale
        la::Vector Lz = cholesky_->l() * la::Vector(z);
        std::vector<double> sample(static_cast<std::size_t>(dimension_));
        for (int j = 0; j < dimension_; ++j)
            sample[static_cast<std::size_t>(j)] = location_[static_cast<std::size_t>(j)] + Lz[j] * scale;

        return sample;
    }

    // Creates a deep copy of this distribution.
    std::unique_ptr<MultivariateDistribution> clone() const override {
        std::unique_ptr<MultivariateStudentT> mvt(new MultivariateStudentT());
        mvt->parameters_valid_ = parameters_valid_;
        mvt->dimension_ = dimension_;
        mvt->degrees_of_freedom_ = degrees_of_freedom_;
        mvt->location_ = location_;
        mvt->scale_matrix_ = scale_matrix_.clone();
        mvt->cholesky_.emplace(scale_matrix_.clone());
        mvt->lnconstant_ = lnconstant_;
        mvt->mvnuni_ = mvnuni_;
        return mvt;
    }

   private:
    // Private parameterless ctor for clone() (matches the C# `private MultivariateStudentT()`).
    MultivariateStudentT() = default;

    // Not shared with MultivariateNormal's identical helpers -- each ported class stays
    // self-contained (matches the small-duplication precedent set by the BVU/bivariate_cdf
    // Gauss-Legendre tables in multivariate_normal.hpp).
    static sampling::MersenneTwister make_clock_seeded() {
        auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return sampling::MersenneTwister(static_cast<std::uint32_t>(ticks));
    }

    static sampling::MersenneTwister make_master(int seed) {
        if (seed > 0) return sampling::MersenneTwister(static_cast<std::uint32_t>(seed));
        return make_clock_seeded();
    }

    // (`kInf` is inherited from MultivariateDistribution -- protected static constexpr,
    // usable unqualified here; not redeclared, matching the Dirichlet/Multinomial/
    // MultivariateNormal precedent.)

    // --- Fields (mirrors the C# field declaration order) ---

    bool parameters_valid_ = true;  // dead state -- see file header divergence note
    int dimension_ = 0;
    double degrees_of_freedom_ = 0.0;
    std::vector<double> location_;
    la::Matrix scale_matrix_ = la::Matrix(0, 0);
    std::optional<la::CholeskyDecomposition> cholesky_;
    double lnconstant_ = 0.0;
    mutable sampling::MersenneTwister mvnuni_ = sampling::MersenneTwister(12345U);
    // (`_variance`/`_standardDeviation` lazy-cache fields omitted -- see file header note)
};

}  // namespace corehydro::numerics::distributions
