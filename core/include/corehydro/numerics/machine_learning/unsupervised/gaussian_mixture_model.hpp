// ported from: Numerics/Machine Learning/Unsupervised/GaussianMixtureModel.cs @ 2a0357a
//
// Gaussian mixture model fitted by EM, initialized from k-means. Generalizes k-means clustering
// by carrying a full covariance per component rather than only a center.
//
// Reference (upstream's): https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model
//
// Six transcription notes, each on something a "cleanup" would silently change:
//
// 1. `oldLogLH` STARTS AT `double.MinValue`, not negative infinity, and the convergence test is
//    `Math.Abs((oldLogLH - newLogLH) / oldLogLH) < Tolerance`. With `lowest()` the first
//    iteration's ratio is essentially 1 and the test cannot fire; with `-infinity` the expression
//    would be `inf / -inf = NaN`, every comparison would be false, and the loop would run to
//    `MaxIterations` every time. Use `std::numeric_limits<double>::lowest()`.
// 2. `LogLikelihood` IS ASSIGNED ONLY INSIDE THE CONVERGENCE BRANCH. A run that exhausts
//    `MaxIterations` therefore leaves it at its default 0. Mirrored.
// 3. `MStep`'s call `MatrixRegularization.MakeSymmetricPositiveDefinite(Sigmas[k]);` DISCARDS THE
//    RETURN VALUE, and that method is pure (it returns a new Matrix and never mutates its
//    argument), so the call is a NO-OP -- the symmetrization and ridge the comment above it
//    promises never reach `Sigmas[k]`. The only thing actually keeping the covariance usable is
//    the diagonal floor a few lines earlier. Mirrored as an explicitly-discarded call so the
//    upstream diff keeps mapping; see docs/upstream-csharp-issues.md.
// 4. The E-step's argmax that sets `Labels[i]` runs over the UNNORMALIZED log values with `idx`
//    seeded at -1 and a strict `>`, so an all-NaN row would leave a label of -1, and ties go to
//    the lowest component index. The normalization then OVERWRITES `LikelihoodMatrix` in place
//    with the responsibilities the M-step reads.
// 5. Both the initialization and the M-step floor each covariance diagonal at `1e-6 * colVar`,
//    where `colVar` is the POPULATION variance of the whole column -- recomputed inside the `d`
//    loop every time, an O(K*D*N) redundancy. The redundancy is harmless but the accumulation
//    ORDER is what the oracle sees, so it is transcribed rather than hoisted.
// 6. `Sigmas[k]` starts as an all-zero `Matrix(Dimension)` and the initialization only fills it
//    when the k-means cluster has MORE THAN ONE member, so a singleton cluster reaches the first
//    E-step carrying only the floored diagonal.
//
// The `float[]` constructor overload is not ported, for the same reason as in k_means.hpp.
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "corehydro/numerics/machine_learning/unsupervised/k_means.hpp"
#include "corehydro/numerics/math/linalg/cholesky_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/matrix_regularization.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::machine_learning {

class GaussianMixtureModel {
   public:
    // Creates a new Gaussian mixture model over a single predictor column.
    GaussianMixtureModel(const std::vector<double>& x, int k)
        : GaussianMixtureModel(math::linalg::Matrix(x), k) {}

    // Creates a new Gaussian mixture model.
    GaussianMixtureModel(const math::linalg::Matrix& x, int k)
        : k_(k),
          x_(x),
          dimension_(x.number_of_columns()),
          means_(static_cast<std::size_t>(k),
                 std::vector<double>(static_cast<std::size_t>(x.number_of_columns()), 0.0)),
          sigmas_(static_cast<std::size_t>(k), math::linalg::Matrix(x.number_of_columns())),
          labels_(static_cast<std::size_t>(x.number_of_rows()), 0) {}

    // The number of clusters.
    int k() const { return k_; }

    // The matrix of predictor values.
    const math::linalg::Matrix& x() const { return x_; }

    // The dimensionality (number of features) of the data space.
    int dimension() const { return dimension_; }

    // The cluster means, k rows by `dimension` columns.
    const std::vector<std::vector<double>>& means() const { return means_; }

    // The cluster covariance matrices, one per component.
    const std::vector<math::linalg::Matrix>& sigmas() const { return sigmas_; }

    // The cluster label assigned to each data point.
    const std::vector<int>& labels() const { return labels_; }

    // The mixing weights.
    const std::vector<double>& weights() const { return weights_; }

    // The responsibility of each data point (row) for each cluster (column), after the last
    // E-step's normalization.
    const std::vector<std::vector<double>>& likelihood_matrix() const {
        return likelihood_matrix_;
    }

    // The total log-likelihood of the fit. See transcription note 2: this is 0 for a run that
    // exhausted `max_iterations()` without converging.
    double log_likelihood() const { return log_likelihood_; }

    // The maximum iterations in the clustering algorithm. Default = 1,000.
    int max_iterations() const { return max_iterations_; }
    void set_max_iterations(int value) { max_iterations_ = value; }

    // The relative tolerance for convergence. Default = 1E-8.
    double tolerance() const { return tolerance_; }
    void set_tolerance(double value) { tolerance_ = value; }

    // The number of iterations required to find the clusters.
    int iterations() const { return iterations_; }

    // Estimates the Gaussian mixture model. A `seed` of zero or less clock-seeds the generator.
    void train(int seed = -1, bool k_means_plus_plus = true) {
        // 1. Initialize clusters from k-means.
        KMeans k_means(x_, k_);
        k_means.train(seed, k_means_plus_plus);
        means_ = k_means.means();

        // Give equal weight to each cluster and initialize the covariance matrix from cluster
        // data variance. Using actual data variance (instead of a tiny constant like 1e-10)
        // prevents the first E-step from computing likelihoods from near-degenerate Gaussians,
        // which can cause numerical overflow or underflow.
        weights_.assign(static_cast<std::size_t>(k_), 0.0);
        likelihood_matrix_.assign(static_cast<std::size_t>(x_.number_of_rows()),
                                  std::vector<double>(static_cast<std::size_t>(k_), 0.0));
        sigmas_.assign(static_cast<std::size_t>(k_), math::linalg::Matrix(dimension_));
        for (int k = 0; k < k_; k++) {
            std::size_t ks = static_cast<std::size_t>(k);
            weights_[ks] = 1.0 / k_;
            sigmas_[ks] = math::linalg::Matrix(dimension_);

            // Compute within-cluster covariance from the k-means labels.
            int cluster_count = 0;
            for (int i = 0; i < x_.number_of_rows(); i++)
                if (k_means.labels()[static_cast<std::size_t>(i)] == k) cluster_count++;

            if (cluster_count > 1) {
                for (int d = 0; d < dimension_; d++) {
                    for (int j = 0; j < dimension_; j++) {
                        double sum = 0;
                        for (int i = 0; i < x_.number_of_rows(); i++) {
                            if (k_means.labels()[static_cast<std::size_t>(i)] == k)
                                sum += (x_(i, d) - means_[ks][static_cast<std::size_t>(d)]) *
                                       (x_(i, j) - means_[ks][static_cast<std::size_t>(j)]);
                        }
                        sigmas_[ks](d, j) = sum / cluster_count;
                    }
                }
            }

            // Ensure positive-definite: floor the diagonal at a fraction of the overall data
            // variance (see transcription note 5).
            for (int d = 0; d < dimension_; d++) {
                double col_var = 0;
                double col_mean = 0;
                for (int i = 0; i < x_.number_of_rows(); i++) col_mean += x_(i, d);
                col_mean /= x_.number_of_rows();
                for (int i = 0; i < x_.number_of_rows(); i++)
                    col_var += (x_(i, d) - col_mean) * (x_(i, d) - col_mean);
                col_var /= x_.number_of_rows();

                sigmas_[ks](d, d) = std::max(sigmas_[ks](d, d), 1e-6 * col_var);
            }
        }

        // 2. Optimize clusters.
        double old_log_lh = std::numeric_limits<double>::lowest();
        double new_log_lh = std::numeric_limits<double>::lowest();
        for (iterations_ = 1; iterations_ <= max_iterations_; iterations_++) {
            // Perform the expectation step.
            new_log_lh = e_step();

            // Check convergence (see transcription note 1).
            if (std::fabs((old_log_lh - new_log_lh) / old_log_lh) < tolerance_) {
                log_likelihood_ = new_log_lh;
                break;
            }

            // Perform the maximization step.
            m_step();

            // Update log-likelihood state.
            old_log_lh = new_log_lh;
        }
    }

   private:
    // The expectation step. Returns the log-likelihood.
    double e_step() {
        std::vector<double> log_det(static_cast<std::size_t>(k_), 0.0);

        // Outer loop for computing the likelihoods.
        for (int k = 0; k < k_; k++) {
            std::size_t ks = static_cast<std::size_t>(k);
            // Decompose the covariance in the outer loop.
            math::linalg::CholeskyDecomposition cholesky(sigmas_[ks]);
            log_det[ks] = cholesky.log_determinant();
            for (int i = 0; i < x_.number_of_rows(); i++) {
                // Inner loop for likelihoods.
                math::linalg::Vector u(dimension_);
                for (int d = 0; d < dimension_; d++)
                    u[d] = x_(i, d) - means_[ks][static_cast<std::size_t>(d)];
                // Solve L*v = u.
                math::linalg::Vector v = cholesky.forward(u);
                double sum = 0;
                for (int d = 0; d < dimension_; d++) sum += sqr(v[d]);
                likelihood_matrix_[static_cast<std::size_t>(i)][ks] =
                    -0.5 * (sum + log_det[ks]) + std::log(weights_[ks]);
            }
        }
        // At this point we have unnormalized logs of the likelihoods. Normalize with
        // log-sum-exp and compute the log-likelihood.
        double log_lh = 0;
        for (int i = 0; i < x_.number_of_rows(); i++) {
            std::size_t is = static_cast<std::size_t>(i);
            // Get the maximum likelihood and its index (see transcription note 4).
            double max = std::numeric_limits<double>::lowest();
            int idx = -1;
            for (int k = 0; k < k_; k++) {
                if (likelihood_matrix_[is][static_cast<std::size_t>(k)] > max) {
                    max = likelihood_matrix_[is][static_cast<std::size_t>(k)];
                    idx = k;
                }
            }
            // Update labels given the likelihoods.
            labels_[is] = idx;

            // log-sum-exp trick begins here.
            double sum = 0;
            for (int k = 0; k < k_; k++)
                sum += std::exp(likelihood_matrix_[is][static_cast<std::size_t>(k)] - max);
            double tmp = max + std::log(sum);
            for (int k = 0; k < k_; k++)
                likelihood_matrix_[is][static_cast<std::size_t>(k)] =
                    std::exp(likelihood_matrix_[is][static_cast<std::size_t>(k)] - tmp);
            log_lh += tmp;
        }
        return log_lh;
    }

    // The maximization step.
    void m_step() {
        for (int k = 0; k < k_; k++) {
            std::size_t ks = static_cast<std::size_t>(k);
            double wgt = 0.0;
            for (int i = 0; i < x_.number_of_rows(); i++)
                wgt += likelihood_matrix_[static_cast<std::size_t>(i)][ks];
            weights_[ks] = wgt / x_.number_of_rows();
            for (int d = 0; d < dimension_; d++) {
                // Compute centroids.
                double sum = 0;
                for (int i = 0; i < x_.number_of_rows(); i++)
                    sum += likelihood_matrix_[static_cast<std::size_t>(i)][ks] * x_(i, d);
                means_[ks][static_cast<std::size_t>(d)] = sum / wgt;
                // Compute covariance.
                for (int j = 0; j < dimension_; j++) {
                    sum = 0;
                    for (int i = 0; i < x_.number_of_rows(); i++) {
                        sum += likelihood_matrix_[static_cast<std::size_t>(i)][ks] *
                               (x_(i, d) - means_[ks][static_cast<std::size_t>(d)]) *
                               (x_(i, j) - means_[ks][static_cast<std::size_t>(j)]);
                    }
                    sigmas_[ks](d, j) = sum / wgt;
                }
            }

            // Floor the diagonal at a fraction of the overall data variance to prevent component
            // collapse. When a component captures very few points, its covariance can become
            // singular, causing Cholesky decomposition in the E-step to fail. This mirrors
            // sklearn's reg_covar parameter.
            for (int d = 0; d < dimension_; d++) {
                double col_var = 0;
                double col_mean = 0;
                for (int i = 0; i < x_.number_of_rows(); i++) col_mean += x_(i, d);
                col_mean /= x_.number_of_rows();
                for (int i = 0; i < x_.number_of_rows(); i++)
                    col_var += (x_(i, d) - col_mean) * (x_(i, d) - col_mean);
                col_var /= x_.number_of_rows();

                sigmas_[ks](d, d) = std::max(sigmas_[ks](d, d), 1e-6 * col_var);
            }

            // Ensure the full covariance matrix remains symmetric positive-definite.
            //
            // TRANSCRIPTION NOTE 3: upstream DISCARDS this return value, and the method is pure,
            // so this line has no effect on `Sigmas[k]`. Kept (with the discard made explicit)
            // so the upstream diff keeps mapping line-for-line. Assigning the result here would
            // be a silent behavior change against every oracle.
            (void)math::linalg::MatrixRegularization::make_symmetric_positive_definite(sigmas_[ks]);
        }
    }

    int k_;
    math::linalg::Matrix x_;
    int dimension_;
    std::vector<std::vector<double>> means_;
    std::vector<math::linalg::Matrix> sigmas_;
    std::vector<int> labels_;
    std::vector<double> weights_;
    std::vector<std::vector<double>> likelihood_matrix_;
    double log_likelihood_ = 0.0;
    int max_iterations_ = 1000;
    double tolerance_ = 1e-8;
    int iterations_ = 0;
};

}  // namespace corehydro::numerics::machine_learning
