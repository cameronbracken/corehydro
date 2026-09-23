// ported from: Numerics/Mathematics/Linear Algebra/EigenValueDecomposition.cs @ 7e8e8d1
//
// Computes all eigenvalues and eigenvectors of a real SYMMETRIC matrix using the classic
// Jacobi rotation method: A = V * D * V^T, where D contains the eigenvalues and the columns
// of V are the corresponding orthonormal eigenvectors. Robust and accurate for small to
// medium problems (Press et al., "Numerical Recipes", 3rd ed.).
//
// Ported with the GMM follow-up (Phase 6, Task B8) because
// `MatrixRegularization::regularize` -- deferred at the end of Phase 3 precisely because it
// needs this decomposition -- is on GeneralizedMethodOfMoments' covariance path.
//
// v2.2.0 retains the public input copy and rotates a separate working matrix.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"

namespace corehydro::numerics::math::linalg {

class EigenValueDecomposition {
   public:
    // Constructs a new symmetric eigen-decomposition of the symmetric input matrix A.
    explicit EigenValueDecomposition(const Matrix& A)
        : n_(A.number_of_rows()),
          a_(A.clone()),
          eigen_vectors_(Matrix::identity(A.number_of_rows())),
          eigen_values_(A.number_of_rows()) {
        if (A.number_of_rows() != A.number_of_columns())
            throw std::out_of_range("The matrix A must be square.");
        if (!A.is_symmetric())
            throw std::invalid_argument("The matrix A must be symmetric for this decomposition.");

        const int n = n_;
        Matrix work = A.clone();
        double element_max = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                element_max = std::max(element_max, std::fabs(work(i, j)));
        const double tol = 1e-12 * element_max;
        const int max_iter = 2000;

        for (int iter = 0; iter < max_iter; ++iter) {
            // Find largest off-diagonal element.
            double max = 0.0;
            int p = 0, q = 0;
            for (int i = 0; i < n - 1; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    double val = std::fabs(work(i, j));
                    if (val > max) {
                        max = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if (max <= tol) break;  // Converged

            // Current largest off-diagonal pair (p, q).
            double app = work(p, p);
            double aqq = work(q, q);
            double apq = work(p, q);

            // Robust angle computation: theta = 0.5 * atan2(2*apq, aqq - app).
            double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
            double c = std::cos(theta);
            double s = std::sin(theta);

            // Rotate the 2x2 pivot block exactly.
            double app_new = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            double aqq_new = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            work(p, p) = app_new;
            work(q, q) = aqq_new;
            work(p, q) = work(q, p) = 0.0;

            // Update the rest of the rows/cols (preserve symmetry).
            for (int k = 0; k < n; ++k) {
                if (k == p || k == q) continue;
                double aik = work(k, p);
                double akq = work(k, q);
                double akp_new = c * aik - s * akq;
                double akq_new = s * aik + c * akq;
                work(k, p) = work(p, k) = akp_new;
                work(k, q) = work(q, k) = akq_new;
            }

            // Accumulate eigenvectors: V = V * J(p, q).
            for (int k = 0; k < n; ++k) {
                double vip = eigen_vectors_(k, p);
                double viq = eigen_vectors_(k, q);
                eigen_vectors_(k, p) = c * vip - s * viq;
                eigen_vectors_(k, q) = s * vip + c * viq;
            }
        }

        // Extract eigenvalues from the diagonal of A.
        for (int i = 0; i < n; ++i) eigen_values_[i] = work(i, i);
    }

    // The retained input matrix A.
    const Matrix& a() const { return a_; }

    // The vector of eigenvalues (length n).
    const Vector& eigen_values() const { return eigen_values_; }

    // The matrix of eigenvectors (n x n). Each column corresponds to an eigenvalue.
    const Matrix& eigen_vectors() const { return eigen_vectors_; }

    // Returns the effective sample size based on Dutilleul's method (1993).
    double effective_sample_size() const {
        double spectral_scale = 0.0;
        for (int i = 0; i < eigen_values_.length(); ++i)
            spectral_scale = std::max(spectral_scale, std::fabs(eigen_values_[i]));
        if (spectral_scale == 0.0) return 0.0;
        double sum = 0;
        double sumsq = 0;
        for (int i = 0; i < eigen_values_.length(); ++i) {
            // Clip tiny negative eigenvalues that can appear from numerical error.
            double lambda = eigen_values_[i] / spectral_scale;
            if (lambda < 0.0 && std::fabs(lambda) <= 1e-10) lambda = 0.0;
            sum += lambda;
            sumsq += lambda * lambda;
        }
        if (sumsq == 0.0) return 0.0;
        double neff = (sum * sum) / sumsq;
        return neff;
    }

   private:
    int n_;  // Size of A
    Matrix a_;
    Matrix eigen_vectors_;
    Vector eigen_values_;
};

}  // namespace corehydro::numerics::math::linalg
