// ported from: Numerics/Mathematics/Linear Algebra/MatrixRegularization.cs @ 7e8e8d1
//
// Phase 1 ported ONLY `MakeSymmetricPositiveDefinite` -- the sole member the then-in-scope
// BestFit Estimation layer (MaximumLikelihood covariance/sandwich estimator,
// MaximumAPosteriori covariance) calls. Its C# body: symmetrize S = 0.5*(M + Mtranspose);
// compute a trace-scaled base ridge (1e-10 * trace(S)/p, or 1e-10 if trace <= 0); try up
// to 8 increasing ridges (baseRidge * 10^k for k = 0..7) added to the diagonal of a clone
// of S, returning the first that Cholesky-factorizes; if all 8 fail, add a last-resort
// ridge (1e-4 * trace(S)/p, or 1e-4 if trace <= 0) and return unconditionally. The
// private `Symmetrize` helper is inlined there via Matrix's own `operator+`/`transpose`/
// scalar `operator*`, matching the C# source (which itself inlines `0.5 * (M +
// M.Transpose())` rather than calling its own `Symmetrize` private method for this
// particular member). C#'s bare `catch { }` around `new CholeskyDecomposition(T)` is
// mirrored as `catch (const std::exception&)`, the faithful analogue per the task brief
// (CholeskyDecomposition's ctor only ever throws std::invalid_argument or
// std::runtime_error, both std::exception, so this is not narrower than the C#).
//
// Phase 6 (Task B8, the GMM follow-up the Phase 3 sever note pointed at) adds the
// previously deferred `Regularize` (Matrix overload) plus its private `Symmetrize` and
// `MedianFromVector` helpers, on top of the newly ported `EigenValueDecomposition`
// (symmetric Jacobi EVD, eigenvalue_decomposition.hpp). `GeneralizedMethodOfMoments::
// GetCovariance` regularizes S before inversion through this member. Still omitted: the
// C# `Regularize(double[,] ...)` overload -- a raw-array convenience wrapper around the
// Matrix overload with no caller in this port (call sites use `Matrix` directly).
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <exception>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/linalg/cholesky_decomposition.hpp"
#include "corehydro/numerics/math/linalg/eigenvalue_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::linalg {

class MatrixRegularization {
   public:
    // Eigen-regularizes a symmetric matrix Vb and returns a PSD matrix suitable for
    // Cholesky (C# `Regularize(Matrix, eps, capMult)`): Vb is (approximately) symmetrized,
    // eigen-decomposed Vb = Q diag(lambda) Q^T, eigenvalues are floored at
    // eps * (trace(Vb)/p) (falling back to eps if trace <= 0) and capped at
    // capMult * median(lambda), and the matrix is reconstructed as Q diag(lambda_reg) Q^T.
    static Matrix regularize(const Matrix& vb, double eps = 1e-6, double cap_mult = 50.0) {
        if (!vb.is_square()) throw std::invalid_argument("Vb must be square.");
        int p = vb.number_of_rows();

        // Ensure exact symmetry: A := (A + A^T)/2 (nice to have before Jacobi).
        Matrix vb_sym = symmetrize(vb);

        // Eigen-decomposition (symmetric): Vb = Q diag(lambda) Q^T.
        EigenValueDecomposition eig(vb_sym);
        const Matrix& q = eig.eigen_vectors();
        const Vector& w = eig.eigen_values();  // length p

        // Compute trace and robust floor.
        double trace = 0.0;
        for (int i = 0; i < p; ++i) trace += w[i];
        double floor = eps * (trace > 0.0 ? trace / p : 1.0);

        // Median of eigenvalues (copy -> sort).
        double median = median_from_vector(w);
        double cap = cap_mult * median;

        // Floor and cap eigenvalues.
        Matrix d(p);
        for (int i = 0; i < p; ++i) {
            double li = w[i];
            if (li < floor) li = floor;
            if (li > cap) li = cap;
            d(i, i) = li;
        }

        // Recompose: Q * D * Q^T (C# `Matrix.Transpose(Q)`, the static alias of the
        // instance `transpose()` -- see matrix.hpp's header).
        return q * d * q.transpose();
    }

    // Makes the matrix symmetric and positive definite (C# `MakeSymmetricPositiveDefinite`).
    static Matrix make_symmetric_positive_definite(const Matrix& m) {
        if (!m.is_square()) throw std::invalid_argument("The matrix must be square.");
        for (int i = 0; i < m.number_of_rows(); ++i)
            for (int j = 0; j < m.number_of_columns(); ++j)
                if (!corehydro::numerics::is_finite(m(i, j)))
                    throw std::invalid_argument("The matrix must contain only finite entries.");
        Matrix s = (m + m.transpose()) * 0.5;
        for (int i = 0; i < s.number_of_rows(); ++i)
            for (int j = 0; j < s.number_of_columns(); ++j)
                if (!corehydro::numerics::is_finite(s(i, j)))
                    throw std::runtime_error(
                        "Matrix symmetrization produced a non-finite entry.");

        Matrix lower(s.number_of_rows());
        int failed_row = -1;
        double failed_pivot = std::numeric_limits<double>::quiet_NaN();
        const double tolerance =
            CholeskyDecomposition::default_relative_tolerance(s.number_of_rows());
        if (CholeskyDecomposition::try_factorize(
                s, tolerance, lower, failed_row, failed_pivot)) return s;

        double tr = 0.0;
        for (int i = 0; i < s.number_of_rows(); ++i) tr += s(i, i);
        double base_ridge = (tr > 0 ? 1e-10 * tr / s.number_of_rows() : 1e-10);
        if (!corehydro::numerics::is_finite(tr) ||
            !corehydro::numerics::is_finite(base_ridge) || base_ridge <= 0.0) {
            throw std::runtime_error(
                "The trace-scaled matrix ridge is not a positive finite number.");
        }

        double ridge = base_ridge;
        for (int k = 0; corehydro::numerics::is_finite(ridge); ++k) {
            Matrix t = s.clone();
            for (int i = 0; i < t.number_of_rows(); ++i) {
                t(i, i) += ridge;
                if (!corehydro::numerics::is_finite(t(i, i))) {
                    throw std::runtime_error(
                        "Matrix ridge escalation overflowed a diagonal before positive "
                        "definiteness was achieved.");
                }
            }
            if (CholeskyDecomposition::try_factorize(
                    t, tolerance, lower, failed_row, failed_pivot)) return t;
            ridge = k < 7
                ? base_ridge * std::pow(10.0, static_cast<double>(k + 1))
                : ridge * 10.0;
        }
        throw std::runtime_error(
            "Matrix ridge escalation exhausted finite candidates before positive "
            "definiteness was achieved.");
    }

   private:
    // Symmetrizes the matrix (C# private `Symmetrize`).
    static Matrix symmetrize(const Matrix& a) {
        int n = a.number_of_rows(), m = a.number_of_columns();
        Matrix s(n, m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                s(i, j) = 0.5 * (a(i, j) + a(j, i));
            }
        }
        return s;
    }

    // Returns the median value from the vector (C# private `MedianFromVector`).
    static double median_from_vector(const Vector& v) {
        int n = v.length();
        std::vector<double> arr(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) arr[static_cast<std::size_t>(i)] = v[i];
        std::sort(arr.begin(), arr.end());
        if ((n & 1) == 1) return arr[static_cast<std::size_t>(n / 2)];
        return 0.5 * (arr[static_cast<std::size_t>(n / 2 - 1)] + arr[static_cast<std::size_t>(n / 2)]);
    }
};

}  // namespace corehydro::numerics::math::linalg
