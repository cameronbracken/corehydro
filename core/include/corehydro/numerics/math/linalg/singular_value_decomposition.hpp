// ported from: Numerics/Mathematics/Linear Algebra/SingularValueDecomposition.cs @ 7e8e8d1
//
// Solves sets of linear equations using Singular Value Decomposition (the Numerical
// Recipes 3rd-edition algorithm). Ported with Task B9 because LinearRegression.FitSVD
// (Numerics/Data/Regression/LinearRegression.cs) estimates its coefficients and their
// covariance through this class; the whole public surface is ported (ctor, threshold,
// U/V/W accessors, inverse_condition, rank/nullity/range/nullspace, both solve overloads,
// log_determinant/log_pseudo_determinant).
//
// Faithful C# details kept on purpose:
//   - `Threshold` is mutable state recomputed by rank()/nullity()/solve() when their
//     `threshold` argument is negative (the C# property setter side effect), so those
//     methods are non-const here.
//   - v2.2.0 makes the 30-iteration convergence guard reachable.
//   - C# Tools.Sign / Tools.DoubleMachineEpsilon map to corehydro::numerics::sign /
//     kDoubleMachineEpsilon.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::linalg {

class SingularValueDecomposition {
   public:
    // Constructs a Singular Value Decomposition of the input matrix A.
    explicit SingularValueDecomposition(const Matrix& A)
        : a_(A),
          n_(A.number_of_columns()),
          m_(A.number_of_rows()),
          u_(A.clone()),
          v_(n_, n_),
          w_(n_) {
        decompose();
        reorder();
        // Default threshold for nonzero singular values.
        threshold_ = 0.5 * std::sqrt(m_ + n_ + 1.0) * w_[0] * eps_;
    }

    // The default threshold value.
    double threshold() const { return threshold_; }

    // Stores the input matrix A that was SV decomposed.
    const Matrix& a() const { return a_; }

    // M x N column-orthogonal matrix U.
    const Matrix& u() const { return u_; }

    // Transpose of an N x N orthogonal matrix V.
    const Matrix& v() const { return v_; }

    // Diagonal matrix W, stored as a vector.
    const Vector& w() const { return w_; }

    // Returns the reciprocal of the condition number of A.
    double inverse_condition() const {
        return (w_[0] <= 0.0 || w_[n_ - 1] <= 0.0) ? 0.0 : w_[n_ - 1] / w_[0];
    }

    // Return the rank of A, after zeroing any singular values smaller than the threshold.
    // A negative threshold selects a default based on estimated roundoff.
    int rank(double threshold = -1) {
        int j, nr = 0;
        int n = a_.number_of_columns();
        int m = a_.number_of_rows();
        threshold_ = (threshold >= 0.0 ? threshold : 0.5 * std::sqrt(m + n + 1.0) * w_[0] * eps_);
        for (j = 0; j < n; j++)
            if (w_[j] > threshold_) nr++;
        return nr;
    }

    // Return the nullity of A, after zeroing any singular values smaller than the threshold.
    int nullity(double threshold = -1) {
        int j, nn = 0;
        threshold_ = (threshold >= 0.0 ? threshold : 0.5 * std::sqrt(m_ + n_ + 1.0) * w_[0] * eps_);
        for (j = 0; j < n_; j++)
            if (w_[j] <= threshold_) nn++;
        return nn;
    }

    // Gives an orthonormal basis for the range of A as the columns of a return matrix.
    Matrix range(double threshold = -1) {
        int i, j, nr = 0;
        Matrix rnge(m_, rank(threshold));
        for (j = 0; j < n_; j++) {
            if (w_[j] > threshold_) {
                for (i = 0; i < m_; i++) rnge(i, nr) = u_(i, j);
                nr++;
            }
        }
        return rnge;
    }

    // Gives an orthonormal basis for the nullspace of A as the columns of a return matrix.
    Matrix nullspace(double threshold = -1) {
        int j, jj, nn = 0;
        Matrix nullsp(n_, nullity(threshold));
        for (j = 0; j < n_; j++) {
            if (w_[j] <= threshold_) {
                for (jj = 0; jj < n_; jj++) nullsp(jj, nn) = v_(jj, j);
                nn++;
            }
        }
        return nullsp;
    }

    // Solves A*x=b for a vector x using the pseudoinverse of A as obtained by SVD. If
    // positive, the threshold is the value below which singular values are considered
    // zero; if negative, a default based on expected roundoff error is used.
    Vector solve(const Vector& B, double threshold = -1) {
        int i, j, jj;
        double s;
        Vector x(n_);
        Vector tmp(n_);
        if (B.length() != m_) {
            throw std::out_of_range(
                "The vector b must have the same number of rows as the matrix A.");
        }
        threshold_ = (threshold >= 0.0 ? threshold : 0.5 * std::sqrt(m_ + n_ + 1.0) * w_[0] * eps_);
        for (j = 0; j < n_; j++)  // Calculate U^T*B
        {
            s = 0.0;
            if (w_[j] > threshold_)  // Nonzero result only if W-j is nonzero.
            {
                for (i = 0; i < m_; i++) s += u_(i, j) * B[i];
                s /= w_[j];  // This s is then divided by W-j.
            }
            tmp[j] = s;
        }
        // Matrix multiply by V to get answer.
        for (j = 0; j < n_; j++) {
            s = 0.0;
            for (jj = 0; jj < n_; jj++) s += v_(j, jj) * tmp[jj];
            x[j] = s;
        }
        return x;
    }

    // Solve m sets of n equations A*X=B using the pseudoinverse of A.
    Matrix solve(const Matrix& B, double threshold = -1) {
        int i, j, p = B.number_of_columns();
        Matrix x(n_, p);
        Vector xx(n_);
        Vector bcol(m_);
        if (B.number_of_rows() != n_) {
            throw std::out_of_range(
                "The matrix B must have the same number of rows as columns in the matrix A.");
        }
        // Copy and solve each column in turn.
        for (j = 0; j < p; j++) {
            for (i = 0; i < m_; i++) bcol[i] = B(i, j);
            xx = solve(bcol, threshold);
            for (i = 0; i < n_; i++) x(i, j) = xx[i];
        }
        return x;
    }

    // Takes the log determinant of the matrix W.
    double log_determinant() const {
        double det = 0;
        for (int i = 0; i < w_.length(); i++) det += std::log(w_[i]);
        return det;
    }

    // Takes the log determinant of the pseudo matrix W (zero singular values skipped).
    double log_pseudo_determinant() const {
        double det = 0;
        for (int i = 0; i < w_.length(); i++)
            if (w_[i] != 0) det += std::log(w_[i]);
        return det;
    }

    double log_pseudo_determinant(double threshold) {
        threshold_ = threshold >= 0.0
            ? threshold
            : 0.5 * std::sqrt(m_ + n_ + 1.0) * w_[0] * eps_;
        double determinant = 0.0;
        for (int i = 0; i < w_.length(); ++i)
            if (w_[i] > threshold_) determinant += std::log(w_[i]);
        return determinant;
    }

   private:
    // Performs the singular value decomposition.
    void decompose() {
        bool flag;
        int i, its, j, jj, k, l = 0, nm = 0;
        double anorm, c, f, g, h, s, scale, x, y, z;
        const int m = m_, n = n_;
        Vector rv1(n);
        g = scale = anorm = 0.0;
        for (i = 0; i < n; i++) {
            l = i + 2;
            rv1[i] = scale * g;
            g = s = scale = 0.0;
            if (i < m) {
                for (k = i; k < m; k++) scale += std::fabs(u_(k, i));
                if (scale != 0.0) {
                    for (k = i; k < m; k++) {
                        u_(k, i) /= scale;
                        s += u_(k, i) * u_(k, i);
                    }
                    f = u_(i, i);
                    g = -numerics::sign(std::sqrt(s), f);
                    h = f * g - s;
                    u_(i, i) = f - g;
                    for (j = l - 1; j < n; j++) {
                        for (s = 0.0, k = i; k < m; k++) s += u_(k, i) * u_(k, j);
                        f = s / h;
                        for (k = i; k < m; k++) u_(k, j) += f * u_(k, i);
                    }
                    for (k = i; k < m; k++) u_(k, i) *= scale;
                }
            }
            w_[i] = scale * g;
            g = s = scale = 0.0;
            if (i + 1 <= m && i + 1 != n) {
                for (k = l - 1; k < n; k++) scale += std::fabs(u_(i, k));
                if (scale != 0.0) {
                    for (k = l - 1; k < n; k++) {
                        u_(i, k) /= scale;
                        s += u_(i, k) * u_(i, k);
                    }
                    f = u_(i, l - 1);
                    g = -numerics::sign(std::sqrt(s), f);
                    h = f * g - s;
                    u_(i, l - 1) = f - g;
                    for (k = l - 1; k < n; k++) rv1[k] = u_(i, k) / h;
                    for (j = l - 1; j < m; j++) {
                        for (s = 0.0, k = l - 1; k < n; k++) s += u_(j, k) * u_(i, k);
                        for (k = l - 1; k < n; k++) u_(j, k) += s * rv1[k];
                    }
                    for (k = l - 1; k < n; k++) u_(i, k) *= scale;
                }
            }
            anorm = std::max(anorm, std::fabs(w_[i]) + std::fabs(rv1[i]));
        }
        for (i = n - 1; i >= 0; i--) {
            if (i < n - 1) {
                if (g != 0.0) {
                    for (j = l; j < n; j++) v_(j, i) = (u_(i, j) / u_(i, l)) / g;
                    for (j = l; j < n; j++) {
                        for (s = 0.0, k = l; k < n; k++) s += u_(i, k) * v_(k, j);
                        for (k = l; k < n; k++) v_(k, j) += s * v_(k, i);
                    }
                }
                for (j = l; j < n; j++) v_(i, j) = v_(j, i) = 0.0;
            }
            v_(i, i) = 1.0;
            g = rv1[i];
            l = i;
        }
        for (i = std::min(m, n) - 1; i >= 0; i--) {
            l = i + 1;
            g = w_[i];
            for (j = l; j < n; j++) u_(i, j) = 0.0;
            if (g != 0.0) {
                g = 1.0 / g;
                for (j = l; j < n; j++) {
                    for (s = 0.0, k = l; k < m; k++) s += u_(k, i) * u_(k, j);
                    f = (s / u_(i, i)) * g;
                    for (k = i; k < m; k++) u_(k, j) += f * u_(k, i);
                }
                for (j = i; j < m; j++) u_(j, i) *= g;
            } else
                for (j = i; j < m; j++) u_(j, i) = 0.0;
            ++u_(i, i);
        }
        for (k = n - 1; k >= 0; k--) {
            for (its = 0; its < 30; its++) {
                flag = true;
                for (l = k; l >= 0; l--) {
                    nm = l - 1;
                    if (l == 0 || std::fabs(rv1[l]) <= eps_ * anorm) {
                        flag = false;
                        break;
                    }
                    if (std::fabs(w_[nm]) <= eps_ * anorm) break;
                }
                if (flag) {
                    c = 0.0;
                    s = 1.0;
                    for (i = l; i < k + 1; i++) {
                        f = s * rv1[i];
                        rv1[i] = c * rv1[i];
                        if (std::fabs(f) <= eps_ * anorm) break;
                        g = w_[i];
                        h = pythag(f, g);
                        w_[i] = h;
                        h = 1.0 / h;
                        c = g * h;
                        s = -f * h;
                        for (j = 0; j < m; j++) {
                            y = u_(j, nm);
                            z = u_(j, i);
                            u_(j, nm) = y * c + z * s;
                            u_(j, i) = z * c - y * s;
                        }
                    }
                }
                z = w_[k];
                if (l == k) {
                    if (z < 0.0) {
                        w_[k] = -z;
                        for (j = 0; j < n; j++) v_(j, k) = -v_(j, k);
                    }
                    break;
                }
                if (its == 29)
                    throw std::invalid_argument("There was no convergence in 30 iterations");
                x = w_[l];
                nm = k - 1;
                y = w_[nm];
                g = rv1[nm];
                h = rv1[k];
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                g = pythag(f, 1.0);
                f = ((x - z) * (x + z) + h * ((y / (f + numerics::sign(g, f))) - h)) / x;
                c = s = 1.0;
                for (j = l; j <= nm; j++) {
                    i = j + 1;
                    g = rv1[i];
                    y = w_[i];
                    h = s * g;
                    g = c * g;
                    z = pythag(f, h);
                    rv1[j] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = g * c - x * s;
                    h = y * s;
                    y *= c;
                    for (jj = 0; jj < n; jj++) {
                        x = v_(jj, j);
                        z = v_(jj, i);
                        v_(jj, j) = x * c + z * s;
                        v_(jj, i) = z * c - x * s;
                    }
                    z = pythag(f, h);
                    w_[j] = z;
                    if (z != 0.0) {
                        z = 1.0 / z;
                        c = f * z;
                        s = h * z;
                    }
                    f = c * g + s * y;
                    x = c * y - s * g;
                    for (jj = 0; jj < m; jj++) {
                        y = u_(jj, j);
                        z = u_(jj, i);
                        u_(jj, j) = y * c + z * s;
                        u_(jj, i) = z * c - y * s;
                    }
                }
                rv1[l] = 0.0;
                rv1[k] = f;
                w_[k] = x;
            }
        }
    }

    // Auxiliary function to reorder.
    void reorder() {
        const int m = m_, n = n_;
        int i, j, k, s, inc = 1;
        double sw;
        Vector su(m);
        Vector sv(n);
        do {
            inc *= 3;
            inc++;
        } while (inc <= n);
        do {
            inc /= 3;
            for (i = inc; i < n; i++) {
                sw = w_[i];
                for (k = 0; k < m; k++) su[k] = u_(k, i);
                for (k = 0; k < n; k++) sv[k] = v_(k, i);
                j = i;
                while (w_[j - inc] < sw) {
                    w_[j] = w_[j - inc];
                    for (k = 0; k < m; k++) u_(k, j) = u_(k, j - inc);
                    for (k = 0; k < n; k++) v_(k, j) = v_(k, j - inc);
                    j -= inc;
                    if (j < inc) break;
                }
                w_[j] = sw;
                for (k = 0; k < m; k++) u_(k, j) = su[k];
                for (k = 0; k < n; k++) v_(k, j) = sv[k];
            }
        } while (inc > 1);
        for (k = 0; k < n; k++) {
            s = 0;
            for (i = 0; i < m; i++)
                if (u_(i, k) < 0.0) s++;
            for (j = 0; j < n; j++)
                if (v_(j, k) < 0.0) s++;
            if (s > (m + n) / 2) {
                for (i = 0; i < m; i++) u_(i, k) = -u_(i, k);
                for (j = 0; j < n; j++) v_(j, k) = -v_(j, k);
            }
        }
    }

    // Computes (a^2 + b^2)^1/2 without destructive underflow or overflow.
    static double pythag(double a, double b) {
        double absa = std::fabs(a), absb = std::fabs(b);
        return (absa > absb
                    ? absa * std::sqrt(1.0 + std::pow(absb / absa, 2))
                    : (absb == 0.0 ? 0.0 : absb * std::sqrt(1.0 + std::pow(absa / absb, 2))));
    }

    static constexpr double eps_ = numerics::kDoubleMachineEpsilon;
    Matrix a_;
    int n_;  // Number of columns in A
    int m_;  // Number of rows in A
    Matrix u_;
    Matrix v_;
    Vector w_;
    double threshold_ = 0.0;
};

}  // namespace corehydro::numerics::math::linalg
