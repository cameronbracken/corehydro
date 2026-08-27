// ported from: Numerics/Mathematics/Linear Algebra/QRDecomposition.cs @ 2a0357a
//
// QR decomposition of a general real m-by-n matrix A = QR using Householder reflections
// (Numerical Recipes-style: accumulate Q^T in place while zeroing R below the diagonal one
// Householder reflector at a time). Member order mirrors the C# source: ctor, `q()`/`r()`
// accessors, `solve(Vector)`, `solve(Matrix)`.
//
// Naming note: the C# property is `RMatrix` (not `R`) because `R` collides with a type
// parameter elsewhere in the C# codebase; that collision does not exist in this namespace,
// so the accessor here is simply `r()`, matching the `q()`/`r()` pair the port's Interfaces
// call for. `RMatrix` -> `r()`, `Q` -> `q()`.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"

namespace corehydro::numerics::math::linalg {

class QRDecomposition {
   public:
    // Constructs a new QR decomposition of the real m-by-n matrix A.
    explicit QRDecomposition(const Matrix& A)
        : m_(A.number_of_rows()), n_(A.number_of_columns()), qt_(Matrix::identity(m_)), r_(A.clone()) {
        for (int k = 0; k < std::min(m_, n_); ++k) {
            double norm_x = 0.0;
            for (int i = k; i < m_; ++i) norm_x += r_(i, k) * r_(i, k);
            norm_x = std::sqrt(norm_x);
            if (norm_x == 0.0) continue;

            if (r_(k, k) >= 0.0) norm_x = -norm_x;

            std::vector<double> v(static_cast<std::size_t>(m_), 0.0);
            v[static_cast<std::size_t>(k)] = r_(k, k) - norm_x;
            for (int i = k + 1; i < m_; ++i) v[static_cast<std::size_t>(i)] = r_(i, k);

            double beta = 0.0;
            for (int i = k; i < m_; ++i) beta += v[static_cast<std::size_t>(i)] * v[static_cast<std::size_t>(i)];
            beta = 2.0 / beta;

            // Apply to R.
            for (int j = k; j < n_; ++j) {
                double s = 0.0;
                for (int i = k; i < m_; ++i) s += v[static_cast<std::size_t>(i)] * r_(i, j);
                s *= beta;
                for (int i = k; i < m_; ++i) r_(i, j) -= s * v[static_cast<std::size_t>(i)];
            }

            // Apply to Q (accumulated as Q^T).
            for (int j = 0; j < m_; ++j) {
                double s = 0.0;
                for (int i = k; i < m_; ++i) s += v[static_cast<std::size_t>(i)] * qt_(i, j);
                s *= beta;
                for (int i = k; i < m_; ++i) qt_(i, j) -= s * v[static_cast<std::size_t>(i)];
            }
        }
    }

    // Gets the orthogonal matrix Q (C# `Q => Matrix.Transpose(_Qt)`).
    Matrix q() const { return qt_.transpose(); }

    // Gets the upper triangular matrix R (C# `RMatrix`; see the file header naming note).
    const Matrix& r() const { return r_; }

    // Solves A*x = b using the QR decomposition.
    Vector solve(const Vector& b) const {
        if (b.length() != m_) throw std::invalid_argument("Vector b must match the number of rows in A.");
        int min = std::min(m_, n_);
        Vector qtb = qt_ * b;
        std::vector<double> x(static_cast<std::size_t>(n_), 0.0);
        for (int i = min - 1; i >= 0; --i) {
            x[static_cast<std::size_t>(i)] = qtb[i];
            for (int j = i + 1; j < min; ++j)
                x[static_cast<std::size_t>(i)] -= r_(i, j) * x[static_cast<std::size_t>(j)];
            x[static_cast<std::size_t>(i)] /= r_(i, i);
        }
        return Vector(std::move(x));
    }

    // Solves A*X = B (B a matrix) using the QR decomposition.
    Matrix solve(const Matrix& B) const {
        if (B.number_of_rows() != m_)
            throw std::invalid_argument("Matrix B must have the same number of rows as A.");
        int min = std::min(m_, n_);
        Matrix X(n_, B.number_of_columns());
        for (int j = 0; j < B.number_of_columns(); ++j) {
            std::vector<double> col(static_cast<std::size_t>(m_));
            for (int i = 0; i < m_; ++i) col[static_cast<std::size_t>(i)] = B(i, j);
            Vector qtb = qt_ * Vector(col);
            std::vector<double> x_col(static_cast<std::size_t>(n_), 0.0);
            for (int i = min - 1; i >= 0; --i) {
                x_col[static_cast<std::size_t>(i)] = qtb[i];
                for (int k = i + 1; k < min; ++k)
                    x_col[static_cast<std::size_t>(i)] -= r_(i, k) * x_col[static_cast<std::size_t>(k)];
                x_col[static_cast<std::size_t>(i)] /= r_(i, i);
            }
            for (int i = 0; i < min; ++i) X(i, j) = x_col[static_cast<std::size_t>(i)];
        }
        return X;
    }

   private:
    int m_;
    int n_;
    Matrix qt_;  // C# _Qt: accumulated Q^T
    Matrix r_;   // C# _R: upper triangular R
};

}  // namespace corehydro::numerics::math::linalg
