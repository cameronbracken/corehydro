// ported from: Numerics/Mathematics/Linear Algebra/CholeskyDecomposition.cs @ 7e8e8d1
//
// Cholesky decomposition A = L*L^T of a symmetric positive-definite matrix, ported
// verbatim (loop order and epsilon checks unchanged). Member order mirrors the C# source
// exactly: ctor, l(), a(), is_positive_definite(), solve(), backward(), forward(),
// inverse_a(), determinant(), log_determinant() -- note the C# source declares
// `Backward` before `Forward`, which differs from the task brief's prose order.
#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>

#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::linalg {

class CholeskyDecomposition {
   public:
    // Constructs new Cholesky Decomposition. Throws std::invalid_argument if A is not
    // square, and std::runtime_error if A is not positive-definite.
    explicit CholeskyDecomposition(const Matrix& A)
        : CholeskyDecomposition(A, default_relative_tolerance(A.number_of_rows())) {}

    CholeskyDecomposition(const Matrix& A, double relative_tolerance)
        : n_(A.number_of_rows()), a_(A.to_array()), l_(A.to_array()),
          relative_tolerance_(relative_tolerance) {
        int failed_row = -1;
        double failed_pivot = std::numeric_limits<double>::quiet_NaN();
        if (!try_factorize(a_, relative_tolerance, l_, failed_row, failed_pivot)) {
            throw std::runtime_error(
                "Cholesky Decomposition failed. The input matrix is not positive-definite.");
        }
        is_positive_definite_ = true;
    }

    static double default_relative_tolerance(int dimension) {
        return dimension > 0
            ? 2.0 * static_cast<double>(dimension) *
                  corehydro::numerics::kDoubleMachineEpsilon
            : 0.0;
    }

    static bool try_factorize(const Matrix& matrix, double relative_tolerance,
                              Matrix& lower, int& failed_row, double& failed_pivot) {
        if (!corehydro::numerics::is_finite(relative_tolerance) || relative_tolerance < 0.0 ||
            relative_tolerance >= 1.0) {
            throw std::out_of_range(
                "relative tolerance must be finite and within [0, 1)");
        }
        if (matrix.number_of_columns() != matrix.number_of_rows()) {
            throw std::invalid_argument("The matrix A must be square.");
        }
        const int dimension = matrix.number_of_rows();
        lower = Matrix(matrix.to_array());
        failed_row = -1;
        failed_pivot = std::numeric_limits<double>::quiet_NaN();
        for (int i = 0; i < dimension; ++i) {
            for (int j = i; j < dimension; ++j) {
                double sum = lower(i, j);
                for (int k = i - 1; k >= 0; --k) sum -= lower(i, k) * lower(j, k);
                if (i == j) {
                    const double diagonal = matrix(i, i);
                    const double threshold = diagonal > 0.0 && std::isfinite(diagonal)
                        ? relative_tolerance * diagonal
                        : 0.0;
                    if (std::isnan(sum) || sum <= 0.0 || sum <= threshold) {
                        failed_row = i;
                        failed_pivot = sum;
                        return false;
                    }
                    lower(i, i) = std::sqrt(sum);
                } else {
                    lower(j, i) = sum / lower(i, i);
                }
            }
        }
        for (int i = 0; i < dimension; ++i)
            for (int j = 0; j < i; ++j) lower(j, i) = 0.0;
        return true;
    }

    // Stores the decomposition (lower triangular L, with A = L*L^T).
    const Matrix& l() const { return l_; }

    // Stores the input matrix A that was decomposed.
    const Matrix& a() const { return a_; }

    // Determines whether the input matrix A is positive definite.
    bool is_positive_definite() const { return is_positive_definite_; }
    double relative_tolerance() const { return relative_tolerance_; }

    // Solves the set of n linear equations A*x=b using the stored decomposition A=L*L^T.
    Vector solve(const Vector& b) const {
        if (b.length() != n_)
            throw std::invalid_argument(
                "The vector b must have the same number of rows as the matrix A.");
        Vector x(n_);
        for (int i = 0; i < n_; ++i) {
            double sum = b[i];
            for (int k = i - 1; k >= 0; --k) sum -= l_(i, k) * x[k];
            x[i] = sum / l_(i, i);
        }
        for (int i = n_ - 1; i >= 0; --i) {
            double sum = x[i];
            for (int k = i + 1; k < n_; ++k) sum -= l_(k, i) * x[k];
            x[i] = sum / l_(i, i);
        }
        return x;
    }

    // Solves the L^T * x = y equation with backward substitution.
    Vector backward(const Vector& y) const {
        if (y.length() != n_)
            throw std::invalid_argument(
                "The vector y must have the same number of rows as the matrix A.");
        Vector x(n_);
        for (int i = n_ - 1; i >= 0; --i) {
            double sum = y[i];
            for (int j = n_ - 1; j > i; --j) sum -= x[j] * l_(j, i);
            x[i] = sum / l_(i, i);
        }
        return x;
    }

    // Solves the L * y = b equation using forward substitution.
    Vector forward(const Vector& b) const {
        if (b.length() != n_)
            throw std::invalid_argument(
                "The vector b must have the same number of rows as the matrix A.");
        Vector y(n_);
        for (int i = 0; i < n_; ++i) {
            double sum = b[i];
            for (int j = 0; j < i; ++j) sum -= l_(i, j) * y[j];
            y[i] = sum / l_(i, i);
        }
        return y;
    }

    // Matrix inverse A^-1 using the stored Cholesky decomposition.
    Matrix inverse_a() const {
        Matrix Ainv(n_);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = i == j ? 1.0 : 0.0;
                for (int k = i - 1; k >= j; --k) sum -= l_(i, k) * Ainv(j, k);
                Ainv(j, i) = sum / l_(i, i);
            }
        }
        for (int i = n_ - 1; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                double sum = i < j ? 0.0 : Ainv(j, i);
                for (int k = i + 1; k < n_; ++k) sum -= l_(k, i) * Ainv(j, k);
                Ainv(j, i) = sum / l_(i, i);
                Ainv(i, j) = Ainv(j, i);
            }
        }
        return Ainv;
    }

    // Using the stored Cholesky decomposition, returns the determinant of the matrix A.
    double determinant() const {
        double d = 1.0;
        for (int i = 0; i < n_; ++i) d *= l_(i, i);
        return std::pow(d, 2.0);
    }

    // Using the stored Cholesky decomposition, returns the logarithm of the determinant.
    double log_determinant() const {
        double sum = 0.0;
        for (int i = 0; i < n_; ++i) sum += std::log(l_(i, i));
        return 2.0 * sum;
    }

   private:
    int n_;  // Number of rows in A
    Matrix a_;
    Matrix l_;
    bool is_positive_definite_ = false;
    double relative_tolerance_ = 0.0;
};

}  // namespace corehydro::numerics::math::linalg
