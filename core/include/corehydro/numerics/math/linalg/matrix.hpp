// ported from: Numerics/Mathematics/Linear Algebra/Support/Matrix.cs @ 2a0357a
//
// Phase 0 shipped a minimal dense square-matrix inverse (Gauss-Jordan with partial
// pivoting) as free-standing `Matrix2D`/`inverse()`; GEV still depends on those exactly
// as they are, so they are kept untouched below. Phase 2 adds the real `Matrix` class
// alongside it for CholeskyDecomposition and the multivariate distributions.
//
// `Matrix` ports the members Phase 2 actually uses: the `(rows, cols)` / `(n)` / 2D-data
// constructors, `identity`, element access, `number_of_rows`/`number_of_columns`,
// `is_square`/`is_symmetric`, instance and static `diagonal`, elementwise `sqrt`,
// `multiply` (+ `operator*` for Matrix*Matrix and Matrix*Vector), `transpose`, and
// `clone`. A `(rows, cols, flat)` constructor is also added -- not present in the C#
// source -- to match this port's fixture convention of passing matrices as flattened
// row-major `args` (see `fixtures/special_functions/cholesky.json`). The square ctor
// `Matrix(int size)` delegates to the `(rows, cols)` ctor, which zero-value-initializes
// its backing `std::vector<std::vector<double>>` -- confirmed this matches C#'s `new
// double[size, size]` default-zero semantics, which RWMH's proposal-sigma init relies on
// (`new Matrix(2)` as an all-zeros starting matrix).
//
// Phase 3 (this port) extends the class with the members RunningCovarianceMatrix.Push
// (Numerics/Data/Statistics/RunningCovarianceMatrix.cs) needs beyond Phase 2's set:
// `add`/`subtract` (+ `operator+`/`operator-`) and scalar `operator*` on both sides (C#
// has both `Matrix*double` and `double*Matrix`, unlike Vector). The task brief's own
// EXTEND note for this file mentioned only the scalar operator; `add`/`subtract` are
// pulled in too because `RunningCovarianceMatrix.Push` needs `Mean +=`/`x - Mean` and the
// C# source governs over the brief when the two disagree on scope (see the task report).
// `Matrix::inverse()` is also added, backed by the newly-ported `LUDecomposition`
// (`lu_decomposition.hpp`) -- C# parity for `Matrix.Inverse()`/`operator!`, needed by
// `RunningCovarianceMatrix.SampleCorrelation`/`PopulationCorrelation` (`!D` in C#) and by
// the MAP init path. `Matrix::determinant()` (`Matrix.Determinant()`) is added alongside
// it for the same LU-backed parity, even though no ported caller needs it yet, because it
// is a two-line sibling of `inverse()` sharing the same `LUDecomposition` instance.
//
// Omitted (UI/serialization or not yet needed; add if a later target needs them):
// `Header`, `Array` (raw-array reference property; `to_array()`/`ToArray()`, a copy, is
// provided instead), `ToXElement`/XElement ctor, the single-column-array/`List<double[]>`
// ctors (RunningCovarianceMatrix.Push uses the existing `(rows, cols, flat)` ctor
// instead, passing a length-N vector as an Nx1 flattened matrix -- an equivalent
// construction), `Row`/`Column`, `UpperTriangle`/`LowerTriangle`/`Trace`, `ColumnMeans`,
// `Apply`/`Sqr`/`Log`/`Exp`, and `Sum`. B10 adds `outer` (C# Matrix.Outer, line 561) and
// scalar `divide`/`operator/` (C# Matrix.Divide, line 638 / operator/, line 742):
// Bulletin17CDistribution::MomentConditions forms S = E[gg']/n - Outer(mean, mean). Also
// omitted: `operator!` (inverse alias) and `operator~` (transpose alias) -- both are just
// operator spellings for the already-ported `inverse()`/`transpose()` methods, left out
// because C++ `operator!`/`operator~` on a non-boolean/non-bitmask type would be
// unidiomatic and no caller needs the operator form; call sites use `.inverse()`/
// `.transpose()` directly (e.g. `!D` in C# becomes `D.inverse()` here). The static
// `Matrix.Transpose(A)` overload is likewise omitted -- callers use the already-ported
// instance `transpose()`, which computes the identical result.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/linalg/vector.hpp"

namespace corehydro::numerics::math::linalg {

using Matrix2D = std::vector<std::vector<double>>;

inline Matrix2D inverse(const Matrix2D& a) {
    const int n = static_cast<int>(a.size());
    // Augment [a | I]
    std::vector<std::vector<double>> m(n, std::vector<double>(2 * n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) m[i][j] = a[i][j];
        m[i][n + i] = 1.0;
    }
    for (int col = 0; col < n; ++col) {
        // Partial pivot
        int pivot = col;
        double best = std::fabs(m[col][col]);
        for (int r = col + 1; r < n; ++r) {
            if (std::fabs(m[r][col]) > best) {
                best = std::fabs(m[r][col]);
                pivot = r;
            }
        }
        if (best == 0.0) throw std::runtime_error("matrix is singular; cannot invert");
        if (pivot != col) std::swap(m[pivot], m[col]);

        double diag = m[col][col];
        for (int j = 0; j < 2 * n; ++j) m[col][j] /= diag;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double factor = m[r][col];
            if (factor == 0.0) continue;
            for (int j = 0; j < 2 * n; ++j) m[r][j] -= factor * m[col][j];
        }
    }
    Matrix2D inv(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) inv[i][j] = m[i][n + j];
    return inv;
}

class Matrix {
   public:
    // Construct a new matrix with specified number of rows and columns, zero-initialized.
    Matrix(int number_of_rows, int number_of_columns)
        : data_(static_cast<std::size_t>(number_of_rows),
                std::vector<double>(static_cast<std::size_t>(number_of_columns), 0.0)) {}

    // Constructs a new square matrix.
    explicit Matrix(int size) : Matrix(size, size) {}

    // Construct a new matrix based on an initial 2D array.
    explicit Matrix(Matrix2D initial_array) : data_(std::move(initial_array)) {}

    // Construct a new matrix from flattened row-major data (see file header comment --
    // this ctor is not in the C# source; it exists for this port's fixture convention).
    Matrix(int number_of_rows, int number_of_columns, const std::vector<double>& flat)
        : data_(static_cast<std::size_t>(number_of_rows),
                std::vector<double>(static_cast<std::size_t>(number_of_columns), 0.0)) {
        if (static_cast<int>(flat.size()) != number_of_rows * number_of_columns)
            throw std::invalid_argument("flattened data length must equal rows*columns");
        for (int i = 0; i < number_of_rows; ++i)
            for (int j = 0; j < number_of_columns; ++j)
                data_[i][j] = flat[static_cast<std::size_t>(i * number_of_columns + j)];
    }

    // Construct an n-by-1 matrix from a single column of values (C# `Matrix(double[] x)`,
    // Matrix.cs -- the 1-D-array ctor every Machine Learning class offers as its single-feature
    // overload). P5 Task 1.
    explicit Matrix(const std::vector<double>& column_values)
        : data_(column_values.size(), std::vector<double>(1, 0.0)) {
        for (std::size_t i = 0; i < column_values.size(); ++i) data_[i][0] = column_values[i];
    }

    // Builds a matrix from a list of COLUMNS (C# `Matrix(List<double[]> x)`, the ctor the
    // Machine Learning test files use to assemble a feature matrix from named feature arrays).
    //
    // This is a named factory rather than a constructor because C# distinguishes
    // `Matrix(double[,])` (rows) from `Matrix(List<double[]>)` (columns) by TYPE, and both map
    // to `std::vector<std::vector<double>>` in C++ -- an overload pair here would be ambiguous
    // and, worse, would silently transpose whichever call bound to the wrong one. The
    // row-oriented form is the existing `explicit Matrix(Matrix2D)`. P5 Task 1.
    static Matrix from_columns(const std::vector<std::vector<double>>& columns) {
        if (columns.empty())
            throw std::invalid_argument("Input must have at least one column.");
        std::size_t rows = columns[0].size();
        for (const std::vector<double>& c : columns)
            if (c.size() != rows)
                throw std::invalid_argument("All columns must have the same number of rows.");
        Matrix m(static_cast<int>(rows), static_cast<int>(columns.size()));
        for (std::size_t j = 0; j < columns.size(); ++j)
            for (std::size_t i = 0; i < rows; ++i) m.data_[i][j] = columns[j][i];
        return m;
    }

    // Gets the number of rows.
    int number_of_rows() const { return static_cast<int>(data_.size()); }

    // Gets the number of columns.
    int number_of_columns() const { return data_.empty() ? 0 : static_cast<int>(data_[0].size()); }

    // Get/set the element at the specific row and column index.
    double operator()(int row_index, int column_index) const { return data_[row_index][column_index]; }
    double& operator()(int row_index, int column_index) { return data_[row_index][column_index]; }

    // Returns row `i` as a copy (C# `Matrix.Row(int)`). P5 Task 1 -- every Machine Learning
    // class walks its predictor matrix a row at a time.
    std::vector<double> row(int row_index) const {
        if (row_index < 0 || row_index >= number_of_rows())
            throw std::out_of_range("Matrix::row: row index out of range");
        return data_[static_cast<std::size_t>(row_index)];
    }

    // Returns column `j` as a copy (C# `Matrix.Column(int)`). P5 Task 1.
    std::vector<double> column(int column_index) const {
        if (column_index < 0 || column_index >= number_of_columns())
            throw std::out_of_range("Matrix::column: column index out of range");
        std::vector<double> c(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i)
            c[i] = data_[i][static_cast<std::size_t>(column_index)];
        return c;
    }

    // Evaluates whether this matrix is symmetric.
    bool is_symmetric() const {
        if (!is_square()) return false;
        const double epsilon = 1e-12;
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = i + 1; j < number_of_columns(); ++j)
                if (std::fabs(data_[i][j] - data_[j][i]) > epsilon) return false;
        return true;
    }

    // Determines whether this matrix is square.
    bool is_square() const { return number_of_rows() == number_of_columns(); }

    // Returns a copy of this matrix.
    Matrix clone() const { return Matrix(data_); }

    // Returns the matrix as a 2D array (copy).
    Matrix2D to_array() const { return data_; }

    // Returns the elements of the diagonal. For non-square matrices, returns
    // min(rows, columns) elements where i == j.
    std::vector<double> diagonal() const {
        int n = std::min(number_of_rows(), number_of_columns());
        std::vector<double> result(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) result[i] = data_[i][i];
        return result;
    }

    // Returns the transpose of the matrix.
    Matrix transpose() const {
        Matrix t(number_of_columns(), number_of_rows());
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < number_of_columns(); ++j) t(j, i) = data_[i][j];
        return t;
    }

    // Returns the matrix determinant, via LUDecomposition (C# `Matrix.Determinant()`).
    // Declared here, defined out-of-line in lu_decomposition.hpp once LUDecomposition is
    // complete (LUDecomposition's own constructor takes a Matrix, so the two headers are
    // mutually dependent; including lu_decomposition.hpp is required to link a call to
    // this method).
    double determinant() const;

    // Returns the matrix inverse A^-1, via LUDecomposition (C# `Matrix.Inverse()` /
    // `operator!`). See the `determinant()` comment above for why this is declared here
    // but defined in lu_decomposition.hpp.
    Matrix inverse() const;

    // Returns the diagonal matrix from a square matrix A (off-diagonal entries zeroed).
    static Matrix diagonal(const Matrix& a) {
        if (!a.is_square()) throw std::invalid_argument("The matrix must be square.");
        Matrix d(a.number_of_columns());
        for (int i = 0; i < a.number_of_rows(); ++i)
            for (int j = 0; j < a.number_of_columns(); ++j) d(i, j) = i == j ? a(i, j) : 0.0;
        return d;
    }

    // Returns the diagonal matrix built from a vector.
    static Matrix diagonal(const Vector& a) {
        Matrix d(a.length());
        for (int i = 0; i < d.number_of_rows(); ++i)
            for (int j = 0; j < d.number_of_columns(); ++j) d(i, j) = i == j ? a[i] : 0.0;
        return d;
    }

    // Returns the identity matrix of size n.
    static Matrix identity(int size) {
        Matrix I(size);
        for (int j = 0; j < size; ++j) I(j, j) = 1.0;
        return I;
    }

    // Returns the outer product A = x y^T, size x.Length x y.Length (C# Matrix.Outer,
    // line 561; B10). The C# null-argument guards are unrepresentable for references.
    static Matrix outer(const Vector& x, const Vector& y) {
        int m = x.length();
        int n = y.length();
        Matrix A(m, n);

        for (int i = 0; i < m; i++) {
            double xi = x[i];
            for (int j = 0; j < n; j++) A(i, j) = xi * y[j];
        }
        return A;
    }

    // Computes the square root of the matrix elementwise, in place (mirrors the C#
    // `void Sqrt()`, which mutates via `Apply(Math.Sqrt)`).
    void sqrt() {
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < number_of_columns(); ++j) data_[i][j] = std::sqrt(data_[i][j]);
    }

    // Multiply by another matrix.
    Matrix multiply(const Matrix& other) const {
        if (number_of_columns() != other.number_of_rows())
            throw std::invalid_argument(
                "The number of rows in the right-hand matrix must be equal to the number of "
                "columns in this matrix.");
        Matrix result(number_of_rows(), other.number_of_columns());
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < other.number_of_columns(); ++j) {
                double sum = 0.0;
                for (int k = 0; k < number_of_columns(); ++k) sum += data_[i][k] * other(k, j);
                result(i, j) = sum;
            }
        return result;
    }

    // Multiply by a vector.
    Vector multiply(const Vector& vector) const {
        if (number_of_columns() != vector.length())
            throw std::invalid_argument(
                "The number of rows in vector must be equal to the number of columns in the "
                "matrix.");
        std::vector<double> result(static_cast<std::size_t>(number_of_rows()));
        for (int i = 0; i < number_of_rows(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < number_of_columns(); ++j) sum += data_[i][j] * vector[j];
            result[i] = sum;
        }
        return Vector(std::move(result));
    }

    // Multiply by a scalar.
    Matrix multiply(double scalar) const {
        Matrix result(number_of_rows(), number_of_columns());
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < number_of_columns(); ++j) result(i, j) = data_[i][j] * scalar;
        return result;
    }

    // Divide by a scalar (C# Matrix.Divide, line 638; B10).
    Matrix divide(double scalar) const {
        Matrix result(number_of_rows(), number_of_columns());
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < number_of_columns(); ++j) result(i, j) = data_[i][j] / scalar;
        return result;
    }

    // Add the matrix. Matrices must have the same number of rows and columns.
    Matrix add(const Matrix& other) const {
        if (number_of_columns() != other.number_of_columns())
            throw std::invalid_argument("The matrix must have the same number of columns.");
        if (number_of_rows() != other.number_of_rows())
            throw std::invalid_argument("The matrix must have the same number of rows.");
        Matrix result(number_of_rows(), number_of_columns());
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < number_of_columns(); ++j) result(i, j) = data_[i][j] + other(i, j);
        return result;
    }

    // Subtract the matrix. Matrices must have the same number of rows and columns.
    Matrix subtract(const Matrix& other) const {
        if (number_of_columns() != other.number_of_columns())
            throw std::invalid_argument("The matrix must have the same number of columns.");
        if (number_of_rows() != other.number_of_rows())
            throw std::invalid_argument("The matrix must have the same number of rows.");
        Matrix result(number_of_rows(), number_of_columns());
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < number_of_columns(); ++j) result(i, j) = data_[i][j] - other(i, j);
        return result;
    }

   private:
    Matrix2D data_;
};

// Multiplies matrix A and B and returns the result as a matrix.
inline Matrix operator*(const Matrix& a, const Matrix& b) { return a.multiply(b); }

// Multiplies matrix A with vector b and returns the result as a vector.
inline Vector operator*(const Matrix& a, const Vector& b) { return a.multiply(b); }

// Multiplies a matrix A with a scalar and returns the result as a matrix.
inline Matrix operator*(const Matrix& a, double scalar) { return a.multiply(scalar); }

// Multiplies a matrix A with a scalar and returns the result as a matrix.
inline Matrix operator*(double scalar, const Matrix& a) { return a.multiply(scalar); }

// Adds matrix A and B and returns the result as a matrix.
inline Matrix operator+(const Matrix& a, const Matrix& b) { return a.add(b); }

// Subtracts matrix A and B and returns the result as a matrix.
inline Matrix operator-(const Matrix& a, const Matrix& b) { return a.subtract(b); }

// Divides a matrix A with a scalar and returns the result as a matrix (C# operator/,
// line 742; B10).
inline Matrix operator/(const Matrix& a, double scalar) { return a.divide(scalar); }

}  // namespace corehydro::numerics::math::linalg
