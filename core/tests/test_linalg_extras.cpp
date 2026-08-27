// Transcribed from: upstream/Numerics/Test_Numerics/Mathematics/Linear Algebra/
// Test_QRDecomposition.cs and Test_GaussJordanElimination.cs (@ 2a0357a) for the P2
// "math extras" phase Task 9 QRDecomposition/GaussJordanElimination port. All 9 QR test
// methods and the 1 Gauss-Jordan test method are transcribed, values and tolerances
// unaltered: Test_QRDecomp, Test_SolveVector, Test_SolveMatrix,
// Test_QR_Solve_Vector_NonSymmetric_Square, Test_QR_Solve_Matrix_NonSymmetric_Square,
// Test_QR_Solve_Vector_NonSymmetric_Overdetermined,
// Test_QR_Solve_Matrix_NonSymmetric_Overdetermined,
// Test_QR_Solve_Vector_NonSymmetric_Underdetermined,
// Test_QR_Solve_Matrix_NonSymmetric_Underdetermined; Test_GaussJordanElim.
//
// Test_GaussJordanElim's C# assertion is exact (`Assert.AreEqual(true_IA[i, j], A[i, j])`,
// no tolerance) and its loop bound is `j < A.NumberOfColumns - 1` (only columns 0 and 1 of
// the 3x3 inverse are asserted by the upstream test, not column 2) -- both are transcribed
// verbatim rather than "fixed", per the port's structural-mirroring rule.
#include <vector>

#include "corehydro/numerics/math/linalg/gauss_jordan_elimination.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/qr_decomposition.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "check.hpp"

using corehydro::numerics::math::linalg::gauss_jordan_solve;
using corehydro::numerics::math::linalg::Matrix;
using corehydro::numerics::math::linalg::QRDecomposition;
using corehydro::numerics::math::linalg::Vector;

namespace {

// C# Test_QRDecomp.
void test_qr_decomp() {
    Matrix A(3);
    A(0, 0) = 1; A(0, 1) = 1; A(0, 2) = 1;
    A(1, 0) = 0; A(1, 1) = 2; A(1, 2) = 5;
    A(2, 0) = 2; A(2, 1) = 5; A(2, 2) = -1;

    QRDecomposition qr(A);
    Matrix Q = qr.q();
    Matrix R = qr.r();
    Matrix QR = Q * R;

    for (int i = 0; i < A.number_of_rows(); ++i)
        for (int j = 0; j < A.number_of_columns(); ++j) CHECK_NEAR(A(i, j), QR(i, j), 1e-10);
}

// C# Test_SolveVector.
void test_solve_vector() {
    Matrix A(3);
    A(0, 0) = 1; A(0, 1) = 1; A(0, 2) = 1;
    A(1, 0) = 0; A(1, 1) = 2; A(1, 2) = 5;
    A(2, 0) = 2; A(2, 1) = 5; A(2, 2) = -1;

    Vector B(std::vector<double>{6, -4, 27});
    QRDecomposition qr(A);
    Vector x = qr.solve(B);

    // Verify that A * x ~ B.
    Vector Ax = A * x;
    for (int i = 0; i < B.length(); ++i) CHECK_NEAR(B[i], Ax[i], 1e-10);
}

// C# Test_SolveMatrix.
void test_solve_matrix() {
    Matrix A(3);
    A(0, 0) = 1; A(0, 1) = 1; A(0, 2) = 1;
    A(1, 0) = 0; A(1, 1) = 2; A(1, 2) = 5;
    A(2, 0) = 2; A(2, 1) = 5; A(2, 2) = -1;

    Matrix matB(3, 1);
    matB(0, 0) = 6;
    matB(1, 0) = -4;
    matB(2, 0) = 27;

    QRDecomposition qr(A);
    Matrix matX = qr.solve(matB);

    Matrix AX = A * matX;
    for (int i = 0; i < AX.number_of_rows(); ++i) CHECK_NEAR(matB(i, 0), AX(i, 0), 1e-10);
}

// C# Test_QR_Solve_Vector_NonSymmetric_Square.
void test_qr_solve_vector_nonsymmetric_square() {
    Matrix A(3, 3, std::vector<double>{1, 2, 3, 0, 1, 4, 5, 6, 0});
    Vector b(std::vector<double>{1, 2, 3});

    QRDecomposition qr(A);
    Vector x = qr.solve(b);
    Vector Ax = A * x;

    for (int i = 0; i < b.length(); ++i) CHECK_NEAR(b[i], Ax[i], 1e-10);
}

// C# Test_QR_Solve_Matrix_NonSymmetric_Square.
void test_qr_solve_matrix_nonsymmetric_square() {
    Matrix A(3, 3, std::vector<double>{1, 2, 3, 0, 1, 4, 5, 6, 0});

    Matrix B(3, 1);
    B(0, 0) = 1;
    B(1, 0) = 2;
    B(2, 0) = 3;

    QRDecomposition qr(A);
    Matrix X = qr.solve(B);
    Matrix AX = A * X;

    for (int i = 0; i < A.number_of_rows(); ++i) CHECK_NEAR(B(i, 0), AX(i, 0), 1e-10);
}

// C# Test_QR_Solve_Vector_NonSymmetric_Overdetermined.
void test_qr_solve_vector_nonsymmetric_overdetermined() {
    Matrix A(4, 3, std::vector<double>{1, 1, 1, 1, 2, 3, 1, 3, 6, 1, 4, 10});
    Vector b(std::vector<double>{6, 0, 0, 6});

    QRDecomposition qr(A);
    Vector x = qr.solve(b);
    Vector Ax = A * x;

    for (int i = 0; i < b.length(); ++i) CHECK_NEAR(b[i], Ax[i], 1e-10);
}

// C# Test_QR_Solve_Matrix_NonSymmetric_Overdetermined.
void test_qr_solve_matrix_nonsymmetric_overdetermined() {
    Matrix A(4, 3, std::vector<double>{1, 1, 1, 1, 2, 3, 1, 3, 6, 1, 4, 10});

    Matrix B(4, 1);
    B(0, 0) = 6;
    B(1, 0) = 0;
    B(2, 0) = 0;
    B(3, 0) = 6;

    QRDecomposition qr(A);
    Matrix X = qr.solve(B);
    Matrix AX = A * X;

    for (int i = 0; i < A.number_of_rows(); ++i) CHECK_NEAR(B(i, 0), AX(i, 0), 1e-10);
}

// C# Test_QR_Solve_Vector_NonSymmetric_Underdetermined.
void test_qr_solve_vector_nonsymmetric_underdetermined() {
    Matrix A(3, 4, std::vector<double>{2, 3, 5, 1, 1, 0, 2, 3, 0, 1, 4, 2});
    Vector b(std::vector<double>{1, 2, 3});

    QRDecomposition qr(A);
    Vector x = qr.solve(b);
    Vector Ax = A * x;

    for (int i = 0; i < b.length(); ++i) CHECK_NEAR(b[i], Ax[i], 1e-10);
}

// C# Test_QR_Solve_Matrix_NonSymmetric_Underdetermined.
void test_qr_solve_matrix_nonsymmetric_underdetermined() {
    Matrix A(3, 4, std::vector<double>{2, 3, 5, 1, 1, 0, 2, 3, 0, 1, 4, 2});

    Matrix B(3, 1);
    B(0, 0) = 1;
    B(1, 0) = 2;
    B(2, 0) = 3;

    QRDecomposition qr(A);
    Matrix X = qr.solve(B);
    Matrix AX = A * X;

    for (int i = 0; i < A.number_of_rows(); ++i) CHECK_NEAR(B(i, 0), AX(i, 0), 1e-10);
}

// C# Test_GaussJordanElim.
void test_gauss_jordan_elim() {
    Matrix A(3, 3, std::vector<double>{1, 3, 3, 1, 4, 3, 1, 3, 4});
    Matrix true_IA(3, 3, std::vector<double>{7, -3, -3, -1, 1, 0, -1, 0, 1});

    Matrix argB(A.number_of_rows(), 0);  // C# `Matrix argB = null` -> zero-column B.
    gauss_jordan_solve(A, argB);

    for (int i = 0; i < A.number_of_rows(); ++i)
        for (int j = 0; j < A.number_of_columns() - 1; ++j) CHECK_EQ(true_IA(i, j), A(i, j));
}

}  // namespace

int main() {
    test_qr_decomp();
    test_solve_vector();
    test_solve_matrix();
    test_qr_solve_vector_nonsymmetric_square();
    test_qr_solve_matrix_nonsymmetric_square();
    test_qr_solve_vector_nonsymmetric_overdetermined();
    test_qr_solve_matrix_nonsymmetric_overdetermined();
    test_qr_solve_vector_nonsymmetric_underdetermined();
    test_qr_solve_matrix_nonsymmetric_underdetermined();
    test_gauss_jordan_elim();
    return chtest::summary("linalg_extras");
}
