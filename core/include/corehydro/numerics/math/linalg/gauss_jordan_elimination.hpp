// ported from: Numerics/Mathematics/Linear Algebra/GaussJordanElimination.cs @ 2a0357a
//
// Gauss-Jordan elimination with full (row and column) pivoting, transcribed verbatim from
// the Numerical Recipes-style C# static `Solve(ref Matrix A, ref Matrix B)`: on return, `A`
// holds its own inverse and `B` holds the corresponding solution set. This is the one
// deliberately in-place, mutating free function under `numerics/math/linalg/` (every other
// header in this directory returns new values); the C# `ref, ref` signature IS the API, so
// it is mirrored as `gauss_jordan_solve(Matrix& a, Matrix& b)` rather than reshaped into a
// value-returning form. `B == null` (solve for the inverse only, no right-hand sides) is
// represented as a zero-column Matrix passed by the caller, since C++ has no null-matrix
// sentinel to mirror the C# `ref Matrix B = null` default.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/linalg/matrix.hpp"

namespace corehydro::numerics::math::linalg {

// Performs Gauss-Jordan elimination. On return, `a` is replaced by its matrix inverse, and
// `b` is replaced by the corresponding set of solution vectors (C# `GaussJordanElimination.
// Solve(ref Matrix A, ref Matrix B)`). Throws std::runtime_error on a singular matrix (C#
// `throw new Exception("Singular matrix")`).
inline void gauss_jordan_solve(Matrix& a, Matrix& b) {
    int n = a.number_of_rows();
    int m = b.number_of_columns();
    int icol = 0;
    int irow = 0;
    double big;
    double dum;
    double pivinv;
    // These integer arrays are used for bookkeeping on the pivoting.
    std::vector<int> indxc(static_cast<std::size_t>(n), 0);
    std::vector<int> indxr(static_cast<std::size_t>(n), 0);
    std::vector<int> ipiv(static_cast<std::size_t>(n), 0);

    // This is the main loop over the columns to be reduced.
    for (int i = 0; i < n; ++i) {
        big = 0.0;
        // This is the outer loop of the search for a pivot element.
        for (int j = 0; j < n; ++j) {
            if (ipiv[static_cast<std::size_t>(j)] != 1) {
                for (int k = 0; k < n; ++k) {
                    if (ipiv[static_cast<std::size_t>(k)] == 0) {
                        if (std::fabs(a(j, k)) >= big) {
                            big = std::fabs(a(j, k));
                            irow = j;
                            icol = k;
                        }
                    } else if (ipiv[static_cast<std::size_t>(k)] > 1) {
                        throw std::runtime_error("Singular matrix");
                    }
                }
            }
        }

        ipiv[static_cast<std::size_t>(icol)] += 1;
        // We now have the pivot element, so we interchange rows, if needed, to put the pivot
        // element on the diagonal. The columns are not physically interchanged, only relabeled:
        // indxc[i], the column of the (i+1)th pivot element, is the (i+1)th column that is
        // reduced, while indxr[i] is the row in which that pivot element was originally located.
        // If indxr[i] != indxc[i], there is an implied column interchange. With this form of
        // bookkeeping, the solution B's will end up in the correct order, and the inverse matrix
        // will be scrambled by columns.
        if (irow != icol) {
            for (int l = 0; l < n; ++l) {
                dum = a(irow, l);
                a(irow, l) = a(icol, l);
                a(icol, l) = dum;
            }
            for (int l = 0; l < m; ++l) {
                dum = b(irow, l);
                b(irow, l) = b(icol, l);
                b(icol, l) = dum;
            }
        }
        // We are now ready to divide the pivot row by the pivot element, located at irow and icol.
        indxr[static_cast<std::size_t>(i)] = irow;
        indxc[static_cast<std::size_t>(i)] = icol;
        if (a(icol, icol) == 0.0) throw std::runtime_error("Singular matrix");

        pivinv = 1.0 / a(icol, icol);
        a(icol, icol) = 1.0;
        for (int l = 0; l < n; ++l) a(icol, l) *= pivinv;
        for (int l = 0; l < m; ++l) b(icol, l) *= pivinv;
        // Now we reduce the rows except for the pivot one, of course.
        for (int ll = 0; ll < n; ++ll) {
            if (ll != icol) {
                dum = a(ll, icol);
                a(ll, icol) = 0.0;
                for (int l = 0; l < n; ++l) a(ll, l) -= a(icol, l) * dum;
                for (int l = 0; l < m; ++l) b(ll, l) -= b(icol, l) * dum;
            }
        }
    }
    // This is the end of the main loop over columns of the reduction. It only remains to
    // unscramble the solution in view of the column interchanges. We do this by interchanging
    // pairs of columns in the reverse order that the permutation was built up.
    for (int l = n - 1; l >= 0; --l) {
        if (indxr[static_cast<std::size_t>(l)] != indxc[static_cast<std::size_t>(l)]) {
            for (int k = 0; k < n; ++k) {
                dum = a(k, indxr[static_cast<std::size_t>(l)]);
                a(k, indxr[static_cast<std::size_t>(l)]) = a(k, indxc[static_cast<std::size_t>(l)]);
                a(k, indxc[static_cast<std::size_t>(l)]) = dum;
            }
        }
    }
    // And we are done.
}

}  // namespace corehydro::numerics::math::linalg
