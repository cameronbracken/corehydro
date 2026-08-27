// ported from: Numerics/Mathematics/Special Functions/Evaluate.cs @ 2a0357a
//
// Polynomial evaluation via Horner's method, in three coefficient orderings: ascending
// (`Polynomial`), descending (`PolynomialRev`), and descending with an implicit unit leading
// coefficient (`PolynomialRev_1`).
//
// Duplication note: gamma.hpp already inlines a private copy of this file's PolynomialRev
// algorithm as `special::detail::polynomial_rev` (a raw-pointer, explicit-degree overload used
// only by stirling/function/lanczos there), predating this file by construction order. That
// copy is left untouched here rather than refactored to share code with
// evaluate_polynomial_rev below -- changing gamma.hpp for this task risks oracle-visible churn
// on every distribution that touches the Gamma function, for zero behavioral gain (both copies
// implement the same C# `Evaluate.PolynomialRev` algorithm).
#pragma once
#include <stdexcept>
#include <vector>

namespace corehydro::numerics::math::special {

// Evaluates a polynomial with coefficients in ASCENDING order (coefficients[0] is the constant
// term): coefficients[n-1]*x^(n-1) + ... + coefficients[1]*x + coefficients[0].
// Ported from Evaluate.Polynomial.
inline double evaluate_polynomial(const std::vector<double>& coefficients, double x) {
    int n = static_cast<int>(coefficients.size());
    double value = coefficients[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        value *= x;
        value += coefficients[i];
    }
    return value;
}

// Evaluates a polynomial with coefficients in DESCENDING order (coefficients[0] is the
// highest-order coefficient): coefficients[0]*x^n + ... + coefficients[n]. `n` optionally
// redefines the polynomial's order to n+1; -1 (the default) uses the full coefficient list.
// Ported from Evaluate.PolynomialRev.
inline double evaluate_polynomial_rev(const std::vector<double>& coefficients, double x,
                                      int n = -1) {
    int size = static_cast<int>(coefficients.size());
    if (n > size)
        throw std::out_of_range(
            "evaluate_polynomial_rev: n cannot be greater than the number of coefficients");
    if (n == -1 || n == size) n = size - 1;

    double value = coefficients[0];
    for (int i = 1; i <= n; ++i) {
        value *= x;
        value += coefficients[i];
    }
    return value;
}

// Evaluates a polynomial with coefficients in DESCENDING order and an implicit leading
// coefficient of 1: x^n + coefficients[0]*x^(n-1) + ... + coefficients[n-1].
// Ported from Evaluate.PolynomialRev_1.
inline double evaluate_polynomial_rev_1(const std::vector<double>& coefficients, double x) {
    int n = static_cast<int>(coefficients.size());
    double value = x + coefficients[0];
    for (int i = 1; i < n; ++i) {
        value *= x;
        value += coefficients[i];
    }
    return value;
}

}  // namespace corehydro::numerics::math::special
