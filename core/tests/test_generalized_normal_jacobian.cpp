// GeneralizedNormal::quantile_jacobian has no fixture/oracle coverage: C# throws
// NotImplementedException on ParameterCovariance and QuantileVariance, and QuantileJacobian's own
// C# test is a tautology (it compares the analytic gradient against the same numerical derivative
// the method calls internally, so there is no independent literal to scrape). This file instead
// checks quantile_jacobian by analytic identity against the method it is built from
// (quantile_gradient, which IS oracle-pinned elsewhere): each Jacobian row must equal the gradient
// at that row's probability, and the returned determinant must equal the 3x3 cofactor expansion of
// those same rows computed independently here.
#include "corehydro/numerics/distributions/generalized_normal.hpp"

#include <vector>

#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "check.hpp"

using corehydro::numerics::distributions::GeneralizedNormal;
using corehydro::numerics::math::linalg::Matrix2D;

int main() {
    // kappa != 0 so all three parameters are in play (kappa == 0 is the degenerate Normal case).
    GeneralizedNormal dist(100.0, 10.0, -0.1);

    std::vector<double> probabilities = {0.1, 0.5, 0.99};
    double determinant = 0.0;
    Matrix2D jac = dist.quantile_jacobian(probabilities, determinant);

    // Each Jacobian row equals quantile_gradient at the corresponding probability.
    for (std::size_t row = 0; row < probabilities.size(); ++row) {
        std::vector<double> expected_row = dist.quantile_gradient(probabilities[row]);
        CHECK_EQ(jac[row].size(), expected_row.size());
        for (std::size_t col = 0; col < expected_row.size(); ++col) {
            CHECK_NEAR(jac[row][col], expected_row[col], 1e-15);
        }
    }

    // The determinant equals the 3x3 cofactor expansion of the gradient rows, computed
    // independently of the method under test.
    double a = jac[0][0], b = jac[0][1], c = jac[0][2];
    double d = jac[1][0], e = jac[1][1], f = jac[1][2];
    double g = jac[2][0], h = jac[2][1], i = jac[2][2];
    double expected_determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    CHECK_NEAR(determinant, expected_determinant, 1e-12);

    // Size-mismatch guard: probabilities must have exactly number_of_parameters() (3) entries.
    {
        std::vector<double> too_few = {0.1, 0.5};
        double det2 = 0.0;
        CHECK_THROWS(dist.quantile_jacobian(too_few, det2));
    }

    return chtest::summary("generalized_normal_jacobian");
}
