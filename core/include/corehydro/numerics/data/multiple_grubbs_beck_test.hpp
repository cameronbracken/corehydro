// ported from: Numerics/Data/Statistics/MultipleGrubbsBeckTest.cs @ 2a0357a
//
// The Multiple Grubbs-Beck low-outlier test (Cohn et al. 2013; Bulletin 17C), converted
// upstream from the FORTRAN source of PeakfqSA, plus the original Grubbs-Beck test with
// the Pilon et al. (1985) polynomial approximation of the 10% critical value.
//
// C# maps onto this port as follows:
// - `Function(double[] X)` -> `function(const std::vector<double>&)`.
// - `GrubbsBeckTest(IList<double> sample, out double XHi, out double XLo)` is void with
//   two `out` params; the closest C++ mirror is reference out-parameters, so this port
//   uses `grubbs_beck_test(sample, double& x_hi, double& x_lo)` (the struct-return idiom
//   mcmc_diagnostics.hpp uses is for the return-value + out-param mix, which does not
//   apply here). Its C# ArgumentException on non-positive values maps to
//   std::invalid_argument.
// - GGBCRITP configures an AdaptiveGaussKronrod with MaxDepth = 25 and
//   ReportFailure = false, returning NaN when Status == Failure. That reads across
//   member for member now that the integrator is a complete port over the Integrator
//   base; before it was, this file drove the minimal port's free `integrate` under a
//   try/catch instead. The two are equivalent -- a failed run leaves Result at the NaN
//   ClearResults set -- but the C# form is what the port carries.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/beta_distribution.hpp"
#include "corehydro/numerics/distributions/noncentral_t.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/integration/adaptive_gauss_kronrod.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"

namespace corehydro::numerics::data {

/// Contains functions for computing the Multiple Grubbs Beck low outlier test.
class MultipleGrubbsBeckTest {
   public:
    /// Generalized Grubbs-Beck Test. Identifies the number of low outliers using the
    /// generalized Grubbs-Beck test. `x` is a vector of floods in real space; returns
    /// the number of low outliers.
    static int function(const std::vector<double>& x) {
        double alpha_out = 0.005;
        double alpha_in = 0.0;
        double alpha_zero_in = 0.1;
        double max_frac_lo = 0.5;
        int n = static_cast<int>(x.size());
        std::vector<double> zt(static_cast<std::size_t>(n));
        std::vector<double> pvaluew(static_cast<std::size_t>(n));
        for (int i = 0; i < n; i++) {
            zt[static_cast<std::size_t>(i)] =
                std::log10(std::max(1.0e-88, x[static_cast<std::size_t>(i)]));
            pvaluew[static_cast<std::size_t>(i)] = -99.0;
        }

        // sort log flows from smallest to largest
        std::sort(zt.begin(), zt.end());

        // set starting point for MGBT search at approximate median position (1/2 N)
        int n2 = static_cast<int>(n * max_frac_lo);
        double s1 = 0.0;
        double s2 = 0.0;
        for (int i = n; i >= n2 + 2; i -= 1) {
            s1 += zt[static_cast<std::size_t>(i - 1)];
            s2 += std::pow(zt[static_cast<std::size_t>(i - 1)], 2.0);
        }

        int nc;
        double xv;
        double xm;
        std::vector<double> w(static_cast<std::size_t>(n));
        for (int i = n2; i >= 1; i -= 1) {
            s1 += zt[static_cast<std::size_t>(i)];
            s2 += std::pow(zt[static_cast<std::size_t>(i)], 2.0);
            nc = n - i;
            xm = s1 / nc;
            xv = (s2 - nc * std::pow(xm, 2.0)) / (nc - 1);
            w[static_cast<std::size_t>(i - 1)] =
                (zt[static_cast<std::size_t>(i - 1)] - xm) / std::sqrt(xv);
            pvaluew[static_cast<std::size_t>(i - 1)] =
                ggbcritp(n, i, w[static_cast<std::size_t>(i - 1)]);
        }

        // Determine Number of low outliers in 2 Or 3 steps.
        // Based on TAC original code And JRS recommendations.
        //
        // Step 1.   Outward sweep from median (always done).
        // alpha level of test = alpha_out
        // number of outliers = J1
        //
        // Step 2.   Inward sweep from largest low outlier identified in Step 1.
        // alpha level of test = alpha_in
        // number of outliers = J2
        //
        // Step 3.   Inward sweep from smallest observation
        // alpha level of test = alpha_zero_in
        // number of outliers = J3

        // Initialize counters
        int j1 = 0;  // Outward sweep number of low outliers
        int j2 = 0;  // Inward sweep number of low outliers
        int j3 = 0;  // Oth Inward sweep number of low outliers

        // 1) Outward sweep check: Loop over low flows up to median
        for (int i = n2; i >= 1; i -= 1) {
            if (pvaluew[static_cast<std::size_t>(i - 1)] < alpha_out) {
                j1 = i;
                break;
            }
        }

        // 2) Inward sweep check with alpha_in
        j2 = j1;
        for (int i = j1 + 1; i <= n2; i++) {
            if (pvaluew[static_cast<std::size_t>(i - 1)] >= alpha_in) {
                j2 = i - 1;
                break;
            }
        }

        // 3) Inward sweep check with alpha_zero_in
        for (int i = 1; i <= n2; i++) {
            if (pvaluew[static_cast<std::size_t>(i - 1)] >= alpha_zero_in) {
                j3 = i - 1;
                break;
            }
        }

        // Set number of low outliers as max of 3 sweeps
        int mgbtp = std::max(j1, std::max(j2, j3));
        return mgbtp;
    }

    /// The original Grubbs and Beck test for detection of outliers. Test is performed at
    /// the 10% significance level based on a normal distribution. Values greater than
    /// `x_hi` are considered high outliers; values lower than `x_lo` low outliers.
    static void grubbs_beck_test(const std::vector<double>& sample, double& x_hi,
                                 double& x_lo) {
        // The following polynomial approximation proposed by Pilon et al. (1985)
        int n = static_cast<int>(sample.size());
        for (double v : sample)
            if (v <= 0.0)
                throw std::invalid_argument(
                    "All sample values must be positive for the Grubbs-Beck test.");
        std::vector<double> log_sample(static_cast<std::size_t>(n));
        for (int i = 0; i < n; i++)
            log_sample[static_cast<std::size_t>(i)] =
                std::log(sample[static_cast<std::size_t>(i)]);
        double mean = data::mean(log_sample);
        double sd = data::standard_deviation(log_sample);
        double kn = -3.62201 + 6.28446 * std::pow(n, 0.25) - 2.49835 * std::pow(n, 0.5) +
                    0.491436 * std::pow(n, 0.75) - 0.037911 * n;
        x_hi = std::exp(mean + kn * sd);
        x_lo = std::exp(mean - kn * sd);
    }

   private:
    /// Auxiliary routine used to compute p-values (GGCRITP) for a Generalized
    /// Grubbs-Beck Test.
    static double ggbcritp(int n, int r, double eta) {
        if (n < 10 || r > n / 2.0) {
            return 0.5;
        }

        // The original FORTRAN source code utilized a globally adaptive Gauss-Kronrod
        // integration method. The number of low outliers computed by this method is
        // consistent with the results from the FORTRAN code.
        math::integration::AdaptiveGaussKronrod sr(
            [&](double pzr) { return fggb(pzr, n, r, eta); }, 1e-16, 1.0 - 1e-16);
        sr.max_depth = 25;
        sr.report_failure = false;
        sr.integrate();
        return sr.status() != math::integration::IntegrationStatus::Failure
                   ? sr.result()
                   : std::numeric_limits<double>::quiet_NaN();
    }

    /// Auxiliary routine used in ggbcritp.
    static double fggb(double pzr, int n, int r, double eta) {
        using distributions::BetaDistribution;
        using distributions::NoncentralT;
        using distributions::Normal;
        namespace special = math::special;

        double df, mu_m, mu_s2, var_m, var_s2, cov_m_s2;
        double ex1, ex2, ex3, ex4;
        double cov_m_s, var_s, alpha, beta;
        double mu_mp, eta_p, h, lambda, mu_s, ncp, q, var_mp, pr, zr, n2;
        double ans;

        // Compute the value of the r-th smallest obs. based on its order statistic
        n2 = n - r;
        BetaDistribution beta_dist(r, n + 1 - r);
        pr = beta_dist.inverse_cdf(pzr);
        zr = Normal::standard_z(pr);

        // Calculate the expected values of M, S2, S and their variances/covariances
        h = Normal::standard_pdf(zr) / std::max(0.0000000001, 1.0 - pr);
        ex1 = h;
        ex2 = 1.0 + h * zr;
        ex3 = 2.0 * ex1 + h * std::pow(zr, 2.0);
        ex4 = 3.0 * ex2 + h * std::pow(zr, 3.0);
        mu_m = ex1;
        mu_s2 = ex2 - std::pow(ex1, 2.0);
        var_m = mu_s2 / n2;
        var_s2 = (ex4 - 4.0 * ex3 * ex1 + 6.0 * ex2 * std::pow(ex1, 2.0) -
                  3.0 * std::pow(ex1, 4.0) - std::pow(mu_s2, 2.0)) /
                     n2 +
                 2.0 / ((n2 - 1.0) * n2) * std::pow(mu_s2, 2.0);
        alpha = mu_s2 * mu_s2 / var_s2;
        beta = mu_s2 / alpha;
        cov_m_s2 = (ex3 - 3.0 * ex2 * ex1 + 2.0 * std::pow(ex1, 3.0)) /
                   std::sqrt(n2 * (n2 - 1.0));
        mu_s = std::sqrt(beta) *
               std::exp(special::log_gamma(alpha + 0.5) - special::log_gamma(alpha));
        cov_m_s = cov_m_s2 / (2.0 * mu_s);
        var_s = mu_s2 - std::pow(mu_s, 2.0);
        lambda = cov_m_s / var_s;
        eta_p = eta + lambda;
        mu_mp = mu_m - lambda * mu_s;
        var_mp = var_m - cov_m_s * cov_m_s / var_s;
        df = 2.0 * alpha;
        ncp = (mu_mp - zr) / std::sqrt(var_mp);
        q = -std::sqrt(mu_s2 / var_mp) * eta_p;
        // Match Fortran FP_TNC_CDF: use normal approximation for df > 20
        double cdf_result;
        if (df > 20.0) {
            double z = (q * (1.0 - 1.0 / (4.0 * df)) - ncp) /
                       std::sqrt(1.0 + q * q / (2.0 * df));
            cdf_result = Normal::standard_cdf(z);
        } else {
            NoncentralT nct_dist(df, ncp);
            cdf_result = nct_dist.cdf(q);
        }
        ans = 1.0 - cdf_result;
        return ans;
    }
};

}  // namespace corehydro::numerics::data
