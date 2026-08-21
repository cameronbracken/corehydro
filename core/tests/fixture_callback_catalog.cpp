// The C++ fixture runner's catalogs of host-language callback counterparts. See
// fixture_callback_catalog.hpp for what these are, and core/CMakeLists.txt for why this file is a
// translation unit of its own: -ffp-contract=off is scoped to THIS file, so the lambdas below
// evaluate the arithmetic the fixtures name, while test_fixtures.cpp compiles the ported core
// exactly as the shipped R and Python packages compile it. Do not fold this file back into
// test_fixtures.cpp -- doing so puts the whole header-only core back under the flag.
#include "fixture_callback_catalog.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "optimization_test_functions.hpp"

namespace tbx = corehydro::numerics::support;
namespace bfsamp = corehydro::numerics::sampling;

namespace fixture_catalog {

// --- optimizer path (Task 8) -------------------------------------------------------------
//
// fixtures/toolbox/optimizers.json cases name a built-in objective by the same name TestFunctions.cs
// uses (DeJong/FXYZ/Booth/McCormick/FX/...); this table maps that name to the real C++ function in
// core/tests/optimization_test_functions.hpp -- the C++ analogue of the native closures the R and
// Python fixture runners write for the same names, so every fixture case exercises the real
// host-language callback path optimizer_runner.hpp exists to protect. `construct` is passed
// straight through to tbx::run_optimizer as the spec JSON (minus the "objective" key, which is not
// part of the runner's own grammar -- see optimizer_runner.hpp's file header on why no objective
// registry lives there).
tbx::Objective optimizer_objective(const std::string& name) {
    if (name == "FXYZ") return tbx::Objective(test_functions::fxyz);
    if (name == "DeJong") return tbx::Objective(test_functions::de_jong);
    if (name == "Booth") return tbx::Objective(test_functions::booth);
    if (name == "McCormick") return tbx::Objective(test_functions::mccormick);
    if (name == "FX")
        return tbx::Objective(
            [](const std::vector<double>& v) { return test_functions::fx(v[0]); });
    throw std::runtime_error("unknown optimizer fixture objective: " + name);
}

// --- callback path (callback surface, Task 1) --------------------------------------------
//
// fixtures/callback/*.json cases name a built-in callback by the same name the fixture's own
// `callbacks` block documents; this table maps that name to a real C++ lambda -- the C++ analogue
// of the native closures the R and Python fixture runners (and the dotnet emitter's delegates)
// write for the same names, so every case exercises the real host-language callback path
// callback_runner.hpp exists to protect. NOTE these names are NOT the optimizer catalog's above:
// `Diff_FXYZ` is Test_Differentiation.FXYZ (x^3 + y^4 + z^5), unrelated to the optimizer
// catalog's `FXYZ`; the `Diff_`/`Root_`/`Quad_` prefixes exist so the catalogs can never be
// confused (`Diff_FX` and `Quad_FX3` are both x^3, from two different upstream test files).
void callback_set(const std::string& name, tbx::CallbackSet& cbs) {
    if (name == "Root_Quadratic") {
        cbs.scalar = [](double x) { return x * x - 2.0; };
    } else if (name == "Root_Cubic") {
        cbs.scalar = [](double x) { return x * x * x - x - 1.0; };
    } else if (name == "Diff_FX") {
        cbs.scalar = [](double x) { return std::pow(x, 3.0); };
    } else if (name == "Diff_FXY") {
        cbs.vector_scalar = [](const std::vector<double>& p) {
            return std::pow(p[0], 2.0) * std::pow(p[1], 3.0);
        };
    } else if (name == "Diff_FXYZ") {
        cbs.vector_scalar = [](const std::vector<double>& p) {
            return std::pow(p[0], 3.0) + std::pow(p[1], 4.0) + std::pow(p[2], 5.0);
        };
    } else if (name == "Diff_FH") {
        cbs.vector_scalar = [](const std::vector<double>& p) {
            return std::pow(p[0], 3.0) - 2.0 * p[0] * p[1] - std::pow(p[1], 6.0);
        };
    } else if (name == "Quad_FX3") {
        cbs.scalar = [](double x) { return std::pow(x, 3.0); };
    } else if (name == "Quad_Cosine") {
        cbs.scalar = [](double x) { return std::cos(x); };
    } else if (name == "Quad_Sine") {
        cbs.scalar = [](double x) { return std::sin(x); };
    } else if (name == "Quad_FXX") {
        cbs.scalar = [](double x) { return 0.5 + 24.0 * x + 3.0 * x * x; };
    } else if (name == "Quad_FXXX") {
        cbs.scalar = [](double x) { return 0.5 + 24.0 * x + 3.0 * x * x + 8.0 * x * x * x; };
    } else if (name == "Quad_Peak") {
        // corehydro addition, no upstream integrand -- the one callback that reaches the
        // subdividing branch of the recursion. Arithmetic only, so all four runners agree bit for
        // bit and the evaluation count is a real oracle. See the fixture's `callbacks` note.
        cbs.scalar = [](double x) { return 1.0 / (1.0 + 1.0e4 * x * x); };
    } else if (name == "Mcmc_GaussianKernel") {
        // The mcmc catalog (fixtures/callback/mcmc.json). Both log-densities are arithmetic only
        // and sum in an explicit loop rather than through any accumulate helper: a Markov chain
        // turns one differing bit into a different chain, so the four runners have to agree to
        // the last bit for these oracles to mean anything. See the fixture's own note.
        cbs.vector_scalar = [](const std::vector<double>& p) {
            const double data[] = {4.9, 5.1, 5.0, 5.2, 4.8};
            double acc = 0.0;
            for (double x : data) acc += (x - p[0]) * (x - p[0]);
            return -0.5 * acc;
        };
    } else if (name == "Mcmc_LinearKernel") {
        cbs.vector_scalar = [](const std::vector<double>& p) {
            const double t[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
            const double y[] = {2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1};
            double acc = 0.0;
            for (std::size_t i = 0; i < 8; ++i) {
                double residual = y[i] - p[0] - p[1] * t[i];
                acc += residual * residual;
            }
            return -0.5 * acc;
        };
    } else if (name == "Mcmc_UniformWidthKernel") {
        // The Gibbs case's model, whose full conditional really IS uniform: with
        // x_i ~ Uniform(mu - 1, mu + 1) and a flat prior, mu given the data is
        // Uniform(max(x) - 1, min(x) + 1), so `Prop_UniformConditional` below is an exact Gibbs
        // step rather than a random walk wearing Gibbs's name. The normalizing term is dropped,
        // as the two kernels above drop theirs; comparisons and arithmetic only.
        cbs.vector_scalar = [](const std::vector<double>& p) {
            const double data[] = {4.9, 5.1, 5.0, 5.2, 4.8};
            for (double x : data)
                if (x - p[0] > 1.0 || p[0] - x > 1.0)
                    return -std::numeric_limits<double>::infinity();
            return 0.0;
        };
    } else if (name == "Prop_UniformConditional") {
        // The proposal catalog: (parameters, rng) -> parameters, upstream's Gibbs.Proposal shape.
        // Draws through the HANDLE, exactly as the R closure and the Python lambda do.
        cbs.vector_rng = [](const std::vector<double>&, bfsamp::MersenneTwister& prng) {
            const double data[] = {4.9, 5.1, 5.0, 5.2, 4.8};
            tbx::RngScope scope(prng);
            double lo = data[0] - 1.0, hi = data[0] + 1.0;
            for (double x : data) {
                if (x - 1.0 > lo) lo = x - 1.0;
                if (x + 1.0 < hi) hi = x + 1.0;
            }
            return std::vector<double>{lo + scope.handle()->uniform(1).at(0) * (hi - lo)};
        };
    } else if (name == "Grad_GaussianKernel") {
        // The gradient catalog: (parameters) -> vector, upstream's HMC.Gradient shape. The
        // analytic derivative of Mcmc_GaussianKernel, d/dmu = sum(x - mu).
        cbs.vector_vector = [](const std::vector<double>& p) {
            const double data[] = {4.9, 5.1, 5.0, 5.2, 4.8};
            double acc = 0.0;
            for (double x : data) acc += x - p[0];
            return std::vector<double>{acc};
        };
    } else if (name == "Mcmc_QuarticKernel") {
        // Task-5 review fix, coverage finding: unlike Mcmc_GaussianKernel, whose derivative
        // sum(x - mu) is LINEAR in mu (so its third derivative is zero and the ported
        // central-difference default agrees with the analytic gradient to rounding, ~4e-16), this
        // kernel's derivative is CUBIC in mu, so the central-difference truncation error is real
        // rather than rounding -- the analytic and default gradients genuinely disagree here, which
        // is what a supplied-vs-ignored gradient regression needs to be caught by the oracle gate.
        // The 0.05 coefficient is load-bearing, not decorative: measured by brute-force sweep
        // (a coefficient of 1 over the same data), an unscaled quartic makes HMC's leapfrog
        // trajectory genuinely CHAOTIC over 200 iterations -- the analytic and default gradients
        // then diverge at the ~0.3% level, the same order this fixture's own cross-language
        // divergence measured at that scale, so no fixed tolerance could pin it. At 0.05 the
        // divergence is small and smooth (~3e-8 relative, four orders past the file's 1e-12
        // tolerance) rather than chaotic, which is what keeps the case usable as an oracle at all.
        cbs.vector_scalar = [](const std::vector<double>& p) {
            const double data[] = {4.9, 5.1, 5.0, 5.2, 4.8};
            double acc = 0.0;
            for (double x : data) {
                double d = x - p[0];
                acc += d * d * d * d;
            }
            return -0.05 * acc;
        };
    } else if (name == "Grad_QuarticKernel") {
        // The analytic derivative of Mcmc_QuarticKernel, d/dmu = 0.05 * 4 * sum((x - mu)^3).
        cbs.vector_vector = [](const std::vector<double>& p) {
            const double data[] = {4.9, 5.1, 5.0, 5.2, 4.8};
            double acc = 0.0;
            for (double x : data) {
                double d = x - p[0];
                acc += d * d * d;
            }
            return std::vector<double>{0.2 * acc};
        };
    } else if (name == "Resample_Iid") {
        // The bootstrap catalog (fixtures/callback/bootstrap.json), upstream's four
        // Bootstrap<TData> delegate shapes. Every one is arithmetic and comparisons only, and the
        // mean is summed in an explicit loop rather than through any accumulate helper: R's sum()
        // and mean() accumulate in extended precision, and one differing bit in a fitted mean moves
        // a percentile. The resample draws every index through the HANDLE, exactly as a user's own
        // resample function does.
        cbs.data_rng = [](const std::vector<double>& data, const std::vector<double>&,
                          bfsamp::MersenneTwister& prng) {
            tbx::RngScope scope(prng);
            const int n = static_cast<int>(data.size());
            std::vector<double> out;
            out.reserve(data.size());
            for (int k : scope.handle()->integers(n, 0, n))
                out.push_back(data[static_cast<std::size_t>(k)]);
            return out;
        };
    } else if (name == "Fit_Mean") {
        cbs.data_vector = [](const std::vector<double>& data) {
            double acc = 0.0;
            for (double x : data) acc += x;
            return std::vector<double>{acc / static_cast<double>(data.size())};
        };
    } else if (name == "Stat_Identity") {
        cbs.vector_vector = [](const std::vector<double>& p) { return p; };
    } else if (name == "Stat_MeanAndSquare") {
        cbs.vector_vector = [](const std::vector<double>& p) {
            return std::vector<double>{p[0], p[0] * p[0]};
        };
    } else if (name == "Jack_LeaveOneOut") {
        cbs.data_index = [](const std::vector<double>& data, int index) {
            std::vector<double> out;
            out.reserve(data.size() - 1);
            for (std::size_t i = 0; i < data.size(); ++i)
                if (static_cast<int>(i) != index) out.push_back(data[i]);
            return out;
        };
    } else if (name == "Mom_NormalMeanVariance") {
        // The gmm catalog (fixtures/callback/gmm.json), upstream's three delegate shapes from
        // GeneralizedMethodOfMoments's delegate constructor. The model is the just-identified
        // two-parameter method-of-moments fit of a Normal: theta = (mu, sigma2) and
        //   g = [mean(x - mu), mean((x - mu)^2 - sigma2)],  S = the covariance of those two
        // whose unique root -- and so the GMM optimum, since q = p makes g = 0 attainable -- is
        // the sample mean and the population variance. Arithmetic and an explicit loop only, never
        // sum()/mean()/Average(): R accumulates both in extended precision where the other three
        // languages accumulate in double, and one differing bit moves a fitted parameter.
        cbs.moment_conditions = [](const std::vector<double>& p) {
            const double data[] = {4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7};
            const double n = 8.0;
            double g0 = 0.0, g1 = 0.0, s00 = 0.0, s01 = 0.0, s11 = 0.0;
            for (double x : data) {
                double a = x - p[0];
                double b = a * a - p[1];
                g0 += a;
                g1 += b;
                s00 += a * a;
                s01 += a * b;
                s11 += b * b;
            }
            tbx::MomentConditionReturn out;
            out.g = {g0 / n, g1 / n};
            out.s = {s00 / n, s01 / n, s01 / n, s11 / n};  // row-major, symmetric
            out.s_rows = 2;
            out.s_cols = 2;
            return out;
        };
    } else if (name == "Mom_NormalThreeMoments") {
        // The OVER-IDENTIFIED member of the same catalog: the identical Normal model and the
        // identical eight observations, with a third moment condition added -- mean((x - mu)^3),
        // zero for a Normal -- so q = 3 > p = 2 and the degrees of freedom become 1. This is the
        // only case in the file that reaches the chi-squared p-value branch of the J-statistic;
        // see the fixture's own note on why J ITSELF still cannot be pinned even here.
        cbs.moment_conditions = [](const std::vector<double>& p) {
            const double data[] = {4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7};
            const double n = 8.0;
            double g0 = 0.0, g1 = 0.0, g2 = 0.0;
            double s00 = 0.0, s01 = 0.0, s02 = 0.0, s11 = 0.0, s12 = 0.0, s22 = 0.0;
            for (double x : data) {
                double a = x - p[0];
                double b = a * a - p[1];
                double c = a * a * a;
                g0 += a;
                g1 += b;
                g2 += c;
                s00 += a * a;
                s01 += a * b;
                s02 += a * c;
                s11 += b * b;
                s12 += b * c;
                s22 += c * c;
            }
            tbx::MomentConditionReturn out;
            out.g = {g0 / n, g1 / n, g2 / n};
            out.s = {s00 / n, s01 / n, s02 / n,  // row-major, symmetric
                     s01 / n, s11 / n, s12 / n,
                     s02 / n, s12 / n, s22 / n};
            out.s_rows = 3;
            out.s_cols = 3;
            return out;
        };
    } else if (name == "Mom_NormalFourthMoment") {
        // The CUBIC-JACOBIAN member of the same catalog, and the one case in the file whose
        // analytic Jacobian is distinguishable from the ported numerical one. theta = (mu, sigma),
        // matched on the first and fourth central moments of a Normal:
        //   g = [mean(x - mu), mean(u^4) - 3 t^4],  u = 100 (x - mu),  t = 100 sigma
        // so dg2/dsigma = -1200 t^3 is cubic in the parameter, where every other case in the file
        // has a Jacobian that is linear in it. The eight observations are the ones above with the
        // decimal point moved two places, which is what makes the fitted sigma (0.00404) small next
        // to the numerical Jacobian's step h = 1e-4 (|theta| + 1); the factors of 100 write the
        // fourth-moment condition on the same order of magnitude as the first, which GMM is
        // invariant to because W = S^-1 absorbs it.
        cbs.moment_conditions = [](const std::vector<double>& p) {
            const double data[] = {0.041, 0.052, 0.048, 0.055, 0.049, 0.051, 0.053, 0.047};
            const double n = 8.0;
            const double t = p[1] * 100.0;
            const double t4 = 3.0 * t * t * t * t;
            double g0 = 0.0, g1 = 0.0, s00 = 0.0, s01 = 0.0, s11 = 0.0;
            for (double x : data) {
                double a = x - p[0];
                double u = a * 100.0;
                double b = u * u * u * u - t4;
                g0 += a;
                g1 += b;
                s00 += a * a;
                s01 += a * b;
                s11 += b * b;
            }
            tbx::MomentConditionReturn out;
            out.g = {g0 / n, g1 / n};
            out.s = {s00 / n, s01 / n, s01 / n, s11 / n};  // row-major, symmetric
            out.s_rows = 2;
            out.s_cols = 2;
            return out;
        };
    } else if (name == "Jac_NormalFourthMoment") {
        // The analytic Jacobian of Mom_NormalFourthMoment, row-major 2 x 2 (one ROW per moment
        // condition): dg1/dmu = -1, dg1/dsigma = 0, dg2/dmu = -400 mean(u^3), dg2/dsigma =
        // -1200 t^3.
        cbs.vector_matrix = [](const std::vector<double>& p) {
            const double data[] = {0.041, 0.052, 0.048, 0.055, 0.049, 0.051, 0.053, 0.047};
            double acc = 0.0;
            for (double x : data) {
                double u = (x - p[0]) * 100.0;
                acc += u * u * u;
            }
            const double t = p[1] * 100.0;
            return std::make_pair(
                std::vector<double>{-1.0, 0.0, -400.0 * acc / 8.0, -1200.0 * t * t * t},
                std::vector<int>{2, 2});
        };
    } else if (name == "Jac_NormalMeanVariance") {
        // The analytic Jacobian of Mom_NormalMeanVariance, row-major 2 x 2 (one ROW per moment
        // condition): dg1/dmu = -1, dg1/dsigma2 = 0, dg2/dmu = -2 mean(x - mu), dg2/dsigma2 = -1.
        cbs.vector_matrix = [](const std::vector<double>& p) {
            const double data[] = {4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7};
            double acc = 0.0;
            for (double x : data) acc += x - p[0];
            return std::make_pair(std::vector<double>{-1.0, 0.0, -2.0 * acc / 8.0, -1.0},
                                  std::vector<int>{2, 2});
        };
    } else if (name == "Pen_SigmaTowardsOne") {
        // A ridge penalty pulling sigma2 towards 1, carrying its own 1/2 as the ported
        // half-quadratic convention expects. Moves the estimate off the closed form, which is what
        // makes the penalty case an oracle rather than a repeat of the first one.
        cbs.vector_scalar = [](const std::vector<double>& p) {
            double d = p[1] - 1.0;
            return 0.5 * d * d;
        };
    } else if (name == "Rng_Uniform") {
        // The rng catalog (fixtures/callback/rng_handle.json). Each of these takes the handle the
        // runner hands it -- the C++ analogue of the R closure calling rng_uniform() and the Python
        // lambda calling rng.uniform() -- rather than reaching for the generator directly, so the
        // fixture pins what a USER's callback draws and not merely what the generator emits.
        cbs.vector_rng = [](const std::vector<double>& p, bfsamp::MersenneTwister& prng) {
            tbx::RngScope scope(prng);
            return scope.handle()->uniform(static_cast<int>(p.at(0)));
        };
    } else if (name == "Rng_Integers") {
        cbs.vector_rng = [](const std::vector<double>& p, bfsamp::MersenneTwister& prng) {
            tbx::RngScope scope(prng);
            std::vector<int> k = scope.handle()->integers(
                static_cast<int>(p.at(0)), static_cast<int>(p.at(1)), static_cast<int>(p.at(2)));
            return std::vector<double>(k.begin(), k.end());
        };
    } else if (name == "Rng_Interleaved") {
        cbs.vector_rng = [](const std::vector<double>&, bfsamp::MersenneTwister& prng) {
            tbx::RngScope scope(prng);
            tbx::RngBorrowPtr rng = scope.handle();
            std::vector<double> out = rng->uniform(2);
            for (int k : rng->integers(2, 0, 100)) out.push_back(static_cast<double>(k));
            out.push_back(rng->uniform(1).at(0));
            return out;
        };
    } else if (name == "Rng_Warmup1000") {
        cbs.vector_rng = [](const std::vector<double>&, bfsamp::MersenneTwister& prng) {
            tbx::RngScope scope(prng);
            tbx::RngBorrowPtr rng = scope.handle();
            rng->uniform(1000);  // discarded, as upstream's own test discards 1000 GenRandInt32
            return rng->uniform(10);
        };
    } else {
        throw std::runtime_error("unknown callback fixture callback: " + name);
    }
}

}  // namespace fixture_catalog
