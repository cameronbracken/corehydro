// cpp11 glue for the multivariate-distribution fixture surface the shared distribution runner
// cannot serve. Every user-facing multivariate verb now goes through ch_mvdist_run_ in
// dist_spec.cpp, which drives the same dist_runner.hpp entry point the C++ fixture runner, the
// Python glue and the dotnet oracle emitter drive. Three groups stay here (see
// test-fixtures.R's delegation comment): the MVNDST integrator internals and Dirichlet's static
// LogMultivariateBeta, which have no runner verb; the non-finite constructs and evaluation
// points, which the runner's JSON grammar has no literal for; and MultivariateNormal's seeded
// MVNUNI sequences, which pin a run of values off ONE object and so cannot come from a
// stateless runner. Entry points are generic-by-method-string rather than one function per
// method, so each retains the arms it had before even where only one is reached today.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <memory>
#include <string>
#include <vector>

#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/distributions/multivariate/bivariate_empirical.hpp"
#include "corehydro/numerics/distributions/multivariate/dirichlet.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_student_t.hpp"

namespace mvd = corehydro::numerics::distributions;
namespace data = corehydro::numerics::data;
using namespace cpp11;

// --- Dirichlet -----------------------------------------------------------------------
// method + flat args in, double out. Methods: dimension, parameters_valid, alpha,
// alpha_sum, mean, variance, mode, covariance, pdf, log_pdf, log_multivariate_beta,
// random_value (args = [sample_size, seed, row, col]; no LHS -- Dirichlet has no
// LatinHypercubeRandomValues in the C# source, see fixtures/README.md). `args` carries
// index/point arguments per method (see corehydror/tests/testthat's dispatch for the exact
// per-method argument shape, mirroring fixtures/README.md).

[[cpp11::register]]
double ch_dirichlet_val_(std::string method, doubles alpha, doubles args) {
    std::vector<double> a(alpha.begin(), alpha.end());
    std::vector<double> ar(args.begin(), args.end());
    mvd::Dirichlet d(a);

    if (method == "dimension") return d.dimension();
    if (method == "parameters_valid") return d.parameters_valid() ? 1.0 : 0.0;
    if (method == "alpha") return d.alpha()[static_cast<std::size_t>(ar[0])];
    if (method == "alpha_sum") return d.alpha_sum();
    if (method == "mean") return d.mean()[static_cast<std::size_t>(ar[0])];
    if (method == "variance") return d.variance()[static_cast<std::size_t>(ar[0])];
    if (method == "mode") return d.mode()[static_cast<std::size_t>(ar[0])];
    if (method == "covariance")
        return d.covariance(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
    if (method == "pdf") return d.pdf(ar);
    if (method == "log_pdf") return d.log_pdf(ar);
    if (method == "log_multivariate_beta") return mvd::Dirichlet::log_multivariate_beta(ar);
    if (method == "random_value") {
        // args = [sample_size, seed, row, col]; stateless (generate_random_values seeds
        // its own MersenneTwister from `seed`).
        auto sample = d.generate_random_values(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
        return sample[static_cast<std::size_t>(ar[2])][static_cast<std::size_t>(ar[3])];
    }
    stop("unknown Dirichlet fixture method '%s'", method.c_str());
}

// --- BivariateEmpirical ------------------------------------------------------------
// Grid passed flattened row-major (nrow = x1.size(), ncol = x2.size()) + nrow. transforms
// is a 3-element character vector: (x1_transform, x2_transform, p_transform), each
// "None"/"Logarithmic"/"NormalZ". Methods: dimension, parameters_valid, cdf (vector arg,
// args = [x1, x2]), cdf_xy (same, scalar-pair entry point).

static data::Transform parse_transform(const std::string& s) {
    if (s == "None") return data::Transform::None;
    if (s == "Logarithmic") return data::Transform::Logarithmic;
    if (s == "NormalZ") return data::Transform::NormalZ;
    stop("unknown transform '%s'", s.c_str());
}

[[cpp11::register]]
double ch_bve_cdf_(std::string method, doubles x1, doubles x2, doubles p_flat, int nrow,
                    strings transforms, doubles args) {
    std::vector<double> x1v(x1.begin(), x1.end());
    std::vector<double> x2v(x2.begin(), x2.end());
    std::size_t ncol = x2v.size();
    std::vector<std::vector<double>> p(static_cast<std::size_t>(nrow), std::vector<double>(ncol));
    for (int i = 0; i < nrow; ++i)
        for (std::size_t j = 0; j < ncol; ++j)
            p[static_cast<std::size_t>(i)][j] = p_flat[static_cast<int>(static_cast<std::size_t>(i) * ncol + j)];

    mvd::BivariateEmpirical bv(x1v, x2v, p, parse_transform(std::string(transforms[0])),
                                parse_transform(std::string(transforms[1])),
                                parse_transform(std::string(transforms[2])));

    if (method == "dimension") return bv.dimension();
    if (method == "parameters_valid") return bv.parameters_valid() ? 1.0 : 0.0;
    if (method == "cdf") return bv.cdf(args[0], args[1]);
    if (method == "cdf_xy") return bv.cdf(args[0], args[1]);
    stop("unknown BivariateEmpirical fixture method '%s'", method.c_str());
}

// v2.1.4: verifies the stale-cache fix (SetParameters now nulls the cached bilinear
// interpolator) in ONE self-contained call -- CDF() once (forces the bilinear cache to
// build against the CURRENT grid), SetParameters() with a REPLACEMENT grid, then CDF()
// again. `p_new_flat` is nrow_new*x2_new.size() row-major, matching `p_flat`'s convention.
[[cpp11::register]]
double ch_bve_cdf_after_set_parameters_(doubles x1, doubles x2, doubles p_flat, int nrow,
                                         strings transforms, doubles x1_new, doubles x2_new,
                                         doubles p_new_flat, int nrow_new, double x1_eval,
                                         double x2_eval) {
    std::vector<double> x1v(x1.begin(), x1.end());
    std::vector<double> x2v(x2.begin(), x2.end());
    std::size_t ncol = x2v.size();
    std::vector<std::vector<double>> p(static_cast<std::size_t>(nrow), std::vector<double>(ncol));
    for (int i = 0; i < nrow; ++i)
        for (std::size_t j = 0; j < ncol; ++j)
            p[static_cast<std::size_t>(i)][j] = p_flat[static_cast<int>(static_cast<std::size_t>(i) * ncol + j)];

    mvd::BivariateEmpirical bv(x1v, x2v, p, parse_transform(std::string(transforms[0])),
                                parse_transform(std::string(transforms[1])),
                                parse_transform(std::string(transforms[2])));
    bv.cdf(x1_eval, x2_eval);

    std::vector<double> x1n(x1_new.begin(), x1_new.end());
    std::vector<double> x2n(x2_new.begin(), x2_new.end());
    std::size_t ncol_new = x2n.size();
    std::vector<std::vector<double>> p_new(static_cast<std::size_t>(nrow_new), std::vector<double>(ncol_new));
    for (int i = 0; i < nrow_new; ++i)
        for (std::size_t j = 0; j < ncol_new; ++j)
            p_new[static_cast<std::size_t>(i)][j] =
                p_new_flat[static_cast<int>(static_cast<std::size_t>(i) * ncol_new + j)];
    bv.set_parameters(std::move(x1n), std::move(x2n), std::move(p_new));
    return bv.cdf(x1_eval, x2_eval);
}

// --- MultivariateNormal (deterministic methods) ---------------------------------------
// Stateless per-call style: a fresh instance is constructed from (mean, cov_flat) on every
// call. Covers everything that does not touch the seeded MVNUNI stream (mean, median,
// mode, sd, variance, covariance, pdf, log_pdf, cdf for dim 1-2, mahalanobis,
// inverse_cdf, dimension, parameters_valid). Seeded CDF (dim>=3) / MVNDST batches need a
// single persistent instance across several calls -- see ch_mvn_cdf_seq_/
// ch_mvn_mvndst_seq_ (added alongside the MVNDST port). random_value/lhs_value (args =
// [sample_size, seed, row, col]) are ALSO stateless despite being "seeded" -- unlike
// MVNUNI, GenerateRandomValues/LatinHypercubeRandomValues seed their own fresh
// MersenneTwister/LatinHypercube draw from the `seed` argument every call, so a fresh
// instance per call is correct (see fixtures/README.md's Statefulness section).

[[cpp11::register]]
double ch_mvn_val_(std::string method, doubles mean, doubles cov_flat, doubles args) {
    std::vector<double> mu(mean.begin(), mean.end());
    std::size_t dim = mu.size();
    std::vector<std::vector<double>> cov(dim, std::vector<double>(dim));
    for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t j = 0; j < dim; ++j) cov[i][j] = cov_flat[static_cast<int>(i * dim + j)];
    std::vector<double> ar(args.begin(), args.end());
    mvd::MultivariateNormal n(mu, cov);

    if (method == "dimension") return n.dimension();
    if (method == "parameters_valid") return n.parameters_valid() ? 1.0 : 0.0;
    if (method == "mean") return n.mean()[static_cast<std::size_t>(ar[0])];
    if (method == "median") return n.median()[static_cast<std::size_t>(ar[0])];
    if (method == "mode") return n.mode()[static_cast<std::size_t>(ar[0])];
    if (method == "sd") return n.standard_deviation()[static_cast<std::size_t>(ar[0])];
    if (method == "variance") return n.variance()[static_cast<std::size_t>(ar[0])];
    if (method == "covariance") return n.covariance(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
    if (method == "pdf") return n.pdf(ar);
    if (method == "log_pdf") return n.log_pdf(ar);
    if (method == "cdf") return n.cdf(ar);
    if (method == "mahalanobis") return n.mahalanobis(ar);
    if (method == "inverse_cdf") {
        // args = [p_1..p_dim, index]
        std::vector<double> p(ar.begin(), ar.end() - 1);
        int idx = static_cast<int>(ar.back());
        return n.inverse_cdf(p)[static_cast<std::size_t>(idx)];
    }
    if (method == "random_value") {
        // args = [sample_size, seed, row, col]; stateless (generate_random_values seeds
        // its own MersenneTwister from `seed` -- independent of the MVNUNI stream the
        // seeded cdf/interval/mvndst batch above uses).
        auto sample = n.generate_random_values(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
        return sample[static_cast<std::size_t>(ar[2])][static_cast<std::size_t>(ar[3])];
    }
    if (method == "lhs_value") {
        auto sample = n.latin_hypercube_random_values(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
        return sample[static_cast<std::size_t>(ar[2])][static_cast<std::size_t>(ar[3])];
    }
    // v2.1.4: IsDensityValid is always true for a freshly-constructed (throwing-ctor)
    // instance -- the interesting false-after-a-failed-Try* transition is a stateful
    // sequence that doesn't fit this stateless dispatch shape; see
    // core/tests/test_multivariate_normal_api.cpp for that half.
    if (method == "is_density_valid") return n.is_density_valid() ? 1.0 : 0.0;
    if (method == "mvndst" || method == "mvndst_inform" || method == "mvndst_error") {
        // Stateless (unseeded) mvndst path -- exercised by the deterministic status-code/
        // collapse cases whose construct carries no "seed" (mvnuni_ is never touched: no
        // -1/finite-bound-count reaches DKBVRC for these). Same args shape as
        // mvndst_seq's per-call arrays, packed flat here -- args = [n_dim, lower...,
        // upper..., infin..., correl..., maxpts, abseps, releps], sizes inferred from
        // n_dim/nl.
        int n_dim = static_cast<int>(ar[0]);
        int nl = n_dim * (n_dim - 1) / 2;
        std::size_t i = 1;
        std::vector<double> lower(ar.begin() + static_cast<long>(i), ar.begin() + static_cast<long>(i + static_cast<std::size_t>(n_dim)));
        i += static_cast<std::size_t>(n_dim);
        std::vector<double> upper(ar.begin() + static_cast<long>(i), ar.begin() + static_cast<long>(i + static_cast<std::size_t>(n_dim)));
        i += static_cast<std::size_t>(n_dim);
        std::vector<int> infin;
        for (std::size_t k = 0; k < static_cast<std::size_t>(n_dim); ++k) infin.push_back(static_cast<int>(ar[i + k]));
        i += static_cast<std::size_t>(n_dim);
        std::vector<double> correl(ar.begin() + static_cast<long>(i), ar.begin() + static_cast<long>(i + static_cast<std::size_t>(nl)));
        i += static_cast<std::size_t>(nl);
        int maxpts = static_cast<int>(ar[i]);
        double abseps = ar[i + 1];
        double releps = ar[i + 2];
        double error = 0, value = 0;
        int inform = 0;
        n.mvndst(n_dim, lower, upper, infin, correl, maxpts, abseps, releps, error, value, inform);
        if (method == "mvndst_inform") return inform;
        if (method == "mvndst_error") return error;
        return value;
    }
    stop("unknown MultivariateNormal fixture method '%s'", method.c_str());
}

// --- MultivariateNormal (seeded batch methods) -----------------------------------------
// `cdf` for dim>=3 draws from the seeded MVNUNI stream (via MVNDST); `mvndst` is the
// Fortran-oracle entry point itself. Each needs a SINGLE persistent instance across a whole
// run of k sequential calls -- rebuilding per call the way ch_mvn_val_ does would silently
// reset the seeded RNG between assertions. The R fixture runner groups consecutive same-method
// seeded assertions within a case and calls these once per run, comparing results
// element-wise (see test-fixtures.R). `Interval` also consumes the stream, but the corpus's
// only Interval case makes exactly one such call, so it delegates through the grammar's
// `seed` key and needs no batch entry point here.

[[cpp11::register]]
doubles ch_mvn_cdf_seq_(doubles mean, doubles cov_flat, int seed, doubles xs_flat, int k) {
    std::vector<double> mu(mean.begin(), mean.end());
    std::size_t dim = mu.size();
    std::vector<std::vector<double>> cov(dim, std::vector<double>(dim));
    for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t j = 0; j < dim; ++j) cov[i][j] = cov_flat[static_cast<int>(i * dim + j)];
    mvd::MultivariateNormal n(mu, cov);
    n.set_mvnuni_seed(seed);

    writable::doubles out(k);
    for (int c = 0; c < k; ++c) {
        std::vector<double> x(dim);
        for (std::size_t d = 0; d < dim; ++d)
            x[d] = xs_flat[static_cast<int>(static_cast<std::size_t>(c) * dim + d)];
        out[c] = n.cdf(x);
    }
    return out;
}

// n_dim: the MVNDST `N` argument (also the dimension used to build the scratch instance
// -- an identity-covariance MultivariateNormal(n_dim), matching the Fortran oracle test's
// `new MultivariateNormal(5)`). Every per-call array is flattened k*n_dim (k*NL for
// correl, NL = n_dim*(n_dim-1)/2); maxpts/abseps/releps are length-k (one per call).
[[cpp11::register]]
doubles ch_mvn_mvndst_seq_(int n_dim, int seed, doubles lower_flat, doubles upper_flat,
                            integers infin_flat, doubles correl_flat, integers maxpts_v,
                            doubles abseps_v, doubles releps_v, int k) {
    mvd::MultivariateNormal mvn(n_dim);
    mvn.set_mvnuni_seed(seed);
    int nl = n_dim * (n_dim - 1) / 2;

    writable::doubles out(k);
    for (int c = 0; c < k; ++c) {
        std::vector<double> lower(static_cast<std::size_t>(n_dim)), upper(static_cast<std::size_t>(n_dim));
        std::vector<int> infin(static_cast<std::size_t>(n_dim));
        std::vector<double> correl(static_cast<std::size_t>(nl));
        for (int d = 0; d < n_dim; ++d) {
            lower[static_cast<std::size_t>(d)] = lower_flat[c * n_dim + d];
            upper[static_cast<std::size_t>(d)] = upper_flat[c * n_dim + d];
            infin[static_cast<std::size_t>(d)] = infin_flat[c * n_dim + d];
        }
        for (int d = 0; d < nl; ++d) correl[static_cast<std::size_t>(d)] = correl_flat[c * nl + d];

        double error = 0, value = 0;
        int inform = 0;
        mvn.mvndst(n_dim, lower, upper, infin, correl, maxpts_v[c], abseps_v[c], releps_v[c], error, value,
                   inform);
        out[c] = value;
    }
    return out;
}

// --- MultivariateStudentT ---------------------------------------------------------------
// Stateless per-call style, mirroring ch_mvn_val_: a fresh instance is constructed from
// (df, location, scale_flat) on every call. Unlike MultivariateNormal, the CDF here (K=200
// stratified chi-squared mixture) is fully deterministic -- no seeded MVNUNI stream, so no
// ch_mvt_*_seq_ batch entry point is needed. Methods: dimension, parameters_valid,
// degrees_of_freedom, mean, median, mode, sd, variance, covariance, pdf, log_pdf, cdf,
// mahalanobis, inverse_cdf, random_value, lhs_value (args = [sample_size, seed, row, col];
// stateless for the same reason as MultivariateNormal's -- see the comment above
// ch_mvn_val_).

[[cpp11::register]]
double ch_mvt_val_(std::string method, double df, doubles location, doubles scale_flat, doubles args) {
    std::vector<double> loc(location.begin(), location.end());
    std::size_t dim = loc.size();
    std::vector<std::vector<double>> scale(dim, std::vector<double>(dim));
    for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t j = 0; j < dim; ++j) scale[i][j] = scale_flat[static_cast<int>(i * dim + j)];
    std::vector<double> ar(args.begin(), args.end());
    mvd::MultivariateStudentT t(df, loc, scale);

    if (method == "dimension") return t.dimension();
    if (method == "parameters_valid") return t.parameters_valid() ? 1.0 : 0.0;
    if (method == "degrees_of_freedom") return t.degrees_of_freedom();
    if (method == "mean") return t.mean()[static_cast<std::size_t>(ar[0])];
    if (method == "median") return t.median()[static_cast<std::size_t>(ar[0])];
    if (method == "mode") return t.mode()[static_cast<std::size_t>(ar[0])];
    if (method == "sd") return t.standard_deviation()[static_cast<std::size_t>(ar[0])];
    if (method == "variance") return t.variance()[static_cast<std::size_t>(ar[0])];
    if (method == "covariance") return t.covariance(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
    if (method == "pdf") return t.pdf(ar);
    if (method == "log_pdf") return t.log_pdf(ar);
    if (method == "cdf") return t.cdf(ar);
    if (method == "mahalanobis") return t.mahalanobis(ar);
    if (method == "inverse_cdf") {
        // args = [p_1..p_dim+1, index]
        std::vector<double> p(ar.begin(), ar.end() - 1);
        int idx = static_cast<int>(ar.back());
        return t.inverse_cdf(p)[static_cast<std::size_t>(idx)];
    }
    if (method == "random_value") {
        // args = [sample_size, seed, row, col]; stateless (generate_random_values seeds
        // its own MersenneTwister from `seed`).
        auto sample = t.generate_random_values(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
        return sample[static_cast<std::size_t>(ar[2])][static_cast<std::size_t>(ar[3])];
    }
    if (method == "lhs_value") {
        auto sample = t.latin_hypercube_random_values(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
        return sample[static_cast<std::size_t>(ar[2])][static_cast<std::size_t>(ar[3])];
    }
    stop("unknown MultivariateStudentT fixture method '%s'", method.c_str());
}
