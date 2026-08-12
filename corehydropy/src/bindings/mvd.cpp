// pybind11 glue for the multivariate-distribution fixture surface the shared distribution
// runner cannot serve. Every user-facing multivariate verb now goes through _core.mvdist_run in
// dist_spec.cpp, which drives the same dist_runner.hpp entry point the C++ fixture runner, the R
// glue and the dotnet oracle emitter drive. Three groups stay here (see test_fixtures.py's
// delegation comment): the MVNDST integrator internals and Dirichlet's static
// LogMultivariateBeta, which have no runner verb; the non-finite constructs and evaluation
// points, which the runner's JSON grammar has no literal for; and MultivariateNormal's seeded
// MVNUNI sequences, which pin a run of values off ONE object and so cannot come from a
// stateless runner. Entry points are generic-by-method-string rather than one function per
// method, so each retains the arms it had before even where only one is reached today.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <string>
#include <vector>

#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/distributions/multivariate/bivariate_empirical.hpp"
#include "corehydro/numerics/distributions/multivariate/dirichlet.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_student_t.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace mvd = corehydro::numerics::distributions;
namespace data = corehydro::numerics::data;

static data::Transform parse_transform(const std::string& s) {
    if (s == "None") return data::Transform::None;
    if (s == "Logarithmic") return data::Transform::Logarithmic;
    if (s == "NormalZ") return data::Transform::NormalZ;
    throw py::value_error("unknown transform: " + s);
}

void register_multivariate(py::module_& m) {
    // --- Dirichlet ---------------------------------------------------------------------
    // method + flat args in, double out. Methods: dimension, parameters_valid, alpha,
    // alpha_sum, mean, variance, mode, covariance, pdf, log_pdf, log_multivariate_beta,
    // random_value (args = [sample_size, seed, row, col]; stateless -- generate_random_values
    // seeds its own MersenneTwister from `seed`; no LHS -- Dirichlet has no
    // LatinHypercubeRandomValues in the C# source, see fixtures/README.md).
    m.def("dirichlet_val", [](const std::string& method, const std::vector<double>& alpha,
                               const std::vector<double>& args) {
        mvd::Dirichlet d(alpha);
        if (method == "dimension") return static_cast<double>(d.dimension());
        if (method == "parameters_valid") return d.parameters_valid() ? 1.0 : 0.0;
        if (method == "alpha") return d.alpha()[static_cast<std::size_t>(args[0])];
        if (method == "alpha_sum") return d.alpha_sum();
        if (method == "mean") return d.mean()[static_cast<std::size_t>(args[0])];
        if (method == "variance") return d.variance()[static_cast<std::size_t>(args[0])];
        if (method == "mode") return d.mode()[static_cast<std::size_t>(args[0])];
        if (method == "covariance")
            return d.covariance(static_cast<int>(args[0]), static_cast<int>(args[1]));
        if (method == "pdf") return d.pdf(args);
        if (method == "log_pdf") return d.log_pdf(args);
        if (method == "log_multivariate_beta") return mvd::Dirichlet::log_multivariate_beta(args);
        if (method == "random_value") {
            auto sample = d.generate_random_values(static_cast<int>(args[0]), static_cast<int>(args[1]));
            return sample[static_cast<std::size_t>(args[2])][static_cast<std::size_t>(args[3])];
        }
        throw py::value_error("unknown Dirichlet fixture method: " + method);
    });

    // --- BivariateEmpirical -------------------------------------------------------------
    // p is a nested list (row i = x1[i], column j = x2[j]). transforms is
    // (x1_transform, x2_transform, p_transform) strings. Methods: dimension,
    // parameters_valid, cdf (vector arg, args = [x1, x2]), cdf_xy (same, scalar pair).
    m.def("bve_cdf", [](const std::string& method, const std::vector<double>& x1,
                         const std::vector<double>& x2,
                         const std::vector<std::vector<double>>& p,
                         const std::vector<std::string>& transforms,
                         const std::vector<double>& args) {
        mvd::BivariateEmpirical bv(x1, x2, p, parse_transform(transforms[0]),
                                    parse_transform(transforms[1]), parse_transform(transforms[2]));
        if (method == "dimension") return static_cast<double>(bv.dimension());
        if (method == "parameters_valid") return bv.parameters_valid() ? 1.0 : 0.0;
        if (method == "cdf") return bv.cdf(args[0], args[1]);
        if (method == "cdf_xy") return bv.cdf(args[0], args[1]);
        throw py::value_error("unknown BivariateEmpirical fixture method: " + method);
    });

    // v2.1.4: verifies the stale-cache fix in ONE self-contained call -- CDF() once
    // (forces the bilinear cache to build against the CURRENT grid), set_parameters()
    // with a REPLACEMENT grid, then CDF() again.
    m.def("bve_cdf_after_set_parameters",
          [](const std::vector<double>& x1, const std::vector<double>& x2,
             const std::vector<std::vector<double>>& p, const std::vector<std::string>& transforms,
             const std::vector<double>& x1_new, const std::vector<double>& x2_new,
             const std::vector<std::vector<double>>& p_new, double x1_eval, double x2_eval) {
        mvd::BivariateEmpirical bv(x1, x2, p, parse_transform(transforms[0]),
                                    parse_transform(transforms[1]), parse_transform(transforms[2]));
        bv.cdf(x1_eval, x2_eval);
        bv.set_parameters(x1_new, x2_new, p_new);
        return bv.cdf(x1_eval, x2_eval);
    });

    // --- MultivariateNormal (deterministic methods) -------------------------------------
    // Stateless per-call style: a fresh instance is constructed from (mean, covariance) on
    // every call. Covers everything that does not touch the seeded MVNUNI stream (mean,
    // median, mode, sd, variance, covariance, pdf, log_pdf, cdf for dim 1-2, mahalanobis,
    // inverse_cdf, dimension, parameters_valid). Seeded CDF (dim>=3) / MVNDST batches need
    // a single persistent instance across several calls -- see mvn_cdf_seq/mvn_mvndst_seq
    // (added alongside the MVNDST port). random_value/lhs_value (args = [sample_size, seed,
    // row, col]) are ALSO stateless despite being "seeded" -- unlike MVNUNI,
    // GenerateRandomValues/LatinHypercubeRandomValues seed their own fresh MersenneTwister/
    // LatinHypercube draw from the `seed` argument every call (see fixtures/README.md's
    // Statefulness section).
    m.def("mvn_val", [](const std::string& method, const std::vector<double>& mean,
                         const std::vector<std::vector<double>>& covariance,
                         const std::vector<double>& args) {
        mvd::MultivariateNormal n(mean, covariance);
        if (method == "dimension") return static_cast<double>(n.dimension());
        if (method == "parameters_valid") return n.parameters_valid() ? 1.0 : 0.0;
        if (method == "mean") return n.mean()[static_cast<std::size_t>(args[0])];
        if (method == "median") return n.median()[static_cast<std::size_t>(args[0])];
        if (method == "mode") return n.mode()[static_cast<std::size_t>(args[0])];
        if (method == "sd") return n.standard_deviation()[static_cast<std::size_t>(args[0])];
        if (method == "variance") return n.variance()[static_cast<std::size_t>(args[0])];
        if (method == "covariance")
            return n.covariance(static_cast<int>(args[0]), static_cast<int>(args[1]));
        if (method == "pdf") return n.pdf(args);
        if (method == "log_pdf") return n.log_pdf(args);
        if (method == "cdf") return n.cdf(args);
        if (method == "mahalanobis") return n.mahalanobis(args);
        if (method == "inverse_cdf") {
            // args = [p_1..p_dim, index]
            std::vector<double> p(args.begin(), args.end() - 1);
            int idx = static_cast<int>(args.back());
            return n.inverse_cdf(p)[static_cast<std::size_t>(idx)];
        }
        if (method == "random_value") {
            auto sample = n.generate_random_values(static_cast<int>(args[0]), static_cast<int>(args[1]));
            return sample[static_cast<std::size_t>(args[2])][static_cast<std::size_t>(args[3])];
        }
        if (method == "lhs_value") {
            auto sample =
                n.latin_hypercube_random_values(static_cast<int>(args[0]), static_cast<int>(args[1]));
            return sample[static_cast<std::size_t>(args[2])][static_cast<std::size_t>(args[3])];
        }
        // v2.1.4: IsDensityValid is always true for a freshly-constructed (throwing-ctor)
        // instance -- the interesting false-after-a-failed-Try* transition is a stateful
        // sequence covered by core/tests/test_multivariate_normal_api.cpp instead.
        if (method == "is_density_valid") return n.is_density_valid() ? 1.0 : 0.0;
        if (method == "mvndst" || method == "mvndst_inform" || method == "mvndst_error") {
            // Stateless (unseeded) mvndst path -- exercised by the deterministic
            // status-code/collapse cases whose construct carries no "seed" (mvnuni_ is
            // never touched: no -1/finite-bound-count reaches DKBVRC for these). Same
            // args shape as the seeded batch's per-call arrays, flattened here to a plain
            // double vector by the Python side (_flatten_mv_args) -- args = [n_dim,
            // [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps];
            // re-derive the sub-vector boundaries from n_dim/nl.
            int n_dim = static_cast<int>(args[0]);
            int nl = n_dim * (n_dim - 1) / 2;
            std::size_t i = 1;
            std::vector<double> lower(args.begin() + static_cast<long>(i),
                                       args.begin() + static_cast<long>(i + static_cast<std::size_t>(n_dim)));
            i += static_cast<std::size_t>(n_dim);
            std::vector<double> upper(args.begin() + static_cast<long>(i),
                                       args.begin() + static_cast<long>(i + static_cast<std::size_t>(n_dim)));
            i += static_cast<std::size_t>(n_dim);
            std::vector<int> infin;
            for (std::size_t k = 0; k < static_cast<std::size_t>(n_dim); ++k)
                infin.push_back(static_cast<int>(args[i + k]));
            i += static_cast<std::size_t>(n_dim);
            std::vector<double> correl(args.begin() + static_cast<long>(i),
                                        args.begin() + static_cast<long>(i + static_cast<std::size_t>(nl)));
            i += static_cast<std::size_t>(nl);
            int maxpts = static_cast<int>(args[i]);
            double abseps = args[i + 1];
            double releps = args[i + 2];
            double error = 0, value = 0;
            int inform = 0;
            n.mvndst(n_dim, lower, upper, infin, correl, maxpts, abseps, releps, error, value, inform);
            if (method == "mvndst_inform") return static_cast<double>(inform);
            if (method == "mvndst_error") return error;
            return value;
        }
        throw py::value_error("unknown MultivariateNormal fixture method: " + method);
    });

    // --- MultivariateNormal (seeded batch methods) ---------------------------------------
    // `cdf` for dim>=3 draws from the seeded MVNUNI stream (via MVNDST); `mvndst` is the
    // Fortran-oracle entry point itself. Each needs a SINGLE persistent instance across a
    // whole run of k sequential calls -- rebuilding per call the way mvn_val does would
    // silently reset the seeded RNG between assertions. The Python fixture runner groups
    // consecutive same-method seeded assertions within a case and calls these once per run,
    // comparing results element-wise (see test_fixtures.py). `Interval` also consumes the
    // stream, but the corpus's only Interval case makes exactly one such call, so it delegates
    // through the grammar's `seed` key and needs no batch entry point here.
    m.def("mvn_cdf_seq", [](const std::vector<double>& mean,
                             const std::vector<std::vector<double>>& covariance, int seed,
                             const std::vector<std::vector<double>>& xs) {
        mvd::MultivariateNormal n(mean, covariance);
        n.set_mvnuni_seed(seed);
        std::vector<double> out;
        out.reserve(xs.size());
        for (const auto& x : xs) out.push_back(n.cdf(x));
        return out;
    });

    // n: the MVNDST `N` argument (also the dimension used to build the scratch instance
    // -- an identity-covariance MultivariateNormal(n), matching the Fortran oracle test's
    // `new MultivariateNormal(5)`). Each of lowers/uppers/infins/correls holds one
    // per-call vector; maxpts_v/abseps_v/releps_v are one scalar per call.
    m.def("mvn_mvndst_seq", [](int n, int seed, const std::vector<std::vector<double>>& lowers,
                                const std::vector<std::vector<double>>& uppers,
                                const std::vector<std::vector<int>>& infins,
                                const std::vector<std::vector<double>>& correls,
                                const std::vector<int>& maxpts_v, const std::vector<double>& abseps_v,
                                const std::vector<double>& releps_v) {
        mvd::MultivariateNormal mvn(n);
        mvn.set_mvnuni_seed(seed);
        std::vector<double> out;
        out.reserve(lowers.size());
        for (std::size_t c = 0; c < lowers.size(); ++c) {
            double error = 0, value = 0;
            int inform = 0;
            mvn.mvndst(n, lowers[c], uppers[c], infins[c], correls[c], maxpts_v[c], abseps_v[c],
                       releps_v[c], error, value, inform);
            out.push_back(value);
        }
        return out;
    });

    // --- MultivariateStudentT ------------------------------------------------------------
    // Stateless per-call style, mirroring mvn_val: a fresh instance is constructed from
    // (df, location, scale) on every call. Unlike MultivariateNormal, the CDF here (K=200
    // stratified chi-squared mixture) is fully deterministic -- no seeded MVNUNI stream, so
    // no mvt_*_seq batch entry point is needed. Methods: dimension, parameters_valid,
    // degrees_of_freedom, mean, median, mode, sd, variance, covariance, pdf, log_pdf, cdf,
    // mahalanobis, inverse_cdf, random_value, lhs_value (args = [sample_size, seed, row,
    // col]; stateless for the same reason as MultivariateNormal's -- see the comment above
    // mvn_val).
    m.def("mvt_val", [](const std::string& method, double df, const std::vector<double>& location,
                         const std::vector<std::vector<double>>& scale, const std::vector<double>& args) {
        mvd::MultivariateStudentT t(df, location, scale);
        if (method == "dimension") return static_cast<double>(t.dimension());
        if (method == "parameters_valid") return t.parameters_valid() ? 1.0 : 0.0;
        if (method == "degrees_of_freedom") return t.degrees_of_freedom();
        if (method == "mean") return t.mean()[static_cast<std::size_t>(args[0])];
        if (method == "median") return t.median()[static_cast<std::size_t>(args[0])];
        if (method == "mode") return t.mode()[static_cast<std::size_t>(args[0])];
        if (method == "sd") return t.standard_deviation()[static_cast<std::size_t>(args[0])];
        if (method == "variance") return t.variance()[static_cast<std::size_t>(args[0])];
        if (method == "covariance")
            return t.covariance(static_cast<int>(args[0]), static_cast<int>(args[1]));
        if (method == "pdf") return t.pdf(args);
        if (method == "log_pdf") return t.log_pdf(args);
        if (method == "cdf") return t.cdf(args);
        if (method == "mahalanobis") return t.mahalanobis(args);
        if (method == "inverse_cdf") {
            // args = [p_1..p_dim+1, index]
            std::vector<double> p(args.begin(), args.end() - 1);
            int idx = static_cast<int>(args.back());
            return t.inverse_cdf(p)[static_cast<std::size_t>(idx)];
        }
        if (method == "random_value") {
            auto sample = t.generate_random_values(static_cast<int>(args[0]), static_cast<int>(args[1]));
            return sample[static_cast<std::size_t>(args[2])][static_cast<std::size_t>(args[3])];
        }
        if (method == "lhs_value") {
            auto sample =
                t.latin_hypercube_random_values(static_cast<int>(args[0]), static_cast<int>(args[1]));
            return sample[static_cast<std::size_t>(args[2])][static_cast<std::size_t>(args[3])];
        }
        throw py::value_error("unknown MultivariateStudentT fixture method: " + method);
    });
}
