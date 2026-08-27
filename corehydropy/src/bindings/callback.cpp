// pybind11 glue for the callback surface (corehydropy.callback). Unlike the toolbox verbs, which
// pass serializable data through toolbox_run, every verb here takes a live Python function, so it
// goes through callback_runner.hpp and its guard. Mirrors corehydror's src/callback.cpp call for
// call.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings.hpp"
#include "corehydro/numerics/support/callback_runner.hpp"
#include "corehydro/numerics/support/rng_handle.hpp"

namespace py = pybind11;
namespace sup = corehydro::numerics::support;
namespace samp = corehydro::numerics::sampling;

namespace {

// --- the RNG handle -------------------------------------------------------------------------
//
// The generator a callback draws from is owned by the C++ frame driving it, so what Python gets is
// an object holding a BORROW (numerics/support/rng_handle.hpp) rather than the generator itself. A
// Python object outlives the C++ frame that made it -- a user may bind it to a name, close over it,
// or stash it in a list -- so the borrow's flag, cleared by RngScope's destructor, is the only
// thing standing between that and a read of freed memory. Every draw goes through the borrow's
// require(), which checks the flag before the generator is touched, and raises RuntimeError (with
// the same sentence corehydror raises) when it is clear.
//
// Kept a thin wrapper rather than binding RngBorrow itself so the class Python sees has exactly
// two methods, no constructor, and no way to reach the raw pointer.
class RngHandle {
   public:
    explicit RngHandle(sup::RngBorrowPtr borrow) : borrow_(std::move(borrow)) {}

    std::vector<double> uniform(int n) const { return borrow_->uniform(n); }
    std::vector<int> integers(int n, int min_inclusive, int max_exclusive) const {
        return borrow_->integers(n, min_inclusive, max_exclusive);
    }
    bool valid() const { return borrow_ && borrow_->valid; }

   private:
    sup::RngBorrowPtr borrow_;
};

// The whole-number argument check, so the two packages refuse the SAME values for the same
// reasons. Left to their defaults they did not: pybind11 rejects any float where an `int` is
// declared (so `rng.uniform(3.0)` failed, though every number R hands a callback is a double),
// while R's own `rng_check_count` silently truncated `n = 2.7` to 2. Neither half of that is
// defensible in a pair of packages whose whole promise is that the same call means the same thing
// in both. The rule now is one sentence in both languages: a whole number is accepted however it
// is spelled (3 or 3.0), and anything with a fraction is refused by the name of the argument.
int as_whole_number(const py::object& v, const char* message) {
    // The accepted magnitude is |value| <= 2147483647, NOT the full int32 range. R's own
    // rng_is_whole() tests `abs(x) <= .Machine$integer.max`, which is 2147483647, so R refuses
    // -2147483648 -- and Python used to accept it, making `integers(1, -2147483648, -2147483000)`
    // a legal call in one package and an error in the other. The bound is symmetric here for
    // exactly that reason, and it applies to the __index__ path below as well as the float one:
    // a Python `int` of -2147483648 casts to `int` without complaint and would otherwise slip
    // past the float check that catches the same value written as a literal -2147483648.0.
    constexpr double kIntMin = -2147483647.0, kIntMax = 2147483647.0;
    if (py::isinstance<py::bool_>(v)) throw py::type_error(message);
    if (py::isinstance<py::float_>(v)) {
        const double d = v.cast<double>();
        if (!std::isfinite(d) || d != std::trunc(d) || d < kIntMin || d > kIntMax)
            throw py::type_error(message);
        return static_cast<int>(d);
    }
    if (!PyIndex_Check(v.ptr())) throw py::type_error(message);
    int out;
    try {
        out = v.cast<int>();
    } catch (const py::cast_error&) {
        throw py::type_error(message);
    }
    if (static_cast<double>(out) < kIntMin) throw py::type_error(message);
    return out;
}

// A callback's return read as a sequence of numbers, ACCEPTING A BARE NUMBER as the length-one
// sequence it stands for. R has no scalar type -- `acc / length(data)` there IS a numeric of
// length one -- so without this fallback every length-one return has to be written `[x]` in Python
// and `x` in R, and the two packages' own documented examples cannot be transliterations of each
// other. (corehydror's `bootstrap_custom` example fits the mean with `acc / length(data)`; the
// Python docstring had to spell the same line `[acc / len(data)]` to work at all.) The precedent is
// row_major_matrix() below, which has always accepted a bare number as the 1 x 1 matrix a
// one-moment-condition model has.
//
// A numpy 0-d array and a numpy scalar both reach the `double` cast, so they are accepted too. A
// string is not: pybind refuses `str -> double`, and the message names what came back.
std::vector<double> as_number_sequence(const py::object& out, const std::string& message) {
    try {
        return out.cast<std::vector<double>>();
    } catch (const py::cast_error&) {
    }
    // Deliberately NOT `py::isinstance<py::float_>`: an int, a numpy float64 and a 0-d array all
    // have to work, and each of them casts to double while none of them is a Python float.
    try {
        return std::vector<double>{out.cast<double>()};
    } catch (const py::cast_error&) {
    }
    throw std::runtime_error(message + "; got " + std::string(py::str(out)));
}

// Worded exactly as corehydror's R checks are, so a caller cannot tell which language refused.
const char* kBadCount = "`n` must be a single positive whole number";
const char* kBadBounds = "`min` and `max` must each be a single finite whole number";

// Wraps `f` as the (parameters, generator) callback the rng/mcmc/bootstrap groups call. The scope
// is a local of this lambda, so the handle it hands Python is invalidated the moment the callback
// returns -- by a normal return or by an unwind, since a destructor runs either way.
//
// Review fix (Task 5, finding 1): refuses a nan element by name, same wording as_vector_vector_fn
// uses, for the same reason -- a Gibbs proposal (the caller this shape serves today) that returns
// nan would otherwise be accepted unconditionally by Gibbs and corrupt the chain silently rather
// than failing loudly. The check lives HERE rather than duplicated per call site because every
// caller of this shape (the Gibbs proposal today, a bootstrap resample in Task 6) shares the
// mistake.
std::function<std::vector<double>(const std::vector<double>&, samp::MersenneTwister&)> as_rng_fn(
    py::function f) {
    return [f](const std::vector<double>& p, samp::MersenneTwister& prng) {
        sup::RngScope scope(prng);
        py::object out = f(p, RngHandle(scope.handle()));
        std::vector<double> result =
            as_number_sequence(out, "the function must return a sequence of numbers");
        for (double value : result)
            if (std::isnan(value))
                throw std::runtime_error("the function returned nan rather than a number");
        return result;
    };
}

// Converts a Python callable into the scalar signature, raising the same "single number" error the
// R glue raises. That raise (a plain C++ exception, not a Python one) is funneled through
// GuardedCall exactly like a genuine Python-side exception (py::error_already_set), which latches
// it and lets run_callback rethrow it once the ported routine returns.
std::function<double(double)> as_scalar_fn(py::function f) {
    return [f](double x) -> double {
        py::object out = f(x);
        try {
            return out.cast<double>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the function must return a single number; got " +
                                     std::string(py::str(out)));
        }
    };
}

// Converts a Python callable of two arguments into the (x, y) -> z signature math/quadrature_2d
// takes (P2 "math extras"). Mirrors as_scalar_fn's cast and error wording.
std::function<double(double, double)> as_scalar_xy_fn(py::function f) {
    return [f](double x, double y) -> double {
        py::object out = f(x, y);
        try {
            return out.cast<double>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the function must return a single number; got " +
                                     std::string(py::str(out)));
        }
    };
}

// f(theta) -> vector, the shape the HMC/NUTS gradient callback and the Gibbs proposal take. The
// length is checked in the core (against the prior count, which is the only place that knows it);
// here the job is refusing something that is not a sequence of numbers at all, and refusing a nan
// element for the same reason as_vector_scalar_fn refuses a nan result: corehydror's twin
// (as_vector_vector_fn in corehydror/src/callback.cpp) rejects NA/NaN by name already, so without
// this check a gradient or proposal returning nan is a clear error in R and a silently garbage
// chain in Python. +/-inf is NOT refused here, matching as_vector_scalar_fn and the R twin -- an
// infinite element is not this callback's own error to raise.
std::function<std::vector<double>(const std::vector<double>&)> as_vector_vector_fn(py::function f) {
    return [f](const std::vector<double>& p) -> std::vector<double> {
        py::object out = f(p);
        std::vector<double> result =
            as_number_sequence(out, "the function must return a sequence of numbers");
        for (double value : result)
            if (std::isnan(value))
                throw std::runtime_error("the function returned nan rather than a number");
        return result;
    };
}

// Converts a Python callable of two arguments into the (x, weight) -> z signature
// math/quadrature_vegas takes (P2 "math extras"), upstream's own Vegas integrand shape -- `x` the
// sample point (a sequence of numbers) and `weight` the importance weight Vegas has already
// computed for it. Mirrors as_scalar_xy_fn's cast and error wording, with `x` marshaled as a
// std::vector<double> the way as_vector_scalar_fn's argument is.
std::function<double(const std::vector<double>&, double)> as_vector_weight_fn(py::function f) {
    return [f](const std::vector<double>& x, double weight) -> double {
        py::object out = f(x, weight);
        try {
            return out.cast<double>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the function must return a single number; got " +
                                     std::string(py::str(out)));
        }
    };
}

std::function<double(const std::vector<double>&)> as_vector_scalar_fn(py::function f) {
    return [f](const std::vector<double>& p) -> double {
        py::object out = f(p);
        double value;
        try {
            value = out.cast<double>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the function must return a single number; got " +
                                     std::string(py::str(out)));
        }
        // nan is refused; +/-inf is NOT. An infinite log-density is the ordinary way to say "this
        // parameter is impossible" and every sampler treats it as a rejected point, but a nan is
        // never an answer. corehydror's as_vector_scalar_fn refuses NA and NaN the same way, which
        // is what keeps a mistake from being a clear error in one package and a silently
        // motionless chain in the other.
        if (std::isnan(value))
            throw std::runtime_error("the function returned nan rather than a number");
        return value;
    };
}

// --- the bootstrap delegates ----------------------------------------------------------------
//
// nan IS TREATED DIFFERENTLY BY THE TWO SHAPES BELOW, on purpose, and corehydror's glue draws the
// same line for the same reason.
//
//   - A delegate returning DATA (resample, jackknife) refuses nan by name. The ported Bootstrap has
//     no finiteness check on data at all, so a nan -- R's answer for an index past the end of a
//     vector, and numpy's for plenty of ordinary mistakes -- would sail into the user's OWN fit
//     function and surface there, two calls from where it was made.
//   - A delegate returning NUMBERS the ported class tests for itself (fit, statistic) passes nan
//     through untouched. `HasExpectedFiniteParameterValues` and `ValidateStatistics` exist
//     precisely to read a non-finite value as "this replicate failed" and retry it; refusing it
//     here would steal a documented upstream behaviour and turn a legitimately hard resample into
//     a hard error.
//
// Both wrappers below build the handle the way as_rng_fn does -- an RngScope local to the lambda,
// so the handle is invalidated the moment the callback returns, by a normal return or by an
// unwind. It is copied rather than shared because upstream's resample delegate is a THREE-argument
// shape, `Func<TData, ParameterSet, Random, TData>`, and as_rng_fn's is (parameters, generator).
// A bare number is accepted as the one-element sequence it stands for -- see as_number_sequence()
// above on why, and on corehydror's own documented `fit` returning `acc / length(data)`.
std::vector<double> as_number_list(const py::object& out, const char* what) {
    return as_number_sequence(out, "the " + std::string(what) +
                                       " function must return a sequence of numbers");
}

// f(data, parameters, rng) -> data. Upstream's `Func<TData, ParameterSet, Random, TData>`.
std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&,
                                  samp::MersenneTwister&)>
as_resample_fn(py::function f) {
    return [f](const std::vector<double>& data, const std::vector<double>& p,
               samp::MersenneTwister& prng) {
        sup::RngScope scope(prng);
        py::object out = f(data, p, RngHandle(scope.handle()));
        std::vector<double> result = as_number_list(out, "resample");
        for (double value : result)
            if (std::isnan(value))
                throw std::runtime_error(
                    "the resample function returned nan rather than a number; rng.integers() draws "
                    "on [0, n), which is already Python's own subscript base");
        return result;
    };
}

// f(data, index) -> data. Upstream's `Func<TData, int, TData>`, called with `index` counting from
// 0 (both packages say so by name).
std::function<std::vector<double>(const std::vector<double>&, int)> as_jackknife_fn(
    py::function f) {
    return [f](const std::vector<double>& data, int index) {
        py::object out = f(data, index);
        std::vector<double> result = as_number_list(out, "jackknife");
        for (double value : result)
            if (std::isnan(value))
                throw std::runtime_error(
                    "the jackknife function returned nan rather than a number; `index` counts from "
                    "0, so the sample without it is data[:index] + data[index + 1:]");
        return result;
    };
}

// f(x) -> numbers, the shape both the fit (data -> parameters) and the statistic (parameters ->
// statistics) take. nan passes through: see the note above.
std::function<std::vector<double>(const std::vector<double>&)> as_numbers_fn(py::function f,
                                                                            const char* what) {
    return [f, what](const std::vector<double>& x) { return as_number_list(f(x), what); };
}

// f(data) -> (parameters, covariance). Upstream's `Func<TData, BootstrapFit>
// FitWithCovarianceFunction`, the delegate the PIVOTAL run type fits through. Defined below
// row_major_matrix(), which it shares with the gmm moment function; declared here so it sits with
// the other bootstrap delegates.
//
// nan PASSES THROUGH BOTH HALVES, unlike the gmm weighting matrix, and the file's own rule says
// why: `IsValidFit` tests the parameters AND the covariance for finiteness itself and reads a
// non-finite one as a failed replicate to retry. Refusing it here would steal that behaviour.
std::function<sup::FitWithCovarianceReturn(const std::vector<double>&)> as_fit_with_covariance_fn(
    py::function f);

// --- the gmm delegates ------------------------------------------------------------------------
//
// The moment condition function returns TWO things at once -- upstream's `(Vector G, Matrix S)`
// tuple -- so the return shape is the mistake this surface invites, and it is checked here by name
// rather than left to produce a covariance matrix quietly built out of a moment vector. The Python
// spelling is the tuple `(g, s)` or a dict with keys `g` and `s`; corehydror accepts a list with
// those two names and raises the same sentence.
//
// nan IS TREATED DIFFERENTLY BY THE PIECES BELOW, on purpose, drawn by what each value MEANS to
// the ported estimator (the same rule the bootstrap delegates above follow, and corehydror's glue
// draws it identically):
//
//   - `g` and the penalty pass nan through. `Q()` ends with `is_finite(qv) ? qv : double.max`: a
//     non-finite quadratic form is the ported estimator's own way of learning that a corner of the
//     parameter box is infeasible, and refusing it here would turn a hard trial point into a hard
//     error.
//   - `s` and the jacobian refuse it by name. Nothing in the ported class tests either for
//     finiteness, so a nan there flows silently into the weighting matrix or the sandwich
//     covariance and comes back as a fitted number with no warning attached to it.
const char* kMomentShape =
    "the moment condition function must return the tuple (g, s) -- or a dict with keys 'g' and "
    "'s' -- where 'g' is the moment vector and 's' is the weighting matrix";

// A matrix flattened ROW-MAJOR, with its shape. A sequence of rows (a list of lists, or a 2-D
// numpy array) is the ordinary spelling and is already row-major. A bare number, or a sequence
// holding exactly one, is accepted as the 1 x 1 matrix a one-moment-condition model has --
// corehydror accepts a length-1 numeric with no `dim` for the same case. Anything else is refused
// naming the element it came from.
//
// `allow_nan` is the one axis callers differ on, and it is drawn by what the value MEANS to the
// ported class rather than by taste: the gmm weighting matrix and jacobian are tested for
// finiteness by NOTHING downstream (so a nan reaches the fitted standard errors in silence and is
// refused here), while the pivotal bootstrap's covariance is tested by `IsValidFit`, which reads a
// non-finite one as a failed replicate to retry (so refusing it here would steal that behaviour).
std::pair<std::vector<double>, std::vector<int>> row_major_matrix(const py::object& value,
                                                                 const std::string& what,
                                                                 bool allow_nan = false) {
    std::vector<std::vector<double>> rows;
    bool nested = true;
    try {
        rows = value.cast<std::vector<std::vector<double>>>();
    } catch (const py::cast_error&) {
        nested = false;
    }
    if (!nested) {
        std::vector<double> flat;
        try {
            flat = value.cast<std::vector<double>>();
        } catch (const py::cast_error&) {
            try {
                flat = {value.cast<double>()};
            } catch (const py::cast_error&) {
                throw std::runtime_error(what + " must be a matrix -- a sequence of rows; got " +
                                         std::string(py::str(value)));
            }
        }
        if (flat.size() != 1)
            throw std::runtime_error(
                what + " must be a matrix -- a sequence of ROWS, not a flat sequence of " +
                std::to_string(flat.size()) + " numbers");
        rows = {flat};
    }
    if (rows.empty()) throw std::runtime_error(what + " must have at least one row; it was empty");
    const std::size_t cols = rows[0].size();
    std::vector<double> out;
    out.reserve(rows.size() * cols);
    for (const std::vector<double>& row : rows) {
        if (row.size() != cols)
            throw std::runtime_error(what + " must be rectangular; its first row holds " +
                                     std::to_string(cols) + " values and another holds " +
                                     std::to_string(row.size()));
        for (double x : row) {
            if (std::isnan(x) && !allow_nan)
                throw std::runtime_error(what +
                                         " returned nan rather than a number; nothing in the "
                                         "estimator checks it, so it would reach the fitted "
                                         "standard errors without a word");
            out.push_back(x);
        }
    }
    return {out, std::vector<int>{static_cast<int>(rows.size()), static_cast<int>(cols)}};
}

// The definition of the bootstrap delegate declared above, placed here because row_major_matrix()
// lives between the two groups. The two names are checked BY NAME for the same reason the moment
// function's are: returning the wrong thing is the mistake this shape invites, and a bare sequence
// would otherwise be read as parameters with no covariance at all. The dict form and the tuple
// `(parameters, covariance)` are both accepted, exactly as the moment function accepts both, and
// corehydror takes a list with the same two names.
const char* kFitCovarianceShape =
    "the fit_with_covariance function must return the tuple (parameters, covariance) -- or a dict "
    "with keys 'parameters' and 'covariance' -- where 'parameters' is the fitted vector and "
    "'covariance' is its covariance matrix";

std::function<sup::FitWithCovarianceReturn(const std::vector<double>&)> as_fit_with_covariance_fn(
    py::function f) {
    return [f](const std::vector<double>& data) {
        py::object out = f(data);
        py::object parameters, covariance;
        if (py::isinstance<py::dict>(out)) {
            py::dict d = out.cast<py::dict>();
            if (!d.contains("parameters") || !d.contains("covariance"))
                throw std::runtime_error(kFitCovarianceShape);
            parameters = d["parameters"];
            covariance = d["covariance"];
        } else {
            // PySequence_Check rather than a cast, for the reason the moment function gives.
            if (!PySequence_Check(out.ptr())) throw std::runtime_error(kFitCovarianceShape);
            py::sequence pair = out.cast<py::sequence>();
            if (py::len(pair) != 2) throw std::runtime_error(kFitCovarianceShape);
            parameters = pair[0];
            covariance = pair[1];
            // THE ONE AMBIGUOUS RETURN on this surface, refused rather than guessed at, exactly as
            // the moment function refuses its own: `[mu, sigma]` -- the likeliest mistake, a
            // two-parameter fit written without its covariance -- is indistinguishable from the
            // pair (parameters, covariance) of a ONE-parameter model, both being two bare numbers.
            // Reading the mistake as a one-parameter fit would bootstrap a different model in
            // silence. R cannot reach this at all: its fit returns a NAMED list, and a bare
            // `c(mu, sigma)` is refused by the same sentence.
            if (!PySequence_Check(parameters.ptr()) && !PySequence_Check(covariance.ptr()))
                throw std::runtime_error(
                    std::string(kFitCovarianceShape) +
                    ". Both came back as bare numbers, which cannot be told apart from a fitted "
                    "vector like [mu, sigma] returned without its covariance: write 'parameters' "
                    "as a one-element sequence, or return the dict, which names them");
        }

        sup::FitWithCovarianceReturn result;
        // nan passes: see the note above.
        result.parameters = as_number_sequence(
            parameters,
            "the fit_with_covariance function's 'parameters' must be a sequence of numbers");
        auto cov = row_major_matrix(covariance, "the fit_with_covariance function's 'covariance'",
                                    /*allow_nan=*/true);
        result.covariance = cov.first;
        result.rows = cov.second[0];
        result.cols = cov.second[1];
        return result;
    };
}

// f(parameters) -> (g, s). Upstream's
// `(Vector G, Matrix S) MomentConditionFunction(double[] parameters)`.
std::function<sup::MomentConditionReturn(const std::vector<double>&)> as_moment_conditions_fn(
    py::function f) {
    return [f](const std::vector<double>& p) {
        py::object out = f(p);
        py::object g_object, s_object;
        if (py::isinstance<py::dict>(out)) {
            py::dict d = out.cast<py::dict>();
            if (!d.contains("g") || !d.contains("s")) throw std::runtime_error(kMomentShape);
            g_object = d["g"];
            s_object = d["s"];
        } else {
            // PySequence_Check rather than a cast: `out.cast<py::sequence>()` raises pybind11's own
            // TypeError for a non-sequence, which would escape past the check below and reach the
            // user as "Object of type 'float' is not an instance of 'sequence'" instead of the one
            // sentence that names both elements.
            if (!PySequence_Check(out.ptr())) throw std::runtime_error(kMomentShape);
            py::sequence pair = out.cast<py::sequence>();
            if (py::len(pair) != 2) throw std::runtime_error(kMomentShape);
            g_object = pair[0];
            s_object = pair[1];
            // THE ONE AMBIGUOUS RETURN on this surface, refused rather than guessed at. Both `g`
            // and `s` may be written as bare numbers -- `g` because R's `c(x)` is a bare number
            // there too, `s` because row_major_matrix() reads one as the 1 x 1 matrix a
            // one-moment-condition model has -- so `(number, number)` could be a q == 1 model, or
            // it could be `return [g0, g1]`, the likeliest mistake when q == 2, where a flat pair
            // of numbers looks like just the moment vector. Nothing downstream can tell them
            // apart, and reading the mistake as a q == 1 model would fit a different model in
            // silence. The dict form names both elements and is never ambiguous; so is
            // `([g0], s)`. R cannot reach this at all: its moment function returns a NAMED list,
            // and a bare `c(g0, g1)` is refused by the same sentence.
            if (!PySequence_Check(g_object.ptr()) && !PySequence_Check(s_object.ptr()))
                throw std::runtime_error(
                    std::string(kMomentShape) +
                    ". Both came back as bare numbers, which cannot be told apart from a flat "
                    "moment vector like [g0, g1]: write 'g' as a one-element sequence, or return "
                    "the dict, which names them");
        }

        sup::MomentConditionReturn result;
        // nan passes: see the note above. A bare number is the one-element moment vector a
        // one-moment-condition model has, exactly as corehydror accepts a length-1 numeric.
        result.g = as_number_sequence(
            g_object,
            "the moment condition function's 'g' (the moment vector) must be a sequence of numbers");
        auto s = row_major_matrix(
            s_object, "the moment condition function's 's' (the weighting matrix)");
        result.s = s.first;
        result.s_rows = s.second[0];
        result.s_cols = s.second[1];
        return result;
    };
}

// f(parameters) -> a q x p matrix. Upstream's `double[,] JacobianFunction(double[])`.
std::function<std::pair<std::vector<double>, std::vector<int>>(const std::vector<double>&)>
as_matrix_fn(py::function f, std::string what) {
    return [f, what](const std::vector<double>& p) { return row_major_matrix(f(p), what); };
}

// f(parameters) -> one number. The shape as_vector_scalar_fn already serves -- except that this one
// must let nan through (see the note above the gmm delegates), so it is its own small lambda rather
// than a reuse that would quietly refuse it.
std::function<double(const std::vector<double>&)> as_penalty_fn(py::function f) {
    return [f](const std::vector<double>& p) -> double {
        py::object out = f(p);
        try {
            return out.cast<double>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the penalty function must return a single number; got " +
                                     std::string(py::str(out)));
        }
    };
}

py::dict pack(const sup::CallbackResult& r) {
    py::dict out;
    out["values"] = r.values;
    out["names"] = r.names;
    out["dims"] = r.dims;
    out["status"] = r.status;
    return out;
}

}  // namespace

void register_callback(py::module_& m) {
    // Runs one method of the callback runner's "math" group against a Python function. The scalar
    // methods (root_find, derivative, quadrature) call `f` with one number; the vector methods
    // (gradient, hessian) call it with the whole point as a list. Either way `f` must return a
    // single number.
    m.def(
        "callback_math",
        [](const std::string& method, const std::string& options_json, py::function f) {
            sup::CallbackSet cbs;
            if (method == "root_find" || method == "derivative" || method == "quadrature")
                cbs.scalar = as_scalar_fn(f);
            else
                cbs.vector_scalar = as_vector_scalar_fn(f);
            return pack(sup::run_callback("math", method, options_json, cbs));
        },
        py::arg("method"), py::arg("options_json"), py::arg("f"));

    // The two-callback half of the math group (P2 "math extras"): "root_find_newton" (f, its
    // analytic derivative df) and "root_find_system" (F, its Jacobian J). Split from
    // "callback_math" above because pybind11 arguments are fixed too -- every other math method
    // takes exactly one Python function, and these two need a second. `f`/`g` play different roles
    // by method: for "root_find_newton" they are the scalar function and its scalar derivative
    // (`cbs.scalar`/`cbs.scalar_deriv`); for "root_find_system" they are the vector-valued system
    // function and its Jacobian (`cbs.vector_vector`/`cbs.vector_matrix`, the same shapes the mcmc
    // gradient and gmm jacobian callbacks already use).
    m.def(
        "callback_math2",
        [](const std::string& method, const std::string& options_json, py::function f,
           py::function g) {
            sup::CallbackSet cbs;
            if (method == "root_find_newton") {
                cbs.scalar = as_scalar_fn(f);
                cbs.scalar_deriv = as_scalar_fn(g);
            } else if (method == "root_find_system") {
                cbs.vector_vector = as_vector_vector_fn(f);
                cbs.vector_matrix = as_matrix_fn(g, "the jacobian function");
            } else {
                throw std::invalid_argument("unknown two-callback math method: " + method);
            }
            return pack(sup::run_callback("math", method, options_json, cbs));
        },
        py::arg("method"), py::arg("options_json"), py::arg("f"), py::arg("g"));

    // The (x, y) half of the math group (P2 "math extras"): "quadrature_2d" and "ode_solve",
    // split from "callback_math" above for the same reason "callback_math2" is -- the arity
    // differs from every other math method's. `f` marshals through as_scalar_xy_fn into
    // `cbs.scalar_xy`; "ode_solve" reuses the same shape with `t` playing `x`'s role (see
    // callback/math.hpp's ode_solve arm).
    m.def(
        "callback_math_xy",
        [](const std::string& method, const std::string& options_json, py::function f) {
            sup::CallbackSet cbs;
            if (method == "quadrature_2d" || method == "ode_solve") {
                cbs.scalar_xy = as_scalar_xy_fn(f);
            } else {
                throw std::invalid_argument("unknown xy-callback math method: " + method);
            }
            return pack(sup::run_callback("math", method, options_json, cbs));
        },
        py::arg("method"), py::arg("options_json"), py::arg("f"));

    // The (x, weight) half of the math group (P2 "math extras"): "quadrature_vegas" alone, split
    // from "callback_math" for the same reason "callback_math_xy" is -- the callback shape differs
    // from every other math method's. `f` marshals through as_vector_weight_fn into
    // `cbs.vector_weight`.
    m.def(
        "callback_math_vw",
        [](const std::string& method, const std::string& options_json, py::function f) {
            sup::CallbackSet cbs;
            if (method == "quadrature_vegas") {
                cbs.vector_weight = as_vector_weight_fn(f);
            } else {
                throw std::invalid_argument("unknown vector-weight-callback math method: " + method);
            }
            return pack(sup::run_callback("math", method, options_json, cbs));
        },
        py::arg("method"), py::arg("options_json"), py::arg("f"));

    // Runs the callback runner's "mcmc" group against a Python log-likelihood: `f` is called with
    // the whole parameter vector as a list and must return a single number, exactly as upstream's
    // own `LogLikelihood` delegate does. The flat result is the layout documented in
    // numerics/support/callback/mcmc.hpp; corehydropy.callback's mcmc_posterior() slices it back
    // into the dict mcmc_sample() returns.
    //
    // INTERRUPTS, measured rather than assumed (Task 4). Ctrl-C during a long chain returns
    // control with a KeyboardInterrupt: CPython raises it at the `f(p)` call above, pybind11 turns
    // it into a py::error_already_set, and GuardedCall latches it for run_callback to rethrow --
    // the same path a genuine Python exception takes, so no extra check is needed here. It is not
    // instant, though, and that is worth knowing before reporting a hang: the ported samplers have
    // no cancellation hook (the C# CancellationTokenSource is a documented omission of this port),
    // so after the latch the chain still runs to the end of its loop with every remaining point
    // rejected -- without re-entering Python -- before the exception surfaces. Measured on a
    // 200,000-iteration single chain at the default thinning: 0.3 seconds from the signal to the
    // KeyboardInterrupt reaching the caller. corehydror's src/callback.cpp header carries the R
    // half of the same measurement.
    //
    // `proposal` and `gradient` are the two other delegates upstream's samplers take, and either
    // may be None: `proposal` is Gibbs's conditional draw, called with (parameters, rng) where
    // `rng` is a handle on THIS chain's generator, and `gradient` is HMC/NUTS's optional analytic
    // gradient, called with the parameter list. A None gradient leaves the ported
    // finite-difference default in force; which sampler may take which is enforced in the core,
    // once, for all four runners.
    m.def(
        "callback_mcmc",
        [](const std::string& options_json, py::function f, py::object proposal,
           py::object gradient) {
            sup::CallbackSet cbs;
            cbs.vector_scalar = as_vector_scalar_fn(f);
            // The proposal takes the SAME (parameters, generator) shape the rng group's probe
            // takes, so it is wrapped by the same as_rng_fn -- there is one place a handle is
            // handed to Python, not two.
            if (!proposal.is_none()) cbs.vector_rng = as_rng_fn(proposal.cast<py::function>());
            if (!gradient.is_none())
                cbs.vector_vector = as_vector_vector_fn(gradient.cast<py::function>());
            return pack(sup::run_callback("mcmc", "sample", options_json, cbs));
        },
        py::arg("options_json"), py::arg("f"), py::arg("proposal") = py::none(),
        py::arg("gradient") = py::none());

    // Runs the callback runner's "bootstrap" group against the delegates upstream's
    // `Bootstrap<TData>` takes as public properties: `resample(data, parameters, rng)`,
    // `fit(data)`, `statistic(parameters)`, `jackknife(data, index)` for the BCa method alone, and
    // `fit_with_covariance(data)` for the pivotal run type alone. `resample` and `statistic` are
    // always required; which of `fit` and `fit_with_covariance` is required is decided by the
    // options' own `run_type`, in the core, once for all four runners -- so both arrive as
    // py::object and either may be None here. The flat result is the layout documented in
    // numerics/support/callback/bootstrap.hpp;
    // corehydropy.callback's bootstrap_custom() reads it back by name.
    //
    // The resample delegate is handed a handle on the replicate's own generator, so a seeded run
    // stays reproducible and agrees with the identical run in R. INTERRUPTS behave exactly as the
    // mcmc note above describes, and for the same reason.
    m.def(
        "callback_bootstrap",
        [](const std::string& options_json, py::function resample, py::object fit,
           py::function statistic, py::object jackknife, py::object fit_with_covariance) {
            sup::CallbackSet cbs;
            cbs.data_rng = as_resample_fn(resample);
            if (!fit.is_none()) cbs.data_vector = as_numbers_fn(fit.cast<py::function>(), "fit");
            cbs.vector_vector = as_numbers_fn(statistic, "statistic");
            if (!jackknife.is_none())
                cbs.data_index = as_jackknife_fn(jackknife.cast<py::function>());
            if (!fit_with_covariance.is_none())
                cbs.data_covariance =
                    as_fit_with_covariance_fn(fit_with_covariance.cast<py::function>());
            return pack(sup::run_callback("bootstrap", "run", options_json, cbs));
        },
        py::arg("options_json"), py::arg("resample"), py::arg("fit"), py::arg("statistic"),
        py::arg("jackknife") = py::none(), py::arg("fit_with_covariance") = py::none());

    // Runs the callback runner's "gmm" group against the delegates upstream's
    // `GeneralizedMethodOfMoments` takes in its delegate-based constructor (C# 143):
    // `moment_conditions(parameters)`, returning the tuple `(g, s)` -- the moment vector and the
    // weighting matrix -- plus the optional `jacobian(parameters)` (a q x p matrix) and
    // `penalty(parameters)` (one number). Either optional argument may be None. The flat result is
    // the layout documented in numerics/support/callback/gmm.hpp; corehydropy.callback's
    // fit_gmm_moments() reads it back by name into the same Fit object fit_gmm() returns.
    //
    // There is no generator on this surface and no RNG anywhere in the fit, so a repeated call
    // returns the identical numbers. INTERRUPTS behave exactly as the mcmc note above describes,
    // and for the same reason.
    m.def(
        "callback_gmm",
        [](const std::string& options_json, py::function moment_conditions, py::object jacobian,
           py::object penalty) {
            sup::CallbackSet cbs;
            cbs.moment_conditions = as_moment_conditions_fn(moment_conditions);
            if (!jacobian.is_none())
                cbs.vector_matrix =
                    as_matrix_fn(jacobian.cast<py::function>(), "the jacobian function");
            if (!penalty.is_none())
                cbs.vector_scalar = as_penalty_fn(penalty.cast<py::function>());
            return pack(sup::run_callback("gmm", "fit", options_json, cbs));
        },
        py::arg("options_json"), py::arg("moment_conditions"), py::arg("jacobian") = py::none(),
        py::arg("penalty") = py::none());

    // The handle a callback is given on the core's seeded generator. No constructor is exposed:
    // it is handed out by the runner and is only usable for the duration of the one call it was
    // handed to (see RngHandle above). `corehydropy.callback` re-exports it as `Rng` so a user can
    // type-annotate a proposal or resample function.
    py::class_<RngHandle>(m, "Rng", R"doc(
The seeded random number generator a corehydro callback is handed.

Draw from this, not from `random` or `numpy.random`, for every random number a callback needs: it
is the same Mersenne Twister the core seeded, so the run stays reproducible from its seed and
agrees value for value with the identical run in R.

The handle borrows the generator for the duration of the one call it was given to. It is not an
object to keep. Storing it and drawing from it after your callback has returned raises
`RuntimeError`, which is deliberate: the generator it pointed at no longer exists, and reading it
would crash the interpreter rather than merely misbehave.

Examples
--------
The Gibbs proposal of :func:`corehydropy.mcmc_posterior` is the verb that hands you one. This
model's full conditional really is uniform: with x_i ~ Uniform(mu - 1, mu + 1) and a flat prior,
mu given the data is Uniform(max(x) - 1, min(x) + 1), so one uniform draw IS the Gibbs step.

>>> import corehydropy as ch
>>> x = [4.9, 5.1, 5.0, 5.2, 4.8]
>>> def ll(p):
...     return 0.0 if all(abs(xi - p[0]) <= 1.0 for xi in x) else float("-inf")
>>> def proposal(parameters, rng):
...     lo, hi = max(x) - 1.0, min(x) + 1.0
...     return [lo + rng.uniform(1)[0] * (hi - lo)]
>>> fit = ch.mcmc_posterior(ll, ch.Distribution("Uniform", [0.0, 10.0]),
...                         sampler="Gibbs", proposal=proposal,
...                         iterations=200, seed=12345, initialize="Randomize")
>>> round(float(fit["posterior_mean"][0]), 1)
5.0

`rng.integers(n, min, max)` draws whole numbers on [min, max) off the same stream -- a resampling
proposal, say, picking one of the observations by index.

>>> def pick_one(parameters, rng):
...     return [x[rng.integers(1, 0, len(x))[0]]]
)doc")
        .def(
            "uniform",
            [](const RngHandle& h, const py::object& n) {
                return h.uniform(as_whole_number(n, kBadCount));
            },
            py::arg("n"), "n draws on [0, 1) (the ported C# MersenneTwister.NextDouble).")
        .def(
            "integers",
            [](const RngHandle& h, const py::object& n, const py::object& min,
               const py::object& max) {
                return h.integers(as_whole_number(n, kBadCount), as_whole_number(min, kBadBounds),
                                  as_whole_number(max, kBadBounds));
            },
            py::arg("n"), py::arg("min"), py::arg("max"),
            "n draws on [min, max) -- the upper bound is EXCLUDED, as in the ported C# "
            "MersenneTwister.Next(minInclusive, maxExclusive). `min` and `max` must be whole "
            "numbers no more than 2147483647 apart (the ported generator draws an int32 span, and "
            "C# throws for a wider one).")
        .def("__repr__", [](const RngHandle& h) {
            return std::string("<corehydropy.Rng ") + (h.valid() ? "live" : "expired") + ">";
        });

    // Seeds a generator, hands `f` a handle on it together with the `parameters` vector from the
    // options, and returns what `f` drew. Internal: it is how corehydropy.callback's _rng_probe()
    // and the fixture runner reach the rng group, and the shape it drives -- a callback of
    // (parameters, generator) -- is the Gibbs proposal's, so what it proves about the handle
    // carries over to the samplers that will use it.
    m.def(
        "rng_probe",
        [](const std::string& options_json, py::function f) {
            sup::CallbackSet cbs;
            cbs.vector_rng = as_rng_fn(f);
            return pack(sup::run_callback("rng", "probe", options_json, cbs));
        },
        py::arg("options_json"), py::arg("f"));
}
