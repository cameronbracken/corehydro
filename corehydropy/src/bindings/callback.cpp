// pybind11 glue for the callback surface (corehydropy.callback). Unlike the toolbox verbs, which
// pass serializable data through toolbox_run, every verb here takes a live Python function, so it
// goes through callback_runner.hpp and its guard. Mirrors corehydror's src/callback.cpp call for
// call.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
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

// Worded exactly as corehydror's R checks are, so a caller cannot tell which language refused.
const char* kBadCount = "`n` must be a single positive whole number";
const char* kBadBounds = "`min` and `max` must each be a single finite whole number";

// Wraps `f` as the (parameters, generator) callback the rng/mcmc/bootstrap groups call. The scope
// is a local of this lambda, so the handle it hands Python is invalidated the moment the callback
// returns -- by a normal return or by an unwind, since a destructor runs either way.
std::function<std::vector<double>(const std::vector<double>&, samp::MersenneTwister&)> as_rng_fn(
    py::function f) {
    return [f](const std::vector<double>& p, samp::MersenneTwister& prng) {
        sup::RngScope scope(prng);
        py::object out = f(p, RngHandle(scope.handle()));
        try {
            return out.cast<std::vector<double>>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the function must return a sequence of numbers; got " +
                                     std::string(py::str(out)));
        }
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
    m.def(
        "callback_mcmc",
        [](const std::string& options_json, py::function f) {
            sup::CallbackSet cbs;
            cbs.vector_scalar = as_vector_scalar_fn(f);
            return pack(sup::run_callback("mcmc", "sample", options_json, cbs));
        },
        py::arg("options_json"), py::arg("f"));

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
