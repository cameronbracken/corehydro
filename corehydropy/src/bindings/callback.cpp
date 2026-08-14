// pybind11 glue for the callback surface (corehydropy.callback). Unlike the toolbox verbs, which
// pass serializable data through toolbox_run, every verb here takes a live Python function, so it
// goes through callback_runner.hpp and its guard. Mirrors corehydror's src/callback.cpp call for
// call.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
        try {
            return out.cast<double>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the function must return a single number; got " +
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
        .def("uniform", &RngHandle::uniform, py::arg("n"),
             "n draws on [0, 1) (the ported C# MersenneTwister.NextDouble).")
        .def("integers", &RngHandle::integers, py::arg("n"), py::arg("min"), py::arg("max"),
             "n draws on [min, max) -- the upper bound is EXCLUDED, as in the ported C# "
             "MersenneTwister.Next(minInclusive, maxExclusive).")
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
