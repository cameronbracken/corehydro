// cpp11 glue for the callback surface (R/callback.R). Unlike the toolbox verbs, which pass
// serializable data through ch_toolbox_run_, every verb here takes a live R function, so it goes
// through callback_runner.hpp and its guard. Mirrors corehydropy's src/bindings/callback.cpp
// call for call.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/support/callback_runner.hpp"
#include "corehydro/numerics/support/rng_handle.hpp"

using namespace cpp11;
namespace sup = corehydro::numerics::support;
namespace samp = corehydro::numerics::sampling;

namespace {

// Converts an R closure into the scalar signature, raising the same "single number" error the
// optimizer glue raises. The throw travels back through GuardedCall, which latches it and lets
// run_callback rethrow it inside this same cpp11-protected frame.
std::function<double(double)> as_scalar_fn(function f) {
    return [f](double x) mutable -> double {
        sexp out = f(writable::doubles({x}));
        doubles v = as_doubles(out);
        if (v.size() != 1)
            throw std::runtime_error(
                "the function must return a single number; got a value of length " +
                std::to_string(static_cast<long long>(v.size())));
        return v[0];
    };
}

std::function<double(const std::vector<double>&)> as_vector_scalar_fn(function f) {
    return [f](const std::vector<double>& p) mutable -> double {
        writable::doubles par(static_cast<R_xlen_t>(p.size()));
        for (std::size_t i = 0; i < p.size(); ++i) par[static_cast<R_xlen_t>(i)] = p[i];
        sexp out = f(par);
        doubles v = as_doubles(out);
        if (v.size() != 1)
            throw std::runtime_error(
                "the function must return a single number; got a value of length " +
                std::to_string(static_cast<long long>(v.size())));
        return v[0];
    };
}

// --- the RNG handle -----------------------------------------------------------------------
//
// The generator a callback draws from is owned by the C++ frame driving it, so what R gets is an
// external pointer to a BORROW (numerics/support/rng_handle.hpp) rather than to the generator.
// An R external pointer outlives the C++ frame that made it -- a user may assign it, capture it in
// a closure, or simply let it survive in an environment -- so the borrow's flag, cleared by
// RngScope's destructor, is the only thing standing between that and a read of freed memory. Every
// draw goes through borrow_from() below, which checks the flag before the generator is touched.
//
// The `corehydro_rng` class attribute is for the user's benefit (print/inherits); the type check
// does not trust it, because an attribute can be forged.
//
// WHY THE TAG. EXTPTRSXP alone is nowhere near enough: every package that hands R an external
// pointer produces the same SEXP type, so `TYPEOF == EXTPTRSXP` admits an Rcpp::XPtr, a curl
// handle, an xml2 document -- anything -- and casting one of those to RngBorrowPtr* and
// dereferencing it is a wild pointer read. It is not theoretical: passing the address slot of a
// registered native routine (base R, no packages needed) segfaulted this function and aborted the
// session. So the pointer is TAGGED with the `corehydro_rng` symbol where it is created, and the
// tag is REQUIRED here before the cast. The tag lives in the SEXP itself, not in an attribute, and
// R offers no way to set it from R code, so unlike the class attribute it cannot be forged from a
// script. Four checks in order: EXTPTRSXP, our tag, a non-null address, the borrow's own flag.
const char* kNotAHandle =
    "`rng` must be a random number generator handle, the one a callback is given as its second "
    "argument";

SEXP rng_tag() { return Rf_install("corehydro_rng"); }

sup::RngBorrowPtr& borrow_from(sexp handle) {
    if (TYPEOF(handle) != EXTPTRSXP) throw std::runtime_error(kNotAHandle);
    if (R_ExternalPtrTag(handle) != rng_tag()) throw std::runtime_error(kNotAHandle);
    void* addr = R_ExternalPtrAddr(handle);
    if (addr == nullptr) throw std::runtime_error(sup::rng_handle_expired_message());
    sup::RngBorrowPtr* p = static_cast<sup::RngBorrowPtr*>(addr);
    if (!*p) throw std::runtime_error(sup::rng_handle_expired_message());
    return *p;
}

// Wraps `f` as the (parameters, generator) callback the rng/mcmc/bootstrap groups call. The scope
// is a local of this lambda, so the handle it hands R is invalidated the moment the callback
// returns -- by a normal return or by an unwind, since a destructor runs either way.
std::function<std::vector<double>(const std::vector<double>&, samp::MersenneTwister&)> as_rng_fn(
    function f) {
    return [f](const std::vector<double>& p, samp::MersenneTwister& prng) mutable {
        writable::doubles par(static_cast<R_xlen_t>(p.size()));
        for (std::size_t i = 0; i < p.size(); ++i) par[static_cast<R_xlen_t>(i)] = p[i];

        sup::RngScope scope(prng);
        external_pointer<sup::RngBorrowPtr> ptr(new sup::RngBorrowPtr(scope.handle()));
        sexp handle(ptr);
        // The tag borrow_from() requires. A symbol is a permanent SEXP, so the check there is a
        // pointer comparison against this same object.
        R_SetExternalPtrTag(handle, rng_tag());
        sexp cls(Rf_mkString("corehydro_rng"));
        Rf_setAttrib(handle, R_ClassSymbol, cls);

        sexp out = f(par, handle);
        doubles v = as_doubles(out);
        return std::vector<double>(v.begin(), v.end());
    };
}

list pack(const sup::CallbackResult& r) {
    writable::doubles values(static_cast<R_xlen_t>(r.values.size()));
    for (std::size_t i = 0; i < r.values.size(); ++i)
        values[static_cast<R_xlen_t>(i)] = r.values[i];
    writable::strings names(static_cast<R_xlen_t>(r.names.size()));
    for (std::size_t i = 0; i < r.names.size(); ++i) names[static_cast<R_xlen_t>(i)] = r.names[i];
    writable::integers dims(static_cast<R_xlen_t>(r.dims.size()));
    for (std::size_t i = 0; i < r.dims.size(); ++i) dims[static_cast<R_xlen_t>(i)] = r.dims[i];
    return writable::list({"values"_nm = values, "names"_nm = names, "dims"_nm = dims,
                           "status"_nm = writable::strings({r.status})});
}

}  // namespace

// Runs one method of the callback runner's "math" group against an R function. The scalar methods
// (root_find, derivative, quadrature) call `f` with one number; the vector methods (gradient,
// hessian) call it with the whole point. Either way `f` must return a single number.
[[cpp11::register]]
list ch_callback_math_(std::string method, std::string options_json, function f) {
    sup::CallbackSet cbs;
    if (method == "root_find" || method == "derivative" || method == "quadrature")
        cbs.scalar = as_scalar_fn(f);
    else
        cbs.vector_scalar = as_vector_scalar_fn(f);
    return pack(sup::run_callback("math", method, options_json, cbs));
}

// Seeds a generator, hands `f` a handle on it together with the `parameters` vector from the
// options, and returns what `f` drew. Registered but unexported: it is how R/callback.R's
// rng_probe() and the fixture runner reach the rng group, and the shape it drives -- a callback of
// (parameters, generator) -- is the Gibbs proposal's, so what it proves about the handle carries
// over to the samplers that will use it.
[[cpp11::register]]
list ch_rng_probe_(std::string options_json, function f) {
    sup::CallbackSet cbs;
    cbs.vector_rng = as_rng_fn(f);
    return pack(sup::run_callback("rng", "probe", options_json, cbs));
}

// `n` draws on [0, 1) from the borrowed generator (the ported C# MersenneTwister.NextDouble).
[[cpp11::register]]
doubles ch_rng_uniform_(sexp handle, int n) {
    std::vector<double> v = borrow_from(handle)->uniform(n);
    writable::doubles out(static_cast<R_xlen_t>(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) out[static_cast<R_xlen_t>(i)] = v[i];
    return out;
}

// `n` draws on [min_inclusive, max_exclusive) from the borrowed generator (the ported C#
// MersenneTwister.Next(int, int)).
[[cpp11::register]]
integers ch_rng_integers_(sexp handle, int n, int min_inclusive, int max_exclusive) {
    std::vector<int> v = borrow_from(handle)->integers(n, min_inclusive, max_exclusive);
    writable::integers out(static_cast<R_xlen_t>(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) out[static_cast<R_xlen_t>(i)] = v[i];
    return out;
}
