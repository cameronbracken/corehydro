// cpp11 glue for the callback surface (R/callback.R). Unlike the toolbox verbs, which pass
// serializable data through ch_toolbox_run_, every verb here takes a live R function, so it goes
// through callback_runner.hpp and its guard. Mirrors corehydropy's src/bindings/callback.cpp
// call for call.
//
// INTERRUPTS, measured rather than assumed (Task 4, mcmc_posterior). Ctrl-C during a long chain
// returns control to the prompt with an `interrupt` condition, and no R_CheckUserInterrupt() call
// is needed in this file to get it. R's own eval() checks for a pending interrupt every thousand
// evaluations, so any callback that evaluates R code at all -- which every callback does -- is a
// check point. R raises the interrupt inside the closure, cpp11 turns it into a C++ exception at
// the call below, GuardedCall latches it, and run_callback rethrows it inside this same protected
// frame, exactly as it does a user's own stop(). Verified over a pseudo-terminal against a real
// interactive session: an added cpp11::safe[R_CheckUserInterrupt]() changed neither the outcome
// nor the timing, so it was taken back out.
//
// What Ctrl-C is NOT is instant, and that is worth knowing before reporting a hang. The ported
// samplers have no cancellation hook (the C# CancellationTokenSource is a documented omission of
// this port), so after the latch the chain still runs to the end of its loop -- every remaining
// point rejected without re-entering R -- before the exception surfaces. Measured on a 100,000-
// iteration single chain at the default thinning: control came back 4.2 seconds after Ctrl-C, out
// of a 10-second run. Python behaves the same way and for the same reason -- CPython raises
// KeyboardInterrupt inside the callback, pybind11 hands it over as py::error_already_set, and the
// guard carries it out -- and is quicker to unwind: 0.3 seconds after SIGINT on a 200,000-
// iteration chain. Neither language hangs and neither segfaults.
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
        // NA and NaN are refused; +/-Inf is NOT. An infinite log-density is the ordinary way to
        // say "this parameter is impossible" and every sampler treats it as a rejected point, but
        // NA is never an answer -- it is R telling you the arithmetic had a hole in it. The
        // commonest hole on this surface is a log-likelihood reading p[3] when the prior list has
        // two entries: R returns NA where Python raises IndexError, so without this check the same
        // mistake is a clear error in one package and a silently motionless chain in the other.
        // corehydropy's as_vector_scalar_fn refuses nan for the same reason.
        if (ISNAN(v[0]))
            throw std::runtime_error(
                "the function returned NA or NaN rather than a number; in R, reading past the end "
                "of a vector gives NA, so check the length of the vector it was passed");
        return v[0];
    };
}

// f(theta) -> vector, the shape the HMC/NUTS gradient callback takes. The length is checked in
// the core (against the prior count, which is the only place that knows it); here the only job is
// refusing something that is not a numeric vector at all, and refusing NA/NaN for the same reason
// as_vector_scalar_fn does -- reading past the end of a vector is the commonest mistake on this
// surface and R answers it with NA rather than an error.
std::function<std::vector<double>(const std::vector<double>&)> as_vector_vector_fn(function f) {
    return [f](const std::vector<double>& p) mutable -> std::vector<double> {
        writable::doubles par(static_cast<R_xlen_t>(p.size()));
        for (std::size_t i = 0; i < p.size(); ++i) par[static_cast<R_xlen_t>(i)] = p[i];
        sexp out = f(par);
        doubles v = as_doubles(out);
        std::vector<double> result(v.begin(), v.end());
        for (double value : result)
            if (ISNAN(value))
                throw std::runtime_error(
                    "the function returned NA or NaN rather than a number; in R, reading past the "
                    "end of a vector gives NA, so check the length of the vector it was passed");
        return result;
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

// THE one place a handle is built, so no caller can forget the tag. A handle without it is
// refused by borrow_from() above -- correctly, but the failure lands on the user as "must be a
// random number generator handle" for a handle we ourselves handed them, which is a confusing way
// to learn that a new verb's glue omitted one line. Every group that hands R a generator (rng,
// mcmc's Gibbs proposal, and the bootstrap delegates in Task 6) calls this rather than building
// its own external pointer.
sexp make_rng_handle(const sup::RngBorrowPtr& borrow) {
    external_pointer<sup::RngBorrowPtr> ptr(new sup::RngBorrowPtr(borrow));
    sexp handle(ptr);
    // The tag borrow_from() requires. A symbol is a permanent SEXP, so the check there is a
    // pointer comparison against this same object.
    R_SetExternalPtrTag(handle, rng_tag());
    sexp cls(Rf_mkString("corehydro_rng"));
    Rf_setAttrib(handle, R_ClassSymbol, cls);
    return handle;
}

// Wraps `f` as the (parameters, generator) callback the rng/mcmc/bootstrap groups call. The scope
// is a local of this lambda, so the handle it hands R is invalidated the moment the callback
// returns -- by a normal return or by an unwind, since a destructor runs either way.
//
// Review fix (Task 5, finding 1): refuses NA/NaN by name, same wording as_vector_vector_fn uses,
// for the same reason -- a Gibbs proposal (the caller this shape serves today) that returns NaN
// would otherwise be accepted unconditionally by Gibbs and corrupt the chain silently rather than
// failing loudly. The check lives HERE rather than duplicated per call site because every caller
// of this shape (the Gibbs proposal today, a bootstrap resample in Task 6) shares the mistake.
std::function<std::vector<double>(const std::vector<double>&, samp::MersenneTwister&)> as_rng_fn(
    function f) {
    return [f](const std::vector<double>& p, samp::MersenneTwister& prng) mutable {
        writable::doubles par(static_cast<R_xlen_t>(p.size()));
        for (std::size_t i = 0; i < p.size(); ++i) par[static_cast<R_xlen_t>(i)] = p[i];

        sup::RngScope scope(prng);
        sexp handle = make_rng_handle(scope.handle());

        sexp out = f(par, handle);
        doubles v = as_doubles(out);
        std::vector<double> result(v.begin(), v.end());
        for (double value : result)
            if (ISNAN(value))
                throw std::runtime_error(
                    "the function returned NA or NaN rather than a number; in R, reading past the "
                    "end of a vector gives NA, so check the length of the vector it was passed");
        return result;
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

// Runs the callback runner's "mcmc" group against an R log-likelihood: `f` is called with the
// whole parameter vector and must return a single number, exactly as upstream's own
// `LogLikelihood` delegate does. The flat result is the layout documented in
// numerics/support/callback/mcmc.hpp; R/callback.R's mcmc_posterior() slices it back into the
// list mcmc_sample() returns.
//
// `proposal` and `gradient` are the two other delegates upstream's samplers take, and either may
// be NULL: `proposal` is Gibbs's conditional draw, called with (parameters, rng) where `rng` is a
// handle on THIS chain's generator, and `gradient` is HMC/NUTS's optional analytic gradient,
// called with the parameter vector. A NULL gradient leaves the ported finite-difference default
// in force; which sampler may take which is enforced in the core, once, for all four runners.
//
// INTERRUPTS. Ctrl-C during a long chain returns control with an interrupt condition, and not
// instantly -- both halves of that are measured and explained in this file's header note.
[[cpp11::register]]
list ch_callback_mcmc_(std::string options_json, function f, sexp proposal, sexp gradient) {
    sup::CallbackSet cbs;
    cbs.vector_scalar = as_vector_scalar_fn(f);
    // The proposal takes the SAME (parameters, generator) shape the rng group's probe takes, so it
    // is wrapped by the same as_rng_fn -- there is one place a handle is handed to R, not two.
    if (!Rf_isNull(proposal)) cbs.vector_rng = as_rng_fn(function(proposal));
    if (!Rf_isNull(gradient)) cbs.vector_vector = as_vector_vector_fn(function(gradient));
    return pack(sup::run_callback("mcmc", "sample", options_json, cbs));
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
