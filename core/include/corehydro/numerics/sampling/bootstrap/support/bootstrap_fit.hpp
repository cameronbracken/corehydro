// ported from: Numerics/Sampling/Bootstrap/Support/BootstrapFit.cs @ 2a0357a
//
// Stores a fitted parameter vector and the covariance matrix associated with that fit. Used
// by the covariance-aware pivotal bootstrap workflow (`Bootstrap<TData>`'s second
// constructor, `RunPivotalBootstrap`, `TransformPivotalBootstrap`) -- the pivotal workflow is
// now ported in full; see `bootstrap.hpp`'s file header for the documentation of record. This
// type is consumed by it.
//
// Namespace note: see bootstrap_results.hpp's header -- C# `BootstrapFit` lives in the flat
// `Numerics.Sampling` namespace despite sitting in a `Sampling/Bootstrap/Support/` folder;
// this port keeps that flat namespace (`corehydro::numerics::sampling`) for parity.
//
// Null-reference guards omitted: C#'s ctor throws `ArgumentNullException` when `covariance`
// is null; a C++ `const Matrix&` reference cannot be null, so that guard has no C++
// equivalent (same rationale the rest of this port uses for reference-vs-pointer parameters).
#pragma once
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"

namespace corehydro::numerics::sampling {

class BootstrapFit {
   public:
    // Constructs a new covariance-aware fit. Clones both inputs -- C# clones `parameters`/
    // `covariance` on input so callers can safely reuse or mutate their source objects after
    // construction (`ParameterSet::clone`/`Matrix::clone` already give value-copy semantics
    // for this port; the clone calls are kept for API-shape parity with the C# source).
    // `Matrix` has no default ctor, so the clones happen in the member-initializer list
    // (before the validation below runs, unlike the C# source's validate-then-clone order --
    // harmless, since `clone()` cannot itself fail: a failed validation still throws out of
    // the constructor and the already-initialized members are simply discarded).
    BootstrapFit(const math::optimization::ParameterSet& parameters, const math::linalg::Matrix& covariance)
        : parameters_(parameters.clone()), covariance_(covariance.clone()) {
        if (parameters.values.empty())
            throw std::invalid_argument("The parameter set must contain at least one value.");
        if (!covariance.is_square()) throw std::invalid_argument("The covariance matrix must be square.");
        if (covariance.number_of_rows() != static_cast<int>(parameters.values.size()))
            throw std::invalid_argument("The covariance dimension must match the parameter count.");
    }

    // Constructs a new covariance-aware fit from raw parameter values (C# overload; fitness
    // defaults to NaN, matching `new ParameterSet(parameters, double.NaN)`).
    BootstrapFit(const std::vector<double>& parameters, const math::linalg::Matrix& covariance)
        : BootstrapFit(math::optimization::ParameterSet(parameters, std::numeric_limits<double>::quiet_NaN()),
                        covariance) {}

    // Gets the fitted parameter set.
    const math::optimization::ParameterSet& parameters() const { return parameters_; }

    // Gets the covariance matrix for `parameters()`.
    const math::linalg::Matrix& covariance() const { return covariance_; }

    // Gets the number of fitted parameters.
    int parameter_count() const { return static_cast<int>(parameters_.values.size()); }

   private:
    math::optimization::ParameterSet parameters_;
    math::linalg::Matrix covariance_;
};

}  // namespace corehydro::numerics::sampling
