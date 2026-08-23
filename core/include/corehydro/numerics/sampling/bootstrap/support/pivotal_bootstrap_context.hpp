// ported from: Numerics/Sampling/Bootstrap/Support/PivotalBootstrapContext.cs @ 2a0357a
//
// Provides accepted raw bootstrap fits to a pivotal bootstrap link factory. Link factories
// can use this context to select fixed links, such as `LogLink` for positive scale
// parameters, or to fit data-adaptive links, such as `YeoJohnsonLink`, from the accepted raw
// bootstrap ensemble. Used by the covariance-aware pivotal bootstrap workflow, which is now
// ported in full -- see `bootstrap.hpp`'s file header for the documentation of record.
//
// Null-reference guards omitted: C#'s ctor throws `ArgumentNullException` when `parentFit` or
// `rawBootstrapFits` is null; a C++ `const BootstrapFit&`/`const std::vector<BootstrapFit>&`
// parameter cannot be null, so that guard has no C++ equivalent (same rationale
// bootstrap_fit.hpp uses for its own omitted null guards).
//
// Reference-vs-value semantics: C# `BootstrapFit` is a reference type, so `ParentFit`/
// `RawBootstrapFits` alias the caller's own objects. `BootstrapFit` in this port has value
// semantics, so the parent fit and raw fits are copied into the context on construction --
// consistent with how `BootstrapFit` itself already clones its `ParameterSet`/`Matrix`
// members.
#pragma once
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/sampling/bootstrap/support/bootstrap_fit.hpp"

namespace corehydro::numerics::sampling {

class PivotalBootstrapContext {
   public:
    // Constructs a new pivotal bootstrap context.
    //
    // parent_fit: the original parent fit.
    // raw_bootstrap_fits: the accepted raw bootstrap fits.
    // Throws std::invalid_argument when a raw fit's parameter count differs from the parent
    // fit's parameter count.
    PivotalBootstrapContext(const BootstrapFit& parent_fit, std::vector<BootstrapFit> raw_bootstrap_fits)
        : parent_fit_(parent_fit), raw_bootstrap_fits_(std::move(raw_bootstrap_fits)) {
        const int p = parent_fit_.parameter_count();
        raw_parameter_values_.assign(raw_bootstrap_fits_.size(), std::vector<double>(p));
        for (std::size_t i = 0; i < raw_bootstrap_fits_.size(); ++i) {
            if (raw_bootstrap_fits_[i].parameter_count() != p)
                throw std::invalid_argument(
                    "Every raw bootstrap fit must have the same parameter count as the parent fit.");

            for (int j = 0; j < p; ++j) raw_parameter_values_[i][j] = raw_bootstrap_fits_[i].parameters().values[j];
        }
    }

    // Gets the original parent fit.
    const BootstrapFit& parent_fit() const { return parent_fit_; }

    // Gets the accepted raw bootstrap fits.
    const std::vector<BootstrapFit>& raw_bootstrap_fits() const { return raw_bootstrap_fits_; }

    // Gets the number of fitted parameters.
    int parameter_count() const { return parent_fit_.parameter_count(); }

    // Gets a copy of the accepted raw bootstrap values for one parameter.
    //
    // parameter_index: the zero-based parameter index.
    // Throws std::out_of_range when parameter_index is outside the parameter range.
    std::vector<double> get_raw_parameter_values(int parameter_index) const {
        if (parameter_index < 0 || parameter_index >= parameter_count())
            throw std::out_of_range("parameter_index is outside the parameter range.");

        std::vector<double> values(raw_bootstrap_fits_.size());
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = raw_parameter_values_[i][parameter_index];
        return values;
    }

   private:
    BootstrapFit parent_fit_;
    std::vector<BootstrapFit> raw_bootstrap_fits_;
    std::vector<std::vector<double>> raw_parameter_values_;
};

}  // namespace corehydro::numerics::sampling
