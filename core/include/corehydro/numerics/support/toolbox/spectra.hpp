// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `spectra` group's dispatch arm: autocorrelation (correlation/covariance/partial,
// with the confidence-interval companion), cross-correlation, and the DFT/real-DFT pair.
// Includes toolbox/common.hpp, which defines the shared ToolboxResult/data_at/scalar helpers
// used here.
#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/numerics/data/autocorrelation.hpp"
#include "corehydro/numerics/math/fourier/fourier.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

inline ToolboxResult run_spectra(const std::string& method,
                                 const std::vector<std::vector<double>>& data,
                                 const JsonValue& options) {
    namespace nd = numerics::data;
    namespace nf = numerics::math::fourier;

    if (method == "autocorrelation") {
        const std::vector<double>& x = data_at(data, 0, "spectra", method);
        int lag_max = options.value_or("lag_max", -1);
        std::string type_name = options.value_or("type", "correlation");
        nd::Autocorrelation::Type type;
        if (type_name == "correlation") type = nd::Autocorrelation::Type::Correlation;
        else if (type_name == "covariance") type = nd::Autocorrelation::Type::Covariance;
        else if (type_name == "partial") type = nd::Autocorrelation::Type::Partial;
        else throw std::runtime_error("unknown spectra.autocorrelation type: " + type_name);

        // nd::Autocorrelation::function() returns nullopt for three distinct reasons (see
        // autocorrelation.hpp), and they are not the same failure: too few observations, an
        // explicit max_lag that can't produce any lag, and (Correlation only) a constant series
        // where the lag-0 denominator is zero. Check the first two before the call so the
        // message names the real cause instead of a generic "too short".
        int n = static_cast<int>(x.size());
        if (n < 2)
            throw std::runtime_error("spectra.autocorrelation: series must have at least two "
                                     "observations");
        if (lag_max == 0)
            throw std::runtime_error("spectra.autocorrelation: max_lag must be at least 1");

        auto acf = nd::Autocorrelation::function(x, lag_max, type);
        if (!acf)
            throw std::runtime_error("spectra.autocorrelation: series has zero variance; "
                                     "correlation is undefined for a constant series");
        ToolboxResult r;
        r.dims = {static_cast<int>(acf->size()), 2};
        r.values.reserve(acf->size() * 2);
        for (const auto& pair : *acf) {
            r.values.push_back(pair[0]);
            r.values.push_back(pair[1]);
        }
        return r;
    }
    if (method == "autocorrelation_ci") {
        int sample_size = options.at("sample_size").as_int();
        double interval = options.value_or("confidence_level", 0.95);
        ToolboxResult r;
        r.values = nd::Autocorrelation::correlation_confidence_interval(sample_size, interval);
        r.names = {"lower", "upper"};
        return r;
    }
    if (method == "cross_correlation") {
        const std::vector<double>& x = data_at(data, 0, "spectra", method);
        const std::vector<double>& y = data_at(data, 1, "spectra", method);
        ToolboxResult r;
        r.values = nf::correlation(x, y);
        return r;
    }
    if (method == "dft" || method == "dft_real") {
        const std::vector<double>& x = data_at(data, 0, "spectra", method);
        bool inverse = options.value_or("inverse", false);
        std::vector<double> copy = x;  // fft()/real_fft() mutate their argument in place
        if (method == "dft")
            nf::fft(copy, inverse);
        else
            nf::real_fft(copy, inverse);
        ToolboxResult r;
        r.values = std::move(copy);
        return r;
    }
    throw std::runtime_error("unknown spectra method: " + method);
}

}  // namespace corehydro::numerics::support::detail
