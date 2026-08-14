// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `histogram` group's dispatch arm: the per-bin table and the summary statistics, both
// off a Histogram built from the Rice-Rule default constructor or the explicit-bin-count one.
// Includes toolbox/common.hpp, which defines the shared ToolboxResult/data_at/scalar helpers
// used here.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/histogram.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

// Histogram: bins omitted (the default) selects the Rice-Rule constructor `Histogram(data)`;
// bins > 0 selects the explicit-bin-count constructor `Histogram(data, bins)`. The real C#
// class has no lower/upper-bound constructor overload -- both ctors derive their range from
// `data` -- so no such option is accepted here. A non-positive explicit `bins` is an error.
inline ToolboxResult run_histogram(const std::string& method,
                                   const std::vector<std::vector<double>>& data,
                                   const JsonValue& options) {
    const std::vector<double>& x = data_at(data, 0, "histogram", method);
    numerics::data::Histogram h = options.contains("bins")
        ? [&x, &options]() {
            int bins = options.at("bins").as_int();
            if (bins <= 0) {
                throw std::runtime_error("bins must be positive; got " + std::to_string(bins));
            }
            return numerics::data::Histogram(x, bins);
        }()
        : numerics::data::Histogram(x);
    if (method == "bins") {
        ToolboxResult r;
        for (int i = 0; i < h.number_of_bins(); ++i) {
            numerics::data::Histogram::Bin b = h.bin(i);
            r.values.push_back(b.lower_bound);
            r.values.push_back(b.upper_bound);
            r.values.push_back(b.midpoint());
            r.values.push_back(static_cast<double>(b.frequency));
        }
        r.names = {"lower", "upper", "midpoint", "frequency"};
        r.dims = {h.number_of_bins(), 4};
        return r;
    }
    if (method == "statistics") {
        ToolboxResult r;
        r.values = {h.mean(),        h.median(),       h.mode(),     h.standard_deviation(),
                    h.lower_bound(), h.upper_bound(),  h.bin_width(),
                    static_cast<double>(h.number_of_bins())};
        r.names = {"mean", "median", "mode", "sd", "lower", "upper", "bin_width", "bins"};
        return r;
    }
    throw std::runtime_error("unknown histogram method: " + method);
}

}  // namespace corehydro::numerics::support::detail
