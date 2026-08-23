// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `interpolation` group's dispatch arms (Linear/Bilinear/CubicSpline/Polynomial)
// plus the two enum parsers (transform, sort order) it needs. Includes toolbox/common.hpp,
// which defines the shared ToolboxResult/data_at/scalar helpers used here.
//
// CubicSpline and Polynomial have neither a transform surface nor an Extrapolate() method in
// C# (only Linear/Bilinear do), so `cubic_spline`/`polynomial` accept only `sort_order` (and,
// for `polynomial`, the required `order`); `x_transform`/`y_transform`/`extrapolate` are simply
// not read for these two methods -- the R/Python `interpolate()` verb is the one that validates
// and rejects them for a non-"linear" method, matching what C# has on these classes.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/interpolation/bilinear.hpp"
#include "corehydro/numerics/data/interpolation/cubic_spline.hpp"
#include "corehydro/numerics/data/interpolation/linear.hpp"
#include "corehydro/numerics/data/interpolation/polynomial.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

inline numerics::data::Transform parse_transform(const std::string& s) {
    if (s == "none") return numerics::data::Transform::None;
    if (s == "log") return numerics::data::Transform::Logarithmic;
    if (s == "normal_z") return numerics::data::Transform::NormalZ;
    throw std::runtime_error("unknown transform '" + s + "'; expected none, log, or normal_z");
}

inline numerics::data::SortOrder parse_sort_order(const std::string& s) {
    if (s == "ascending") return numerics::data::SortOrder::Ascending;
    if (s == "descending") return numerics::data::SortOrder::Descending;
    throw std::runtime_error("unknown sort order '" + s + "'; expected ascending or descending");
}

// Linear/Bilinear: mirrors Numerics.Data.Interpolation.Linear/Bilinear exactly, including their
// x/y transforms (None/Logarithmic/NormalZ) and Linear's separate Extrapolate() surface (the
// clamp-to-end-knot behavior of interpolate() vs. the linear extension of extrapolate() is a
// real C# API distinction, not an option on one method).
inline ToolboxResult run_interpolation(const std::string& method,
                                       const std::vector<std::vector<double>>& data,
                                       const JsonValue& options) {
    namespace nd = numerics::data;
    nd::SortOrder order = parse_sort_order(options.value_or("sort_order", "ascending"));

    if (method == "linear") {
        const std::vector<double>& x = data_at(data, 0, "interpolation", method);
        const std::vector<double>& y = data_at(data, 1, "interpolation", method);
        const std::vector<double>& xout = data_at(data, 2, "interpolation", method);
        nd::Linear interp(x, y, order);
        interp.x_transform = parse_transform(options.value_or("x_transform", "none"));
        interp.y_transform = parse_transform(options.value_or("y_transform", "none"));
        bool extrapolate = options.value_or("extrapolate", false);
        ToolboxResult r;
        for (double v : xout)
            r.values.push_back(extrapolate ? interp.extrapolate(v) : interp.interpolate(v));
        return r;
    }

    if (method == "bilinear") {
        const std::vector<double>& x1 = data_at(data, 0, "interpolation", method);
        const std::vector<double>& x2 = data_at(data, 1, "interpolation", method);
        const std::vector<double>& flat = data_at(data, 2, "interpolation", method);
        const std::vector<double>& x1out = data_at(data, 3, "interpolation", method);
        const std::vector<double>& x2out = data_at(data, 4, "interpolation", method);
        if (flat.size() != x1.size() * x2.size())
            throw std::runtime_error("bilinear 'y' holds " + std::to_string(flat.size()) +
                                     " values, expected " + std::to_string(x1.size()) + " x " +
                                     std::to_string(x2.size()));
        if (x1out.size() != x2out.size())
            throw std::runtime_error("bilinear needs one x2 value per x1 value");
        std::vector<std::vector<double>> y(x1.size(), std::vector<double>(x2.size()));
        for (std::size_t i = 0; i < x1.size(); ++i)
            for (std::size_t j = 0; j < x2.size(); ++j) y[i][j] = flat[i * x2.size() + j];
        nd::Bilinear interp(x1, x2, y, order);
        interp.x1_transform = parse_transform(options.value_or("x1_transform", "none"));
        interp.x2_transform = parse_transform(options.value_or("x2_transform", "none"));
        interp.y_transform = parse_transform(options.value_or("y_transform", "none"));
        ToolboxResult r;
        for (std::size_t i = 0; i < x1out.size(); ++i)
            r.values.push_back(interp.interpolate(x1out[i], x2out[i]));
        return r;
    }

    if (method == "cubic_spline") {
        const std::vector<double>& x = data_at(data, 0, "interpolation", method);
        const std::vector<double>& y = data_at(data, 1, "interpolation", method);
        const std::vector<double>& xout = data_at(data, 2, "interpolation", method);
        nd::CubicSpline interp(x, y, order);
        ToolboxResult r;
        for (double v : xout) r.values.push_back(interp.interpolate(v));
        return r;
    }

    if (method == "polynomial") {
        const std::vector<double>& x = data_at(data, 0, "interpolation", method);
        const std::vector<double>& y = data_at(data, 1, "interpolation", method);
        const std::vector<double>& xout = data_at(data, 2, "interpolation", method);
        if (!options.contains("order"))
            throw std::runtime_error("toolbox method 'interpolation.polynomial' needs an "
                                     "'order' option");
        int poly_order = options.at("order").as_int();
        nd::Polynomial interp(poly_order, x, y, order);
        ToolboxResult r;
        for (double v : xout) r.values.push_back(interp.interpolate(v));
        return r;
    }
    throw std::runtime_error("unknown interpolation method: " + method);
}

}  // namespace corehydro::numerics::support::detail
