// ported from: Numerics/Data/Paired Data/Ordinate.cs @ 2a0357a
//
// A single (X, Y) ordinate with an IsValid flag (false whenever either coordinate is
// infinite or NaN), plus the monotonicity checks (OrdinateValid/OrdinateErrors) and the
// Transform helper the Paired Data classes (OrderedPairedData and friends, Task 8) use to
// validate and preprocess a series of ordinates.
//
// Deliberately NOT ported (project-wide severance -- see e.g.
// models/data_frame/data_frame.hpp's header): the XElement constructor and ToXElement(). XML
// round-tripping has no C++ counterpart anywhere in this port; ctest skips Test_ToXElement.
// GetHashCode/Equals(object) are likewise not ported (no untyped-equality/hashing consumer in
// this port's scope; operator== below is the one equality entry point).
//
// Transcription note 1 (C# lines ~289-291, the operator== comment, verbatim in spirit): the
// equality operator tests `fabs(diff) > kDoubleMachineEpsilon` per coordinate and returns
// false only when that predicate is true. `fabs(NaN - anything) > eps` is ALWAYS false (NaN
// comparisons are false), so an ordinate holding a NaN coordinate compares EQUAL to every
// other ordinate on that coordinate. This is a documented upstream quirk, not a bug: the C#
// comment says so directly, and Test_Construction asserts `Ordinate(NaN, 4) == Ordinate(2, 4)`.
// It is pinned here exactly the same way.
//
// Transcription note 2: this is one of two independent PerpendicularDistance-shaped formulas
// in the (partially ported) Paired Data subsystem. This file has no such method itself, but
// line_simplification.hpp's `perpendicular_distance` (normalized segment direction, guarded
// against a degenerate zero-length segment) is NOT the same formula as
// `OrderedPairedData::perpendicular_distance` (Task 8: triangle-area-over-base, no degenerate
// guard) -- see line_simplification.hpp's own note for the other side of this split.
#pragma once
#include <cmath>
#include <string>
#include <vector>

#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data::paired_data {

struct Ordinate {
    double x = 0.0;
    double y = 0.0;
    bool is_valid = false;

    Ordinate() = default;

    // C# lines 35-44: computes is_valid, false whenever either coordinate is +-infinity or NaN.
    Ordinate(double x_value, double y_value) : x(x_value), y(y_value), is_valid(true) {
        if (std::isinf(x) || std::isnan(x) || std::isinf(y) || std::isnan(y)) {
            is_valid = false;
        }
    }

    // C# OrdinateValid (lines ~103-183): test if `compare` is a valid next/previous ordinate
    // in a monotonic series against this ordinate, per the strictness/order criteria.
    bool ordinate_valid(const Ordinate& compare, bool strict_x, bool strict_y,
                         SortOrder x_order, SortOrder y_order,
                         bool compare_ordinate_is_next) const {
        if (!is_valid) return false;

        if (compare_ordinate_is_next) {
            // Looking forward.
            if (strict_y && y_order != SortOrder::None) {
                if (compare.y == y) return false;
            }
            if (y_order == SortOrder::Descending) {
                if (compare.y > y) return false;
            } else if (y_order == SortOrder::Ascending) {
                if (compare.y < y) return false;
            }

            if (strict_x && x_order != SortOrder::None) {
                if (compare.x == x) return false;
            }
            if (x_order == SortOrder::Descending) {
                if (compare.x > x) return false;
            } else if (x_order == SortOrder::Ascending) {
                if (compare.x < x) return false;
            }
        } else {
            // Looking back.
            if (strict_y && y_order != SortOrder::None) {
                if (y == compare.y) return false;
            }
            if (y_order == SortOrder::Descending) {
                if (y > compare.y) return false;
            } else if (y_order == SortOrder::Ascending) {
                if (y < compare.y) return false;
            }

            if (strict_x && x_order != SortOrder::None) {
                if (x == compare.x) return false;
            }
            if (x_order == SortOrder::Descending) {
                if (x > compare.x) return false;
            } else if (x_order == SortOrder::Ascending) {
                if (x < compare.x) return false;
            }
        }
        return true;
    }

    // C# OrdinateErrors(ordinateToCompare, ...) (lines ~192-249): the monotonicity error
    // messages, in emission order -- this-ordinate's own validity errors first, then the
    // strict-equality messages (Y before X), then the direction messages (Y before X).
    std::vector<std::string> ordinate_errors(const Ordinate& compare, bool strict_x,
                                              bool strict_y, SortOrder x_order,
                                              SortOrder y_order,
                                              bool compare_ordinate_is_next) const {
        std::vector<std::string> result = ordinate_errors();

        std::string order_y = (y_order == SortOrder::Descending) ? "decreasing" : "increasing";
        std::string order_x = (x_order == SortOrder::Descending) ? "decreasing" : "increasing";

        if (strict_y && y_order != SortOrder::None) {
            if (compare.y == y) result.push_back("Y values must be strictly " + order_y + ".");
        }
        if (strict_x && x_order != SortOrder::None) {
            if (compare.x == x) result.push_back("X values must be strictly " + order_x + ".");
        }

        if (compare_ordinate_is_next) {
            // Looking forward.
            if (y_order == SortOrder::Descending) {
                if (compare.y > y) result.push_back("Y values must decrease.");
            } else if (y_order == SortOrder::Ascending) {
                if (compare.y < y) result.push_back("Y values must increase.");
            }
            if (x_order == SortOrder::Descending) {
                if (compare.x > x) result.push_back("X values must decrease.");
            } else if (x_order == SortOrder::Ascending) {
                if (compare.x < x) result.push_back("X values must increase.");
            }
        } else {
            // Looking back.
            if (y_order == SortOrder::Descending) {
                if (y > compare.y) result.push_back("Y values must decrease.");
            } else if (y_order == SortOrder::Ascending) {
                if (y < compare.y) result.push_back("Y values must increase.");
            }
            if (x_order == SortOrder::Descending) {
                if (x > compare.x) result.push_back("X values must decrease.");
            } else if (x_order == SortOrder::Ascending) {
                if (x < compare.x) result.push_back("X values must increase.");
            }
        }
        return result;
    }

    // C# OrdinateErrors() (lines ~255-268): errors in this ordinate's own data, in emission
    // order X-infinity, X-NaN, Y-infinity, Y-NaN.
    std::vector<std::string> ordinate_errors() const {
        std::vector<std::string> result;
        if (!is_valid) {
            if (std::isinf(x)) result.push_back("X value can not be infinity.");
            if (std::isnan(x)) result.push_back("X value must be a valid number.");
            if (std::isinf(y)) result.push_back("Y value can not be infinity.");
            if (std::isnan(y)) result.push_back("Y value must be a valid number.");
        }
        return result;
    }

    // C# Transform (lines ~275-303): Tools.Log10 is corehydro::numerics::clamped_log10;
    // Normal.StandardZ is distributions::Normal::standard_z. Transform::None leaves the
    // coordinate unchanged (the switch has no default case in C#).
    Ordinate transform(Transform x_transform, Transform y_transform) const {
        double transformed_x = x;
        double transformed_y = y;
        switch (x_transform) {
            case Transform::Logarithmic:
                transformed_x = corehydro::numerics::clamped_log10(x);
                break;
            case Transform::NormalZ:
                transformed_x = distributions::Normal::standard_z(x);
                break;
            case Transform::None:
                break;
        }
        switch (y_transform) {
            case Transform::Logarithmic:
                transformed_y = corehydro::numerics::clamped_log10(y);
                break;
            case Transform::NormalZ:
                transformed_y = distributions::Normal::standard_z(y);
                break;
            case Transform::None:
                break;
        }
        return Ordinate(transformed_x, transformed_y);
    }
};

// C# operator== (lines ~311-316). See transcription note 1 above: this is where the pinned
// NaN-compares-equal quirk lives.
inline bool operator==(const Ordinate& l, const Ordinate& r) {
    if (std::fabs(l.x - r.x) > corehydro::numerics::kDoubleMachineEpsilon) return false;
    if (std::fabs(l.y - r.y) > corehydro::numerics::kDoubleMachineEpsilon) return false;
    return true;
}

// C# operator!= (lines ~324-327).
inline bool operator!=(const Ordinate& l, const Ordinate& r) { return !(l == r); }

}  // namespace corehydro::numerics::data::paired_data
