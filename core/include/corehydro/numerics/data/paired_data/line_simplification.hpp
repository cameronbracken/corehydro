// ported from: Numerics/Data/Paired Data/LineSimplification.cs @ 2a0357a
//
// The Ramer-Douglas-Peucker curve-decimation algorithm: given a curve made of ordinates,
// finds a similar curve with fewer points, where "similar" is bounded by a user-supplied
// maximum perpendicular distance (epsilon) between the original and simplified curves.
//
// The C# class is static with two public static methods; ported here as free functions in
// their own namespace rather than a class with only statics (no instance state to carry).
//
// Exception mapping: C# ArgumentOutOfRangeException ("Not enough points to simplify") maps to
// std::invalid_argument; the generic C# Exception ("Problem assembling output") maps to
// std::runtime_error. Neither is expected to be reachable from a well-formed caller -- the
// first guards fewer than two input ordinates, the second guards an internal recursion
// invariant -- but both are preserved because they are part of the C# contract.
//
// Transcription note: PerpendicularDistance here normalizes the segment direction (dx, dy) to
// a unit vector and guards the degenerate zero-length segment with `mag > 0.0`, falling back
// to the point-to-line-start distance when the segment has zero length (line_start ==
// line_end) -- this is the case Test_EqualPoints exercises. This is a DIFFERENT formula from
// `OrderedPairedData::perpendicular_distance` (Task 8, corehydro/numerics/data/paired_data/
// ordered_paired_data.hpp): that one is triangle-area-over-base and has NO degenerate guard.
// The two are independent upstream implementations for two different purposes and are both
// kept as-is; see ordinate.hpp's own transcription note 2 for the other side of this split.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/paired_data/ordinate.hpp"

namespace corehydro::numerics::data::paired_data::line_simplification {

// C# PerpendicularDistance (lines ~85-105).
inline double perpendicular_distance(const Ordinate& pt, const Ordinate& line_start,
                                      const Ordinate& line_end) {
    double dx = line_end.x - line_start.x;
    double dy = line_end.y - line_start.y;

    // Normalize.
    double mag = std::sqrt(dx * dx + dy * dy);
    if (mag > 0.0) {
        dx /= mag;
        dy /= mag;
    }
    double pvx = pt.x - line_start.x;
    double pvy = pt.y - line_start.y;

    // Get dot product (project pv onto normalized direction).
    double pvdot = dx * pvx + dy * pvy;

    // Scale line direction vector and subtract it from pv.
    double ax = pvx - pvdot * dx;
    double ay = pvy - pvdot * dy;

    return std::sqrt(ax * ax + ay * ay);
}

// C# RamerDouglasPeucker (lines ~31-77).
inline void ramer_douglas_peucker(const std::vector<Ordinate>& ordinates, double epsilon,
                                   std::vector<Ordinate>& output) {
    if (ordinates.size() < 2) {
        throw std::invalid_argument("Not enough points to simplify");
    }

    // Find the point with the maximum distance from the line between the start and end.
    double dmax = 0.0;
    std::size_t index = 0;
    std::size_t end = ordinates.size() - 1;
    for (std::size_t i = 1; i < end; ++i) {
        double d = perpendicular_distance(ordinates[i], ordinates[0], ordinates[end]);
        if (d > dmax) {
            index = i;
            dmax = d;
        }
    }

    // If max distance is greater than epsilon, recursively simplify.
    if (dmax > epsilon) {
        std::vector<Ordinate> rec_results1;
        std::vector<Ordinate> rec_results2;
        std::vector<Ordinate> first_line(ordinates.begin(), ordinates.begin() + index + 1);
        std::vector<Ordinate> last_line(ordinates.begin() + index, ordinates.end());
        ramer_douglas_peucker(first_line, epsilon, rec_results1);
        ramer_douglas_peucker(last_line, epsilon, rec_results2);

        // Build the result list. C# does not clear `output` here (unlike the else branch
        // below): the recursive calls above always hand `rec_results1`/`rec_results2` a fresh
        // empty vector, and `output` itself is fresh at every call site in this file, so
        // AddRange-without-Clear and this insert-without-clear are equivalent for every
        // reachable call. A caller handing a pre-populated `output` to the top-level call
        // would see it appended to rather than replaced, exactly as C# would.
        output.insert(output.end(), rec_results1.begin(),
                       rec_results1.end() - (rec_results1.empty() ? 0 : 1));
        output.insert(output.end(), rec_results2.begin(), rec_results2.end());
        if (output.size() < 2) throw std::runtime_error("Problem assembling output");
    } else {
        // Just return start and end points.
        output.clear();
        output.push_back(ordinates.front());
        output.push_back(ordinates.back());
    }
}

}  // namespace corehydro::numerics::data::paired_data::line_simplification
