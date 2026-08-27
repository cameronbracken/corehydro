// ported from: Numerics/Mathematics/Optimization/Local/GoldenSection.cs @ 2a0357a
//
// The Golden-Section optimization algorithm: successively narrows a bracketing interval by the
// golden ratio until it collapses on an interior optimum. The function need not be
// differentiable, and no derivatives are taken ("Numerical Recipes, Routines and Examples in
// Basic", J.C. Sprott, 1991; see the C# file's references).
//
// Transcription notes:
//
// 1. One-dimensional objective. C# takes a `Func<double, double>` and forwards it to the base
//    as `base((x) => objectiveFunction(x[0]), 1)`. The same adapter lambda is used here, so
//    the base's Evaluate/best-parameter tracking sees a single-element point exactly as in C#.
//
// 2. The golden-ratio constants (`R = 0.61803399`, `C = 1.0 - R`) are C# method locals; they
//    live at file scope as `constexpr` here because MSVC raises C3493 on the implicit use of a
//    `const` function-local inside a capture-less lambda (see CLAUDE.md's portability notes),
//    and file-scope constants are the repo's established answer to that.
//
// 3. Loop-condition quirk, deliberately preserved: the while condition is
//    `|x3 - x0| > AbsoluteTolerance || |x3 - x0| > RelativeTolerance * (|x1| + |x2|)`. An OR of
//    two "not yet converged" tests means BOTH must be satisfied before the loop exits, so the
//    stricter of the two governs -- where a Numerical-Recipes golden-section loop would use the
//    relative test alone. This is upstream behavior and is NOT "fixed" here; the C# oracle
//    tests pin the results it produces.
#pragma once
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/optimization/support/optimizer.hpp"

namespace corehydro::numerics::math::optimization {

// C# method locals `R` / `C`; hoisted to file scope for MSVC C3493 (see class header note 2).
inline constexpr double kGoldenSectionR = 0.61803399;
inline constexpr double kGoldenSectionC = 1.0 - kGoldenSectionR;

class GoldenSection : public Optimizer {
   public:
    // Construct a new Golden-Section optimization method.
    GoldenSection(std::function<double(double)> objective_function, double lower_bound,
                  double upper_bound)
        : Optimizer([objective_function](std::vector<double>& x) { return objective_function(x[0]); },
                    1) {
        // validate inputs
        if (upper_bound < lower_bound) {
            throw ArgumentException("The upper bound cannot be less than the lower bound.");
        }
        lower_bound_ = lower_bound;
        upper_bound_ = upper_bound;
    }

    // The lower bound (inclusive) of the interval containing the optimal point.
    double lower_bound() const { return lower_bound_; }

    // The upper bound (inclusive) of the interval containing the optimal point.
    double upper_bound() const { return upper_bound_; }

   protected:
    void optimize() override {
        // Define variables
        bool cancel = false;
        // Bracket problem
        double ax = lower_bound_, bx = 0.5 * (upper_bound_ + lower_bound_), cx = upper_bound_;
        double R = kGoldenSectionR, C = kGoldenSectionC;
        double x1, x2;
        double x0 = ax;
        double x3 = cx;
        if (std::fabs(cx - bx) > std::fabs(bx - ax)) {
            x1 = bx;
            x2 = bx + C * (cx - bx);
        } else {
            x2 = bx;
            x1 = bx - C * (bx - ax);
        }
        std::vector<double> point1 = {x1};
        std::vector<double> point2 = {x2};
        double f1 = evaluate(point1, cancel);
        double f2 = evaluate(point2, cancel);
        while (std::fabs(x3 - x0) > absolute_tolerance ||
               std::fabs(x3 - x0) > relative_tolerance * (std::fabs(x1) + std::fabs(x2))) {
            if (f2 < f1) {
                x0 = x1;
                x1 = x2;
                x2 = R * x2 + C * x3;
                f1 = f2;
                point2 = {x2};
                f2 = evaluate(point2, cancel);
            } else {
                x3 = x2;
                x2 = x1;
                x1 = R * x1 + C * x0;
                f2 = f1;
                point1 = {x1};
                f1 = evaluate(point1, cancel);
            }
            if (cancel == true) return;
            iterations_ += 1;
            if (iterations_ >= max_iterations) {
                update_status(OptimizationStatus::MaximumIterationsReached);
                return;
            }
        }
        update_status(OptimizationStatus::Success);
    }

   private:
    double lower_bound_ = 0.0;
    double upper_bound_ = 0.0;
};

}  // namespace corehydro::numerics::math::optimization
