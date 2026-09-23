// ported from: Numerics/Mathematics/Optimization/Local/NelderMead.cs @ 7e8e8d1
//             + Numerics/Mathematics/Optimization/Support/Optimizer.cs (base behavior)
//
// Nelder-Mead downhill simplex (Sprott 1991 / Press et al.). The Optimizer-base
// best-tracking, bound repair, convergence test, and minimize/maximize sign handling
// are folded in here for Phase 0 (the full optimizer hierarchy lands in a later phase).
// The start-point probe (disabled by default upstream) and Hessian computation
// (unused by the GEV MLE, which derives covariance analytically) are intentionally omitted.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "corehydro/numerics/math/optimization/support/optimization_status.hpp"

namespace corehydro::numerics::math::optimization {

class NelderMead {
   public:
    // MUTABLE-POINT SEMANTICS (M14, C#-fidelity): non-const reference, matching the C#
    // `Func<double[], double>` objective (arrays are reference types -- an objective CAN
    // write back into the evaluated point, and RMC.BestFit's MixtureModel does: its
    // DataLogLikelihood normalizes the weight entries in place, so the reflection/
    // contraction points this solver stores back into the simplex are the NORMALIZED ones,
    // exactly like upstream). Lambdas taking `const std::vector<double>&` still convert to
    // this std::function unchanged, so every non-mutating Phase 0/1 caller is unaffected.
    // Buffer lifetimes mirror the C# source: the init loop evaluates a COPY of each vertex
    // (PT), so vertex rows stay raw there; the loop evaluates pr/prr/z in place, so their
    // post-evaluation values are what lands in the simplex -- identical to upstream.
    using Objective = std::function<double(std::vector<double>&)>;

    NelderMead(Objective objective, int num_parameters, std::vector<double> initial_values,
               std::vector<double> lower_bounds, std::vector<double> upper_bounds)
        : objective_(std::move(objective)),
          D_(num_parameters),
          initial_(std::move(initial_values)),
          lower_(std::move(lower_bounds)),
          upper_(std::move(upper_bounds)) {}

    double simplex_scale = 0.05;
    int max_iterations = 10000;
    double relative_tolerance = 1E-8;
    double absolute_tolerance = 1E-8;

    void maximize() {
        function_scale_ = -1;
        clear_and_optimize();
    }
    void minimize() {
        function_scale_ = 1;
        clear_and_optimize();
    }

    const std::vector<double>& best_parameters() const { return best_values_; }
    OptimizationStatus status() const { return status_; }

   private:
    Objective objective_;
    int D_;
    std::vector<double> initial_, lower_, upper_;
    int function_scale_ = 1;
    int iterations_ = 0;
    bool has_best_ = false;
    double best_fitness_ = 0.0;
    std::vector<double> best_values_;
    OptimizationStatus status_ = OptimizationStatus::None;

    void clear_and_optimize() {
        iterations_ = 0;
        has_best_ = false;
        best_values_.clear();
        status_ = OptimizationStatus::None;
        optimize();
    }

    // Non-const `values` (see the Objective note above); the best-values copy is taken AFTER
    // the objective call, so a mutating objective's normalized point is what gets recorded --
    // matching the C# base's Evaluate() (`values.Clone()` after `ObjectiveFunction(values)`).
    double evaluate(std::vector<double>& values) {
        double fitness = function_scale_ * objective_(values);
        if (!has_best_ || fitness <= best_fitness_) {
            best_values_ = values;
            best_fitness_ = fitness;
            has_best_ = true;
        }
        return fitness;
    }

    static double repair(double value, double lo, double hi) {
        if (value < lo) value = lo;
        if (value > hi) value = hi;
        return value;
    }

    bool converged(double old_value, double new_value) const {
        if (std::isnan(old_value) || std::isnan(new_value) || std::isinf(old_value) ||
            std::isinf(new_value))
            return false;
        return 2.0 * std::fabs(new_value - old_value) /
                   (std::fabs(new_value) + std::fabs(old_value) + absolute_tolerance) <
               relative_tolerance;
    }

    void optimize() {
        const double alpha = 1.0, beta = 0.5, gamma = 2.0;
        const int D = D_;
        const int mpts = D + 1;
        std::vector<std::vector<double>> p(mpts, std::vector<double>(D));
        std::vector<double> f(mpts), pr(D), prr(D), pbar(D);
        const double epsAbsFrac = 1e-4;

        for (int j = 0; j < D; ++j) p[0][j] = initial_[j];

        for (int i = 1; i < mpts; ++i) {
            for (int j = 0; j < D; ++j) p[i][j] = p[0][j];
            int d = i - 1;
            double range = std::max(upper_[d] - lower_[d], 0.0);
            double absFloor = std::isinf(range) ? 1.0 : std::max(range * epsAbsFrac, 1e-12);
            double baseScale = std::max(std::fabs(p[0][d]), 1.0);
            if (!std::isinf(range)) baseScale = std::min(baseScale, 0.5 * range);
            double h = std::max(simplex_scale * baseScale, absFloor);
            double up = p[0][d] + h, dn = p[0][d] - h;
            bool placed = false;
            if (up <= upper_[d]) {
                p[i][d] = up;
                placed = true;
            } else if (dn >= lower_[d]) {
                p[i][d] = dn;
                placed = true;
            } else {
                for (int it = 0; it < 20 && !placed; ++it) {
                    h *= 0.5;
                    up = p[0][d] + h;
                    dn = p[0][d] - h;
                    if (up <= upper_[d]) {
                        p[i][d] = up;
                        placed = true;
                        break;
                    }
                    if (dn >= lower_[d]) {
                        p[i][d] = dn;
                        placed = true;
                        break;
                    }
                }
                if (!placed) p[i][d] = repair(p[0][d] + h, lower_[d], upper_[d]);
            }
        }

        std::vector<double> PT(D);
        for (int i = 0; i < mpts; ++i) {
            for (int j = 0; j < D; ++j) PT[j] = p[i][j];
            f[i] = evaluate(PT);
        }

        while (iterations_ < max_iterations) {
            int ilo = 1, ihi, inhi;
            if (f[0] > f[1]) {
                ihi = 0;
                inhi = 1;
            } else {
                ihi = 1;
                inhi = 0;
            }
            for (int i = 0; i < mpts; ++i) {
                if (f[i] < f[ilo]) ilo = i;
                if (f[i] > f[ihi]) {
                    inhi = ihi;
                    ihi = i;
                } else if (f[i] > f[inhi]) {
                    if (i != ihi) inhi = i;
                }
            }

            if (converged(f[ihi], f[ilo])) {
                status_ = OptimizationStatus::Success;
                return;
            }

            iterations_ += 1;
            for (int j = 0; j < D; ++j) pbar[j] = 0.0;
            for (int i = 0; i < mpts; ++i) {
                if (i == ihi) continue;
                for (int j = 0; j < D; ++j) pbar[j] += p[i][j];
            }
            for (int j = 0; j < D; ++j) {
                pbar[j] = pbar[j] / D;
                pr[j] = (1.0 + alpha) * pbar[j] - alpha * p[ihi][j];
                pr[j] = repair(pr[j], lower_[j], upper_[j]);
            }
            double fpr = evaluate(pr);

            if (fpr <= f[ilo]) {
                for (int j = 0; j < D; ++j) {
                    prr[j] = gamma * pr[j] + (1.0 - gamma) * pbar[j];
                    prr[j] = repair(prr[j], lower_[j], upper_[j]);
                }
                double fprr = evaluate(prr);
                if (fprr < f[ilo]) {
                    for (int j = 0; j < D; ++j) p[ihi][j] = prr[j];
                    f[ihi] = fprr;
                } else {
                    for (int j = 0; j < D; ++j) p[ihi][j] = pr[j];
                    f[ihi] = fpr;
                }
            } else if (fpr >= f[inhi]) {
                if (fpr < f[ihi]) {
                    for (int j = 0; j < D; ++j) p[ihi][j] = pr[j];
                    f[ihi] = fpr;
                }
                for (int j = 0; j < D; ++j) {
                    prr[j] = beta * p[ihi][j] + (1.0 - beta) * pbar[j];
                    prr[j] = repair(prr[j], lower_[j], upper_[j]);
                }
                double fprr = evaluate(prr);
                if (fprr < f[ihi]) {
                    for (int j = 0; j < D; ++j) p[ihi][j] = prr[j];
                    f[ihi] = fprr;
                } else {
                    for (int i = 0; i < mpts; ++i) {
                        if (i == ilo) continue;
                        for (int j = 0; j < D; ++j) {
                            pr[j] = 0.5 * (p[i][j] + p[ilo][j]);
                            pr[j] = repair(pr[j], lower_[j], upper_[j]);
                            p[i][j] = pr[j];
                        }
                        f[i] = evaluate(pr);
                    }
                }
            } else {
                for (int j = 0; j < D; ++j) p[ihi][j] = pr[j];
                f[ihi] = fpr;
            }
        }
        status_ = OptimizationStatus::MaximumIterationsReached;
    }
};

}  // namespace corehydro::numerics::math::optimization
