// ported from: Numerics/Mathematics/Optimization/Global/ShuffledComplexEvolution.cs @ 2a0357a
//
// The Shuffled Complex Evolution (SCE-UA) global optimizer of Duan et al. (1992): a population
// is partitioned into complexes, each complex is evolved independently by the competitive
// complex evolution (CCE) sub-algorithm (reflection / contraction / mutation on a randomly drawn
// sub-complex), and the whole population is then reshuffled and reranked. No gradient needed.
//
// Transcription hazards:
//
// 1. Two independent streams from ONE seed, plus one child stream per complex. C# constructs
//    `new MersenneTwister(PRNGSeed)` AND passes the same `PRNGSeed` to
//    `LatinHypercube.Random(populationSize, D, PRNGSeed)` -- two separately-seeded streams, as in
//    DifferentialEvolution/ParticleSwarm. The parent `prng` is then used for exactly one thing:
//    drawing `Complexes` child seeds via `prng.Next()` (the INTEGER draw, not `NextDouble()`),
//    one per complex PRNG. Every random number consumed inside the CCE loop comes from a complex
//    PRNG, never from the parent, so each complex's stream is independent of how many evaluations
//    the other complexes performed.
//
// 2. Sorting. Both `List<T>.Sort` calls use an explicit comparison that breaks a fitness tie on
//    `Index`. `std::sort` is NOT a faithful stand-in for `List<T>.Sort` in general -- .NET's
//    introsort has its own tie permutation, which is why models/data_frame/data_frame_plotting.hpp
//    ports it outright -- but a stand-in is exact whenever the comparator is a strict total order,
//    because then only one sorted permutation exists. That holds for both call sites here:
//      * `Dpoints` and each `Acomplex` carry pairwise-DISTINCT `Index` values (Dpoints is
//        re-indexed 0..n-1 every outer iteration; the complexes deal those distinct indices out
//        and only ever overwrite an element's ParameterSet, never its Index), so no two elements
//        can compare equal.
//      * The sub-complex `B` CAN hold two entries with the same Index, because the trapezoidal
//        draw samples the complex WITH replacement -- but two such entries are clones of the very
//        same complex slot, i.e. byte-identical in every field the algorithm ever reads, so their
//        relative order is unobservable. (The one theoretical residual is a mutated `B.Last()`
//        landing on a bit-exact fitness tie with its own duplicate; that is a floating-point
//        coincidence, not a structural tie.)
//    Written down here so a future reader does not have to re-derive it.
//
// 3. `Array.FindIndex(cdf, p => rnd <= p)` returns -1 when the draw exceeds every cumulative
//    probability (reachable, since the trapezoidal CDF's last entry is 1 only up to rounding);
//    C# then falls back to `cdf.Length - 1`. Kept.
//
// 4. UPSTREAM ARRAY ALIASING, DELIBERATELY REPRODUCED. C# `new ParameterSet(double[] values,
//    double fitness)` stores the array REFERENCE -- it does not clone (Optimizer.Evaluate clones,
//    ParameterSet's constructor does not; see ParameterSet.cs). EvolveComplex reuses ONE scratch
//    array `p` for the reflection / contraction / mutation point across the whole beta loop and
//    assigns `B.Last().ParameterSet = new ParameterSet(p, fitness)`, so from that moment the
//    sub-complex entry's Values IS the scratch array: a later write to `p` silently rewrites that
//    entry's position while leaving its Fitness stale. This is observable whenever `alpha > 1`
//    (i.e. after the first converging outer iteration), because only then does a second alpha pass
//    read a `B` entry that a previous pass aliased -- most visibly in Contraction, which then
//    contracts toward the previous REFLECTION point rather than toward the worst point.
//    `std::vector<double>` has value semantics and cannot alias, so this port tracks the aliasing
//    explicitly: `PointFitness::values_alias_p` (a port-only bookkeeping flag with no C#
//    counterpart) marks every entry whose Values array C# would have pointed at `p`, and
//    `sync_alias(...)` copies `p` back into all of them immediately after each of the three
//    writers (Reflection / Contraction / SmallestHypercube). Syncing AFTER the whole write loop --
//    rather than element by element -- is exactly equivalent: each of those loops reads index `i`
//    of the aliased entry before writing index `i` of `p`, and never reads an index it has
//    already written, so the values C# observes mid-loop are precisely the pre-write ones.
//    Do NOT "fix" this by cloning; it changes the search path and therefore the oracle.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/sampling/latin_hypercube.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::math::optimization {

class ShuffledComplexEvolution : public Optimizer {
   public:
    // Construct a new shuffled complex evolution (SCE-UA) optimization method.
    ShuffledComplexEvolution(Objective objective_function, int number_of_parameters,
                             std::vector<double> lower_bounds, std::vector<double> upper_bounds)
        : Optimizer(std::move(objective_function), number_of_parameters),
          lower_bounds_(std::move(lower_bounds)),
          upper_bounds_(std::move(upper_bounds)) {
        // Check if the length of the lower and upper bounds equal the number of parameters
        if (static_cast<int>(lower_bounds_.size()) != number_of_parameters_ ||
            static_cast<int>(upper_bounds_.size()) != number_of_parameters_) {
            throw ArgumentException(
                "The lower and upper bounds must be the same length as the number of parameters.");
        }
        // Check if the lower bounds are less than the upper bounds
        for (std::size_t i = 0; i < lower_bounds_.size(); ++i) {
            if (upper_bounds_[i] <= lower_bounds_[i])
                throw ArgumentException("The upper bound cannot be less than or equal to the lower bound.");
        }
        cce_iterations = 2 * number_of_parameters_ + 1;
    }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    // The pseudo random number generator (PRNG) seed.
    int prng_seed = 12345;

    // The number of complexes. Default = 5.
    int complexes = 5;

    // The number of iterations in the inner loop (CCE algorithm). Default = 2 *
    // NumberOfParameters + 1, as recommended by Duan et al (1994). C# has no field initializer
    // here; the ctor is the only assignment.
    int cce_iterations = 0;

    // The number of iterations where the improvement is within the relative tolerance required
    // to confirm convergence. Default = 20.
    int tolerance_steps = 20;

   protected:
    void optimize() override {
        if (complexes < 1) throw ArgumentException("The number of complexes must be greater than 0.");
        if (cce_iterations < 1)
            throw ArgumentException("The number of CCE iterations must be greater than 0.");
        if (tolerance_steps < 1)
            throw ArgumentException("The number of tolerance steps must be greater than 0.");

        int D = number_of_parameters_;
        bool cancel = false;
        int population_size = complexes * cce_iterations;
        int converge = 0, alpha = 1;
        double old_fit, new_fit;

        // Initialize the population of points. Two independently-seeded streams from the SAME
        // PRNGSeed -- see transcription hazard 1 above.
        sampling::MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        auto rnd = sampling::LatinHypercube::random(population_size, D, prng_seed);
        std::vector<PointFitness> Dpoints;
        Dpoints.reserve(static_cast<std::size_t>(population_size));
        for (int i = 0; i < population_size; ++i) {
            std::vector<double> values(static_cast<std::size_t>(D));
            for (int j = 0; j < D; ++j) {
                std::size_t jj = static_cast<std::size_t>(j);
                values[jj] = lower_bounds_[jj] + (upper_bounds_[jj] - lower_bounds_[jj]) *
                                                     rnd[static_cast<std::size_t>(i)][jj];
            }

            PointFitness point;
            point.parameter_set = ParameterSet(values, evaluate(values, cancel));
            point.index = i;
            Dpoints.push_back(std::move(point));
            if (cancel) return;
        }
        old_fit = Dpoints[0].parameter_set.fitness;
        iterations_ += 1;

        // Initialize shuffled complex evolution (SCE) algorithm
        // Create list of complex prngs
        std::vector<sampling::MersenneTwister> complex_prngs;
        complex_prngs.reserve(static_cast<std::size_t>(complexes));
        for (int i = 0; i < complexes; ++i)
            complex_prngs.emplace_back(static_cast<std::uint32_t>(prng.next()));

        // Create trapezoidal cumulative probability for points in complex
        auto cdf = trapezoidal(cce_iterations);

        // Perform the Shuffled complex evolution loop
        while (iterations_ < max_iterations) {
            // Partition populations into p complexes distributing points evenly between complexes
            std::vector<std::vector<PointFitness>> Acomplex;  // The A complex
            for (int i = 0; i < complexes; ++i) {
                Acomplex.emplace_back();
                for (int j = 0; j < cce_iterations; ++j)
                    Acomplex[static_cast<std::size_t>(i)].push_back(
                        Dpoints[static_cast<std::size_t>(i + complexes * j)].clone());
            }

            // Evolve each complex according to competitive complex evolution (CCE) algorithm
            for (int i = 0; i < complexes; ++i) {
                auto& arg_acomplex = Acomplex[static_cast<std::size_t>(i)];
                evolve_complex(alpha, arg_acomplex, cdf, complex_prngs[static_cast<std::size_t>(i)],
                               cancel);
                if (cancel) return;
            }

            // Replace A into D
            for (int i = 0; i < complexes; ++i) {
                for (int j = 0; j < cce_iterations; ++j)
                    Dpoints[static_cast<std::size_t>(i + complexes * j)] =
                        Acomplex[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].clone();
            }

            // Rank the points in terms of fitness (i.e., order of increasing function value).
            // std::sort is exact here -- see transcription hazard 2.
            std::sort(Dpoints.begin(), Dpoints.end(), compare_point_fitness);

            // Next reset the indexes, so that i = 0 represents the point With the smallest
            // function value.
            for (std::size_t i = 0; i < Dpoints.size(); ++i) Dpoints[i].index = static_cast<int>(i);

            new_fit = Dpoints.front().parameter_set.fitness;

            // Check convergence
            if (check_convergence(old_fit, new_fit)) {
                // If no improvement increase alpha to increase chance of sub-complexes evolving
                converge += 1;
                alpha = std::min(3, converge + 1);
                old_fit = new_fit;

                if (converge >= tolerance_steps) {
                    update_status(OptimizationStatus::Success);
                    return;
                }
            } else {
                old_fit = new_fit;
                converge = 0;
                alpha = 1;
            }

            iterations_ += 1;
        }

        // If we made it to here, the maximum iterations were reached.
        update_status(OptimizationStatus::MaximumIterationsReached);
    }

   private:
    // Class for keeping track of parameter sets with indexes.
    struct PointFitness {
        ParameterSet parameter_set;
        int index = 0;

        // PORT-ONLY (no C# counterpart): true when C# would have this entry's `Values` array
        // pointing at EvolveComplex's scratch array `p` -- see transcription hazard 4. Never set
        // by clone(), because C# `Clone()` deep-copies the array and so breaks the alias.
        bool values_alias_p = false;

        // Creates a deep copy of this PointFitness instance.
        PointFitness clone() const {
            PointFitness copy;
            copy.parameter_set = parameter_set.clone();
            copy.index = index;
            return copy;
        }
    };

    // The comparison both `List<T>.Sort` call sites pass: fitness ascending, ties broken on
    // Index. Expressed as a strict weak ordering for std::sort (see transcription hazard 2).
    static bool compare_point_fitness(const PointFitness& x, const PointFitness& y) {
        if (x.parameter_set.fitness < y.parameter_set.fitness) return true;
        if (y.parameter_set.fitness < x.parameter_set.fitness) return false;
        return x.index < y.index;
    }

    // Copies the scratch point `p` into every sub-complex entry whose C# `Values` array would BE
    // `p` -- the explicit stand-in for the upstream aliasing (transcription hazard 4).
    static void sync_alias(std::vector<PointFitness>& B, const std::vector<double>& p) {
        for (auto& point : B)
            if (point.values_alias_p) point.parameter_set.values = p;
    }

    // Evolve complex according to the competitive complex evolution (CCE) algorithm.
    //   alpha:    The worst point in the complex is reflected or contracted to seek an
    //             improvement alpha times.
    //   Acomplex: The complex to evolve.
    //   cdf:      The trapezoidal cumulative probability for points in complex.
    //   prng:     The prng.
    //   cancel:   By ref. Determines if the solver should be canceled.
    void evolve_complex(int alpha, std::vector<PointFitness>& Acomplex, const std::vector<double>& cdf,
                        sampling::MersenneTwister& prng, bool& cancel) {
        int beta = static_cast<int>(Acomplex.size());
        int q = number_of_parameters_ + 1;
        std::vector<double> g(static_cast<std::size_t>(number_of_parameters_));  // centroid g
        std::vector<double> p(static_cast<std::size_t>(number_of_parameters_));
        double fitness;

        // The Beta loop. Allow sub-complex to evolve beta times.
        for (int i = 0; i < beta; ++i) {
            // Select parents by randomly choosing q distinct points from A complex
            std::vector<PointFitness> B;
            for (int j = 1; j <= q; ++j) {
                double rnd = prng.next_double();
                int rndi = -1;
                for (std::size_t k = 0; k < cdf.size(); ++k) {
                    if (rnd <= cdf[k]) {
                        rndi = static_cast<int>(k);
                        break;
                    }
                }
                if (rndi == -1) rndi = static_cast<int>(cdf.size()) - 1;

                B.push_back(Acomplex[static_cast<std::size_t>(rndi)].clone());
                B.back().index = rndi;
            }

            // Alpha loop. The worst point in the complex is reflected or contracted to seek an
            // improvement alpha times.
            for (int j = 0; j < alpha; ++j) {
                // Rank the B points in terms of fitness (i.e., order of increasing function value.
                std::sort(B.begin(), B.end(), compare_point_fitness);

                // Find centroid g excluding worst point and compute reflection of worst point
                // about centroid r = 2g - u(worst)
                reflection(B, g, p);
                sync_alias(B, p);

                // Check if r is feasible
                if (is_feasible(p) == true) {
                    // The parameter set is feasible, so evaluate it.
                    fitness = evaluate(p, cancel);
                    if (cancel) return;
                } else {
                    // The parameters were infeasible, so perform mutation
                    // Compute the smallest hypercube enclosing A complex and randomly sample a
                    // point within it.
                    smallest_hypercube(Acomplex, prng, p);
                    sync_alias(B, p);
                    fitness = evaluate(p, cancel);
                    if (cancel) return;
                }

                // Now, either replace worst point with better point
                if (fitness < B.back().parameter_set.fitness) {
                    B.back().parameter_set = ParameterSet(p, fitness);
                    B.back().values_alias_p = true;
                } else {
                    // Or contract to midpoint between centroid and worst point
                    contraction(B, g, p);
                    sync_alias(B, p);
                    fitness = evaluate(p, cancel);
                    if (cancel) return;

                    // If better than worst point replace worst point
                    if (fitness < B.back().parameter_set.fitness) {
                        B.back().parameter_set = ParameterSet(p, fitness);
                        B.back().values_alias_p = true;
                    } else {
                        // Otherwise perform mutation
                        // Compute the smallest hypercube enclosing A complex and randomly sample
                        // a point within it.
                        smallest_hypercube(Acomplex, prng, p);
                        sync_alias(B, p);
                        fitness = evaluate(p, cancel);
                        if (cancel) return;
                        // Replace worst point with new point regardless of its value
                        B.back().parameter_set = ParameterSet(p, fitness);
                        B.back().values_alias_p = true;
                    }
                }
            }

            // Replace B into A according to L And sort A in order of increasing function value
            for (std::size_t j = 0; j < Acomplex.size(); ++j) {
                for (int k = 0; k < q; ++k) {
                    if (B[static_cast<std::size_t>(k)].index == static_cast<int>(j) &&
                        B[static_cast<std::size_t>(k)].parameter_set.fitness <
                            Acomplex[j].parameter_set.fitness) {
                        Acomplex[j].parameter_set = B[static_cast<std::size_t>(k)].parameter_set.clone();
                    }
                }
            }

            std::sort(Acomplex.begin(), Acomplex.end(), compare_point_fitness);
        }
    }

    // Determines if the point is within the feasible parameter space.
    bool is_feasible(const std::vector<double>& point) const {
        for (std::size_t i = 0; i < point.size(); ++i)
            if (point[i] < lower_bounds_[i] || point[i] > upper_bounds_[i]) return false;
        return true;
    }

    // Compute reflection of worst point about centroid r = 2g - u(worst)
    void reflection(const std::vector<PointFitness>& U, std::vector<double>& g,
                    std::vector<double>& r) const {
        for (int i = 0; i < number_of_parameters_; ++i) {
            std::size_t ii = static_cast<std::size_t>(i);
            g[ii] = 0.0;
            for (std::size_t j = 0; j + 1 < U.size(); ++j) g[ii] += U[j].parameter_set.values[ii];
            g[ii] /= static_cast<double>(U.size() - 1);
            r[ii] = 2.0 * g[ii] - U.back().parameter_set.values[ii];  // u(worst)
        }
    }

    // Contract to midpoint between centroid and worst point
    void contraction(const std::vector<PointFitness>& U, const std::vector<double>& g,
                     std::vector<double>& c) const {
        for (int i = 0; i < number_of_parameters_; ++i) {
            std::size_t ii = static_cast<std::size_t>(i);
            c[ii] = 0.5 * (g[ii] + U.back().parameter_set.values[ii]);  // u(worst)
        }
    }

    // Compute the smallest hypercube enclosing A complex and randomly sample a point within it.
    void smallest_hypercube(const std::vector<PointFitness>& Acomplex,
                            sampling::MersenneTwister& prng, std::vector<double>& z) const {
        for (int i = 0; i < number_of_parameters_; ++i) {
            std::size_t ii = static_cast<std::size_t>(i);
            double low = Acomplex[0].parameter_set.values[ii];
            double high = low;
            for (std::size_t j = 1; j < Acomplex.size(); ++j) {
                low = std::min(low, Acomplex[j].parameter_set.values[ii]);
                high = std::max(high, Acomplex[j].parameter_set.values[ii]);
            }
            z[ii] = low + (high - low) * prng.next_double();
        }
    }

    // The Trapezoidal cumulative probability distribution, for a sample size of N.
    static std::vector<double> trapezoidal(int N) {
        std::vector<double> PP(static_cast<std::size_t>(N));
        // C#: `2 * (N + 1 - i) / (N * (N + 1.0))` -- an INT numerator over a DOUBLE denominator,
        // so this is a double divide, not integer division.
        for (int i = 1; i <= N; ++i)
            PP[static_cast<std::size_t>(i - 1)] =
                static_cast<double>(2 * (N + 1 - i)) / (static_cast<double>(N) * (N + 1.0));
        for (int i = 1; i < N; ++i)
            PP[static_cast<std::size_t>(i)] += PP[static_cast<std::size_t>(i - 1)];  // cumulative sum
        return PP;
    }

    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};

}  // namespace corehydro::numerics::math::optimization
