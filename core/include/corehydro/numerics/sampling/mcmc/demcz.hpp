// ported from: Numerics/Sampling/MCMC/DEMCz.cs @ 7e8e8d1
//
// Differential Evolution Markov Chain (DE-MCz): a POPULATION sampler (`IsPopulationSampler =
// true`, requiring >= 3 chains). Instead of proposing from a per-chain distribution centered
// at the current state (RWMH/ARWMH), each chain proposes a jump along the difference vector
// between two OTHER, uniformly-at-random-without-replacement-drawn rows of the shared
// `PopulationMatrix` (every chain's every recorded state, accumulated across the whole run --
// see mcmc_sampler.hpp's `sample()` "Record output" loop): `x* <- xi + G*(zR1 - zR2) + e`,
// where `G` is `Jump` (default `2.38 / sqrt(2*D)`, the literature-standard DE-MC scaling
// constant) with a `JumpThreshold`-probability chance of being forced to 1.0 (a "big jump" that
// lets the sampler cross between disconnected modes), and `e` is per-parameter noise drawn from
// `Normal(0, Noise)` via its inverse-CDF.
//
// ChainIteration draw order (verbatim): (1) ONE uniform for the jump-vs-big-jump test
// (`_chainPRNGs[index].NextDouble() < JumpThreshold`, STRICT `<`, unlike DEMCzs's `<=` -- see
// demczs.hpp); (2) ONE integer draw (`Next(0, M)`) for R1; (3) a DISTINCTNESS re-draw loop (`do
// r2 = Next(0, M); while (r2 == r1);`) for R2 -- one or more integer draws, consuming exactly as
// many as it takes to land on a value != r1; (4) `D` interleaved uniforms, ONE PER PARAMETER,
// each immediately consumed by `_b.InverseCDF` to produce that parameter's noise term `e` (NOT a
// single `NextDoubles(D)` batch draw like RWMH/ARWMH -- the noise draw is INSIDE the per-
// parameter loop, interleaved with the (non-drawing) feasibility check on each iteration); (5)
// ONE more uniform for the accept/reject draw. `M = PopulationMatrix.Count`; DEMCz (unlike
// DEMCzs) explicitly guards `M < 2` with a thrown exception before drawing R1 -- transcribed
// verbatim below, not one of the "guard for C++ UB where C# throws" cases (C# ITSELF already
// guards this explicitly; no fixup needed).
//
// `InitialIterations = 100 * NumberOfChains` -- NOTE this is `NumberOfChains`, not
// `NumberOfParameters` as RWMH/ARWMH/DEMCzs (below) all use. Transcribed exactly as written; not
// a typo to "fix".
#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"

namespace corehydro::numerics::sampling::mcmc {

class DEMCz : public MCMCSampler {
   public:
    // Constructs a new DEMCz sampler.
    DEMCz(std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors,
          LogLikelihood log_likelihood_function)
        : MCMCSampler(std::move(priors), std::move(log_likelihood_function)) {
        set_initial_iterations(100 * number_of_chains());
        // DE-MCz options.
        is_population_sampler_ = true;
        // Jump parameter. Default = 2.38/SQRT(2*D).
        jump = 2.38 / std::sqrt(2.0 * number_of_parameters());
        // Jump threshold. Default = 0.1 or 10% of the time.
        jump_threshold = 0.1;
        b_ = distributions::Normal(0.0, noise_);
    }

    // The jumping parameter used to jump from one mode region to another in the target
    // distribution. Plain public field (matches the C# auto-property `Jump { get; set; }`; no
    // reset() side effect).
    double jump = 0.0;

    // Determines how often the jump parameter switches to 1.0; e.g., 0.10 will result in a
    // large jump 10% of the time. Plain public field (matches the C# auto-property
    // `JumpThreshold { get; set; }`; no reset() side effect).
    double jump_threshold = 0.0;

    // The noise parameter (b). Setting it rebuilds the internal `Normal(0, Noise)` used to
    // draw the per-parameter noise term `e` -- mirrors the C# custom property setter
    // (`Noise { get; set { _noise = value; _b = new Normal(0, _noise); } }`).
    double noise() const { return noise_; }
    void set_noise(double value) {
        noise_ = value;
        b_ = distributions::Normal(0.0, noise_);
    }

   protected:
    void validate_custom_settings() override {
        if (number_of_chains() < 3) throw std::invalid_argument("There must be at least 3 chains.");
        if (jump <= 0.0 || jump >= 2.0)
            throw std::invalid_argument("The jump parameter must be between 0 and 2.");
        if (jump_threshold < 0.0 || jump_threshold >= 1.0)
            throw std::invalid_argument("The jump threshold must be between 0 and 1.");
        if (noise_ < 0.0) throw std::invalid_argument("The noise parameter must be greater than 0.");
    }

    ParameterSet chain_iteration(int index, ParameterSet state) override {
        // Update the sample count.
        sample_count_[static_cast<std::size_t>(index)] += 1;

        // The adaptation for the algorithm to allow for jumps from one mode region to
        // another in the target distribution.
        double G = chain_prngs_[static_cast<std::size_t>(index)].next_double() < jump_threshold ? 1.0 : jump;

        // Sample uniformly at random without replacement two numbers R1 and R2 from the
        // numbers 1, 2, ..., M.
        int M = static_cast<int>(population_matrix_.size());
        if (M < 2) throw std::runtime_error("PopulationMatrix must contain at least 2 elements.");
        int r1 = chain_prngs_[static_cast<std::size_t>(index)].next(0, M);
        int r2;
        do {
            r2 = chain_prngs_[static_cast<std::size_t>(index)].next(0, M);
        } while (r2 == r1);

        // Calculate the proposal vector.
        // x* <- xi + gamma(zR1 - zR2) + e
        // where zR1 and zR2 are rows R1 and R2 of the population matrix.
        std::vector<double> xp(static_cast<std::size_t>(number_of_parameters()));
        for (int i = 0; i < number_of_parameters(); ++i) {
            double xi = state.values[static_cast<std::size_t>(i)];
            double zr1 = population_matrix_[static_cast<std::size_t>(r1)].values[static_cast<std::size_t>(i)];
            double zr2 = population_matrix_[static_cast<std::size_t>(r2)].values[static_cast<std::size_t>(i)];
            double e = b_.inverse_cdf(chain_prngs_[static_cast<std::size_t>(index)].next_double());
            xp[static_cast<std::size_t>(i)] = xi + G * (zr1 - zr2) + e;

            // Check if the parameter is feasible (within the constraints).
            const auto& prior = prior_distributions_[static_cast<std::size_t>(i)];
            if (xp[static_cast<std::size_t>(i)] < prior->minimum() || xp[static_cast<std::size_t>(i)] > prior->maximum()) {
                // The proposed parameter vector was infeasible, so leave xi unchanged.
                return state;
            }
        }

        // Evaluate fitness.
        double log_lh_p = log_likelihood_function_(xp);
        double log_lh_i = state.fitness;

        // Calculate the Metropolis ratio.
        double log_ratio = log_lh_p - log_lh_i;

        // Accept the proposal with probability min(1, r); otherwise leave xi unchanged.
        double log_u = std::log(chain_prngs_[static_cast<std::size_t>(index)].next_double());
        if (log_u <= log_ratio) {
            // The proposal is accepted.
            accept_count_[static_cast<std::size_t>(index)] += 1;
            return ParameterSet(xp, log_lh_p);
        }
        return state;
    }

   private:
    double noise_ = 1e-12;
    distributions::Normal b_;
};

}  // namespace corehydro::numerics::sampling::mcmc
