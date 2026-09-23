// ported from: Numerics/Sampling/MCMC/DEMCzs.cs @ 7e8e8d1
//
// Differential Evolution Markov Chain with snooker update (DE-MCzs): DEMCz (see demcz.hpp) plus
// a `SnookerThreshold`-probability "snooker update" move that, once the chain is past a short
// warmup (`SampleCount[index] > 5 * ThinningInterval`), proposes along the projection of two
// OTHER chains' current directions onto the line connecting this chain's state and a third
// chain's state, with a Jacobian correction term for the non-unit projection Jacobian
// (Braak & Vrugt 2008, section 3.2). Two DELIBERATE differences from DEMCz worth flagging
// (both transcribed verbatim, not typos): (1) DEMCzs's `InitialIterations = 100 *
// NumberOfParameters` (DEMCz uses `100 * NumberOfChains`); (2) DEMCzs's jump-threshold test
// uses `<=` (DEMCz uses strict `<`), and DEMCzs's `JumpThreshold`/`SnookerThreshold` validation
// bounds are INCLUSIVE on the upper end (`> 1`/`> 0.5`) where DEMCz's `JumpThreshold` bound is
// EXCLUSIVE (`>= 1`).
//
// CROSS-LANGUAGE REPRODUCIBILITY (both this class and demcz.hpp, `IsPopulationSampler = true`):
// unlike RWMH/ARWMH's per-chain-independent proposals, every chain here reads `PopulationMatrix`
// -- a SINGLE, ever-growing pool every chain writes into and every chain proposes from. A
// sub-ULP accept/reject divergence anywhere (occasionally inevitable over enough draws, same
// root cause as the RWMH MAP-path/BrentSearch findings) immediately contaminates every OTHER
// chain's future proposals through this shared pool, so long-run aggregates (`AcceptanceRates`,
// and anything derived from `Output`, e.g. `ESS`) are measurably less cross-language-reproducible
// than the early-`MarkovChains` digest window, even though the underlying draw STREAM is
// bit-identical. See `fixtures/README.md`'s "Population-sampler divergence finding" for the full
// diagnosis (confirmed via an exhaustive draw-by-draw sweep against the real C# library) and the
// resulting tolerance policy for `demcz.json`/`demczs.json`.
//
// ChainIteration draw order (verbatim): (1) ONE uniform for the snooker-vs-parallel-direction
// test (`_chainPRNGs[index].NextDouble() <= SnookerThreshold`) -- drawn UNCONDITIONALLY every
// iteration, since it is the left operand of a `&&` (C#/C++ `&&` both short-circuit on the LEFT
// operand's truthiness, so the right operand, `SampleCount[index] > 5 * ThinningInterval`, is
// evaluated -- with no PRNG draw of its own -- only when the left operand is true). If that test
// passes, control transfers to SnookerUpdate() (below) and NO further draws happen in
// ChainIteration itself. Otherwise: (2) ONE uniform for the jump-vs-big-jump test (`<=
// JumpThreshold`, not `<` -- see file header); (3) ONE integer draw for R1 (`Next(0, M)`, where
// `M = PopulationMatrix.Count` -- UNLIKE DEMCz, there is NO `M < 2` guard here, transcribed
// faithfully); (4) a distinctness re-draw loop for R2; (5) `D` interleaved uniforms, one per
// parameter, each immediately consumed by `_b.InverseCDF` (same interleaving as DEMCz -- see its
// file header); (6) ONE more uniform for the accept/reject draw.
//
// SnookerUpdate() draw order (verbatim): (1) ONE uniform, consumed by `_g.InverseCDF`
// (`_g = Uniform(1.2, 2.2)`), for the snooker jump scale `G`; (2) a re-draw loop for chain index
// `c` (`Next(0, NumberOfChains)` until `c != index`); (3) a re-draw loop for `c1` (until `c1 !=
// c` -- NOT also excluding `index`); (4) a re-draw loop for `c2` (until `c2 != c1 && c2 != c` --
// again NOT also excluding `index`); (5) ONE more uniform for the accept/reject draw at the end
// (there is no further draw for the Jacobian-correction underflow guard -- that branch returns
// `state` with no PRNG consumption). SNAPSHOT SEMANTICS (the brief's flagged off-by-one hazard):
// `int n = MarkovChains[c].Count();` is computed ONCE, from chain `c` ONLY, and that SAME `n` is
// then reused to index `MarkovChains[c][n-1]`, `MarkovChains[c1][n-1]`, AND `MarkovChains[c2][n-1]`
// -- NOT `MarkovChains[c1].Count()-1`/`MarkovChains[c2].Count()-1` computed independently per
// chain. This is safe (not a bug) because `mcmc_sampler.hpp`'s `sample()` "Record output" loop
// appends exactly one state to EVERY chain's `MarkovChains[j]` per outer iteration, in lockstep
// -- so at any point during `ChainIteration`, every chain's `MarkovChains[j].size()` is equal,
// and reusing `n` is equivalent to (but cheaper than) three independent `.size()` calls. The
// warmup gate (`SampleCount[index] > 5 * ThinningInterval`) additionally guarantees `n >= 1`
// whenever SnookerUpdate() runs (SampleCount[index] cannot exceed `5 * ThinningInterval` until
// at least 5 outer `sample()` iterations have completed, by which point every chain -- including
// `c`/`c1`/`c2` -- has at least 5 recorded states), so `MarkovChains[c][n - 1]` never indexes an
// empty history. All THREE lookups reference `MarkovChains` (the per-outer-iteration recorded
// snapshot), never `PopulationMatrix` (the flat, ever-growing cross-chain pool DEMCz's own
// parallel-direction branch draws from) -- transcribing which collection at which index exactly
// is the hazard this file header calls out per the task brief.
#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"

namespace corehydro::numerics::sampling::mcmc {

namespace linalg = corehydro::numerics::math::linalg;

class DEMCzs : public MCMCSampler {
   public:
    // Constructs a new DEMCzs sampler.
    DEMCzs(std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors,
           LogLikelihood log_likelihood_function)
        : MCMCSampler(std::move(priors), std::move(log_likelihood_function)) {
        set_initial_iterations(100 * number_of_parameters());
        // DE-MCz options.
        is_population_sampler_ = true;
        // Jump parameter. Default = 2.38/SQRT(2*D).
        jump = 2.38 / std::sqrt(2.0 * number_of_parameters());
        // Adaptation threshold. Default = 0.1 or 10% of the time.
        jump_threshold = 0.1;
        // Snooker update. Default = 0.1 or 10% of the time.
        snooker_threshold = 0.1;
        b_ = distributions::Normal(0.0, noise_);
        g_ = distributions::Uniform(1.2, 2.2);
    }

    // The jumping parameter used to jump from one mode region to another in the target
    // distribution. Plain public field (matches the C# auto-property; no reset() side effect).
    double jump = 0.0;

    // Determines how often the jump parameter switches to 1.0; e.g., 0.10 will result in a
    // large jump 10% of the time. Plain public field (matches the C# auto-property; no reset()
    // side effect).
    double jump_threshold = 0.0;

    // Determines how often to perform the snooker update; e.g., 0.10 will result in an update
    // 10% of the time. Plain public field (matches the C# auto-property; no reset() side
    // effect).
    double snooker_threshold = 0.0;

    // The noise parameter (b). Setting it rebuilds the internal `Normal(0, Noise)` used to
    // draw the per-parameter noise term `e` -- mirrors the C# custom property setter.
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
        if (jump_threshold < 0.0 || jump_threshold > 1.0)
            throw std::invalid_argument("The jump threshold must be between 0 and 1.");
        if (snooker_threshold < 0.0 || snooker_threshold > 0.5)
            throw std::invalid_argument("The snooker threshold must be between 0 and 0.5.");
        if (noise_ < 0.0) throw std::invalid_argument("The noise parameter must be greater than 0.");
    }

    ParameterSet chain_iteration(int index, ParameterSet state) override {
        // Update the sample count.
        sample_count_[static_cast<std::size_t>(index)] += 1;

        // 10% snooker updates and 90% parallel direction updates. The left-operand uniform is
        // drawn UNCONDITIONALLY every iteration -- see file header.
        if (chain_prngs_[static_cast<std::size_t>(index)].next_double() <= snooker_threshold &&
            sample_count_[static_cast<std::size_t>(index)] > 5 * thinning_interval_) {
            return snooker_update(index, state);
        }

        // The adaptation for the algorithm to allow for jumps from one mode region to
        // another in the target distribution.
        double G = chain_prngs_[static_cast<std::size_t>(index)].next_double() <= jump_threshold ? 1.0 : jump;

        // Sample uniformly at random without replacement two numbers R1 and R2 from the
        // numbers 1, 2, ..., M. Unlike DEMCz, no `M < 2` guard -- transcribed faithfully.
        int M = static_cast<int>(population_matrix_.size());
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
    // Returns a proposed MCMC iteration based on the Snooker Update method. `index`: the
    // Markov Chain zero-based index. `state`: the current chain state to compare against.
    ParameterSet snooker_update(int index, ParameterSet state) {
        // Get Jump -- uniform random number between 1.2 and 2.2.
        double G = g_.inverse_cdf(chain_prngs_[static_cast<std::size_t>(index)].next_double());

        // Select another chain, which is in state z.
        int c = index;
        do {
            c = chain_prngs_[static_cast<std::size_t>(index)].next(0, number_of_chains());
        } while (c == index);
        // Select two other random chains, zR1 and zR2.
        int c1 = c;
        int c2 = c;
        do {
            c1 = chain_prngs_[static_cast<std::size_t>(index)].next(0, number_of_chains());
        } while (c1 == c);
        do {
            c2 = chain_prngs_[static_cast<std::size_t>(index)].next(0, number_of_chains());
        } while (c2 == c1 || c2 == c);

        // Define z. See file header for why `n` is computed ONCE (from chain `c` only) and
        // reused for `c1`/`c2`.
        int n = static_cast<int>(markov_chains_[static_cast<std::size_t>(c)].size());
        linalg::Vector z(markov_chains_[static_cast<std::size_t>(c)][static_cast<std::size_t>(n - 1)].values);
        linalg::Vector xi(state.values);
        // Define line xi - z.
        linalg::Vector line = xi - z;
        // Orthogonally project zR1 and zR2 onto the line xi - z.
        linalg::Vector zr1(markov_chains_[static_cast<std::size_t>(c1)][static_cast<std::size_t>(n - 1)].values);
        linalg::Vector zr2(markov_chains_[static_cast<std::size_t>(c2)][static_cast<std::size_t>(n - 1)].values);
        linalg::Vector zp1 = linalg::Vector::project(zr1, line);
        linalg::Vector zp2 = linalg::Vector::project(zr2, line);

        // Calculate the proposal vector.
        // x* <- xi + gamma(zP1 - zP2)
        linalg::Vector xp(number_of_parameters());
        for (int i = 0; i < number_of_parameters(); ++i) {
            xp[i] = xi[i] + G * (zp1[i] - zp2[i]);

            // Check if the parameter is feasible (within the constraints).
            const auto& prior = prior_distributions_[static_cast<std::size_t>(i)];
            if (xp[i] < prior->minimum() || xp[i] > prior->maximum()) {
                // The proposed parameter vector was infeasible, so leave xi unchanged.
                return state;
            }
        }

        // Evaluate fitness.
        double log_lh_p = log_likelihood_function_(xp.to_array());
        double log_lh_i = state.fitness;

        // Euclidean distance terms for Jacobian correction.
        double dist_p = linalg::Vector::distance(xp, z);
        double dist_i = linalg::Vector::distance(xi, z);

        // Avoid underflow in log terms.
        if (dist_p < 1e-12 || dist_i < 1e-12) return state;

        double log_ed_p = (number_of_parameters() - 1) * std::log(dist_p);
        double log_ed_i = (number_of_parameters() - 1) * std::log(dist_i);

        // Calculate the Metropolis ratio.
        double log_ratio = log_lh_p + log_ed_p - log_lh_i - log_ed_i;
        // Accept the proposal with probability min(1, r); otherwise leave xi unchanged.
        double log_u = std::log(chain_prngs_[static_cast<std::size_t>(index)].next_double());
        if (log_u <= log_ratio) {
            // The proposal is accepted.
            accept_count_[static_cast<std::size_t>(index)] += 1;
            return ParameterSet(xp.to_array(), log_lh_p);
        }
        return state;
    }

    double noise_ = 1e-12;
    distributions::Normal b_;
    distributions::Uniform g_;
};

}  // namespace corehydro::numerics::sampling::mcmc
