// ported from: Numerics/Sampling/MCMC/SNIS.cs @ 7e8e8d1
//
// Self-Normalizing Importance Sampling (SNIS): not a Markov chain at all -- every draw is
// independent, either straight from the priors (naive Monte Carlo) or from an optional
// importance distribution `multivariate_normal` (if supplied at construction, or hot-started
// from a successful MAP fit's Fisher-information covariance, inflated 2.5x). `ChainIteration`
// is therefore a no-op stub (never called -- see below); `Sample()` is entirely overridden.
//
// `Sample()` draw order (verbatim): (1) `InitializeChains()` is called for its SIDE EFFECTS
// ONLY (running MAP/DifferentialEvolution when `Initialize == MAP`, setting the base class's
// protected `map_successful_`/`mvn_` -- its own returned initial ParameterSet vector is
// discarded, exactly as C# calls `InitializeChains();` as a bare statement); (2)
// `InitializeCustomSettings()` builds `snis_mvn_` from that MAP fit if importance sampling
// wasn't already configured via the ctor; (3) `ValidateSettings()` -- SNIS's own COMPLETE
// REPLACEMENT of the base validation (not calling the base implementation, since the base
// requires `WarmupIterations >= 1` while SNIS's ctor forces `WarmupIterations == 0` -- these
// are irreconcilable, so C# doesn't call `base.ValidateSettings()` either); (4) `_masterPRNG`
// (this port's `master_prng_`) is RE-CREATED from `PRNGSeed` here, a SECOND time (the first
// being the ordinary one inside `Reset()`/the ctor) -- this is why `MarkovChains`/`Output`/
// `AcceptCount`/`SampleCount`/`MeanLogLikelihood`/`MAP` are all rebuilt inline rather than
// relying on `reset()`'s versions; (5) `_masterPRNG.NextDoubles(Iterations, NumberOfParameters)`
// draws the ENTIRE parameter-uniform grid up front, using the sub-MersenneTwister-per-column
// pattern from `extension_methods.hpp` (P3.2) -- one sub-generator seeded per DIMENSION, each
// advanced `Iterations` times; getting this single call wrong (e.g. drawing per-iteration
// instead) desyncs every downstream value against the real C# output.
//
// `Parallel.For` (twice: the per-draw evaluation loop, and the posterior-weight
// normalization loop) is not ported -- see mcmc_sampler.hpp's file header on why this port is
// single-threaded by design. Both C# loops write to DISTINCT, pre-sized array slots with no
// cross-iteration dependency (unlike RWMH/ARWMH's sequential per-chain PRNG stream), so an
// ordinary `for` loop reproduces the exact same values regardless of iteration order.
//
// SORT-COMPARATOR DISCREPANCY (transcribed verbatim, not "fixed" -- a new upstream finding,
// logged in docs/upstream-csharp-issues.md): the C# comment above the sort call reads "Sort
// list in ascending order of posterior weights", but the comparator itself is
// `(x, y) => x.Fitness.CompareTo(y.Fitness)` -- it sorts by `Fitness` (the raw
// log-likelihood), NOT by `Weight` (the just-computed normalized posterior weight the comment
// describes and the very next lines' CDF construction actually consume). For naive Monte
// Carlo (no importance distribution) `Weight` and `Fitness` coincide, so this is
// unobservable; WITH importance sampling they diverge (`Weight = Fitness - mvn.LogPDF(...)`),
// so the two orderings are genuinely different sorts. `std::stable_sort` matches v2.2.0's
// `OrderBy(x => x.Fitness)`, so tied fitness values retain seeded draw order across runtimes.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/search.hpp"
#include "corehydro/numerics/data/plotting_positions.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::sampling::mcmc {

namespace linalg = corehydro::numerics::math::linalg;
namespace ext = corehydro::numerics::utilities;

class SNIS : public MCMCSampler {
   public:
    // Constructs a self-normalizing importance sampler. `multivariate_normal`: optional
    // importance distribution; if omitted, naive Monte Carlo sampling from the priors is
    // performed (unless a MAP fit later hot-starts one -- see initialize_custom_settings()).
    SNIS(std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors,
         LogLikelihood log_likelihood_function,
         std::optional<distributions::MultivariateNormal> multivariate_normal = std::nullopt)
        : MCMCSampler(std::move(priors), std::move(log_likelihood_function)),
          snis_mvn_(std::move(multivariate_normal)),
          use_importance_sampling_(snis_mvn_.has_value()) {
        // Create default settings -- see gibbs.hpp's file header for why these bypass the
        // reset()-triggering setters.
        number_of_chains_ = 1;
        warmup_iterations_ = 0;
        thinning_interval_ = 1;
        initial_iterations_ = 1;
        iterations_ = 100000;
        output_length = 10000;
        reset();
    }

    // Perform importance sampling. Entirely replaces the base class's driver -- see file
    // header.
    void sample() override {
        initialize_chains();  // Side effects only (MAP fit); return value discarded.
        initialize_custom_settings();
        validate_settings();
        // ParallelizeChains / ResumeSimulation / CancellationTokenSource are omitted --
        // desktop/threading concerns, not math (see mcmc_sampler.hpp's file header).
        resume_simulation = false;

        // Create inputs.
        master_prng_.emplace(static_cast<std::uint32_t>(prng_seed_));
        markov_chains_.assign(1, std::vector<ParameterSet>(static_cast<std::size_t>(iterations_)));
        output_.assign(1, std::vector<ParameterSet>());

        // Create sample & accept counts.
        accept_count_.assign(1, 0);
        sample_count_.assign(1, 0);

        // Create mean log-likelihood list.
        mean_log_likelihood_.clear();

        // Keep track of the best parameter set.
        map_ = ParameterSet({}, -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity());

        // Create parameter random values: the sub-MersenneTwister-per-dimension grid (see
        // file header).
        auto rnds = ext::next_doubles(*master_prng_, iterations_, number_of_parameters());

        // Perform sampling.
        for (int idx = 0; idx < iterations_; ++idx) {
            std::vector<double> parameters(static_cast<std::size_t>(number_of_parameters()));
            double log_lh = 0.0;
            double weight = 0.0;
            if (use_importance_sampling_ && snis_mvn_.has_value()) {
                parameters = snis_mvn_->inverse_cdf(rnds[static_cast<std::size_t>(idx)]);
                log_lh = log_likelihood_function_(parameters);
                weight = log_lh - snis_mvn_->log_pdf(parameters);
            } else {
                for (int i = 0; i < number_of_parameters(); ++i)
                    parameters[static_cast<std::size_t>(i)] =
                        prior_distributions_[static_cast<std::size_t>(i)]->inverse_cdf(
                            rnds[static_cast<std::size_t>(idx)][static_cast<std::size_t>(i)]);
                log_lh = log_likelihood_function_(parameters);
                weight = log_lh;
            }
            // Record sample.
            markov_chains_[0][static_cast<std::size_t>(idx)] = ParameterSet(parameters, log_lh, weight);
        }

        // Get the maximum a posteriori.
        for (int i = 0; i < iterations_; ++i) {
            const auto& ps = markov_chains_[0][static_cast<std::size_t>(i)];
            if (is_finite(ps.weight) && ps.weight > map_.weight) map_ = ps.clone();
        }

        // Get the normalization factor.
        double max_w = map_.weight;
        if (!is_finite(max_w))
            throw std::runtime_error("SNIS failed because all importance weights are non-finite.");

        double sum = 0.0;
        for (int i = 0; i < iterations_; ++i) {
            const auto& ps = markov_chains_[0][static_cast<std::size_t>(i)];
            if (is_finite(ps.weight)) sum += std::exp(ps.weight - max_w);
        }
        if (!is_finite(sum) || sum <= 0.0)
            throw std::runtime_error("SNIS failed because the finite importance weights could not be normalized.");

        double normalization = max_w + std::log(sum);
        if (!is_finite(normalization))
            throw std::runtime_error("SNIS failed because the importance-weight normalization is non-finite.");

        // Compute the posterior weights.
        for (int idx = 0; idx < iterations_; ++idx) {
            const auto& ps = markov_chains_[0][static_cast<std::size_t>(idx)];
            double w = is_finite(ps.weight) ? std::exp(ps.weight - normalization) : 0.0;
            markov_chains_[0][static_cast<std::size_t>(idx)] = ParameterSet(ps.values, ps.fitness, w);
        }

        // Sort list in ascending order -- BY FITNESS, not by the just-computed Weight; see
        // file header's SORT-COMPARATOR DISCREPANCY note.
        std::stable_sort(markov_chains_[0].begin(), markov_chains_[0].end(),
                          [](const ParameterSet& a, const ParameterSet& b) { return a.fitness < b.fitness; });

        std::vector<double> cdf(static_cast<std::size_t>(iterations_));
        cdf[0] = std::max(0.0, markov_chains_[0][0].weight);

        // Create CDF.
        for (int i = 1; i < iterations_; ++i)
            cdf[static_cast<std::size_t>(i)] =
                cdf[static_cast<std::size_t>(i - 1)] +
                std::max(0.0, markov_chains_[0][static_cast<std::size_t>(i)].weight);

        // Record output.
        auto rnd_out = data::plotting_positions::weibull(output_length);
        int idx = 0;
        for (int i = 0; i < output_length; ++i) {
            idx = data::search::sequential(rnd_out[static_cast<std::size_t>(i)], cdf, idx);
            output_[0].push_back(markov_chains_[0][static_cast<std::size_t>(idx)].clone());
        }
        // C# does not increment `Simulations` on this path (no `Simulations += 1` anywhere
        // in SNIS.Sample()) -- transcribed verbatim; harmless since `ResumeSimulation` is
        // forced false above, so every Sample() call fully reinitializes regardless.
    }

   protected:
    void initialize_custom_settings() override {
        if (!snis_mvn_.has_value() && !use_importance_sampling_ && initialize == InitializationType::MAP &&
            map_successful_ && mvn_.has_value()) {
            // Copy-construct from the MAP-fit MVN (equivalent to C#'s `(MultivariateNormal)
            // _MVN.Clone()` -- MultivariateNormal's own Clone() override just deep-copies its
            // member state, the same thing this port's implicit copy constructor does; see
            // multivariate_normal.hpp's file header on the MVNUNI value-copy).
            distributions::MultivariateNormal inflated = *mvn_;
            auto covar = inflated.covariance();
            for (auto& row : covar)
                for (auto& v : row) v *= 2.5;  // Inflate to cover the posterior.
            inflated.set_parameters(inflated.mean(), covar);
            snis_mvn_ = std::move(inflated);
            use_importance_sampling_ = true;
        }
    }

    // Complete replacement of the base validation (not calling it) -- see file header.
    void validate_settings() override {
        if (number_of_chains_ != 1) throw std::invalid_argument("There can only be 1 chain with this method.");
        if (output_length < 100) throw std::invalid_argument("The output length must be at least 100.");
        if (iterations_ < output_length)
            throw std::invalid_argument("The number of iterations cannot be less than the output length.");
        if (warmup_iterations_ != 0) throw std::invalid_argument("There are no warmup iterations with this method.");
        if (thinning_interval_ != 1) throw std::invalid_argument("The thinning interval must be 1 for this method.");
        if (initial_iterations_ != 1) throw std::invalid_argument("The initial population must be 1 for this method.");
        if (snis_mvn_.has_value() && !snis_mvn_->parameters_valid())
            throw std::invalid_argument("The multivariate Normal importance distribution is invalid.");
    }

    // Never called: SNIS's own sample() never invokes sample_chain()/chain_iteration(). Kept
    // as a no-op stub only to satisfy the base class's pure virtual (matches C#'s own no-op
    // override, `return new ParameterSet();`).
    ParameterSet chain_iteration(int /*index*/, ParameterSet /*state*/) override { return ParameterSet(); }

   private:
    static bool is_finite(double value) { return !std::isnan(value) && !std::isinf(value); }

    // Declaration order matters: the ctor's initializer list sets `use_importance_sampling_`
    // from `snis_mvn_.has_value()`, and member initialization follows DECLARATION order (not
    // initializer-list order) -- `snis_mvn_` must therefore be declared first.
    std::optional<distributions::MultivariateNormal> snis_mvn_;
    bool use_importance_sampling_ = false;
};

}  // namespace corehydro::numerics::sampling::mcmc
