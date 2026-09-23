// ported from: Numerics/Sampling/MCMC/Base/MCMCSampler.cs @ 7e8e8d1
//
// Abstract base class for every Bayesian MCMC sampler (RWMH, and later ARWMH/DEMCz/DEMCzs/
// HMC/NUTS/Gibbs/SNIS): inputs (PRNG seed, chain/iteration/thinning counts, initialization
// mode), the shared simulation driver (Sample/InitializeChains/SampleChain), and outputs
// (MarkovChains, Output, MAP, acceptance rates, mean log-likelihood).
//
// `LogLikelihood` (C# `public delegate double LogLikelihood(double[] parameters);`)
// INCLUDES the prior log-density -- the caller's closure is responsible for summing data
// likelihood + prior likelihood, exactly as the C# XML doc says: "This function should
// account for the data likelihood as well as the prior likelihood of the model
// parameters." The `prior_distributions_` held by the sampler itself are used ONLY for (a)
// feasibility bounds (RWMH's ChainIteration rejects a proposal outside [Minimum, Maximum])
// and (b) chain initialization (InitializeChains' Randomize/MAP paths) -- never re-added to
// the likelihood internally.
//
// `priors` is `std::vector<std::shared_ptr<UnivariateDistributionBase>>` -- a DOCUMENTED
// deviation from this core's usual `unique_ptr` ownership norm. The Gibbs sampler's
// conjugate-prior proposal (a later task) mutates the SAME prior objects the sampler holds,
// mirroring the C# source's reference-type aliasing (`List<IUnivariateDistribution>` is a
// list of references in C#; MCMCSampler and a Gibbs conjugate proposal can hold and mutate
// the same underlying distribution instance). `shared_ptr` reproduces that aliasing;
// `unique_ptr` could not (only one owner could mutate it).
//
// OMITTED (desktop/threading/eventing concerns, not math): `ParallelizeChains`,
// `ProgressChanged`/`ReportProgress`/`ProgressChangedRate`, `CancellationTokenSource`/
// `CancelSimulation`, `[Serializable]`. C# runs the `NumberOfChains` chains via
// `Parallel.For` when `ParallelizeChains` (default true); each chain draws ONLY from its own
// per-chain `MersenneTwister` and all shared state (MarkovChains/Output/AcceptCount/
// SampleCount/MeanLogLikelihood/MAP/PopulationMatrix) is written back SEQUENTIALLY in the
// "Record output" loop directly below the parallel block -- so serial execution (this port,
// always) reproduces the exact same per-chain streams and the exact same sequential
// record-phase writes as the C# `Parallel.For` path, bit-for-bit. This port is
// single-threaded by design (no threading dependency for CRAN/PyPI); the C#
// `ParallelizeChains == false` branch (an explicit sequential `for` loop, functionally
// identical to the `Parallel.For` case for our purposes) is what `sample()` below mirrors.
//
// Reset() seeding cascade (C# lines ~602-622) is transcribed VERBATIM: `master_prng_ =
// MersenneTwister(prng_seed_)` is constructed FIRST; then, in chain-index order 0..N-1,
// `chain_prngs_[i] = MersenneTwister(master_prng_.next())` -- exactly one `master_prng_`
// draw per chain, consumed strictly in order. Getting this order (or interleaving any other
// master-PRNG draw among these calls) wrong desyncs every downstream chain's stream from
// the C# oracle.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/math/linalg/lu_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/differential_evolution.hpp"
#include "corehydro/numerics/math/optimization/support/optimization_status.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/latin_hypercube.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::sampling::mcmc {

namespace opt = corehydro::numerics::math::optimization;
using opt::ParameterSet;

// The log-likelihood function to evaluate. Must include the prior log-density -- see file
// header.
using LogLikelihood = std::function<double(const std::vector<double>&)>;

class MCMCSampler {
   public:
    // Constructs a new MCMC sampler.
    MCMCSampler(std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors,
                LogLikelihood log_likelihood_function)
        : log_likelihood_function_(std::move(log_likelihood_function)),
          prior_distributions_(std::move(priors)) {
        reset();
    }

    virtual ~MCMCSampler() = default;

    // --- Inputs ----------------------------------------------------------------------------

    // Enumerates the chain-initialization strategies.
    enum class InitializationType {
        // Initialize the chains using the Maximum a Posteriori (MAP) estimate and
        // covariance matrix. If the MAP optimization fails, chains automatically fall back
        // to random samples from the priors.
        MAP,
        // Automatically initialize the chains with random samples from the priors. Default.
        Randomize,
        // Initialize the chains from user-defined points (the last state of each existing
        // chain -- requires a prior Sample() call).
        UserDefined,
    };

    // The pseudo random number generator (PRNG) seed.
    int prng_seed() const { return prng_seed_; }
    void set_prng_seed(int value) {
        prng_seed_ = value;
        reset();
    }

    // The number of iterations used to initialize the chains. Recommended >= 10x the
    // number of parameters.
    int initial_iterations() const { return initial_iterations_; }
    void set_initial_iterations(int value) {
        initial_iterations_ = value;
        reset();
    }

    // The number of warm up MCMC iterations to discard at the beginning of the simulation
    // (post-hoc, in diagnostics only -- see sample()'s comment on why WarmupIterations does
    // not gate anything inside the sampling loop itself).
    int warmup_iterations() const { return warmup_iterations_; }
    void set_warmup_iterations(int value) {
        warmup_iterations_ = value;
        reset();
    }

    // The number of MCMC iterations to simulate (post-warmup).
    int iterations() const { return iterations_; }
    void set_iterations(int value) {
        iterations_ = value;
        reset();
    }

    // The number of Markov Chains.
    int number_of_chains() const { return number_of_chains_; }
    void set_number_of_chains(int value) {
        number_of_chains_ = value;
        reset();
    }

    // The thinning interval: how often MCMC iterations are recorded and evaluated.
    int thinning_interval() const { return thinning_interval_; }
    void set_thinning_interval(int value) {
        thinning_interval_ = value;
        reset();
    }

    // The Log-Likelihood function to evaluate.
    const LogLikelihood& log_likelihood_function() const { return log_likelihood_function_; }

    // The list of prior distributions for the model parameters.
    const std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>>& prior_distributions()
        const {
        return prior_distributions_;
    }

    // The number of parameters to evaluate.
    int number_of_parameters() const { return static_cast<int>(prior_distributions_.size()); }

    // Number of transitions performed by one chain in sample(). Warmup is already a
    // subset of iterations and is not added separately. Invalid zero-chain settings
    // report zero rather than dividing by zero before validate_settings() runs.
    std::int64_t transition_count() const {
        if (number_of_chains_ < 1) return 0;
        return (static_cast<std::int64_t>(iterations_) + output_iterations()) *
               static_cast<std::int64_t>(thinning_interval_);
    }

    // Total transition work across all chains. This is not a likelihood evaluation
    // count for gradient samplers, which perform multiple evaluations per transition.
    std::int64_t total_transition_count() const {
        return transition_count() * static_cast<std::int64_t>(number_of_chains_);
    }

    // Whether to update the population matrix when chain states are recorded.
    bool is_population_sampler() const { return is_population_sampler_; }

    // Whether the MCMC simulation should be resumed on the next Sample() call. Default =
    // true (plain public field, no reset() side effect -- matches the C# auto-property).
    bool resume_simulation = true;

    // Determines whether to initialize chains via MAP, Randomize, or UserDefined. Default =
    // Randomize (plain public field, no reset() side effect -- matches the C# auto-property;
    // note InitializeChains() itself may flip this to Randomize on a failed MAP attempt).
    InitializationType initialize = InitializationType::Randomize;

    // Whether MAP initialization was attempted but failed, falling back to random
    // initialization.
    bool map_initialization_failed() const { return map_initialization_failed_; }

    // --- UserDefined seeding hook (ADDITIVE, X2) -------------------------------------------
    //
    // The C# MixtureAnalysis.RunAsync EM-seeds the sampler by mutating
    // `sampler.PopulationMatrix` (`.Add(...)`) and `sampler.MarkovChains[i]` (`.Add(...)`)
    // DIRECTLY, after `SetUpSampler()` and just before `Sample()`, with `Initialize =
    // UserDefined` and NO intervening setter. A caller of this port cannot mutate
    // `population_matrix_` / `markov_chains_` for the same effect, because the LAST
    // reset-triggering setter it touches would wipe the seed (reset() clears
    // `population_matrix_` and re-assigns `markov_chains_` to N empty chains), and any harness
    // that re-seeds then flips a setter would silently lose it.
    //
    // These hooks store the seed in members reset() does NOT clear (`seeded_population_` /
    // `seeded_chains_`). When `initialize == InitializationType::UserDefined`,
    // `initialize_chains()` re-materializes `population_matrix_` / `markov_chains_` from the
    // stored seed at the point sample() consumes them -- so the seed SURVIVES a reset()
    // triggered after seeding, mirroring the C#'s effective ordering. ADDITIVE ONLY: for a
    // never-seeded sampler both members are empty and the Randomize / MAP paths are
    // byte-identical (the existing seeded-chain digest fixtures guard this).

    // Seeds the whole population matrix used by population-based samplers (mirrors the C#
    // `sampler.PopulationMatrix.Add(...)` loop collapsed to a single assignment). Honored only
    // when `initialize == InitializationType::UserDefined`.
    void seed_population(std::vector<ParameterSet> population) {
        seeded_population_ = std::move(population);
    }

    // Appends a user-defined initial state for chain `chain_index` (mirrors the C#
    // `sampler.MarkovChains[i].Add(...)` seed). Grows the seed store as needed. Honored only
    // when `initialize == InitializationType::UserDefined`.
    void seed_chain(int chain_index, ParameterSet state) {
        if (chain_index < 0) throw std::out_of_range("chain_index must be non-negative.");
        if (static_cast<std::size_t>(chain_index) >= seeded_chains_.size())
            seeded_chains_.resize(static_cast<std::size_t>(chain_index) + 1);
        seeded_chains_[static_cast<std::size_t>(chain_index)].push_back(std::move(state));
    }

    // Clears any stored UserDefined seed (both the population and the per-chain seeds).
    void clear_seed() {
        seeded_population_.clear();
        seeded_chains_.clear();
    }

    // --- Outputs ---------------------------------------------------------------------------

    // The population matrix used for population-based samplers.
    const std::vector<ParameterSet>& population_matrix() const { return population_matrix_; }

    // The list of sampled Markov Chains (one vector of ParameterSet per chain).
    const std::vector<std::vector<ParameterSet>>& markov_chains() const { return markov_chains_; }

    // The number of accepted samples per chain.
    const std::vector<int>& accept_count() const { return accept_count_; }

    // The number of calls to the proposal sampler per chain.
    const std::vector<int>& sample_count() const { return sample_count_; }

    // The acceptance rate per chain.
    std::vector<double> acceptance_rates() const {
        std::vector<double> ar(static_cast<std::size_t>(number_of_chains_));
        for (int i = 0; i < number_of_chains_; ++i) {
            ar[static_cast<std::size_t>(i)] =
                sample_count_[static_cast<std::size_t>(i)] > 0
                    ? static_cast<double>(accept_count_[static_cast<std::size_t>(i)]) /
                          static_cast<double>(sample_count_[static_cast<std::size_t>(i)])
                    : 0.0;
        }
        return ar;
    }

    // Acceptance statistic reported in MCMCResults. NUTS overrides this with its mean
    // Hamiltonian acceptance probability because every NUTS transition retains a state.
    virtual std::vector<double> reported_acceptance_rates() const {
        return acceptance_rates();
    }

    // The average log-likelihood across each chain for each iteration.
    const std::vector<double>& mean_log_likelihood() const { return mean_log_likelihood_; }

    // The number of posterior parameter sets to output. Default = 10000 (plain public
    // field, no reset() side effect -- matches the C# auto-property).
    int output_length = 10000;

    // Output posterior parameter sets, recorded after `iterations()` have completed (one
    // vector of ParameterSet per chain).
    const std::vector<std::vector<ParameterSet>>& output() const { return output_; }

    // The output parameter set that produced the maximum likelihood (the maximum a
    // posteriori, MAP).
    const ParameterSet& map() const { return map_; }

    // --- Simulation methods ------------------------------------------------------------

    // Resets simulation results: reseeds the master/chain PRNGs and clears every output
    // container. Called automatically by every reset-triggering setter above, and once from
    // the constructor. See the file header for the exact seeding cascade this must
    // reproduce.
    void reset() {
        simulations_ = 0;
        map_initialization_failed_ = false;

        master_prng_.emplace(static_cast<std::uint32_t>(prng_seed_));
        chain_prngs_.clear();
        chain_prngs_.reserve(static_cast<std::size_t>(number_of_chains_));
        population_matrix_.clear();
        markov_chains_.assign(static_cast<std::size_t>(number_of_chains_), std::vector<ParameterSet>());
        output_.assign(static_cast<std::size_t>(number_of_chains_), std::vector<ParameterSet>());
        for (int i = 0; i < number_of_chains_; ++i) {
            // Exactly one master_prng_ draw per chain, in order -- see file header.
            chain_prngs_.emplace_back(static_cast<std::uint32_t>(master_prng_->next()));
        }
        accept_count_.assign(static_cast<std::size_t>(number_of_chains_), 0);
        sample_count_.assign(static_cast<std::size_t>(number_of_chains_), 0);
        mean_log_likelihood_.clear();
        map_ = ParameterSet({}, -std::numeric_limits<double>::infinity());
    }

    // Samples the Markov Chains.
    virtual void sample() {
        validate_settings();

        if (!resume_simulation || simulations_ < 1) {
            chain_states_ = initialize_chains();
            initialize_custom_settings();
        }

        int total_iterations = iterations_ + output_iterations();
        int output_count = 0;
        output_.assign(static_cast<std::size_t>(number_of_chains_), std::vector<ParameterSet>());

        for (int i = 1; i <= total_iterations; ++i) {
            for (int j = 0; j < number_of_chains_; ++j) {
                chain_states_[static_cast<std::size_t>(j)] =
                    sample_chain(j, chain_states_[static_cast<std::size_t>(j)]);
            }

            // Record output.
            for (int j = 0; j < number_of_chains_; ++j) {
                if (is_population_sampler_)
                    population_matrix_.push_back(chain_states_[static_cast<std::size_t>(j)].clone(false));

                if (i <= iterations_) {
                    if (static_cast<int>(mean_log_likelihood_.size()) < i) mean_log_likelihood_.push_back(0.0);
                    mean_log_likelihood_[static_cast<std::size_t>(i - 1)] +=
                        chain_states_[static_cast<std::size_t>(j)].fitness / number_of_chains_;

                    markov_chains_[static_cast<std::size_t>(j)].push_back(
                        chain_states_[static_cast<std::size_t>(j)].clone(false));
                } else if (i > iterations_ && output_count < output_length) {
                    output_[static_cast<std::size_t>(j)].push_back(
                        chain_states_[static_cast<std::size_t>(j)].clone(false));
                    ++output_count;
                    // MAP tracking (v2.1.4 fix, formerly a documented scale bug -- see
                    // initialize_chains()'s MAP branch): map_.fitness is now negated back
                    // onto the same UNSCALED log-likelihood scale chain_states_[j].fitness
                    // uses everywhere else, so this comparison is meaningful and can
                    // actually re-trigger when a later-sampled chain state beats the
                    // initial MAP estimate (formerly map_.fitness carried
                    // DifferentialEvolution's internally-negated scale, making it a large
                    // POSITIVE number for any typical negative log-likelihood, so the
                    // comparison was nearly always false and MAP stayed frozen at the DE
                    // estimate for the rest of Sample() -- retired history logged in
                    // docs/upstream-csharp-issues.md).
                    if (chain_states_[static_cast<std::size_t>(j)].fitness > map_.fitness)
                        map_ = chain_states_[static_cast<std::size_t>(j)].clone();
                }
            }

            // Progress reporting / cancellation are omitted -- see file header.
        }

        simulations_ += 1;
    }

   protected:
    // --- Protected state (direct-access by design, matching this port's convention of
    // exposing C#'s `protected` fields directly to subclasses rather than via further
    // accessor indirection; see optimizer.hpp/differential_evolution.hpp for the same
    // pattern). ---

    int prng_seed_ = 12345;
    int initial_iterations_ = 10;
    int warmup_iterations_ = 1750;
    int iterations_ = 3500;
    int number_of_chains_ = 4;
    int thinning_interval_ = 20;

    int simulations_ = 0;

    // The master PRNG.
    std::optional<sampling::MersenneTwister> master_prng_;

    // The PRNG for each Markov Chain.
    std::vector<sampling::MersenneTwister> chain_prngs_;

    // The current states of each chain.
    std::vector<ParameterSet> chain_states_;

    LogLikelihood log_likelihood_function_;
    std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> prior_distributions_;

    bool is_population_sampler_ = false;

    // Whether the MAP optimization succeeded during InitializeChains().
    bool map_successful_ = false;

    bool map_initialization_failed_ = false;

    // The Multivariate Normal proposal distribution set from the MAP estimate (C# `_MVN`).
    std::optional<distributions::MultivariateNormal> mvn_;

    std::vector<ParameterSet> population_matrix_;
    std::vector<std::vector<ParameterSet>> markov_chains_;

    // UserDefined seed store (ADDITIVE, X2). NOT cleared by reset(); materialized into
    // population_matrix_ / markov_chains_ by initialize_chains() when initialize ==
    // UserDefined. Empty for every never-seeded sampler (Randomize / MAP paths untouched).
    std::vector<ParameterSet> seeded_population_;
    std::vector<std::vector<ParameterSet>> seeded_chains_;
    std::vector<int> accept_count_;
    std::vector<int> sample_count_;
    std::vector<double> mean_log_likelihood_;
    std::vector<std::vector<ParameterSet>> output_;
    ParameterSet map_;

    // Validate the sampler settings.
    virtual void validate_settings() {
        if (number_of_chains_ < 1) throw std::invalid_argument("There must be at least 1 chain.");
        if (iterations_ < 100) throw std::invalid_argument("The number of iterations cannot be less than 100.");
        if (warmup_iterations_ < 1)
            throw std::invalid_argument("The number of warm up iterations cannot be less than 1.");
        if (warmup_iterations_ > static_cast<int>(0.5 * iterations_))
            throw std::invalid_argument(
                "The number of warm up iterations cannot be greater than half the number of iterations.");
        if (thinning_interval_ < 1) throw std::invalid_argument("The thinning interval cannot be less than 1.");
        if (initial_iterations_ < number_of_chains_)
            throw std::invalid_argument("The initial population cannot be less than the number of chains.");
        if (output_length < 100) throw std::invalid_argument("The output length must be at least 100.");
        validate_custom_settings();
    }

    // Validate any custom MCMC sampler settings. No-op here; overridden by concrete
    // samplers (e.g. RWMH checks its proposal covariance matrix).
    virtual void validate_custom_settings() {}

    int output_iterations() const {
        return static_cast<int>(
            std::ceil(static_cast<double>(output_length) / static_cast<double>(number_of_chains_)));
    }

    // Initialize any custom MCMC sampler settings. No-op here; overridden by concrete
    // samplers.
    virtual void initialize_custom_settings() {}

    // Initialize the Markov Chains.
    virtual std::vector<ParameterSet> initialize_chains() {
        if (initialize == InitializationType::UserDefined) {
            // Materialize any stored UserDefined seed past reset()'s wipe (ADDITIVE, X2 hook).
            // For a never-seeded resume (the pre-existing use case: markov_chains_ already
            // holds prior-Sample() states, seeds empty) both stores are empty and this is a
            // no-op -- behavior is byte-identical to before the hook.
            if (!seeded_population_.empty()) population_matrix_ = seeded_population_;
            for (std::size_t i = 0; i < seeded_chains_.size() && i < markov_chains_.size(); ++i) {
                if (!seeded_chains_[i].empty()) markov_chains_[i] = seeded_chains_[i];
            }

            // Return the last state of each existing chain.
            std::vector<ParameterSet> chain_states(static_cast<std::size_t>(number_of_chains_));
            for (int i = 0; i < number_of_chains_; ++i)
                chain_states[static_cast<std::size_t>(i)] = markov_chains_[static_cast<std::size_t>(i)].back();
            return chain_states;
        }

        sampling::MersenneTwister prng(static_cast<std::uint32_t>(prng_seed_));
        auto rnds = sampling::LatinHypercube::random(initial_iterations_, number_of_parameters(), prng.next());
        std::vector<double> parameters(static_cast<std::size_t>(number_of_parameters()));
        std::vector<ParameterSet> temp_population;
        std::vector<ParameterSet> initials(static_cast<std::size_t>(number_of_chains_));
        double log_lh = 0.0;

        if (initialize == InitializationType::MAP) {
            // Use differential evolution to find a global optimum.
            std::vector<double> lower_bounds(static_cast<std::size_t>(number_of_parameters()));
            std::vector<double> upper_bounds(static_cast<std::size_t>(number_of_parameters()));
            for (int j = 0; j < number_of_parameters(); ++j) {
                lower_bounds[static_cast<std::size_t>(j)] = prior_distributions_[static_cast<std::size_t>(j)]->minimum();
                upper_bounds[static_cast<std::size_t>(j)] = prior_distributions_[static_cast<std::size_t>(j)]->maximum();
            }
            // NOTE: C# also computes `var inititals = lowerBounds.Add(upperBounds).Divide(2d);`
            // here (a midpoint vector) but the result is never referenced again anywhere in
            // the method -- dead code (confirmed by inspection of the full method body).
            // Omitted; logged in docs/upstream-csharp-issues.md.
            opt::DifferentialEvolution de(
                [this](const std::vector<double>& x) { return log_likelihood_function_(x); },
                number_of_parameters(), lower_bounds, upper_bounds);
            de.report_failure = false;
            de.maximize();
            if (de.status() == opt::OptimizationStatus::Success) {
                try {
                    map_successful_ = true;
                    // Get MAP. Fitness is negated back onto the log-likelihood scale
                    // (v2.1.4 fix): DifferentialEvolution::maximize() sets function_scale_
                    // = -1, so best_parameter_set().fitness is that internally *scaled*
                    // (-logLH) value, not the plain log-likelihood chain_states_[j].fitness
                    // uses everywhere else in this sampler. Negating it here puts map_
                    // back on that common scale -- see sample()'s output-phase MAP-tracking
                    // comparison, which depends on this.
                    map_ = ParameterSet(de.best_parameter_set().values, -de.best_parameter_set().fitness);
                    // Get Fisher Information Matrix.
                    if (!de.hessian().has_value())
                        throw std::runtime_error("Hessian matrix is not available.");
                    auto fisher = de.hessian().value() * -1.0;
                    // Invert it to get the covariance matrix, and scale to give wider
                    // coverage.
                    auto covar = fisher.inverse() * 2.0;

                    // Set up proposal distribution.
                    mvn_.emplace(map_.values, covar.to_array());
                    // Then randomly sample from the proposal.
                    for (int i = 0; i < initial_iterations_; ++i) {
                        parameters = mvn_->inverse_cdf(rnds[static_cast<std::size_t>(i)]);
                        log_lh = log_likelihood_function_(parameters);
                        if (is_population_sampler_) population_matrix_.emplace_back(parameters, log_lh);
                        temp_population.emplace_back(parameters, log_lh);
                    }

                    // Set the initial vectors randomly from the MVN proposal.
                    for (int i = 0; i < number_of_chains_; ++i)
                        initials[static_cast<std::size_t>(i)] = temp_population[static_cast<std::size_t>(i)].clone();

                    return initials;
                } catch (...) {
                    // If this fails, go to naive initialization below.
                    initialize = InitializationType::Randomize;
                    map_initialization_failed_ = true;
                }
            }
        }

        // *** If not using MAP or if MAP fails, then use random initialization *** //

        // First add the mean of the priors.
        for (int j = 0; j < number_of_parameters(); ++j)
            parameters[static_cast<std::size_t>(j)] = prior_distributions_[static_cast<std::size_t>(j)]->mean();
        log_lh = log_likelihood_function_(parameters);
        if (is_population_sampler_) population_matrix_.emplace_back(parameters, log_lh);
        temp_population.emplace_back(parameters, log_lh);

        // If the initial population and the number of chains is 1, just take the mean of
        // the priors.
        if (initial_iterations_ == 1 && number_of_chains_ == 1) {
            initials[0] = temp_population.front().clone();
            return initials;
        }

        // Then randomly sample from the priors.
        for (int i = 1; i < initial_iterations_; ++i) {
            for (int j = 0; j < number_of_parameters(); ++j) {
                parameters[static_cast<std::size_t>(j)] = prior_distributions_[static_cast<std::size_t>(j)]->inverse_cdf(
                    rnds[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            }
            log_lh = log_likelihood_function_(parameters);
            if (is_population_sampler_) population_matrix_.emplace_back(parameters, log_lh);
            temp_population.emplace_back(parameters, log_lh);
        }

        // Preserve draw order when likelihoods tie. Upstream now uses stable
        // OrderByDescending because tied negative-infinity prior draws otherwise make
        // chain initialization implementation dependent.
        std::stable_sort(temp_population.begin(), temp_population.end(),
                          [](const ParameterSet& a, const ParameterSet& b) { return a.fitness > b.fitness; });

        // Set the initial vectors to the best-performing parameter sets.
        for (int i = 0; i < number_of_chains_; ++i)
            initials[static_cast<std::size_t>(i)] = temp_population[static_cast<std::size_t>(i)].clone();

        return initials;
    }

    // Samples one Markov Chain: iterates ChainIteration `thinning_interval_` times, then
    // returns the final state.
    virtual ParameterSet sample_chain(int index, ParameterSet state) {
        for (int j = 1; j <= thinning_interval_; ++j) {
            state = chain_iteration(index, state).clone(false);
        }
        return state;
    }

    // Returns a proposed MCMC parameter set and its fitness. Concrete samplers (RWMH, and
    // later ARWMH/DEMCz/.../Gibbs) implement the proposal/accept-reject mechanics here.
    virtual ParameterSet chain_iteration(int index, ParameterSet state) = 0;
};

}  // namespace corehydro::numerics::sampling::mcmc
