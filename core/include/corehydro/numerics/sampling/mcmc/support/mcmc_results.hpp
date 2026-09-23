// ported from: Numerics/Sampling/MCMC/Support/MCMCResults.cs @ 7e8e8d1
//
// Post-processes a finished MCMCSampler run (or a raw (MAP, parameter-set list) pair) into
// posterior summaries: cloned MarkovChains + a flattened combined Output, AcceptanceRates,
// MeanLogLikelihood, MAP, per-parameter ParameterResults (mean/sd/median/CI/histogram/
// autocorrelation, via ParameterResults + MCMCDiagnostics), and PosteriorMean.
//
// SKIPPED: ToByteArray/FromByteArray + the JsonConverters-based (de)serialization surface --
// GUI/persistence, out of scope for the numerical core (consistent with every other ported
// class's Serialization region).
#pragma once
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_diagnostics.hpp"
#include "corehydro/numerics/sampling/mcmc/support/parameter_results.hpp"

namespace corehydro::numerics::sampling::mcmc {

using opt::ParameterSet;

class MCMCResults {
   public:
    // Constructs an empty MCMC results.
    MCMCResults() = default;

    // Constructs and post-processes MCMC results. `alpha`: the confidence level; default
    // 0.1 -> 90% confidence intervals.
    explicit MCMCResults(const MCMCSampler& sampler, double alpha = 0.1) {
        // Clone the Markov Chains and Output.
        markov_chains.assign(static_cast<std::size_t>(sampler.number_of_chains()), std::vector<ParameterSet>());
        for (int i = 0; i < sampler.number_of_chains(); ++i) {
            markov_chains[static_cast<std::size_t>(i)] = sampler.markov_chains()[static_cast<std::size_t>(i)];
            const auto& out_i = sampler.output()[static_cast<std::size_t>(i)];
            output.insert(output.end(), out_i.begin(), out_i.end());
        }
        acceptance_rates = sampler.reported_acceptance_rates();
        mean_log_likelihood = sampler.mean_log_likelihood();
        map = sampler.map().clone();
        process_parameter_results(sampler, alpha);
    }

    // Constructs and post-processes MCMC results from a raw MAP + parameter-set list (no
    // sampler / MarkovChains / AcceptanceRates / MeanLogLikelihood / Rhat / ESS available).
    MCMCResults(ParameterSet map_in, std::vector<ParameterSet> parameter_sets, double alpha = 0.1)
        : output(std::move(parameter_sets)), map(std::move(map_in)) {
        process_parameter_results(alpha);
    }

    // The list of sampled Markov Chains.
    std::vector<std::vector<ParameterSet>> markov_chains;

    // Output posterior parameter sets, flattened across every chain.
    std::vector<ParameterSet> output;

    // The average log-likelihood across each chain for each iteration.
    std::vector<double> mean_log_likelihood;

    // The acceptance rate for each chain.
    std::vector<double> acceptance_rates;

    // Parameter results using the output posterior parameter sets.
    std::vector<ParameterResults> parameter_results;

    // The output parameter set that produced the maximum likelihood (MAP).
    ParameterSet map;

    // The mean of the posterior distribution of each parameter.
    ParameterSet posterior_mean;

    // Recomputes parameter summary statistics at a new credible-interval level (alpha)
    // without rerunning the chain. Preserves Rhat, ESS, autocorrelation, MarkovChains,
    // AcceptanceRates, MeanLogLikelihood, MAP, and Output.
    void recompute_parameter_results(double alpha) {
        if (parameter_results.empty() || output.empty()) return;

        std::size_t n = parameter_results.size();
        std::vector<double> rhats(n), esss(n);
        std::vector<std::vector<std::array<double, 2>>> acfs(n);
        for (std::size_t i = 0; i < n; ++i) {
            rhats[i] = parameter_results[i].summary_statistics.rhat;
            esss[i] = parameter_results[i].summary_statistics.ess;
            acfs[i] = parameter_results[i].autocorrelation;
        }

        // Recompute (replaces parameter_results and posterior_mean -- output untouched).
        process_parameter_results(alpha);

        // Restore preserved diagnostics.
        for (std::size_t i = 0; i < parameter_results.size(); ++i) {
            parameter_results[i].summary_statistics.rhat = rhats[i];
            parameter_results[i].summary_statistics.ess = esss[i];
            parameter_results[i].autocorrelation = acfs[i];
        }
    }

   private:
    // Process parameter results using the (already-populated) sampler + this->output.
    void process_parameter_results(const MCMCSampler& sampler, double alpha = 0.1) {
        auto gr = gelman_rubin(sampler.markov_chains(), sampler.warmup_iterations());
        auto ess_result = effective_sample_size(sampler.output());

        std::vector<double> post_mean(static_cast<std::size_t>(sampler.number_of_parameters()));
        parameter_results.assign(static_cast<std::size_t>(sampler.number_of_parameters()), ParameterResults());
        for (int i = 0; i < sampler.number_of_parameters(); ++i) {
            std::vector<double> x(output.size());
            for (std::size_t k = 0; k < output.size(); ++k) x[k] = output[k].values[static_cast<std::size_t>(i)];
            parameter_results[static_cast<std::size_t>(i)] = ParameterResults(x, alpha);
            parameter_results[static_cast<std::size_t>(i)].summary_statistics.rhat = gr[static_cast<std::size_t>(i)];
            parameter_results[static_cast<std::size_t>(i)].summary_statistics.ess =
                ess_result.ess[static_cast<std::size_t>(i)];
            parameter_results[static_cast<std::size_t>(i)].autocorrelation =
                ess_result.average_acf[static_cast<std::size_t>(i)];
            post_mean[static_cast<std::size_t>(i)] = parameter_results[static_cast<std::size_t>(i)].summary_statistics.mean;
        }
        // Set the posterior mean parameter set.
        double post_mean_log_lh = sampler.log_likelihood_function()(post_mean);
        posterior_mean = ParameterSet(post_mean, post_mean_log_lh);
    }

    // Process parameter results using only this->output (the raw (MAP, list) ctor path).
    void process_parameter_results(double alpha = 0.1) {
        if (output.empty()) throw std::invalid_argument("Sequence contains no elements.");
        int p = static_cast<int>(output.front().values.size());
        std::vector<double> post_mean(static_cast<std::size_t>(p));
        parameter_results.assign(static_cast<std::size_t>(p), ParameterResults());
        for (int i = 0; i < p; ++i) {
            std::vector<double> x(output.size());
            for (std::size_t k = 0; k < output.size(); ++k) x[k] = output[k].values[static_cast<std::size_t>(i)];
            parameter_results[static_cast<std::size_t>(i)] = ParameterResults(x, alpha);
            post_mean[static_cast<std::size_t>(i)] = parameter_results[static_cast<std::size_t>(i)].summary_statistics.mean;
        }
        // Set the posterior mean parameter set (no sampler to evaluate the log-likelihood
        // function against; matches C#'s `double.NaN` fitness on this path).
        posterior_mean = ParameterSet(post_mean, std::numeric_limits<double>::quiet_NaN());
    }
};

}  // namespace corehydro::numerics::sampling::mcmc
