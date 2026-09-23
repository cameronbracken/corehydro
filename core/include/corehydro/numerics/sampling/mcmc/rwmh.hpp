// ported from: Numerics/Sampling/MCMC/RWMH.cs @ 7e8e8d1
//
// Random Walk Metropolis-Hastings (RWMH): the simplest MCMCSampler concretization, and this
// port's end-to-end exemplar for the whole sampler family. Every chain proposes a new point
// from a Multivariate Normal centered at the current state with covariance `proposal_sigma`
// (fixed, or -- when `initialize == MAP` and the MAP optimization succeeded -- overridden by
// the Fisher-information-derived covariance in `initialize_custom_settings()`), rejects
// proposals outside the prior support (feasibility check, not a Metropolis reject), and
// otherwise accepts/rejects via the standard log-Metropolis ratio.
//
// ChainIteration draw order (verbatim): (1) `NumberOfParameters` uniforms via
// `_chainPRNGs[index].NextDoubles(NumberOfParameters)`; (2) the MVN inverse-CDF transform of
// those uniforms into a proposal vector `xp`; (3) a bounds/feasibility check against each
// prior's [Minimum, Maximum] (an in-place reject -- returns the unchanged `state`, consuming
// no further PRNG draws); (4) the log-Metropolis ratio; (5) one more
// `_chainPRNGs[index].NextDouble()` for the accept/reject draw. Getting this order wrong
// (e.g. drawing the accept/reject uniform before the feasibility check) desyncs the stream.
#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::sampling::mcmc {

namespace linalg = corehydro::numerics::math::linalg;
namespace ext = corehydro::numerics::utilities;

class RWMH : public MCMCSampler {
   public:
    // Constructs a new RWMH sampler. `proposal_sigma`: the covariance matrix Sigma for the
    // proposal distribution.
    RWMH(std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors,
         LogLikelihood log_likelihood_function, linalg::Matrix proposal_sigma)
        : MCMCSampler(std::move(priors), std::move(log_likelihood_function)),
          proposal_sigma_(std::move(proposal_sigma)) {
        set_initial_iterations(100 * number_of_parameters());
    }

    // The covariance matrix Sigma for the proposal distribution.
    const linalg::Matrix& proposal_sigma() const { return proposal_sigma_; }

   protected:
    void validate_custom_settings() override {
        // C# also checks `ProposalSigma == null` -- unreachable here: `Matrix` is a value
        // type in this port, and `proposal_sigma_` is a required, always-constructed ctor
        // argument (the same "no C++ equivalent for a null value type" omission as
        // multivariate_normal.hpp's mean/covariance null-checks).
        if (proposal_sigma_.number_of_rows() != proposal_sigma_.number_of_columns())
            throw std::invalid_argument("The proposal covariance matrix must be square.");
        if (proposal_sigma_.number_of_rows() != number_of_parameters())
            throw std::invalid_argument(
                "The proposal covariance matrix must have the same number of rows and "
                "columns as the number of parameters.");
    }

    void initialize_custom_settings() override {
        // Set up the proposal matrix: if MAP initialization succeeded, adopt its
        // Fisher-information-derived covariance instead of the ctor-supplied
        // `proposal_sigma_`.
        if (initialize == InitializationType::MAP && map_successful_ && mvn_.has_value()) {
            proposal_sigma_ = linalg::Matrix(mvn_->covariance());
        }
        // Factorize the fixed covariance once. A chain iteration only translates the mean.
        mvn_per_chain_.clear();
        mvn_per_chain_.reserve(static_cast<std::size_t>(number_of_chains()));
        for (int i = 0; i < number_of_chains(); ++i) {
            mvn_per_chain_.emplace_back(number_of_parameters());
            mvn_per_chain_.back().set_parameters(
                std::vector<double>(static_cast<std::size_t>(number_of_parameters()), 0.0),
                proposal_sigma_.to_array());
        }
    }

    ParameterSet chain_iteration(int index, ParameterSet state) override {
        // Update the sample count.
        sample_count_[static_cast<std::size_t>(index)] += 1;

        // Get proposal vector.
        mvn_per_chain_[static_cast<std::size_t>(index)].set_mean(state.values);
        auto xp = mvn_per_chain_[static_cast<std::size_t>(index)].inverse_cdf(
            ext::next_doubles(chain_prngs_[static_cast<std::size_t>(index)], number_of_parameters()));

        for (int i = 0; i < number_of_parameters(); ++i) {
            // Check if the parameter is feasible (within the prior's constraints).
            const auto& prior = prior_distributions_[static_cast<std::size_t>(i)];
            if (xp[static_cast<std::size_t>(i)] < prior->minimum() || xp[static_cast<std::size_t>(i)] > prior->maximum()) {
                // The proposed parameter vector was infeasible, so leave the state
                // unchanged.
                return state;
            }
        }

        // Evaluate fitness.
        double log_lh_p = log_likelihood_function_(xp);
        double log_lh_i = state.fitness;

        // Calculate the Metropolis ratio.
        double log_ratio = log_lh_p - log_lh_i;

        // Accept the proposal with probability min(1, r); otherwise leave the state
        // unchanged.
        double log_u = std::log(chain_prngs_[static_cast<std::size_t>(index)].next_double());
        if (log_u <= log_ratio) {
            // The proposal is accepted.
            accept_count_[static_cast<std::size_t>(index)] += 1;
            return ParameterSet(xp, log_lh_p);
        }
        return state;
    }

   private:
    linalg::Matrix proposal_sigma_;
    std::vector<distributions::MultivariateNormal> mvn_per_chain_;
};

}  // namespace corehydro::numerics::sampling::mcmc
