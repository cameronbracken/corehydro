// ported from: Numerics/Sampling/MCMC/ARWMH.cs @ 7e8e8d1
//
// Adaptive Random Walk Metropolis-Hastings (ARWMH): like RWMH, but the proposal covariance
// adapts online from the chain's own history via a per-chain RunningCovarianceMatrix (P3.2),
// scaled by `Scale = 2.38^2 / D` (the standard Roberts-Rosenthal optimal-scaling constant),
// with a small `Beta`-probability (default 0.05) of falling back to a fixed, non-adaptive
// identity-scaled proposal (`sigmaIdentity = I(D) * (0.1^2 / D)`) -- both to keep the chain
// exploring during the (first `100*D` samples, always-identity) warmup phase and to guard
// against the adaptive covariance collapsing/degenerating later on.
//
// ChainIteration draw order (verbatim, and the brief's flagged hazard -- see the self-review
// note in the task report): (1) ONE uniform for the Beta test
// (`_chainPRNGs[index].NextDouble() <= Beta`) -- drawn UNCONDITIONALLY every iteration since
// it is the left operand of a `||`, even when the right operand (`SampleCount[index] <= 100 *
// NumberOfParameters`) alone would already select the identity branch; (2) `D` uniforms via
// `NextDoubles(NumberOfParameters)` for the MVN inverse-CDF proposal; (3) the feasibility
// check (no draw); (4) the log-Metropolis ratio; (5) one more uniform for the accept/reject
// draw. Drawing the Beta-test uniform anywhere but first desyncs the stream.
//
// The adaptive covariance is updated from the realized retained chain state on every
// iteration. Rejected and infeasible proposals contribute the repeated current state.
#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/running_covariance_matrix.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::sampling::mcmc {

namespace linalg = corehydro::numerics::math::linalg;
namespace ext = corehydro::numerics::utilities;

class ARWMH : public MCMCSampler {
   public:
    // Constructs a new ARWMH sampler.
    ARWMH(std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors,
          LogLikelihood log_likelihood_function)
        : MCMCSampler(std::move(priors), std::move(log_likelihood_function)),
          // The initial scale & identity covariance matrix. Computed in the member
          // initializer list (Matrix has no default ctor); `number_of_parameters()` is safe
          // to call here since the MCMCSampler base subobject (which owns
          // `prior_distributions_`) is already fully constructed by this point.
          sigma_identity_(linalg::Matrix::identity(number_of_parameters()) *
                           (0.1 * 0.1 / number_of_parameters())) {
        set_initial_iterations(100 * number_of_parameters());

        // The optimal scale & adaptive covariance matrix.
        scale = 2.38 * 2.38 / number_of_parameters();
        beta = 0.05;
    }

    // The scaling parameter used to scale the adaptive covariance matrix. Plain public field
    // (matches the C# auto-property `Scale { get; set; }`; no reset() side effect).
    double scale = 0.0;

    // Determines how often to sample from the small identity covariance matrix; e.g. 0.05
    // will result in sampling 5% of the time. Plain public field (matches the C#
    // auto-property `Beta { get; set; }`; no reset() side effect).
    double beta = 0.0;

    // The covariance matrix Sigma for the proposal distribution, one per chain: the small
    // fixed identity matrix while `sigma_[i].n() < 2`, else the running empirical covariance
    // scaled by `1 / (N - 1)`.
    std::vector<linalg::Matrix> proposal_sigma() const {
        std::vector<linalg::Matrix> sigmas;
        sigmas.reserve(static_cast<std::size_t>(number_of_chains()));
        for (int i = 0; i < number_of_chains(); ++i) {
            const auto& s = sigma_[static_cast<std::size_t>(i)];
            if (s.n() < 2)
                sigmas.push_back(sigma_identity_);
            else
                sigmas.push_back(s.covariance() * (1.0 / static_cast<double>(s.n() - 1)));
        }
        return sigmas;
    }

   protected:
    void validate_custom_settings() override {
        if (scale <= 0.0) throw std::invalid_argument("The scale parameter must greater than 0.");
        if (beta < 0.0 || beta > 1.0) throw std::invalid_argument("Beta must be between 0 and 1.");
    }

    void initialize_custom_settings() override {
        // Set up multivariate Normal distributions and the adaptive covariance matrix for
        // each chain.
        mvn_per_chain_.clear();
        sigma_.clear();
        mvn_per_chain_.reserve(static_cast<std::size_t>(number_of_chains()));
        sigma_.reserve(static_cast<std::size_t>(number_of_chains()));
        for (int i = 0; i < number_of_chains(); ++i) {
            mvn_per_chain_.emplace_back(number_of_parameters());
            sigma_.emplace_back(number_of_parameters());

            if (initialize == InitializationType::MAP && map_successful_ && mvn_.has_value()) {
                // Hot start the covariance matrix.
                for (int j = 0; j < number_of_parameters(); ++j) {
                    for (int k = 0; k < number_of_parameters(); ++k) {
                        sigma_[static_cast<std::size_t>(i)].covariance_mutable()(j, k) = mvn_->covariance(j, k);
                    }
                }
            }
        }
    }

    ParameterSet chain_iteration(int index, ParameterSet state) override {
        auto& mvn_i = mvn_per_chain_[static_cast<std::size_t>(index)];
        auto& sigma_i = sigma_[static_cast<std::size_t>(index)];

        // Update the sample count.
        sample_count_[static_cast<std::size_t>(index)] += 1;

        // The Beta-test uniform is drawn FIRST, unconditionally -- see file header.
        if (chain_prngs_[static_cast<std::size_t>(index)].next_double() <= beta ||
            sample_count_[static_cast<std::size_t>(index)] <= 100 * number_of_parameters()) {
            // Use the identity matrix the first 100*D samples.
            mvn_i.set_parameters(state.values, sigma_identity_.to_array());
        } else {
            // Use the adaptive covariance matrix.
            auto proposal_cov = sigma_i.covariance() * (scale / static_cast<double>(sigma_i.n() - 1));
            mvn_i.set_parameters(state.values, proposal_cov.to_array());
        }

        // Get proposal vector.
        auto xp = mvn_i.inverse_cdf(
            ext::next_doubles(chain_prngs_[static_cast<std::size_t>(index)], number_of_parameters()));

        bool feasible = true;
        for (int i = 0; i < number_of_parameters(); ++i) {
            const auto& prior = prior_distributions_[static_cast<std::size_t>(i)];
            if (xp[static_cast<std::size_t>(i)] < prior->minimum() || xp[static_cast<std::size_t>(i)] > prior->maximum()) {
                feasible = false;
                break;
            }
        }

        ParameterSet retained = state;
        if (feasible) {
            const double proposed_log_likelihood = log_likelihood_function_(xp);
            const double log_ratio = proposed_log_likelihood - state.fitness;
            const double log_uniform =
                std::log(chain_prngs_[static_cast<std::size_t>(index)].next_double());
            if (log_uniform <= log_ratio) {
                accept_count_[static_cast<std::size_t>(index)] += 1;
                retained = ParameterSet(xp, proposed_log_likelihood);
            }
        }
        sigma_i.push(retained.values);
        return retained;
    }

   private:
    linalg::Matrix sigma_identity_;
    std::vector<data::RunningCovarianceMatrix> sigma_;
    std::vector<distributions::MultivariateNormal> mvn_per_chain_;
};

}  // namespace corehydro::numerics::sampling::mcmc
