// ported from: Numerics/Sampling/MCMC/HMC.cs @ 7e8e8d1
//
// Hamiltonian Monte Carlo (HMC): a gradient-based sampler. Each ChainIteration jitters the
// leapfrog step size/step count, draws a momentum vector phi ~ N(0, Mass), simulates
// Hamiltonian dynamics via leapfrog integration (reflecting phi and clamping xp at any prior
// boundary it crosses), and accepts/rejects via the standard log-Metropolis ratio on the
// TOTAL energy (log-likelihood + kinetic energy) -- NOT a feasibility-only reject like
// RWMH/ARWMH/DEMCz(s): an infeasible xp can occur only transiently mid-leapfrog (immediately
// reflected/clamped back in bounds), never as a final accept/reject test here.
//
// `Gradient = std::function<linalg::Vector(const std::vector<double>&)>` mirrors C#'s
// `public delegate Vector Gradient(IList<double> parameters);` -- NUTS.cs (a later slice)
// reuses this exact alias. The default gradient function -- used whenever the ctor's
// `gradient_function` argument is left empty -- is a bound-aware finite-difference gradient
// of `SafeLogLikelihood` (differentiation::gradient(), the P3.3 NumericalDerivative port)
// evaluated against the cached prior [Minimum, Maximum] bounds, so finite-difference probes
// never leave the feasible region even when `xp` is currently pinned at a boundary.
//
// SafeLogLikelihood: C# catches `ArgumentOutOfRangeException` from the user's log-likelihood
// closure and maps it to `double.NegativeInfinity` -- ported directly (`std::out_of_range` is
// this port's closest analogue; no distribution ported so far actually throws it from a
// likelihood path, since our guards return -inf/NaN instead, but the C# doc comment
// explicitly anticipates a future closure that could).
//
// ChainIteration's own `catch (ArithmeticException)` wraps the ENTIRE proposal (jitter draws
// through the accept/reject test) and specifically catches the "non-finite gradient" failure
// mode: NumericalDerivative.Gradient throws `ArithmeticException` when `f(theta)` itself is
// not finite (SafeLogLikelihood having mapped an out-of-range draw to -infinity). This port's
// `differentiation::gradient()` (see numerical_derivative.hpp) throws `std::domain_error` for
// that exact condition -- the direct C++ translation of the same C# guard -- so the catch
// clause here is `catch (const std::domain_error&)`, not a literal `ArithmeticException`
// spelling (C++ has no such type). A CUSTOM `gradient_function` supplied by the caller could
// throw any exception type in either language; only the specific type each language's default
// gradient path actually throws is caught, matching the C# source's scope exactly (nothing
// broader).
//
// ChainIteration draw order (verbatim): (1) ONE uniform, `_stepSizeU.InverseCDF(...)`, for the
// jittered step size; (2) ONE uniform, `_stepsU.InverseCDF(...)` then `Math.Ceiling`'d, for the
// jittered step count; (3) `D` uniforms, one per parameter, each immediately consumed by
// `Normal::standard_z` for the momentum draw `phi[i]`; (4) the leapfrog integration itself
// draws NOTHING (every `GradientFunction` call is deterministic given `xp`); (5) ONE final
// uniform for the accept/reject draw. Getting this order wrong (e.g. drawing the step-size/
// step-count jitter uniforms in the wrong order, or interleaving a draw inside the leapfrog
// loop) desyncs the stream.
#pragma once
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/distributions/uniform_discrete.hpp"
#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::sampling::mcmc {

namespace linalg = corehydro::numerics::math::linalg;
namespace diff = corehydro::numerics::math::differentiation;

class HMC : public MCMCSampler {
   public:
    // The function for evaluating the gradient of the log-likelihood function.
    using Gradient = std::function<linalg::Vector(const std::vector<double>&)>;

    // Constructs a new HMC sampler. `mass`: optional momentum-distribution mass vector
    // (default = identity, i.e. all 1s -- mirrors C#'s `mass == null` default; `std::nullopt`
    // is this port's null-value-type stand-in, same convention as `mcmc_sampler.hpp`'s
    // `mvn_`). `step_size`: leapfrog step size (default 0.1). `steps`: number of leapfrog
    // steps (default 50). `gradient_function`: optional custom gradient function; default = a
    // bound-aware finite-difference gradient of `SafeLogLikelihood`.
    HMC(std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors,
        LogLikelihood log_likelihood_function, std::optional<linalg::Vector> mass = std::nullopt,
        double step_size = 0.1, int steps = 50, Gradient gradient_function = nullptr)
        : MCMCSampler(std::move(priors), std::move(log_likelihood_function)),
          mass_(mass.has_value() ? std::move(mass).value() : linalg::Vector(number_of_parameters(), 1.0)),
          inverse_mass_(number_of_parameters()) {
        set_initial_iterations(100 * number_of_parameters());

        // Set the inverse mass vector.
        for (int i = 0; i < number_of_parameters(); ++i) inverse_mass_[i] = 1.0 / mass_[i];

        // Set leapfrog inputs.
        set_step_size(step_size);
        set_steps(steps);

        // Cache prior distribution bounds for the gradient function.
        lower_bounds_.resize(static_cast<std::size_t>(number_of_parameters()));
        upper_bounds_.resize(static_cast<std::size_t>(number_of_parameters()));
        for (int i = 0; i < number_of_parameters(); ++i) {
            lower_bounds_[static_cast<std::size_t>(i)] = prior_distributions_[static_cast<std::size_t>(i)]->minimum();
            upper_bounds_[static_cast<std::size_t>(i)] = prior_distributions_[static_cast<std::size_t>(i)]->maximum();
        }

        // Set the gradient function with prior bounds so finite-difference probes stay in
        // valid region.
        if (gradient_function == nullptr) {
            gradient_function_ = [this](const std::vector<double>& x) {
                return linalg::Vector(diff::gradient([this](const std::vector<double>& y) { return safe_log_likelihood(y); },
                                                       x, lower_bounds_, upper_bounds_));
            };
        } else {
            gradient_function_ = std::move(gradient_function);
        }
    }

    // The mass vector for the momentum distribution.
    const linalg::Vector& mass() const { return mass_; }

    // The leapfrog step size. Default = 0.1.
    double step_size() const { return step_size_; }
    void set_step_size(double value) {
        step_size_ = value;
        step_size_u_ = distributions::Uniform(0.0, 2.0 * step_size_);
    }

    // The number of leapfrog steps. Default = 50.
    int steps() const { return steps_; }
    void set_steps(int value) {
        steps_ = value;
        steps_u_ = distributions::UniformDiscrete(1, 2.0 * steps_);
    }

    // The function for evaluating the gradient of the log-likelihood.
    const Gradient& gradient_function() const { return gradient_function_; }

    // Computes the quadratic form phi^T M^-1 phi, used for the kinetic energy. `phi` and
    // `inverse_mass` must be the same length.
    static double quadratic_form(const linalg::Vector& phi, const linalg::Vector& inverse_mass) {
        if (phi.length() != inverse_mass.length())
            throw std::invalid_argument("Vectors must be the same length to compute the quadratic form.");
        double sum = 0.0;
        for (int i = 0; i < phi.length(); ++i) sum += inverse_mass[i] * phi[i] * phi[i];
        return sum;
    }

   protected:
    void validate_custom_settings() override {
        if (mass_.length() != number_of_parameters())
            throw std::invalid_argument("The mass vector must be the same length as the number of parameters.");
        if (step_size_ <= 0.0) throw std::invalid_argument("The leapfrog step size must be positive.");
        if (steps_ < 1) throw std::invalid_argument("The number of leapfrog steps must be at least one.");
    }

    void initialize_custom_settings() override {
        start_gradient_positions_.assign(static_cast<std::size_t>(number_of_chains()), {});
        start_gradient_values_.assign(static_cast<std::size_t>(number_of_chains()), {});
        start_gradient_occupied_.assign(static_cast<std::size_t>(number_of_chains()), false);
    }

    ParameterSet chain_iteration(int index, ParameterSet state) override {
        // Update the sample count.
        sample_count_[static_cast<std::size_t>(index)] += 1;

        try {
            // Jigger the step size and number of steps.
            double leapfrog_step_size =
                step_size_u_.inverse_cdf(chain_prngs_[static_cast<std::size_t>(index)].next_double());
            int leapfrog_steps = static_cast<int>(
                std::ceil(steps_u_.inverse_cdf(chain_prngs_[static_cast<std::size_t>(index)].next_double())));

            // Step 1. Sample phi from a N~(0, M).
            linalg::Vector phi(number_of_parameters());
            for (int i = 0; i < number_of_parameters(); ++i)
                phi[i] = std::sqrt(mass_[i]) *
                         distributions::Normal::standard_z(chain_prngs_[static_cast<std::size_t>(index)].next_double());

            // Get kinetic energy of the current state.
            double log_ki = -0.5 * quadratic_form(phi, inverse_mass_);

            // Step 2. Perform leapfrog steps to get proposal vector.
            linalg::Vector xp(std::vector<double>(state.values));
            phi = phi + start_gradient(index, xp.to_array()) * leapfrog_step_size * 0.5;
            std::vector<double> proposal_position;
            linalg::Vector proposal_gradient(number_of_parameters());
            for (int i = 0; i < leapfrog_steps; ++i) {
                xp = xp + inverse_mass_ * phi * leapfrog_step_size;

                // Ensure the parameters are feasible (within the constraints).
                for (int j = 0; j < number_of_parameters(); ++j) {
                    const auto& prior = prior_distributions_[static_cast<std::size_t>(j)];
                    if (xp[j] < prior->minimum()) {
                        xp[j] = prior->minimum() + corehydro::numerics::kDoubleMachineEpsilon;
                        phi[j] = -phi[j];
                    }
                    if (xp[j] > prior->maximum()) {
                        xp[j] = prior->maximum() - corehydro::numerics::kDoubleMachineEpsilon;
                        phi[j] = -phi[j];
                    }
                }

                if (i == leapfrog_steps - 1) {
                    proposal_position = xp.to_array();
                    proposal_gradient = gradient_function_(xp.to_array());
                    phi = phi + proposal_gradient * leapfrog_step_size * 0.5;
                } else {
                    phi = phi + gradient_function_(xp.to_array()) * leapfrog_step_size;
                }
            }
            phi = phi * -1.0;

            // Get kinetic energy of the proposal state.
            double log_kp = -0.5 * quadratic_form(phi, inverse_mass_);

            // Evaluate fitness.
            double log_lh_p = safe_log_likelihood(xp.to_array());
            double log_lh_i = state.fitness;

            // Calculate the Metropolis ratio.
            double log_ratio = log_lh_p - log_lh_i + log_kp - log_ki;

            // Accept the proposal with probability min(1, r); otherwise leave xi unchanged.
            double log_u = std::log(chain_prngs_[static_cast<std::size_t>(index)].next_double());
            if (log_u <= log_ratio) {
                // The proposal is accepted.
                accept_count_[static_cast<std::size_t>(index)] += 1;
                store_start_gradient(index, std::move(proposal_position), proposal_gradient);
                return ParameterSet(xp.to_array(), log_lh_p);
            }
            return state;
        } catch (const std::domain_error&) {
            // Non-finite gradient encountered during leapfrog integration. This occurs when
            // parameters drift into regions where the log-likelihood returns -Infinity.
            // Reject the proposal and return the current state, consistent with Metropolis
            // rejection behavior. See file header for the ArithmeticException -> domain_error
            // mapping.
            return state;
        }
    }

   private:
    linalg::Vector start_gradient(int index, const std::vector<double>& position) {
        const std::size_t chain = static_cast<std::size_t>(index);
        if (start_gradient_occupied_[chain] &&
            matches_bitwise(start_gradient_positions_[chain], position))
            return linalg::Vector(start_gradient_values_[chain]);
        const std::vector<double> recorded = position;
        linalg::Vector gradient = gradient_function_(position);
        store_start_gradient(index, recorded, gradient);
        return gradient;
    }

    void store_start_gradient(int index, std::vector<double> position,
                              const linalg::Vector& gradient) {
        const std::size_t chain = static_cast<std::size_t>(index);
        if (gradient.length() != number_of_parameters()) {
            start_gradient_occupied_[chain] = false;
            return;
        }
        start_gradient_positions_[chain] = std::move(position);
        start_gradient_values_[chain] = gradient.to_array();
        start_gradient_occupied_[chain] = true;
    }

    static bool matches_bitwise(const std::vector<double>& stored,
                                const std::vector<double>& position) {
        return stored.size() == position.size() &&
               std::memcmp(stored.data(), position.data(),
                           stored.size() * sizeof(double)) == 0;
    }

    // Evaluates the log-likelihood, returning negative infinity if the parameters are out of
    // range. This prevents an out-of-range exception from propagating during leapfrog
    // integration when the sampler explores parameter values that violate distribution
    // constraints.
    double safe_log_likelihood(const std::vector<double>& parameters) const {
        try {
            return log_likelihood_function_(parameters);
        } catch (const std::out_of_range&) {
            return -std::numeric_limits<double>::infinity();
        }
    }

    linalg::Vector mass_;
    linalg::Vector inverse_mass_;
    distributions::Uniform step_size_u_{0.0, 0.2};
    distributions::UniformDiscrete steps_u_{1, 100};
    double step_size_ = 0.1;
    int steps_ = 50;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    Gradient gradient_function_;
    std::vector<std::vector<double>> start_gradient_positions_;
    std::vector<std::vector<double>> start_gradient_values_;
    std::vector<bool> start_gradient_occupied_;
};

}  // namespace corehydro::numerics::sampling::mcmc
