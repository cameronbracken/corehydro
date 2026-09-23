// ported from: Numerics/Sampling/MCMC/NUTS.cs @ 7e8e8d1
//
// No-U-Turn Sampler (NUTS): an adaptive extension of HMC that eliminates hand-tuning the
// leapfrog step size/step count. Each ChainIteration recursively doubles a binary tree of
// leapfrog states (BuildTree) in a randomly chosen direction until a U-turn is detected or
// MaxTreeDepth is reached, selects a candidate via multinomial sampling weighted by
// exp(-Hamiltonian), and during warmup adapts the step size via dual averaging (Hoffman &
// Gelman 2014, Algorithm 5) and -- when AdaptMassMatrix is enabled -- a diagonal mass
// matrix via Stan-style windowed Welford accumulation. NUTS ALWAYS ACCEPTS its candidate
// (AcceptCount is incremented unconditionally every ChainIteration -- it is not a
// Metropolis acceptance RATE here, unlike RWMH/ARWMH/DEMCz(s)/HMC; `acceptance_rate()` is
// therefore always 1.0 for every NUTS chain by construction).
//
// `HMC::Gradient` (this port's `hmc.hpp`) is reused verbatim as `NUTS::Gradient` -- C#
// itself types NUTS's constructor parameter/property as `HMC.Gradient`, not a distinct
// delegate. SafeLogLikelihood (the ArgumentOutOfRangeException -> -Infinity mapping) is
// re-implemented privately here exactly as it is in hmc.hpp -- the C# source does not share
// it between the two classes either (each is a private, non-virtual method on its own
// class).
//
// GRADIENT PATH (v2.1.4 fix, formerly a documented gotcha): both `leapfrog_in_place` (the
// step-size heuristic of Hoffman & Gelman Algorithm 4, used only by
// `find_reasonable_epsilon`/`try_single_step_log_acceptance`) and `leapfrog` (used by
// `build_tree`, i.e. every actual trajectory step) now call `gradient_function_(...)`.
// Previously `leapfrog_in_place` bypassed `gradient_function_` entirely and called the
// numerical gradient of SafeLogLikelihood directly, even when the caller supplied a custom
// `gradient_function` -- so a custom gradient affected the sampled trajectory but not the
// step-size heuristic. Both call sites now honor whatever gradient function is configured,
// matching upstream. Stream stability: for the DEFAULT gradient function (no custom
// `gradient_function` argument at construction), the default closure below performs EXACTLY
// the same bound-aware finite-difference computation `leapfrog_in_place`'s old hardcoded
// call did, so every seeded NUTS fixture (all of which use the default gradient) reproduces
// bit-for-bit unchanged by this fix.
//
// `Vector` (linalg::Vector) is a C++ value type (no reference semantics), so every C#
// `.Clone()` call in NUTS.cs is simply a copy in this port -- copy-construction/assignment
// already produces an independent value, matching `Clone()`'s intent with no explicit
// `clone()` member needed (see vector.hpp's own omission note).
//
// `Math.Max`/`Math.Min` on doubles are transcribed via the file-local `nan_max`/`nan_min`
// NaN-propagating helpers (matching running_statistics.hpp's `detail::nan_min`/`nan_max`
// convention) rather than `std::max`/`std::min`, which silently favor the non-NaN argument
// instead of propagating NaN. `Math.Max`/`Math.Min` calls on `int` operands (the adaptation
// window sizing in InitializeCustomSettings) use plain `std::max` since an `int` cannot be
// NaN.
//
// TreeState is a private nested struct exactly as in C# (a value type there too, so `var
// tree = BuildTree(...)` in the recursive case of BuildTree is a genuine independent copy
// in both languages).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/sampling/mcmc/hmc.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::sampling::mcmc {

namespace linalg = corehydro::numerics::math::linalg;
namespace diff = corehydro::numerics::math::differentiation;

namespace detail {
// Math.Max(double, double) / Math.Min(double, double) NaN-propagating narrow ports; see
// running_statistics.hpp's detail::nan_min/nan_max for the same convention.
inline double nuts_nan_max(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return b > a ? b : a;
}
inline double nuts_nan_min(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return b < a ? b : a;
}
}  // namespace detail

class NUTS : public MCMCSampler {
   public:
    // Reuses HMC's Gradient delegate type verbatim -- see file header.
    using Gradient = HMC::Gradient;

    // Constructs a new NUTS sampler. `mass`: optional initial momentum-distribution mass
    // vector (default = identity; adapted during warmup when AdaptMassMatrix is enabled).
    // `step_size`: initial leapfrog step size (default 0.1; adapted during warmup).
    // `max_tree_depth`: maximum binary tree depth, capping the trajectory at
    // 2^max_tree_depth leapfrog steps (default 10). `gradient_function`: optional custom
    // gradient function; default = a bound-aware finite-difference gradient of
    // SafeLogLikelihood.
    NUTS(std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors,
         LogLikelihood log_likelihood_function, std::optional<linalg::Vector> mass = std::nullopt,
         double step_size = 0.1, int max_tree_depth = 10, Gradient gradient_function = nullptr)
        : MCMCSampler(std::move(priors), std::move(log_likelihood_function)),
          mass_(mass.has_value() ? std::move(mass).value() : linalg::Vector(number_of_parameters(), 1.0)),
          inverse_mass_(number_of_parameters()) {
        set_initial_iterations(100 * number_of_parameters());

        // Set the inverse mass vector.
        for (int i = 0; i < number_of_parameters(); ++i) inverse_mass_[i] = 1.0 / mass_[i];

        // Set defaults.
        initial_step_size_ = step_size;
        max_tree_depth_ = max_tree_depth;

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

    // The maximum binary tree depth. Default = 10.
    int max_tree_depth() const { return max_tree_depth_; }
    void set_max_tree_depth(int value) { max_tree_depth_ = value; }

    // The function for evaluating the gradient of the log-likelihood.
    const Gradient& gradient_function() const { return gradient_function_; }

    // Target Metropolis acceptance probability for dual averaging adaptation.
    double target_acceptance_rate() const { return target_acceptance_rate_; }
    void set_target_acceptance_rate(double value) { target_acceptance_rate_ = value; }

    std::vector<double> hamiltonian_acceptance_rates() const {
        if (diagnostic_sample_counts_.size() != static_cast<std::size_t>(number_of_chains_) ||
            hamiltonian_acceptance_sums_.size() != static_cast<std::size_t>(number_of_chains_))
            return {};
        std::vector<double> result(static_cast<std::size_t>(number_of_chains_), 0.0);
        for (int i = 0; i < number_of_chains_; ++i) {
            const std::size_t chain = static_cast<std::size_t>(i);
            if (diagnostic_sample_counts_[chain] > 0)
                result[chain] = hamiltonian_acceptance_sums_[chain] /
                                diagnostic_sample_counts_[chain];
        }
        return result;
    }

    std::vector<double> reported_acceptance_rates() const override {
        return hamiltonian_acceptance_rates();
    }

    const std::vector<int>& diagnostic_sample_counts() const { return diagnostic_sample_counts_; }
    const std::vector<int>& divergence_counts() const { return divergence_counts_; }
    const std::vector<int>& max_tree_depth_hit_counts() const { return max_tree_depth_hit_counts_; }
    const std::vector<double>& step_sizes() const { return chain_step_sizes_; }

    std::vector<double> mean_tree_depths() const { return diagnostic_means(tree_depth_sums_); }
    std::vector<double> mean_leapfrog_steps() const { return diagnostic_means(leapfrog_step_sums_); }
    std::vector<double> energy_bfmi() const {
        if (diagnostic_sample_counts_.size() != static_cast<std::size_t>(number_of_chains_)) return {};
        std::vector<double> result(static_cast<std::size_t>(number_of_chains_));
        for (int i = 0; i < number_of_chains_; ++i) {
            const std::size_t chain = static_cast<std::size_t>(i);
            result[chain] = compute_energy_bfmi(diagnostic_sample_counts_[chain], energy_m2_[chain],
                                                energy_squared_difference_sums_[chain]);
        }
        return result;
    }

    static double compute_energy_bfmi(int sample_count, double energy_m2,
                                      double squared_difference_sum) {
        if (sample_count < 2 || !(energy_m2 > 0.0))
            return std::numeric_limits<double>::quiet_NaN();
        return (squared_difference_sum / sample_count) /
               (energy_m2 / (sample_count - 1.0));
    }

    // Whether to adapt the diagonal mass matrix during warmup. Default = true.
    bool adapt_mass_matrix = true;

   protected:
    void validate_custom_settings() override {
        if (mass_.length() != number_of_parameters())
            throw std::invalid_argument("The mass vector must be the same length as the number of parameters.");
        if (initial_step_size_ <= 0.0) throw std::invalid_argument("The leapfrog step size must be positive.");
        if (max_tree_depth_ < 1) throw std::invalid_argument("The maximum tree depth must be at least 1.");
        if (!std::isfinite(target_acceptance_rate_) || target_acceptance_rate_ <= 0.0 ||
            target_acceptance_rate_ >= 1.0)
            throw std::invalid_argument(
                "The target acceptance rate must be greater than 0 and less than 1.");
    }

    void initialize_custom_settings() override {
        int d = number_of_parameters();
        int n = number_of_chains_;

        // Initialize dual averaging state.
        chain_step_sizes_.assign(static_cast<std::size_t>(n), 0.0);
        chain_log_eps_bar_.assign(static_cast<std::size_t>(n), 0.0);
        chain_h_bar_.assign(static_cast<std::size_t>(n), 0.0);
        chain_mu_.assign(static_cast<std::size_t>(n), 0.0);
        chain_adapt_step_.assign(static_cast<std::size_t>(n), 0);

        hamiltonian_acceptance_sums_.assign(static_cast<std::size_t>(n), 0.0);
        diagnostic_sample_counts_.assign(static_cast<std::size_t>(n), 0);
        divergence_counts_.assign(static_cast<std::size_t>(n), 0);
        max_tree_depth_hit_counts_.assign(static_cast<std::size_t>(n), 0);
        tree_depth_sums_.assign(static_cast<std::size_t>(n), 0.0);
        leapfrog_step_sums_.assign(static_cast<std::size_t>(n), 0.0);
        energy_means_.assign(static_cast<std::size_t>(n), 0.0);
        energy_m2_.assign(static_cast<std::size_t>(n), 0.0);
        energy_squared_difference_sums_.assign(static_cast<std::size_t>(n), 0.0);
        previous_energy_.assign(static_cast<std::size_t>(n), 0.0);
        has_previous_energy_.assign(static_cast<std::size_t>(n), false);

        // Initialize diagonal mass matrix and Welford accumulators.
        welford_mean_.assign(static_cast<std::size_t>(n), std::vector<double>());
        welford_m2_.assign(static_cast<std::size_t>(n), std::vector<double>());
        welford_count_.assign(static_cast<std::size_t>(n), 0);
        mass_matrix_.assign(static_cast<std::size_t>(n), std::vector<double>());
        inverse_mass_matrix_.assign(static_cast<std::size_t>(n), std::vector<double>());

        gradient_cache_positions_.assign(
            static_cast<std::size_t>(n),
            std::vector<std::vector<double>>(kGradientCacheSize,
                                             std::vector<double>(static_cast<std::size_t>(d))));
        gradient_cache_values_ = gradient_cache_positions_;
        gradient_cache_occupied_.assign(
            static_cast<std::size_t>(n), std::vector<bool>(kGradientCacheSize, false));
        gradient_cache_next_slot_.assign(static_cast<std::size_t>(n), 0);

        for (int i = 0; i < n; ++i) {
            // Start with identity mass matrix (or user-provided mass).
            mass_matrix_[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(d));
            inverse_mass_matrix_[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(d));
            for (int j = 0; j < d; ++j) {
                mass_matrix_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = mass_[j];
                inverse_mass_matrix_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 1.0 / mass_[j];
            }

            welford_mean_[static_cast<std::size_t>(i)].assign(static_cast<std::size_t>(d), 0.0);
            welford_m2_[static_cast<std::size_t>(i)].assign(static_cast<std::size_t>(d), 0.0);
            welford_count_[static_cast<std::size_t>(i)] = 0;
        }

        // Compute adaptation window boundaries following Stan exactly.
        // Stan defaults: init_buffer=75, term_buffer=50, base_window=25.
        // Phase 1 (init_buffer): step size adaptation only, identity mass matrix.
        // Phase 2 (slow adaptation): mass matrix + step size in doubling windows.
        // Phase 3 (term_buffer): final step size tuning with fixed mass matrix.
        int total_warmup = warmup_iterations_ * thinning_interval_;

        // Stan default window sizes, scaled if warmup is too short.
        init_buffer_ = 75;
        term_buffer_ = 50;
        int base_window = 25;
        if (init_buffer_ + base_window + term_buffer_ > total_warmup) {
            // Fallback: redistribute proportionally.
            init_buffer_ = std::max(1, static_cast<int>(0.15 * total_warmup));
            term_buffer_ = std::max(1, static_cast<int>(0.10 * total_warmup));
            base_window = std::max(1, total_warmup - init_buffer_ - term_buffer_);
        }

        // Build doubling window boundaries with Stan's look-ahead merging. If the
        // next-next window would overshoot, stretch the current window.
        std::vector<int> window_ends;
        int adapt_end = total_warmup - term_buffer_;
        if (adapt_end > init_buffer_) {
            int window_size = base_window;
            int next_window_end = init_buffer_ + window_size - 1;

            while (next_window_end < adapt_end) {
                // Look ahead: if (current end + 2 * next window size) > adaptEnd, stretch
                // this window to fill the remaining space.
                int next_next_end = next_window_end + 2 * window_size;
                if (next_next_end >= adapt_end) {
                    next_window_end = adapt_end - 1;
                }
                window_ends.push_back(next_window_end);

                window_size *= 2;
                next_window_end += window_size;
            }
            // Ensure we always have the final boundary.
            if (window_ends.empty() || window_ends.back() != adapt_end - 1) {
                window_ends.push_back(adapt_end - 1);
            }
        }
        adapt_window_ends_ = std::move(window_ends);

        // Find reasonable initial step size per chain (Hoffman & Gelman Algorithm 4) and
        // initialize dual averaging state.
        // Note: chain_states_ is populated by initialize_chains() before this method is
        // called.
        for (int i = 0; i < n; ++i) {
            const std::vector<double>& theta0 = chain_states_[static_cast<std::size_t>(i)].values;
            double log_lh0 = chain_states_[static_cast<std::size_t>(i)].fitness;

            double eps0;
            try {
                eps0 = find_reasonable_epsilon(theta0, log_lh0, i);
            } catch (...) {
                eps0 = initial_step_size_;
            }

            chain_step_sizes_[static_cast<std::size_t>(i)] = eps0;
            chain_log_eps_bar_[static_cast<std::size_t>(i)] = std::log(eps0);
            chain_h_bar_[static_cast<std::size_t>(i)] = 0.0;
            chain_mu_[static_cast<std::size_t>(i)] = std::log(10.0 * eps0);
            chain_adapt_step_[static_cast<std::size_t>(i)] = 0;
        }
    }

    ParameterSet chain_iteration(int index, ParameterSet state) override {
        // Update the sample count.
        sample_count_[static_cast<std::size_t>(index)] += 1;
        const int sample_num = sample_count_[static_cast<std::size_t>(index)];
        const int warmup_steps = warmup_iterations_ * thinning_interval_;

        double eps = chain_step_sizes_[static_cast<std::size_t>(index)];
        int d = number_of_parameters();

        // Step 1: Sample momentum from N(0, M) using the per-chain mass matrix.
        linalg::Vector phi(d);
        for (int i = 0; i < d; ++i)
            phi[i] = std::sqrt(mass_matrix_[static_cast<std::size_t>(index)][static_cast<std::size_t>(i)]) *
                     distributions::Normal::standard_z(chain_prngs_[static_cast<std::size_t>(index)].next_double());

        // Compute initial Hamiltonian using the per-chain inverse mass matrix.
        double h0 = -state.fitness +
                    0.5 * diagonal_quadratic_form_vec(phi, inverse_mass_matrix_[static_cast<std::size_t>(index)]);

        // Step 2: Initialize tree.
        linalg::Vector theta(state.values);
        linalg::Vector theta_minus = theta;
        linalg::Vector theta_plus = theta;
        linalg::Vector r_minus = phi;
        linalg::Vector r_plus = phi;

        linalg::Vector candidate = theta;
        double candidate_log_lh = state.fitness;
        double log_sum_weight = -h0;

        int depth = 0;
        double sum_alpha = 0.0;
        int num_alpha = 0;
        int leapfrog_steps = 0;
        int trajectory_depth = 0;
        bool trajectory_divergent = false;

        // Step 3: Build tree by doubling until U-turn or max depth.
        while (depth < max_tree_depth_) {
            // Choose a random direction.
            int v = chain_prngs_[static_cast<std::size_t>(index)].next_double() < 0.5 ? -1 : 1;

            TreeState subtree;
            if (v == -1) {
                subtree = build_tree(theta_minus, r_minus, -eps, depth, h0, index);
                theta_minus = subtree.theta_minus;
                r_minus = subtree.momentum_minus;
            } else {
                subtree = build_tree(theta_plus, r_plus, eps, depth, h0, index);
                theta_plus = subtree.theta_plus;
                r_plus = subtree.momentum_plus;
            }

            // If the subtree is valid, consider accepting its candidate.
            if (subtree.valid) {
                double log_sum_weight_new = log_sum_exp(log_sum_weight, subtree.log_sum_weight);
                double accept_prob = std::exp(subtree.log_sum_weight - log_sum_weight_new);
                if (chain_prngs_[static_cast<std::size_t>(index)].next_double() < accept_prob) {
                    candidate = subtree.theta_prime;
                    candidate_log_lh = subtree.log_likelihood_prime;
                }
                log_sum_weight = log_sum_weight_new;
            }

            leapfrog_steps += subtree.leaf_count;
            trajectory_depth = depth + 1;
            trajectory_divergent = trajectory_divergent || subtree.divergent;

            // Accumulate adaptation statistics.
            sum_alpha += subtree.sum_alpha;
            num_alpha += subtree.num_alpha;

            // Check stopping criterion: divergence or U-turn at the top level.
            if (!subtree.valid) break;

            linalg::Vector d_theta = theta_plus - theta_minus;
            if (linalg::Vector::dot_product(d_theta, r_minus) < 0 || linalg::Vector::dot_product(d_theta, r_plus) < 0)
                break;

            ++depth;
        }

        const double average_acceptance = num_alpha > 0 ? sum_alpha / num_alpha : 0.0;
        if (sample_num <= warmup_steps) {
            // Always do dual averaging step size adaptation during warmup.
            double avg_alpha = num_alpha > 0 ? average_acceptance : target_acceptance_rate_;
            dual_averaging_update(index, avg_alpha);

            // Accumulate Welford statistics during mass matrix adaptation windows (Phase 2).
            if (adapt_mass_matrix && sample_num > init_buffer_ && sample_num <= warmup_steps - term_buffer_) {
                accumulate_welford_statistics(index, candidate.to_array());

                // Check if we're at the end of an adaptation window.
                if (is_end_of_adaptation_window(sample_num)) {
                    ParameterSet current_state(candidate.to_array(), candidate_log_lh);
                    update_mass_matrix(index, current_state);
                }
            }
        } else if (sample_num == warmup_steps + 1) {
            // After warmup, fix step size to the smoothed value.
            chain_step_sizes_[static_cast<std::size_t>(index)] = std::exp(chain_log_eps_bar_[static_cast<std::size_t>(index)]);
        }

        if (sample_num > warmup_steps) {
            record_diagnostics(index, average_acceptance, trajectory_divergent,
                               depth >= max_tree_depth_, trajectory_depth,
                               leapfrog_steps, h0);
        }

        // NUTS always accepts.
        accept_count_[static_cast<std::size_t>(index)] += 1;
        return ParameterSet(candidate.to_array(), candidate_log_lh);
    }

   private:
    std::vector<double> diagnostic_means(const std::vector<double>& sums) const {
        if (sums.size() != static_cast<std::size_t>(number_of_chains_) ||
            diagnostic_sample_counts_.size() != static_cast<std::size_t>(number_of_chains_))
            return {};
        std::vector<double> result(static_cast<std::size_t>(number_of_chains_), 0.0);
        for (int i = 0; i < number_of_chains_; ++i) {
            const std::size_t chain = static_cast<std::size_t>(i);
            if (diagnostic_sample_counts_[chain] > 0)
                result[chain] = sums[chain] / diagnostic_sample_counts_[chain];
        }
        return result;
    }

    void record_diagnostics(int chain_index, double acceptance_probability, bool divergent,
                            bool hit_maximum_depth, int tree_depth, int leapfrog_steps,
                            double energy) {
        const std::size_t chain = static_cast<std::size_t>(chain_index);
        const int count = ++diagnostic_sample_counts_[chain];
        hamiltonian_acceptance_sums_[chain] += acceptance_probability;
        if (divergent) ++divergence_counts_[chain];
        if (hit_maximum_depth) ++max_tree_depth_hit_counts_[chain];
        tree_depth_sums_[chain] += tree_depth;
        leapfrog_step_sums_[chain] += leapfrog_steps;
        if (has_previous_energy_[chain]) {
            const double difference = energy - previous_energy_[chain];
            energy_squared_difference_sums_[chain] += difference * difference;
        }
        previous_energy_[chain] = energy;
        has_previous_energy_[chain] = true;
        const double delta = energy - energy_means_[chain];
        energy_means_[chain] += delta / count;
        energy_m2_[chain] += delta * (energy - energy_means_[chain]);
    }

    // Finds a reasonable initial step size using the heuristic from Hoffman and Gelman
    // (2014), Algorithm 4. Searches for a step size that gives roughly 50% acceptance
    // probability for a single leapfrog step.
    double find_reasonable_epsilon(const std::vector<double>& theta0, double log_lh0, int chain_index) {
        int d = number_of_parameters();
        double epsilon = 1.0;

        // Sample momentum from the current mass matrix.
        std::vector<double> r0(static_cast<std::size_t>(d));
        for (int j = 0; j < d; ++j)
            r0[static_cast<std::size_t>(j)] =
                std::sqrt(mass_matrix_[static_cast<std::size_t>(chain_index)][static_cast<std::size_t>(j)]) *
                distributions::Normal::standard_z(chain_prngs_[static_cast<std::size_t>(chain_index)].next_double());

        // Compute initial Hamiltonian.
        double h0 = -log_lh0 + 0.5 * diagonal_quadratic_form(r0, inverse_mass_matrix_[static_cast<std::size_t>(chain_index)]);

        std::vector<double> theta_prime = theta0;
        std::vector<double> r_prime = r0;
        double log_alpha = try_single_step_log_acceptance(theta0, r0, theta_prime, r_prime, epsilon, h0, chain_index);

        // Determine direction: double epsilon if acceptance too high, halve if too low.
        double a = log_alpha > std::log(0.5) ? 1.0 : -1.0;

        for (int iter = 0; iter < 25; ++iter) {
            // Check if we've crossed the 0.5 threshold.
            if (a * log_alpha <= -a * std::log(2.0)) break;

            epsilon *= std::pow(2.0, a);

            // Safety bounds: don't let epsilon get absurdly small or large.
            if (epsilon < 1e-8 || epsilon > 1e6) break;

            theta_prime = theta0;
            r_prime = r0;
            log_alpha = try_single_step_log_acceptance(theta0, r0, theta_prime, r_prime, epsilon, h0, chain_index);
        }

        return detail::nuts_nan_max(1e-8, detail::nuts_nan_min(epsilon, 1e6));
    }

    // Attempts one leapfrog step and converts invalid Hamiltonian states to a low
    // log-acceptance value.
    double try_single_step_log_acceptance(const std::vector<double>& theta0, const std::vector<double>& r0,
                                           std::vector<double>& theta_prime, std::vector<double>& r_prime,
                                           double epsilon, double initial_hamiltonian, int chain_index) {
        try {
            leapfrog_in_place(theta_prime, r_prime, epsilon, chain_index);

            double log_lh = safe_log_likelihood(theta_prime);
            double hamiltonian =
                -log_lh + 0.5 * diagonal_quadratic_form(r_prime, inverse_mass_matrix_[static_cast<std::size_t>(chain_index)]);
            double log_alpha = initial_hamiltonian - hamiltonian;
            if (!corehydro::numerics::is_finite(log_alpha)) return -1000.0;

            return log_alpha;
        } catch (const std::domain_error&) {
            theta_prime = theta0;
            r_prime = r0;
            return -1000.0;
        }
    }

    const std::vector<double>& evaluate_gradient(const std::vector<double>& position,
                                                 int chain_index) {
        const std::size_t chain = static_cast<std::size_t>(chain_index);
        for (std::size_t slot = 0; slot < kGradientCacheSize; ++slot) {
            if (!gradient_cache_occupied_[chain][slot]) continue;
            const auto& stored = gradient_cache_positions_[chain][slot];
            if (stored.size() == position.size() &&
                std::memcmp(stored.data(), position.data(),
                            position.size() * sizeof(double)) == 0)
                return gradient_cache_values_[chain][slot];
        }

        const std::size_t slot = gradient_cache_next_slot_[chain];
        gradient_cache_occupied_[chain][slot] = false;
        gradient_cache_positions_[chain][slot] = position;
        const std::vector<double> gradient = gradient_function_(position).to_array();
        if (gradient.size() != static_cast<std::size_t>(number_of_parameters())) {
            malformed_gradient_ = std::move(gradient);
            return malformed_gradient_;
        }
        gradient_cache_values_[chain][slot] = gradient;
        gradient_cache_occupied_[chain][slot] = true;
        gradient_cache_next_slot_[chain] = (slot + 1) % kGradientCacheSize;
        return gradient_cache_values_[chain][slot];
    }

    // Performs a single leapfrog step in-place on raw arrays, using the per-chain mass
    // matrix. Used by find_reasonable_epsilon to avoid Vector allocations. Calls
    // gradient_function_ (v2.1.4 fix, formerly called diff::gradient directly) -- see file
    // header's gradient-path note.
    void leapfrog_in_place(std::vector<double>& theta, std::vector<double>& momentum, double epsilon, int chain_index) {
        int d = number_of_parameters();
        double half_eps = epsilon * 0.5;
        const std::vector<double>& inv_mass = inverse_mass_matrix_[static_cast<std::size_t>(chain_index)];

        // Half-step momentum update.
        const std::vector<double>* grad = &evaluate_gradient(theta, chain_index);
        for (int j = 0; j < d; ++j)
            momentum[static_cast<std::size_t>(j)] += (*grad)[static_cast<std::size_t>(j)] * half_eps;

        // Full-step position update.
        for (int j = 0; j < d; ++j) {
            theta[static_cast<std::size_t>(j)] +=
                inv_mass[static_cast<std::size_t>(j)] * momentum[static_cast<std::size_t>(j)] * epsilon;
            if (theta[static_cast<std::size_t>(j)] < lower_bounds_[static_cast<std::size_t>(j)])
                theta[static_cast<std::size_t>(j)] = lower_bounds_[static_cast<std::size_t>(j)] + corehydro::numerics::kDoubleMachineEpsilon;
            if (theta[static_cast<std::size_t>(j)] > upper_bounds_[static_cast<std::size_t>(j)])
                theta[static_cast<std::size_t>(j)] = upper_bounds_[static_cast<std::size_t>(j)] - corehydro::numerics::kDoubleMachineEpsilon;
        }

        // Half-step momentum update.
        grad = &evaluate_gradient(theta, chain_index);
        for (int j = 0; j < d; ++j)
            momentum[static_cast<std::size_t>(j)] += (*grad)[static_cast<std::size_t>(j)] * half_eps;
    }

    // Computes the diagonal quadratic form phi^T M^-1 phi using raw arrays.
    static double diagonal_quadratic_form(const std::vector<double>& momentum, const std::vector<double>& inverse_mass) {
        double sum = 0.0;
        for (std::size_t j = 0; j < momentum.size(); ++j) sum += momentum[j] * momentum[j] * inverse_mass[j];
        return sum;
    }

    // Accumulates sample statistics using Welford's online algorithm for computing the
    // diagonal mass matrix during warmup.
    void accumulate_welford_statistics(int chain_index, const std::vector<double>& sample) {
        welford_count_[static_cast<std::size_t>(chain_index)]++;
        int n = welford_count_[static_cast<std::size_t>(chain_index)];
        for (int j = 0; j < number_of_parameters(); ++j) {
            double delta = sample[static_cast<std::size_t>(j)] - welford_mean_[static_cast<std::size_t>(chain_index)][static_cast<std::size_t>(j)];
            welford_mean_[static_cast<std::size_t>(chain_index)][static_cast<std::size_t>(j)] += delta / n;
            double delta2 = sample[static_cast<std::size_t>(j)] - welford_mean_[static_cast<std::size_t>(chain_index)][static_cast<std::size_t>(j)];
            welford_m2_[static_cast<std::size_t>(chain_index)][static_cast<std::size_t>(j)] += delta * delta2;
        }
    }

    // Updates the diagonal mass matrix from the accumulated Welford statistics at the end
    // of an adaptation window. Resets Welford accumulators and dual averaging state so the
    // step size can re-adapt to the new mass matrix.
    void update_mass_matrix(int chain_index, const ParameterSet& current_state) {
        int n = welford_count_[static_cast<std::size_t>(chain_index)];
        if (n < 2) return;

        std::vector<double> estimated_variance(
            static_cast<std::size_t>(number_of_parameters()));
        double largest_window_variance = 0.0;
        for (int j = 0; j < number_of_parameters(); ++j) {
            double variance = welford_m2_[static_cast<std::size_t>(chain_index)][static_cast<std::size_t>(j)] / (n - 1);
            double prior_range = upper_bounds_[static_cast<std::size_t>(j)] - lower_bounds_[static_cast<std::size_t>(j)];
            double fallback_variance = (prior_range * prior_range) / 36.0;
            if (!corehydro::numerics::is_finite(fallback_variance) || fallback_variance <= 0) fallback_variance = 1.0;
            const bool usable = n >= kMinAdaptWindowCount && std::isfinite(variance) && variance > 0.0;
            estimated_variance[static_cast<std::size_t>(j)] =
                usable ? variance : fallback_variance;
            if (usable) largest_window_variance = std::max(largest_window_variance, variance);
        }
        const double variance_floor = largest_window_variance * kRelativeVarianceFloor;
        for (int j = 0; j < number_of_parameters(); ++j) {
            const double inverse_mass =
                std::max(estimated_variance[static_cast<std::size_t>(j)], variance_floor);
            inverse_mass_matrix_[static_cast<std::size_t>(chain_index)][static_cast<std::size_t>(j)] = inverse_mass;
            mass_matrix_[static_cast<std::size_t>(chain_index)][static_cast<std::size_t>(j)] = 1.0 / inverse_mass;
        }

        // Reset Welford accumulators for next window.
        std::fill(welford_mean_[static_cast<std::size_t>(chain_index)].begin(),
                  welford_mean_[static_cast<std::size_t>(chain_index)].end(), 0.0);
        std::fill(welford_m2_[static_cast<std::size_t>(chain_index)].begin(),
                  welford_m2_[static_cast<std::size_t>(chain_index)].end(), 0.0);
        welford_count_[static_cast<std::size_t>(chain_index)] = 0;

        // Critical: after metric change, find a new reasonable step size (Stan's
        // init_stepsize), then reset dual averaging to re-adapt from the new starting point.
        double eps0;
        try {
            eps0 = find_reasonable_epsilon(current_state.values, current_state.fitness, chain_index);
        } catch (...) {
            eps0 = chain_step_sizes_[static_cast<std::size_t>(chain_index)];
        }
        chain_step_sizes_[static_cast<std::size_t>(chain_index)] = eps0;
        chain_h_bar_[static_cast<std::size_t>(chain_index)] = 0.0;
        chain_adapt_step_[static_cast<std::size_t>(chain_index)] = 0;
        chain_mu_[static_cast<std::size_t>(chain_index)] = std::log(10.0 * eps0);
        chain_log_eps_bar_[static_cast<std::size_t>(chain_index)] = std::log(eps0);
    }

    // Checks whether the given sample number falls at the end of a mass matrix adaptation
    // window.
    bool is_end_of_adaptation_window(int sample_num) const {
        for (int end : adapt_window_ends_)
            if (sample_num == end) return true;
        return false;
    }

    // Internal state of a binary tree node used during the NUTS tree-building recursion.
    struct TreeState {
        // Leftmost position in the subtree.
        linalg::Vector theta_minus{0};
        // Leftmost momentum in the subtree.
        linalg::Vector momentum_minus{0};
        // Rightmost position in the subtree.
        linalg::Vector theta_plus{0};
        // Rightmost momentum in the subtree.
        linalg::Vector momentum_plus{0};
        // Candidate position selected by multinomial sampling.
        linalg::Vector theta_prime{0};
        // Log of the sum of weights for multinomial sampling.
        double log_sum_weight = -std::numeric_limits<double>::infinity();
        // Log-likelihood of the candidate position.
        double log_likelihood_prime = -std::numeric_limits<double>::infinity();
        // Number of leaf nodes in the subtree.
        int leaf_count = 0;
        // Whether the subtree is valid (no divergence, no U-turn).
        bool valid = false;
        // Whether the subtree contains a divergent or non-finite trajectory.
        bool divergent = false;
        // Sum of per-leaf Metropolis acceptance probabilities (for dual averaging).
        double sum_alpha = 0.0;
        // Number of leaves contributing to sum_alpha.
        int num_alpha = 0;
    };

    // Creates an invalid tree state for a trajectory that entered a non-finite log-density
    // region.
    static TreeState invalid_tree_state(const linalg::Vector& theta, const linalg::Vector& momentum) {
        TreeState s;
        s.theta_minus = theta;
        s.momentum_minus = momentum;
        s.theta_plus = theta;
        s.momentum_plus = momentum;
        s.theta_prime = theta;
        s.log_sum_weight = -std::numeric_limits<double>::infinity();
        s.log_likelihood_prime = -std::numeric_limits<double>::infinity();
        s.leaf_count = 1;
        s.valid = false;
        s.divergent = true;
        s.sum_alpha = 0.0;
        s.num_alpha = 1;
        return s;
    }

    // Performs a single leapfrog integration step with boundary enforcement, using the
    // per-chain diagonal mass matrix.
    std::pair<linalg::Vector, linalg::Vector> leapfrog(const linalg::Vector& theta, const linalg::Vector& momentum,
                                                         double epsilon, int chain_index) {
        int d = number_of_parameters();
        const std::vector<double>& inv_mass = inverse_mass_matrix_[static_cast<std::size_t>(chain_index)];

        // Half-step momentum update.
        linalg::Vector grad(evaluate_gradient(theta.to_array(), chain_index));
        linalg::Vector r = momentum + grad * (epsilon * 0.5);

        // Full-step position update using the per-chain inverse mass matrix.
        linalg::Vector q(d);
        for (int j = 0; j < d; ++j) q[j] = theta[j] + inv_mass[static_cast<std::size_t>(j)] * r[j] * epsilon;

        // Enforce parameter bounds.
        for (int j = 0; j < d; ++j) {
            if (q[j] < prior_distributions_[static_cast<std::size_t>(j)]->minimum())
                q[j] = prior_distributions_[static_cast<std::size_t>(j)]->minimum() + corehydro::numerics::kDoubleMachineEpsilon;
            if (q[j] > prior_distributions_[static_cast<std::size_t>(j)]->maximum())
                q[j] = prior_distributions_[static_cast<std::size_t>(j)]->maximum() - corehydro::numerics::kDoubleMachineEpsilon;
        }

        // Half-step momentum update.
        grad = linalg::Vector(evaluate_gradient(q.to_array(), chain_index));
        r = r + grad * (epsilon * 0.5);

        return {q, r};
    }

    // Computes the diagonal quadratic form phi^T M^-1 phi using a Vector and raw array.
    static double diagonal_quadratic_form_vec(const linalg::Vector& phi, const std::vector<double>& inverse_mass) {
        double sum = 0.0;
        for (int j = 0; j < phi.length(); ++j) sum += phi[j] * phi[j] * inverse_mass[static_cast<std::size_t>(j)];
        return sum;
    }

    // Recursively builds a balanced binary tree of leapfrog states.
    TreeState build_tree(const linalg::Vector& theta, const linalg::Vector& momentum, double epsilon, int depth,
                          double h0, int chain_index) {
        if (depth == 0) {
            // Base case: take one leapfrog step.
            linalg::Vector theta_prime{0};
            linalg::Vector momentum_prime{0};
            try {
                auto pair = leapfrog(theta, momentum, epsilon, chain_index);
                theta_prime = pair.first;
                momentum_prime = pair.second;
            } catch (const std::domain_error&) {
                return invalid_tree_state(theta, momentum);
            }

            double log_lh = safe_log_likelihood(theta_prime.to_array());
            double h = -log_lh + 0.5 * diagonal_quadratic_form_vec(momentum_prime, inverse_mass_matrix_[static_cast<std::size_t>(chain_index)]);
            double log_weight = -h;
            if (!corehydro::numerics::is_finite(log_lh) || !corehydro::numerics::is_finite(h) ||
                !corehydro::numerics::is_finite(log_weight))
                return invalid_tree_state(theta_prime, momentum_prime);

            bool divergent = (h - h0) > kMaxDeltaH;
            double alpha = detail::nuts_nan_min(1.0, std::exp(h0 - h));
            if (std::isnan(alpha)) alpha = 0.0;

            TreeState s;
            s.theta_minus = theta_prime;
            s.momentum_minus = momentum_prime;
            s.theta_plus = theta_prime;
            s.momentum_plus = momentum_prime;
            s.theta_prime = theta_prime;
            s.log_sum_weight = log_weight;
            s.log_likelihood_prime = log_lh;
            s.leaf_count = 1;
            s.valid = !divergent;
            s.divergent = divergent;
            s.sum_alpha = alpha;
            s.num_alpha = 1;
            return s;
        }

        // Recursive case: build first half-tree.
        TreeState tree = build_tree(theta, momentum, epsilon, depth - 1, h0, chain_index);

        if (tree.valid) {
            // Build second half-tree.
            TreeState tree2;
            if (epsilon > 0) {
                tree2 = build_tree(tree.theta_plus, tree.momentum_plus, epsilon, depth - 1, h0, chain_index);
                tree.theta_plus = tree2.theta_plus;
                tree.momentum_plus = tree2.momentum_plus;
            } else {
                tree2 = build_tree(tree.theta_minus, tree.momentum_minus, epsilon, depth - 1, h0, chain_index);
                tree.theta_minus = tree2.theta_minus;
                tree.momentum_minus = tree2.momentum_minus;
            }

            // Multinomial sampling: accept candidate from tree2 with appropriate
            // probability.
            double log_sum_weight_new = log_sum_exp(tree.log_sum_weight, tree2.log_sum_weight);
            double accept_tree2_prob = std::exp(tree2.log_sum_weight - log_sum_weight_new);
            if (chain_prngs_[static_cast<std::size_t>(chain_index)].next_double() < accept_tree2_prob) {
                tree.theta_prime = tree2.theta_prime;
                tree.log_likelihood_prime = tree2.log_likelihood_prime;
            }

            tree.log_sum_weight = log_sum_weight_new;
            tree.leaf_count += tree2.leaf_count;
            tree.divergent = tree.divergent || tree2.divergent;
            tree.sum_alpha += tree2.sum_alpha;
            tree.num_alpha += tree2.num_alpha;

            // Check U-turn criterion on the combined tree.
            linalg::Vector d_theta = tree.theta_plus - tree.theta_minus;
            bool uturn = linalg::Vector::dot_product(d_theta, tree.momentum_minus) < 0 ||
                         linalg::Vector::dot_product(d_theta, tree.momentum_plus) < 0;
            tree.valid = tree2.valid && !uturn;
        }

        return tree;
    }

    // Updates the step size using the dual averaging scheme from Hoffman and Gelman
    // (2014), Algorithm 5.
    void dual_averaging_update(int chain_index, double avg_accept_prob) {
        chain_adapt_step_[static_cast<std::size_t>(chain_index)]++;
        int m = chain_adapt_step_[static_cast<std::size_t>(chain_index)];

        // Update running average of the acceptance statistic.
        chain_h_bar_[static_cast<std::size_t>(chain_index)] =
            (1.0 - 1.0 / (m + kT0)) * chain_h_bar_[static_cast<std::size_t>(chain_index)] +
            (target_acceptance_rate_ - avg_accept_prob) / (m + kT0);

        // Compute new log step size.
        double log_eps = chain_mu_[static_cast<std::size_t>(chain_index)] -
                          std::sqrt(static_cast<double>(m)) / kGamma * chain_h_bar_[static_cast<std::size_t>(chain_index)];

        // Update smoothed log step size (exponential moving average).
        double m_pow = std::pow(static_cast<double>(m), -kKappa);
        chain_log_eps_bar_[static_cast<std::size_t>(chain_index)] =
            m_pow * log_eps + (1.0 - m_pow) * chain_log_eps_bar_[static_cast<std::size_t>(chain_index)];

        // Set current step size (during adaptation, use the un-smoothed value).
        chain_step_sizes_[static_cast<std::size_t>(chain_index)] = std::exp(log_eps);

        // Clamp step size to prevent extreme values.
        if (chain_step_sizes_[static_cast<std::size_t>(chain_index)] < 1e-10)
            chain_step_sizes_[static_cast<std::size_t>(chain_index)] = 1e-10;
        if (chain_step_sizes_[static_cast<std::size_t>(chain_index)] > 1e5)
            chain_step_sizes_[static_cast<std::size_t>(chain_index)] = 1e5;
    }

    // Computes log(exp(a) + exp(b)) in a numerically stable way.
    static double log_sum_exp(double a, double b) {
        double max = detail::nuts_nan_max(a, b);
        if (max == -std::numeric_limits<double>::infinity()) return -std::numeric_limits<double>::infinity();
        return max + std::log(std::exp(a - max) + std::exp(b - max));
    }

    // Evaluates the log-likelihood, returning negative infinity if the parameters are out
    // of range. This prevents an out-of-range exception from propagating during leapfrog
    // integration when the sampler explores parameter values that violate distribution
    // constraints.
    double safe_log_likelihood(const std::vector<double>& parameters) const {
        try {
            return log_likelihood_function_(parameters);
        } catch (const std::out_of_range&) {
            return -std::numeric_limits<double>::infinity();
        }
    }

    // Dual averaging hyperparameters (Hoffman & Gelman 2014, Section 3.2).
    static constexpr double kDeltaTarget = 0.80;
    static constexpr double kGamma = 0.05;
    static constexpr double kT0 = 10.0;
    static constexpr double kKappa = 0.75;

    // Divergence threshold: if H - H0 exceeds this, the trajectory is considered divergent.
    static constexpr double kMaxDeltaH = 1000.0;
    static constexpr std::size_t kGradientCacheSize = 4;
    static constexpr int kMinAdaptWindowCount = 10;
    static constexpr double kRelativeVarianceFloor = 1e-12;

    linalg::Vector mass_;
    linalg::Vector inverse_mass_;
    double initial_step_size_ = 0.1;
    int max_tree_depth_ = 10;
    double target_acceptance_rate_ = kDeltaTarget;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    Gradient gradient_function_;

    // Per-chain dual averaging state.
    std::vector<double> chain_step_sizes_;
    std::vector<double> chain_log_eps_bar_;
    std::vector<double> chain_h_bar_;
    std::vector<double> chain_mu_;
    std::vector<int> chain_adapt_step_;

    std::vector<double> hamiltonian_acceptance_sums_;
    std::vector<int> diagnostic_sample_counts_;
    std::vector<int> divergence_counts_;
    std::vector<int> max_tree_depth_hit_counts_;
    std::vector<double> tree_depth_sums_;
    std::vector<double> leapfrog_step_sums_;
    std::vector<double> energy_means_;
    std::vector<double> energy_m2_;
    std::vector<double> energy_squared_difference_sums_;
    std::vector<double> previous_energy_;
    std::vector<bool> has_previous_energy_;

    // Per-chain diagonal mass matrix adaptation (Welford's online algorithm).
    std::vector<std::vector<double>> welford_mean_;
    std::vector<std::vector<double>> welford_m2_;
    std::vector<int> welford_count_;
    std::vector<std::vector<double>> mass_matrix_;
    std::vector<std::vector<double>> inverse_mass_matrix_;

    std::vector<std::vector<std::vector<double>>> gradient_cache_positions_;
    std::vector<std::vector<std::vector<double>>> gradient_cache_values_;
    std::vector<std::vector<bool>> gradient_cache_occupied_;
    std::vector<std::size_t> gradient_cache_next_slot_;
    std::vector<double> malformed_gradient_;

    // Adaptation window boundaries (computed per chain in initialize_custom_settings()).
    int init_buffer_ = 0;
    int term_buffer_ = 0;
    std::vector<int> adapt_window_ends_;
};

}  // namespace corehydro::numerics::sampling::mcmc
