// ported from: Numerics/Mathematics/Integration/Vegas.cs @ 2a0357a
//
// Vegas: adaptive importance-sampling Monte Carlo integration (Lepage's algorithm, "Numerical
// Recipes" Sec. 7.8), with the upstream Power Transform enhancement for rare-tail-event sampling
// (`ApplyPowerTransform`/`PowerTransformJacobian`/`TailFocusParameter`/
// `configure_for_rare_events`). `vegas_algorithm()`/`refine_grid()` below are direct,
// method-for-method transcriptions of the C# private `vegas()`/`RefineGrid()` (`vegas()` is
// renamed here to avoid colliding with the class name).
//
// STRUCTURAL NOTE: C# stores its algorithm-internal loop counters (`i`, `it`, `j`, `k`,
// `gridRefinementFlag`, ...) as class-instance fields rather than method locals -- almost
// certainly a mechanical habit (or C# 7-vs-earlier scoping preference) rather than a deliberate
// design, since nothing outside `vegas()`/`RefineGrid()` reads them. This port keeps the fields
// that genuinely carry state ACROSS `integrate()` calls (the grid, bin contributions, the running
// sums, `previous_bin_count_`/`current_bin_count_`/`grid_refinement_flag_`/... -- everything the
// `initialize > 0` "inherit the grid" re-entry path depends on) as private members, matching C#
// 1:1, but declares the pure iteration-scratch counters (`i`, `j`, `k`, `it`, `n`) as C++ locals
// inside the methods that use them -- semantically identical, since no external code ever reads
// them between calls.
//
// SOBOL PATH (corehydro addition, no C# counterpart): see miser.hpp's file header for the
// rationale -- the same applies here. `_sobol` is unconditionally constructed in the ctor exactly
// as C# does, so the ctor below takes a trailing `sobol_path` argument (default `""`, valid only
// when `dimensions == 1`).
//
// RNG: see monte_carlo_integration.hpp's file header for the `random` member's C#-parity note.
// Vegas only reads `random` (via `next_doubles`) when `use_sobol_sequence` is false.
//
// GRID ACCESSOR: C#'s `Grid` property returns the raw `double[Dimensions, NumberOfBins]` 2-D
// array (`stratificationGrid[dimension, bin]`). This port exposes the same data as
// `grid()`, a `const std::vector<std::vector<double>>&` sized `[dimensions][number_of_bins]` --
// i.e. row-major by DIMENSION (each outer element is one dimension's full bin-edge row), matching
// the C# array's first index.
#pragma once
#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/integration/support/integration_status.hpp"
#include "corehydro/numerics/math/integration/support/integrator.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/sampling/sobol.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::integration {

/// Vegas: adaptive Monte Carlo integration for multidimensional integrands, with an optional
/// Power Transform for rare tail-event sampling.
class Vegas : public Integrator {
   public:
    /// Constructs a new Vegas class. `function` is the multidimensional function to integrate --
    /// its second argument is the sample weight -- `dimensions` the number of dimensions it
    /// takes (max 20), `min`/`max` the per-dimension bounds the integral is computed under.
    /// `sobol_path` is a corehydro addition -- see the file header.
    Vegas(std::function<double(const std::vector<double>&, double)> function, int dimensions,
          std::vector<double> min, std::vector<double> max, const std::string& sobol_path = "")
        : dimensions_(validate_dimensions(dimensions, min, max)), sobol_(dimensions_, sobol_path) {
        for (std::size_t i = 0; i < min.size(); ++i) {
            if (max[i] <= min[i])
                throw std::out_of_range(
                    "The maximum values cannot be less than or equal to the minimum values.");
        }
        if (!function) throw std::invalid_argument("The function cannot be null.");

        function_ = std::move(function);
        min_ = std::move(min);
        max_ = std::move(max);

        relative_tolerance = 1E-3;
        initialize_parameters();
    }

    /// The multidimensional function to integrate; its second argument is the sample weight.
    const std::function<double(const std::vector<double>&, double)>& function() const {
        return function_;
    }

    /// The number of dimensions in the function to evaluate.
    int dimensions() const { return dimensions_; }

    /// The minimum values under which the integral must be computed.
    const std::vector<double>& min() const { return min_; }

    /// The maximum values under which the integral must be computed.
    const std::vector<double>& max() const { return max_; }

    /// The random number generator used within the Monte Carlo integration. See
    /// monte_carlo_integration.hpp's file header.
    sampling::MersenneTwister random;

    /// Determines whether to use a Sobol sequence or a pseudo-random number generator.
    bool use_sobol_sequence = true;

    /// Determines whether to check convergence and exit when integrating.
    bool check_convergence = true;

    /// Determines how to initialize the Vegas routine. If 0, cold start. If 1, inherit the grid
    /// from a previous call but not its answers. If 2, inherit the grid and its answers. Default
    /// = 0.
    int initialize = 0;

    /// The number of statistically independent evaluations of the integral, per iteration.
    /// Default = 1000.
    int independent_evaluations = 1000;

    /// The number of function evaluations within each independent evaluation. Default = 10,000.
    int function_calls = 10000;

    /// The refinement damping parameter for the grid. Default = 1.5.
    double alpha = 1.5;

    /// The number of stratification bins for each dimension. Default = 50. Setting this
    /// re-initializes the parameter arrays (matches the C# property setter).
    int number_of_bins() const { return number_of_bins_; }
    void set_number_of_bins(int value) {
        number_of_bins_ = value;
        initialize_parameters();
    }

    /// Power transform parameter for tail-focused rare-event sampling. Default = 1.0 (standard
    /// uniform sampling, backward compatible). Set gamma > 1 to focus sampling on the upper tail
    /// (p -> 1) for rare events. The power transform uses p' = 1 - (1-p)^gamma to concentrate
    /// samples in the upper tail; weights are corrected by the Jacobian dp'/dp = gamma(1-p)^
    /// (gamma-1), which stays O(1) -- unlike z-space transforms, this stays numerically stable in
    /// high dimensions.
    double tail_focus_parameter = 1.0;

    /// The stratification grid boundaries, sized `[dimensions][number_of_bins]` -- see the file
    /// header's GRID ACCESSOR note.
    const std::vector<std::vector<double>>& grid() const { return stratification_grid_; }

    /// The integration standard error.
    double standard_error() const { return standard_error_; }

    /// The Chi-Squared statistic.
    double chi_squared() const { return chi_squared_; }

    /// Configures Vegas for rare tail events, automatically setting `tail_focus_parameter` based
    /// on `target_probability` (e.g. 1e-6). Chosen so the target probability appears in ~5% of
    /// transformed samples: gamma ~= ln(targetProbability) / ln(0.05), clamped to [1, 20].
    void configure_for_rare_events(double target_probability) {
        double gamma = std::log(target_probability) / std::log(0.05);
        gamma = std::max(1.0, std::min(gamma, 20.0));

        tail_focus_parameter = gamma;
        set_number_of_bins(std::max(100, number_of_bins()));
        alpha = 1.8;  // More aggressive grid adaptation
    }

    /// Evaluates the integral.
    void integrate() override {
        validate();

        try {
            vegas_algorithm();

            if (function_evaluations_ >= max_function_evaluations) {
                status_ = IntegrationStatus::MaximumFunctionEvaluationsReached;
            } else {
                status_ = IntegrationStatus::Success;
            }
        } catch (...) {
            status_ = IntegrationStatus::Failure;
            if (report_failure) throw;
        }
    }

   private:
    // Validates dimensions (>= 1, matching min/max length, <= kMaxDimensions) and returns it, so
    // the checks run -- and throw C#'s ArgumentOutOfRangeException-mapped std::out_of_range --
    // BEFORE `sobol_`'s member initializer constructs a SobolSequence with a possibly-invalid
    // dimension. Member initializers run in declaration order, so this must run ahead of `sobol_`
    // below (mirrors miser.hpp's `require_dimensions`, extended with the two extra C# ctor
    // checks that precede `_sobol`'s construction there too).
    static int validate_dimensions(int dimensions, const std::vector<double>& min,
                                   const std::vector<double>& max) {
        if (dimensions < 1)
            throw std::out_of_range("There must be at least 1 dimension to evaluate.");
        if (static_cast<int>(min.size()) != dimensions || static_cast<int>(max.size()) != dimensions)
            throw std::out_of_range(
                "The minimum and maximum values must be the same length as the number of "
                "dimensions.");
        if (dimensions > kMaxDimensions)
            throw std::out_of_range("The maximum number of dimensions is 20.");
        return dimensions;
    }

    static constexpr int kMaxDimensions = 20;
    static constexpr double kTinyValue = 1.0e-30;

    std::function<double(const std::vector<double>&, double)> function_;
    int dimensions_ = 0;
    std::vector<double> min_;
    std::vector<double> max_;
    sampling::SobolSequence sobol_;

    std::vector<double> region_;  // computed, unused elsewhere -- mirrors C#'s dead `_region`.
    int number_of_bins_ = 50;
    double standard_error_ = 0.0;
    double chi_squared_ = 0.0;

    // Persistent algorithm state -- carried across `integrate()` calls by the `initialize > 0`
    // re-entry path, mirroring the C# instance fields 1:1 (see the file header's STRUCTURAL NOTE
    // for why these stay members while pure loop counters do not).
    int grid_refinement_flag_ = 1;
    int current_bin_count_ = 0;
    int previous_bin_count_ = 1;
    int stratification_levels_ = 0;
    int samples_per_bin_group_ = 0;
    double total_function_calls_ = 0.0;
    double bin_volume_ = 0.0;
    double bin_spacing_ = 0.0;
    double jacobian_determinant_ = 0.0;
    double normalized_bin_count_ = 0.0;
    double sum_chi_squared_ = 0.0;
    double sum_weighted_results_ = 0.0;
    double sum_integration_weights_ = 0.0;

    std::vector<int> bin_indexes_;
    std::vector<int> region_bin_counts_;
    std::vector<double> dimension_contributions_;
    std::vector<double> dx_;
    std::vector<double> bin_densities_;
    std::vector<double> random_point_;
    std::vector<double> bin_edges_;
    std::vector<std::vector<double>> bin_contributions_;           // [number_of_bins][dimensions]
    std::vector<std::vector<double>> refined_bin_contributions_;   // [number_of_bins][dimensions]
    std::vector<std::vector<double>> stratification_grid_;         // [dimensions][number_of_bins]

    // Initializes the parameter arrays.
    void initialize_parameters() {
        bin_indexes_.assign(static_cast<std::size_t>(dimensions_), 0);
        region_bin_counts_.assign(static_cast<std::size_t>(dimensions_), 0);
        dimension_contributions_.assign(static_cast<std::size_t>(dimensions_), 0.0);
        dx_.assign(static_cast<std::size_t>(dimensions_), 0.0);
        bin_densities_.assign(static_cast<std::size_t>(number_of_bins_), 0.0);
        random_point_.assign(static_cast<std::size_t>(dimensions_), 0.0);
        bin_edges_.assign(static_cast<std::size_t>(number_of_bins_), 0.0);
        bin_contributions_.assign(static_cast<std::size_t>(number_of_bins_),
                                  std::vector<double>(static_cast<std::size_t>(dimensions_), 0.0));
        refined_bin_contributions_.assign(
            static_cast<std::size_t>(number_of_bins_),
            std::vector<double>(static_cast<std::size_t>(dimensions_), 0.0));
        stratification_grid_.assign(
            static_cast<std::size_t>(dimensions_),
            std::vector<double>(static_cast<std::size_t>(number_of_bins_), 0.0));

        // Create region array
        region_.assign(static_cast<std::size_t>(2 * dimensions_), 0.0);
        for (int i = 0; i < dimensions_; ++i) region_[static_cast<std::size_t>(i)] =
            min_[static_cast<std::size_t>(i)];
        for (int i = dimensions_; i < 2 * dimensions_; ++i)
            region_[static_cast<std::size_t>(i)] = max_[static_cast<std::size_t>(i - dimensions_)];

        iterations_ = 0;
    }

    // Applies the power transform to probability `p` for tail-focused sampling: p' = 1-(1-p)^gamma
    // concentrates samples in the upper tail when gamma > 1. Identity when gamma == 1.
    double apply_power_transform(double p) const {
        double gamma = tail_focus_parameter;
        if (std::fabs(gamma - 1.0) < 1e-10) return p;
        return 1.0 - std::pow(1.0 - p, gamma);
    }

    // Computes the Jacobian for the power transform: dp'/dp = gamma(1-p)^(gamma-1). Returns 1.0
    // (identity) when gamma == 1. (1-p) is clamped away from zero to avoid numerical issues.
    double power_transform_jacobian(double p) const {
        double gamma = tail_focus_parameter;
        if (std::fabs(gamma - 1.0) < 1e-10) return 1.0;
        double one_minus_p = std::max(1.0 - p, 1e-15);
        return gamma * std::pow(one_minus_p, gamma - 1.0);
    }

    // Performs the Vegas algorithm. Direct transcription of the C# private `vegas()` method.
    void vegas_algorithm() {
        if (initialize <= 0) {
            // Normal entry. Enter here on a cold start. Set grid_refinement_flag_ = 0 to disable
            // stratified sampling, i.e. use importance sampling only.
            grid_refinement_flag_ = previous_bin_count_ = 1;
            for (int d = 0; d < dimensions_; ++d) {
                stratification_grid_[static_cast<std::size_t>(d)][0] = 1.0;
            }
        }
        if (initialize <= 1) {
            // Enter here to inherit the grid from a previous call, but not its answers.
            sum_weighted_results_ = sum_integration_weights_ = sum_chi_squared_ = 0.0;
        }
        if (initialize <= 2) {
            // Inherit the grid and previous results for further refinement.
            current_bin_count_ = number_of_bins_;
            stratification_levels_ = 1;

            // Set up stratification grid based on the number of function calls and dimensions
            if (grid_refinement_flag_ != 0) {
                stratification_levels_ = static_cast<int>(
                    std::pow(function_calls / 2.0 + 0.25, 1.0 / dimensions_));
                grid_refinement_flag_ = 1;

                if ((2 * stratification_levels_ - number_of_bins_) >= 0) {
                    grid_refinement_flag_ = -1;
                    samples_per_bin_group_ = stratification_levels_ / number_of_bins_ + 1;
                    current_bin_count_ = stratification_levels_ / samples_per_bin_group_;
                    stratification_levels_ = samples_per_bin_group_ * current_bin_count_;
                }
            }

            int total_sample_groups = 1;
            for (int d = 0; d < dimensions_; ++d) total_sample_groups *= stratification_levels_;

            samples_per_bin_group_ = std::max(function_calls / total_sample_groups, 2);
            total_function_calls_ =
                static_cast<double>(samples_per_bin_group_) * static_cast<double>(total_sample_groups);

            bin_volume_ = 1.0;
            bin_spacing_ = 1.0 / stratification_levels_;
            for (int d = 0; d < dimensions_; ++d) bin_volume_ *= bin_spacing_;

            bin_volume_ = sqr(total_function_calls_ * bin_volume_) / samples_per_bin_group_ /
                         samples_per_bin_group_ / (samples_per_bin_group_ - 1.0);
            normalized_bin_count_ = current_bin_count_;
            bin_spacing_ *= normalized_bin_count_;

            jacobian_determinant_ = 1.0 / total_function_calls_;
            for (int d = 0; d < dimensions_; ++d) {
                std::size_t dd = static_cast<std::size_t>(d);
                dx_[dd] = max_[dd] - min_[dd];
                jacobian_determinant_ *= dx_[dd];
            }

            // Refine the grid binning if necessary
            if (current_bin_count_ != previous_bin_count_) {
                for (int b = 0; b < std::max(current_bin_count_, previous_bin_count_); ++b) {
                    bin_densities_[static_cast<std::size_t>(b)] = 1.0;
                }
                for (int d = 0; d < dimensions_; ++d) {
                    refine_grid(static_cast<double>(previous_bin_count_) / normalized_bin_count_,
                               current_bin_count_, bin_densities_, bin_edges_, stratification_grid_,
                               d);
                }
                previous_bin_count_ = current_bin_count_;
            }
        }

        // Main iteration loop. Can enter here (initialize >= 3) to do additional independent
        // evaluations with all other parameters unchanged.
        for (int it = 0; it < independent_evaluations; ++it) {
            iterations_++;

            double total_integral_estimate = 0.0, total_error_estimate = 0.0;

            // Initialize bin contributions and stratification indexes
            for (int d = 0; d < dimensions_; ++d) {
                std::size_t dd = static_cast<std::size_t>(d);
                region_bin_counts_[dd] = 1;
                for (int b = 0; b < current_bin_count_; ++b) {
                    bin_contributions_[static_cast<std::size_t>(b)][dd] =
                        refined_bin_contributions_[static_cast<std::size_t>(b)][dd] = 0.0;
                }
            }

            // Stratified sampling loop
            int rk = 0;
            while (true) {
                double function_value_sum = 0.0, function_value_squared_sum = 0.0;
                for (int s = 0; s < samples_per_bin_group_; ++s) {
                    double weight = jacobian_determinant_;

                    // Generate random or Sobol sequences
                    std::vector<double> rnd =
                        use_sobol_sequence ? sobol_.next_double() : random.next_doubles(dimensions_);

                    for (int d = 0; d < dimensions_; ++d) {
                        std::size_t dd = static_cast<std::size_t>(d);
                        double random_bin_index = (region_bin_counts_[dd] - rnd[dd]) * bin_spacing_ + 1.0;
                        bin_indexes_[dd] = std::max(
                            std::min(static_cast<int>(random_bin_index), number_of_bins_), 1);

                        double bin_spacing_delta, random_point_in_bin;
                        std::size_t bidx = static_cast<std::size_t>(bin_indexes_[dd]);
                        if (bin_indexes_[dd] > 1) {
                            bin_spacing_delta = stratification_grid_[dd][bidx - 1] -
                                                stratification_grid_[dd][bidx - 2];
                            random_point_in_bin =
                                stratification_grid_[dd][bidx - 2] +
                                (random_bin_index - bin_indexes_[dd]) * bin_spacing_delta;
                        } else {
                            bin_spacing_delta = stratification_grid_[dd][bidx - 1];
                            random_point_in_bin = (random_bin_index - bin_indexes_[dd]) * bin_spacing_delta;
                        }

                        // Apply power transform if enabled (gamma > 1). Grid position is in
                        // [0, 1]; transform to focus on tails (identity if gamma == 1).
                        double p_uniform = random_point_in_bin;
                        double p_transformed = apply_power_transform(p_uniform);

                        // Scale to [Min, Max] range
                        random_point_[dd] = min_[dd] + p_transformed * dx_[dd];

                        // Weight includes Jacobian correction (1.0 if gamma == 1)
                        double jacobian = power_transform_jacobian(p_uniform);
                        weight *= bin_spacing_delta * normalized_bin_count_ * jacobian;
                    }

                    // Evaluate integrand function
                    double function_value = weight * function_(random_point_, weight);
                    function_evaluations_++;

                    function_value_sum += function_value;
                    double function_value_squared = function_value * function_value;
                    function_value_squared_sum += function_value_squared;

                    for (int d = 0; d < dimensions_; ++d) {
                        std::size_t dd = static_cast<std::size_t>(d);
                        std::size_t bidx = static_cast<std::size_t>(bin_indexes_[dd] - 1);
                        refined_bin_contributions_[bidx][dd] += function_value;
                        if (grid_refinement_flag_ >= 0) {
                            bin_contributions_[bidx][dd] += function_value_squared;
                        }
                    }
                }

                function_value_squared_sum = std::sqrt(function_value_squared_sum * samples_per_bin_group_);
                function_value_squared_sum =
                    (function_value_squared_sum - function_value_sum) *
                    (function_value_squared_sum + function_value_sum);
                if (function_value_squared_sum <= 0.0) function_value_squared_sum = kTinyValue;

                total_integral_estimate += function_value_sum;
                total_error_estimate += function_value_squared_sum;

                if (grid_refinement_flag_ < 0) {
                    for (int d = 0; d < dimensions_; ++d) {
                        std::size_t dd = static_cast<std::size_t>(d);
                        bin_contributions_[static_cast<std::size_t>(bin_indexes_[dd] - 1)][dd] +=
                            function_value_squared_sum;
                    }
                }
                // Increment stratification indexes
                for (rk = dimensions_ - 1; rk >= 0; --rk) {
                    std::size_t rr = static_cast<std::size_t>(rk);
                    region_bin_counts_[rr] %= stratification_levels_;
                    if (++region_bin_counts_[rr] != 1) break;
                }
                if (rk < 0) break;
            }

            // Compute final results for this iteration
            total_error_estimate *= bin_volume_;
            double weight = 1.0 / total_error_estimate;
            sum_integration_weights_ += weight;
            sum_weighted_results_ += weight * total_integral_estimate;
            sum_chi_squared_ += weight * total_integral_estimate * total_integral_estimate;

            result_ = sum_weighted_results_ / sum_integration_weights_;
            standard_error_ = std::sqrt(1.0 / sum_integration_weights_);
            total_error_estimate = std::sqrt(total_error_estimate);

            // Ensure chi-squared is non-negative
            chi_squared_ = (sum_chi_squared_ - sum_weighted_results_ * result_) / (it + 1);
            if (chi_squared_ < 0.0) chi_squared_ = 0.0;

            // Check convergence
            if ((check_convergence && iterations_ > 1 &&
                std::fabs(standard_error_ / result_) < relative_tolerance) ||
                function_evaluations_ >= max_function_evaluations) {
                break;
            }

            // Refine the grid. The refinement is damped to avoid rapid, destabilizing changes,
            // and compressed in range by the exponent `alpha`.
            for (int d = 0; d < dimensions_; ++d) {
                std::size_t dd = static_cast<std::size_t>(d);
                double lower_contribution = bin_contributions_[0][dd];
                double upper_contribution = bin_contributions_[1][dd];
                bin_contributions_[0][dd] = (lower_contribution + upper_contribution) / 2.0;
                dimension_contributions_[dd] = bin_contributions_[0][dd];
                for (int b = 2; b < current_bin_count_; ++b) {
                    std::size_t bb = static_cast<std::size_t>(b);
                    double region_contribution = lower_contribution + upper_contribution;
                    lower_contribution = upper_contribution;
                    upper_contribution = bin_contributions_[bb][dd];
                    bin_contributions_[bb - 1][dd] = (region_contribution + upper_contribution) / 3.0;
                    dimension_contributions_[dd] += bin_contributions_[bb - 1][dd];
                }

                std::size_t last = static_cast<std::size_t>(current_bin_count_ - 1);
                bin_contributions_[last][dd] = (lower_contribution + upper_contribution) / 2.0;
                dimension_contributions_[dd] += bin_contributions_[last][dd];
            }
            for (int d = 0; d < dimensions_; ++d) {
                std::size_t dd = static_cast<std::size_t>(d);
                double region_contribution = 0.0;
                for (int b = 0; b < current_bin_count_; ++b) {
                    std::size_t bb = static_cast<std::size_t>(b);
                    if (bin_contributions_[bb][dd] < kTinyValue) bin_contributions_[bb][dd] = kTinyValue;
                    bin_densities_[bb] = std::pow(
                        (1.0 - bin_contributions_[bb][dd] / dimension_contributions_[dd]) /
                            (std::log(dimension_contributions_[dd]) - std::log(bin_contributions_[bb][dd])),
                        alpha);
                    region_contribution += bin_densities_[bb];
                }
                refine_grid(region_contribution / normalized_bin_count_, current_bin_count_,
                           bin_densities_, bin_edges_, stratification_grid_, d);
            }
        }
    }

    // Refines the grid by redistributing bin boundaries based on the contribution densities.
    // `total_density` is the total density to distribute across bins, `num_bins` the number of
    // bins in dimension `dimension_index`, `bin_densities` the density values per bin;
    // `new_bin_edges` receives the adjusted bin edges (scratch, reused across calls) before they
    // are written back into `grid`.
    void refine_grid(double total_density, int num_bins, std::vector<double>& bin_densities,
                     std::vector<double>& new_bin_edges, std::vector<std::vector<double>>& grid,
                     int dimension_index) {
        std::size_t dim = static_cast<std::size_t>(dimension_index);
        int current_bin = 0;
        double cumulative_density = 0.0, upper_edge = 0.0, lower_edge = 0.0;
        for (int i = 0; i < num_bins - 1; ++i) {
            // Accumulate densities until it exceeds the target for the current boundary
            while (total_density > cumulative_density) {
                cumulative_density += bin_densities[static_cast<std::size_t>(++current_bin - 1)];
            }

            // Compute the new bin edge
            if (current_bin > 1) {
                lower_edge = grid[dim][static_cast<std::size_t>(current_bin - 2)];
            }
            upper_edge = grid[dim][static_cast<std::size_t>(current_bin - 1)];
            cumulative_density -= total_density;

            // Calculate the adjusted bin edge based on cumulative density
            new_bin_edges[static_cast<std::size_t>(i)] =
                upper_edge - (upper_edge - lower_edge) * cumulative_density /
                                 bin_densities[static_cast<std::size_t>(current_bin - 1)];
        }
        // Update the stratification grid
        for (int i = 0; i < num_bins - 1; ++i) {
            grid[dim][static_cast<std::size_t>(i)] = new_bin_edges[static_cast<std::size_t>(i)];
        }
        grid[dim][static_cast<std::size_t>(num_bins - 1)] = 1.0;
    }
};

}  // namespace corehydro::numerics::math::integration
