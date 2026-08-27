// ported from: Numerics/Mathematics/Integration/MonteCarloIntegration.cs @ 2a0357a
//
// Plain Monte Carlo integration for multidimensional integrands: draw uniform points in the
// hyper-rectangle [min, max], evaluate the integrand at each, and report the running mean times
// the volume as the estimate, with StandardError shrinking as ~1/sqrt(Iterations).
//
// DEAD PROPERTY (C# source, not a port bug): `UseSobolSequence` is declared as a public settable
// property but `Integrate()` never reads it -- every sample is drawn from `Random.NextDouble()`
// regardless. This port keeps `use_sobol_sequence` for API parity with the C# class (and with its
// Miser/Vegas siblings, where the flag DOES matter) but does not consult it either.
//
// RNG: C#'s `public Random Random { get; set; }` is a settable property (default-constructed
// `MersenneTwister()`, clock-seeded). This port exposes the same shape as a public mutable
// `sampling::MersenneTwister random` member -- assign it directly (`mc.random =
// sampling::MersenneTwister(12345);`) to reproduce a seeded C# run bit-for-bit, since every draw
// in `integrate()` below goes through `random.next_double()` in the same order as C#'s
// `Random.NextDouble()` calls.
#pragma once
#include <cmath>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/integration/support/integration_status.hpp"
#include "corehydro/numerics/math/integration/support/integrator.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::math::integration {

/// Monte Carlo integration for multidimensional integrands.
class MonteCarloIntegration : public Integrator {
   public:
    /// Constructs a new Monte Carlo integration class. `function` is the multidimensional
    /// function to integrate, `dimensions` the number of dimensions it takes, `min`/`max` the
    /// per-dimension bounds the integral is computed under.
    MonteCarloIntegration(std::function<double(const std::vector<double>&)> function,
                          int dimensions, std::vector<double> min, std::vector<double> max) {
        if (dimensions < 1)
            throw std::out_of_range("There must be at least 1 dimension to evaluate.");
        if (static_cast<int>(min.size()) != dimensions || static_cast<int>(max.size()) != dimensions)
            throw std::out_of_range(
                "The minimum and maximum values must be the same length as the number of "
                "dimensions.");
        for (std::size_t i = 0; i < min.size(); ++i) {
            if (max[i] <= min[i])
                throw std::out_of_range(
                    "The maximum values cannot be less than or equal to the minimum values.");
        }
        if (!function) throw std::invalid_argument("The function cannot be null.");

        function_ = std::move(function);
        dimensions_ = dimensions;
        min_ = std::move(min);
        max_ = std::move(max);
    }

    /// The multidimensional function to integrate.
    const std::function<double(const std::vector<double>&)>& function() const { return function_; }

    /// The number of dimensions in the function to evaluate.
    int dimensions() const { return dimensions_; }

    /// The minimum values under which the integral must be computed.
    const std::vector<double>& min() const { return min_; }

    /// The maximum values under which the integral must be computed.
    const std::vector<double>& max() const { return max_; }

    /// The random number generator used within the Monte Carlo integration. Default-constructed
    /// (clock-seeded), matching C#'s `Random = new MersenneTwister()`; assign a seeded instance
    /// for a reproducible run.
    sampling::MersenneTwister random;

    /// The integration standard error.
    double standard_error() const { return standard_error_; }

    /// Determines whether to use a Sobol sequence or a pseudo-random number generator. See the
    /// file header: this flag is a documented dead property, unused by `integrate()`, kept only
    /// for parity with the C# class.
    bool use_sobol_sequence = true;

    /// Evaluates the integral.
    void integrate() override {
        clear_results();
        validate();

        try {
            std::vector<double> sample(static_cast<std::size_t>(dimensions_));

            double sum = 0.0, sum2 = 0.0;
            double avg = 0.0, avg2 = 0.0;

            double dx = 1.0;
            for (int i = 0; i < dimensions_; ++i) dx *= (max_[static_cast<std::size_t>(i)] -
                                                          min_[static_cast<std::size_t>(i)]);

            for (int iter = 1; iter <= max_iterations; ++iter) {
                for (int j = 0; j < dimensions_; ++j) {
                    std::size_t jj = static_cast<std::size_t>(j);
                    sample[jj] = min_[jj] + random.next_double() * (max_[jj] - min_[jj]);
                }

                double f = function_(sample);

                iterations_++;
                function_evaluations_++;
                sum += f;
                sum2 += f * f;

                avg = sum / iterations_;
                avg2 = sum2 / iterations_;

                result_ = avg * dx;
                standard_error_ = std::sqrt((avg2 - avg * avg) / iterations_) * dx;

                if (iterations_ > min_iterations &&
                    std::fabs(standard_error_ / result_) < relative_tolerance) {
                    break;
                }
            }

            if (function_evaluations_ >= max_function_evaluations) {
                status_ = IntegrationStatus::MaximumFunctionEvaluationsReached;
            } else {
                status_ = IntegrationStatus::Success;
            }
        } catch (...) {
            update_status(IntegrationStatus::Failure, std::current_exception());
        }
    }

   private:
    std::function<double(const std::vector<double>&)> function_;
    int dimensions_ = 0;
    std::vector<double> min_;
    std::vector<double> max_;
    double standard_error_ = 0.0;
};

}  // namespace corehydro::numerics::math::integration
