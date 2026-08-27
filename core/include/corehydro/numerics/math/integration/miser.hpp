// ported from: Numerics/Mathematics/Integration/Miser.cs @ 2a0357a
//
// Miser: the recursive stratified-sampling Monte Carlo algorithm (Press et al., "Numerical
// Recipes", Sec. 7.9). Bisects the integration region along the coordinate axis that minimizes
// combined sub-region variance, recurses down to a user-specified depth, and integrates each
// terminal sub-region with plain Monte Carlo. `miser()`/`ranpt()` below are direct,
// method-for-method transcriptions of the C# private helpers of the same name.
//
// SOBOL PATH (corehydro addition, no C# counterpart): C#'s `_sobol = new SobolSequence(Dimensions)`
// reads its direction numbers from an embedded resource; this port's `SobolSequence` instead takes
// a filesystem path (see sampling/sobol.hpp), required whenever `dimensions > 1`. The ctor below
// therefore takes an extra trailing `sobol_path` argument (default `""`, valid only when
// `dimensions == 1`) so `_sobol` -- unconditionally constructed here exactly as C# does -- has
// somewhere to read from. Path resolution is a wrapper concern in every other corehydro Sobol
// consumer (see numerics/support/toolbox/sampling.hpp's file header); this follows the same
// pattern.
//
// RNG: see monte_carlo_integration.hpp's file header for the `random` member's C#-parity note.
// Miser only reads `random` when `use_sobol_sequence` is false (see `ranpt()`).
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

/// Miser: recursive stratified-sampling Monte Carlo integration for multidimensional integrands.
class Miser : public Integrator {
   public:
    /// Constructs a new Miser class. `function` is the multidimensional function to integrate,
    /// `dimensions` the number of dimensions it takes, `min`/`max` the per-dimension bounds the
    /// integral is computed under. `sobol_path` is a corehydro addition -- see the file header.
    Miser(std::function<double(const std::vector<double>&)> function, int dimensions,
          std::vector<double> min, std::vector<double> max, const std::string& sobol_path = "")
        : dimensions_(require_dimensions(dimensions)), sobol_(dimensions_, sobol_path) {
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

    /// The random number generator used within the Monte Carlo integration. See
    /// monte_carlo_integration.hpp's file header.
    sampling::MersenneTwister random;

    /// The integration standard error.
    double standard_error() const { return standard_error_; }

    /// The fraction of remaining function evaluations used at each stage to explore the
    /// variance of the function. Default = 0.1.
    double fraction = 0.1;

    /// The minimum number of points and function evaluations performed in each subregion.
    /// Default = 15.
    int minimum_number_of_subregion_points = 15;

    /// A subregion is further bisected only if this number of function evaluations are
    /// available. Default = 4 * 15 = 60.
    int minimum_number_of_bisections = 60;

    /// Dither should normally be set to 0.0, but can be set to 0.1 if the integrand's active
    /// region falls on the boundary of a power-of-two subdivision of a region.
    double dither = 0.0;

    /// Determines whether to use a Sobol sequence or a pseudo-random number generator.
    bool use_sobol_sequence = true;

    /// Evaluates the integral.
    void integrate() override {
        clear_results();
        validate();

        try {
            std::vector<double> regn(static_cast<std::size_t>(2 * dimensions_));
            for (int i = 0; i < dimensions_; ++i) regn[static_cast<std::size_t>(i)] =
                min_[static_cast<std::size_t>(i)];
            for (int i = dimensions_; i < 2 * dimensions_; ++i)
                regn[static_cast<std::size_t>(i)] = max_[static_cast<std::size_t>(i - dimensions_)];

            double volume = 1.0;
            for (int i = 0; i < dimensions_; ++i)
                volume *= (max_[static_cast<std::size_t>(i)] - min_[static_cast<std::size_t>(i)]);

            double result = 0.0, error = 0.0;
            miser(regn, max_function_evaluations, dither, result, error);
            result_ = result * volume;
            standard_error_ = std::sqrt(error) * volume;
        } catch (...) {
            update_status(IntegrationStatus::Failure, std::current_exception());
        }
    }

   private:
    // Validates `dimensions >= 1` and returns it, so the check runs (and throws C#'s
    // ArgumentOutOfRangeException-mapped std::out_of_range) BEFORE `sobol_`'s member
    // initializer constructs a SobolSequence with a possibly-invalid dimension -- member
    // initializers run in declaration order, so this must be declared, and must run, ahead of
    // `sobol_` below.
    static int require_dimensions(int dimensions) {
        if (dimensions < 1)
            throw std::out_of_range("There must be at least 1 dimension to evaluate.");
        return dimensions;
    }

    std::function<double(const std::vector<double>&)> function_;
    int dimensions_ = 0;
    std::vector<double> min_;
    std::vector<double> max_;
    sampling::SobolSequence sobol_;
    double standard_error_ = 0.0;

    // Monte Carlo samples `function_` in the rectangular volume specified by
    // `regn[0..2*ndim-1]` (ndim "lower-left" coordinates followed by ndim "upper-right"
    // coordinates), sampled `npts` times via recursive stratified sampling. `ave` receives the
    // mean value of the function in the region; `var` receives an estimate of the statistical
    // uncertainty of `ave` (square of standard deviation). `dith` should normally be 0, but can
    // be set to (e.g.) 0.1 if the integrand's active region falls on the boundary of a
    // power-of-two subdivision of the region.
    void miser(const std::vector<double>& regn, int npts, double dith, double& ave, double& var) {
        const int MNPT = minimum_number_of_subregion_points;
        const int MNBS = minimum_number_of_bisections;
        const double PFAC = fraction;
        constexpr double TINY = 1.0e-30;
        constexpr double BIG = 1.0e30;

        // PFAC is the fraction of remaining function evaluations used at each stage to explore
        // the variance of the integrand. At least MNPT function evaluations are performed in any
        // terminal subregion; a subregion is further bisected only if at least MNBS function
        // evaluations are available. We take MNBS = 4 * MNPT.

        int iran = 0;
        int ndim = dimensions_;
        std::vector<double> pt(static_cast<std::size_t>(ndim));

        if (npts < MNBS) {
            // Too few points to bisect, do straight Monte Carlo
            double summ = 0.0, summ2 = 0.0;
            for (int n = 0; n < npts; ++n) {
                ranpt(pt, regn);
                double fval = function_(pt);
                function_evaluations_++;
                summ += fval;
                summ2 += fval * fval;
            }
            ave = summ / npts;
            var = std::max(TINY, (summ2 - summ * summ / npts) / (static_cast<double>(npts) * npts));
        } else {
            // Do the preliminary (uniform sampling)
            std::vector<double> rmid(static_cast<std::size_t>(ndim));
            int npre = std::max(static_cast<int>(npts * PFAC), MNPT);
            std::vector<double> fmaxl(static_cast<std::size_t>(ndim));
            std::vector<double> fmaxr(static_cast<std::size_t>(ndim));
            std::vector<double> fminl(static_cast<std::size_t>(ndim));
            std::vector<double> fminr(static_cast<std::size_t>(ndim));

            // Initialize the left and right bounds for each dimension
            for (int j = 0; j < ndim; ++j) {
                std::size_t jj = static_cast<std::size_t>(j);
                iran = (iran * 2661 + 36979) % 175000;
                double s = sign(dith, static_cast<double>(iran - 87500));
                rmid[jj] = (0.5 + s) * regn[jj] + (0.5 - s) * regn[static_cast<std::size_t>(ndim) + jj];
                fminl[jj] = fminr[jj] = BIG;
                fmaxl[jj] = fmaxr[jj] = -BIG;
            }
            // Loop over the points in the sample
            for (int n = 0; n < npre; ++n) {
                ranpt(pt, regn);
                double fval = function_(pt);
                function_evaluations_++;
                // Find the left and right bounds for each dimension
                for (int j = 0; j < ndim; ++j) {
                    std::size_t jj = static_cast<std::size_t>(j);
                    if (pt[jj] <= rmid[jj]) {
                        fminl[jj] = std::min(fminl[jj], fval);
                        fmaxl[jj] = std::max(fmaxl[jj], fval);
                    } else {
                        fminr[jj] = std::min(fminr[jj], fval);
                        fmaxr[jj] = std::max(fmaxr[jj], fval);
                    }
                }
            }
            // Choose which dimension jb to bisect
            double sumb = BIG;
            int jb = -1;
            double siglb = 1.0, sigrb = 1.0;
            for (int j = 0; j < ndim; ++j) {
                std::size_t jj = static_cast<std::size_t>(j);
                if (fmaxl[jj] > fminl[jj] && fmaxr[jj] > fminr[jj]) {
                    double sigl = std::max(TINY, std::pow(fmaxl[jj] - fminl[jj], 2.0 / 3.0));
                    double sigr = std::max(TINY, std::pow(fmaxr[jj] - fminr[jj], 2.0 / 3.0));
                    double sum = sigl + sigr;  // Equation (7.9.24), see text.
                    if (sum <= sumb) {
                        sumb = sum;
                        jb = j;
                        siglb = sigl;
                        sigrb = sigr;
                    }
                }
            }
            if (jb == -1) jb = (ndim * iran) / 175000;  // MNPT may be too small
            // Apportion the remaining points between left and right
            std::size_t jbz = static_cast<std::size_t>(jb);
            double rgl = regn[jbz];
            double rgm = rmid[jbz];
            double rgr = regn[static_cast<std::size_t>(ndim) + jbz];
            double fracl = std::fabs((rgm - rgl) / (rgr - rgl));
            int nptl = static_cast<int>(
                MNPT + (npts - npre - 2 * MNPT) * fracl * siglb /
                           (fracl * siglb + (1.0 - fracl) * sigrb));  // Equation (7.9.23)
            int nptr = npts - npre - nptl;

            // Now allocate and integrate the two sub-regions
            std::vector<double> regn_temp(static_cast<std::size_t>(2 * ndim));
            for (int j = 0; j < ndim; ++j) {
                std::size_t jj = static_cast<std::size_t>(j);
                regn_temp[jj] = regn[jj];
                regn_temp[static_cast<std::size_t>(ndim) + jj] = regn[static_cast<std::size_t>(ndim) + jj];
            }
            double avel = 0.0, varl = 0.0;
            regn_temp[static_cast<std::size_t>(ndim) + jbz] = rmid[jbz];
            miser(regn_temp, nptl, dith, avel, varl);
            regn_temp[jbz] = rmid[jbz];
            regn_temp[static_cast<std::size_t>(ndim) + jbz] = regn[static_cast<std::size_t>(ndim) + jbz];
            miser(regn_temp, nptr, dith, ave, var);
            ave = fracl * avel + (1 - fracl) * ave;
            var = fracl * fracl * varl + (1 - fracl) * (1 - fracl) * var;
            // Combine left and right regions by equation (7.9.11)
        }
    }

    // Returns a uniformly random point `pt` in an n-dimensional rectangular region, used by
    // `miser()`. `regn` is a vector consisting of `ndim` "lower-left" coordinates followed by
    // `ndim` "upper-right" coordinates.
    void ranpt(std::vector<double>& pt, const std::vector<double>& regn) {
        int n = static_cast<int>(pt.size());
        std::vector<double> rnd;
        bool have_rnd = use_sobol_sequence;
        if (have_rnd) rnd = sobol_.next_double();

        for (int j = 0; j < n; ++j) {
            std::size_t jj = static_cast<std::size_t>(j);
            double u = have_rnd ? rnd[jj] : random.next_double();
            pt[jj] = regn[jj] + (regn[static_cast<std::size_t>(n) + jj] - regn[jj]) * u;
        }
    }
};

}  // namespace corehydro::numerics::math::integration
