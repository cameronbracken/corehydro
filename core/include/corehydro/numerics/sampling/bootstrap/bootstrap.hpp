// ported from: Numerics/Sampling/Bootstrap/Bootstrap.cs @ 2a0357a
//
// A general-purpose bootstrap class for parametric or non-parametric bootstrap analysis.
// This port covers BOTH workflows the C# class supports:
//   - the REGULAR bootstrap: Run(), RunDoubleBootstrap(), RunWithStudentizedBootstrap(),
//     GetConfidenceIntervals (Percentile / BiasCorrected / BCa / Normal / BootstrapT), the
//     cube-root default Transform/InverseTransform, and the leave-one-out jackknife / BCa
//     acceleration-constant machinery;
//   - the covariance-aware PIVOTAL bootstrap: the second (`BootstrapFit`) constructor,
//     `RunPivotalBootstrap`, `TransformPivotalBootstrap` (both overloads),
//     `GetRawPivotalConfidenceIntervals`, `TryCreatePivotalDraw`, `TryApplyInvalidPolicy`,
//     `LinkCovariance`, `CreatePivotalLinks`, every `Pivotal*` property, the shared state the
//     pivotal mode owns (`_originalCovariance`, `_rawBootstrap*`, `_pivotalLinks`,
//     `_pivotalDiagnostics`), and the `BootstrapRunType.Pivotal` enum member with its branches
//     in `GetConfidenceIntervals` and `ValidateConfidenceIntervalRequest`.
// The pivotal region was a documented omission in the earlier P3.10 port and is now ported in
// full; the two modes share result storage but validate their delegates and their supported
// confidence-interval methods separately, exactly as the C# source does.
//
// Pivotal link ownership: C# `ILinkFunction` is a reference type, so `_pivotalLinks` and the
// `LinkController` built from it hold the SAME link objects, and `PivotalLinks`'s
// `Array.Clone()` is a shallow copy. This port's `LinkController` OWNS its links through
// `unique_ptr`, so the factory contract here is a `std::vector<std::shared_ptr<ILinkFunction>>`
// (shared ownership reproduces C#'s reference sharing; a null element still means identity,
// preserving `LinkController`'s null-means-identity path exactly) and `detail::SharedLink`
// below bridges a shared link into the controller's owning storage. `pivotal_links()` returns
// a copy of the vector whose elements point at the same links -- the same shallow clone C#
// returns.
//
// Namespace note: see bootstrap_results.hpp's header -- C# `Bootstrap<TData>` lives in the
// flat `Numerics.Sampling` namespace despite sitting in a `Sampling/Bootstrap/` folder; this
// port keeps that flat namespace (`corehydro::numerics::sampling`) for parity.
//
// Delegate ports: C#'s `Func<...>` properties become `std::function` public data members
// (default-constructed/empty, i.e. falsy via `operator bool`, unless assigned -- mirrors a
// null C# delegate). `Transform`/`InverseTransform` default to the cube-root/cube pair
// (`x => Math.Pow(x, 1d / 3d)` / `x => Math.Pow(x, 3d)`), transcribed via `std::pow` (not
// `std::cbrt`, which -- unlike `Math.Pow` -- returns the real cube root for negative `x`;
// `std::pow(x, 1.0/3.0)` matches `Math.Pow`'s NaN-for-negative-base behavior exactly).
//
// Threading: `Parallel.For(0, Replicates, idx => {...})` becomes a plain serial `for` loop.
// This is PROVABLY order-independent: each replicate `idx` reads only `seeds[idx]` (an
// up-front array, computed once before the loop) and writes only to `_bootstrapParameterSets
// [idx]`/`_bootstrapStatistics[idx, *]`/`_validFlags[idx]` -- no replicate reads or mutates
// another replicate's slot, so serial vs. parallel execution order cannot change any output
// value. `Interlocked.Increment(ref _failedCount)` becomes a plain `++failed_count_` (safe
// under the same no-cross-replicate-aliasing argument).
//
// Threading, the PIVOTAL loop specifically (`RunPivotalBootstrap`'s own `Parallel.For`): for
// pure-function delegates with no side effects, serial and parallel execution produce
// identical results. For delegates that carry mutable state (e.g. a callback closing over a
// counter), C#'s `Parallel.For` order is nondeterministic while this port's serial loop is
// deterministic and fixed -- the port is STRICTLY MORE DETERMINISTIC, and no oracle depends
// on the parallel ordering. Replicate `index` reads: `seeds[index]` (an up-front array
// computed before the loop), `_originalData`, `parentFit.parameters()` (all immutable for the
// duration), plus the read-only members `max_retries`, `resample_function`, and
// `fit_with_covariance_function`. Each replicate writes only to `rawFits[index]` plus
// `Interlocked.Increment(ref failed)`, which is a COUNT and therefore order-independent by
// construction (integer addition, no floating-point reassociation). No replicate reads or
// mutates another's slot. The `rawFits.Where(f => f != null)` compaction that follows
// preserves index order in both cases, so the accepted-fit sequence handed to the transform is
// identical too. The transformation loop inside `TransformPivotalBootstrap` is a plain serial
// `for` in C# as well -- that matters, because it is the loop that consumes `jitterRng`,
// whose draw ORDER is observable in the output when `AddPivotalJitter` is on.
//
// Per-replicate seeding cascade (transcribed verbatim, load-bearing for cross-language
// reproducibility, and identical for `Run`/`RunDoubleBootstrap`/`RunWithStudentizedBootstrap`/
// `RunPivotalBootstrap`): ONE master `MersenneTwister(PRNGSeed)` draws `Replicates` seeds up
// front via `NextIntegers` (`utilities::next_integers`, in order); replicate `idx`'s FIRST
// attempt seeds a fresh `MersenneTwister` from `seeds[idx]`; a failed attempt's retry `r`
// (1-based count) reseeds from `seeds[idx] + 10 * r`. The `+` is computed in `uint32_t`
// (matching C#'s unchecked `int` addition, which wraps mod 2^32 in two's-complement -- the
// identical bit pattern to unsigned wraparound) to avoid signed-integer-overflow UB while
// staying bit-exact with the C# stream for every seed value, including near-`int.MaxValue`
// ones. The pivotal transform then opens a SECOND, independent generator,
// `new MersenneTwister(PRNGSeed)` (`jitterRng`), seeded from the same `PRNGSeed` but never
// advanced by the resampling loop -- it is consumed only when `AddPivotalJitter` is true, one
// `NextDouble()` per pivotal vector component, in accepted-fit order.
//
// BCa HAZARD (see `compute_acceleration_constants`'s own comment): C#'s
// `ComputeAccelerationConstants` uses `Tools.ParallelAdd` inside its own `Parallel.For` --  an
// order-DEPENDENT floating-point reduction that is NOT bit-reproducible even run-to-run in
// the real C# library (confirmed by running the oracle emitter twice and diffing `--dump`
// output). This port replaces it with a plain serial accumulation in jackknife-index order --
// deterministic within this port, but not guaranteed to match either C# run bit-for-bit. The
// BCa fixture case therefore uses a LOOSE tolerance sized from the measured C# run-to-run
// wobble (see fixtures/README.md's bootstrap schema).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/functions/i_link_function.hpp"
#include "corehydro/numerics/functions/link_controller.hpp"
#include "corehydro/numerics/math/linalg/cholesky_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/matrix_regularization.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/bootstrap/bootstrap_results.hpp"
#include "corehydro/numerics/sampling/bootstrap/support/bootstrap_fit.hpp"
#include "corehydro/numerics/sampling/bootstrap/support/pivotal_bootstrap_context.hpp"
#include "corehydro/numerics/sampling/bootstrap/support/pivotal_bootstrap_diagnostics.hpp"
#include "corehydro/numerics/sampling/bootstrap/support/pivotal_bootstrap_invalid_draw_policy.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::sampling {

namespace opt = corehydro::numerics::math::optimization;

namespace detail {

// Non-owning forwarder that lets a `shared_ptr`-held link be handed to `LinkController`,
// which owns its links through `unique_ptr` -- see the file header's "Pivotal link ownership"
// note. Behaviorally transparent: every call forwards to the shared link.
class SharedLink final : public corehydro::numerics::functions::ILinkFunction {
   public:
    explicit SharedLink(std::shared_ptr<corehydro::numerics::functions::ILinkFunction> inner)
        : inner_(std::move(inner)) {}
    double link(double x) const override { return inner_->link(x); }
    double inverse_link(double eta) const override { return inner_->inverse_link(eta); }
    double d_link(double x) const override { return inner_->d_link(x); }

   private:
    std::shared_ptr<corehydro::numerics::functions::ILinkFunction> inner_;
};

}  // namespace detail

template <typename TData>
class Bootstrap {
   public:
    // --- Delegates (C# `Func<...>` properties) ---------------------------------------------
    using ResampleFn = std::function<TData(const TData&, const opt::ParameterSet&, MersenneTwister&)>;
    using FitFn = std::function<opt::ParameterSet(const TData&)>;
    using StatisticFn = std::function<std::vector<double>(const opt::ParameterSet&)>;
    using JackknifeFn = std::function<TData(const TData&, int)>;
    using SampleSizeFn = std::function<int(const TData&)>;
    using TransformFn = std::function<double(double)>;
    using FitWithCovarianceFn = std::function<BootstrapFit(const TData&)>;
    // C# `ILinkFunction?[]` -- one optional link per fitted parameter; see the file header's
    // "Pivotal link ownership" note for why the element type is `shared_ptr` here.
    using LinkArray = std::vector<std::shared_ptr<corehydro::numerics::functions::ILinkFunction>>;
    using PivotalLinkFactoryFn = std::function<LinkArray(const PivotalBootstrapContext&)>;
    using PivotalReplicateFilterFn = std::function<bool(const BootstrapFit&)>;
    using PivotalParameterValidatorFn = std::function<bool(const std::vector<double>&)>;

    // --- Construction ------------------------------------------------------------------------

    // Constructs a new regular bootstrap analysis.
    Bootstrap(TData original_data, opt::ParameterSet original_parameters)
        : original_data_(std::move(original_data)), original_parameters_(std::move(original_parameters)) {}

    // Constructs a new covariance-aware bootstrap analysis. C#'s `ArgumentNullException` guard
    // on `originalFit` has no C++ equivalent (a `const BootstrapFit&` cannot be null), the same
    // rationale bootstrap_fit.hpp uses for its own omitted null guards.
    // (Member-initializer order follows DECLARATION order -- `original_covariance` is a public
    // property declared above the private fields -- not the C# assignment order.)
    Bootstrap(TData original_data, const BootstrapFit& original_fit)
        : original_covariance(original_fit.covariance().clone()),
          original_data_(std::move(original_data)),
          original_parameters_(original_fit.parameters().clone()) {}

    // --- Properties ----------------------------------------------------------------------------

    // Delegate for resampling the original data given the current parameters and PRNG.
    ResampleFn resample_function;
    // Delegate for fitting a model to data and returning a parameter set.
    FitFn fit_function;
    // Delegate for extracting statistics from a fitted parameter set.
    StatisticFn statistic_function;
    // Delegate for fitting a model to data and returning a parameter set plus covariance
    // matrix. Required by `run_pivotal_bootstrap()`.
    FitWithCovarianceFn fit_with_covariance_function;
    // Optional delegate for computing a leave-one-out jackknife sample. Required for BCa.
    JackknifeFn jackknife_function;
    // Optional delegate returning the number of observations in the data. Required for BCa.
    SampleSizeFn sample_size_function;

    // Optional transform applied to statistic values before Normal/BootstrapT CI computation.
    // Default cube-root (see file header for why `std::pow`, not `std::cbrt`).
    TransformFn transform = [](double x) { return std::pow(x, 1.0 / 3.0); };
    // Optional inverse transform corresponding to `transform`. Default cube.
    TransformFn inverse_transform = [](double x) { return std::pow(x, 3.0); };

    // Number of bootstrap replicates. Default = 10,000.
    int replicates = 10000;
    // PRNG seed for reproducibility. Default = 12345.
    int prng_seed = 12345;
    // Maximum number of retries for a failed bootstrap replicate. Default = 20.
    int max_retries = 20;
    // Number of inner bootstrap replicates for Bootstrap-t standard error estimation.
    // Default = 300.
    int inner_replicates = 300;

    // The original covariance matrix used by the pivotal bootstrap. C#'s property clones on
    // both get and set, because a C# `Matrix` is a reference type and a caller who kept the
    // matrix it assigned could otherwise mutate this object's copy. `std::optional<Matrix>`
    // already copies on assignment and on read, so a plain public member reproduces that with
    // no clone calls (unset == C# null, and `Matrix` has no default constructor, which is the
    // other reason for the optional). The one behavior C# has and this does not is protection
    // against a caller mutating THROUGH the member itself, which C++ value semantics cannot
    // express through a public field and no caller in this port does.
    std::optional<math::linalg::Matrix> original_covariance;

    // Factory returning one optional link function per fitted parameter for the pivotal
    // bootstrap. Null elements are treated as identity links by `LinkController`.
    PivotalLinkFactoryFn pivotal_link_factory;
    // Optional filter applied to valid raw covariance-aware fits before pivotal transformation.
    PivotalReplicateFilterFn pivotal_replicate_filter;
    // Optional validator applied to transformed pivotal parameter values.
    PivotalParameterValidatorFn pivotal_parameter_validator;
    // Policy used when a pivotal draw is invalid. Default = Drop.
    PivotalBootstrapInvalidDrawPolicy pivotal_invalid_draw_policy = PivotalBootstrapInvalidDrawPolicy::Drop;
    // Whether pivotal covariance matrices are regularized before Cholesky decomposition.
    bool regularize_pivotal_covariances = true;
    // Optional absolute component limit for the standardized pivotal vector (C# `double?`).
    std::optional<double> pivotal_z_limit;
    // Whether Gaussian jitter is added to pivotal vectors before the optional limit check.
    bool add_pivotal_jitter = false;
    // Base standard deviation of the optional Gaussian jitter. The applied scale is this value
    // divided by sqrt(parameter count).
    double pivotal_jitter_scale = 0.01;

    // The active bootstrapped model parameter sets. For the pivotal bootstrap, these are the
    // retained pivotal parameter draws.
    const std::vector<opt::ParameterSet>& bootstrap_parameter_sets() const { return bootstrap_parameter_sets_; }
    // The active bootstrapped statistics, [replicate][statistic].
    const std::vector<std::vector<double>>& bootstrap_statistics() const { return bootstrap_statistics_; }
    // The accepted raw covariance-aware fits from the most recent pivotal bootstrap run.
    const std::vector<BootstrapFit>& raw_bootstrap_fits() const { return raw_bootstrap_fits_; }
    // The accepted raw parameter sets from the most recent pivotal bootstrap run.
    const std::vector<opt::ParameterSet>& raw_bootstrap_parameter_sets() const {
        return raw_bootstrap_parameter_sets_;
    }
    // The accepted raw statistics from the most recent pivotal bootstrap run, when
    // `statistic_function` was supplied (C# `double[,]?`).
    const std::optional<std::vector<std::vector<double>>>& raw_bootstrap_statistics() const {
        return raw_bootstrap_statistics_;
    }
    // The pivotal link functions used by the most recent pivotal bootstrap run. Returned by
    // value: a shallow copy whose elements are the same links, exactly like C#'s `Array.Clone`.
    LinkArray pivotal_links() const { return pivotal_links_; }
    // Diagnostics from the most recent pivotal bootstrap run (C# `PivotalBootstrapDiagnostics?`).
    const std::optional<PivotalBootstrapDiagnostics>& pivotal_diagnostics() const { return pivotal_diagnostics_; }
    // The number of replicates that failed after all retries. For the pivotal bootstrap, this
    // is the raw covariance-aware fit failure count.
    int failed_replicates() const { return failed_count_; }

    // --- Run Methods -----------------------------------------------------------------------

    // Runs the regular bootstrap procedure with error handling and retry logic.
    void run() {
        validate_core_delegates();
        validate_replication_settings();
        initialize_state();
        reset_pivotal_state();

        MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        std::vector<int> seeds = utilities::next_integers(prng, replicates);

        for (int idx = 0; idx < replicates; ++idx) {
            bool succeeded = false;
            for (int retry = 0; retry < max_retries; ++retry) {
                try {
                    MersenneTwister rng(replicate_seed(seeds[static_cast<std::size_t>(idx)], retry));
                    TData sample = resample_function(original_data_, original_parameters_, rng);
                    opt::ParameterSet fit_result = fit_function(sample);
                    if (!has_expected_finite_parameter_values(fit_result, num_params_)) continue;
                    std::vector<double> stat = validate_statistics(statistic_function(fit_result), num_stats_);

                    bootstrap_parameter_sets_[static_cast<std::size_t>(idx)] = fit_result;
                    for (int k = 0; k < num_stats_; ++k)
                        bootstrap_statistics_[static_cast<std::size_t>(idx)][static_cast<std::size_t>(k)] = stat[static_cast<std::size_t>(k)];
                    valid_flags_[static_cast<std::size_t>(idx)] = true;
                    succeeded = true;
                } catch (...) {
                    // Retry failed regular bootstrap replicates.
                }
                if (succeeded) break;
            }
            if (!succeeded) mark_failed(idx);
        }

        run_type_ = BootstrapRunType::Regular;
    }

    // Runs the double bootstrap procedure with bias correction.
    void run_double_bootstrap(int inner_reps = 300) {
        validate_core_delegates();
        validate_replication_settings();
        if (inner_reps < 1)
            throw std::invalid_argument("The number of inner replicates must be positive.");

        initialize_state();
        reset_pivotal_state();

        MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        std::vector<int> seeds = utilities::next_integers(prng, replicates);

        for (int idx = 0; idx < replicates; ++idx) {
            bool succeeded = false;
            for (int retry = 0; retry < max_retries; ++retry) {
                try {
                    MersenneTwister rng(replicate_seed(seeds[static_cast<std::size_t>(idx)], retry));

                    TData outer_sample = resample_function(original_data_, original_parameters_, rng);
                    opt::ParameterSet outer_fit = fit_function(outer_sample);
                    if (!has_expected_finite_parameter_values(outer_fit, num_params_)) continue;
                    std::vector<double> outer_stat = validate_statistics(statistic_function(outer_fit), num_stats_);

                    int p = num_params_;
                    std::vector<double> params_inner_sum(static_cast<std::size_t>(p), 0.0);
                    std::vector<double> stats_inner_sum(static_cast<std::size_t>(num_stats_), 0.0);
                    int valid_inner = 0;

                    for (int k = 0; k < inner_reps; ++k) {
                        try {
                            TData inner_sample = resample_function(outer_sample, outer_fit, rng);
                            opt::ParameterSet inner_fit = fit_function(inner_sample);
                            if (!has_expected_finite_parameter_values(inner_fit, p)) continue;
                            std::vector<double> inner_stat = validate_statistics(statistic_function(inner_fit), num_stats_);

                            for (int i = 0; i < p; ++i)
                                params_inner_sum[static_cast<std::size_t>(i)] += inner_fit.values[static_cast<std::size_t>(i)];
                            for (int i = 0; i < num_stats_; ++i)
                                stats_inner_sum[static_cast<std::size_t>(i)] += inner_stat[static_cast<std::size_t>(i)];
                            ++valid_inner;
                        } catch (...) {
                            // Skip failed inner replicate.
                        }
                    }

                    if (valid_inner == 0) continue;

                    std::vector<double> bias_corrected_parms(static_cast<std::size_t>(p));
                    for (int i = 0; i < p; ++i) {
                        double inner_mean = params_inner_sum[static_cast<std::size_t>(i)] / valid_inner;
                        bias_corrected_parms[static_cast<std::size_t>(i)] =
                            outer_fit.values[static_cast<std::size_t>(i)] - (inner_mean - outer_fit.values[static_cast<std::size_t>(i)]);
                    }

                    std::vector<double> bias_corrected_stats(static_cast<std::size_t>(num_stats_));
                    for (int i = 0; i < num_stats_; ++i) {
                        double inner_mean = stats_inner_sum[static_cast<std::size_t>(i)] / valid_inner;
                        bias_corrected_stats[static_cast<std::size_t>(i)] =
                            outer_stat[static_cast<std::size_t>(i)] - (inner_mean - outer_stat[static_cast<std::size_t>(i)]);
                    }

                    bootstrap_parameter_sets_[static_cast<std::size_t>(idx)] =
                        opt::ParameterSet(bias_corrected_parms, outer_fit.fitness, outer_fit.weight);
                    for (int k = 0; k < num_stats_; ++k)
                        bootstrap_statistics_[static_cast<std::size_t>(idx)][static_cast<std::size_t>(k)] = bias_corrected_stats[static_cast<std::size_t>(k)];
                    valid_flags_[static_cast<std::size_t>(idx)] = true;
                    succeeded = true;
                } catch (...) {
                    // Retry failed outer replicate.
                }
                if (succeeded) break;
            }
            if (!succeeded) mark_failed(idx);
        }

        run_type_ = BootstrapRunType::DoubleBootstrap;
    }

    // Runs the regular bootstrap procedure with nested inner bootstrap for studentized
    // Bootstrap-t confidence intervals.
    void run_with_studentized_bootstrap() {
        validate_core_delegates();
        validate_replication_settings();
        if (inner_replicates < 1)
            throw std::invalid_argument("The number of inner replicates must be positive.");

        reset_pivotal_state();

        std::vector<double> original_stats = validate_statistics(statistic_function(original_parameters_));
        parent_statistics_ = original_stats;
        num_stats_ = static_cast<int>(original_stats.size());
        num_params_ = static_cast<int>(original_parameters_.values.size());

        std::vector<double> pop_transformed(static_cast<std::size_t>(num_stats_));
        for (int i = 0; i < num_stats_; ++i)
            pop_transformed[static_cast<std::size_t>(i)] = apply_transform(original_stats[static_cast<std::size_t>(i)]);

        bootstrap_parameter_sets_.assign(static_cast<std::size_t>(replicates), opt::ParameterSet{});
        bootstrap_statistics_.assign(static_cast<std::size_t>(replicates),
                                      std::vector<double>(static_cast<std::size_t>(num_stats_), 0.0));
        valid_flags_.assign(static_cast<std::size_t>(replicates), false);
        failed_count_ = 0;
        studentized_values_ =
            std::vector<std::vector<double>>(static_cast<std::size_t>(replicates), std::vector<double>(static_cast<std::size_t>(num_stats_), 0.0));
        transformed_statistics_ =
            std::vector<std::vector<double>>(static_cast<std::size_t>(replicates), std::vector<double>(static_cast<std::size_t>(num_stats_), 0.0));

        MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        std::vector<int> seeds = utilities::next_integers(prng, replicates);

        for (int idx = 0; idx < replicates; ++idx) {
            bool succeeded = false;
            for (int retry = 0; retry < max_retries; ++retry) {
                try {
                    MersenneTwister rng(replicate_seed(seeds[static_cast<std::size_t>(idx)], retry));
                    TData sample = resample_function(original_data_, original_parameters_, rng);
                    opt::ParameterSet outer_fit = fit_function(sample);
                    if (!has_expected_finite_parameter_values(outer_fit, num_params_)) continue;
                    std::vector<double> outer_stats = validate_statistics(statistic_function(outer_fit), num_stats_);

                    bootstrap_parameter_sets_[static_cast<std::size_t>(idx)] = outer_fit;
                    for (int k = 0; k < num_stats_; ++k)
                        bootstrap_statistics_[static_cast<std::size_t>(idx)][static_cast<std::size_t>(k)] = outer_stats[static_cast<std::size_t>(k)];

                    std::vector<double> outer_transformed(static_cast<std::size_t>(num_stats_));
                    for (int j = 0; j < num_stats_; ++j)
                        outer_transformed[static_cast<std::size_t>(j)] = apply_transform(outer_stats[static_cast<std::size_t>(j)]);

                    // The inner PRNG is seeded from `seeds[idx]` DIRECTLY -- NOT offset by
                    // `retry` -- matching C#'s `new MersenneTwister(seeds[idx])` exactly.
                    MersenneTwister inner_prng(static_cast<std::uint32_t>(seeds[static_cast<std::size_t>(idx)]));
                    std::vector<int> inner_seeds = utilities::next_integers(inner_prng, inner_replicates);
                    std::vector<std::vector<double>> inner_transformed(
                        static_cast<std::size_t>(inner_replicates), std::vector<double>(static_cast<std::size_t>(num_stats_)));
                    int valid_inner = 0;

                    for (int k = 0; k < inner_replicates; ++k) {
                        try {
                            MersenneTwister inner_rng(static_cast<std::uint32_t>(inner_seeds[static_cast<std::size_t>(k)]));
                            TData inner_sample = resample_function(sample, outer_fit, inner_rng);
                            opt::ParameterSet inner_fit = fit_function(inner_sample);
                            if (!has_expected_finite_parameter_values(inner_fit, num_params_)) {
                                for (int j = 0; j < num_stats_; ++j)
                                    inner_transformed[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] = kNaN;
                                continue;
                            }
                            std::vector<double> inner_stats = validate_statistics(statistic_function(inner_fit), num_stats_);

                            for (int j = 0; j < num_stats_; ++j)
                                inner_transformed[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] =
                                    apply_transform(inner_stats[static_cast<std::size_t>(j)]);
                            ++valid_inner;
                        } catch (...) {
                            for (int j = 0; j < num_stats_; ++j)
                                inner_transformed[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] = kNaN;
                        }
                    }

                    if (valid_inner < 2) continue;

                    for (int j = 0; j < num_stats_; ++j) {
                        std::vector<double> col = get_column(inner_transformed, j);
                        std::vector<double> valid_col;
                        valid_col.reserve(col.size());
                        for (double v : col)
                            if (corehydro::numerics::is_finite(v)) valid_col.push_back(v);
                        double se = valid_col.size() > 1 ? data::standard_deviation(valid_col) : kNaN;
                        (*transformed_statistics_)[static_cast<std::size_t>(idx)][static_cast<std::size_t>(j)] =
                            outer_transformed[static_cast<std::size_t>(j)];
                        (*studentized_values_)[static_cast<std::size_t>(idx)][static_cast<std::size_t>(j)] =
                            se > 0 ? (pop_transformed[static_cast<std::size_t>(j)] - outer_transformed[static_cast<std::size_t>(j)]) / se : kNaN;
                    }

                    valid_flags_[static_cast<std::size_t>(idx)] = true;
                    succeeded = true;
                } catch (...) {
                    // Retry failed outer replicate.
                }
                if (succeeded) break;
            }

            if (!succeeded) {
                mark_failed(idx);
                for (int j = 0; j < num_stats_; ++j) {
                    (*transformed_statistics_)[static_cast<std::size_t>(idx)][static_cast<std::size_t>(j)] = kNaN;
                    (*studentized_values_)[static_cast<std::size_t>(idx)][static_cast<std::size_t>(j)] = kNaN;
                }
            }
        }

        run_type_ = BootstrapRunType::Studentized;
    }

    // --- Pivotal Bootstrap ---------------------------------------------------------------------

    // Runs the covariance-aware pivotal bootstrap as a distinct bootstrap mode.
    void run_pivotal_bootstrap() {
        validate_pivotal_delegates();
        validate_replication_settings();

        BootstrapFit parent_fit = create_original_fit();

        // C#'s `Stopwatch` around the resampling loop is dropped: the two `TimeSpan`
        // diagnostics it fed are not ported (see pivotal_bootstrap_diagnostics.hpp's header --
        // wall-clock duration is not oracle-comparable across languages).
        MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        std::vector<int> seeds = utilities::next_integers(prng, replicates);
        std::vector<std::optional<BootstrapFit>> raw_fits(static_cast<std::size_t>(replicates));
        int failed = 0;

        for (int index = 0; index < replicates; ++index) {
            bool succeeded = false;
            for (int retry = 0; retry < max_retries; ++retry) {
                try {
                    MersenneTwister rng(replicate_seed(seeds[static_cast<std::size_t>(index)], retry));
                    TData sample = resample_function(original_data_, parent_fit.parameters(), rng);
                    BootstrapFit raw_fit = fit_with_covariance_function(sample);
                    if (!is_valid_fit(raw_fit, parent_fit.parameter_count())) continue;

                    raw_fits[static_cast<std::size_t>(index)] = raw_fit;
                    succeeded = true;
                } catch (...) {
                    // Retry failed pivotal bootstrap resamples or covariance-aware fits.
                }

                if (succeeded) break;
            }

            if (!succeeded) ++failed;  // plain increment replacing Interlocked.Increment
        }

        std::vector<BootstrapFit> accepted_raw_fits;
        accepted_raw_fits.reserve(raw_fits.size());
        for (const auto& f : raw_fits)
            if (f.has_value()) accepted_raw_fits.push_back(*f);

        transform_pivotal_bootstrap(accepted_raw_fits, replicates, failed);
    }

    // Transforms precomputed raw covariance-aware bootstrap fits into pivotal bootstrap draws.
    // C#'s `ArgumentNullException` guard on `rawFits` has no C++ equivalent (a `std::vector`
    // cannot be null).
    void transform_pivotal_bootstrap(const std::vector<BootstrapFit>& raw_fits) {
        transform_pivotal_bootstrap(raw_fits, static_cast<int>(raw_fits.size()), 0);
    }

   private:
    // Applies the pivotal transformation to raw fits and stores the resulting active bootstrap
    // ensemble. C#'s `resamplingTime` parameter is dropped along with the timing diagnostics.
    void transform_pivotal_bootstrap(const std::vector<BootstrapFit>& raw_fits, int requested_replicates,
                                      int failed_raw_replicates) {
        BootstrapFit parent_fit = create_original_fit();
        int p = parent_fit.parameter_count();

        std::vector<BootstrapFit> accepted_raw_fits;
        accepted_raw_fits.reserve(raw_fits.size());
        for (const auto& f : raw_fits)
            if (is_accepted_raw_fit(f, p)) accepted_raw_fits.push_back(f);

        if (accepted_raw_fits.empty())
            throw std::runtime_error("No valid raw bootstrap fits were supplied for pivotal transformation.");

        LinkArray links = create_pivotal_links(parent_fit, accepted_raw_fits);
        functions::LinkController link_controller = make_link_controller(links);
        std::vector<double> parent_eta = link_controller.link(parent_fit.parameters().values);
        validate_transformed_values(parent_eta, "The parent link transformation produced a non-finite value.");

        math::linalg::Matrix parent_link_covariance = link_covariance(parent_fit, link_controller);
        math::linalg::CholeskyDecomposition parent_cholesky(parent_link_covariance);
        std::vector<opt::ParameterSet> pivotal_parameter_sets;
        pivotal_parameter_sets.reserve(accepted_raw_fits.size());
        MersenneTwister jitter_rng(static_cast<std::uint32_t>(prng_seed));
        int invalid = 0;

        for (std::size_t i = 0; i < accepted_raw_fits.size(); ++i) {
            const BootstrapFit& raw_fit = accepted_raw_fits[i];
            opt::ParameterSet pivotal_parameters;
            if (try_create_pivotal_draw(raw_fit, parent_fit, link_controller, parent_eta, parent_cholesky,
                                        jitter_rng, pivotal_parameters)) {
                pivotal_parameter_sets.push_back(pivotal_parameters);
            } else {
                ++invalid;
                opt::ParameterSet fallback_parameters;
                if (try_apply_invalid_policy(raw_fit, parent_fit, fallback_parameters))
                    pivotal_parameter_sets.push_back(fallback_parameters);
            }
        }

        raw_bootstrap_fits_ = accepted_raw_fits;
        raw_bootstrap_parameter_sets_.clear();
        raw_bootstrap_parameter_sets_.reserve(accepted_raw_fits.size());
        for (const auto& f : accepted_raw_fits) raw_bootstrap_parameter_sets_.push_back(f.parameters().clone());
        bootstrap_parameter_sets_ = pivotal_parameter_sets;
        num_params_ = p;
        failed_count_ = failed_raw_replicates;
        valid_flags_.assign(bootstrap_parameter_sets_.size(), true);
        studentized_values_.reset();
        transformed_statistics_.reset();
        pivotal_links_ = links;

        PivotalBootstrapDiagnostics diagnostics;
        diagnostics.requested_replicates = requested_replicates;
        diagnostics.failed_raw_replicates = failed_raw_replicates;
        diagnostics.rejected_raw_replicates = static_cast<int>(raw_fits.size() - accepted_raw_fits.size());
        diagnostics.accepted_raw_replicates = static_cast<int>(accepted_raw_fits.size());
        diagnostics.invalid_pivotal_replicates = invalid;
        diagnostics.retained_pivotal_replicates = static_cast<int>(bootstrap_parameter_sets_.size());
        pivotal_diagnostics_ = diagnostics;

        if (statistic_function) {
            parent_statistics_ = validate_statistics(statistic_function(parent_fit.parameters()));
            num_stats_ = static_cast<int>(parent_statistics_->size());
            raw_bootstrap_statistics_ = compute_statistics(raw_bootstrap_parameter_sets_, statistic_function);
            bootstrap_statistics_ = compute_statistics(bootstrap_parameter_sets_, statistic_function);
        } else {
            parent_statistics_ = std::vector<double>{};
            num_stats_ = 0;
            raw_bootstrap_statistics_ =
                std::vector<std::vector<double>>(raw_bootstrap_parameter_sets_.size(), std::vector<double>{});
            bootstrap_statistics_ =
                std::vector<std::vector<double>>(bootstrap_parameter_sets_.size(), std::vector<double>{});
        }

        run_type_ = BootstrapRunType::Pivotal;
    }

    // Attempts to create one pivotal draw from an accepted raw fit. C# assigns
    // `pivotalParameters = default` up front; here the out parameter is simply left untouched
    // when the method returns false, and no caller reads it in that case. `parent_fit` is
    // unused -- the C# method takes it and never reads it either (the parent enters through
    // `parentEta`/`parentCholesky`); the parameter is kept for signature parity.
    bool try_create_pivotal_draw(const BootstrapFit& raw_fit, const BootstrapFit& parent_fit,
                                  const functions::LinkController& link_controller,
                                  const std::vector<double>& parent_eta,
                                  const math::linalg::CholeskyDecomposition& parent_cholesky,
                                  MersenneTwister& jitter_rng, opt::ParameterSet& pivotal_parameters) const {
        (void)parent_fit;
        try {
            std::vector<double> raw_eta = link_controller.link(raw_fit.parameters().values);
            validate_transformed_values(raw_eta, "The raw link transformation produced a non-finite value.");

            math::linalg::Matrix raw_link_covariance = link_covariance(raw_fit, link_controller);
            math::linalg::CholeskyDecomposition raw_cholesky(raw_link_covariance);
            std::vector<double> difference(parent_eta.size());
            for (std::size_t j = 0; j < difference.size(); ++j) difference[j] = parent_eta[j] - raw_eta[j];

            std::vector<double> z = raw_cholesky.forward(math::linalg::Vector(difference)).to_array();
            if (!contains_only_finite_values(z)) return false;

            if (add_pivotal_jitter && pivotal_jitter_scale > 0.0) {
                double jitter_scale =
                    pivotal_jitter_scale / std::sqrt(static_cast<double>(std::max<std::size_t>(1, z.size())));
                for (std::size_t j = 0; j < z.size(); ++j) {
                    // C# `Tools.Clamp` -- identical semantics to std::clamp.
                    double u = std::clamp(jitter_rng.next_double(), 1e-16, 1.0 - 1e-16);
                    z[j] += jitter_scale * distributions::Normal::standard_z(u);
                }
            }

            if (pivotal_z_limit.has_value()) {
                for (double value : z)
                    if (std::fabs(value) > *pivotal_z_limit) return false;
            }

            std::vector<double> reinflated = (parent_cholesky.l() * math::linalg::Vector(z)).to_array();
            std::vector<double> pivotal_eta(parent_eta.size());
            for (std::size_t j = 0; j < pivotal_eta.size(); ++j) pivotal_eta[j] = parent_eta[j] + reinflated[j];

            std::vector<double> pivotal_values = link_controller.inverse_link(pivotal_eta);
            if (!contains_only_finite_values(pivotal_values)) return false;
            if (pivotal_parameter_validator && !pivotal_parameter_validator(pivotal_values)) return false;

            pivotal_parameters =
                opt::ParameterSet(pivotal_values, raw_fit.parameters().fitness, raw_fit.parameters().weight);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Applies the configured invalid pivotal draw policy.
    bool try_apply_invalid_policy(const BootstrapFit& raw_fit, const BootstrapFit& parent_fit,
                                   opt::ParameterSet& parameter_set) const {
        switch (pivotal_invalid_draw_policy) {
            case PivotalBootstrapInvalidDrawPolicy::Drop:
                return false;
            case PivotalBootstrapInvalidDrawPolicy::UseRaw:
                parameter_set = raw_fit.parameters().clone();
                return true;
            case PivotalBootstrapInvalidDrawPolicy::UseParent:
                parameter_set = parent_fit.parameters().clone();
                return true;
        }
        throw std::invalid_argument("Unknown invalid-draw policy.");
    }

    // Computes the link-space covariance for a bootstrap fit using the supplied link controller.
    math::linalg::Matrix link_covariance(const BootstrapFit& fit,
                                          const functions::LinkController& link_controller) const {
        math::linalg::Matrix jacobian = link_controller.link_jacobian(fit.parameters().values);
        for (int i = 0; i < jacobian.number_of_rows(); ++i) {
            if (!corehydro::numerics::is_finite(jacobian(i, i)))
                throw std::invalid_argument("The link derivative produced a non-finite value.");
        }

        math::linalg::Matrix link_cov = jacobian * fit.covariance() * jacobian.transpose();
        return regularize_pivotal_covariances
                   ? math::linalg::MatrixRegularization::make_symmetric_positive_definite(link_cov)
                   : link_cov;
    }

    // Builds the pivotal link array from `pivotal_link_factory` or the identity default (an
    // all-null array of the right length). C#'s `links == null` guard has no C++ equivalent (a
    // `std::vector` cannot be null); the length check is transcribed. The return is a shallow
    // vector copy -- C#'s `links.Clone()`.
    LinkArray create_pivotal_links(const BootstrapFit& parent_fit,
                                    const std::vector<BootstrapFit>& accepted_raw_fits) const {
        LinkArray links = pivotal_link_factory
                              ? pivotal_link_factory(PivotalBootstrapContext(parent_fit, accepted_raw_fits))
                              : LinkArray(static_cast<std::size_t>(parent_fit.parameter_count()));

        if (static_cast<int>(links.size()) != parent_fit.parameter_count())
            throw std::invalid_argument("The pivotal link factory must return one link per parameter.");

        return links;
    }

    // Wraps a shared link array in a `LinkController` (see the file header's "Pivotal link
    // ownership" note). Null elements stay null so the controller's null-means-identity path is
    // reached, exactly as in C#.
    static functions::LinkController make_link_controller(const LinkArray& links) {
        std::vector<std::unique_ptr<functions::ILinkFunction>> owned;
        owned.reserve(links.size());
        for (const auto& link : links)
            owned.push_back(link ? std::unique_ptr<functions::ILinkFunction>(new detail::SharedLink(link))
                                 : nullptr);
        return functions::LinkController(std::move(owned));
    }

   public:
    // --- Confidence Intervals ----------------------------------------------------------------

    // Computes bootstrap confidence intervals using the specified method.
    BootstrapResults get_confidence_intervals(BootstrapCIMethod method, double alpha = 0.1) {
        validate_confidence_interval_request(method, alpha);

        if (run_type_ == BootstrapRunType::Pivotal) {
            return create_percentile_results(bootstrap_parameter_sets_, &bootstrap_statistics_,
                                             original_parameters_.values, parent_statistics_, alpha,
                                             failed_count_, method);
        }

        if (!statistic_function) throw std::runtime_error("StatisticFunction must be set.");
        std::vector<double> original_stats = validate_statistics(statistic_function(original_parameters_));

        std::optional<std::vector<double>> accel_constants;
        if (method == BootstrapCIMethod::BCa) accel_constants = compute_acceleration_constants(original_stats);

        BootstrapResults results;
        results.method = method;
        results.alpha = alpha;
        results.statistic_results.resize(static_cast<std::size_t>(num_stats_));
        results.parameter_results.resize(static_cast<std::size_t>(num_params_));
        results.failed_replicates = failed_count_;

        for (int i = 0; i < num_stats_; ++i) {
            std::vector<double> values = get_column(bootstrap_statistics_, i);
            switch (method) {
                case BootstrapCIMethod::Percentile:
                    results.statistic_results[static_cast<std::size_t>(i)] =
                        compute_percentile_ci(values, original_stats[static_cast<std::size_t>(i)], alpha);
                    break;
                case BootstrapCIMethod::BiasCorrected:
                    results.statistic_results[static_cast<std::size_t>(i)] =
                        compute_bias_corrected_ci(values, original_stats[static_cast<std::size_t>(i)], alpha);
                    break;
                case BootstrapCIMethod::BCa:
                    results.statistic_results[static_cast<std::size_t>(i)] = compute_bca_ci(
                        values, original_stats[static_cast<std::size_t>(i)], alpha, (*accel_constants)[static_cast<std::size_t>(i)]);
                    break;
                case BootstrapCIMethod::Normal:
                    results.statistic_results[static_cast<std::size_t>(i)] =
                        compute_normal_ci(values, original_stats[static_cast<std::size_t>(i)], alpha);
                    break;
                case BootstrapCIMethod::BootstrapT:
                    results.statistic_results[static_cast<std::size_t>(i)] =
                        compute_bootstrap_t_ci(i, original_stats[static_cast<std::size_t>(i)], alpha);
                    break;
            }
        }

        for (int i = 0; i < num_params_; ++i) {
            std::vector<double> values;
            values.reserve(bootstrap_parameter_sets_.size());
            for (const auto& ps : bootstrap_parameter_sets_) values.push_back(ps.values[static_cast<std::size_t>(i)]);
            results.parameter_results[static_cast<std::size_t>(i)] =
                compute_percentile_ci(values, original_parameters_.values[static_cast<std::size_t>(i)], alpha);
        }

        return results;
    }

    // Computes raw-bootstrap percentile confidence intervals after a pivotal bootstrap run.
    BootstrapResults get_raw_pivotal_confidence_intervals(double alpha = 0.1) {
        if (run_type_ != BootstrapRunType::Pivotal)
            throw std::runtime_error(
                "RunPivotalBootstrap() or TransformPivotalBootstrap() must be called before requesting raw "
                "pivotal-bootstrap confidence intervals.");
        if (alpha <= 0.0 || alpha >= 1.0) throw std::invalid_argument("Alpha must be between 0 and 1.");

        return create_percentile_results(
            raw_bootstrap_parameter_sets_,
            raw_bootstrap_statistics_.has_value() ? &*raw_bootstrap_statistics_ : nullptr,
            original_parameters_.values, parent_statistics_, alpha, failed_count_,
            BootstrapCIMethod::Percentile);
    }

   private:
    // Identifies the workflow that produced the active bootstrap results.
    enum class BootstrapRunType { None, Regular, DoubleBootstrap, Studentized, Pivotal };

    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    TData original_data_;
    opt::ParameterSet original_parameters_;
    std::vector<opt::ParameterSet> bootstrap_parameter_sets_;
    std::vector<std::vector<double>> bootstrap_statistics_;  // [replicate][statistic]
    std::vector<opt::ParameterSet> raw_bootstrap_parameter_sets_;
    std::optional<std::vector<std::vector<double>>> raw_bootstrap_statistics_;
    std::vector<BootstrapFit> raw_bootstrap_fits_;
    LinkArray pivotal_links_;
    std::optional<PivotalBootstrapDiagnostics> pivotal_diagnostics_;
    int num_stats_ = 0;
    int num_params_ = 0;
    int failed_count_ = 0;
    std::vector<bool> valid_flags_;
    std::optional<std::vector<std::vector<double>>> studentized_values_;
    std::optional<std::vector<std::vector<double>>> transformed_statistics_;
    std::optional<std::vector<double>> parent_statistics_;
    BootstrapRunType run_type_ = BootstrapRunType::None;

    // --- CI Methods --------------------------------------------------------------------------

    // Computes percentile confidence intervals for a single statistic or parameter.
    BootstrapStatisticResult compute_percentile_ci(const std::vector<double>& values, double population_estimate,
                                                    double alpha) const {
        std::vector<double> valid;
        valid.reserve(values.size());
        for (double v : values)
            if (corehydro::numerics::is_finite(v)) valid.push_back(v);
        std::sort(valid.begin(), valid.end());

        double lower_p = alpha / 2.0;
        double upper_p = 1.0 - alpha / 2.0;

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = !valid.empty() ? data::percentile(valid, lower_p, true) : kNaN;
        r.upper_ci = !valid.empty() ? data::percentile(valid, upper_p, true) : kNaN;
        r.valid_count = static_cast<int>(valid.size());
        r.total_count = static_cast<int>(values.size());
        r.standard_error = valid.size() > 1 ? data::standard_deviation(valid) : kNaN;
        r.mean = !valid.empty() ? data::mean(valid) : kNaN;
        return r;
    }

    // Computes bias-corrected confidence intervals for a single statistic.
    BootstrapStatisticResult compute_bias_corrected_ci(const std::vector<double>& values, double population_estimate,
                                                        double alpha) const {
        std::vector<double> valid;
        valid.reserve(values.size());
        for (double v : values)
            if (corehydro::numerics::is_finite(v)) valid.push_back(v);
        int valid_n = static_cast<int>(valid.size());
        if (valid_n == 0) return empty_result(population_estimate, static_cast<int>(values.size()));

        int count_leq = 0;
        for (int i = 0; i < valid_n; ++i)
            if (valid[static_cast<std::size_t>(i)] <= population_estimate) ++count_leq;
        double p0 = static_cast<double>(count_leq) / (valid_n + 1);

        std::sort(valid.begin(), valid.end());

        double z0 = distributions::Normal::standard_z(p0);
        double z_lower = distributions::Normal::standard_z(alpha / 2.0);
        double z_upper = distributions::Normal::standard_z(1.0 - alpha / 2.0);
        double bc_lower = distributions::Normal::standard_cdf(2.0 * z0 + z_lower);
        double bc_upper = distributions::Normal::standard_cdf(2.0 * z0 + z_upper);

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = data::percentile(valid, bc_lower, true);
        r.upper_ci = data::percentile(valid, bc_upper, true);
        r.valid_count = valid_n;
        r.total_count = static_cast<int>(values.size());
        r.standard_error = valid_n > 1 ? data::standard_deviation(valid) : kNaN;
        r.mean = data::mean(valid);
        return r;
    }

    // Computes bias-corrected and accelerated confidence intervals for a single statistic.
    BootstrapStatisticResult compute_bca_ci(const std::vector<double>& values, double population_estimate,
                                             double alpha, double acceleration) const {
        std::vector<double> valid;
        valid.reserve(values.size());
        for (double v : values)
            if (corehydro::numerics::is_finite(v)) valid.push_back(v);
        int valid_n = static_cast<int>(valid.size());
        if (valid_n == 0) return empty_result(population_estimate, static_cast<int>(values.size()));

        int count_leq = 0;
        for (int i = 0; i < valid_n; ++i)
            if (valid[static_cast<std::size_t>(i)] <= population_estimate) ++count_leq;
        double p0 = static_cast<double>(count_leq + 1) / (valid_n + 1);

        std::sort(valid.begin(), valid.end());

        double z0 = distributions::Normal::standard_z(p0);
        double z_lower = distributions::Normal::standard_z(alpha / 2.0);
        double z_upper = distributions::Normal::standard_z(1.0 - alpha / 2.0);

        double num_lower = z0 + z_lower;
        double den_lower = 1.0 - acceleration * num_lower;
        double bc_lower = distributions::Normal::standard_cdf(z0 + num_lower / den_lower);

        double num_upper = z0 + z_upper;
        double den_upper = 1.0 - acceleration * num_upper;
        double bc_upper = distributions::Normal::standard_cdf(z0 + num_upper / den_upper);

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = data::percentile(valid, bc_lower, true);
        r.upper_ci = data::percentile(valid, bc_upper, true);
        r.valid_count = valid_n;
        r.total_count = static_cast<int>(values.size());
        r.standard_error = valid_n > 1 ? data::standard_deviation(valid) : kNaN;
        r.mean = data::mean(valid);
        return r;
    }

    // Computes Normal confidence intervals for a single statistic using `transform`.
    BootstrapStatisticResult compute_normal_ci(const std::vector<double>& values, double population_estimate,
                                                double alpha) const {
        double pop_transformed = apply_transform(population_estimate);
        std::vector<double> valid_slice;
        valid_slice.reserve(values.size());
        for (double v : values)
            if (corehydro::numerics::is_finite(v)) valid_slice.push_back(apply_transform(v));

        if (valid_slice.size() < 2) return empty_result(population_estimate, static_cast<int>(values.size()));

        double se = data::standard_deviation(valid_slice);
        double z_lower = distributions::Normal::standard_z(alpha / 2.0);
        double z_upper = distributions::Normal::standard_z(1.0 - alpha / 2.0);

        double lower_transformed = pop_transformed + se * z_lower;
        double upper_transformed = pop_transformed + se * z_upper;

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = apply_inverse_transform(lower_transformed);
        r.upper_ci = apply_inverse_transform(upper_transformed);
        r.valid_count = static_cast<int>(valid_slice.size());
        r.total_count = static_cast<int>(values.size());
        r.standard_error = se;
        r.mean = data::mean(valid_slice);
        return r;
    }

    // Computes Bootstrap-t confidence intervals for a single statistic.
    BootstrapStatisticResult compute_bootstrap_t_ci(int statistic_index, double population_estimate,
                                                      double alpha) const {
        double pop_transformed = apply_transform(population_estimate);
        std::vector<double> x_col = get_column(*transformed_statistics_, statistic_index);
        std::vector<double> t_col = get_column(*studentized_values_, statistic_index);

        std::vector<double> valid_x, valid_t;
        valid_x.reserve(x_col.size());
        valid_t.reserve(t_col.size());
        for (double v : x_col)
            if (corehydro::numerics::is_finite(v)) valid_x.push_back(v);
        for (double v : t_col)
            if (corehydro::numerics::is_finite(v)) valid_t.push_back(v);

        if (valid_t.size() < 2) return empty_result(population_estimate, replicates);

        double se = data::standard_deviation(valid_x);
        std::sort(valid_t.begin(), valid_t.end());

        double t_lower = data::percentile(valid_t, alpha / 2.0, true);
        double t_upper = data::percentile(valid_t, 1.0 - alpha / 2.0, true);

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = apply_inverse_transform(pop_transformed + se * t_lower);
        r.upper_ci = apply_inverse_transform(pop_transformed + se * t_upper);
        r.valid_count = static_cast<int>(valid_t.size());
        r.total_count = replicates;
        r.standard_error = se;
        r.mean = data::mean(valid_x);
        return r;
    }

    // --- BCa Support ---------------------------------------------------------------------------

    // Computes acceleration constants for each statistic using leave-one-out jackknife
    // samples. See file header's BCa HAZARD note: this is a plain serial sum, NOT a
    // reproduction of C#'s order-dependent `Tools.ParallelAdd` reduction.
    std::vector<double> compute_acceleration_constants(const std::vector<double>& population_estimates) const {
        int n = sample_size_function(original_data_);
        std::vector<double> i2(static_cast<std::size_t>(num_stats_), 0.0);
        std::vector<double> i3(static_cast<std::size_t>(num_stats_), 0.0);
        std::vector<double> a(static_cast<std::size_t>(num_stats_), 0.0);

        for (int idx = 0; idx < n; ++idx) {
            try {
                TData jack_data = jackknife_function(original_data_, idx);
                opt::ParameterSet jack_fit = fit_function(jack_data);
                std::vector<double> jack_stats = statistic_function(jack_fit);

                for (int i = 0; i < num_stats_; ++i) {
                    double diff = population_estimates[static_cast<std::size_t>(i)] - jack_stats[static_cast<std::size_t>(i)];
                    i2[static_cast<std::size_t>(i)] += diff * diff;
                    i3[static_cast<std::size_t>(i)] += diff * diff * diff;
                }
            } catch (...) {
                // Skip failed jackknife samples.
            }
        }

        for (int i = 0; i < num_stats_; ++i)
            a[static_cast<std::size_t>(i)] = i3[static_cast<std::size_t>(i)] / (std::pow(i2[static_cast<std::size_t>(i)], 1.5) * 6.0);

        return a;
    }

    // --- Private Helpers ---------------------------------------------------------------------

    // Combines a replicate's up-front seed with its (0-based) retry count, in `uint32_t`
    // (see file header's seeding-cascade note).
    static std::uint32_t replicate_seed(int seed, int retry) {
        return static_cast<std::uint32_t>(seed) + static_cast<std::uint32_t>(10 * retry);
    }

    void validate_core_delegates() const {
        if (!resample_function) throw std::runtime_error("ResampleFunction must be set.");
        if (!fit_function) throw std::runtime_error("FitFunction must be set.");
        if (!statistic_function) throw std::runtime_error("StatisticFunction must be set.");
    }

    // Validates that the pivotal bootstrap delegates and parent covariance are set.
    void validate_pivotal_delegates() const {
        if (!resample_function)
            throw std::runtime_error("ResampleFunction must be set before running the pivotal bootstrap.");
        if (!fit_with_covariance_function)
            throw std::runtime_error(
                "FitWithCovarianceFunction must be set before running the pivotal bootstrap.");
        if (!original_covariance.has_value())
            throw std::runtime_error("OriginalCovariance must be set before running the pivotal bootstrap.");
    }

    void validate_replication_settings() const {
        if (replicates < 1) throw std::invalid_argument("The number of replicates must be positive.");
        if (max_retries < 1) throw std::invalid_argument("The maximum retry count must be positive.");
    }

    // Initializes regular-bootstrap state arrays before a `run()` or `run_double_bootstrap()`.
    void initialize_state() {
        std::vector<double> original_stats = validate_statistics(statistic_function(original_parameters_));
        parent_statistics_ = original_stats;
        num_stats_ = static_cast<int>(original_stats.size());
        num_params_ = static_cast<int>(original_parameters_.values.size());

        bootstrap_parameter_sets_.assign(static_cast<std::size_t>(replicates), opt::ParameterSet{});
        bootstrap_statistics_.assign(static_cast<std::size_t>(replicates),
                                      std::vector<double>(static_cast<std::size_t>(num_stats_), 0.0));
        valid_flags_.assign(static_cast<std::size_t>(replicates), false);
        failed_count_ = 0;
        studentized_values_.reset();
        transformed_statistics_.reset();
        run_type_ = BootstrapRunType::None;
    }

    // Clears stored pivotal-bootstrap state before a regular bootstrap run.
    void reset_pivotal_state() {
        raw_bootstrap_fits_.clear();
        raw_bootstrap_parameter_sets_.clear();
        raw_bootstrap_statistics_.reset();
        pivotal_links_.clear();
        pivotal_diagnostics_.reset();
    }

    // Marks a replicate as failed with NaN parameter and statistic values.
    void mark_failed(int idx) {
        std::vector<double> nan_params(static_cast<std::size_t>(num_params_), kNaN);
        bootstrap_parameter_sets_[static_cast<std::size_t>(idx)] = opt::ParameterSet(nan_params, kNaN);
        for (int k = 0; k < num_stats_; ++k) bootstrap_statistics_[static_cast<std::size_t>(idx)][static_cast<std::size_t>(k)] = kNaN;
        valid_flags_[static_cast<std::size_t>(idx)] = false;
        ++failed_count_;  // plain increment replacing Interlocked.Increment -- see file header
    }

    double apply_transform(double value) const { return transform ? transform(value) : value; }
    double apply_inverse_transform(double value) const { return inverse_transform ? inverse_transform(value) : value; }

    static BootstrapStatisticResult empty_result(double population_estimate, int total_count) {
        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = kNaN;
        r.upper_ci = kNaN;
        r.valid_count = 0;
        r.total_count = total_count;
        r.standard_error = kNaN;
        r.mean = kNaN;
        return r;
    }

    // Creates percentile results for parameter and statistic ensembles. C#'s `double[,]?
    // statistics` becomes a pointer (nullptr == C# null); `GetLength(1)` becomes the inner row
    // width of this port's [replicate][statistic] nesting, which is 0 for an empty ensemble
    // just as C#'s `new double[0, 0]` is.
    BootstrapResults create_percentile_results(const std::vector<opt::ParameterSet>& parameter_sets,
                                                const std::vector<std::vector<double>>* statistics,
                                                const std::vector<double>& parameter_estimates,
                                                const std::optional<std::vector<double>>& statistic_estimates,
                                                double alpha, int failed_reps,
                                                BootstrapCIMethod method) const {
        int statistic_count =
            (statistics == nullptr || statistics->empty()) ? 0 : static_cast<int>((*statistics)[0].size());
        BootstrapResults results;
        results.method = method;
        results.alpha = alpha;
        results.statistic_results.resize(static_cast<std::size_t>(statistic_count));
        results.parameter_results.resize(parameter_estimates.size());
        results.failed_replicates = failed_reps;

        for (int i = 0; i < statistic_count; ++i) {
            double estimate =
                statistic_estimates.has_value() && i < static_cast<int>(statistic_estimates->size())
                    ? (*statistic_estimates)[static_cast<std::size_t>(i)]
                    : kNaN;
            results.statistic_results[static_cast<std::size_t>(i)] =
                compute_percentile_ci(get_column(*statistics, i), estimate, alpha);
        }

        for (std::size_t i = 0; i < parameter_estimates.size(); ++i)
            results.parameter_results[i] =
                compute_percentile_ci(get_parameter_column(parameter_sets, static_cast<int>(i)),
                                      parameter_estimates[i], alpha);

        return results;
    }

    // Extracts one parameter column from an ensemble of parameter sets.
    static std::vector<double> get_parameter_column(const std::vector<opt::ParameterSet>& parameter_sets,
                                                      int parameter_index) {
        std::vector<double> values(parameter_sets.size());
        for (std::size_t i = 0; i < parameter_sets.size(); ++i)
            values[i] = parameter_sets[i].values[static_cast<std::size_t>(parameter_index)];
        return values;
    }

    // Computes statistic values for each parameter set.
    static std::vector<std::vector<double>> compute_statistics(
        const std::vector<opt::ParameterSet>& parameter_sets, const StatisticFn& statistic) {
        if (parameter_sets.empty()) return {};

        std::vector<double> first = validate_statistics(statistic(parameter_sets[0]));
        std::vector<std::vector<double>> values(parameter_sets.size(), std::vector<double>(first.size(), 0.0));
        for (std::size_t j = 0; j < first.size(); ++j) values[0][j] = first[j];

        for (std::size_t i = 1; i < parameter_sets.size(); ++i) {
            std::vector<double> row =
                validate_statistics(statistic(parameter_sets[i]), static_cast<int>(first.size()));
            for (std::size_t j = 0; j < first.size(); ++j) values[i][j] = row[j];
        }

        return values;
    }

    // Creates the original covariance-aware fit for the pivotal bootstrap.
    BootstrapFit create_original_fit() const {
        if (!original_covariance.has_value())
            throw std::runtime_error(
                "OriginalCovariance must be set before running or transforming a pivotal bootstrap.");

        BootstrapFit fit(original_parameters_, *original_covariance);
        if (!is_valid_fit(fit, std::nullopt))
            throw std::invalid_argument(
                "The original fit must contain finite parameters and a finite square covariance matrix.");

        return fit;
    }

    // Determines whether a raw covariance-aware fit should be accepted before the pivotal
    // transformation.
    bool is_accepted_raw_fit(const BootstrapFit& fit, int parameter_count) const {
        if (!is_valid_fit(fit, parameter_count)) return false;
        if (pivotal_replicate_filter && !pivotal_replicate_filter(fit)) return false;
        return true;
    }

    // Determines whether a covariance-aware fit has finite parameters and a finite square
    // covariance matrix. C#'s `fit == null` arm has no C++ equivalent (a `const BootstrapFit&`
    // cannot be null), and `BootstrapFit`'s own constructor already enforces the non-empty,
    // square, dimension-matching invariants; the finiteness checks are transcribed.
    static bool is_valid_fit(const BootstrapFit& fit, std::optional<int> parameter_count) {
        if (fit.parameters().values.empty()) return false;
        if (parameter_count.has_value() && static_cast<int>(fit.parameters().values.size()) != *parameter_count)
            return false;
        if (!contains_only_finite_values(fit.parameters().values)) return false;
        if (!fit.covariance().is_square() ||
            fit.covariance().number_of_rows() != static_cast<int>(fit.parameters().values.size()))
            return false;

        for (int i = 0; i < fit.covariance().number_of_rows(); ++i) {
            for (int j = 0; j < fit.covariance().number_of_columns(); ++j) {
                if (!corehydro::numerics::is_finite(fit.covariance()(i, j))) return false;
            }
        }

        return true;
    }

    // Validates that transformed values are finite.
    static void validate_transformed_values(const std::vector<double>& values, const char* message) {
        if (!contains_only_finite_values(values)) throw std::invalid_argument(message);
    }

    // Extracts column `col` from a [replicate][stat]-indexed 2D vector (this port's
    // representation of C#'s `double[,]`; mirrors `ExtensionMethods.GetColumn`, which is not
    // otherwise ported -- see extension_methods.hpp's own header note).
    static std::vector<double> get_column(const std::vector<std::vector<double>>& matrix, int col) {
        std::vector<double> values(matrix.size());
        for (std::size_t i = 0; i < matrix.size(); ++i) values[i] = matrix[i][static_cast<std::size_t>(col)];
        return values;
    }

    // Validates statistic values returned by `statistic_function`. The C# overload's null
    // check has no C++ equivalent (a `std::vector` cannot be null); every other check is
    // transcribed.
    static std::vector<double> validate_statistics(const std::vector<double>& statistics,
                                                     std::optional<int> expected_length = std::nullopt) {
        if (statistics.empty())
            throw std::runtime_error("The statistic function must return at least one statistic.");
        if (expected_length.has_value() && static_cast<int>(statistics.size()) != *expected_length)
            throw std::runtime_error("The statistic function must return the same number of statistics for every draw.");
        if (!contains_only_finite_values(statistics))
            throw std::runtime_error("The statistic function returned a non-finite value.");
        return statistics;
    }

    static bool has_expected_finite_parameter_values(const opt::ParameterSet& parameters, int expected_length) {
        return static_cast<int>(parameters.values.size()) == expected_length &&
               contains_only_finite_values(parameters.values);
    }

    static bool contains_only_finite_values(const std::vector<double>& values) {
        for (double v : values)
            if (!corehydro::numerics::is_finite(v)) return false;
        return true;
    }

    // Validates a confidence interval request against the last bootstrap run mode. C#'s
    // `_bootstrapStatistics == null` half of the first guard has no C++ equivalent (a
    // `std::vector` cannot be null; `run_type_ == None` already covers the never-run case).
    void validate_confidence_interval_request(BootstrapCIMethod method, double alpha) const {
        if (run_type_ == BootstrapRunType::None)
            throw std::runtime_error("A bootstrap run must be completed before requesting confidence intervals.");
        if (alpha <= 0.0 || alpha >= 1.0) throw std::invalid_argument("Alpha must be between 0 and 1.");

        if (run_type_ == BootstrapRunType::Pivotal) {
            if (method != BootstrapCIMethod::Percentile)
                throw std::runtime_error(
                    "Only percentile confidence intervals are supported after a pivotal bootstrap run.");
            return;
        }

        if (method == BootstrapCIMethod::BCa && (!jackknife_function || !sample_size_function))
            throw std::runtime_error("JackknifeFunction and SampleSizeFunction must be set for BCa method.");
        if (method == BootstrapCIMethod::BootstrapT && !studentized_values_.has_value())
            throw std::runtime_error(
                "RunWithStudentizedBootstrap() must be called before requesting Bootstrap-t confidence intervals.");
    }
};

}  // namespace corehydro::numerics::sampling
