// ported from: Numerics/Data/Statistics/Probability.cs @ 7e8e8d1
//
// Selective port. `Probability` is a large static utility class (basic two-event rules,
// joint-probability dispatch, unions, exclusive-combination enumeration, and three
// different correlated-joint-probability engines: HPCM, the original Pandey PCM, and an
// MVN-CDF-backed path). This file ports ONLY the members that CompetingRisks.cs's
// CDF/PDF/CumulativeIncidenceFunctions paths actually call, traced call-by-call from the
// C# source rather than from this class's full public surface.
//
// IMPORTANT finding (overload resolution, not a judgment call): CompetingRisks.cs's
// correlated-mode CDF calls read
//     Probability.UnionPCM(cdf, _mvn.Covariance)
//     Probability.JointProbability(cdf, ind, _mvn.Covariance)
// The second call's 3rd argument is a `double[,]` (`_mvn.Covariance`), which only matches
// the overload `JointProbability(IList<double>, int[], double[,]? correlationMatrix = null,
// DependencyType dependency = DependencyType.CorrelationMatrix)` -- there is no
// `JointProbability(IList<double>, int[], MultivariateNormal)` overload in the C# source.
// With a non-null correlationMatrix and the DEFAULT dependency (CorrelationMatrix), that
// overload unconditionally dispatches to `JointProbabilityHPCM` ("Haden Smith's
// modification of Pandey's Product of Conditional Marginals" method) -- never to
// `JointProbabilityMVN`. `UnionPCM` reaches the same 3-arg overload internally for each
// inclusion-exclusion term. CompetingRisks constructs a `MultivariateNormal` purely to
// hold its `.Covariance` matrix (mu/sigma bookkeeping for PerfectlyNegative's synthetic
// rho, or to pass through a user CorrelationMatrix) -- that MVN instance's `.CDF()` is
// NEVER called anywhere in CompetingRisks.cs. Consequence: the correlated CDF/PDF paths
// this file supports are fully DETERMINISTIC for any number of components D -- ONLY
// `MultivariateNormal.BivariateCDF` (Drezner/Genz closed-form bivariate normal CDF, no
// RNG) is used, never the seeded Genz-Bretz MVNDST quasi-Monte-Carlo integrator that
// backs `MultivariateNormal.CDF()` for dimension >= 3 (see multivariate_normal.hpp's
// `mvnuni_` note and Task 6's carry-forward note in .superpowers/sdd/progress.md, which
// both concern a DIFFERENT code path that CompetingRisks does not reach). See
// docs/upstream-csharp-issues.md for a short write-up of this call-path finding.
//
// Ported (the narrow subset CompetingRisks.cs's call sites reach):
//   DependencyType; JointProbability (2-arg dependency dispatch, and the 4-arg
//   indicators+correlationMatrix overload); IndependentJointProbability (1-arg + 2-arg);
//   PositiveJointProbability (1-arg + 2-arg); NegativeJointProbability (1-arg + 2-arg);
//   JointProbabilityHPCM; Union (2-arg dependency dispatch, renamed `union_probability`
//   -- `union` is a C++ keyword); IndependentUnion; PositivelyDependentUnion;
//   NegativelyDependentUnion; UnionPCM (2-arg + 6-arg).
//
// v2.2.0 additionally ports the exclusive-enumeration and equicorrelated single-factor
// foundations added to the public upstream surface. Explicitly OMITTED (unreachable from
// CompetingRisks.cs and unchanged by the v2.2.0 scope unless noted elsewhere):
// caller needs them):
//   JointProbabilityMVN / JointProbabilitiesMVN / UnionMVN (the actual MVN-CDF-backed
//     joint-probability path -- see the finding above: CompetingRisks never calls these),
//   JointProbabilityPCM (Pandey's ORIGINAL, non-Haden-Smith-modified PCM -- HPCM is a
//     distinct, separately-implemented method in the C# source, not a generalization),
//   JointProbabilitiesPCM (array/Parallel.For variant of JointProbabilityPCM),
//   the `out List<...>` UnionPCM overload (diagnostic variant returning per-term detail),
//   the entire "Basic Probability Rules" region (AAndB/AOrB/ANotB/BNotA/AGivenB/BGivenA),
//   NegativelyDependentExclusive, ExclusiveMVN, CommonCauseAdjustment, and
//   MutuallyExclusiveAdjustment.
//
// Small Tools.cs helpers this file needs (Clamp; Sum/Product/Min/Max with an indicators
// overload) are reimplemented narrowly in the `detail` namespace below rather than
// pulling in a full Tools.cs port (mirrors this project's existing "port only what's
// called" precedent for that class -- see tools.hpp's header note). `useComplement`
// (false in every call site reached here) is omitted from the indicator-overload helpers.
// Every site below that transcribes a C# `Math.Min`/`Math.Max`/`Tools.Max` call goes
// through `detail::nan_min`/`detail::nan_max`/`detail::max_value` rather than plain
// `std::min`/`std::max`/`std::max_element`: the BCL versions propagate NaN (return NaN if
// any operand is NaN), while the bare STL algorithms do not (a comparison against NaN is
// always false, so e.g. `std::min(a, b)` silently returns `a` when `b` is NaN).
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/integration/adaptive_gauss_kronrod.hpp"
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data::probability {

namespace sf = corehydro::numerics::math::special;

// Same underlying type as MultivariateNormal::covariance()'s return type
// (corehydro::numerics::math::linalg::Matrix2D); aliased locally for a shorter, class-
// scoped name mirroring the C# `double[,]` parameters this file ports.
using Matrix2D = corehydro::numerics::math::linalg::Matrix2D;

// Mirrors Probability.DependencyType.
enum class DependencyType { Independent, PerfectlyPositive, PerfectlyNegative, CorrelationMatrix };

namespace detail {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Tools.Clamp narrow port.
inline double clamp(double x, double min_v, double max_v) {
    return x < min_v ? min_v : (x > max_v ? max_v : x);
}

// Math.Min(double, double) narrow port. The BCL's Math.Min propagates NaN (returns NaN if
// either argument is NaN); plain `std::min` does NOT (a comparison against NaN is always
// false, so `std::min(a, b)` degrades to "return a" when b is NaN). Used at every site that
// transcribes a C# `Math.Min` call so NaN inputs propagate identically to the C# source.
inline double nan_min(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return kNaN;
    return b < a ? b : a;
}

// Math.Max(double, double) narrow port; see nan_min above.
inline double nan_max(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return kNaN;
    return b > a ? b : a;
}

// Tools.Max(IList<double>) narrow port: returns NaN if any element is NaN. Plain
// `std::max_element` is NOT NaN-aware (it would silently return a position-dependent
// finite value unless NaN happens to be the very first element).
inline double max_value(const std::vector<double>& values) {
    double m = -std::numeric_limits<double>::infinity();
    for (double v : values) {
        if (std::isnan(v)) return kNaN;
        if (v > m) m = v;
    }
    return m;
}

// Tools.Sum(IList<double>, IList<int>, useComplement=false) narrow port.
inline double sum(const std::vector<double>& values, const std::vector<int>& indicators) {
    double s = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i)
        if (indicators[i] == 1) s += values[i];
    return s;
}

// Tools.Product(IList<double>, IList<int>, useComplement=false) narrow port. (Note: the
// 1-arg `Tools.Product(IList<double>)` overload is NOT ported here -- no member reached
// from CompetingRisks.cs calls it; `IndependentJointProbability(IList<double>)` below
// inlines its own product loop in the C# source rather than calling `Tools.Product`.)
inline double product(const std::vector<double>& values, const std::vector<int>& indicators) {
    double p = 1.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (indicators[i] == 1) {
            p *= values[i];
            if (p == 0.0) return 0.0;
        }
    }
    return p;
}

// Tools.Min(IList<double>) narrow port.
inline double min_value(const std::vector<double>& values) {
    double m = std::numeric_limits<double>::max();
    for (double v : values) {
        if (std::isnan(v)) return kNaN;
        if (v < m) m = v;
    }
    return m;
}

// Tools.Min(IList<double>, IList<int>, useComplement=false) narrow port.
inline double min_value(const std::vector<double>& values, const std::vector<int>& indicators) {
    double m = std::numeric_limits<double>::max();
    bool any = false;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (indicators[i] == 1) {
            if (std::isnan(values[i])) return kNaN;
            if (values[i] < m) m = values[i];
            any = true;
        }
    }
    return any ? m : kNaN;
}

}  // namespace detail

// --- Joint Probability (dispatch by DependencyType) ---

// Mirrors Probability.IndependentJointProbability(IList<double>).
inline double independent_joint_probability(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    double p = 1.0;
    for (double v : probabilities) {
        p *= v;
        if (p == 0.0) return 0.0;
    }
    return detail::clamp(p, 0.0, 1.0);
}

// Mirrors Probability.PositiveJointProbability(IList<double>).
inline double positive_joint_probability(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    return detail::clamp(detail::min_value(probabilities), 0.0, 1.0);
}

// Mirrors Probability.NegativeJointProbability(IList<double>).
inline double negative_joint_probability(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    double s = 0.0;
    for (double v : probabilities) s += v;
    return detail::nan_max(0.0, detail::nan_min(1.0, s) - 1.0);
}

// Mirrors Probability.JointProbability(IList<double>, DependencyType = Independent).
inline double joint_probability(const std::vector<double>& probabilities,
                                 DependencyType dependency = DependencyType::Independent) {
    if (dependency == DependencyType::Independent) return independent_joint_probability(probabilities);
    if (dependency == DependencyType::PerfectlyPositive) return positive_joint_probability(probabilities);
    if (dependency == DependencyType::PerfectlyNegative) return negative_joint_probability(probabilities);
    return detail::kNaN;
}

// --- Joint Probability with indicators + correlation matrix (correlated-mode CDF/PDF) ---

// Mirrors Probability.IndependentJointProbability(IList<double>, int[]).
inline double independent_joint_probability(const std::vector<double>& probabilities,
                                             const std::vector<int>& indicators) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (indicators.empty()) throw std::invalid_argument("indicators must have at least one row");
    if (probabilities.size() != indicators.size())
        throw std::invalid_argument("probabilities and indicators must have the same length");
    return detail::clamp(detail::product(probabilities, indicators), 0.0, 1.0);
}

// Mirrors Probability.PositiveJointProbability(IList<double>, int[]).
inline double positive_joint_probability(const std::vector<double>& probabilities,
                                          const std::vector<int>& indicators) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (indicators.empty()) throw std::invalid_argument("indicators must have at least one row");
    if (probabilities.size() != indicators.size())
        throw std::invalid_argument("probabilities and indicators must have the same length");
    return detail::clamp(detail::min_value(probabilities, indicators), 0.0, 1.0);
}

// Mirrors Probability.NegativeJointProbability(IList<double>, int[]).
inline double negative_joint_probability(const std::vector<double>& probabilities,
                                          const std::vector<int>& indicators) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (indicators.empty()) throw std::invalid_argument("indicators must have at least one row");
    if (probabilities.size() != indicators.size())
        throw std::invalid_argument("probabilities and indicators must have the same length");
    return detail::nan_max(0.0, detail::nan_min(1.0, detail::sum(probabilities, indicators)) - 1.0);
}

// Computes the joint probability of multiple events with dependency, using Haden Smith's
// modification of Pandey's method for the Product of Conditional Marginals (PCM).
// Mirrors Probability.JointProbabilityHPCM. v2.1.4 sync (Numerics 33dc1af): FIXED, not
// mirrored -- the "First cycle" block below used to leave its `cdf < 1e-300` underflow
// guard commented out (`//if (cdf < 1e-300) cdf = 1e-300;`) while the structurally
// identical "Remaining cycles" loop (only reached for n >= 3) kept the same guard active
// three lines apart -- an asymmetry that looked like an accidental omission (see
// docs/upstream-csharp-issues.md, marked RESOLVED). Upstream now applies the guard (named
// `minimumCdf` there) in both places; this port does too, via the shared kMinimumCdf
// constant below.
inline double joint_probability_hpcm(const std::vector<double>& probabilities,
                                      const std::vector<int>& indicators,
                                      const Matrix2D& correlation_matrix,
                                      std::vector<double>* conditional_probabilities = nullptr) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (indicators.empty()) throw std::invalid_argument("indicators must have at least one row");
    if (probabilities.size() != indicators.size())
        throw std::invalid_argument("probabilities and indicators must have the same length");
    int n = static_cast<int>(probabilities.size());
    if (static_cast<int>(correlation_matrix.size()) != n ||
        (n > 0 && static_cast<int>(correlation_matrix[0].size()) != n))
        throw std::invalid_argument(
            "correlation matrix must be square with dimensions equal to the length of probabilities");

    constexpr double zMin = -9.0, zMax = 9.0;
    constexpr double kMinimumCdf = 1e-300;
    Matrix2D R = correlation_matrix;  // mirrors `Array.Copy(correlationMatrix, R, ...)`

    for (int i = 0; i < n; ++i) {
        std::size_t ii = static_cast<std::size_t>(i);
        if (indicators[ii] == 0)
            R[ii][ii] = detail::clamp(distributions::Normal::standard_z(1.0), zMin, zMax);
        else
            R[ii][ii] = detail::clamp(distributions::Normal::standard_z(probabilities[ii]), zMin, zMax);
    }

    // First cycle
    double z1 = R[0][0];
    double pdf = distributions::Normal::standard_pdf(z1);
    double cdf = distributions::Normal::standard_cdf(z1);
    if (cdf < kMinimumCdf) cdf = kMinimumCdf;
    double A = pdf / cdf;
    double B = A * (z1 + A);
    for (int k = 1; k < n; ++k) {
        std::size_t kk = static_cast<std::size_t>(k);
        double z2 = R[kk][kk];
        double r12 = R[0][kk];
        r12 = std::fabs(r12) < 1e-3 ? 0.0 : r12;
        double p21 = distributions::MultivariateNormal::bivariate_cdf(-z1, -z2, r12) / cdf;
        p21 = detail::nan_max(0.0, detail::nan_min(1.0, p21));
        double z21 = detail::clamp(distributions::Normal::standard_z(p21), zMin, zMax);
        R[kk][0] = z21;
    }
    for (int ir = 1; ir < n - 1; ++ir) {
        std::size_t iir = static_cast<std::size_t>(ir);
        for (int ic = ir + 1; ic < n; ++ic) {
            std::size_t iic = static_cast<std::size_t>(ic);
            R[iir][iic] = (R[iir][iic] - R[0][iir] * R[0][iic] * B) /
                std::sqrt((1.0 - R[0][iir] * R[0][iir] * B) * (1.0 - R[0][iic] * R[0][iic] * B));
        }
    }

    // Remaining cycles (only reached when n >= 3)
    for (int j = 1; j < n - 1; ++j) {
        std::size_t jj = static_cast<std::size_t>(j);
        z1 = R[jj][jj - 1];
        pdf = distributions::Normal::standard_pdf(z1);
        cdf = distributions::Normal::standard_cdf(z1);
        if (cdf < kMinimumCdf) cdf = kMinimumCdf;
        A = pdf / cdf;
        B = A * (z1 + A);
        for (int k = j + 1; k < n; ++k) {
            std::size_t kk = static_cast<std::size_t>(k);
            double z2 = R[kk][jj - 1];
            double r12 = R[jj][kk];
            r12 = std::fabs(r12) < 1e-3 ? 0.0 : r12;
            double p21 = distributions::MultivariateNormal::bivariate_cdf(-z1, -z2, r12) / cdf;
            p21 = detail::nan_max(0.0, detail::nan_min(1.0, p21));
            double z21 = detail::clamp(distributions::Normal::standard_z(p21), zMin, zMax);
            R[kk][jj] = z21;
        }
        for (int ir = j + 1; ir < n - 1; ++ir) {
            std::size_t iir = static_cast<std::size_t>(ir);
            for (int ic = ir + 1; ic < n; ++ic) {
                std::size_t iic = static_cast<std::size_t>(ic);
                R[iir][iic] = (R[iir][iic] - R[jj][iir] * R[jj][iic] * B) /
                    std::sqrt((1.0 - R[jj][iir] * R[jj][iir] * B) * (1.0 - R[jj][iic] * R[jj][iic] * B));
            }
        }
    }

    double jp = std::log(distributions::Normal::standard_cdf(R[0][0]));
    if (conditional_probabilities != nullptr &&
        conditional_probabilities->size() == static_cast<std::size_t>(n))
        (*conditional_probabilities)[0] = distributions::Normal::standard_cdf(R[0][0]);
    for (int i = 1; i < n; ++i) {
        std::size_t ii = static_cast<std::size_t>(i);
        jp += std::log(distributions::Normal::standard_cdf(R[ii][ii - 1]));
        if (conditional_probabilities != nullptr &&
            conditional_probabilities->size() == static_cast<std::size_t>(n))
            (*conditional_probabilities)[ii] = distributions::Normal::standard_cdf(R[ii][ii - 1]);
    }
    jp = std::exp(jp);
    jp = std::min(1.0, std::max(0.0, jp));
    if (std::isnan(jp)) jp = 0.0;
    return jp;
}

// Mirrors Probability.JointProbability(IList<double>, int[], double[,]? = null,
// DependencyType = CorrelationMatrix). CompetingRisks.cs always calls this with a
// non-null correlationMatrix and the default (CorrelationMatrix) dependency, which
// dispatches to joint_probability_hpcm above -- see this file's header comment.
inline double joint_probability(const std::vector<double>& probabilities,
                                 const std::vector<int>& indicators,
                                 const Matrix2D* correlation_matrix = nullptr,
                                 DependencyType dependency = DependencyType::CorrelationMatrix) {
    if (dependency == DependencyType::CorrelationMatrix && correlation_matrix != nullptr)
        return joint_probability_hpcm(probabilities, indicators, *correlation_matrix);
    if (dependency == DependencyType::Independent)
        return independent_joint_probability(probabilities, indicators);
    if (dependency == DependencyType::PerfectlyPositive)
        return positive_joint_probability(probabilities, indicators);
    if (dependency == DependencyType::PerfectlyNegative)
        return negative_joint_probability(probabilities, indicators);
    return detail::kNaN;
}

// --- Probability of Union ---
// `Union` is renamed `union_probability` (`union` is a reserved C++ keyword).

// Mirrors Probability.IndependentUnion(IList<double>) (De Morgan's rule).
inline double independent_union(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (probabilities.size() == 1) return probabilities[0];
    double numerator = 1.0;
    for (double p : probabilities) {
        double q = 1.0 - p;
        if (q == 0.0) return 1.0;
        numerator *= q;
    }
    return 1.0 - numerator;
}

inline double independent_exclusive(const std::vector<double>& probabilities,
                                    const std::vector<int>& indicators) {
    if (probabilities.empty() || indicators.empty() || probabilities.size() != indicators.size()) {
        throw std::invalid_argument("probabilities and indicators must have the same nonzero length");
    }
    double result = 1.0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (std::isnan(probabilities[i])) return detail::kNaN;
        result *= indicators[i] == 1 ? probabilities[i] : 1.0 - probabilities[i];
    }
    return detail::clamp(result, 0.0, 1.0);
}

inline void validate_exclusive_metadata(const std::vector<double>& probabilities,
                                        const std::vector<int>& binomial_combinations,
                                        const std::vector<std::vector<int>>& indicators,
                                        double absolute_tolerance,
                                        double relative_tolerance) {
    if (probabilities.empty() || indicators.empty()) {
        throw std::invalid_argument("probabilities and indicators must be non-empty");
    }
    if (binomial_combinations.size() != probabilities.size()) {
        throw std::invalid_argument(
            "binomial metadata must contain one count for each subset size");
    }
    if (!corehydro::numerics::is_finite(absolute_tolerance) || absolute_tolerance < 0.0 ||
        !corehydro::numerics::is_finite(relative_tolerance) || relative_tolerance < 0.0) {
        throw std::out_of_range("tolerances must be finite and non-negative");
    }
    std::size_t row_count = 0;
    const int n = static_cast<int>(probabilities.size());
    for (int i = 0; i < n; ++i) {
        const double probability = probabilities[static_cast<std::size_t>(i)];
        if (!corehydro::numerics::is_finite(probability) || probability < 0.0 ||
            probability > 1.0) {
            throw std::out_of_range("probabilities must be finite and between zero and one");
        }
        const auto expected = static_cast<std::size_t>(
            sf::factorial::binomial_coefficient(n, i + 1));
        if (binomial_combinations[static_cast<std::size_t>(i)] !=
            static_cast<int>(expected)) {
            throw std::invalid_argument(
                "binomial metadata does not match the probability count");
        }
        row_count += expected;
    }
    if (row_count != indicators.size()) {
        throw std::invalid_argument(
            "indicator row count does not match the binomial metadata");
    }
    for (const auto& row : indicators) {
        if (row.size() != probabilities.size()) {
            throw std::invalid_argument(
                "probabilities and indicator rows must have the same length");
        }
    }
}

// Mirrors the caller-owned-output IndependentExclusive overload. std::vector capacity and
// nested rows are retained across calls when their dimensions already match.
inline bool independent_exclusive(
    const std::vector<double>& probabilities,
    const std::vector<int>& binomial_combinations,
    const std::vector<std::vector<int>>& indicators,
    std::vector<double>& event_probabilities,
    std::vector<std::vector<int>>& event_indicators,
    double absolute_tolerance = 1e-4, double relative_tolerance = 1e-4) {
    validate_exclusive_metadata(probabilities, binomial_combinations, indicators,
                                absolute_tolerance, relative_tolerance);
    const std::size_t n = probabilities.size();
    std::size_t used = 0;
    event_probabilities.clear();
    auto place_row = [&](std::size_t source) -> const std::vector<int>& {
        if (used >= event_indicators.size()) event_indicators.emplace_back(n);
        auto& row = event_indicators[used];
        if (row.size() != n) row.resize(n);
        row = indicators[source];
        ++used;
        return row;
    };

    double union_value = 0.0;
    double sign = 1.0;
    std::size_t block = 0;
    std::size_t boundary = static_cast<std::size_t>(binomial_combinations[0]);
    double inclusion = detail::kNaN;
    double exclusion = detail::kNaN;
    for (std::size_t i = 0; i < indicators.size(); ++i) {
        if (i == boundary) {
            if (block > 0) {
                if (sign == 1.0) inclusion = union_value;
                else if (sign == -1.0) exclusion = union_value;
            }
            const double difference = std::fabs(inclusion - exclusion);
            if (block > 0 && block < binomial_combinations.size() &&
                difference <= absolute_tolerance &&
                difference <= relative_tolerance * std::min(inclusion, exclusion)) {
                place_row(indicators.size() - 1);
                event_probabilities.push_back(detail::clamp(0.5 * difference, 0.0, 1.0));
                event_indicators.resize(used);
                return true;
            }
            sign *= -1.0;
            ++block;
            if (block < binomial_combinations.size()) {
                boundary += static_cast<std::size_t>(binomial_combinations[block]);
            }
        }
        const auto& row = place_row(i);
        event_probabilities.push_back(independent_exclusive(probabilities, row));
        union_value += sign *
            (i < probabilities.size() ? probabilities[i]
                                      : independent_joint_probability(probabilities, row));
    }
    event_indicators.resize(used);
    return false;
}

enum class ExclusiveEnumerationStatus { Complete, Converged, Capped };

struct ExclusiveEnumerationResult {
    ExclusiveEnumerationStatus status;
    std::vector<double> probabilities;
    std::vector<std::vector<int>> indicators;
};

inline ExclusiveEnumerationResult independent_exclusive_lazy(
    const std::vector<double>& probabilities, bool include_no_event_row = false,
    long long max_emitted_combinations = 0, double absolute_tolerance = 1e-4,
    double relative_tolerance = 1e-4) {
    if (probabilities.empty()) throw std::invalid_argument("probabilities must not be empty");
    if (!corehydro::numerics::is_finite(absolute_tolerance) || absolute_tolerance < 0.0 ||
        !corehydro::numerics::is_finite(relative_tolerance) || relative_tolerance < 0.0) {
        throw std::out_of_range("tolerances must be finite and non-negative");
    }
    for (double probability : probabilities) {
        if (!corehydro::numerics::is_finite(probability) || probability < 0.0 || probability > 1.0) {
            throw std::out_of_range("probabilities must be finite and between zero and one");
        }
    }
    ExclusiveEnumerationResult output{ExclusiveEnumerationStatus::Complete, {}, {}};
    const int n = static_cast<int>(probabilities.size());
    auto close = [&](double mass, ExclusiveEnumerationStatus status) {
        output.indicators.emplace_back(probabilities.size(), 1);
        output.probabilities.push_back(detail::clamp(mass, 0.0, 1.0));
        output.status = status;
        return output;
    };
    double no_event_mass = 1.0;
    for (double probability : probabilities) no_event_mass *= 1.0 - probability;
    const double total_output_mass = include_no_event_row ? 1.0 : 1.0 - no_event_mass;
    double emitted_mass = 0.0;
    if (include_no_event_row) {
        output.indicators.emplace_back(probabilities.size(), 0);
        output.probabilities.push_back(detail::clamp(no_event_mass, 0.0, 1.0));
        emitted_mass += no_event_mass;
    }
    double union_value = 0.0;
    double sign = 1.0;
    double inclusion = detail::kNaN;
    double exclusion = detail::kNaN;
    long long emitted = 0;
    for (int k = 1; k <= n; ++k) {
        if (k >= 2) {
            const int block = k - 2;
            if (block > 0) {
                if (sign == 1.0)
                    inclusion = union_value;
                else if (sign == -1.0)
                    exclusion = union_value;
            }
            const double difference = std::fabs(inclusion - exclusion);
            if (block > 0 && block < n && difference <= absolute_tolerance &&
                difference <= relative_tolerance * std::min(inclusion, exclusion)) {
                return close(0.5 * difference, ExclusiveEnumerationStatus::Converged);
            }
            sign *= -1.0;
        }
        std::vector<int> combination(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) combination[static_cast<std::size_t>(i)] = i;
        do {
            if (max_emitted_combinations > 0 && emitted >= max_emitted_combinations) {
                return close(total_output_mass - emitted_mass, ExclusiveEnumerationStatus::Capped);
            }
            std::vector<int> row(probabilities.size(), 0);
            for (int index : combination) row[static_cast<std::size_t>(index)] = 1;
            const double exclusive = independent_exclusive(probabilities, row);
            output.indicators.push_back(row);
            output.probabilities.push_back(exclusive);
            emitted_mass += exclusive;
            ++emitted;
            union_value += sign *
                (k == 1 ? probabilities[static_cast<std::size_t>(combination[0])]
                        : independent_joint_probability(probabilities, row));
        } while (sf::factorial::detail::next_combination_unchecked(combination, n));
    }
    return output;
}

inline double positively_dependent_exclusive(const std::vector<double>& probabilities,
                                             const std::vector<int>& indicators) {
    if (probabilities.empty() || indicators.empty() || probabilities.size() != indicators.size()) {
        throw std::invalid_argument("probabilities and indicators must have the same nonzero length");
    }
    double minimum = 1.0;
    double maximum = 0.0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (std::isnan(probabilities[i])) return detail::kNaN;
        if (indicators[i] == 1) minimum = std::min(minimum, probabilities[i]);
        else maximum = std::max(maximum, probabilities[i]);
    }
    return detail::clamp(minimum - maximum, 0.0, 1.0);
}

inline ExclusiveEnumerationResult positively_dependent_exclusive_lazy(
    const std::vector<double>& probabilities, double absolute_tolerance = 1e-4,
    double relative_tolerance = 1e-4) {
    if (probabilities.empty()) throw std::invalid_argument("probabilities must not be empty");
    ExclusiveEnumerationResult output{ExclusiveEnumerationStatus::Complete, {}, {}};
    const int n = static_cast<int>(probabilities.size());
    double union_value = 0.0;
    double sign = 1.0;
    double inclusion = detail::kNaN;
    double exclusion = detail::kNaN;
    for (int subset_size = 1; subset_size <= n; ++subset_size) {
        if (subset_size >= 2) {
            const int block = subset_size - 2;
            if (block > 0) {
                if (sign == 1.0) inclusion = union_value;
                else if (sign == -1.0) exclusion = union_value;
            }
            const double difference = std::fabs(inclusion - exclusion);
            if (block > 0 && block < n && difference <= absolute_tolerance &&
                difference <= relative_tolerance * std::min(inclusion, exclusion)) {
                output.indicators.emplace_back(probabilities.size(), 1);
                output.probabilities.push_back(detail::clamp(0.5 * difference, 0.0, 1.0));
                output.status = ExclusiveEnumerationStatus::Converged;
                return output;
            }
            sign *= -1.0;
        }
        std::vector<int> combination(static_cast<std::size_t>(subset_size));
        for (int i = 0; i < subset_size; ++i) combination[static_cast<std::size_t>(i)] = i;
        do {
            std::vector<int> row(probabilities.size(), 0);
            for (int index : combination) row[static_cast<std::size_t>(index)] = 1;
            output.probabilities.push_back(
                positively_dependent_exclusive(probabilities, row));
            output.indicators.push_back(row);
            union_value += sign *
                (subset_size == 1
                     ? probabilities[static_cast<std::size_t>(combination[0])]
                     : positive_joint_probability(probabilities, row));
        } while (sf::factorial::detail::next_combination_unchecked(combination, n));
    }
    return output;
}

// Mirrors Probability.PositivelyDependentUnion(IList<double>).
inline double positively_dependent_union(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (probabilities.size() == 1) return probabilities[0];
    double m = detail::max_value(probabilities);
    return detail::clamp(m, 0.0, 1.0);
}

// Mirrors Probability.NegativelyDependentUnion(IList<double>).
inline double negatively_dependent_union(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (probabilities.size() == 1) return probabilities[0];
    double s = 0.0;
    for (double p : probabilities) s += p;
    return detail::clamp(s, 0.0, 1.0);
}

// Mirrors Probability.Union(IList<double>, DependencyType = Independent).
inline double union_probability(const std::vector<double>& probabilities,
                                 DependencyType dependency = DependencyType::Independent) {
    if (dependency == DependencyType::Independent) return independent_union(probabilities);
    if (dependency == DependencyType::PerfectlyPositive) return positively_dependent_union(probabilities);
    if (dependency == DependencyType::PerfectlyNegative) return negatively_dependent_union(probabilities);
    return detail::kNaN;
}

inline void single_factor_conditional_probabilities(const std::vector<double>& normal_thresholds,
                                                    double rho, double z,
                                                    std::vector<double>& conditional) {
    if (conditional.size() < normal_thresholds.size()) {
        throw std::invalid_argument("conditional buffer must be at least as long as thresholds");
    }
    if (!corehydro::numerics::is_finite(rho) || rho < 0.0 || rho >= 1.0) {
        throw std::out_of_range("common correlation must be within [0, 1)");
    }
    if (!corehydro::numerics::is_finite(z)) throw std::out_of_range("factor value must be finite");
    const double sqrt_rho = std::sqrt(rho);
    const double sqrt_complement = std::sqrt(1.0 - rho);
    for (std::size_t i = 0; i < normal_thresholds.size(); ++i) {
        conditional[i] = distributions::Normal::standard_cdf(
            (normal_thresholds[i] - sqrt_rho * z) / sqrt_complement);
    }
}

inline double union_single_factor(const std::vector<double>& probabilities, double rho,
                                  double relative_tolerance = 1e-8) {
    if (probabilities.empty()) throw std::invalid_argument("probabilities must not be empty");
    for (double probability : probabilities) {
        if (!corehydro::numerics::is_finite(probability) || probability < 0.0 || probability > 1.0) {
            throw std::out_of_range("probabilities must be finite and within [0, 1]");
        }
    }
    if (!corehydro::numerics::is_finite(rho) || rho < 0.0 || rho > 1.0) {
        throw std::out_of_range("common correlation must be within [0, 1]");
    }
    if (!corehydro::numerics::is_finite(relative_tolerance) || relative_tolerance < 1e-15 ||
        relative_tolerance > 1.0) {
        throw std::out_of_range("relative tolerance must be within [1e-15, 1]");
    }

    double maximum_probability = 0.0;
    std::vector<double> thresholds;
    thresholds.reserve(probabilities.size());
    for (double probability : probabilities) {
        if (probability >= 1.0) return 1.0;
        maximum_probability = std::max(maximum_probability, probability);
        if (probability > 0.0) thresholds.push_back(distributions::Normal::standard_z(probability));
    }
    if (thresholds.empty()) return 0.0;
    if (rho == 1.0) return maximum_probability;

    const double sqrt_rho = std::sqrt(rho);
    const double sqrt_complement = std::sqrt(1.0 - rho);
    std::vector<double> conditional(thresholds.size());
    auto integrand = [&](double u) {
        const double z = distributions::Normal::standard_z(u);
        double log_survival = 0.0;
        for (std::size_t i = 0; i < thresholds.size(); ++i) {
            conditional[i] = distributions::Normal::standard_cdf(
                (thresholds[i] - sqrt_rho * z) / sqrt_complement);
            log_survival += std::log1p(-conditional[i]);
        }
        return -corehydro::numerics::expm1(log_survival);
    };
    math::integration::AdaptiveGaussKronrod quadrature(integrand, 1e-16, 1.0 - 1e-16);
    quadrature.relative_tolerance = relative_tolerance;
    quadrature.absolute_tolerance = 1e-15;
    quadrature.min_depth = 2;
    quadrature.report_failure = true;
    quadrature.integrate();
    if (quadrature.status() != math::integration::IntegrationStatus::Success) {
        throw std::runtime_error(
            "single-factor union exhausted its evaluation budget before meeting tolerance");
    }
    return detail::clamp(quadrature.result(), 0.0, 1.0);
}

inline double union_pcm_lazy(const std::vector<double>& probabilities,
                             const Matrix2D& correlation_matrix,
                             ExclusiveEnumerationStatus& status,
                             double absolute_tolerance = 1e-4,
                             double relative_tolerance = 1e-4) {
    if (probabilities.empty() || correlation_matrix.empty()) {
        throw std::invalid_argument(
            "probabilities and correlation matrix must be non-empty");
    }
    const int n = static_cast<int>(probabilities.size());
    std::vector<int> row(probabilities.size(), 0);
    double union_value = 0.0;
    double sign = 1.0;
    double inclusion = detail::kNaN;
    double exclusion = detail::kNaN;
    for (int subset_size = 1; subset_size <= n; ++subset_size) {
        if (subset_size >= 2) {
            const int block = subset_size - 2;
            if (block > 0) {
                if (sign == 1.0) inclusion = union_value;
                else if (sign == -1.0) exclusion = union_value;
            }
            const double difference = std::fabs(inclusion - exclusion);
            if (block > 0 && block < n && difference <= absolute_tolerance &&
                difference <= relative_tolerance * std::min(inclusion, exclusion)) {
                status = ExclusiveEnumerationStatus::Converged;
                return detail::clamp(union_value + 0.5 * difference, 0.0, 1.0);
            }
            sign *= -1.0;
        }
        std::vector<int> combination(static_cast<std::size_t>(subset_size));
        for (int i = 0; i < subset_size; ++i) combination[static_cast<std::size_t>(i)] = i;
        do {
            std::fill(row.begin(), row.end(), 0);
            for (int index : combination) row[static_cast<std::size_t>(index)] = 1;
            const double joint = subset_size == 1
                ? probabilities[static_cast<std::size_t>(combination[0])]
                : joint_probability(probabilities, row, &correlation_matrix);
            union_value += sign * joint;
        } while (sf::factorial::detail::next_combination_unchecked(combination, n));
    }
    status = ExclusiveEnumerationStatus::Complete;
    return detail::clamp(union_value, 0.0, 1.0);
}

inline double sum_search(const std::vector<double>& values,
                         const std::vector<int>& required,
                         const std::vector<std::vector<int>>& indicators,
                         std::size_t start, std::size_t end) {
    double result = 0.0;
    for (std::size_t row = start; row < end; ++row) {
        bool inclusive = true;
        for (std::size_t column = 0; column < required.size(); ++column) {
            if (required[column] == 1 && indicators[row][column] == 0) {
                inclusive = false;
                break;
            }
        }
        if (inclusive) result += values[row];
    }
    return result;
}

inline ExclusiveEnumerationResult exclusive_pcm_lazy(
    const std::vector<double>& probabilities, const Matrix2D& correlation_matrix,
    double absolute_tolerance = 1e-4, double relative_tolerance = 1e-4) {
    if (probabilities.empty() || correlation_matrix.empty()) {
        throw std::invalid_argument(
            "probabilities and correlation matrix must be non-empty");
    }
    ExclusiveEnumerationResult output{ExclusiveEnumerationStatus::Complete, {}, {}};
    std::vector<double> joint_probabilities;
    std::vector<std::size_t> cumulative_combinations;
    const int n = static_cast<int>(probabilities.size());
    double union_value = 0.0;
    double sign = 1.0;
    double inclusion = detail::kNaN;
    double exclusion = detail::kNaN;
    for (int subset_size = 1; subset_size <= n; ++subset_size) {
        if (subset_size >= 2) {
            const int block = subset_size - 2;
            if (block > 0) {
                if (sign == 1.0) inclusion = union_value;
                else if (sign == -1.0) exclusion = union_value;
            }
            const double difference = std::fabs(inclusion - exclusion);
            if (block > 0 && block < n && difference <= absolute_tolerance &&
                difference <= relative_tolerance * std::min(inclusion, exclusion)) {
                output.indicators.emplace_back(probabilities.size(), 1);
                joint_probabilities.push_back(detail::clamp(0.5 * difference, 0.0, 1.0));
                output.status = ExclusiveEnumerationStatus::Converged;
                break;
            }
            sign *= -1.0;
        }
        std::vector<int> combination(static_cast<std::size_t>(subset_size));
        for (int i = 0; i < subset_size; ++i) combination[static_cast<std::size_t>(i)] = i;
        do {
            std::vector<int> row(probabilities.size(), 0);
            for (int index : combination) row[static_cast<std::size_t>(index)] = 1;
            const double joint = subset_size == 1
                ? probabilities[static_cast<std::size_t>(combination[0])]
                : joint_probability(probabilities, row, &correlation_matrix);
            output.indicators.push_back(row);
            joint_probabilities.push_back(detail::clamp(joint, 0.0, 1.0));
            union_value += sign * joint;
        } while (sf::factorial::detail::next_combination_unchecked(combination, n));
        if (subset_size < n) cumulative_combinations.push_back(joint_probabilities.size());
    }

    std::size_t combination_block = 0;
    std::size_t next_block_start = cumulative_combinations.empty()
        ? std::numeric_limits<std::size_t>::max()
        : cumulative_combinations[0];
    output.probabilities.reserve(joint_probabilities.size());
    for (std::size_t i = 0; i < joint_probabilities.size(); ++i) {
        if (i == next_block_start) {
            ++combination_block;
            next_block_start = combination_block < cumulative_combinations.size()
                ? cumulative_combinations[combination_block]
                : std::numeric_limits<std::size_t>::max();
        }
        double exclusive = joint_probabilities[i];
        double association = 1.0;
        for (std::size_t block = combination_block;
             block < cumulative_combinations.size(); ++block) {
            association *= -1.0;
            const std::size_t start = cumulative_combinations[block];
            const std::size_t end = block + 1 == cumulative_combinations.size()
                ? cumulative_combinations[block] + 1
                : cumulative_combinations[block + 1];
            exclusive += association * sum_search(joint_probabilities, output.indicators[i],
                                                  output.indicators, start, end);
        }
        output.probabilities.push_back(detail::clamp(exclusive, 0.0, 1.0));
    }
    return output;
}

// Returns the probability of union using the inclusion-exclusion method, with dependence
// captured by joint_probability_hpcm via correlation_matrix. Mirrors the 6-arg
// Probability.UnionPCM(IList<double>, int[], int[,], double[,], double, double) verbatim.
inline double union_pcm(const std::vector<double>& probabilities,
                         const std::vector<int>& binomial_combinations,
                         const std::vector<std::vector<int>>& indicators,
                         const Matrix2D& correlation_matrix, double absolute_tolerance = 1e-4,
                         double relative_tolerance = 1e-4) {
    if (probabilities.empty() || binomial_combinations.empty() || indicators.empty())
        throw std::invalid_argument(
            "probabilities, binomial_combinations, and indicators must be non-empty");

    double result = 0.0;
    double s = 1.0;
    std::size_t j = 0;
    int c = binomial_combinations[j];
    double inc = detail::kNaN;
    double exc = detail::kNaN;
    std::size_t num_indicators = indicators.size();

    for (std::size_t i = 0; i < num_indicators; ++i) {
        if (static_cast<int>(i) == c) {
            if (j > 0) {
                if (s == 1.0) inc = result;
                else if (s == -1.0) exc = result;
            }
            double diff = std::fabs(inc - exc);
            if (j > 0 && j < binomial_combinations.size() && diff <= absolute_tolerance &&
                diff <= relative_tolerance * std::min(inc, exc)) {
                return result + 0.5 * diff;
            }
            s *= -1.0;
            ++j;
            if (j < binomial_combinations.size()) c += binomial_combinations[j];
        }
        if (i < probabilities.size()) {
            result += s * probabilities[i];
        } else {
            result += s * joint_probability(probabilities, indicators[i], &correlation_matrix);
        }
    }
    return result;
}

// Mirrors the 2-arg Probability.UnionPCM(IList<double>, double[,], double, double)
// convenience overload: builds the binomial-combination counts and all-subsets indicator
// table, then delegates to the 6-arg overload above.
inline double union_pcm(const std::vector<double>& probabilities, const Matrix2D& correlation_matrix,
                         double absolute_tolerance = 1e-4, double relative_tolerance = 1e-4) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities and correlation matrix must be non-empty");
    ExclusiveEnumerationStatus status = ExclusiveEnumerationStatus::Complete;
    return union_pcm_lazy(probabilities, correlation_matrix, status, absolute_tolerance,
                          relative_tolerance);
}

}  // namespace corehydro::numerics::data::probability
