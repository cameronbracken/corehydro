// ported from: Numerics/Data/Paired Data/UncertainOrdinate.cs @ 2a0357a
//
// P4 Task 9: a single (X, Y) ordinate where X is a double and Y is a whole continuous
// distribution (nullable), plus the monotonicity checks (OrdinateValid/OrdinateErrors) that
// UncertainOrderedPairedData uses to validate a curve whose Y coordinate carries uncertainty.
// Sibling of Task 7's Ordinate (ordinate.hpp), whose ordinate_valid/ordinate_errors this class
// probes at three points -- min_percentile, the mean or median, and 1 - min_percentile -- rather
// than comparing a single (X, Y) pair directly.
//
// Deliberately NOT ported (project-wide severance -- see e.g. ordinate.hpp's own header): the
// three XElement constructors and ToXElement(). GetHashCode/Equals(object) are likewise not
// ported (no untyped-equality/hashing consumer in this port's scope; operator== is the one
// equality entry point, matching Ordinate's own precedent).
//
// CONSTRUCTION, a corehydro addition with no C# analogue: C#'s `Y` is a nullable reference
// (`UnivariateDistributionBase? Y`), a shared, aliasable handle -- two UncertainOrdinates can
// point at the SAME distribution instance, and the constructors only clone where the C# source
// explicitly calls `.Clone()` (see uncertain_ordered_paired_data.hpp). C++ has no ownership-free
// reference type to mirror that with; `std::unique_ptr<UnivariateDistributionBase>` (this port's
// choice, matching the task brief) forces UncertainOrdinate to be exactly one owner. Rather than
// leave it move-only (which would make `std::vector<UncertainOrdinate>` unable to reallocate by
// copy, and would make the many call sites below that need to set aside a value and later feed
// it back into the collection -- e.g. Test_IList's `ordinate = pairedData[2]; ...;
// pairedData.Insert(2, ordinate);` -- impossible without a bespoke clone call at every site),
// this port gives UncertainOrdinate its own copy constructor/assignment that deep-clones Y via
// the distribution's own `clone()`. That makes every COPY here a clone, where C# only clones at
// specific points and shares references everywhere else -- strictly MORE cloning than upstream,
// never less. This is behaviorally invisible to every method ported in this file and its sibling
// (nothing in the ported surface observes distribution IDENTITY, only distribution VALUE via
// operator==/OrdinateValid/OrdinateErrors), so it cannot diverge from an oracle. A move
// constructor/assignment is declared too so the common case (pushing a freshly-built, soon-to-be-
// discarded temporary into a vector) doesn't pay a clone.
//
// The convenience two-argument constructor below (`UncertainOrdinate(x, SomeDistribution{...})`)
// is likewise a corehydro addition: it lets call sites write `UncertainOrdinate(3, Triangular(6,
// 8, 12))` the way the C# source's `new UncertainOrdinate(3, new Triangular(6, 8, 12))` reads,
// rather than spelling out `std::make_unique<Triangular>(6, 8, 12)` at every call site.
//
// Transcription note 2 (C# OrdinateValid ~lines 152-183 / OrdinateErrors ~lines 192-249):
// `OrdinateValid` probes at `min_percentile`, the MEAN (`GetOrdinate()`), and
// `1 - min_percentile`, while `OrdinateErrors` probes at `min_percentile`, the MEDIAN
// (`GetOrdinate(0.5d)`), and `1 - min_percentile` -- a real, upstream mismatch, not a transcription
// slip on this port's part. For a skewed Y distribution the mean and median differ, so a pair of
// ordinates can fail the central-tendency `OrdinateValid` probe (declaring the curve invalid) while
// `OrdinateErrors`'s median probe passes, silently omitting the very error that made `IsValid`
// false. Both are ported exactly as written; `min_percentile` is `0.05` when
// `Y->type()` is `PertPercentile` or `PertPercentileZ`, else `1e-5`.
//
// Transcription note 3 (C# operator== ~lines 249-260): `X` is compared with EXACT inequality
// (`left.X != right.X`), unlike `Ordinate::operator==` (ordinate.hpp), which allows
// `kDoubleMachineEpsilon` slack. Y-equality then delegates to the distribution's own comparison
// (C# `UnivariateDistributionBase.operator==`, itself type-then-parameters with a special case for
// Empirical's X/probability arrays and a `NotImplementedException` for KernelDensity -- ported
// below as the free function `detail::distributions_equal` since this port's
// `UnivariateDistributionBase` does not carry that operator itself, see
// `univariate_distribution_base.hpp`'s own header note on why equality operators are excluded from
// the math core).
//
// Documented gap (untested in this port's scope, unreachable by any transcribed test): C#
// `OrdinateErrors(ordinateToCompare, ...)`'s invalid-comparand branch, when
// `ordinateToCompare.Y.ParametersValid == false`, calls
// `ordinateToCompare.Y.ValidateParameters(ordinateToCompare.Y.GetParameters, false).Message` and
// appends that message. This port's `UnivariateDistributionBase` exposes only `parameters_valid()`
// (a bool), not a generic "get the validation error text" accessor (no distribution class in this
// port returns its parameter-validation message publicly), so that specific sub-branch has no C++
// counterpart and is omitted; the two errors ABOVE it (X-infinity, X-NaN) and the Y-null branch
// are ported. Every transcribed test in test_uncertain_paired_data.cpp reaches this method only
// with a comparand whose Y (when non-null) has valid parameters, so the omission is never
// exercised.
#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/data/paired_data/ordinate.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data::paired_data {

namespace detail {

// C# `UnivariateDistributionBase.operator==` (Numerics/Distributions/Univariate/Base/
// UnivariateDistributionBase.cs ~lines 716-772), reproduced here as a free function since this
// port's UnivariateDistributionBase does not carry the operator itself. Type must match first;
// Empirical compares its X/probability arrays elementwise; KernelDensity has no ported comparison
// (C# throws NotImplementedException there); every other family compares GetParameters()
// elementwise, all at kDoubleMachineEpsilon absolute tolerance.
inline bool distributions_equal(const distributions::UnivariateDistributionBase* l,
                                 const distributions::UnivariateDistributionBase* r) {
    if (l->type() != r->type()) return false;
    if (l->type() == distributions::UnivariateDistributionType::Empirical) {
        const auto* le = dynamic_cast<const distributions::EmpiricalDistribution*>(l);
        const auto* re = dynamic_cast<const distributions::EmpiricalDistribution*>(r);
        if (le == nullptr || re == nullptr) return false;
        const auto& lx = le->x_values();
        const auto& rx = re->x_values();
        if (lx.size() != rx.size()) return false;
        for (std::size_t i = 0; i < lx.size(); ++i) {
            if (std::fabs(lx[i] - rx[i]) > corehydro::numerics::kDoubleMachineEpsilon) return false;
        }
        const auto& lp = le->p_values();
        const auto& rp = re->p_values();
        for (std::size_t i = 0; i < lp.size(); ++i) {
            if (std::fabs(lp[i] - rp[i]) > corehydro::numerics::kDoubleMachineEpsilon) return false;
        }
        return true;
    }
    if (l->type() == distributions::UnivariateDistributionType::KernelDensity) {
        throw std::logic_error("Equality is not implemented for KernelDensity distributions.");
    }
    auto lp = l->get_parameters();
    auto rp = r->get_parameters();
    for (std::size_t i = 0; i < lp.size(); ++i) {
        if (std::fabs(lp[i] - rp[i]) > corehydro::numerics::kDoubleMachineEpsilon) return false;
    }
    return true;
}

}  // namespace detail

struct UncertainOrdinate {
    double x = 0.0;
    std::unique_ptr<distributions::UnivariateDistributionBase> y;
    bool is_valid = false;

    UncertainOrdinate() = default;

    // C# constructor (lines ~38-44): computes is_valid, false whenever X is +-infinity/NaN, Y is
    // null, or Y's own parameters are invalid.
    UncertainOrdinate(double x_value, std::unique_ptr<distributions::UnivariateDistributionBase> y_value)
        : x(x_value), y(std::move(y_value)) {
        is_valid = true;
        if (std::isinf(x) || std::isnan(x) || !y || !y->parameters_valid()) is_valid = false;
    }

    // Corehydro addition (see header note): lets a call site write
    // `UncertainOrdinate(3, Triangular(6, 8, 12))` directly. SFINAE-restricted to concrete
    // distribution types so it never competes with the unique_ptr constructor above (needed for
    // `UncertainOrdinate(x, nullptr)`).
    template <typename Dist, typename = std::enable_if_t<
                                  std::is_base_of_v<distributions::UnivariateDistributionBase, Dist>>>
    UncertainOrdinate(double x_value, Dist dist)
        : UncertainOrdinate(x_value, std::make_unique<Dist>(std::move(dist))) {}

    // Move: cheap (no clone).
    UncertainOrdinate(UncertainOrdinate&&) = default;
    UncertainOrdinate& operator=(UncertainOrdinate&&) = default;

    // Copy: deep-clones Y. See header note.
    UncertainOrdinate(const UncertainOrdinate& other)
        : x(other.x), y(other.y ? other.y->clone() : nullptr), is_valid(other.is_valid) {}
    UncertainOrdinate& operator=(const UncertainOrdinate& other) {
        if (this != &other) {
            x = other.x;
            y = other.y ? other.y->clone() : nullptr;
            is_valid = other.is_valid;
        }
        return *this;
    }

    // C# GetOrdinate(double) (lines ~117-124).
    Ordinate get_ordinate(double probability) const {
        if (!y) throw std::runtime_error("Cannot get ordinate when Y distribution is null.");
        return Ordinate(x, y->inverse_cdf(probability));
    }

    // C# GetOrdinate() (lines ~130-137).
    Ordinate get_ordinate() const {
        if (!y) throw std::runtime_error("Cannot get ordinate when Y distribution is null.");
        return Ordinate(x, y->mean());
    }

    // C# OrdinateValid (lines ~152-183). See transcription note 2 above for the min_percentile/
    // mean probe set (OrdinateErrors below probes at the median instead).
    bool ordinate_valid(const UncertainOrdinate& compare, bool strict_x, bool strict_y, SortOrder x_order,
                         SortOrder y_order, bool compare_ordinate_is_next,
                         bool allow_different_types = false) const {
        if (!is_valid) return false;
        if (!compare.is_valid) return false;
        if (!y || !compare.y) return false;
        if (!allow_different_types && compare.y->type() != y->type()) return false;

        double min_percentile =
            (y->type() == distributions::UnivariateDistributionType::PertPercentile ||
             y->type() == distributions::UnivariateDistributionType::PertPercentileZ)
                ? 0.05
                : 1e-5;

        if (!get_ordinate(min_percentile)
                 .ordinate_valid(compare.get_ordinate(min_percentile), strict_x, strict_y, x_order,
                                 y_order, compare_ordinate_is_next))
            return false;
        if (!get_ordinate().ordinate_valid(compare.get_ordinate(), strict_x, strict_y, x_order, y_order,
                                            compare_ordinate_is_next))
            return false;
        if (!get_ordinate(1 - min_percentile)
                 .ordinate_valid(compare.get_ordinate(1 - min_percentile), strict_x, strict_y, x_order,
                                 y_order, compare_ordinate_is_next))
            return false;
        return true;
    }

    // C# OrdinateErrors(ordinateToCompare, ...) (lines ~192-249). See transcription note 2 above
    // for the min_percentile/median probe set, and the documented gap above for the one omitted
    // sub-branch.
    std::vector<std::string> ordinate_errors(const UncertainOrdinate& compare, bool strict_x,
                                              bool strict_y, SortOrder x_order, SortOrder y_order,
                                              bool compare_ordinate_is_next,
                                              bool allow_different_types = false) const {
        std::vector<std::string> result = ordinate_errors();

        if (!compare.is_valid) {
            if (std::isinf(compare.x)) result.push_back("Ordinate X value can not be infinity.");
            if (std::isnan(compare.x)) result.push_back("Ordinate X value must be a valid number.");
            if (!compare.y) {
                result.push_back("Ordinate Y value must be defined.");
            }
            // else if (!compare.y->parameters_valid()): documented gap above, no C++ counterpart.
        }

        if (y && compare.y && !allow_different_types && compare.y->type() != y->type()) {
            result.push_back("Can't compare two ordinates with different distribution types.");
        }

        if (is_valid && compare.is_valid && y && compare.y) {
            double min_percentile =
                (y->type() == distributions::UnivariateDistributionType::PertPercentile ||
                 y->type() == distributions::UnivariateDistributionType::PertPercentileZ)
                    ? 0.05
                    : 1e-5;

            auto lo = get_ordinate(min_percentile)
                          .ordinate_errors(compare.get_ordinate(min_percentile), strict_x, strict_y,
                                           x_order, y_order, compare_ordinate_is_next);
            result.insert(result.end(), lo.begin(), lo.end());

            auto mid = get_ordinate(0.5).ordinate_errors(compare.get_ordinate(0.5), strict_x, strict_y,
                                                          x_order, y_order, compare_ordinate_is_next);
            result.insert(result.end(), mid.begin(), mid.end());

            auto hi = get_ordinate(1 - min_percentile)
                          .ordinate_errors(compare.get_ordinate(1 - min_percentile), strict_x, strict_y,
                                           x_order, y_order, compare_ordinate_is_next);
            result.insert(result.end(), hi.begin(), hi.end());
        }
        return result;
    }

    // C# OrdinateErrors() (lines ~255-268). See documented gap above for the one omitted
    // sub-branch (Y.ParametersValid == false, no C++ message accessor).
    std::vector<std::string> ordinate_errors() const {
        std::vector<std::string> result;
        if (!is_valid) {
            if (std::isinf(x)) result.push_back("Ordinate X value can not be infinity.");
            if (std::isnan(x)) result.push_back("Ordinate X value must be a valid number.");
            if (!y) {
                result.push_back("Ordinate Y value must be defined.");
            }
            // else if (!y->parameters_valid()): documented gap above, no C++ counterpart.
        }
        return result;
    }
};

// C# operator== (lines ~249-260). See transcription note 3 above: exact X equality (unlike
// Ordinate's kDoubleMachineEpsilon slack), then delegates Y-equality to the distribution's own
// comparison.
inline bool operator==(const UncertainOrdinate& l, const UncertainOrdinate& r) {
    if (l.x != r.x) return false;
    if (!l.y && !r.y) return true;
    if (!l.y || !r.y) return false;
    return detail::distributions_equal(l.y.get(), r.y.get());
}

// C# operator!= (lines ~269-273).
inline bool operator!=(const UncertainOrdinate& l, const UncertainOrdinate& r) { return !(l == r); }

}  // namespace corehydro::numerics::data::paired_data
