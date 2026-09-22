// ported from: Numerics/Distributions/Univariate/Base/DistributionSnapshot.cs @ 7e8e8d1
#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"

namespace corehydro::numerics::distributions {

// Immutable bitwise state used to invalidate distribution caches. This is deliberately an
// internal comparison object, not a serialization format.
class DistributionSnapshot {
   public:
    static std::optional<DistributionSnapshot> try_capture(
        const UnivariateDistributionBase* distribution) {
        if (distribution == nullptr) return std::nullopt;
        return DistributionSnapshot(*distribution);
    }

    bool matches(const UnivariateDistributionBase* distribution) const {
        if (distribution == nullptr || distribution->type() != type_) return false;
        const auto parameters = distribution->get_parameters();
        if (parameters.size() != scalar_bits_.size()) return false;
        for (std::size_t i = 0; i < parameters.size(); ++i)
            if (bits(parameters[i]) != scalar_bits_[i]) return false;
        return true;
    }

   private:
    explicit DistributionSnapshot(const UnivariateDistributionBase& distribution)
        : type_(distribution.type()) {
        const auto parameters = distribution.get_parameters();
        scalar_bits_.reserve(parameters.size());
        for (double parameter : parameters) scalar_bits_.push_back(bits(parameter));
    }

    static std::uint64_t bits(double value) {
        std::uint64_t result;
        static_assert(sizeof(result) == sizeof(value));
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    UnivariateDistributionType type_;
    std::vector<std::uint64_t> scalar_bits_;
};

}  // namespace corehydro::numerics::distributions
