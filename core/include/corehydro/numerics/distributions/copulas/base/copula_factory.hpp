// ported from: Numerics/Distributions/Bivariate Copulas/Base/CopulaFactory.cs @ 7e8e8d1
// factory: copulas are always constructed directly by concrete type (e.g. `new
// ClaytonCopula(...)`), the same situation univariate_distribution_factory.hpp's header
// comment describes for UnivariateDistributionBase before that factory was added. This
// factory exists purely so the fixture runners and R/Python glue can go generic-by-string
// the same way, keyed by the EXACT CopulaType enum names (AliMikhailHaq, Clayton, Frank,
// Gumbel, Joe, Normal, StudentT) so the C# oracle emitter can `Enum.Parse<CopulaType>` the
// identical fixture "target" string. Justified because every copula shares BivariateCopula's
// uniform parameter API (theta/get_copula_parameters/set_copula_parameters/pdf/cdf/...),
// collapsing all runner/glue dispatch to one generic path with zero per-type branches
// (see core/tests/test_fixtures.cpp's run_bivariate_copula and fixtures/README.md).
//
// Only ported copulas have a case -- requesting an unported type throws rather than
// silently returning a placeholder, so gaps surface immediately (mirrors
// univariate_distribution_factory.hpp's create_distribution).
#pragma once
#include <memory>
#include <stdexcept>
#include <string>

#include "corehydro/numerics/distributions/copulas/amh_copula.hpp"
#include "corehydro/numerics/distributions/copulas/base/bivariate_copula.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/distributions/copulas/clayton_copula.hpp"
#include "corehydro/numerics/distributions/copulas/frank_copula.hpp"
#include "corehydro/numerics/distributions/copulas/gumbel_copula.hpp"
#include "corehydro/numerics/distributions/copulas/independence_copula.hpp"
#include "corehydro/numerics/distributions/copulas/joe_copula.hpp"
#include "corehydro/numerics/distributions/copulas/normal_copula.hpp"
#include "corehydro/numerics/distributions/copulas/student_t_copula.hpp"

namespace corehydro::numerics::distributions::copulas {

inline std::unique_ptr<BivariateCopula> create_copula(CopulaType type) {
    switch (type) {
        case CopulaType::AliMikhailHaq:
            return std::make_unique<AMHCopula>();
        case CopulaType::Clayton:
            return std::make_unique<ClaytonCopula>();
        case CopulaType::Frank:
            return std::make_unique<FrankCopula>();
        case CopulaType::Gumbel:
            return std::make_unique<GumbelCopula>();
        case CopulaType::Joe:
            return std::make_unique<JoeCopula>();
        case CopulaType::Normal:
            return std::make_unique<NormalCopula>();
        case CopulaType::StudentT:
            return std::make_unique<StudentTCopula>();
        case CopulaType::Independence:
            return std::make_unique<IndependenceCopula>();
        default:
            throw std::invalid_argument("copula type not yet ported");
    }
}

// Construct from the C# CopulaType enum name (the value stored in fixtures' "target"
// field).
inline std::unique_ptr<BivariateCopula> create_copula(const std::string& name) {
    if (name == "AliMikhailHaq") return create_copula(CopulaType::AliMikhailHaq);
    if (name == "Clayton") return create_copula(CopulaType::Clayton);
    if (name == "Frank") return create_copula(CopulaType::Frank);
    if (name == "Gumbel") return create_copula(CopulaType::Gumbel);
    if (name == "Joe") return create_copula(CopulaType::Joe);
    if (name == "Normal") return create_copula(CopulaType::Normal);
    if (name == "StudentT") return create_copula(CopulaType::StudentT);
    if (name == "Independence") return create_copula(CopulaType::Independence);
    throw std::invalid_argument("unknown copula name: " + name);
}

}  // namespace corehydro::numerics::distributions::copulas
