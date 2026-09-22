// C++-only ctest for the v2.1.4 Numerics BivariateCopula.Clone() deep-copy fix -- the new
// protected static `BivariateCopula.CloneMarginal`, called from every concrete copula's
// `Clone()` override (Numerics/Distributions/Bivariate Copulas/Base/BivariateCopula.cs @
// 2a0357a) -- together with the ArchimedeanCopula ParametersValid sentinel fix (see
// archimedean_copula.hpp) that this test also exercises generically for every family.
//
// Mirrors each family's new `Test_Clone` addition (Numerics/Test_Numerics/Distributions/
// Bivariate Copulas/Test_<Family>Copula.cs @ 2a0357a): AreNotSame identity on
// MarginalDistributionX/Y, an identical marginal quantile post-clone (a deep COPY, not an
// independently-parameterized object), and post-clone mutation independence of Theta. Neither
// AreNotSame nor cross-object mutation independence fits the declarative fixture shape: the
// former is an identity comparison across two independently-constructed objects, and the
// fixture assertion shape (one scalar value per assertion against ONE constructed object) can
// express neither -- see fixtures/README.md's "C++-only ctest" precedent, e.g. test_mixture.cpp
// (Task 6). Pinned here instead as a single, generic-over-all-seven-types ctest built through
// copula_factory.hpp: every copula shares the same BivariateCopula::marginal_distribution_x/y +
// clone()/theta()/set_theta()/parameters_valid() surface, so one parameterized check covers
// every family with no per-type branching (mirrors copula_factory.hpp's own rationale). See
// each fixtures/distributions/copulas/*.json's "note" field for the cross-reference.
#include <memory>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/copulas/base/copula_factory.hpp"
#include "corehydro/numerics/distributions/gumbel.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::numerics::distributions::Gumbel;
using corehydro::numerics::distributions::Normal;
namespace cop = corehydro::numerics::distributions::copulas;

namespace {

// Mirrors every family's Test_Clone body: attach distinct Normal/Gumbel marginals to a
// default-constructed copula, clone it, and confirm (1) the clone's marginals are
// independently-constructed objects (AreNotSame, not aliased), (2) the cloned marginal
// reproduces the SAME quantile as the original (a deep COPY, not a divergent independent
// object), (3) the clone is ParametersValid (the sentinel-fix regression: before v2.1.4 this
// was unconditionally false for Clayton/Gumbel/Joe regardless of theta), and (4) mutating the
// clone's Theta afterward leaves the original untouched (ordinary value independence, part of
// every family's Test_Clone body; unaffected by the marginal fix but confirmed alongside it).
void check_clone_deep_copies_marginals(const std::string& copula_name) {
    auto copula = cop::create_copula(copula_name);
    copula->marginal_distribution_x = std::make_shared<Normal>(100.0, 10.0);
    copula->marginal_distribution_y = std::make_shared<Gumbel>(50.0, 5.0);

    auto clone = copula->clone();

    CHECK_TRUE(clone->marginal_distribution_x.get() != copula->marginal_distribution_x.get());
    CHECK_TRUE(clone->marginal_distribution_y.get() != copula->marginal_distribution_y.get());
    CHECK_NEAR(copula->marginal_distribution_y->inverse_cdf(0.9),
               clone->marginal_distribution_y->inverse_cdf(0.9), 1e-12);

    CHECK_TRUE(copula->parameters_valid());
    CHECK_TRUE(clone->parameters_valid());

    double original_theta = copula->theta();
    clone->set_theta(clone->theta() + 1.0);
    CHECK_NEAR(copula->theta(), original_theta, 0.0);
}

void test_clone_deep_copies_marginals_every_family() {
    for (const std::string& name :
         {"AliMikhailHaq", "Clayton", "Frank", "Gumbel", "Joe", "Normal", "StudentT",
          "Independence"}) {
        check_clone_deep_copies_marginals(name);
    }
}

// A null (unattached) marginal passes through clone_marginal unchanged -- confirmed here on
// Clayton (arbitrary representative; every family shares the same clone_marginal call).
void test_clone_with_no_marginals_attached() {
    auto copula = cop::create_copula("Clayton");
    auto clone = copula->clone();
    CHECK_TRUE(clone->marginal_distribution_x == nullptr);
    CHECK_TRUE(clone->marginal_distribution_y == nullptr);
}

}  // namespace

int main() {
    test_clone_deep_copies_marginals_every_family();
    test_clone_with_no_marginals_attached();
    return chtest::summary("test_copula_clone");
}
