// The C++ fixture runner's catalogs of host-language callback counterparts.
//
// fixtures/callback/*.json and fixtures/toolbox/optimizers.json name their callbacks and
// objectives by name; the R runner writes an R closure for each name, the Python runner a Python
// function, the dotnet emitter a C# delegate, and this catalog the C++ lambda. Every one of the
// four is specified as the SAME arithmetic in double, which is what makes
// fixtures/callback/callback_cross_language.json's bit-equality assertions meaningful.
//
// The definitions live in fixture_callback_catalog.cpp rather than here (or in
// test_fixtures.cpp) so the build can compile them, and ONLY them, with -ffp-contract=off --
// see core/CMakeLists.txt for why the flag exists and why it must not reach the ported core.
#pragma once
#include <string>

#include "corehydro/numerics/support/callback_runner.hpp"
#include "corehydro/numerics/support/optimizer_runner.hpp"

namespace fixture_catalog {

// fixtures/toolbox/optimizers.json's objective names (the TestFunctions.cs names:
// DeJong/FXYZ/Booth/McCormick/FX) -> the real C++ function in optimization_test_functions.hpp.
corehydro::numerics::support::Objective optimizer_objective(const std::string& name);

// fixtures/callback/*.json's callback names -> a real C++ lambda, assigned to whichever member of
// `cbs` that name's delegate shape belongs to. Called once per delegate a case supplies, so a
// single CallbackSet accumulates a group's required and optional callbacks.
void callback_set(const std::string& name, corehydro::numerics::support::CallbackSet& cbs);

}  // namespace fixture_catalog
