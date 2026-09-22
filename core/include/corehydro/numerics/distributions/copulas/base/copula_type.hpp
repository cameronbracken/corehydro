// ported from: Numerics/Distributions/Bivariate Copulas/Base/CopulaType.cs @ 7e8e8d1
//
// Enumeration of every bivariate copula. Enumerator names are transcribed VERBATIM
// (including capitalization) because the copula factory (copula_factory.hpp, a corehydro
// addition) and the fixture/emitter glue key construction off these exact strings so the
// C# oracle emitter can `Enum.Parse<CopulaType>` the same fixture "target" value.
#pragma once

namespace corehydro::numerics::distributions::copulas {

enum class CopulaType {
    AliMikhailHaq,
    Clayton,
    Frank,
    Gumbel,
    Joe,
    Normal,
    StudentT,
    Independence
};

}  // namespace corehydro::numerics::distributions::copulas
