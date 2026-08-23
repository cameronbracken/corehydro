// ported from: Numerics/Functions/IUnivariateFunction.cs @ 2a0357a
//
// Interface for univariate functions: a scalar function of one variable, `Function(x)`, plus its
// analytic `InverseFunction(y)`, over a parameter set the implementation validates itself
// (`ParametersValid`/`ValidateParameters`) and an optional non-deterministic path
// (`IsDeterministic`/`ConfidenceLevel`) for a function whose output carries noise. Style-model:
// `i_link_function.hpp` -- sibling interface, same directory, same snake_case member convention
// -- but this interface is wider (12 members against `ILinkFunction`'s 3) because it also owns
// its own domain clamp (`Minimum`/`Maximum`), parameter bounds
// (`MinimumOfParameters`/`MaximumOfParameters`), and the deterministic/uncertain switch.
// `ArgumentOutOfRangeException? ValidateParameters(...)` (null = valid) becomes `bool
// validate_parameters(...)` (true = valid) with the same `throw_on_error` flag, matching the
// `std::out_of_range` convention the sibling link-function headers already use for
// `ArgumentOutOfRangeException`.
//
// TWO implementations, not three: `LinearFunction` and `PowerFunction` below. The upstream third
// implementation, `TabularFunction` (`Numerics/Functions/TabularFunction.cs`), is NOT ported --
// it is built entirely on `UncertainOrderedPairedData`/`OrderedPairedData`/`Ordinate`/
// `UncertainOrdinate` (the `Numerics.Data` "Paired Data" subsystem, `Numerics/Data/Paired Data/`),
// which this repo has not ported. That subsystem is scheduled for Phase P4 (per
// `docs/superpowers/specs/2026-08-20-remaining-port-and-v1-release-design.md`); TabularFunction
// is deferred alongside it rather than ported against a stand-in container. See
// `upstream/CLAUDE.md`'s "What is deliberately not ported" section for the matching severance
// entry, and `core/tests/test_univariate_functions.cpp`'s header for why
// `Test_Tabular_Function` (`Test_Numerics/Functions/Test_Functions.cs`) is not transcribed.
#pragma once

#include <vector>

namespace corehydro::numerics::functions {

class IUnivariateFunction {
   public:
    virtual ~IUnivariateFunction() = default;

    // The number of function parameters.
    virtual int number_of_parameters() const = 0;

    // Whether the current parameters are valid. If not, calling function()/inverse_function()
    // throws.
    virtual bool parameters_valid() const = 0;

    // The minimum/maximum X value supported by the function. Defaults mirror C#'s
    // double.MinValue/double.MaxValue via std::numeric_limits<double>::lowest()/max().
    virtual double minimum() const = 0;
    virtual void set_minimum(double value) = 0;
    virtual double maximum() const = 0;
    virtual void set_maximum(double value) = 0;

    // The minimum/maximum values allowable for each parameter, in parameter order.
    virtual std::vector<double> minimum_of_parameters() const = 0;
    virtual std::vector<double> maximum_of_parameters() const = 0;

    // Whether the function is deterministic or carries uncertainty.
    virtual bool is_deterministic() const = 0;
    virtual void set_is_deterministic(bool value) = 0;

    // The confidence level to evaluate at when the function is not deterministic.
    virtual double confidence_level() const = 0;
    virtual void set_confidence_level(double value) = 0;

    // Sets every parameter from a flat vector, in the same order minimum_of_parameters()/
    // maximum_of_parameters() report.
    virtual void set_parameters(const std::vector<double>& parameters) = 0;

    // Tests whether `parameters` are valid. Returns true (valid) or false (invalid); when
    // `throw_on_error` is true and the parameters are invalid, throws std::out_of_range instead
    // of returning false (mirrors the C# ArgumentOutOfRangeException? return: null == valid).
    virtual bool validate_parameters(const std::vector<double>& parameters,
                                     bool throw_on_error) const = 0;

    // Evaluates the function at x. If the function is uncertain, evaluates at the set confidence
    // level.
    virtual double function(double x) const = 0;

    // Evaluates the inverse function at y. If the function is uncertain, evaluates at the set
    // confidence level.
    virtual double inverse_function(double y) const = 0;
};

}  // namespace corehydro::numerics::functions
