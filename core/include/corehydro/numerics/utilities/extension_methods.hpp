// ported from: Numerics/Utilities/ExtensionMethods.cs @ 2a0357a
//
// C# extends `System.Random`/arrays/`Vector`/`Matrix` with instance-style extension
// methods; C++ has no such mechanism, so each ported member becomes a free function
// taking the receiver as its first parameter. This header ports the random-sampling
// helpers the MCMC/Bootstrap prerequisites need: `next_doubles(rng, n)`,
// `next_doubles(rng, n, dim)`, `next_integers(rng, n)`, and (A3) the two ranged
// `next_integers(rng, minValue, maxValue, length[, replace])` overloads the DataFrame
// resampling surface consumes (`GetRow` is not ported -- neither of the two `NextDoubles`
// overloads calls it, unlike the brief's initial guess).
//
// `next_doubles(rng, n, dim)` transcribes a specific quirk exactly (source ~lines
// 130-157): rather than drawing n*dim values off ONE stream, it constructs `dim`
// independent sub-`MersenneTwister`s -- each seeded by a single `rng.next()` draw off the
// PARENT generator, taken in dimension order -- and fills column `i` by advancing that
// sub-generator's own stream `n` times. This sub-stream-per-column pattern must stay
// bit-exact: a later task (SNIS) depends on reproducing it against the real C# output.
//
// Omitted (no ported caller needs them; the C# source has many more extension-method
// regions -- Enum, 1-D/2-D Array, Vector, Matrix -- covering `Apply`/`Map`/
// `RandomSubset`/`Fill`/`GetColumn`/`SetRow`/`SetColumn`/etc.): every `ExtensionMethods.cs`
// member outside the ones ported below. `Subset` left that list in P6 -- its two 1-D array
// overloads are ported (see their own comments); the 2-D array, Vector and Matrix overloads
// of the same name are not. `almost_equals` (the `Double` region's sole
// member) was added additively for the BestFit v2.0.0 DataFrame plotting-position rewrite,
// which uses it to detect/separate tied Hirsch-Stedinger positions.
#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::utilities {

// Returns true when `a` and `b` differ by no more than `epsilon` (matches C# `AlmostEquals
// (this double a, double b, double epsilon = 1E-15)`).
inline bool almost_equals(double a, double b, double epsilon = 1e-15) {
    return std::fabs(a - b) <= epsilon;
}

// Returns the elements from `start_index` to the end (C# `Subset<T>(this T[] array,
// int startIndex)`, source line 266). P6 un-severed the two 1-D array `Subset` overloads for
// `TimeSeries::summary_hypothesis_test`, which splits its value array with both of them; the
// 2-D array, Vector and Matrix overloads stay omitted (no ported caller).
inline std::vector<double> subset(const std::vector<double>& array, int start_index) {
    std::vector<double> result;
    result.reserve(array.size() - static_cast<std::size_t>(start_index));
    for (std::size_t i = static_cast<std::size_t>(start_index); i < array.size(); ++i)
        result.push_back(array[i]);
    return result;
}

// Returns the elements from `start_index` to `end_index` INCLUSIVE (C# `Subset<T>(this T[]
// array, int startIndex, int endIndex)`, source line 285 -- its loop is `i <= endIndex` and its
// result length is `endIndex - startIndex + 1`).
inline std::vector<double> subset(const std::vector<double>& array, int start_index,
                                  int end_index) {
    std::vector<double> result;
    if (end_index >= start_index)
        result.reserve(static_cast<std::size_t>(end_index - start_index + 1));
    for (int i = start_index; i <= end_index; ++i)
        result.push_back(array[static_cast<std::size_t>(i)]);
    return result;
}

// Returns an array of `length` random doubles (matches C# `NextDoubles(this Random,
// int length)`).
inline std::vector<double> next_doubles(sampling::MersenneTwister& random, int length) {
    std::vector<double> values(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i) values[static_cast<std::size_t>(i)] = random.next_double();
    return values;
}

// Returns a 2-D array of random doubles shaped [length][dimension] (row = sample index,
// column = dimension index -- matches C#'s `values[j, i]` indexing, where `i` is the
// dimension/outer loop and `j` the sample/inner loop). See the file header comment for
// the sub-MersenneTwister-per-column quirk this transcribes exactly (matches C#
// `NextDoubles(this Random, int length, int dimension)`).
inline std::vector<std::vector<double>> next_doubles(sampling::MersenneTwister& random, int length,
                                                       int dimension) {
    std::vector<std::vector<double>> values(
        static_cast<std::size_t>(length), std::vector<double>(static_cast<std::size_t>(dimension)));
    for (int i = 0; i < dimension; ++i) {
        sampling::MersenneTwister sub(static_cast<std::uint32_t>(random.next()));
        for (int j = 0; j < length; ++j)
            values[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = sub.next_double();
    }
    return values;
}

// Returns an array of `length` random integers (matches C# `NextIntegers(this Random,
// int length)`, which draws each element via the parameterless `Next()`).
inline std::vector<int> next_integers(sampling::MersenneTwister& random, int length) {
    std::vector<int> values(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i) values[static_cast<std::size_t>(i)] = random.next();
    return values;
}

// Returns an array of `length` random integers on [minValue, maxValue) (matches C#
// `NextIntegers(this Random, int minValue, int maxValue, int length)`, which draws each
// element via `random.Next(minValue, maxValue)`).
inline std::vector<int> next_integers(sampling::MersenneTwister& random, int min_value,
                                      int max_value, int length) {
    std::vector<int> values(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i)
        values[static_cast<std::size_t>(i)] = random.next(min_value, max_value);
    return values;
}

// Returns an array of `length` random integers on [minValue, maxValue), with or without
// replacement (matches C# `NextIntegers(this Random, int minValue, int maxValue, int
// length, bool replace = true)`). The `replace == true` branch is identical to the 4-arg
// overload. The `replace == false` branch fills a bin list [minValue, maxValue) and draws
// without replacement via `random.Next(0, bins.Count)` + RemoveAt, throwing when `length`
// exceeds the range (all mirroring the C# source).
inline std::vector<int> next_integers(sampling::MersenneTwister& random, int min_value,
                                      int max_value, int length, bool replace) {
    if (replace) {
        std::vector<int> values(static_cast<std::size_t>(length));
        for (int i = 0; i < length; ++i)
            values[static_cast<std::size_t>(i)] = random.next(min_value, max_value);
        return values;
    }
    if (length > max_value - min_value)
        throw std::invalid_argument(
            "When sampling without replacement, the length must be less than or equal to "
            "the range of values.");

    std::vector<int> bins;
    bins.reserve(static_cast<std::size_t>(max_value - min_value));
    for (int i = min_value; i < max_value; ++i) bins.push_back(i);

    // Sample random bin without replacement.
    std::vector<int> values(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i) {
        int r = random.next(0, static_cast<int>(bins.size()));
        values[static_cast<std::size_t>(i)] = bins[static_cast<std::size_t>(r)];
        bins.erase(bins.begin() + r);
    }
    return values;
}

}  // namespace corehydro::numerics::utilities
