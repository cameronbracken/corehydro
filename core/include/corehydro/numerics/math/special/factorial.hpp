// ported from: Numerics/Mathematics/Special Functions/Factorial.cs @ 7e8e8d1
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"

namespace corehydro::numerics::math::special::factorial {

namespace detail {

// 171-element factorial cache [0! .. 170!], exact table from Factorial.cs.
inline const double* factorial_cache() {
    static constexpr double cache[171] = {
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
        3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
        1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0,
        1.21645100408832e+17, 2.43290200817664e+18, 5.109094217170944e+19,
        1.1240007277776077e+21, 2.5852016738884978e+22, 6.2044840173323941e+23,
        1.5511210043330986e+25, 4.0329146112660565e+26, 1.0888869450418352e+28,
        3.0488834461171384e+29, 8.8417619937397008e+30, 2.6525285981219103e+32,
        8.2228386541779224e+33, 2.6313083693369352e+35, 8.6833176188118859e+36,
        2.9523279903960412e+38, 1.0333147966386144e+40, 3.7199332678990118e+41,
        1.3763753091226343e+43, 5.23022617466601e+44, 2.0397882081197442e+46,
        8.1591528324789768e+47, 3.3452526613163803e+49, 1.4050061177528798e+51,
        6.0415263063373834e+52, 2.6582715747884485e+54, 1.1962222086548019e+56,
        5.5026221598120885e+57, 2.5862324151116818e+59, 1.2413915592536073e+61,
        6.0828186403426752e+62, 3.0414093201713376e+64, 1.5511187532873822e+66,
        8.0658175170943877e+67, 4.2748832840600255e+69, 2.3084369733924138e+71,
        1.2696403353658276e+73, 7.1099858780486348e+74, 4.0526919504877221e+76,
        2.3505613312828789e+78, 1.3868311854568987e+80, 8.3209871127413916e+81,
        5.0758021387722484e+83, 3.1469973260387939e+85, 1.98260831540444e+87,
        1.2688693218588417e+89, 8.2476505920824715e+90, 5.4434493907744307e+92,
        3.6471110918188683e+94, 2.4800355424368305e+96, 1.711224524281413e+98,
        1.197857166996989e+100, 8.5047858856786218e+101, 6.1234458376886077e+103,
        4.4701154615126834e+105, 3.3078854415193856e+107, 2.4809140811395391e+109,
        1.8854947016660499e+111, 1.4518309202828584e+113, 1.1324281178206295e+115,
        8.9461821307829729e+116, 7.1569457046263779e+118, 5.7971260207473655e+120,
        4.75364333701284e+122, 3.9455239697206569e+124, 3.314240134565352e+126,
        2.8171041143805494e+128, 2.4227095383672724e+130, 2.1077572983795269e+132,
        1.8548264225739836e+134, 1.6507955160908453e+136, 1.4857159644817607e+138,
        1.3520015276784023e+140, 1.24384140546413e+142, 1.1567725070816409e+144,
        1.0873661566567424e+146, 1.0329978488239052e+148, 9.916779348709491e+149,
        9.6192759682482062e+151, 9.426890448883242e+153, 9.33262154439441e+155,
        9.33262154439441e+157, 9.4259477598383536e+159, 9.6144667150351211e+161,
        9.9029007164861754e+163, 1.0299016745145622e+166, 1.0813967582402904e+168,
        1.1462805637347078e+170, 1.2265202031961373e+172, 1.3246418194518284e+174,
        1.4438595832024928e+176, 1.5882455415227421e+178, 1.7629525510902437e+180,
        1.9745068572210728e+182, 2.2311927486598123e+184, 2.5435597334721862e+186,
        2.9250936934930141e+188, 3.3931086844518965e+190, 3.969937160808719e+192,
        4.6845258497542883e+194, 5.5745857612076033e+196, 6.6895029134491239e+198,
        8.09429852527344e+200, 9.8750442008335976e+202, 1.2146304367025325e+205,
        1.5061417415111404e+207, 1.8826771768889254e+209, 2.3721732428800459e+211,
        3.0126600184576582e+213, 3.8562048236258025e+215, 4.9745042224772855e+217,
        6.4668554892204716e+219, 8.4715806908788174e+221, 1.1182486511960039e+224,
        1.4872707060906852e+226, 1.9929427461615181e+228, 2.6904727073180495e+230,
        3.6590428819525472e+232, 5.01288874827499e+234, 6.9177864726194859e+236,
        9.6157231969410859e+238, 1.346201247571752e+241, 1.89814375907617e+243,
        2.6953641378881614e+245, 3.8543707171800706e+247, 5.5502938327393013e+249,
        8.0479260574719866e+251, 1.17499720439091e+254, 1.7272458904546376e+256,
        2.5563239178728637e+258, 3.8089226376305671e+260, 5.7133839564458505e+262,
        8.6272097742332346e+264, 1.3113358856834518e+267, 2.0063439050956811e+269,
        3.0897696138473489e+271, 4.7891429014633912e+273, 7.47106292628289e+275,
        1.1729568794264138e+278, 1.8532718694937338e+280, 2.9467022724950369e+282,
        4.714723635992059e+284, 7.5907050539472148e+286, 1.2296942187394488e+289,
        2.0044015765453015e+291, 3.2872185855342945e+293, 5.423910666131586e+295,
        9.0036917057784329e+297, 1.5036165148649983e+300, 2.5260757449731969e+302,
        4.2690680090047027e+304, 7.257415615307994e+306
    };
    return cache;
}

}  // namespace detail

inline double function(int n) {
    if (n < 0) throw std::out_of_range("factorial::function: n must be non-negative");
    if (n < 171) return detail::factorial_cache()[n];
    return std::numeric_limits<double>::infinity();
}

inline double log_factorial(int n) {
    if (n < 0) throw std::out_of_range("factorial::log_factorial: n must be non-negative");
    if (n <= 0) return 0.0;
    if (n < 171) return std::log(detail::factorial_cache()[n]);
    return corehydro::numerics::math::special::log_gamma(static_cast<double>(n) + 1.0);
}

inline double binomial_coefficient(int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0.0;
    return std::floor(0.5 + std::exp(log_factorial(n) - log_factorial(k) - log_factorial(n - k)));
}

namespace detail {

// Helper for find_combinations(): recursively fills `buffer` with each increasing-index
// m-subset of {begin, ..., end-1}, appending each completed subset to `out`. Mirrors C#
// Factorial.FindCombosRecursive.
inline void find_combos_recursive(std::vector<int>& buffer, int done, int begin, int end,
                                   std::vector<std::vector<int>>& out) {
    for (int i = begin; i < end; ++i) {
        buffer[static_cast<std::size_t>(done)] = i;
        if (done == static_cast<int>(buffer.size()) - 1)
            out.push_back(buffer);
        else
            find_combos_recursive(buffer, done + 1, i + 1, end, out);
    }
}

}  // namespace detail

// Finds each m-combination within n (0-indexed), in increasing-index order. Mirrors C#
// Factorial.FindCombinations. Only called with m >= 1 by all_combinations() below (the
// C# source's own only caller); m <= 0 returns no combinations (defensive, not exercised
// by the C# source since it never calls FindCombinations(0, ...)).
inline std::vector<std::vector<int>> find_combinations(int m, int n) {
    std::vector<std::vector<int>> out;
    if (m <= 0) return out;
    std::vector<int> buffer(static_cast<std::size_t>(m));
    detail::find_combos_recursive(buffer, 0, 0, n, out);
    return out;
}

// Finds all combinations of m within n without replacement, represented as 0/1 indicator
// rows grouped by increasing subset size (1, 2, ..., n), each group in the increasing-
// index order find_combinations() produces. Mirrors C# Factorial.AllCombinations
// verbatim, including the n >= 31 overflow guard (2^n would exceed Int32 range).
inline std::vector<std::vector<int>> all_combinations(int n) {
    if (n < 0) throw std::out_of_range("factorial::all_combinations: n must be non-negative");
    if (n >= 31)
        throw std::out_of_range(
            "factorial::all_combinations: n is too large; number of combinations exceeds Int32 range");
    int f = (1 << n) - 1;
    std::vector<std::vector<int>> output(static_cast<std::size_t>(f),
                                          std::vector<int>(static_cast<std::size_t>(n), 0));
    int t = 0;
    for (int i = 1; i <= n; ++i) {
        for (const auto& combo : find_combinations(i, n)) {
            for (int col : combo) output[static_cast<std::size_t>(t)][static_cast<std::size_t>(col)] = 1;
            ++t;
        }
    }
    return output;
}

namespace detail {

inline bool next_combination_unchecked(std::vector<int>& combination, int n) {
    const int k = static_cast<int>(combination.size());
    int i = k - 1;
    while (i >= 0 && combination[static_cast<std::size_t>(i)] == n - k + i) --i;
    if (i < 0) return false;
    ++combination[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < k; ++j) {
        combination[static_cast<std::size_t>(j)] =
            combination[static_cast<std::size_t>(j - 1)] + 1;
    }
    return true;
}

}  // namespace detail

inline bool next_combination(std::vector<int>& combination, int n) {
    if (n < 0) throw std::out_of_range("factorial::next_combination: n must be non-negative");
    if (combination.empty() || combination.size() > static_cast<std::size_t>(n)) {
        throw std::invalid_argument(
            "factorial::next_combination: combination length must be between one and n");
    }
    int previous = -1;
    for (int index : combination) {
        if (index <= previous || index >= n) {
            throw std::invalid_argument(
                "factorial::next_combination: indexes must be strictly increasing and within [0, n)");
        }
        previous = index;
    }
    return detail::next_combination_unchecked(combination, n);
}

// Materializes the lazy C# enumeration while preserving its subset-size then lexicographic order.
inline std::vector<std::vector<int>> all_combinations_lazy(int n) {
    if (n < 0) throw std::out_of_range("factorial::all_combinations_lazy: n must be non-negative");
    std::vector<std::vector<int>> output;
    for (int k = 1; k <= n; ++k) {
        std::vector<int> combination(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) combination[static_cast<std::size_t>(j)] = j;
        do {
            output.push_back(combination);
        } while (detail::next_combination_unchecked(combination, n));
    }
    return output;
}

}  // namespace corehydro::numerics::math::special::factorial
