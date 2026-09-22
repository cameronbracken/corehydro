// ported from: Numerics/Sampling/SobolSequence.cs @ 7e8e8d1
//
// Sobol quasi-random low-discrepancy sequence. Faithful C++17 port of the C#
// SobolSequence (itself derived from Apache Commons Math). Bit-exact with the C#
// source when given the same new-joe-kuo-6 direction-number file. The file is
// loaded from a path passed at construction time so the header stays self-contained
// with no embedded resource dependency.
//
// Divergence note: the C# version embeds the direction data as a compiled resource
// (Properties.Resources.new_joe_kuo_6). The C++ port instead takes a filesystem
// path to the same file; callers must supply it (R via system.file("extdata", ...),
// Python via importlib.resources.files("corehydropy") / "data" / ...).
#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::sampling {

class SobolSequence {
   public:
    // Number of bits used for the direction vectors and scaling.
    static constexpr int kBits = 52;
    // Scaling factor: 2^kBits.
    static constexpr double kScale = 4503599627370496.0;  // 2^52
    // Maximum supported spatial dimension (matches the direction-number file).
    static constexpr int kMaxDimension = 21201;

    // Construct a Sobol sequence.
    //
    // dimension: spatial dimension in [1, kMaxDimension].
    // path: path to the new-joe-kuo-6.21201 direction-numbers file.
    //       Required when dimension > 1; may be empty when dimension == 1.
    explicit SobolSequence(int dimension = 1, const std::string& path = "")
        : dimension_(dimension)
        , direction_(dimension, std::vector<std::int64_t>(kBits + 1, 0LL))
        , x_(dimension, 0LL)
        , shift_(dimension, 0LL)
        , count_(0)
    {
        if (dimension < 1 || dimension > kMaxDimension) {
            throw std::invalid_argument(
                "The dimension must be between 1 and " + std::to_string(kMaxDimension));
        }
        initialize(path);
    }

    SobolSequence(int dimension, const std::string& path, int seed)
        : SobolSequence(dimension, path) {
        seed_ = seed;
        MersenneTwister prng(static_cast<std::uint32_t>(seed));
        scramble_directions(prng);
    }

    // Spatial dimension passed at construction.
    int dimension() const { return dimension_; }
    std::optional<int> seed() const { return seed_; }

    // Returns the next point in the sequence as a vector of length dimension_.
    // Each component is in [0, 1).
    std::vector<double> next_double() {
        std::vector<double> v(dimension_, 0.0);
        if (count_ == 0) {
            count_++;
            // Mirror C# behaviour: increment count but do NOT skip the first point
            // (the commented-out `//return v;` is intentional — the first point is
            // computed the same way as all subsequent ones).
        }

        // Find index c of the rightmost 0 bit in (count_ - 1).
        int c = 1;
        int value = count_ - 1;
        while ((value & 1) == 1) {
            value >>= 1;
            c++;
        }

        for (int i = 0; i < dimension_; ++i) {
            x_[i] ^= direction_[i][c];
            v[i] = static_cast<double>(x_[i] ^ shift_[i]) / kScale;
        }
        count_++;
        return v;
    }

    // Skip to a specific index in the sequence and return the point there.
    // Uses the Gray-code trick for O(kBits * dimension) reconstruction.
    std::vector<double> skip_to(int index) {
        if (index == 0) {
            // Reset internal state.
            std::fill(x_.begin(), x_.end(), 0LL);
        } else {
            int i = index - 1;
            // Gray code of i: i XOR floor(i / 2).
            std::int64_t gray_code = static_cast<std::int64_t>(i) ^
                                     static_cast<std::int64_t>(i >> 1);
            for (int j = 0; j < dimension_; ++j) {
                std::int64_t result = 0;
                for (int k = 1; k <= kBits; ++k) {
                    std::int64_t shift = gray_code >> (k - 1);
                    if (shift == 0) break;  // all remaining bits are zero
                    std::int64_t ik = shift & 1;
                    result ^= ik * direction_[j][k];
                }
                x_[j] = result;
            }
        }
        count_ = index;
        return next_double();
    }

   private:
    int dimension_;
    // direction_[d][i]: i-th direction number for dimension d (1-indexed in i).
    std::vector<std::vector<std::int64_t>> direction_;
    // Current state vector.
    std::vector<std::int64_t> x_;
    std::vector<std::int64_t> shift_;
    std::optional<int> seed_;
    // Sequence counter (0 before the first call to next_double()).
    int count_;

    static bool parity(std::int64_t value) {
        value ^= value >> 32;
        value ^= value >> 16;
        value ^= value >> 8;
        value ^= value >> 4;
        value ^= value >> 2;
        value ^= value >> 1;
        return (value & 1LL) != 0;
    }

    static std::int64_t apply_linear_scramble(
        const std::vector<std::int64_t>& row_masks, std::int64_t vector) {
        std::int64_t result = 0;
        for (int i = 1; i <= kBits; ++i)
            if (parity(row_masks[static_cast<std::size_t>(i - 1)] & vector))
                result |= std::int64_t(1) << (kBits - i);
        return result;
    }

    void scramble_directions(MersenneTwister& prng) {
        std::vector<std::int64_t> row_masks(kBits);
        for (int d = 0; d < dimension_; ++d) {
            for (int i = 1; i <= kBits; ++i) {
                std::int64_t mask = std::int64_t(1) << (kBits - i);
                for (int j = 1; j < i; ++j)
                    if (prng.next(2) == 1) mask |= std::int64_t(1) << (kBits - j);
                row_masks[static_cast<std::size_t>(i - 1)] = mask;
            }
            for (int c = 1; c <= kBits; ++c)
                direction_[d][c] = apply_linear_scramble(row_masks, direction_[d][c]);
            std::int64_t shift = 0;
            for (int i = 1; i <= kBits; ++i)
                if (prng.next(2) == 1) shift |= std::int64_t(1) << (kBits - i);
            shift_[d] = shift;
        }
    }

    void initialize(const std::string& path) {
        // Dimension 1 (index 0): unit initialization — no file needed.
        for (int i = 1; i <= kBits; ++i) {
            direction_[0][i] = std::int64_t(1) << (kBits - i);
        }

        if (dimension_ == 1) return;

        if (path.empty()) {
            throw std::invalid_argument(
                "A path to the direction-numbers file is required for dimension > 1");
        }

        std::ifstream reader(path);
        if (!reader.is_open()) {
            throw std::runtime_error("Cannot open direction-numbers file: " + path);
        }

        // Skip the header line.
        std::string header;
        std::getline(reader, header);

        int index = 1;
        std::string line;
        while (std::getline(reader, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            int dim;
            if (!(ss >> dim)) continue;

            if (dim >= 2 && dim <= dimension_) {
                int s = 0, a = 0;
                ss >> s >> a;
                std::vector<int> m(s + 1, 0);
                for (int j = 1; j <= s; ++j) ss >> m[j];
                init_direction_vector(index++, a, m);
            }

            if (dim > dimension_) break;
        }
    }

    // Populate direction_[d][*] from the primitive polynomial degree s, coefficient a,
    // and initial direction numbers m[1..s].  Mirrors C# initDirectionVector exactly.
    void init_direction_vector(int d, int a, const std::vector<int>& m) {
        int s = static_cast<int>(m.size()) - 1;
        for (int i = 1; i <= s; ++i) {
            direction_[d][i] = static_cast<std::int64_t>(m[i]) << (kBits - i);
        }
        for (int i = s + 1; i <= kBits; ++i) {
            direction_[d][i] =
                direction_[d][i - s] ^ (direction_[d][i - s] >> s);
            for (int k = 1; k <= s - 1; ++k) {
                direction_[d][i] ^=
                    static_cast<std::int64_t>((a >> (s - 1 - k)) & 1) *
                    direction_[d][i - k];
            }
        }
    }
};

}  // namespace corehydro::numerics::sampling
