// ported from: Numerics/Mathematics/Linear Algebra/Support/Vector.cs @ 7e8e8d1
//
// Minimal dense vector: storage, length, indexed access, array conversion, and the
// arithmetic subset the MCMC samplers use (DEMCzs's SnookerUpdate and HMC's leapfrog in
// Numerics/Sampling/MCMC/{DEMCzs,HMC}.cs) -- vector+vector, vector-vector, elementwise
// vector*vector, vector*scalar, `sum`/`norm`/`norm_squared` (the last needed internally
// by `project`, mirroring C#'s `B.NormSquared()` call), and the static `distance`/
// `dot_product`/`project` helpers. `norm_squared()` is ported alongside `norm()` even
// though no sampler calls it directly, because `project()` cannot be transcribed
// faithfully without it (C# `Project` calls `B.NormSquared()`, a distinct public member
// from `Norm()`).
//
// Omitted (not needed by any ported caller): the `Header` UI-metadata property, `Array`
// (raw-array reference property; `to_array()`/`ToArray()`, a copy, is provided instead),
// `Clone`/`Clear`/`CopyFrom`/`ToList`, `Multiply`/`Add`/`Subtract` overloads taking a
// raw `double[]` (only the `Vector`-`Vector` forms are used), `operator^` (power), and
// `operator*(Vector, Matrix)`. Also omitted: `operator*(double, Vector)` (scalar on the
// left) -- unlike `Matrix.cs`, `Vector.cs` has no such overload (only
// `operator*(Vector, double)`), so none is added here either, matching the C# source
// exactly. B10 adds `divide`/`operator/` (C# Vector.Divide, line 318 / operator/, line
// 371): Bulletin17CDistribution::MomentConditions normalizes its accumulated moment sum
// with `mean /= (double)n`.
#pragma once
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace corehydro::numerics::math::linalg {

class Vector {
   public:
    // Construct a new vector with specified length, zero-initialized.
    explicit Vector(int length) : data_(static_cast<std::size_t>(length), 0.0) {}

    // Construct a new vector with specified length and fill it with a constant value.
    Vector(int length, double fill) : data_(static_cast<std::size_t>(length), fill) {}

    // Construct a new vector based on an initial array.
    explicit Vector(std::vector<double> initial_array) : data_(std::move(initial_array)) {}

    // The length of the vector.
    int length() const { return static_cast<int>(data_.size()); }

    // Get/set the element at the specific index.
    double operator[](int index) const { return data_[static_cast<std::size_t>(index)]; }
    double& operator[](int index) { return data_[static_cast<std::size_t>(index)]; }

    // Returns the vector as an array (copy).
    std::vector<double> to_array() const { return data_; }

    // Returns the sum of the vector.
    double sum() const {
        double s = 0.0;
        for (double v : data_) s += v;
        return s;
    }

    // Returns the vector norm.
    double norm() const { return std::sqrt(norm_squared()); }

    // Returns the vector norm squared.
    double norm_squared() const {
        double d = 0.0;
        for (double v : data_) d += v * v;
        return d;
    }

    // Add the vector.
    Vector add(const Vector& other) const {
        if (length() != other.length()) throw std::invalid_argument("Vectors must be the same length.");
        std::vector<double> result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) result[i] = data_[i] + other.data_[i];
        return Vector(std::move(result));
    }

    // Subtract the vector.
    Vector subtract(const Vector& other) const {
        if (length() != other.length()) throw std::invalid_argument("Vectors must be the same length.");
        std::vector<double> result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) result[i] = data_[i] - other.data_[i];
        return Vector(std::move(result));
    }

    // Multiply by a vector, elementwise. Vectors A and B must be the same size.
    Vector multiply(const Vector& other) const {
        if (length() != other.length()) throw std::invalid_argument("The vectors must be the same length.");
        std::vector<double> result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) result[i] = data_[i] * other.data_[i];
        return Vector(std::move(result));
    }

    // Multiply by a scalar.
    Vector multiply(double scalar) const {
        std::vector<double> result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) result[i] = data_[i] * scalar;
        return Vector(std::move(result));
    }

    // Divide by a scalar (C# Vector.Divide, line 318; B10).
    Vector divide(double scalar) const {
        std::vector<double> result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) result[i] = data_[i] / scalar;
        return Vector(std::move(result));
    }

    // Returns the Euclidean distance between two vectors ||x - y||.
    static double distance(const Vector& a, const Vector& b) {
        if (a.length() != b.length()) throw std::invalid_argument("The vectors must be the same length.");
        double d = 0.0;
        for (int i = 0; i < a.length(); ++i) {
            double dx = a[i] - b[i];
            d += dx * dx;
        }
        return std::sqrt(d);
    }

    // Returns the dot product of two vectors.
    static double dot_product(const Vector& a, const Vector& b) {
        if (a.length() != b.length()) throw std::invalid_argument("The vectors must be the same length.");
        double sum = 0.0;
        for (int i = 0; i < a.length(); ++i) sum += a[i] * b[i];
        return sum;
    }

    // Project vector A onto B.
    static Vector project(const Vector& a, const Vector& b) {
        double ab = dot_product(a, b);
        double bb = b.norm_squared();
        return b.multiply(ab / bb);
    }

   private:
    std::vector<double> data_;
};

// Adds vectors A and B by summing corresponding elements. Vectors A and B must be the same size.
inline Vector operator+(const Vector& a, const Vector& b) { return a.add(b); }

// Subtracts vectors A and B by subtracting corresponding elements. Vectors A and B must be the same size.
inline Vector operator-(const Vector& a, const Vector& b) { return a.subtract(b); }

// Multiplies vectors A and B by multiplying corresponding elements. Vectors A and B must be the same size.
inline Vector operator*(const Vector& a, const Vector& b) { return a.multiply(b); }

// Multiplies a vector A with a scalar.
inline Vector operator*(const Vector& a, double scalar) { return a.multiply(scalar); }

// Divides a vector A by a scalar (C# operator/, line 371; B10).
inline Vector operator/(const Vector& a, double scalar) { return a.divide(scalar); }

}  // namespace corehydro::numerics::math::linalg
