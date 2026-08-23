// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `linalg` group's dispatch arms over QRDecomposition and gauss_jordan_solve
// (numerics/math/linalg/{qr_decomposition,gauss_jordan_elimination}.hpp). Matrices travel
// as ONE flattened row-major data vector with `rows`/`cols` in `options` -- the same
// convention `interpolation.hpp`'s `bilinear` arm uses for its flattened `y` -- rather than
// as a nested array, keeping the toolbox grammar (flat double vectors + a small options
// object) uniform across every group. `to_matrix` is this group's local reassembly helper.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/linalg/gauss_jordan_elimination.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/qr_decomposition.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

// Reassembles a row-major flattened matrix from a data vector plus its declared shape.
inline math::linalg::Matrix to_matrix(const std::string& method, const std::vector<double>& flat,
                                      int rows, int cols) {
    if (static_cast<int>(flat.size()) != rows * cols)
        throw std::runtime_error("toolbox method 'linalg." + method + "' matrix data holds " +
                                 std::to_string(flat.size()) + " values, expected " +
                                 std::to_string(rows) + " x " + std::to_string(cols));
    return math::linalg::Matrix(rows, cols, flat);
}

// Serializes a matrix back to a ToolboxResult, row-major, with `dims = {rows, cols}`.
inline ToolboxResult matrix_result(const math::linalg::Matrix& m) {
    ToolboxResult r;
    r.dims = {m.number_of_rows(), m.number_of_columns()};
    r.values.reserve(static_cast<std::size_t>(m.number_of_rows() * m.number_of_columns()));
    for (int i = 0; i < m.number_of_rows(); ++i)
        for (int j = 0; j < m.number_of_columns(); ++j) r.values.push_back(m(i, j));
    return r;
}

// QRDecomposition (qr_q/qr_r/qr_solve/qr_solve_matrix) and gauss_jordan_solve
// (gauss_jordan_inverse/gauss_jordan_solution).
inline ToolboxResult run_linalg(const std::string& method,
                                const std::vector<std::vector<double>>& data,
                                const JsonValue& options) {
    if (!options.contains("rows") || !options.contains("cols"))
        throw std::runtime_error("toolbox method 'linalg." + method +
                                 "' needs 'rows' and 'cols' options for matrix A");
    int rows = options.at("rows").as_int();
    int cols = options.at("cols").as_int();
    const std::vector<double>& a_flat = data_at(data, 0, "linalg", method);
    math::linalg::Matrix A = to_matrix(method, a_flat, rows, cols);

    if (method == "qr_q") {
        math::linalg::QRDecomposition qr(A);
        return matrix_result(qr.q());
    }

    if (method == "qr_r") {
        math::linalg::QRDecomposition qr(A);
        return matrix_result(qr.r());
    }

    if (method == "qr_solve") {
        const std::vector<double>& b = data_at(data, 1, "linalg", method);
        math::linalg::QRDecomposition qr(A);
        math::linalg::Vector x = qr.solve(math::linalg::Vector(b));
        ToolboxResult r;
        r.values = x.to_array();
        return r;
    }

    if (method == "qr_solve_matrix") {
        if (!options.contains("b_cols"))
            throw std::runtime_error("toolbox method 'linalg.qr_solve_matrix' needs a 'b_cols' option");
        int b_cols = options.at("b_cols").as_int();
        const std::vector<double>& b_flat = data_at(data, 1, "linalg", method);
        math::linalg::Matrix B = to_matrix(method, b_flat, rows, b_cols);
        math::linalg::QRDecomposition qr(A);
        return matrix_result(qr.solve(B));
    }

    if (method == "gauss_jordan_inverse" || method == "gauss_jordan_solution") {
        if (!options.contains("b_cols"))
            throw std::runtime_error("toolbox method 'linalg." + method + "' needs a 'b_cols' option");
        int b_cols = options.at("b_cols").as_int();
        const std::vector<double>& b_flat = data_at(data, 1, "linalg", method);
        math::linalg::Matrix B = to_matrix(method, b_flat, rows, b_cols);
        math::linalg::gauss_jordan_solve(A, B);  // in-place: A -> inverse, B -> solution set
        return matrix_result(method == "gauss_jordan_inverse" ? A : B);
    }

    throw std::runtime_error("unknown linalg method: " + method);
}

}  // namespace corehydro::numerics::support::detail
