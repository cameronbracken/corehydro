// ported from: Numerics/Distributions/Bivariate Copulas/Base/BivariateCopulaEstimation.cs @ 7e8e8d1
//
// Free-function port of the C# `static class BivariateCopulaEstimation`: a single public
// `estimate(copula, x, y, method)` (mirrors `Estimate(ref BivariateCopula, ...)`) dispatches
// to the three private methods below. `ref BivariateCopula` becomes a plain `BivariateCopula&`
// -- the C# body only ever mutates the referenced object in place (SetCopulaParameters,
// MarginalDistributionX/Y assignment); it never reassigns the `ref` parameter to a whole new
// object, so a C++ reference reproduces every call site unchanged.
//
// Supports copulas with one or more parameters: BrentSearch (1D) for single-parameter
// copulas (all Archimedeans in Phase 2), NelderMead for multi-parameter copulas (StudentT,
// a later task). Bayesian estimation (the C# remarks mention ARWMH) is NOT ported -- no
// Phase 2 caller needs it and MCMC is out of scope for this phase.
#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/copulas/base/bivariate_copula.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_estimation_method.hpp"
#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"

namespace corehydro::numerics::distributions::copulas {

namespace detail {

// The maximum pseudo likelihood method. Automatically selects BrentSearch for
// single-parameter copulas or NelderMead for multi-parameter copulas.
// sample_data_x/y should be the plotting positions of the data when estimating with
// pseudo likelihood.
inline void mpl(BivariateCopula& copula, const std::vector<double>& sample_data_x,
                 const std::vector<double>& sample_data_y) {
    auto constraints = copula.parameter_constraints(sample_data_x, sample_data_y);
    int n_params = copula.number_of_copula_parameters();

    if (n_params == 1) {
        math::optimization::BrentSearch::Objective func = [&](double x) {
            auto c = copula.clone();
            c->set_copula_parameters({x});
            return c->pseudo_log_likelihood(sample_data_x, sample_data_y);
        };
        math::optimization::BrentSearch brent(func, constraints[0][0], constraints[0][1]);
        brent.maximize();
        copula.set_copula_parameters({brent.best_parameter()});
    } else {
        auto initials = copula.get_copula_parameters();
        std::vector<double> lowers(static_cast<std::size_t>(n_params)),
            uppers(static_cast<std::size_t>(n_params));
        for (int i = 0; i < n_params; ++i) {
            lowers[static_cast<std::size_t>(i)] = constraints[static_cast<std::size_t>(i)][0];
            uppers[static_cast<std::size_t>(i)] = constraints[static_cast<std::size_t>(i)][1];
            // Clamp initials to be within bounds.
            initials[static_cast<std::size_t>(i)] =
                std::max(lowers[static_cast<std::size_t>(i)],
                         std::min(uppers[static_cast<std::size_t>(i)], initials[static_cast<std::size_t>(i)]));
        }

        math::optimization::NelderMead::Objective func = [&](const std::vector<double>& x) {
            auto c = copula.clone();
            c->set_copula_parameters(x);
            return c->pseudo_log_likelihood(sample_data_x, sample_data_y);
        };

        math::optimization::NelderMead solver(func, n_params, initials, lowers, uppers);
        solver.maximize();
        copula.set_copula_parameters(solver.best_parameters());
    }
}

// The inference from margins method. Automatically selects BrentSearch for
// single-parameter copulas or NelderMead for multi-parameter copulas.
inline void ifm(BivariateCopula& copula, const std::vector<double>& sample_data_x,
                 const std::vector<double>& sample_data_y) {
    auto constraints = copula.parameter_constraints(sample_data_x, sample_data_y);
    int n_params = copula.number_of_copula_parameters();

    if (n_params == 1) {
        math::optimization::BrentSearch::Objective func = [&](double x) {
            auto c = copula.clone();
            c->set_copula_parameters({x});
            return c->ifm_log_likelihood(sample_data_x, sample_data_y);
        };
        math::optimization::BrentSearch brent(func, constraints[0][0], constraints[0][1]);
        brent.maximize();
        copula.set_copula_parameters({brent.best_parameter()});
    } else {
        auto initials = copula.get_copula_parameters();
        std::vector<double> lowers(static_cast<std::size_t>(n_params)),
            uppers(static_cast<std::size_t>(n_params));
        for (int i = 0; i < n_params; ++i) {
            lowers[static_cast<std::size_t>(i)] = constraints[static_cast<std::size_t>(i)][0];
            uppers[static_cast<std::size_t>(i)] = constraints[static_cast<std::size_t>(i)][1];
            initials[static_cast<std::size_t>(i)] =
                std::max(lowers[static_cast<std::size_t>(i)],
                         std::min(uppers[static_cast<std::size_t>(i)], initials[static_cast<std::size_t>(i)]));
        }

        math::optimization::NelderMead::Objective func = [&](const std::vector<double>& x) {
            auto c = copula.clone();
            c->set_copula_parameters(x);
            return c->ifm_log_likelihood(sample_data_x, sample_data_y);
        };

        math::optimization::NelderMead solver(func, n_params, initials, lowers, uppers);
        solver.maximize();
        copula.set_copula_parameters(solver.best_parameters());
    }
}

// The maximum likelihood estimation method. Jointly estimates copula parameters and
// marginal distribution parameters using NelderMead optimization.
inline void mle(BivariateCopula& copula, const std::vector<double>& sample_data_x,
                 const std::vector<double>& sample_data_y) {
    // See if marginals are estimable.
    auto* margin1 = dynamic_cast<corehydro::numerics::distributions::IMaximumLikelihoodEstimation*>(
        copula.marginal_distribution_x.get());
    auto* margin2 = dynamic_cast<corehydro::numerics::distributions::IMaximumLikelihoodEstimation*>(
        copula.marginal_distribution_y.get());
    if (margin1 == nullptr || margin2 == nullptr)
        throw std::out_of_range(
            "The marginal distributions must implement the IMaximumLikelihoodEstimation "
            "interface to use this method.");

    int n_copula = copula.number_of_copula_parameters();
    int np1 = copula.marginal_distribution_x->number_of_parameters();
    int np2 = copula.marginal_distribution_y->number_of_parameters();
    int total_params = n_copula + np1 + np2;

    std::vector<double> initials(static_cast<std::size_t>(total_params)),
        lowers(static_cast<std::size_t>(total_params)), uppers(static_cast<std::size_t>(total_params));

    // Get ranks and plotting positions for initial MPL estimate.
    auto rank1 = corehydro::numerics::data::ranks_in_place(sample_data_x);
    auto rank2 = corehydro::numerics::data::ranks_in_place(sample_data_y);
    for (std::size_t i = 0; i < rank1.size(); ++i) {
        rank1[i] = rank1[i] / (static_cast<double>(rank1.size()) + 1.0);
        rank2[i] = rank2[i] / (static_cast<double>(rank2.size()) + 1.0);
    }

    // Get copula parameter constraints and initial estimates via MPL.
    auto copula_constraints = copula.parameter_constraints(sample_data_x, sample_data_y);
    mpl(copula, rank1, rank2);
    auto copula_params = copula.get_copula_parameters();
    for (int i = 0; i < n_copula; ++i) {
        initials[static_cast<std::size_t>(i)] = copula_params[static_cast<std::size_t>(i)];
        lowers[static_cast<std::size_t>(i)] = copula_constraints[static_cast<std::size_t>(i)][0];
        uppers[static_cast<std::size_t>(i)] = copula_constraints[static_cast<std::size_t>(i)][1];
    }

    // Estimate marginals.
    auto* est1 = dynamic_cast<corehydro::numerics::distributions::IEstimation*>(
        copula.marginal_distribution_x.get());
    auto* est2 = dynamic_cast<corehydro::numerics::distributions::IEstimation*>(
        copula.marginal_distribution_y.get());
    est1->estimate(sample_data_x, corehydro::numerics::distributions::ParameterEstimationMethod::MaximumLikelihood);
    est2->estimate(sample_data_y, corehydro::numerics::distributions::ParameterEstimationMethod::MaximumLikelihood);

    std::vector<double> con_initials, con_lowers, con_uppers;
    margin1->get_parameter_constraints(sample_data_x, con_initials, con_lowers, con_uppers);
    auto parms = copula.marginal_distribution_x->get_parameters();
    for (int i = 0; i < np1; ++i) {
        initials[static_cast<std::size_t>(n_copula + i)] = parms[static_cast<std::size_t>(i)];
        lowers[static_cast<std::size_t>(n_copula + i)] = con_lowers[static_cast<std::size_t>(i)];
        uppers[static_cast<std::size_t>(n_copula + i)] = con_uppers[static_cast<std::size_t>(i)];
    }
    margin2->get_parameter_constraints(sample_data_y, con_initials, con_lowers, con_uppers);
    parms = copula.marginal_distribution_y->get_parameters();
    for (int i = 0; i < np2; ++i) {
        initials[static_cast<std::size_t>(n_copula + np1 + i)] = parms[static_cast<std::size_t>(i)];
        lowers[static_cast<std::size_t>(n_copula + np1 + i)] = con_lowers[static_cast<std::size_t>(i)];
        uppers[static_cast<std::size_t>(n_copula + np1 + i)] = con_uppers[static_cast<std::size_t>(i)];
    }

    // Log-likelihood function.
    math::optimization::NelderMead::Objective log_lh = [&](const std::vector<double>& x) {
        // Set copula parameters.
        auto c = copula.clone();
        std::vector<double> copula_vals(x.begin(), x.begin() + n_copula);
        c->set_copula_parameters(copula_vals);

        // Marginal 1.
        auto m1 = copula.marginal_distribution_x->clone();
        std::vector<double> p1(x.begin() + n_copula, x.begin() + n_copula + np1);
        m1->set_parameters(p1);

        // Marginal 2.
        auto m2 = copula.marginal_distribution_y->clone();
        std::vector<double> p2(x.begin() + n_copula + np1, x.begin() + n_copula + np1 + np2);
        m2->set_parameters(p2);

        c->marginal_distribution_x = std::move(m1);
        c->marginal_distribution_y = std::move(m2);
        return c->log_likelihood(sample_data_x, sample_data_y);
    };

    math::optimization::NelderMead solver(log_lh, total_params, initials, lowers, uppers);
    solver.maximize();
    const auto& best = solver.best_parameters();

    // Set copula parameters.
    std::vector<double> best_copula(best.begin(), best.begin() + n_copula);
    copula.set_copula_parameters(best_copula);

    // Set marginal 1 parameters.
    std::vector<double> par1(best.begin() + n_copula, best.begin() + n_copula + np1);
    copula.marginal_distribution_x->set_parameters(par1);

    // Set marginal 2 parameters.
    std::vector<double> par2(best.begin() + n_copula + np1, best.begin() + n_copula + np1 + np2);
    copula.marginal_distribution_y->set_parameters(par2);
}

}  // namespace detail

// Estimate the bivariate copula.
inline void estimate(BivariateCopula& copula, const std::vector<double>& sample_data_x,
                      const std::vector<double>& sample_data_y, CopulaEstimationMethod estimation_method) {
    if (copula.number_of_copula_parameters() == 0) return;
    switch (estimation_method) {
        case CopulaEstimationMethod::PseudoLikelihood:
            detail::mpl(copula, sample_data_x, sample_data_y);
            break;
        case CopulaEstimationMethod::InferenceFromMargins:
            detail::ifm(copula, sample_data_x, sample_data_y);
            break;
        case CopulaEstimationMethod::FullLikelihood:
            detail::mle(copula, sample_data_x, sample_data_y);
            break;
    }
}

}  // namespace corehydro::numerics::distributions::copulas
