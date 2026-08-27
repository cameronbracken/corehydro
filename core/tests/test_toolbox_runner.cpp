// ctest for the shared toolbox runner: dispatch, option parsing, and error messages.
#include <cmath>
#include <string>
#include <vector>

#include "corehydro/numerics/support/toolbox_runner.hpp"
#include "check.hpp"

namespace tb = corehydro::numerics::support;

int main() {
    const std::vector<double> x{14.0, 8.0, 32.0, 7.0, 3.0, 15.0};
    const std::vector<double> y{10.0, 5.0, 7.0, 4.0, 3.0, 8.0};

    auto pearson = tb::run_toolbox("correlation", "pearson", {x, y}, "{}");
    CHECK_EQ(pearson.values.size(), std::size_t{1});
    CHECK_NEAR(pearson.values[0], 0.54502739907793, 1e-12);

    auto spearman = tb::run_toolbox("correlation", "spearman", {x, y}, "{}");
    CHECK_NEAR(spearman.values[0], 0.771428571428571, 1e-12);

    auto tau = tb::run_toolbox("correlation", "kendall", {x, y}, "{}");
    CHECK_NEAR(tau.values[0], 0.6, 1e-12);

    // An unknown group names the group; an unknown method names the method.
    bool threw_group = false;
    try {
        tb::run_toolbox("nope", "pearson", {x, y}, "{}");
    } catch (const std::exception& e) {
        threw_group = std::string(e.what()).find("nope") != std::string::npos;
    }
    CHECK_TRUE(threw_group);

    bool threw_method = false;
    try {
        tb::run_toolbox("correlation", "nope", {x, y}, "{}");
    } catch (const std::exception& e) {
        threw_method = std::string(e.what()).find("nope") != std::string::npos;
    }
    CHECK_TRUE(threw_method);

    // Too few data vectors is an error, not a crash.
    bool threw_arity = false;
    try {
        tb::run_toolbox("correlation", "pearson", {x}, "{}");
    } catch (const std::exception&) {
        threw_arity = true;
    }
    CHECK_TRUE(threw_arity);

    // gof: the named set and the individual metric agree, and the set is labelled.
    const std::vector<double> obs{2.0, 4.0, 6.0, 8.0, 10.0};
    const std::vector<double> mod{2.2, 3.9, 6.4, 7.5, 10.1};
    auto set = tb::run_toolbox("gof", "metrics", {obs, mod}, "{}");
    CHECK_EQ(set.values.size(), set.names.size());
    CHECK_EQ(set.values.size(), std::size_t{17});
    auto one = tb::run_toolbox("gof", "nse", {obs, mod}, "{}");
    std::size_t nse_at = 0;
    for (std::size_t i = 0; i < set.names.size(); ++i)
        if (set.names[i] == "nse") nse_at = i;
    CHECK_NEAR(set.values[nse_at], one.values[0], 0.0);

    // aic takes its arguments from options, not from a data vector.
    auto aic = tb::run_toolbox("gof", "aic", {}, "{\"k\":2,\"log_likelihood\":-121.01131220612}");
    CHECK_NEAR(aic.values[0], 246.02262441224, 1e-9);

    // The distribution-backed tests build their model from a dist spec in the options.
    auto ks = tb::run_toolbox("gof", "ks", {obs},
                              "{\"model\":{\"family\":\"Normal\",\"parameters\":[6.0,3.0]}}");
    CHECK_TRUE(ks.values[0] > 0.0 && ks.values[0] < 1.0);

    // --- statistics group ----------------------------------------------------------------
    const std::vector<double> series{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};

    auto summary = tb::run_toolbox("statistics", "summary", {series}, "{}");
    CHECK_EQ(summary.values.size(), summary.names.size());
    CHECK_EQ(summary.values.size(), std::size_t{13});
    std::size_t mean_at = 0, n_at = 0;
    for (std::size_t i = 0; i < summary.names.size(); ++i) {
        if (summary.names[i] == "mean") mean_at = i;
        if (summary.names[i] == "n") n_at = i;
    }
    CHECK_NEAR(summary.values[mean_at], 5.0, 1e-12);
    CHECK_EQ(static_cast<int>(summary.values[n_at]), 8);

    // A chunked "running" call seeded from a prior "summary" call's state matches one shot.
    auto part1 = tb::run_toolbox("statistics", "summary", {{2.0, 4.0, 4.0, 4.0}}, "{}");
    const std::vector<std::string> state_keys{"m1", "m2", "m3", "m4", "minimum", "maximum"};
    std::string state = "{\"state\":{";
    for (const auto& key : state_keys) {
        std::size_t at = 0;
        for (std::size_t i = 0; i < part1.names.size(); ++i)
            if (part1.names[i] == key) at = i;
        state += "\"" + key + "\":" + std::to_string(part1.values[at]) + ",";
    }
    state += "\"n\":" + std::to_string(static_cast<int>(part1.values[n_at])) + "}}";
    auto part2 = tb::run_toolbox("statistics", "running", {{5.0, 5.0, 7.0, 9.0}}, state);
    std::size_t p2_mean_at = 0;
    for (std::size_t i = 0; i < part2.names.size(); ++i)
        if (part2.names[i] == "mean") p2_mean_at = i;
    CHECK_NEAR(part2.values[p2_mean_at], summary.values[mean_at], 1e-9);

    auto pm = tb::run_toolbox("statistics", "product_moments", {obs}, "{}");
    CHECK_EQ(pm.values.size(), std::size_t{4});
    CHECK_EQ(pm.names.size(), std::size_t{4});

    auto lm = tb::run_toolbox("statistics", "l_moments", {obs}, "{}");
    CHECK_EQ(lm.values.size(), std::size_t{4});

    // Tie handling: the two 1.0 entries (indices 1 and 3) split rank 1.5; the lone 2.0 (index 2)
    // gets rank 3; the lone 3.0 (index 0) gets rank 4.
    auto ranks = tb::run_toolbox("statistics", "ranks", {{3.0, 1.0, 2.0, 1.0}}, "{}");
    CHECK_NEAR(ranks.values[0], 4.0, 0.0);
    CHECK_NEAR(ranks.values[1], 1.5, 0.0);
    CHECK_NEAR(ranks.values[2], 3.0, 0.0);
    CHECK_NEAR(ranks.values[3], 1.5, 0.0);

    auto pct = tb::run_toolbox("statistics", "percentile", {obs, {0.0, 0.5, 1.0}}, "{}");
    CHECK_EQ(pct.values.size(), std::size_t{3});
    CHECK_NEAR(pct.values[0], 2.0, 1e-12);   // min
    CHECK_NEAR(pct.values[2], 10.0, 1e-12);  // max

    // running_covariance: five identical pushes of the same 5-element vector, one per variable.
    // The oracle value for this exact case lives in fixtures/special_functions/
    // running_covariance.json; this ctest instead checks relationships that hold by
    // construction and would catch a wiring bug (wrong axis, wrong block offset) independent of
    // any pinned literal. The mean must equal the input's own mean computed here from first
    // principles. For the covariance block: since every one of the five variables is the
    // identical vector, the data contribution to Cov(i, j) is the same scalar for every pair
    // (i, j) -- so every diagonal entry must equal every other diagonal entry, and every
    // off-diagonal entry must equal every other off-diagonal entry. The two differ by exactly 1:
    // running_covariance_matrix.hpp seeds the accumulator's covariance at the identity matrix
    // before the first push, which adds 1 to the diagonal and 0 off it, and nothing after that
    // touches the two groups differently.
    std::vector<double> col{1.0, 2.0, 3.0, 4.0, 5.0};
    double expected_mean = 0.0;
    for (double v : col) expected_mean += v;
    expected_mean /= static_cast<double>(col.size());

    auto rc = tb::run_toolbox("statistics", "running_covariance", {col, col, col, col, col}, "{}");
    CHECK_EQ(static_cast<int>(rc.values[0]), 5);   // n
    CHECK_EQ(rc.dims.size(), std::size_t{2});
    CHECK_EQ(rc.dims[0], 5);
    CHECK_EQ(rc.dims[1], 5);
    // mean block starts at index 1
    for (int i = 0; i < 5; ++i)
        CHECK_NEAR(rc.values[static_cast<std::size_t>(1 + i)], expected_mean, 1e-12);
    // covariance block starts at index 1+5=6, 25 entries (5x5, row-major)
    double diag_value = rc.values[6];      // (0,0)
    double offdiag_value = rc.values[7];   // (0,1)
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::size_t idx = static_cast<std::size_t>(6 + i * 5 + j);
            CHECK_NEAR(rc.values[idx], i == j ? diag_value : offdiag_value, 1e-9);
        }
    }
    CHECK_NEAR(diag_value - offdiag_value, 1.0, 1e-9);

    // --- spectra group ---------------------------------------------------------------------
    const std::vector<double> ac_series{5.0, 6.0, 4.0, 7.0, 3.0, 8.0, 2.0, 9.0, 1.0, 10.0};

    auto acf_corr = tb::run_toolbox("spectra", "autocorrelation", {ac_series}, "{\"lag_max\":3}");
    CHECK_EQ(acf_corr.dims.size(), std::size_t{2});
    CHECK_EQ(acf_corr.dims[0], 4);
    CHECK_EQ(acf_corr.dims[1], 2);
    CHECK_NEAR(acf_corr.values[1], 1.0, 1e-12);  // lag 0 correlation is 1 by construction

    auto acf_cov = tb::run_toolbox("spectra", "autocorrelation", {ac_series},
                                   "{\"lag_max\":3,\"type\":\"covariance\"}");
    auto acf_partial = tb::run_toolbox("spectra", "autocorrelation", {ac_series},
                                       "{\"lag_max\":3,\"type\":\"partial\"}");
    CHECK_EQ(acf_partial.dims[0], 3);  // lag_max, not lag_max + 1
    CHECK_NEAR(acf_partial.values[1], acf_corr.values[3], 1e-9);  // PACF at lag 1 == ACF at lag 1

    auto ci = tb::run_toolbox("spectra", "autocorrelation_ci", {}, "{\"sample_size\":100}");
    CHECK_EQ(ci.values.size(), std::size_t{2});
    CHECK_NEAR(ci.values[0] + ci.values[1], 0.0, 1e-12);

    const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> b{4.0, 3.0, 2.0, 1.0};
    auto xcorr = tb::run_toolbox("spectra", "cross_correlation", {a, b}, "{}");
    CHECK_EQ(xcorr.values.size(), std::size_t{4});

    auto fwd = tb::run_toolbox("spectra", "dft", {{1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0}}, "{}");
    CHECK_EQ(fwd.values.size(), std::size_t{8});
    auto real_fwd = tb::run_toolbox("spectra", "dft_real", {a}, "{}");
    CHECK_EQ(real_fwd.values.size(), std::size_t{4});

    // --- histogram group ---------------------------------------------------------------------
    // Rice-rule bin count, and the frequencies sum to the sample size.
    const std::vector<double> h{1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 8.0, 9.0};
    auto bins = tb::run_toolbox("histogram", "bins", {h}, "{}");
    CHECK_EQ(bins.dims.size(), std::size_t{2});
    CHECK_EQ(bins.dims[1], 4);
    double total = 0.0;
    for (int row = 0; row < bins.dims[0]; ++row)
        total += bins.values[static_cast<std::size_t>(row * 4 + 3)];
    CHECK_NEAR(total, 10.0, 0.0);

    // --- interpolation group -------------------------------------------------------------------
    // a point on a knot returns that knot's y exactly.
    const std::vector<double> ix{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> iy{10.0, 20.0, 30.0, 40.0};
    auto lin = tb::run_toolbox("interpolation", "linear", {ix, iy, {2.5, 3.0}}, "{}");
    CHECK_EQ(lin.values.size(), std::size_t{2});
    CHECK_NEAR(lin.values[0], 25.0, 1e-12);
    CHECK_NEAR(lin.values[1], 30.0, 1e-12);

    // P4 whole-branch-review finding M2: "log" and "logarithmic" are synonyms for
    // Transform::Logarithmic in EVERY group that reads a "*_transform" option, not just the
    // group each host verb historically used ("log" for interpolate()/interpolate_2d(),
    // "logarithmic" for curve_interpolate()/tabular_function()). Both tokens must parse
    // identically through BOTH the "interpolation" group (which only ever accepted "log" before
    // this fix) and the "paired_data" group (which only ever accepted "logarithmic").
    {
        std::string opts_log = "{\"x_transform\":\"log\",\"y_transform\":\"none\"}";
        std::string opts_logarithmic = "{\"x_transform\":\"logarithmic\",\"y_transform\":\"none\"}";
        auto a = tb::run_toolbox("interpolation", "linear", {ix, iy, {2.5, 3.0}}, opts_log);
        auto b = tb::run_toolbox("interpolation", "linear", {ix, iy, {2.5, 3.0}}, opts_logarithmic);
        CHECK_EQ(a.values.size(), b.values.size());
        for (std::size_t i = 0; i < a.values.size(); ++i) CHECK_NEAR(a.values[i], b.values[i], 0.0);

        const std::vector<double> px{50.0, 100.0, 150.0, 200.0, 250.0};
        const std::vector<double> py{100.0, 200.0, 300.0, 400.0, 500.0};
        std::string popts_log = "{\"x_transform\":\"log\",\"y_transform\":\"none\"}";
        std::string popts_logarithmic = "{\"x_transform\":\"logarithmic\",\"y_transform\":\"none\"}";
        auto pa = tb::run_toolbox("paired_data", "interpolate_y", {px, py, {75.0}}, popts_log);
        auto pb = tb::run_toolbox("paired_data", "interpolate_y", {px, py, {75.0}}, popts_logarithmic);
        CHECK_EQ(pa.values.size(), pb.values.size());
        for (std::size_t i = 0; i < pa.values.size(); ++i) CHECK_NEAR(pa.values[i], pb.values[i], 0.0);
    }

    // --- regression group ------------------------------------------------------------------
    // A two-predictor model whose exact solution is known by construction: y = 3 + 2*x1 - x2
    // with no noise, so the fitted coefficients must recover [3, 2, -1] exactly and R^2 == 1.
    const std::vector<double> rx1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const std::vector<double> rx2{2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0};
    std::vector<double> rflat;
    std::vector<double> ry;
    for (std::size_t i = 0; i < rx1.size(); ++i) {
        rflat.push_back(rx1[i]);
        rflat.push_back(rx2[i]);
        ry.push_back(3.0 + 2.0 * rx1[i] - rx2[i]);
    }
    std::string ropts = "{\"rows\":8,\"columns\":2,\"intercept\":true}";
    auto rfit = tb::run_toolbox("regression", "fit", {rflat, ry}, ropts);
    // values: beta_1, beta_2, beta_3, se_1, se_2, se_3, r_squared, adj_r_squared, sigma, df, n
    CHECK_EQ(rfit.values.size(), std::size_t{11});
    CHECK_NEAR(rfit.values[0], 3.0, 1e-9);   // intercept
    CHECK_NEAR(rfit.values[1], 2.0, 1e-9);   // x1 coefficient
    CHECK_NEAR(rfit.values[2], -1.0, 1e-9);  // x2 coefficient
    std::size_t r2_at = 0;
    for (std::size_t i = 0; i < rfit.names.size(); ++i)
        if (rfit.names[i] == "r_squared") r2_at = i;
    CHECK_NEAR(rfit.values[r2_at], 1.0, 1e-12);

    auto rcov = tb::run_toolbox("regression", "covariance", {rflat, ry}, ropts);
    CHECK_EQ(rcov.dims.size(), std::size_t{2});
    CHECK_EQ(rcov.dims[0], 3);
    CHECK_EQ(rcov.dims[1], 3);

    auto rres = tb::run_toolbox("regression", "residuals", {rflat, ry}, ropts);
    CHECK_EQ(rres.values.size(), std::size_t{8});
    for (double v : rres.values) CHECK_NEAR(v, 0.0, 1e-9);  // no noise: exact fit

    std::string ropts_pred = "{\"rows\":8,\"columns\":2,\"intercept\":true,\"predict_rows\":1}";
    auto rpred = tb::run_toolbox("regression", "predict", {rflat, ry, {1.0, 2.0}}, ropts_pred);
    CHECK_EQ(rpred.values.size(), std::size_t{1});
    CHECK_NEAR(rpred.values[0], 3.0, 1e-9);  // 3 + 2*1 - 2 = 3

    auto rpi = tb::run_toolbox("regression", "prediction_intervals", {rflat, ry, {1.0, 2.0}}, ropts_pred);
    CHECK_EQ(rpi.dims.size(), std::size_t{2});
    CHECK_EQ(rpi.dims[0], 1);
    CHECK_EQ(rpi.dims[1], 3);
    CHECK_NEAR(rpi.values[2], 3.0, 1e-9);  // mean column matches predict()

    // --- sampling group ----------------------------------------------------------------------
    // Structural checks only -- the oracle-pinned Sobol/stratify values (from
    // Test_SobolSequence.cs/Test_Stratification.cs) live in fixtures/toolbox/sampling.json and
    // are exercised cross-language by the generic fixture runner, not re-pinned here. Dimension
    // 1 needs no direction-numbers file (SobolSequence's unit initialization), so "path" is the
    // empty string throughout -- run_sampling's "sobol" arm still requires the key to be
    // present, just unused when dimension == 1.
    auto sob = tb::run_toolbox("sampling", "sobol", {},
                               "{\"dimension\":1,\"n\":10,\"skip\":0,\"path\":\"\"}");
    CHECK_EQ(sob.dims.size(), std::size_t{2});
    CHECK_EQ(sob.dims[0], 10);
    CHECK_EQ(sob.dims[1], 1);
    CHECK_EQ(sob.values.size(), std::size_t{10});
    for (double v : sob.values) CHECK_TRUE(v >= 0.0 && v < 1.0);

    // skip moves the stream: point 1 with skip = 1 equals point 2 (0-based index 1) with skip = 0.
    auto sob_seq = tb::run_toolbox("sampling", "sobol", {}, "{\"dimension\":1,\"n\":2,\"path\":\"\"}");
    auto sob_skip = tb::run_toolbox("sampling", "sobol", {},
                                    "{\"dimension\":1,\"n\":1,\"skip\":1,\"path\":\"\"}");
    CHECK_NEAR(sob_skip.values[0], sob_seq.values[1], 0.0);

    // stratify: on a [0, 1] axis (non-probability), bin weights (defaulting to bin width) sum
    // to exactly the axis length, i.e. 1.
    auto strat = tb::run_toolbox("sampling", "stratify", {}, "{\"lower\":0,\"upper\":1,\"bins\":10}");
    CHECK_EQ(strat.dims.size(), std::size_t{2});
    CHECK_EQ(strat.dims[0], 10);
    CHECK_EQ(strat.dims[1], 4);
    double weight_sum = 0.0;
    for (int row = 0; row < strat.dims[0]; ++row)
        weight_sum += strat.values[static_cast<std::size_t>(row * 4 + 3)];
    CHECK_NEAR(weight_sum, 1.0, 1e-12);

    // probability = true always yields zero bins (Stratify::XValues's documented early return).
    auto strat_prob = tb::run_toolbox("sampling", "stratify", {},
                                      "{\"lower\":0,\"upper\":1,\"bins\":10,\"probability\":true}");
    CHECK_EQ(strat_prob.dims[0], 0);

    // --- probability group --------------------------------------------------------------------
    // Hand-computable arithmetic (independent_joint_probability is a product, positive is a
    // min) -- the upstream-pinned oracle cases (Test_Probability.cs) live in
    // fixtures/toolbox/joint_probability.json.
    auto jp_ind = tb::run_toolbox("probability", "joint", {{0.5, 0.5}}, "{\"dependency\":\"independent\"}");
    CHECK_NEAR(jp_ind.values[0], 0.25, 1e-12);
    auto jp_pos = tb::run_toolbox("probability", "joint", {{0.5, 0.5}}, "{\"dependency\":\"positive\"}");
    CHECK_NEAR(jp_pos.values[0], 0.5, 1e-12);

    // dependency = "correlation" with no indicator vector: the runner mirrors C#'s
    // JointProbability fallthrough (returns NaN, does not throw -- see probability.hpp).
    // Rejecting this combination outright is a wrapper concern, covered by the R/Python wrapper
    // tests, not the runner.
    auto jp_no_indicators =
        tb::run_toolbox("probability", "joint", {{0.5, 0.5}}, "{\"dependency\":\"correlation\"}");
    CHECK_TRUE(std::isnan(jp_no_indicators.values[0]));

    // --- link group ---------------------------------------------------------------------------
    // Round-trip identity (inverse_link(link(x)) == x to 1e-12) and a finite derivative, for
    // every one of the twelve link types the toolbox exposes (the seven Numerics links, with
    // YeoJohnson counted once, plus the five BestFit-specific links) at a value inside its own
    // domain. The upstream-pinned known-value oracles (Test_LinkFunctions.cs and friends) live
    // in fixtures/toolbox/link_functions.json and are exercised cross-language by the generic
    // fixture runner, not re-pinned here.
    struct LinkCase {
        const char* name;
        const char* options;
        double x;
    };
    const LinkCase link_cases[] = {
        {"Identity", "{\"link\":{\"type\":\"Identity\"}}", 5.0},
        {"Log", "{\"link\":{\"type\":\"Log\"}}", 2.0},
        {"Logit", "{\"link\":{\"type\":\"Logit\"}}", 0.3},
        {"Probit", "{\"link\":{\"type\":\"Probit\"}}", 0.4},
        {"ComplementaryLogLog", "{\"link\":{\"type\":\"ComplementaryLogLog\"}}", 0.4},
        {"FisherZ", "{\"link\":{\"type\":\"FisherZ\"}}", 0.3},
        {"YeoJohnson", "{\"link\":{\"type\":\"YeoJohnson\",\"parameters\":{\"lambda\":0.5}}}", 2.0},
        {"ASinH", "{\"link\":{\"type\":\"ASinH\",\"parameters\":{\"gamma0\":0.5,\"scale\":0.3}}}", 1.0},
        {"SES", "{\"link\":{\"type\":\"SES\",\"parameters\":{\"a\":1.0}}}", 1.0},
        {"LogSES", "{\"link\":{\"type\":\"LogSES\",\"parameters\":{\"sigma0\":7.5}}}", 2.0},
        {"LogASinH", "{\"link\":{\"type\":\"LogASinH\",\"parameters\":{\"sigma0\":10.0,\"log_scale\":0.25}}}", 15.0},
        {"Centered",
         "{\"link\":{\"type\":\"Centered\",\"parameters\":{\"mu0\":100.0,\"scale\":20.0},"
         "\"inner\":{\"type\":\"Identity\"}}}",
         120.0},
    };
    for (const LinkCase& lc : link_cases) {
        auto eta = tb::run_toolbox("link", "link", {{lc.x}}, lc.options);
        CHECK_EQ(eta.values.size(), std::size_t{1});
        auto back = tb::run_toolbox("link", "inverse_link", {{eta.values[0]}}, lc.options);
        CHECK_NEAR(back.values[0], lc.x, 1e-12);
        auto d = tb::run_toolbox("link", "d_link", {{lc.x}}, lc.options);
        CHECK_TRUE(std::isfinite(d.values[0]));
    }

    // "names" is the single source of truth link_names() (R/Python) calls through to: exactly
    // the twelve types accepted above, and every one of them round-tripped in link_cases.
    auto link_type_list = tb::run_toolbox("link", "names", {}, "{}");
    CHECK_EQ(link_type_list.names.size(), std::size_t{12});
    for (const LinkCase& lc : link_cases) {
        bool found = false;
        for (const std::string& n : link_type_list.names)
            if (n == lc.name) found = true;
        CHECK_TRUE(found);
    }

    // A missing "link" spec names the group; an unknown link type names the type; an unknown
    // method names the method.
    CHECK_THROWS_MSG(tb::run_toolbox("link", "link", {{1.0}}, "{}"), "link");
    CHECK_THROWS_MSG(tb::run_toolbox("link", "link", {{1.0}}, "{\"link\":{\"type\":\"Nope\"}}"),
                     "Nope");
    CHECK_THROWS_MSG(
        tb::run_toolbox("link", "nope", {{1.0}}, "{\"link\":{\"type\":\"Identity\"}}"), "nope");

    // Centered without an "inner" spec names what is missing.
    CHECK_THROWS_MSG(tb::run_toolbox("link", "link", {{1.0}},
                                     "{\"link\":{\"type\":\"Centered\",\"parameters\":"
                                     "{\"mu0\":0.0}}}"),
                     "inner");

    // --- trend group ---------------------------------------------------------------------------
    // Every method builds a fresh trend from its type's own class defaults (all parameter
    // values 0, i.e. predict() is well-defined even with no explicit "values"), then applies
    // "start_index"/"values" from the spec. Upstream-pinned oracles (the ten
    // Test_*TrendTests.cs files) live in fixtures/toolbox/trend_functions.json.
    auto lin_trend = tb::run_toolbox(
        "trend", "predict", {{1950.0, 1960.0, 1940.0}},
        "{\"trend\":{\"type\":\"Linear\",\"start_index\":1950,\"values\":[100.0,0.5]}}");
    CHECK_EQ(lin_trend.values.size(), std::size_t{3});
    CHECK_NEAR(lin_trend.values[0], 100.0, 1e-10);
    CHECK_NEAR(lin_trend.values[1], 105.0, 1e-10);
    CHECK_NEAR(lin_trend.values[2], 95.0, 1e-10);

    // A bare Constant trend (no explicit values) predicts its class-default 0 everywhere.
    auto def = tb::run_toolbox("trend", "predict", {{0.0, 100.0}}, "{\"trend\":{\"type\":\"Constant\"}}");
    CHECK_NEAR(def.values[0], 0.0, 1e-12);
    CHECK_NEAR(def.values[1], 0.0, 1e-12);

    // parameters(): names and values both come back, matching the C# ModelParameter naming
    // ("(α)" == "(α)", ConstantTrendTests.Test_Constructor_EmptyConstructor_CreatesDefaultModel).
    auto params = tb::run_toolbox("trend", "parameters", {}, "{\"trend\":{\"type\":\"Constant\"}}");
    CHECK_EQ(params.values.size(), std::size_t{1});
    CHECK_EQ(params.names.size(), std::size_t{1});
    CHECK_EQ(params.names[0], std::string("(\xCE\xB1)"));
    CHECK_NEAR(params.values[0], 0.0, 1e-12);

    // GeneralLinear falls through to ConstantTrend, mirroring the C# SetTrendModel if-chain
    // (see trend_model_factory.hpp): a single explicit value predicts as a constant.
    auto gl = tb::run_toolbox("trend", "predict", {{0.0, 50.0}},
                              "{\"trend\":{\"type\":\"GeneralLinear\",\"values\":[7.0]}}");
    CHECK_NEAR(gl.values[0], 7.0, 1e-12);
    CHECK_NEAR(gl.values[1], 7.0, 1e-12);

    // "names" is the single source of truth trend_names() (R/Python) calls through to.
    auto trend_type_list = tb::run_toolbox("trend", "names", {}, "{}");
    CHECK_EQ(trend_type_list.names.size(), std::size_t{11});
    bool has_general_linear = false;
    for (const std::string& n : trend_type_list.names)
        if (n == "GeneralLinear") has_general_linear = true;
    CHECK_TRUE(has_general_linear);

    // A missing "trend" spec, an unknown trend type, and an unknown method all name the thing
    // that's wrong.
    CHECK_THROWS_MSG(tb::run_toolbox("trend", "predict", {{0.0}}, "{}"), "trend");
    CHECK_THROWS_MSG(
        tb::run_toolbox("trend", "predict", {{0.0}}, "{\"trend\":{\"type\":\"Nope\"}}"), "Nope");
    CHECK_THROWS_MSG(
        tb::run_toolbox("trend", "nope", {{0.0}}, "{\"trend\":{\"type\":\"Constant\"}}"), "nope");

    return chtest::summary("toolbox_runner");
}
