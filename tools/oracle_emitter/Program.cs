// Oracle verifier: reproduce every fixture assertion against the upstream Numerics library.
//
// For each fixtures/*.json the same construct + assertions the C++/R/Python runners use are
// replayed against the real C# distribution, and the fixture's "expected" value is checked to
// its stated tolerance. This is the "run dotnet" half of the hybrid oracle workflow: the
// scraped/curated fixtures are confirmed reproducible by the source they were ported from.
//
// Usage:  dotnet run --project tools/oracle_emitter -- [fixtures-dir]
//   exits non-zero if any supported assertion fails to reproduce.
// GEV standard-error methods are reported as "skipped" (verified in Phase 0, not re-checked here).

using System.Text.Json;
using Numerics;
using Numerics.Data;
using Numerics.Data.Statistics;
using Numerics.Distributions;
using Numerics.Distributions.Copulas;
using Numerics.Mathematics;
using Numerics.Mathematics.LinearAlgebra;
using Numerics.Mathematics.Optimization;
using Numerics.Mathematics.SpecialFunctions;
using Numerics.Sampling;
using Numerics.Sampling.MCMC;
// Task 7: the link and trend function layers. Numerics.Functions holds the seven standard
// links (Identity/Log/Logit/Probit/ComplementaryLogLog/YeoJohnson/FisherZ, LinkFunctionFactory,
// LinkFunctionType); the five link-specific BestFit types and the eleven trend types are already
// compiled in via the `$(BF)/Models/**/*.cs` glob in OracleEmitter.csproj (Models/LinkFunctions/
// and Models/TrendFunctions/ are both under Models/), so only their namespaces are new here.
using Numerics.Functions;
using RMC.BestFit.Models.LinkFunctions;
using RMC.BestFit.Models.TrendFunctions;
using RMC.BestFit.Models.TrendFunctions.Support;
// Task T12: the real RMC.BestFit estimation path (subset-compiled -- see OracleEmitter.csproj).
// `RMC.BestFit.Models` holds the DataFrame series/data types (ExactSeries/ExactData/etc.) AND
// DataFrame + the UnivariateDistribution model; `RMC.BestFit.Estimation` holds MaximumLikelihood /
// MaximumAPosteriori / BayesianAnalysis / OptimizationMethod. v2.0.0 moved the DataFrame series
// classes out of the bare `RMC.BestFit` namespace and into `RMC.BestFit.Models` (upstream commit
// range fc28c0c..c2e6192, `namespace RMC.BestFit` -> `namespace RMC.BestFit.Models` across
// Models/DataFrame/**); a plain `using RMC.BestFit;` no longer resolves the unqualified series/data
// type names used below, so it's replaced by a plain `using RMC.BestFit.Models;` (verified: still no
// clash with Numerics). The `BestFitModels` alias is kept for the explicitly-qualified
// `BestFitModels.DataFrame` call sites already in this file.
using RMC.BestFit.Models;
using RMC.BestFit.Estimation;
using BestFitModels = RMC.BestFit.Models;
// Task A11: the real RMC.BestFit user-facing Analyses layer (subset-compiled -- see
// OracleEmitter.csproj). Aliased to avoid clashing with RMC.BestFit.Estimation.BayesianAnalysis.
using BestFitAnalyses = RMC.BestFit.Analyses;

static double ParseNum(JsonElement v)
{
    if (v.ValueKind == JsonValueKind.String)
    {
        return v.GetString() switch
        {
            "nan" => double.NaN,
            "inf" => double.PositiveInfinity,
            "-inf" => double.NegativeInfinity,
            var s => throw new Exception($"unexpected string value: {s}"),
        };
    }
    return v.GetDouble();
}

static ParameterEstimationMethod ParseMethod(string m) => m switch
{
    "mom" => ParameterEstimationMethod.MethodOfMoments,
    "lmom" => ParameterEstimationMethod.MethodOfLinearMoments,
    _ => ParameterEstimationMethod.MaximumLikelihood,
};

// Build a component distribution from {"target": "...", "params": [...]} (or "fit").
// Recursive: components can nest for future Mixture / CompetingRisks.
static UnivariateDistributionBase BuildComponent(JsonElement desc,
                                                  Dictionary<string, double[]> datasets)
{
    var compTarget = desc.GetProperty("target").GetString()!;
    var compType = Enum.Parse<UnivariateDistributionType>(compTarget);
    // v2.1.4: the C# factory switch now has an explicit VonMises case (previously it fell
    // through the old if/else chain to `new Deterministic()`), so the bypass this repo used to
    // need is retired -- see docs/upstream-csharp-issues.md and the T7 factory header note.
    UnivariateDistributionBase compDist = UnivariateDistributionFactory.CreateDistribution(compType);
    if (desc.TryGetProperty("params", out var compPs))
    {
        compDist.SetParameters(compPs.EnumerateArray().Select(ParseNum).ToArray());
        return compDist;
    }
    if (desc.TryGetProperty("fit", out var compFit))
    {
        var compData = datasets[compFit.GetProperty("dataset").GetString()!];
        if (compDist is IEstimation compEst)
            compEst.Estimate(compData, ParseMethod(compFit.GetProperty("method").GetString()!));
        else
            throw new Exception($"{compTarget} does not support estimation");
        return compDist;
    }
    throw new Exception($"BuildComponent: missing 'params' or 'fit' for {compTarget}");
}

// Build composite distributions from their structured construct schemas.
// Future composites (Mixture, CompetingRisks) add a case here.
static UnivariateDistributionBase BuildComposite(string target, JsonElement construct,
                                                  Dictionary<string, double[]> datasets)
{
    if (target == "TruncatedDistribution")
    {
        var baseDist = BuildComponent(construct.GetProperty("base"), datasets);
        var boundsArr = construct.GetProperty("bounds").EnumerateArray().ToArray();
        double lo = ParseNum(boundsArr[0]), hi = ParseNum(boundsArr[1]);
        return new TruncatedDistribution(baseDist, lo, hi);
    }
    if (target == "Empirical")
    {
        var xArr = construct.GetProperty("x").EnumerateArray().Select(ParseNum).ToArray();
        var pArr = construct.GetProperty("p").EnumerateArray().Select(ParseNum).ToArray();
        // v2.1.4: p_descending DECLARES the probability order via the 4-arg SetParameters
        // overload (SortOrder.Ascending/Descending) rather than the 2-arg overload's hardcoded
        // SortOrder.Ascending -- probabilityOrder is NOT auto-detected from the data (an
        // earlier corehydro auto-detect design reproduced a false positive this gate caught;
        // see empirical_distribution.hpp's header note). The two overloads are equivalent when
        // p_descending is false, so this always goes through the 4-arg one.
        bool pDescending = construct.TryGetProperty("p_descending", out var pdEl) && pdEl.GetBoolean();
        var emp = new EmpiricalDistribution(xArr, pArr, Numerics.Data.SortOrder.Ascending,
            pDescending ? Numerics.Data.SortOrder.Descending : Numerics.Data.SortOrder.Ascending);
        // Optional p_transform: "None" or "NormalZ" (default NormalZ)
        // Fully qualified: v2.0.0 moved the RMC.BestFit DataFrame series/data types into the
        // `RMC.BestFit.Models` namespace (see the `using RMC.BestFit.Models;` note above), which
        // now also owns a `Transform` enum (the TimeSeries None/Logarithmic/BoxCox/YeoJohnson
        // transform) that collides with this `Numerics.Data.Transform` (None/Logarithmic/NormalZ).
        if (construct.TryGetProperty("p_transform", out var pt))
        {
            emp.ProbabilityTransform = pt.GetString() switch
            {
                "None"    => Numerics.Data.Transform.None,
                "NormalZ" => Numerics.Data.Transform.NormalZ,
                var s     => throw new Exception($"unknown p_transform: {s}")
            };
        }
        return emp;
    }
    if (target == "KernelDensity")
    {
        var data = datasets[construct.GetProperty("data").GetString()!];
        var kernelStr = construct.TryGetProperty("kernel", out var k) ? k.GetString()! : "Gaussian";
        var kernelType = kernelStr switch
        {
            "Epanechnikov" => KernelDensity.KernelType.Epanechnikov,
            "Gaussian"     => KernelDensity.KernelType.Gaussian,
            "Triangular"   => KernelDensity.KernelType.Triangular,
            "Uniform"      => KernelDensity.KernelType.Uniform,
            var s          => throw new Exception($"unknown kernel type: {s}")
        };
        // ParseNum (not bw.GetDouble()) so a "nan"/"inf" string literal (the v2.1.4
        // Bandwidth NaN/Infinity-rejection case) parses instead of throwing.
        KernelDensity kde = construct.TryGetProperty("bandwidth", out var bw)
            ? new KernelDensity(data, kernelType, ParseNum(bw))
            : new KernelDensity(data, kernelType);
        if (construct.TryGetProperty("bounded_by_data", out var bd))
            kde.BoundedByData = bd.GetBoolean();
        return kde;
    }
    if (target == "Mixture")
    {
        var comps = construct.GetProperty("components").EnumerateArray()
            .Select(c => BuildComponent(c, datasets)).ToArray();
        var wts = construct.GetProperty("weights").EnumerateArray()
            .Select(e => e.GetDouble()).ToArray();
        var mix = new Mixture(wts, comps);
        // Optional zero-inflation (v2.1.4): "zero_inflated" (default false) / "zero_weight"
        // (default 0.0). IsZeroInflated set before ZeroWeight, matching Clone()'s object
        // initializer order and every other call site (mixture.hpp's header note) -- the
        // setters renormalize component weights as a side effect.
        mix.IsZeroInflated = construct.TryGetProperty("zero_inflated", out var ziEl)
            && ziEl.GetBoolean();
        mix.ZeroWeight = construct.TryGetProperty("zero_weight", out var zwEl)
            ? ParseNum(zwEl) : 0.0;
        return mix;
    }
    if (target == "CompetingRisks")
    {
        var comps = construct.GetProperty("components").EnumerateArray()
            .Select(c => BuildComponent(c, datasets)).ToArray();
        var cr = new CompetingRisks(comps);
        if (construct.TryGetProperty("minimum_of_random_variables", out var minOfRV))
            cr.MinimumOfRandomVariables = minOfRV.GetBoolean();
        if (construct.TryGetProperty("dependency", out var depEl))
        {
            cr.Dependency = depEl.GetString() switch
            {
                "Independent" => Probability.DependencyType.Independent,
                "PerfectlyPositive" => Probability.DependencyType.PerfectlyPositive,
                "PerfectlyNegative" => Probability.DependencyType.PerfectlyNegative,
                "CorrelationMatrix" => Probability.DependencyType.CorrelationMatrix,
                var s => throw new Exception($"unknown dependency type: {s}")
            };
        }
        if (construct.TryGetProperty("correlation", out var corrEl))
        {
            var rows = corrEl.EnumerateArray().ToArray();
            int n = rows.Length;
            var corr = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                var row = rows[i].EnumerateArray().ToArray();
                for (int j = 0; j < row.Length; j++) corr[i, j] = ParseNum(row[j]);
            }
            cr.CorrelationMatrix = corr;
        }
        return cr;
    }
    throw new Exception($"unknown composite target: {target}");
}

static UnivariateDistributionBase Build(string target, JsonElement construct,
                                        Dictionary<string, double[]> datasets)
{
    // Composite distributions use bespoke construction (no flat enum entry in C# or C++).
    if (target == "TruncatedDistribution" || target == "Empirical" || target == "KernelDensity"
        || target == "Mixture" || target == "CompetingRisks")
        return BuildComposite(target, construct, datasets);

    // Empirical is constructed from x/p arrays, not flat params -- handled above as composite.
    var type = Enum.Parse<UnivariateDistributionType>(target);
    // v2.1.4: the C# factory switch now has an explicit VonMises case (see BuildComponent's
    // matching note above), so this no longer needs to bypass the factory.
    UnivariateDistributionBase dist = UnivariateDistributionFactory.CreateDistribution(type);
    if (construct.TryGetProperty("params", out var ps))
    {
        var p = ps.EnumerateArray().Select(ParseNum).ToArray();
        dist.SetParameters(p);
        return dist;
    }
    var fit = construct.GetProperty("fit");
    var data = datasets[fit.GetProperty("dataset").GetString()!];
    if (dist is IEstimation est)
        est.Estimate(data, ParseMethod(fit.GetProperty("method").GetString()!));
    else
        throw new Exception($"{target} does not support estimation");
    return dist;
}

// Returns null when the method is intentionally not reproduced here (GEV standard errors).
static double? Dispatch(UnivariateDistributionBase d, string m, JsonElement[] a)
{
    switch (m)
    {
        // Mutates the already-built `d` in place with a new flat parameter vector -- lets a
        // case exercise a "construct valid -> SetParameters invalid -> recheck ->
        // SetParameters valid -> recheck" sequence on ONE persistent object (mirrors
        // test_fixtures.cpp's dispatch_generic; needed for TruncatedDistribution's
        // parameter-validity fixture). Returns a dummy value; pair with a
        // mode:"equal", expected:0 assertion.
        case "set_parameters": d.SetParameters(a.Select(ParseNum).ToArray()); return 0;
        // CompetingRisks-only: verifies the v2.1.4 Dependency setter fix (changing
        // Dependency mid-lifetime invalidates the cached MVN) and that PerfectlyNegative no
        // longer zeroes the public CorrelationMatrix. ONE self-contained call (mirrors
        // test_fixtures.cpp's dispatch_generic "dependency_change" branch -- works
        // identically whether a runner holds the persistent `d` across a case's
        // assertions or rebuilds fresh per dispatch, like R/Python): CDF under the CURRENT
        // dependency, read back CorrelationMatrix[i, j], switch to args[1], CDF again --
        // returns the value named by args[4] ("cdf1"/"correlation"/"cdf2").
        // args = [x, dependency2, i, j, field].
        case "dependency_change":
        {
            var cr = (CompetingRisks)d;
            double x = a[0].GetDouble();
            double cdf1 = cr.CDF(x);
            double corrIj = cr.CorrelationMatrix[a[2].GetInt32(), a[3].GetInt32()];
            cr.Dependency = a[1].GetString() switch
            {
                "Independent" => Probability.DependencyType.Independent,
                "PerfectlyPositive" => Probability.DependencyType.PerfectlyPositive,
                "PerfectlyNegative" => Probability.DependencyType.PerfectlyNegative,
                "CorrelationMatrix" => Probability.DependencyType.CorrelationMatrix,
                var s => throw new Exception($"unknown dependency type: {s}")
            };
            double cdf2 = cr.CDF(x);
            return a[4].GetString() switch
            {
                "cdf1" => cdf1,
                "correlation" => corrIj,
                "cdf2" => cdf2,
                var f => throw new Exception($"unknown dependency_change field: {f}")
            };
        }
        case "mean": return d.Mean;
        case "median": return d.Median;
        case "mode": return d.Mode;
        case "sd": return d.StandardDeviation;
        case "skewness": return d.Skewness;
        case "kurtosis": return d.Kurtosis;
        case "minimum": return d.Minimum;
        case "maximum": return d.Maximum;
        case "pdf": return d.PDF(a[0].GetDouble());
        case "log_pdf": return d.LogPDF(a[0].GetDouble());
        case "cdf": return d.CDF(a[0].GetDouble());
        case "quantile": return d.InverseCDF(a[0].GetDouble());
        case "param":
            if (a[0].ValueKind == JsonValueKind.String)
            {
                int idx = a[0].GetString() switch { "location" => 0, "scale" => 1, "shape" => 2, _ => -1 };
                return d.GetParameters[idx];
            }
            return d.GetParameters[a[0].GetInt32()];
        case "linear_moment":
            if (d is ILinearMomentEstimation lm)
                return lm.LinearMomentsFromParameters(d.GetParameters)[a[0].GetInt32()];
            throw new Exception("distribution has no L-moments");
        // args: [sample_size, seed, index] -- one draw from the seeded MT stream.
        case "random_value": return d.GenerateRandomValues(a[0].GetInt32(), a[1].GetInt32())[a[2].GetInt32()];
        // args: the whole sample, inline. Sums LogPDF over it (UnivariateDistributionBase.
        // LogLikelihood(IList<double>)).
        case "log_likelihood": return d.LogLikelihood(a.Select(ParseNum).ToArray());
        // Static GammaDistribution utility, not tied to `d`'s own parameters -- args:
        // [skewness, probability].
        case "partial_kp": return GammaDistribution.PartialKp(a[0].GetDouble(), a[1].GetDouble());
        // The IStandardError surface. GEV's four bespoke standard-error methods were validated
        // in Phase 0 against the bespoke C# calls and are not re-checked here -- they are the 11
        // documented skips. Every other implementer is dispatched for real, by the same
        // capability cast the C++, R and Python runners use, with the estimation method fixed at
        // MaximumLikelihood (matching dist_runner.hpp and test_fixtures.cpp's dispatch_generic).
        // Args follow the flattened fixture convention those runners already speak:
        // parameter_covariance [sample_size, row, col], quantile_variance [probability,
        // sample_size], quantile_gradient [probability, index], quantile_se [probability,
        // sample_size].
        case "quantile_gradient":
        case "parameter_covariance":
        case "quantile_variance":
        case "quantile_se":
        {
            if (d is GeneralizedExtremeValue || d is not IStandardError se) return null;
            var seMethod = ParameterEstimationMethod.MaximumLikelihood;
            return m switch
            {
                "parameter_covariance" =>
                    se.ParameterCovariance(a[0].GetInt32(), seMethod)[a[1].GetInt32(),
                                                                     a[2].GetInt32()],
                "quantile_variance" =>
                    se.QuantileVariance(a[0].GetDouble(), a[1].GetInt32(), seMethod),
                "quantile_gradient" => se.QuantileGradient(a[0].GetDouble())[a[1].GetInt32()],
                _ => Math.Sqrt(se.QuantileVariance(a[0].GetDouble(), a[1].GetInt32(), seMethod)),
            };
        }
        default:
            throw new Exception($"unknown fixture method: {m}");
    }
}

static bool Compare(double actual, JsonElement assertion)
{
    string mode = assertion.GetProperty("mode").GetString()!;
    var exp = assertion.GetProperty("expected");
    switch (mode)
    {
        case "equal":
            double e = ParseNum(exp);
            return double.IsNaN(e) ? double.IsNaN(actual) : actual == e;
        case "abs":
            return Math.Abs(actual - exp.GetDouble()) <= assertion.GetProperty("tol").GetDouble();
        case "rel":
            double r = exp.GetDouble();
            return Math.Abs(actual - r) / Math.Abs(r) <= assertion.GetProperty("tol").GetDouble();
        default:
            throw new Exception($"unknown comparison mode: {mode}");
    }
}

// Cholesky fixture args are a flattened row-major n*n matrix, with n inferred from the
// args length per the convention documented in fixtures/special_functions/cholesky.json.
static int CholeskySquareN(int len)
{
    int n = (int)Math.Round(Math.Sqrt(len));
    if (n * n != len) throw new Exception("Cholesky fixture args: length is not a perfect square");
    return n;
}

// solve_element args are [flattened n*n matrix, n-length rhs vector, index i], i.e.
// n*n + n + 1 == len; solve the quadratic for n.
static int CholeskySolveN(int len)
{
    int n = (int)Math.Round((-1.0 + Math.Sqrt(1.0 + 4.0 * (len - 1))) / 2.0);
    if (n * n + n + 1 != len) throw new Exception("Cholesky fixture args: length does not fit n*n+n+1");
    return n;
}

static Matrix CholeskyMatrixFromFlat(double[] a, int n)
{
    var m = new Matrix(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i, j] = a[i * n + j];
    return m;
}

// Correlation fixture args are [x..., y...] concatenated and split at the midpoint
// (equal-length samples) -- see fixtures/special_functions/correlation.json for the convention.
static (double[] X, double[] Y) CorrelationSplit(double[] a)
{
    int mid = a.Length / 2;
    return (a[..mid], a[mid..]);
}

// Special-function dispatch table: maps "Module.method" → Func<double[], double>.
static Func<double[], double>? ResolveSpecialFunction(string target) => target switch
{
    // Cholesky family (args: flattened row-major matrix, n inferred from length -- see
    // fixtures/special_functions/cholesky.json for the full convention)
    "Cholesky.determinant" => a =>
    {
        int n = CholeskySquareN(a.Length);
        return new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n)).Determinant();
    },
    "Cholesky.log_determinant" => a =>
    {
        int n = CholeskySquareN(a.Length);
        return new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n)).LogDeterminant();
    },
    "Cholesky.inverse_element" => a =>
    {
        int n = CholeskySquareN(a.Length - 2);
        var chol = new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n));
        int i = (int)a[n * n];
        int j = (int)a[n * n + 1];
        return chol.InverseA()[i, j];
    },
    "Cholesky.solve_element" => a =>
    {
        int n = CholeskySolveN(a.Length);
        var chol = new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n));
        var rhs = new double[n];
        Array.Copy(a, n * n, rhs, 0, n);
        int i = (int)a[n * n + n];
        return chol.Solve(new Vector(rhs))[i];
    },
    // Returns 1.0 if the matrix is positive-definite (construction succeeds), 0.0 if the
    // ctor throws. Upstream CholeskyDecomposition.cs throws a bare `new Exception(...)`
    // for the non-PD/NaN case (no dedicated exception type), so this must catch the
    // general `Exception`, unlike the C++ port's narrower `std::runtime_error` catch.
    "Cholesky.is_positive_definite" => a =>
    {
        int n = CholeskySquareN(a.Length);
        try
        {
            var chol = new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n));
            return chol.IsPositiveDefinite ? 1.0 : 0.0;
        }
        catch (Exception)
        {
            return 0.0;
        }
    },
    // Erf family
    "Erf.function"     => a => Erf.Function(a[0]),
    "Erf.erfc"         => a => Erf.Erfc(a[0]),
    "Erf.inverse_erf"  => a => Erf.InverseErf(a[0]),
    "Erf.inverse_erfc" => a => Erf.InverseErfc(a[0]),
    // Gamma family
    "Gamma.function"                 => a => Gamma.Function(a[0]),
    "Gamma.log_gamma"                => a => Gamma.LogGamma(a[0]),
    "Gamma.digamma"                  => a => Gamma.Digamma(a[0]),
    "Gamma.trigamma"                 => a => Gamma.Trigamma(a[0]),
    "Gamma.lower_incomplete"         => a => Gamma.LowerIncomplete(a[0], a[1]),
    "Gamma.upper_incomplete"         => a => Gamma.UpperIncomplete(a[0], a[1]),
    "Gamma.inverse_lower_incomplete" => a => Gamma.InverseLowerIncomplete(a[0], a[1]),
    "Gamma.inverse_upper_incomplete" => a => Gamma.InverseUpperIncomplete(a[0], a[1]),
    // Beta family
    "Beta.function"           => a => Beta.Function(a[0], a[1]),
    "Beta.incomplete"         => a => Beta.Incomplete(a[0], a[1], a[2]),
    "Beta.incbcf"             => a => Beta.Incbcf(a[0], a[1], a[2]),
    "Beta.incbd"              => a => Beta.Incbd(a[0], a[1], a[2]),
    "Beta.power_series"       => a => Beta.PowerSeries(a[0], a[1], a[2]),
    "Beta.incomplete_inverse" => a => Beta.IncompleteInverse(a[0], a[1], a[2]),
    // Factorial family
    "Factorial.function"             => a => Factorial.Function((int)a[0]),
    "Factorial.log_factorial"        => a => Factorial.LogFactorial((int)a[0]),
    "Factorial.binomial_coefficient" => a => Factorial.BinomialCoefficient((int)a[0], (int)a[1]),
    // Bessel family
    "Bessel.i0" => a => Bessel.I0(a[0]),
    "Bessel.i1" => a => Bessel.I1(a[0]),
    // Correlation family (args: [x..., y...], split at the midpoint -- see
    // fixtures/special_functions/correlation.json for the full convention)
    "Correlation.pearson" => a =>
    {
        var (x, y) = CorrelationSplit(a);
        return Correlation.Pearson(x, y);
    },
    "Correlation.spearman" => a =>
    {
        var (x, y) = CorrelationSplit(a);
        return Correlation.Spearman(x, y);
    },
    "Correlation.kendalls_tau" => a =>
    {
        var (x, y) = CorrelationSplit(a);
        return Correlation.KendallsTau(x, y);
    },
    // LU family (args: flattened row-major matrix, n inferred from length -- reuses the
    // Cholesky-fixture flatten helpers above, which are generic matrix-args conventions,
    // not Cholesky-specific; see fixtures/special_functions/lu_decomposition.json)
    "LU.determinant" => a =>
    {
        int n = CholeskySquareN(a.Length);
        return new LUDecomposition(CholeskyMatrixFromFlat(a, n)).Determinant();
    },
    "LU.inverse_element" => a =>
    {
        int n = CholeskySquareN(a.Length - 2);
        var lu = new LUDecomposition(CholeskyMatrixFromFlat(a, n));
        int i = (int)a[n * n];
        int j = (int)a[n * n + 1];
        return lu.InverseA()[i, j];
    },
    "LU.solve_element" => a =>
    {
        int n = CholeskySolveN(a.Length);
        var lu = new LUDecomposition(CholeskyMatrixFromFlat(a, n));
        var rhs = new double[n];
        Array.Copy(a, n * n, rhs, 0, n);
        int i = (int)a[n * n + n];
        return lu.Solve(new Vector(rhs))[i];
    },
    // Percentile (args: [data_1..data_n, k, data_is_sorted (0.0/1.0)] -- see
    // fixtures/special_functions/percentile.json for the convention)
    "Statistics.percentile" => a =>
    {
        int n = a.Length - 2;
        var data = a[..n];
        double k = a[n];
        bool sorted = a[n + 1] != 0.0;
        return Statistics.Percentile(data, k, sorted);
    },
    // Extensions/MersenneTwister ranged-draw family (see
    // fixtures/special_functions/extension_methods.json for the conventions)
    "Extensions.next_doubles_grid" => a =>
    {
        // args: [n, dim, seed, row, col]
        int n = (int)a[0];
        int dim = (int)a[1];
        var rng = new MersenneTwister((int)a[2]);
        int row = (int)a[3];
        int col = (int)a[4];
        var grid = rng.NextDoubles(n, dim);
        return grid[row, col];
    },
    "Extensions.next_integers_at" => a =>
    {
        // args: [n, seed, i]
        int n = (int)a[0];
        var rng = new MersenneTwister((int)a[1]);
        int i = (int)a[2];
        var values = rng.NextIntegers(n);
        return values[i];
    },
    "Mt.next_range" => a =>
    {
        // args: [seed, min, max, i] -- draws Next(min, max) (i+1) times, 0-based,
        // returning the i-th draw.
        var rng = new MersenneTwister((int)a[0]);
        int minV = (int)a[1];
        int maxV = (int)a[2];
        int i = (int)a[3];
        int result = 0;
        for (int k = 0; k <= i; k++) result = rng.Next(minV, maxV);
        return (double)result;
    },
    // RunningCovarianceMatrix family (args: [size, num_pushes, data_flat, trailing
    // index/indices] -- see fixtures/special_functions/running_covariance.json)
    "RunningCovariance.mean_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        return rcm.Mean[i, 0];
    },
    "RunningCovariance.covariance_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.Covariance[i, j];
    },
    "RunningCovariance.sample_covariance_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.SampleCovariance[i, j];
    },
    "RunningCovariance.sample_correlation_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.SampleCorrelation[i, j];
    },
    "RunningCovariance.population_covariance_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.PopulationCovariance[i, j];
    },
    "RunningCovariance.population_correlation_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.PopulationCorrelation[i, j];
    },
    // RunningStatistics family (args: the flat sample; see
    // fixtures/special_functions/running_statistics.json)
    "RunningStatistics.mean" => a => new RunningStatistics(a).Mean,
    "RunningStatistics.variance" => a => new RunningStatistics(a).Variance,
    "RunningStatistics.standard_deviation" => a => new RunningStatistics(a).StandardDeviation,
    "RunningStatistics.population_variance" => a => new RunningStatistics(a).PopulationVariance,
    "RunningStatistics.population_standard_deviation" => a => new RunningStatistics(a).PopulationStandardDeviation,
    "RunningStatistics.coefficient_of_variation" => a => new RunningStatistics(a).CoefficientOfVariation,
    "RunningStatistics.skewness" => a => new RunningStatistics(a).Skewness,
    "RunningStatistics.population_skewness" => a => new RunningStatistics(a).PopulationSkewness,
    "RunningStatistics.kurtosis" => a => new RunningStatistics(a).Kurtosis,
    "RunningStatistics.population_kurtosis" => a => new RunningStatistics(a).PopulationKurtosis,
    "RunningStatistics.minimum" => a => new RunningStatistics(a).Minimum,
    "RunningStatistics.maximum" => a => new RunningStatistics(a).Maximum,
    "RunningStatistics.count" => a => (double)new RunningStatistics(a).Count,
    // RunningStatistics combine family (args: [n1, sample1(n1), sample2(m)] -- see
    // RunningStatisticsCombined() above and fixtures/special_functions/running_statistics.json)
    "RunningStatistics.combined_minimum" => a => RunningStatisticsCombined(a).Minimum,
    "RunningStatistics.combined_maximum" => a => RunningStatisticsCombined(a).Maximum,
    "RunningStatistics.combined_mean" => a => RunningStatisticsCombined(a).Mean,
    "RunningStatistics.combined_variance" => a => RunningStatisticsCombined(a).Variance,
    "RunningStatistics.combined_standard_deviation" => a => RunningStatisticsCombined(a).StandardDeviation,
    "RunningStatistics.combined_coefficient_of_variation" => a => RunningStatisticsCombined(a).CoefficientOfVariation,
    "RunningStatistics.combined_skewness" => a => RunningStatisticsCombined(a).Skewness,
    "RunningStatistics.combined_kurtosis" => a => RunningStatisticsCombined(a).Kurtosis,
    "RunningStatistics.combined_count" => a => (double)RunningStatisticsCombined(a).Count,
    // RunningStatistics.clone_* family (args: the flat sample -- see RunningStatisticsClone()
    // below and fixtures/special_functions/running_statistics.json)
    "RunningStatistics.clone_mean" => a => RunningStatisticsClone(a).Mean,
    "RunningStatistics.clone_variance" => a => RunningStatisticsClone(a).Variance,
    "RunningStatistics.clone_skewness" => a => RunningStatisticsClone(a).Skewness,
    "RunningStatistics.clone_kurtosis" => a => RunningStatisticsClone(a).Kurtosis,
    "RunningStatistics.clone_minimum" => a => RunningStatisticsClone(a).Minimum,
    "RunningStatistics.clone_maximum" => a => RunningStatisticsClone(a).Maximum,
    "RunningStatistics.clone_count" => a => (double)RunningStatisticsClone(a).Count,
    // Fourier family (see Fourier*At() below for the args conventions -- mirrors
    // fourier_*_at() in core/tests/test_fixtures.cpp exactly)
    "Fourier.fft_at" => FourierFftAt,
    "Fourier.real_fft_at" => FourierRealFftAt,
    "Fourier.correlation_at" => FourierCorrelationAt,
    "Fourier.autocorrelation_at" => FourierAutocorrelationAt,
    // NumericalDerivative family (closed function registry; MUST match
    // numerical_derivative_{quadratic,normal_loglik} in core/tests/test_fixtures.cpp)
    "NumericalDerivative.gradient_element_quadratic" => a => NumericalDerivativeGradientElement(NumericalDerivativeQuadratic, a),
    "NumericalDerivative.gradient_element_normal_loglik" => a => NumericalDerivativeGradientElement(NumericalDerivativeNormalLoglik, a),
    "NumericalDerivative.hessian_element_quadratic" => a => NumericalDerivativeHessianElement(NumericalDerivativeQuadratic, a),
    "NumericalDerivative.hessian_element_normal_loglik" => a => NumericalDerivativeHessianElement(NumericalDerivativeNormalLoglik, a),
    // DifferentialEvolution family (closed function registry; REUSES
    // NumericalDerivativeQuadratic/NumericalDerivativeNormalLoglik above -- MUST match
    // differential_evolution_best_value() in core/tests/test_fixtures.cpp)
    "DifferentialEvolution.best_value" => DifferentialEvolutionBestValue,
    // Histogram family (args: [explicit_bins, data..., trailing probe?] -- see
    // HistogramBuild() below and fixtures/special_functions/histogram.json)
    "Histogram.number_of_bins" => a => (double)HistogramBuild(a, 0).NumberOfBins,
    "Histogram.bin_width" => a => HistogramBuild(a, 0).BinWidth,
    "Histogram.lower_bound" => a => HistogramBuild(a, 0).LowerBound,
    "Histogram.upper_bound" => a => HistogramBuild(a, 0).UpperBound,
    "Histogram.data_count" => a => (double)HistogramBuild(a, 0).DataCount,
    "Histogram.mean" => a => HistogramBuild(a, 0).Mean,
    "Histogram.median" => a => HistogramBuild(a, 0).Median,
    "Histogram.mode" => a => HistogramBuild(a, 0).Mode,
    "Histogram.standard_deviation" => a => HistogramBuild(a, 0).StandardDeviation,
    "Histogram.bin_lower_bound_at" => a => HistogramBuild(a, 1)[(int)a[^1]].LowerBound,
    "Histogram.bin_upper_bound_at" => a => HistogramBuild(a, 1)[(int)a[^1]].UpperBound,
    "Histogram.bin_frequency_at" => a => (double)HistogramBuild(a, 1)[(int)a[^1]].Frequency,
    "Histogram.get_bin_index_of" => a => (double)HistogramBuild(a, 1).GetBinIndexOf(a[^1]),
    // Histogram.adapt_* family (args: [explicit_bins, num_adds, data..., adds...] -- see
    // HistogramBuildAdapt() below and fixtures/special_functions/histogram.json)
    "Histogram.adapt_lower_bound" => a => HistogramBuildAdapt(a).LowerBound,
    "Histogram.adapt_upper_bound" => a => HistogramBuildAdapt(a).UpperBound,
    "Histogram.adapt_bin_first_lower_bound" => a => HistogramBuildAdapt(a)[0].LowerBound,
    "Histogram.adapt_bin_last_upper_bound" => a => { var h = HistogramBuildAdapt(a); return h[h.NumberOfBins - 1].UpperBound; },
    "Histogram.adapt_bin_first_frequency" => a => (double)HistogramBuildAdapt(a)[0].Frequency,
    "Histogram.adapt_bin_last_frequency" => a => { var h = HistogramBuildAdapt(a); return (double)h[h.NumberOfBins - 1].Frequency; },
    "Histogram.adapt_data_count" => a => (double)HistogramBuildAdapt(a).DataCount,
    // PlottingPositions family (args: [N, alpha, i] for function_at; [N, i] for
    // weibull_at -- see fixtures/special_functions/plotting_positions.json)
    "PlottingPositions.function_at" => a => PlottingPositions.Function((int)a[0], a[1])[(int)a[2]],
    "PlottingPositions.weibull_at" => a => PlottingPositions.Weibull((int)a[0])[(int)a[1]],
    // Search family (args: [values..., x, start] -- see fixtures/special_functions/search.json)
    "Search.sequential" => a => Search.Sequential(a[^2], a[..^2], (int)a[^1]),
    "Search.bisection" => a => Search.Bisection(a[^2], a[..^2], (int)a[^1]),
    // MCMCDiagnostics.MinimumSampleSize (args: [quantile, tolerance, probability] -- see
    // fixtures/special_functions/mcmc_diagnostics.json)
    "MCMCDiagnostics.minimum_sample_size" => a => MCMCDiagnostics.MinimumSampleSize(a[0], a[1], a[2]),
    // Search.*_descending family (args: [values..., x, start], same convention as
    // Search.sequential/bisection above but with SortOrder.Descending -- MUST match
    // core/tests/test_fixtures.cpp's Search.*_descending entries)
    "Search.sequential_descending" => a => Search.Sequential(a[^2], a[..^2], (int)a[^1], SortOrder.Descending),
    "Search.bisection_descending" => a => Search.Bisection(a[^2], a[..^2], (int)a[^1], SortOrder.Descending),
    // Bilinear.log_floor_value (args: [x1_query, x2_query] -- see BilinearLogFloorValue()
    // below and fixtures/special_functions/bilinear.json)
    "Bilinear.log_floor_value" => BilinearLogFloorValue,
    // Probability.hpcm_* family (args: see ProbabilityHpcmJoint()/
    // ProbabilityHpcmConditionalAt() below and fixtures/special_functions/probability.json)
    "Probability.hpcm_joint" => a => ProbabilityHpcmJoint(a),
    "Probability.hpcm_conditional_at" => ProbabilityHpcmConditionalAt,
    // Tools.log10 (args: [x] -- see fixtures/special_functions/tools.json)
    "Tools.log10" => a => Tools.Log10(a[0]),
    _ => null,
};

// RunningCovariance fixture args: [size, num_pushes, data_flat(num_pushes*size), trailing
// index/indices] -- see fixtures/special_functions/running_covariance.json for the
// convention. Builds a RunningCovarianceMatrix and replays `numPushes` Push()es of
// `size`-length rows sliced from the flattened data.
static RunningCovarianceMatrix RunningCovarianceBuild(double[] a, int size, int numPushes)
{
    var rcm = new RunningCovarianceMatrix(size);
    for (int p = 0; p < numPushes; p++)
    {
        var row = new double[size];
        Array.Copy(a, 2 + p * size, row, 0, size);
        rcm.Push(row);
    }
    return rcm;
}

// RunningStatistics combine fixture args: [n1, sample1(n1 values), sample2(remaining
// values)] -- a "split-index" convention, distinct from Correlation's equal-length
// two-halves split (Test_Combine/Test_Add split their 69-value sample into UNEQUAL 48/21
// sub-samples, so a fixed midpoint doesn't apply). See
// fixtures/special_functions/running_statistics.json for the full convention. Uses the
// `+` operator (rather than calling RunningStatistics.Combine directly), which exercises
// both -- operator+ is a one-line forwarder to Combine.
static RunningStatistics RunningStatisticsCombined(double[] a)
{
    int n1 = (int)a[0];
    var sample1 = a[1..(1 + n1)];
    var sample2 = a[(1 + n1)..];
    return new RunningStatistics(sample1) + new RunningStatistics(sample2);
}

// RunningStatistics.clone_* fixture args convention -- MUST mirror running_statistics_clone()
// in core/tests/test_fixtures.cpp: args = the flat sample.
static RunningStatistics RunningStatisticsClone(double[] a) => new RunningStatistics(a).Clone();

// Fourier fixture args conventions -- MUST mirror fourier_*_at() in
// core/tests/test_fixtures.cpp exactly (see fixtures/special_functions/fourier.json).
static double FourierFftAt(double[] a)
{
    int n = a.Length - 2;
    var data = a[..n];
    bool inverse = a[n] != 0.0;
    int index = (int)a[n + 1];
    Fourier.FFT(data, inverse);
    return data[index];
}
static double FourierRealFftAt(double[] a)
{
    int n = a.Length - 2;
    var data = a[..n];
    bool inverse = a[n] != 0.0;
    int index = (int)a[n + 1];
    Fourier.RealFFT(data, inverse);
    return data[index];
}
static double FourierCorrelationAt(double[] a)
{
    int n = (a.Length - 1) / 2;
    var data1 = a[..n];
    var data2 = a[n..(2 * n)];
    int index = (int)a[2 * n];
    var corr = Fourier.Correlation(data1, data2);
    return corr[index];
}
static double FourierAutocorrelationAt(double[] a)
{
    int n = a.Length - 2;
    var series = a[..n];
    int lagMax = (int)a[n];
    int lag = (int)a[n + 1];
    var acf = Fourier.Autocorrelation(series, lagMax);
    if (acf is null) throw new Exception("Fourier.autocorrelation_at: Autocorrelation returned null");
    return acf[lag, 1];
}

// Closed registry of named functions for the numerical_derivative fixture -- MUST match
// numerical_derivative_{quadratic,normal_loglik} in core/tests/test_fixtures.cpp exactly.
// (A top-level-statements Program.cs cannot declare a `static readonly` field outside a
// type, so the embedded sample is a local function returning a fresh array each call,
// mirroring the C++ side's static-local-inside-a-function pattern.)
static double[] NumericalDerivativeNormalSample() => [9.0, 10.0, 11.0, 12.0, 13.0];
static double NumericalDerivativeQuadratic(double[] x)
{
    double s = 0;
    for (int i = 0; i < x.Length; i++) { double d = x[i] - i; s += d * d; }
    return s;
}
static double NumericalDerivativeNormalLoglik(double[] x)
{
    var n = new Normal(x[0], x[1]);
    return n.LogLikelihood(NumericalDerivativeNormalSample());
}

// Task 8 (the optimizer runner): a handful of Test_Numerics/Mathematics/Optimization/
// TestFunctions.cs formulas, inlined here (that project is not referenced by this emitter --
// see the NumericalDerivativeQuadratic/NormalLoglik precedent above for the same reason) so
// fixtures/toolbox/optimizers.json's `construct.objective` names can be driven against the REAL
// C# optimizer classes. MUST mirror core/tests/optimization_test_functions.hpp exactly.
static double OptimizerTestFX(double x) => (x + 3d) * Math.Pow(x - 1d, 2d);
static double OptimizerTestFXYZ(double[] p) =>
    Math.Pow(4d * p[0] - 0.5d, 2d) + Math.Pow(3d * p[1] - 0.6d, 2d) + Math.Pow(2d * p[2] - 0.7d, 2d);
static double OptimizerTestDeJong(double[] p) => p.Sum(v => v * v);
static double OptimizerTestBooth(double[] p) =>
    Math.Pow(p[0] + 2d * p[1] - 7d, 2d) + Math.Pow(2d * p[0] + p[1] - 5d, 2d);
static double OptimizerTestMcCormick(double[] p) =>
    Math.Sin(p[0] + p[1]) + Math.Pow(p[0] - p[1], 2d) - 1.5d * p[0] + 2.5d * p[1] + 1d;
static Func<double[], double> OptimizerTestFunction(string name) => name switch
{
    "FXYZ" => OptimizerTestFXYZ,
    "DeJong" => OptimizerTestDeJong,
    "Booth" => OptimizerTestBooth,
    "McCormick" => OptimizerTestMcCormick,
    "FX" => p => OptimizerTestFX(p[0]),
    _ => throw new Exception($"unknown optimizer fixture objective: {name}")
};

// The callback surface, Task 1: the Test_Brent/Test_Differentiation formulas
// fixtures/callback/math.json names by string, inlined here for the same reason as the optimizer
// catalog above (Test_Numerics is not referenced by this emitter). These are the C# delegates the
// REAL Brent/NumericalDerivative are driven with, the counterpart of the native closures the
// C++/R/Python fixture runners write for the same names -- so every case exercises a real
// host-language callback in each of the four runners. NOTE the names here are deliberately NOT the
// optimizer catalog's: `Diff_FXYZ` is Test_Differentiation.FXYZ (x^3 + y^4 + z^5), unrelated to
// OptimizerTestFXYZ above. `Quad_*` are Mathematics/Integration/Integrands.*; `Diff_FX` and
// `Quad_FX3` are both x^3, from two different upstream test files -- hence the prefixes.
static Func<double, double>? CallbackScalarFunction(string name) => name switch
{
    "Root_Quadratic" => x => Math.Pow(x, 2) - 2,
    // P2 "math extras": TestFunctions.Quadratic_Deriv, the newton catalog's counterpart of
    // Root_Quadratic. Same Func<double, double> shape as every other member of this switch, so it
    // lives here rather than in a derivative-only catalog of its own.
    "RootD_Quadratic" => x => 2d * x,
    "Root_Cubic" => x => x * x * x - x - 1d,
    // TestFunctions.Trigonometric, root ~1.12191713 on [0, pi].
    "Root_Trigonometric" => x => 2d * Math.Sin(x) - 3d * Math.Cos(x) - 0.5,
    // TestFunctions.Trigonometric_Deriv.
    "RootD_Trigonometric" => x => 2d * Math.Cos(x) + 3d * Math.Sin(x),
    "Diff_FX" => x => Math.Pow(x, 3.0),
    "Quad_FX3" => x => Math.Pow(x, 3d),
    "Quad_Cosine" => x => Math.Cos(x),
    "Quad_Sine" => x => Math.Sin(x),
    "Quad_FXX" => x => 0.5 + 24 * x + 3 * x * x,
    "Quad_FXXX" => x => 0.5 + 24 * x + 3 * x * x + 8 * x * x * x,
    // corehydro addition, no upstream integrand: every Integrands.cs function the upstream tests
    // integrate converges on the first whole-interval G10K21 evaluation, so none of them reaches
    // the subdividing branch of the recursion. This Lorentzian peak (half-width 0.01 on [-1, 1])
    // does, and it is written with multiplication and division only -- no transcendental -- so the
    // C++/R/Python/C# closures for it are bit-identical and the evaluation count is an oracle
    // rather than a libm coincidence.
    "Quad_Peak" => x => 1d / (1d + 1e4 * x * x),
    _ => null
};
static Func<double[], double>? CallbackVectorFunction(string name) => name switch
{
    "Diff_FXY" => p => Math.Pow(p[0], 2) * Math.Pow(p[1], 3),
    "Diff_FXYZ" => p => Math.Pow(p[0], 3.0) + Math.Pow(p[1], 4.0) + Math.Pow(p[2], 5.0),
    "Diff_FH" => p => Math.Pow(p[0], 3.0) - 2 * p[0] * p[1] - Math.Pow(p[1], 6),
    _ => null
};

// P2 "math extras", the math/root_find_system catalog: Test_NewtonRaphson.Test_Multi_LinearSystem's
// system, F([x;y]) = [3x + y - 9, x + 2y - 8] with the constant Jacobian [[3, 1], [1, 2]], whose
// unique root is [2, 3]. `double[,]`, not `Matrix`, so this catalog's shape matches
// CallbackJacobianFunction's (the gmm group's) rather than adding a third jacobian shape.
static Func<double[], double[]>? CallbackSystemFunction(string name) => name switch
{
    "Sys_Linear_F" => v => new[] { 3.0 * v[0] + v[1] - 9.0, v[0] + 2.0 * v[1] - 8.0 },
    _ => null
};

static Func<double[], double[,]>? CallbackSystemJacobianFunction(string name) => name switch
{
    "Sys_Linear_J" => _ => new double[,] { { 3.0, 1.0 }, { 1.0, 2.0 } },
    _ => null
};

// The callback surface, Task 3: the rng-group catalog for fixtures/callback/rng_handle.json. The
// delegate signature is `(double[] parameters, Random prng)` -- upstream's OWN Gibbs.Proposal
// signature -- and these are the C# counterparts of the four runners' native closures, each drawing
// from the generator the runner seeded and hands in. What they pin is the property the RNG handle
// exists for: a draw taken inside a host-language callback is a draw off THIS stream, so R and
// Python agree with C# value for value. Loops rather than LINQ, so the draw ORDER is written down
// rather than left to deferred execution.
static Func<double[], Random, double[]>? CallbackRngFunction(string name) => name switch
{
    // n uniforms; n comes from the parameters vector so one entry serves any count.
    "Rng_Uniform" => (p, prng) =>
    {
        var v = new double[(int)p[0]];
        for (int i = 0; i < v.Length; i++) v[i] = prng.NextDouble();
        return v;
    },
    // n integers on [min, max), i.e. Next(minInclusive, maxExclusive): parameters are n, min, max.
    "Rng_Integers" => (p, prng) =>
    {
        var v = new double[(int)p[0]];
        for (int i = 0; i < v.Length; i++) v[i] = prng.Next((int)p[1], (int)p[2]);
        return v;
    },
    // Two uniforms, two integers, one uniform, off ONE generator. Uniforms and integers share the
    // state, so this pins the draw ORDER as well as the values: a runner that took its integers
    // from a second generator, or that reordered the two verbs, would reproduce neither the third
    // uniform nor the integers.
    "Rng_Interleaved" => (p, prng) =>
    {
        var v = new double[5];
        v[0] = prng.NextDouble();
        v[1] = prng.NextDouble();
        v[2] = prng.Next(0, 100);
        v[3] = prng.Next(0, 100);
        v[4] = prng.NextDouble();
        return v;
    },
    // 1000 draws discarded, then 10 kept -- the shape of upstream's own Test_MersenneTwister
    // (Sampling/Test_MersenneTwister.cs), whose ten trueDouble literals come from the reference
    // mt19937ar.c output file. NextDouble() is GenRandReal2, one GenRandInt32 per call, so
    // discarding 1000 of them advances the state exactly as the upstream test's 1000 GenRandInt32
    // calls do and the eleventh through twentieth draws are its literals.
    "Rng_Warmup1000" => (p, prng) =>
    {
        for (int i = 0; i < 1000; i++) prng.NextDouble();
        var v = new double[10];
        for (int i = 0; i < v.Length; i++) v[i] = prng.NextDouble();
        return v;
    },
    _ => null
};

// The callback surface, Task 4: the mcmc-group catalog for fixtures/callback/mcmc.json. These are
// the C# `LogLikelihood` delegates the REAL Numerics samplers are constructed with -- upstream's
// own `MCMCSampler(List<IUnivariateDistribution>, LogLikelihood)` constructor -- and the
// counterpart of the native closures the C++/R/Python fixture runners write for the same names.
//
// BOTH are deliberately built from `+ - * /` alone, with the summation written as an explicit
// loop rather than any language's sum(): a Markov chain amplifies one differing bit into a
// different chain outright (one flipped accept/reject changes every state after it), so a
// transcendental call, or R's extended-precision `sum()`, would make these oracles a coincidence
// of one platform's math library rather than a reproducible fact. The datasets are inlined here
// exactly as they are in the other three runners.
// (The datasets are built inside each delegate rather than held in fields: this file is a C#
// top-level program, which has no place to declare one.)
static Func<double[], double>? CallbackMcmcFunction(string name) => name switch
{
    // One parameter: the Gaussian kernel -0.5 * sum((x - mu)^2), unit variance, no normalizer.
    "Mcmc_GaussianKernel" => p =>
    {
        double[] data = { 4.9, 5.1, 5.0, 5.2, 4.8 };
        double acc = 0d;
        foreach (double x in data) acc += (x - p[0]) * (x - p[0]);
        return -0.5 * acc;
    },
    // Two parameters: the same kernel over the residuals of a straight line, y = a + b * t.
    "Mcmc_LinearKernel" => p =>
    {
        double[] t = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        double[] y = { 2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1 };
        double acc = 0d;
        for (int i = 0; i < t.Length; i++)
        {
            double residual = y[i] - p[0] - p[1] * t[i];
            acc += residual * residual;
        }
        return -0.5 * acc;
    },
    // The Gibbs case's model, whose full conditional really IS uniform: with
    // x_i ~ Uniform(mu - 1, mu + 1) and a flat prior, mu given the data is
    // Uniform(max(x) - 1, min(x) + 1), so CallbackProposalFunction below is an EXACT Gibbs step
    // rather than a random walk wearing Gibbs's name. The normalizing term is dropped, as the two
    // kernels above drop theirs; comparisons and arithmetic only.
    "Mcmc_UniformWidthKernel" => p =>
    {
        double[] data = { 4.9, 5.1, 5.0, 5.2, 4.8 };
        foreach (double x in data)
            if (x - p[0] > 1d || p[0] - x > 1d) return double.NegativeInfinity;
        return 0d;
    },
    // Task-5 review fix, coverage finding: unlike Mcmc_GaussianKernel, whose derivative
    // sum(x - mu) is LINEAR in mu (so its third derivative is zero and the ported
    // central-difference default agrees with the analytic gradient to rounding, ~4e-16), this
    // kernel's derivative is CUBIC in mu, so the central-difference truncation error is real rather
    // than rounding -- the analytic and default gradients genuinely disagree, which is what a
    // supplied-vs-ignored gradient regression needs to be caught by the oracle gate. The 0.05
    // coefficient is load-bearing, not decorative: measured by brute-force sweep (a coefficient of
    // 1 over the same data), an unscaled quartic makes HMC's leapfrog trajectory genuinely CHAOTIC
    // over 200 iterations -- the analytic and default gradients then diverge at the ~0.3% level, the
    // same order this fixture's own cross-language divergence measured at that scale, so no fixed
    // tolerance could pin it. At 0.05 the divergence is small and smooth (~3e-8 relative, four
    // orders past the file's 1e-12 tolerance) rather than chaotic, which is what keeps the case
    // usable as an oracle at all.
    "Mcmc_QuarticKernel" => p =>
    {
        double[] data = { 4.9, 5.1, 5.0, 5.2, 4.8 };
        double acc = 0d;
        foreach (double x in data)
        {
            double d = x - p[0];
            acc += d * d * d * d;
        }
        return -0.05d * acc;
    },
    _ => null
};

// The callback surface, Task 5: the two OTHER delegates upstream's samplers take, driven against
// the REAL C# Gibbs and HMC/NUTS. `Gibbs.Proposal(double[] parameters, Random prng)` is required
// by Gibbs and accepted by nothing else; `HMC.Gradient(IList<double> parameters)` is optional for
// HMC/NUTS and, left null, leaves the C# class's own bound-aware finite-difference gradient in
// force -- which is exactly what a case with no `gradient` key pins.
static Gibbs.Proposal? CallbackProposalFunction(string name) => name switch
{
    // One uniform draw from the full conditional Uniform(max(x) - 1, min(x) + 1). The draw comes
    // off the generator the sampler hands in, which is the whole point of the delegate's signature.
    "Prop_UniformConditional" => (parameters, prng) =>
    {
        double[] data = { 4.9, 5.1, 5.0, 5.2, 4.8 };
        double lo = data[0] - 1d, hi = data[0] + 1d;
        foreach (double x in data)
        {
            if (x - 1d > lo) lo = x - 1d;
            if (x + 1d < hi) hi = x + 1d;
        }
        return new[] { lo + prng.NextDouble() * (hi - lo) };
    },
    // The TWO-parameter member of the catalog, written for the second case of
    // fixtures/callback/callback_cross_language.json. An INDEPENDENCE proposal: it ignores the
    // state it is handed and draws each parameter from a fixed interval, lo + u * (hi - lo),
    // exactly as the one above does for a single parameter. Gibbs accepts every proposal, so the
    // recorded chain is a sequence of independent draws from that box and the only mark the
    // log-density leaves on the run is the fitness it reports. Two successive NextDouble calls, in
    // the order the other three runners' single length-two draw consumes them.
    "Prop_UniformBox" => (parameters, prng) =>
    {
        double[] lo = { -1.0, 1.5 };
        double[] hi = { 1.0, 2.5 };
        var v = new double[2];
        double[] u = { prng.NextDouble(), prng.NextDouble() };
        for (int j = 0; j < 2; j++) v[j] = lo[j] + u[j] * (hi[j] - lo[j]);
        return v;
    },
    _ => null
};
static HMC.Gradient? CallbackGradientFunction(string name) => name switch
{
    // The analytic derivative of Mcmc_GaussianKernel, d/dmu = sum(x - mu), summed in an explicit
    // loop for the same reason the kernels are.
    "Grad_GaussianKernel" => parameters =>
    {
        double[] data = { 4.9, 5.1, 5.0, 5.2, 4.8 };
        double acc = 0d;
        foreach (double x in data) acc += x - parameters[0];
        return new Vector(new[] { acc });
    },
    // The analytic derivative of Mcmc_QuarticKernel, d/dmu = 0.05 * 4 * sum((x - mu)^3).
    "Grad_QuarticKernel" => parameters =>
    {
        double[] data = { 4.9, 5.1, 5.0, 5.2, 4.8 };
        double acc = 0d;
        foreach (double x in data)
        {
            double d = x - parameters[0];
            acc += d * d * d;
        }
        return new Vector(new[] { 0.2d * acc });
    },
    _ => null
};

// The callback surface, Task 6: the bootstrap-group catalog for fixtures/callback/bootstrap.json.
// These are the four C# delegates the REAL Numerics `Bootstrap<double[]>` is driven with --
// upstream's own `ResampleFunction` / `FitFunction` / `StatisticFunction` / `JackknifeFunction`
// properties -- and the counterpart of the native closures the C++/R/Python fixture runners write
// for the same names.
//
// All four are built from `+ - * /` and comparisons alone, with the mean summed in an explicit loop
// rather than through LINQ's Average(): R's sum()/mean() accumulate in extended precision where the
// other three languages accumulate in double, and one differing bit in a fitted mean moves a
// percentile. They are also pure functions of their arguments, which matters here and nowhere else
// in this file: C#'s Bootstrap.Run drives them from a Parallel.For.
static Func<double[], ParameterSet, Random, double[]>? CallbackResampleFunction(string name) => name switch
{
    // The ordinary nonparametric resample: n draws of data[prng.Next(0, n)], every index off the
    // generator the replicate hands in (which is the whole point of the delegate's signature).
    "Resample_Iid" => (data, parameters, prng) =>
    {
        var v = new double[data.Length];
        for (int i = 0; i < v.Length; i++) v[i] = data[prng.Next(0, data.Length)];
        return v;
    },
    _ => null
};
static Func<double[], ParameterSet>? CallbackFitFunction(string name) => name switch
{
    // A one-parameter model whose fit is the sample mean. NaN fitness matches what the C++ runner
    // and both glues construct; nothing in the bootstrap reads it.
    "Fit_Mean" => data =>
    {
        double acc = 0d;
        foreach (double x in data) acc += x;
        return new ParameterSet(new[] { acc / data.Length }, double.NaN);
    },
    // The CONTRACTION-BEARING member of the catalog, written for the second case of
    // fixtures/callback/callback_cross_language.json: the ordinary least-squares line of the
    // sample against its position t = 1..n, in the centered form
    //   slope = sum(dt * dy) / sum(dt * dt),   intercept = ybar - slope * tbar
    // returned as [intercept, slope]. Every accumulation is `acc + a * b`, the shape clang and gcc
    // fuse into a multiply-add by default; RyuJIT never fuses one on its own (C# spells that
    // Math.FusedMultiplyAdd), so this delegate computes the written arithmetic and the C++ catalog
    // matches it only because core/CMakeLists.txt turns contraction off for that translation unit.
    "Fit_LinearTrend" => data =>
    {
        double n = data.Length;
        double st = 0d, sy = 0d;
        for (int i = 0; i < data.Length; i++) { st += (double)(i + 1); sy += data[i]; }
        double tbar = st / n, ybar = sy / n;
        double num = 0d, den = 0d;
        for (int i = 0; i < data.Length; i++)
        {
            double dt = (double)(i + 1) - tbar;
            double dy = data[i] - ybar;
            num += dt * dy;
            den += dt * dt;
        }
        double slope = num / den;
        return new ParameterSet(new[] { ybar - slope * tbar, slope }, double.NaN);
    },
    _ => null
};
static Func<ParameterSet, double[]>? CallbackStatisticFunction(string name) => name switch
{
    "Stat_Identity" => ps => (double[])ps.Values.Clone(),
    "Stat_MeanAndSquare" => ps => new[] { ps.Values[0], ps.Values[0] * ps.Values[0] },
    _ => null
};
// The PIVOTAL member of the same catalog: upstream's `Func<TData, BootstrapFit>
// FitWithCovarianceFunction`, the delegate that run type fits through. The model is the
// two-parameter Normal location-scale MLE -- theta = (mu, sigma) with sigma the POPULATION standard
// deviation -- whose covariance is analytic, diag(s2 / n, s2 / (2n)), so the whole delegate is
// arithmetic plus one Math.Sqrt. Sqrt is the one libm function IEEE 754 requires to be correctly
// rounded, so unlike Log or Exp it is the same value in all four runners; the sums are explicit
// loops for the reason Fit_Mean above gives. `ss += (x - mu) * (x - mu)` is itself a
// contraction-bearing shape, so this zero-tolerance guarantee also depends on the C++ catalog's
// own -ffp-contract=off scoping in core/CMakeLists.txt.
static Func<double[], BootstrapFit>? CallbackFitWithCovarianceFunction(string name) => name switch
{
    "FitCov_NormalMLE" => data =>
    {
        double n = data.Length;
        double acc = 0d;
        foreach (double x in data) acc += x;
        double mu = acc / n;
        double ss = 0d;
        foreach (double x in data) ss += (x - mu) * (x - mu);
        double s2 = ss / n;
        var covariance = new Matrix(2);
        covariance[0, 0] = s2 / n;
        covariance[1, 1] = s2 / (2d * n);
        return new BootstrapFit(new[] { mu, Math.Sqrt(s2) }, covariance);
    },
    _ => null
};
static Func<double[], int, double[]>? CallbackJackknifeFunction(string name) => name switch
{
    // The leave-one-out sample ComputeAccelerationConstants needs for BCa. `index` counts from 0,
    // as upstream's own delegate does.
    "Jack_LeaveOneOut" => (data, index) =>
    {
        var v = new double[data.Length - 1];
        int at = 0;
        for (int i = 0; i < data.Length; i++)
            if (i != index) v[at++] = data[i];
        return v;
    },
    _ => null
};

// Flattens a callback fixture file into the (caseName, subLabel, subCase) triples its one branch
// below drives. A "callback"-kind case IS its own single sub-block, so it yields one triple with an
// empty label; a "callback_cross_language"-kind case nests one block per key OTHER than "name" --
// "mcmc", "bootstrap", "pivotal" -- each shaped exactly like a "callback"-kind case's
// construct/assertions, so it yields one triple each. The labels are read OFF the case rather than
// listed here, so a case may nest one block or five without an emitter change. The branch body is
// then written once and reached identically by both kinds, which is what keeps the cross-language
// fixture from growing an evaluation path of its own.
static IEnumerable<(string caseName, string subLabel, JsonElement subCase)> CallbackSubCases(
    JsonElement root, string kind)
{
    foreach (var c in root.GetProperty("cases").EnumerateArray())
    {
        string caseName = c.GetProperty("name").GetString()!;
        if (kind == "callback")
        {
            yield return (caseName, "", c);
        }
        else
        {
            foreach (var property in c.EnumerateObject())
            {
                if (property.Name == "name") continue;
                yield return (caseName, property.Name, property.Value);
            }
        }
    }
}

// Builds + configures + runs one callback-group bootstrap case, mirroring callback/bootstrap.hpp's
// run_bootstrap() decision for decision: theta-hat is the `parameters` option or the fit of the
// original data, the ci_method picks the workflow (BootstrapT is the studentized one, everything
// else the regular Run), and an absent key leaves the C# class's OWN default in force.
static (double[] values, string[] names, int[] dims) RunCallbackBootstrap(
    JsonElement options,
    Func<double[], ParameterSet, Random, double[]> resample,
    Func<double[], ParameterSet> fit,
    Func<ParameterSet, double[]> statistic,
    Func<double[], int, double[]>? jackknife)
{
    bool Has(string key) => options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out _);
    double Num(string key, double dflt) =>
        options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out var v)
            ? ParseNum(v) : dflt;

    if (!Has("data")) throw new Exception("bootstrap/run requires the option 'data'");
    double[] data = options.GetProperty("data").EnumerateArray().Select(ParseNum).ToArray();
    string ciName = options.ValueKind == JsonValueKind.Object &&
                    options.TryGetProperty("ci_method", out var ciEl)
        ? ciEl.GetString()! : "Percentile";
    var method = ParseBootstrapCIMethod(ciName);
    double alpha = Num("alpha", 0.1);

    ParameterSet original = Has("parameters")
        ? new ParameterSet(options.GetProperty("parameters").EnumerateArray().Select(ParseNum).ToArray(),
                           double.NaN)
        : fit(data);

    var boot = new Bootstrap<double[]>(data, original)
    {
        ResampleFunction = resample,
        FitFunction = fit,
        StatisticFunction = statistic
    };
    if (jackknife != null)
    {
        boot.JackknifeFunction = jackknife;
        // Not a user callback on this surface: TData is a double[], so its length IS the sample
        // size (see callback/bootstrap.hpp).
        boot.SampleSizeFunction = d => d.Length;
    }
    if (Has("replicates")) boot.Replicates = (int)Num("replicates", 0);
    if (Has("seed")) boot.PRNGSeed = (int)Num("seed", 0);
    if (Has("prng_seed")) boot.PRNGSeed = (int)Num("prng_seed", 0);
    if (Has("max_retries")) boot.MaxRetries = (int)Num("max_retries", 0);
    if (Has("inner_replicates")) boot.InnerReplicates = (int)Num("inner_replicates", 0);

    if (method == BootstrapCIMethod.BootstrapT) boot.RunWithStudentizedBootstrap();
    else boot.Run();
    var results = boot.GetConfidenceIntervals(method, alpha);

    // The layout callback/bootstrap.hpp's bootstrap_flatten() documents.
    var values = new List<double>();
    var names = new List<string>();
    void Push(string name, double value) { names.Add(name); values.Add(value); }
    void PushBlock(string label, BootstrapStatisticResult[] block)
    {
        for (int i = 0; i < block.Length; i++)
        {
            string ix = $"[{i}]";
            Push(label + ix, block[i].PopulationEstimate);
            Push(label + "_lower" + ix, block[i].LowerCI);
            Push(label + "_upper" + ix, block[i].UpperCI);
            Push(label + "_se" + ix, block[i].StandardError);
            Push(label + "_mean" + ix, block[i].Mean);
            Push(label + "_valid" + ix, block[i].ValidCount);
        }
    }

    Push("replicates", boot.Replicates);
    Push("failed_replicates", results.FailedReplicates);
    Push("alpha", results.Alpha);
    PushBlock("statistic", results.StatisticResults);
    PushBlock("parameter", results.ParameterResults);

    return (values.ToArray(), names.ToArray(),
            new[] { results.StatisticResults.Length, results.ParameterResults.Length });
}

// Builds + runs one callback-group bootstrap case with `"run_type": "pivotal"`, mirroring
// callback/bootstrap.hpp's run_bootstrap_pivotal() decision for decision: the parent fit is the
// `parameters`/`original_covariance` options over the covariance-aware fit of the original data
// (C#'s second constructor), every pivotal property is set only when its key is present so the C#
// class's OWN default stays in force otherwise, and the result carries the raw block and the six
// diagnostic counts after the regular layout.
static (double[] values, string[] names, int[] dims) RunCallbackBootstrapPivotal(
    JsonElement options,
    Func<double[], ParameterSet, Random, double[]> resample,
    Func<double[], BootstrapFit> fitWithCovariance,
    Func<ParameterSet, double[]> statistic)
{
    bool Has(string key) => options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out _);
    double Num(string key, double dflt) =>
        options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out var v)
            ? ParseNum(v) : dflt;

    if (!Has("data")) throw new Exception("bootstrap/run requires the option 'data'");
    double[] data = options.GetProperty("data").EnumerateArray().Select(ParseNum).ToArray();
    string ciName = Has("ci_method") ? options.GetProperty("ci_method").GetString()! : "Percentile";
    if (ParseBootstrapCIMethod(ciName) != BootstrapCIMethod.Percentile)
        throw new Exception("Only percentile confidence intervals are supported after a pivotal bootstrap run.");
    double alpha = Num("alpha", 0.1);

    Matrix? parentCovariance = null;
    if (Has("original_covariance"))
    {
        var rows = options.GetProperty("original_covariance").EnumerateArray()
            .Select(row => row.EnumerateArray().Select(ParseNum).ToArray()).ToArray();
        var array = new double[rows.Length, rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[i].Length; j++) array[i, j] = rows[i][j];
        parentCovariance = new Matrix(array);
    }
    double[] parentParameters = Has("parameters")
        ? options.GetProperty("parameters").EnumerateArray().Select(ParseNum).ToArray()
        : Array.Empty<double>();
    if (parentCovariance == null || parentParameters.Length == 0)
    {
        var probe = fitWithCovariance(data);
        if (parentParameters.Length == 0) parentParameters = (double[])probe.Parameters.Values.Clone();
        parentCovariance ??= probe.Covariance;
    }

    var boot = new Bootstrap<double[]>(data, new BootstrapFit(parentParameters, parentCovariance))
    {
        ResampleFunction = resample,
        FitWithCovarianceFunction = fitWithCovariance,
        StatisticFunction = statistic
    };
    if (Has("replicates")) boot.Replicates = (int)Num("replicates", 0);
    if (Has("seed")) boot.PRNGSeed = (int)Num("seed", 0);
    if (Has("prng_seed")) boot.PRNGSeed = (int)Num("prng_seed", 0);
    if (Has("max_retries")) boot.MaxRetries = (int)Num("max_retries", 0);

    if (Has("pivotal_links"))
    {
        var links = options.GetProperty("pivotal_links").EnumerateArray().Select(entry =>
            entry.ValueKind == JsonValueKind.Null
                ? null
                : LinkFunctionFactory.Create(Enum.Parse<LinkFunctionType>(entry.GetString()!))).ToArray();
        if (links.Length != parentParameters.Length)
            throw new Exception("'pivotal_links' must name one link per parameter");
        boot.PivotalLinkFactory = _ => links;
    }
    if (Has("pivotal_invalid_draw_policy"))
    {
        boot.PivotalInvalidDrawPolicy = options.GetProperty("pivotal_invalid_draw_policy").GetString() switch
        {
            "drop" => PivotalBootstrapInvalidDrawPolicy.Drop,
            "use_raw" => PivotalBootstrapInvalidDrawPolicy.UseRaw,
            "use_parent" => PivotalBootstrapInvalidDrawPolicy.UseParent,
            var p => throw new Exception($"unknown pivotal_invalid_draw_policy '{p}'"),
        };
    }
    if (Has("regularize_pivotal_covariances"))
        boot.RegularizePivotalCovariances = options.GetProperty("regularize_pivotal_covariances").GetBoolean();
    if (Has("pivotal_z_limit")) boot.PivotalZLimit = Num("pivotal_z_limit", 0d);
    if (Has("add_pivotal_jitter"))
        boot.AddPivotalJitter = options.GetProperty("add_pivotal_jitter").GetBoolean();
    if (Has("pivotal_jitter_scale")) boot.PivotalJitterScale = Num("pivotal_jitter_scale", 0.01d);

    boot.RunPivotalBootstrap();
    var results = boot.GetConfidenceIntervals(BootstrapCIMethod.Percentile, alpha);
    var rawResults = boot.GetRawPivotalConfidenceIntervals(alpha);

    // The layout callback/bootstrap.hpp's bootstrap_flatten() plus its pivotal additions document.
    var values = new List<double>();
    var names = new List<string>();
    void Push(string name, double value) { names.Add(name); values.Add(value); }
    void PushBlock(string label, BootstrapStatisticResult[] block)
    {
        for (int i = 0; i < block.Length; i++)
        {
            string ix = $"[{i}]";
            Push(label + ix, block[i].PopulationEstimate);
            Push(label + "_lower" + ix, block[i].LowerCI);
            Push(label + "_upper" + ix, block[i].UpperCI);
            Push(label + "_se" + ix, block[i].StandardError);
            Push(label + "_mean" + ix, block[i].Mean);
            Push(label + "_valid" + ix, block[i].ValidCount);
        }
    }

    Push("replicates", boot.Replicates);
    Push("failed_replicates", results.FailedReplicates);
    Push("alpha", results.Alpha);
    PushBlock("statistic", results.StatisticResults);
    PushBlock("parameter", results.ParameterResults);
    PushBlock("raw_statistic", rawResults.StatisticResults);
    PushBlock("raw_parameter", rawResults.ParameterResults);
    var diagnostics = boot.PivotalDiagnostics;
    if (diagnostics != null)
    {
        Push("requested_replicates", diagnostics.RequestedReplicates);
        Push("rejected_raw_replicates", diagnostics.RejectedRawReplicates);
        Push("failed_raw_replicates", diagnostics.FailedRawReplicates);
        Push("accepted_raw_replicates", diagnostics.AcceptedRawReplicates);
        Push("invalid_pivotal_replicates", diagnostics.InvalidPivotalReplicates);
        Push("retained_pivotal_replicates", diagnostics.RetainedPivotalReplicates);
    }

    return (values.ToArray(), names.ToArray(),
            new[] { results.StatisticResults.Length, results.ParameterResults.Length });
}

// The callback surface, Task 7: the gmm-group catalog for fixtures/callback/gmm.json. These are the
// three C# delegates the REAL RMC.BestFit `GeneralizedMethodOfMoments` delegate constructor (C# 143)
// takes -- upstream's own `MomentConditionFunction` / `JacobianFunction` / `PenaltyFunction` -- and
// the counterpart of the native closures the C++/R/Python fixture runners write for the same names.
//
// The model is the just-identified two-parameter method-of-moments fit of a Normal: theta =
// (mu, sigma2) and g = [mean(x - mu), mean((x - mu)^2 - sigma2)], whose unique root -- and so the
// GMM optimum, since q = p makes g = 0 attainable -- is the sample mean and the population
// variance. All three are built from `+ - * /` alone, with every sum written as an explicit loop
// rather than through LINQ's Average(): R's sum()/mean() accumulate in extended precision where the
// other three languages accumulate in double, and one differing bit moves a fitted parameter.
// A method rather than a field: this file is a top-level-statements program, where a `static
// readonly` field is not legal.
static double[] CallbackGmmData() => new[] { 4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7 };

// The same eight observations with the decimal point moved two places, so the fitted scale
// parameter is small (0.00404) next to the ported numerical Jacobian's step h = 1e-4 (|theta| + 1).
// That is what makes the analytic and numerical Jacobians of the fourth-moment condition below
// genuinely disagree; see the fixture's note on Mom_NormalFourthMoment.
static double[] CallbackGmmSmallScaleData() =>
    new[] { 0.041, 0.052, 0.048, 0.055, 0.049, 0.051, 0.053, 0.047 };

static MomentConditionFunction? CallbackMomentConditionFunction(string name) => name switch
{
    "Mom_NormalMeanVariance" => parameters =>
    {
        double n = 8d;
        double g0 = 0d, g1 = 0d, s00 = 0d, s01 = 0d, s11 = 0d;
        foreach (double x in CallbackGmmData())
        {
            double a = x - parameters[0];
            double b = a * a - parameters[1];
            g0 += a; g1 += b; s00 += a * a; s01 += a * b; s11 += b * b;
        }
        var g = new Vector(new[] { g0 / n, g1 / n });
        var s = new Matrix(new[,] { { s00 / n, s01 / n }, { s01 / n, s11 / n } });
        return (g, s);
    },
    // The OVER-IDENTIFIED member of the same catalog: the identical Normal model and the identical
    // eight observations, with a third moment condition added -- mean((x - mu)^3), zero for a
    // Normal -- so q = 3 > p = 2 and DegreeOfFreedom becomes 1. The only case in the file that
    // reaches the chi-squared p-value branch of PostProcess().
    "Mom_NormalThreeMoments" => parameters =>
    {
        double n = 8d;
        double g0 = 0d, g1 = 0d, g2 = 0d;
        double s00 = 0d, s01 = 0d, s02 = 0d, s11 = 0d, s12 = 0d, s22 = 0d;
        foreach (double x in CallbackGmmData())
        {
            double a = x - parameters[0];
            double b = a * a - parameters[1];
            double c = a * a * a;
            g0 += a; g1 += b; g2 += c;
            s00 += a * a; s01 += a * b; s02 += a * c;
            s11 += b * b; s12 += b * c; s22 += c * c;
        }
        var g = new Vector(new[] { g0 / n, g1 / n, g2 / n });
        var s = new Matrix(new[,] { { s00 / n, s01 / n, s02 / n },
                                    { s01 / n, s11 / n, s12 / n },
                                    { s02 / n, s12 / n, s22 / n } });
        return (g, s);
    },
    // theta = (mu, sigma), matched on the FIRST and FOURTH central moments of a Normal:
    //   g = [mean(x - mu), mean(u^4) - 3 t^4]   with u = 100 (x - mu) and t = 100 sigma
    // so dg2/dsigma = -1200 t^3 is CUBIC in the parameter. A central difference is EXACT for a
    // linear or quadratic derivative and is not for a cubic one, which is the whole point of this
    // member of the catalog -- see Jac_NormalFourthMoment.
    "Mom_NormalFourthMoment" => parameters =>
    {
        double n = 8d;
        double t = parameters[1] * 100d;
        double t4 = 3d * t * t * t * t;
        double g0 = 0d, g1 = 0d, s00 = 0d, s01 = 0d, s11 = 0d;
        foreach (double x in CallbackGmmSmallScaleData())
        {
            double a = x - parameters[0];
            double u = a * 100d;
            double b = u * u * u * u - t4;
            g0 += a; g1 += b; s00 += a * a; s01 += a * b; s11 += b * b;
        }
        var g = new Vector(new[] { g0 / n, g1 / n });
        var s = new Matrix(new[,] { { s00 / n, s01 / n }, { s01 / n, s11 / n } });
        return (g, s);
    },
    _ => null
};

static JacobianFunction? CallbackJacobianFunction(string name) => name switch
{
    // One ROW per moment condition: dg1/dmu = -1, dg1/dsigma2 = 0, dg2/dmu = -2 mean(x - mu),
    // dg2/dsigma2 = -1.
    "Jac_NormalMeanVariance" => parameters =>
    {
        double acc = 0d;
        foreach (double x in CallbackGmmData()) acc += x - parameters[0];
        return new[,] { { -1d, 0d }, { -2d * acc / 8d, -1d } };
    },
    // The analytic Jacobian of Mom_NormalFourthMoment, one ROW per moment condition:
    // dg1/dmu = -1, dg1/dsigma = 0, dg2/dmu = -400 mean(u^3), dg2/dsigma = -1200 t^3.
    "Jac_NormalFourthMoment" => parameters =>
    {
        double acc = 0d;
        foreach (double x in CallbackGmmSmallScaleData())
        {
            double u = (x - parameters[0]) * 100d;
            acc += u * u * u;
        }
        double t = parameters[1] * 100d;
        return new[,] { { -1d, 0d }, { -400d * acc / 8d, -1200d * t * t * t } };
    },
    _ => null
};

static PenaltyFunction? CallbackPenaltyFunction(string name) => name switch
{
    // A ridge penalty pulling sigma2 towards 1, carrying its own 1/2 as the half-quadratic
    // convention in Q() expects.
    "Pen_SigmaTowardsOne" => parameters => 0.5d * (parameters[1] - 1d) * (parameters[1] - 1d),
    _ => null
};

// Builds + configures + fits one callback-group gmm case, mirroring callback/gmm.hpp's run_gmm()
// decision for decision: q is MEASURED by probing the moment condition function once at the initial
// values (never declared), the optimizer/strategy names parse the same way fit_gmm()'s do, and an
// absent key leaves the C# class's OWN default in force. PostProcess(sandwich: true,
// computeJstat: true) matches both the C++ group and fit_gmm()'s model path.
static (double[] values, string[] names, int[] dims) RunCallbackGmm(
    JsonElement options,
    MomentConditionFunction moments,
    JacobianFunction? jacobian,
    PenaltyFunction? penalty)
{
    bool Has(string key) => options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out _);
    double[] Vector1(string key)
    {
        if (!Has(key)) throw new Exception($"gmm/fit requires the option '{key}'");
        return options.GetProperty(key).EnumerateArray().Select(ParseNum).ToArray();
    }

    double[] initial = Vector1("initial"), lower = Vector1("lower"), upper = Vector1("upper");
    if (!Has("sample_size")) throw new Exception("gmm/fit requires the option 'sample_size'");
    int sampleSize = (int)ParseNum(options.GetProperty("sample_size"));
    int p = initial.Length;
    int q = moments(initial).G.Length;   // the up-front probe, exactly as the C++ group does

    var gmm = new GeneralizedMethodOfMoments(moments, p, q, sampleSize, initial, lower, upper,
                                             null, jacobian, penalty, null);
    gmm.OptimizerMethod = Has("optimizer")
        ? ParseOptimizationMethod(options.GetProperty("optimizer").GetString()!)
        : OptimizationMethod.BFGS;
    if (Has("strategy"))
        gmm.EstimationStrategy = ParseGmmStrategy(options.GetProperty("strategy").GetString()!);
    if (Has("max_gmm_iterations"))
        gmm.MaxGMMIterations = (int)ParseNum(options.GetProperty("max_gmm_iterations"));

    if (!gmm.Estimate())
        throw new Exception("GeneralizedMethodOfMoments.Estimate() failed with optimizer " +
                            (Has("optimizer") ? options.GetProperty("optimizer").GetString() : "BFGS"));
    // The J-statistic is allowed to fail, exactly as callback/gmm.hpp's drive site allows it:
    // on a just-identified fit the moment residual covariance is theoretically zero, so inverting
    // it throws as readily as it returns noise, and neither is worth failing an exact fit over.
    // The C# members initialize to 0, not NaN, so the flag is what keeps an uncomputable
    // J-statistic from being reported as a p-value of exactly zero.
    bool jstatComputed = true;
    try { gmm.PostProcess(useSandwich: true, computeJstat: true); }
    catch (Exception) { jstatComputed = false; }

    // The layout callback/gmm.hpp's result block documents.
    var values = new List<double>();
    var names = new List<string>();
    void Push(string name, double value) { names.Add(name); values.Add(value); }

    for (int j = 0; j < p; j++) Push($"parameter[{j}]", gmm.BestParameterSet.Values[j]);
    var se = gmm.GetStandardErrors();
    for (int j = 0; j < p; j++) Push($"standard_error[{j}]", se[j]);
    var cov = gmm.GetCovarianceMatrix();
    var corr = gmm.GetCorrelationMatrix();
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) Push($"covariance[{i},{j}]", cov[i, j]);
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) Push($"correlation[{i},{j}]", corr[i, j]);
    Push("j_stat", jstatComputed ? gmm.JStat : double.NaN);
    Push("j_stat_pval", jstatComputed ? gmm.JStatPval : double.NaN);
    Push("degree_of_freedom", gmm.DegreeOfFreedom);
    Push("gmm_iterations", gmm.GMMIterations);
    Push("converged_within_tolerance", gmm.ConvergedWithinTolerance ? 1d : 0d);
    Push("optimizer_fallback_count", gmm.OptimizerFallbackCount);
    Push("sample_size", gmm.SampleSize);
    Push("number_of_parameters", p);
    Push("number_of_moment_conditions", q);

    return (values.ToArray(), names.ToArray(), new[] { p, p });
}

// Builds the prior list a callback-group mcmc case names, from the same {"family", "parameters"}
// spec grammar dist_spec.hpp builds from on the C++ side.
static List<IUnivariateDistribution> CallbackMcmcPriors(JsonElement options)
{
    if (options.ValueKind != JsonValueKind.Object || !options.TryGetProperty("priors", out var pel))
        throw new Exception("mcmc/sample requires the option 'priors'");
    var priors = new List<IUnivariateDistribution>();
    foreach (var spec in pel.EnumerateArray())
    {
        var dist = UnivariateDistributionFactory.CreateDistribution(
            Enum.Parse<UnivariateDistributionType>(spec.GetProperty("family").GetString()!));
        dist.SetParameters(spec.GetProperty("parameters").EnumerateArray().Select(ParseNum).ToArray());
        priors.Add(dist);
    }
    if (priors.Count == 0) throw new Exception("mcmc/sample requires at least one prior distribution");
    return priors;
}

// Builds + configures + samples one sampler from a callback-group mcmc case's options, mirroring
// mcmc_run.hpp's build_sampler() arm for arm (the five short user-facing keys plus the sampler's
// own setting names; an absent key leaves the C# class's OWN default in force).
static MCMCSampler BuildAndSampleCallbackMcmc(JsonElement options, LogLikelihood logLikelihood,
                                              Gibbs.Proposal? proposal = null,
                                              HMC.Gradient? gradient = null)
{
    var priors = CallbackMcmcPriors(options);
    int d = priors.Count;
    bool Has(string key) => options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out _);
    double Num(string key, double dflt) =>
        options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out var v)
            ? ParseNum(v) : dflt;
    string? Str(string key) =>
        options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out var v)
            ? v.GetString() : null;
    // The short key wins where both exist, matching callback/mcmc.hpp's read order.
    bool HasEither(string shortKey, string longKey) => Has(shortKey) || Has(longKey);
    int IntEither(string shortKey, string longKey, int dflt) =>
        Has(shortKey) ? (int)Num(shortKey, dflt) : (int)Num(longKey, dflt);

    string samplerType = Str("sampler") ?? "RWMH";
    MCMCSampler sampler = samplerType switch
    {
        "RWMH" => new RWMH(priors, logLikelihood,
            Str("proposal_sigma") switch
            {
                null => new Matrix(d),
                "zeros" => new Matrix(d),
                "identity" => Matrix.Identity(d),
                var s => throw new Exception($"unknown proposal_sigma sentinel: {s}")
            }),
        // `gradientFunction: null!` is the C# default, i.e. the class's own bound-aware
        // finite-difference gradient -- which is exactly what a case with no `gradient` key must
        // exercise, so it is passed through rather than replaced by a hand-rolled equivalent.
        "HMC" => new HMC(priors, logLikelihood, stepSize: Num("step_size", 0.1),
                          steps: (int)Num("steps", 50), gradientFunction: gradient!),
        "NUTS" => new NUTS(priors, logLikelihood, stepSize: Num("step_size", 0.1),
                            maxTreeDepth: (int)Num("max_tree_depth", 10), gradientFunction: gradient),
        "ARWMH" => new ARWMH(priors, logLikelihood),
        "Gibbs" => new Gibbs(priors, logLikelihood,
            proposal ?? throw new Exception("the Gibbs sampler requires a proposal function")),
        "SNIS" => new SNIS(priors, logLikelihood),
        "DEMCz" => new DEMCz(priors, logLikelihood),
        "DEMCzs" => new DEMCzs(priors, logLikelihood),
        _ => throw new Exception($"unknown MCMC sampler '{samplerType}'")
    };

    if (Str("initialize") is string init) sampler.Initialize = ParseInitialize(init);
    if (HasEither("seed", "prng_seed")) sampler.PRNGSeed = IntEither("seed", "prng_seed", 12345);
    if (Has("initial_iterations")) sampler.InitialIterations = (int)Num("initial_iterations", 0);
    if (HasEither("warmup", "warmup_iterations"))
        sampler.WarmupIterations = IntEither("warmup", "warmup_iterations", 0);
    if (Has("iterations")) sampler.Iterations = (int)Num("iterations", 0);
    if (HasEither("chains", "number_of_chains"))
        sampler.NumberOfChains = IntEither("chains", "number_of_chains", 0);
    if (HasEither("thinning", "thinning_interval"))
        sampler.ThinningInterval = IntEither("thinning", "thinning_interval", 0);
    if (Has("output_length")) sampler.OutputLength = (int)Num("output_length", 0);
    if (sampler is ARWMH arwmhC)
    {
        if (Has("scale")) arwmhC.Scale = Num("scale", 0);
        if (Has("beta")) arwmhC.Beta = Num("beta", 0);
    }
    if (sampler is DEMCz demczC)
    {
        if (Has("jump")) demczC.Jump = Num("jump", 0);
        if (Has("jump_threshold")) demczC.JumpThreshold = Num("jump_threshold", 0);
        if (Has("noise")) demczC.Noise = Num("noise", 0);
    }
    if (sampler is DEMCzs demczsC)
    {
        if (Has("jump")) demczsC.Jump = Num("jump", 0);
        if (Has("jump_threshold")) demczsC.JumpThreshold = Num("jump_threshold", 0);
        if (Has("snooker_threshold")) demczsC.SnookerThreshold = Num("snooker_threshold", 0);
        if (Has("noise")) demczsC.Noise = Num("noise", 0);
    }
    if (sampler is NUTS nutsC && Has("adapt_mass_matrix"))
        nutsC.AdaptMassMatrix = options.GetProperty("adapt_mass_matrix").GetBoolean();

    sampler.Sample();
    return sampler;
}

// Flattens a finished run into the layout callback/mcmc.hpp's mcmc_flatten() documents: a named
// summary block, then the draws row-major by [chain][draw][parameter].
static (double[] values, string[] names, int[] dims) FlattenCallbackMcmc(MCMCSampler sampler)
{
    var results = new MCMCResults(sampler);
    int chains = sampler.NumberOfChains, p = sampler.NumberOfParameters;
    var values = new List<double>();
    var names = new List<string>();
    void Push(string name, double value) { names.Add(name); values.Add(value); }
    void PushEach(string label, Func<int, double> read, int count)
    {
        for (int j = 0; j < count; j++) Push($"{label}[{j}]", read(j));
    }

    Push("map_fitness", results.MAP.Fitness);
    PushEach("acceptance_rate", j => sampler.AcceptanceRates[j], chains);
    PushEach("map", j => results.MAP.Values[j], p);
    PushEach("posterior_mean", j => results.ParameterResults[j].SummaryStatistics.Mean, p);
    PushEach("posterior_sd", j => results.ParameterResults[j].SummaryStatistics.StandardDeviation, p);
    PushEach("posterior_median", j => results.ParameterResults[j].SummaryStatistics.Median, p);
    PushEach("posterior_lower_ci", j => results.ParameterResults[j].SummaryStatistics.LowerCI, p);
    PushEach("posterior_upper_ci", j => results.ParameterResults[j].SummaryStatistics.UpperCI, p);
    PushEach("rhat", j => results.ParameterResults[j].SummaryStatistics.Rhat, p);
    PushEach("ess", j => results.ParameterResults[j].SummaryStatistics.ESS, p);
    int nSummary = values.Count;

    int draws = sampler.MarkovChains[0].Count;
    for (int c = 0; c < chains; c++)
        for (int i = 0; i < draws; i++)
            for (int j = 0; j < p; j++) values.Add(sampler.MarkovChains[c][i].Values[j]);

    return (values.ToArray(), names.ToArray(), new[] { nSummary, chains, draws, p });
}

// numerical_derivative fixture args convention -- MUST mirror
// numerical_derivative_parse()/gradient_element()/hessian_element() in
// core/tests/test_fixtures.cpp exactly.
static void NumericalDerivativeParse(double[] a, out double[] theta, out double[] lower,
                                      out double[] upper, out int next)
{
    int p = (int)a[0];
    theta = a[1..(1 + p)];
    lower = a[(1 + p)..(1 + 2 * p)];
    upper = a[(1 + 2 * p)..(1 + 3 * p)];
    next = 1 + 3 * p;
}
static double NumericalDerivativeGradientElement(Func<double[], double> f, double[] a)
{
    NumericalDerivativeParse(a, out var theta, out var lower, out var upper, out int next);
    int index = (int)a[next];
    var grad = NumericalDerivative.Gradient(f, theta, lower, upper);
    return grad[index];
}
static double NumericalDerivativeHessianElement(Func<double[], double> f, double[] a)
{
    NumericalDerivativeParse(a, out var theta, out var lower, out var upper, out int next);
    int i = (int)a[next];
    int j = (int)a[next + 1];
    var hess = NumericalDerivative.Hessian(f, theta, lower, upper);
    return hess[i, j];
}

// DifferentialEvolution fixture args convention -- MUST mirror
// differential_evolution_best_value() in core/tests/test_fixtures.cpp exactly:
// args = [fnId, direction, D, lower(D), upper(D), index]. fnId: 0 = Quadratic, 1 =
// NormalLoglik (reuses the closed registry above). direction: 0 = Minimize(), 1 =
// Maximize(). index: 0..D-1 selects BestParameterSet.Values[index]; index == D selects
// BestParameterSet.Fitness. Every other knob uses the library default.
static double DifferentialEvolutionBestValue(double[] a)
{
    int fnId = (int)a[0];
    int direction = (int)a[1];
    int D = (int)a[2];
    var lower = a[3..(3 + D)];
    var upper = a[(3 + D)..(3 + 2 * D)];
    int index = (int)a[3 + 2 * D];

    Func<double[], double> f = fnId switch
    {
        0 => NumericalDerivativeQuadratic,
        1 => NumericalDerivativeNormalLoglik,
        _ => throw new Exception($"unknown DifferentialEvolution function id: {fnId}")
    };

    var de = new DifferentialEvolution(f, D, lower, upper);
    if (direction == 0) de.Minimize(); else de.Maximize();
    return index == D ? de.BestParameterSet.Fitness : de.BestParameterSet.Values[index];
}

// Histogram fixture args convention -- MUST mirror histogram_build() in
// core/tests/test_fixtures.cpp exactly.
static Histogram HistogramBuild(double[] a, int trailing)
{
    int explicitBins = (int)a[0];
    var data = a[1..(a.Length - trailing)];
    return explicitBins > 0 ? new Histogram(data, explicitBins) : new Histogram(data);
}

// Histogram.adapt_* fixture args convention -- MUST mirror histogram_build_adapt() in
// core/tests/test_fixtures.cpp exactly: args = [explicit_bins, num_adds, data..., adds...].
static Histogram HistogramBuildAdapt(double[] a)
{
    int explicitBins = (int)a[0];
    int numAdds = (int)a[1];
    var data = a[2..(a.Length - numAdds)];
    var h = explicitBins > 0 ? new Histogram(data, explicitBins) : new Histogram(data);
    foreach (var value in a[^numAdds..]) h.AddData(value);
    return h;
}

// Bilinear.log_floor_value fixture args convention -- MUST mirror bilinear_log_floor_value()
// in core/tests/test_fixtures.cpp: args = [x1_query, x2_query] against a FIXED 3x3 identity
// grid ({0, 1E-15, 1} on both axes) with X1Transform/X2Transform/YTransform all Logarithmic.
static double BilinearLogFloorValue(double[] a)
{
    var coords = new[] { 0d, 1e-15, 1d };
    var y = new[,] { { 0d, 0d, 0d }, { 1e-15, 1e-15, 1e-15 }, { 1d, 1d, 1d } };
    var bilinear = new Bilinear(coords, coords, y)
    {
        X1Transform = Numerics.Data.Transform.Logarithmic,
        X2Transform = Numerics.Data.Transform.Logarithmic,
        YTransform = Numerics.Data.Transform.Logarithmic
    };
    return bilinear.Interpolate(a[0], a[1]);
}

// Probability.hpcm_* fixture args convention -- MUST mirror probability_hpcm_n()/
// probability_hpcm_joint() in core/tests/test_fixtures.cpp: args = [p_0..p_(n-1),
// ind_0..ind_(n-1), corr(n*n flattened row-major)] for hpcm_joint; hpcm_conditional_at
// appends one trailing 0-based component index.
static int ProbabilityHpcmN(int len)
{
    for (int n = 1; n <= 20; n++)
        if (2 * n + n * n == len) return n;
    throw new Exception("cannot infer n for Probability.hpcm args");
}

static double ProbabilityHpcmJoint(double[] a, double[]? conditional = null)
{
    int n = ProbabilityHpcmN(a.Length);
    var probabilities = a[..n];
    var indicators = new int[n];
    for (int i = 0; i < n; i++) indicators[i] = (int)a[n + i];
    var corr = new double[n, n];
    int baseIdx = 2 * n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            corr[i, j] = a[baseIdx + i * n + j];
    return Probability.JointProbabilityHPCM(probabilities, indicators, corr, conditional);
}

static double ProbabilityHpcmConditionalAt(double[] a)
{
    int idx = (int)a[^1];
    var body = a[..^1];
    var cond = new double[ProbabilityHpcmN(body.Length)];
    ProbabilityHpcmJoint(body, cond);
    return cond[idx];
}

// --- multivariate_distribution branch -----------------------------------------------------
// Mirrors the univariate Build/Dispatch split. Dirichlet/Multinomial/BivariateEmpirical have
// no common Mean/Variance/Covariance signature (unlike UnivariateDistributionBase), so
// DispatchMultivariate downcasts to the concrete type for anything beyond
// Dimension/PDF/LogPDF/CDF/ParametersValid. Extensible: additional multivariate targets add
// a case to each of Build/Dispatch.

static MultivariateDistribution BuildMultivariate(string target, JsonElement construct)
{
    if (target == "Dirichlet")
    {
        var alpha = construct.GetProperty("alpha").EnumerateArray().Select(ParseNum).ToArray();
        return new Dirichlet(alpha);
    }
    if (target == "Multinomial")
    {
        int n = construct.GetProperty("n").GetInt32();
        var p = construct.GetProperty("p").EnumerateArray().Select(ParseNum).ToArray();
        return new Multinomial(n, p);
    }
    if (target == "BivariateEmpirical")
    {
        var x1 = construct.GetProperty("x1").EnumerateArray().Select(ParseNum).ToArray();
        var x2 = construct.GetProperty("x2").EnumerateArray().Select(ParseNum).ToArray();
        var pRows = construct.GetProperty("p").EnumerateArray().ToArray();
        var p = new double[pRows.Length, x2.Length];
        for (int i = 0; i < pRows.Length; i++)
        {
            var row = pRows[i].EnumerateArray().Select(ParseNum).ToArray();
            for (int j = 0; j < row.Length; j++) p[i, j] = row[j];
        }
        Numerics.Data.Transform ParseT(string key) => construct.TryGetProperty(key, out var t)
            ? t.GetString() switch
            {
                "None" => Numerics.Data.Transform.None,
                "Logarithmic" => Numerics.Data.Transform.Logarithmic,
                "NormalZ" => Numerics.Data.Transform.NormalZ,
                var s => throw new Exception($"unknown transform: {s}")
            }
            : Numerics.Data.Transform.None;
        return new BivariateEmpirical(x1, x2, p, ParseT("x1_transform"), ParseT("x2_transform"),
                                       ParseT("p_transform"));
    }
    if (target == "MultivariateNormal")
    {
        var mean = construct.GetProperty("mean").EnumerateArray().Select(ParseNum).ToArray();
        var covRows = construct.GetProperty("covariance").EnumerateArray().ToArray();
        var covariance = new double[covRows.Length, mean.Length];
        for (int i = 0; i < covRows.Length; i++)
        {
            var row = covRows[i].EnumerateArray().Select(ParseNum).ToArray();
            for (int j = 0; j < row.Length; j++) covariance[i, j] = row[j];
        }
        var mvn = new MultivariateNormal(mean, covariance);
        if (construct.TryGetProperty("seed", out var seedEl))
            mvn.MVNUNI = new MersenneTwister(seedEl.GetInt32());
        if (construct.TryGetProperty("max_evaluations", out var maxEvalEl))
            mvn.MaxEvaluations = maxEvalEl.GetInt32();
        if (construct.TryGetProperty("abs_error", out var absErrEl))
            mvn.AbsoluteError = absErrEl.GetDouble();
        if (construct.TryGetProperty("rel_error", out var relErrEl))
            mvn.RelativeError = relErrEl.GetDouble();
        return mvn;
    }
    if (target == "MultivariateStudentT")
    {
        double df = construct.GetProperty("df").GetDouble();
        var location = construct.GetProperty("location").EnumerateArray().Select(ParseNum).ToArray();
        var scaleRows = construct.GetProperty("scale").EnumerateArray().ToArray();
        var scale = new double[scaleRows.Length, location.Length];
        for (int i = 0; i < scaleRows.Length; i++)
        {
            var row = scaleRows[i].EnumerateArray().Select(ParseNum).ToArray();
            for (int j = 0; j < row.Length; j++) scale[i, j] = row[j];
        }
        return new MultivariateStudentT(df, location, scale);
    }
    throw new Exception($"unknown multivariate target: {target}");
}

// Shared lookup for the "random_value"/"lhs_value" seeded-sampling oracle methods, common
// to every multivariate target that implements GenerateRandomValues (all four) /
// LatinHypercubeRandomValues (MultivariateNormal, MultivariateStudentT only -- see
// fixtures/README.md). args = [sample_size, seed, row, col]: `generate` is a method-group
// reference to the (int sampleSize, int seed) => double[,] overload itself, so this is
// stateless -- no persistent-instance batching needed, unlike MultivariateNormal's
// MVNUNI-seeded cdf/interval/mvndst path above.
static double SampleValueAt(Func<int, int, double[,]> generate, JsonElement[] a)
{
    var sample = generate(a[0].GetInt32(), a[1].GetInt32());
    return sample[a[2].GetInt32(), a[3].GetInt32()];
}

static double DispatchMultivariate(MultivariateDistribution d, string target, string m, JsonElement[] a)
{
    switch (m)
    {
        case "dimension": return d.Dimension;
        // The C# property for the PMF/PDF is PDF(double[]); LogPMF is a Multinomial-only
        // method that LogPDF forwards to, so LogPDF works generically across all three.
        case "pdf": return d.PDF(a[0].EnumerateArray().Select(ParseNum).ToArray());
        case "log_pdf": return d.LogPDF(a[0].EnumerateArray().Select(ParseNum).ToArray());
        case "cdf": return d.CDF(a[0].EnumerateArray().Select(ParseNum).ToArray());
    }
    if (target == "Dirichlet")
    {
        var dd = (Dirichlet)d;
        switch (m)
        {
            case "alpha": return dd.Alpha[a[0].GetInt32()];
            case "alpha_sum": return dd.AlphaSum;
            case "mean": return dd.Mean[a[0].GetInt32()];
            case "variance": return dd.Variance[a[0].GetInt32()];
            case "mode": return dd.Mode[a[0].GetInt32()];
            case "covariance": return dd.Covariance(a[0].GetInt32(), a[1].GetInt32());
            case "log_multivariate_beta": return Dirichlet.LogMultivariateBeta(a.Select(ParseNum).ToArray());
            case "random_value": return SampleValueAt(dd.GenerateRandomValues, a);
        }
    }
    else if (target == "Multinomial")
    {
        var mm = (Multinomial)d;
        switch (m)
        {
            case "number_of_trials": return mm.NumberOfTrials;
            case "mean": return mm.Mean[a[0].GetInt32()];
            case "variance": return mm.Variance[a[0].GetInt32()];
            case "covariance": return mm.Covariance(a[0].GetInt32(), a[1].GetInt32());
            case "random_value": return SampleValueAt(mm.GenerateRandomValues, a);
        }
    }
    else if (target == "BivariateEmpirical")
    {
        var bb = (BivariateEmpirical)d;
        if (m == "cdf_xy") return bb.CDF(a[0].GetDouble(), a[1].GetDouble());
        // v2.1.4: verifies the stale-cache fix in ONE self-contained call -- CDF() once
        // (forces the bilinear cache to build against the CURRENT grid), SetParameters()
        // with a REPLACEMENT grid, then CDF() again. args = [[x1_new...], [x2_new...],
        // [[p_row0...], ...], x1_eval, x2_eval].
        if (m == "cdf_xy_after_set_parameters")
        {
            bb.CDF(a[3].GetDouble(), a[4].GetDouble());
            var x1 = a[0].EnumerateArray().Select(ParseNum).ToArray();
            var x2 = a[1].EnumerateArray().Select(ParseNum).ToArray();
            var pRows = a[2].EnumerateArray().ToArray();
            var p = new double[pRows.Length, x2.Length];
            for (int i = 0; i < pRows.Length; i++)
            {
                var row = pRows[i].EnumerateArray().Select(ParseNum).ToArray();
                for (int j = 0; j < row.Length; j++) p[i, j] = row[j];
            }
            bb.SetParameters(x1, x2, p);
            return bb.CDF(a[3].GetDouble(), a[4].GetDouble());
        }
    }
    else if (target == "MultivariateNormal")
    {
        var nn = (MultivariateNormal)d;
        switch (m)
        {
            case "mean": return nn.Mean[a[0].GetInt32()];
            case "median": return nn.Median[a[0].GetInt32()];
            case "mode": return nn.Mode[a[0].GetInt32()];
            case "sd": return nn.StandardDeviation[a[0].GetInt32()];
            case "variance": return nn.Variance[a[0].GetInt32()];
            case "covariance": return nn.Covariance[a[0].GetInt32(), a[1].GetInt32()];
            case "mahalanobis": return nn.Mahalanobis(a[0].EnumerateArray().Select(ParseNum).ToArray());
            case "inverse_cdf":
            {
                // args = [[p_1..p_dim], index]
                var p = a[0].EnumerateArray().Select(ParseNum).ToArray();
                int idx = a[1].GetInt32();
                return nn.InverseCDF(p)[idx];
            }
            case "interval":
            {
                // args = [[lower...], [upper...]]
                var lower = a[0].EnumerateArray().Select(ParseNum).ToArray();
                var upper = a[1].EnumerateArray().Select(ParseNum).ToArray();
                return nn.Interval(lower, upper);
            }
            case "mvndst":
            {
                // args = [n, [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps]
                int n = a[0].GetInt32();
                var lower = a[1].EnumerateArray().Select(ParseNum).ToArray();
                var upper = a[2].EnumerateArray().Select(ParseNum).ToArray();
                var infin = a[3].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var correl = a[4].EnumerateArray().Select(ParseNum).ToArray();
                int maxpts = a[5].GetInt32();
                double abseps = a[6].GetDouble();
                double releps = a[7].GetDouble();
                double error = 0, val = 0;
                int inform = 0;
                nn.MVNDST(n, lower, upper, infin, correl, maxpts, abseps, releps, ref error, ref val, ref inform);
                return val;
            }
            case "random_value": return SampleValueAt(nn.GenerateRandomValues, a);
            case "lhs_value": return SampleValueAt(nn.LatinHypercubeRandomValues, a);
            // v2.1.4: MVNDST status-code cases assert INFORM/ERROR alongside VALUE
            // (same args shape as "mvndst" above; a separate case since DispatchMultivariate
            // returns one double per call).
            case "mvndst_inform":
            case "mvndst_error":
            {
                int n = a[0].GetInt32();
                var lower = a[1].EnumerateArray().Select(ParseNum).ToArray();
                var upper = a[2].EnumerateArray().Select(ParseNum).ToArray();
                var infin = a[3].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var correl = a[4].EnumerateArray().Select(ParseNum).ToArray();
                int maxpts = a[5].GetInt32();
                double abseps = a[6].GetDouble();
                double releps = a[7].GetDouble();
                double error = 0, val = 0;
                int inform = 0;
                nn.MVNDST(n, lower, upper, infin, correl, maxpts, abseps, releps, ref error, ref val, ref inform);
                return m == "mvndst_inform" ? inform : error;
            }
            // v2.1.4: IsDensityValid, as a double (1.0/0.0) for the uniform dispatch shape.
            case "is_density_valid": return nn.IsDensityValid ? 1.0 : 0.0;
            // v2.1.4: Marginal(indices)/Conditional(observedIndices, observedValues) --
            // args = [[indices...], out_index] / [[obs_indices...], [obs_values...], out_index]
            // (mean/dimension) or [[indices...], i, j] (covariance) or
            // [[indices...], [x...]] (log_pdf). Every case constructs the marginal/
            // conditional distribution fresh from `nn` and reads one scalar off it --
            // stateless, matching every other MultivariateNormal dispatch method here.
            case "marginal_mean":
            {
                var indices = a[0].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var marginal = nn.Marginal(indices);
                return marginal.Mean[a[1].GetInt32()];
            }
            case "marginal_covariance":
            {
                var indices = a[0].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var marginal = nn.Marginal(indices);
                return marginal.Covariance[a[1].GetInt32(), a[2].GetInt32()];
            }
            case "marginal_log_pdf":
            {
                var indices = a[0].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var marginal = nn.Marginal(indices);
                var x = a[1].EnumerateArray().Select(ParseNum).ToArray();
                return marginal.LogPDF(x);
            }
            case "marginal_dimension":
            {
                var indices = a[0].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                return nn.Marginal(indices).Dimension;
            }
            case "conditional_mean":
            {
                var obsIndices = a[0].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var obsValues = a[1].EnumerateArray().Select(ParseNum).ToArray();
                var conditional = nn.Conditional(obsIndices, obsValues);
                return conditional.Mean[a[2].GetInt32()];
            }
            case "conditional_covariance":
            {
                var obsIndices = a[0].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var obsValues = a[1].EnumerateArray().Select(ParseNum).ToArray();
                var conditional = nn.Conditional(obsIndices, obsValues);
                return conditional.Covariance[a[2].GetInt32(), a[3].GetInt32()];
            }
            case "conditional_dimension":
            {
                var obsIndices = a[0].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var obsValues = a[1].EnumerateArray().Select(ParseNum).ToArray();
                return nn.Conditional(obsIndices, obsValues).Dimension;
            }
        }
    }
    else if (target == "MultivariateStudentT")
    {
        var tt = (MultivariateStudentT)d;
        switch (m)
        {
            case "degrees_of_freedom": return tt.DegreesOfFreedom;
            case "mean": return tt.Mean[a[0].GetInt32()];
            case "median": return tt.Median[a[0].GetInt32()];
            case "mode": return tt.Mode[a[0].GetInt32()];
            case "sd": return tt.StandardDeviation[a[0].GetInt32()];
            case "variance": return tt.Variance[a[0].GetInt32()];
            case "covariance": return tt.Covariance[a[0].GetInt32(), a[1].GetInt32()];
            case "mahalanobis": return tt.Mahalanobis(a[0].EnumerateArray().Select(ParseNum).ToArray());
            case "inverse_cdf":
            {
                // args = [[p_1..p_dim+1], index]
                var p = a[0].EnumerateArray().Select(ParseNum).ToArray();
                int idx = a[1].GetInt32();
                return tt.InverseCDF(p)[idx];
            }
            case "random_value": return SampleValueAt(tt.GenerateRandomValues, a);
            case "lhs_value": return SampleValueAt(tt.LatinHypercubeRandomValues, a);
        }
    }
    throw new Exception($"unknown multivariate fixture method: {target}/{m}");
}

// --- bivariate_copula branch --------------------------------------------------------------
// Every copula shares BivariateCopula's uniform Theta/GetCopulaParameters/PDF/CDF/... API
// (unlike MultivariateDistribution, whose targets share no common surface), so BuildCopula/
// DispatchCopula are fully generic through Enum.Parse<CopulaType> + the real
// BivariateCopulaEstimation.Estimate static -- no per-target branching, mirroring the C++
// core's copula_factory.hpp rationale. The one exception is the "tau" method-of-moments fit
// (SetThetaFromTau is a member of each concrete Archimedean class in the C# source, not
// part of IBivariateCopula/IArchimedeanCopula), which SetThetaFromTauDispatch resolves by
// target name; each new tau-capable copula adds one branch there.

static BivariateCopula BuildCopula(string target, JsonElement construct,
                                    Dictionary<string, double[]> datasets)
{
    var type = Enum.Parse<CopulaType>(target);
    BivariateCopula copula = type switch
    {
        CopulaType.AliMikhailHaq => new AMHCopula(),
        CopulaType.Clayton => new ClaytonCopula(),
        CopulaType.Frank => new FrankCopula(),
        CopulaType.Gumbel => new GumbelCopula(),
        CopulaType.Joe => new JoeCopula(),
        CopulaType.Normal => new NormalCopula(),
        CopulaType.StudentT => new StudentTCopula(),
        _ => throw new Exception($"copula type not yet ported: {target}")
    };

    if (construct.TryGetProperty("theta", out var thetaEl))
    {
        var parms = new List<double> { ParseNum(thetaEl) };
        if (construct.TryGetProperty("df", out var dfEl)) parms.Add(ParseNum(dfEl));
        copula.SetCopulaParameters(parms.ToArray());
        // {"marginals": {"targets": [..], "params": [[..], [..]]}} attaches marginals
        // directly via the C# `Copula(theta, marginX, marginY)` ctor path -- used by the
        // seeded "random_value" sampling oracles, which back-transform through the
        // marginals when set. Distinct from the "fit"-construct's marginals (a bare
        // 2-element type-name array; see below), since this path sets FIXED marginal
        // parameters rather than fitting them.
        if (construct.TryGetProperty("marginals", out var directMarginalsEl))
        {
            var targets = directMarginalsEl.GetProperty("targets").EnumerateArray().ToArray();
            var paramArrays = directMarginalsEl.GetProperty("params").EnumerateArray().ToArray();
            var mx = UnivariateDistributionFactory.CreateDistribution(
                Enum.Parse<UnivariateDistributionType>(targets[0].GetString()!));
            var my = UnivariateDistributionFactory.CreateDistribution(
                Enum.Parse<UnivariateDistributionType>(targets[1].GetString()!));
            mx.SetParameters(paramArrays[0].EnumerateArray().Select(ParseNum).ToArray());
            my.SetParameters(paramArrays[1].EnumerateArray().Select(ParseNum).ToArray());
            copula.MarginalDistributionX = mx;
            copula.MarginalDistributionY = my;
        }
        return copula;
    }

    var fit = construct.GetProperty("fit");
    var x = datasets[fit.GetProperty("x").GetString()!];
    var y = datasets[fit.GetProperty("y").GetString()!];
    string method = fit.GetProperty("method").GetString()!;

    if (fit.TryGetProperty("marginals", out var marginalsEl))
    {
        var margArr = marginalsEl.EnumerateArray().ToArray();
        var mx = UnivariateDistributionFactory.CreateDistribution(
            Enum.Parse<UnivariateDistributionType>(margArr[0].GetString()!));
        var my = UnivariateDistributionFactory.CreateDistribution(
            Enum.Parse<UnivariateDistributionType>(margArr[1].GetString()!));
        if (method == "ifm")
        {
            ((IEstimation)mx).Estimate(x, ParameterEstimationMethod.MaximumLikelihood);
            ((IEstimation)my).Estimate(y, ParameterEstimationMethod.MaximumLikelihood);
        }
        copula.MarginalDistributionX = mx;
        copula.MarginalDistributionY = my;
    }

    switch (method)
    {
        case "tau":
            SetThetaFromTauDispatch(copula, target, x, y);
            break;
        case "mpl":
            BivariateCopulaEstimation.Estimate(ref copula, x, y, CopulaEstimationMethod.PseudoLikelihood);
            break;
        case "ifm":
            BivariateCopulaEstimation.Estimate(ref copula, x, y, CopulaEstimationMethod.InferenceFromMargins);
            break;
        case "mle":
            BivariateCopulaEstimation.Estimate(ref copula, x, y, CopulaEstimationMethod.FullLikelihood);
            break;
        default:
            throw new Exception($"unknown copula fit method: {method}");
    }
    return copula;
}

// SetThetaFromTau is not part of the shared copula API (see file header); each
// tau-capable copula type adds a branch here.
static void SetThetaFromTauDispatch(BivariateCopula copula, string target, double[] x, double[] y)
{
    if (target == "Clayton") { ((ClaytonCopula)copula).SetThetaFromTau(x, y); return; }
    if (target == "AliMikhailHaq") { ((AMHCopula)copula).SetThetaFromTau(x, y); return; }
    if (target == "Gumbel") { ((GumbelCopula)copula).SetThetaFromTau(x, y); return; }
    // NOTE: JoeCopula has no SetThetaFromTau in the C# source; intentionally not branched
    // here (see joe_copula.hpp's file header and .superpowers/sdd/task-8-report.md).
    throw new Exception($"copula '{target}' has no tau-based method-of-moments fit");
}

// The Weibull plotting positions of a sample, rank / (n + 1) over Statistics.RanksInPlace --
// transcribed from Test_ClaytonCopula.cs's Test_MPL_Fit (and repeated verbatim in every other
// Test_*Copula.cs MPL test and in BivariateCopulaEstimation.MLE). PseudoLogLikelihood is
// defined on values already on (0, 1) and does NOT rank internally, so its caller does this;
// mirrors dist_spec.hpp's support::plotting_positions, which the C++/R/Python runners apply
// inside the "log_likelihood_pseudo" arm.
static double[] CopulaPlottingPositions(double[] sample)
{
    var pp = Statistics.RanksInPlace(sample);
    for (int i = 0; i < pp.Length; i++) pp[i] = pp[i] / (pp.Length + 1d);
    return pp;
}

// `datasets` is consulted only by the three log_likelihood_* verbs, whose args name their two
// sample arrays rather than spelling 200 numbers per assertion (see fixtures/README.md and
// the matching copula_sample_args helpers in the C++/R/Python runners).
static double DispatchCopula(BivariateCopula c, string m, JsonElement[] a,
                             Dictionary<string, double[]> datasets)
{
    double[] Sample(int i)
    {
        string name = a[i].GetString()!;
        if (!datasets.TryGetValue(name, out var s))
            throw new Exception($"copula log-likelihood args name an unknown dataset: {name}");
        return s;
    }

    switch (m)
    {
        case "pdf": return c.PDF(a[0].GetDouble(), a[1].GetDouble());
        case "log_pdf": return c.LogPDF(a[0].GetDouble(), a[1].GetDouble());
        case "cdf": return c.CDF(a[0].GetDouble(), a[1].GetDouble());
        case "inverse_cdf": return c.InverseCDF(a[0].GetDouble(), a[1].GetDouble())[a[2].GetInt32()];
        case "upper_tail_dependence": return c.UpperTailDependence;
        case "lower_tail_dependence": return c.LowerTailDependence;
        case "theta": return c.Theta;
        case "df": return c.GetCopulaParameters[1];
        case "or_exceedance": return c.ORJointExceedanceProbability(a[0].GetDouble(), a[1].GetDouble());
        case "and_exceedance": return c.ANDJointExceedanceProbability(a[0].GetDouble(), a[1].GetDouble());
        case "theta_minimum": return c.ThetaMinimum;
        case "theta_maximum": return c.ThetaMaximum;
        // args = ["<x dataset>", "<y dataset>"]: the RAW paired observations for all three.
        // Only the pseudo arm transforms them (to plotting positions); IFM and the full
        // likelihood push them through the attached marginals' CDF/LogPDF themselves.
        case "log_likelihood_pseudo":
            return c.PseudoLogLikelihood(CopulaPlottingPositions(Sample(0)), CopulaPlottingPositions(Sample(1)));
        case "log_likelihood_ifm": return c.IFMLogLikelihood(Sample(0), Sample(1));
        case "log_likelihood_full": return c.LogLikelihood(Sample(0), Sample(1));
        case "marginal_param":
        {
            string which = a[0].GetString()!;
            int idx = a[1].GetInt32();
            var marg = which == "x" ? c.MarginalDistributionX : c.MarginalDistributionY;
            return marg!.GetParameters[idx];
        }
        case "random_value":
        {
            // args = [sample_size, seed, row, col]. Stateless: GenerateRandomValues seeds
            // its own internal LatinHypercube draw from `seed`, so no persistent-instance
            // batching is needed (mirrors SampleValueAt() for MultivariateDistribution above).
            var sample = c.GenerateRandomValues(a[0].GetInt32(), a[1].GetInt32());
            return sample[a[2].GetInt32(), a[3].GetInt32()];
        }
        default: throw new Exception($"unknown copula fixture method: {m}");
    }
}

// --- mcmc_sampler helpers (Task P3.5 / P3.6) ----------------------------------------------
//
// The model builder mirrors Test_RWMH.cs's Test_RWMH_NormalDist_RStan construction VERBATIM
// for "uniform_constraints", and Test_Gibbs.cs's Test_Gibbs_NormalDist_RStan (v2.1.4 rework:
// ConditionalMeanParameters/ConditionalVarianceParameters) for "normal_conjugate_gibbs" -- see
// model_registry.hpp's header comment for the v2.1.4 split this mirrors: `muInitializationPrior`/
// `sigmaInitializationPrior` seed only `priors` (feasibility bounds), while `conditionalMean`/
// `conditionalVariance` are separate, freshly constructed locals the proposal closure alone
// mutates every Gibbs step.
static (List<IUnivariateDistribution> priors, LogLikelihood logLikelihood, Gibbs.Proposal? proposal) BuildMcmcModel(
    string name, string family, double[] data)
{
    if (name == "uniform_constraints")
    {
        // Family-generic (mirrors model_registry.hpp's C++ "uniform_constraints" entry,
        // which builds any `family` via the factory + IMaximumLikelihoodEstimation rather
        // than being Normal-only): uninformative Uniform priors bounded by `family`'s own
        // GetParameterConstraints(data) lower/upper arrays, and a log-likelihood closure
        // that refits a fresh `family` instance's parameters at each proposal. Originally
        // hard-coded to Normal only (P3.5, Test_RWMH_NormalDist_RStan); P3.6's
        // ARWMH/SNIS rstan cases need Logistic/Gumbel/Weibull too (Test_ARWMH.cs).
        var probe = UnivariateDistributionFactory.CreateDistribution(Enum.Parse<UnivariateDistributionType>(family));
        if (probe is not IMaximumLikelihoodEstimation imle)
            throw new Exception($"BuildMcmcModel: family '{family}' does not implement " +
                                 "IMaximumLikelihoodEstimation");
        var constraints = imle.GetParameterConstraints(data);
        var priors = new List<IUnivariateDistribution>();
        for (int i = 0; i < constraints.Item2.Length; i++)
            priors.Add(new Uniform(constraints.Item2[i], constraints.Item3[i]));
        double LogLH(double[] x)
        {
            var dist = UnivariateDistributionFactory.CreateDistribution(Enum.Parse<UnivariateDistributionType>(family));
            dist.SetParameters(x);
            return dist.LogLikelihood(data);
        }
        return (priors, LogLH, null);
    }
    if (name == "normal_conjugate_gibbs")
    {
        if (family != "Normal")
            throw new Exception($"BuildMcmcModel: family '{family}' not supported for " +
                                 "'normal_conjugate_gibbs' (mirrors Test_Gibbs.cs, Normal-only)");
        int n = data.Length;
        var mu = Statistics.Mean(data);
        double mu0 = 0, sigma0 = 5E5;
        // Preserve the reference model's limiting non-informative inverse-gamma prior for
        // the conditional variance update.
        double variancePriorShape = 0d, variancePriorScale = 0d;
        // Proper distributions used SOLELY to initialize the sampler state (feasibility
        // bounds / initial draws) -- see model_registry.hpp's v2.1.4 split note.
        double initializationShape = 2d, initializationScale = 0.001d;
        var muInitializationPrior = new Normal(mu0, sigma0);
        var sigmaInitializationPrior = new InverseGamma(initializationScale, initializationShape);
        var conditionalMean = new Normal();
        var conditionalVariance = new InverseGamma();
        var priors = new List<IUnivariateDistribution> { muInitializationPrior, sigmaInitializationPrior };

        double LogLH(double[] x)
        {
            var dist = new Normal(x[0], x[1]);
            return dist.LogLikelihood(data);
        }

        double[] Proposal(double[] x, Random random)
        {
            // Sample the conditional mean given the current standard deviation
            // (ConditionalMeanParameters): the corrected conjugate-Normal posterior
            // mean/variance, replacing the pre-v2.1.4 `mu0 / 2` bug.
            double likelihoodVariance = x[1] * x[1];
            double priorVariance = sigma0 * sigma0;
            double posteriorVariance = 1d / (n / likelihoodVariance + 1d / priorVariance);
            double posteriorMean = posteriorVariance * (n * mu / likelihoodVariance + mu0 / priorVariance);
            conditionalMean.SetParameters(posteriorMean, Math.Sqrt(posteriorVariance));
            double mup = conditionalMean.InverseCDF(random.NextDouble());

            // Sample the conditional variance (ConditionalVarianceParameters), then return
            // its square root as sigma.
            double sumOfSquaredErrors = 0;
            for (int i = 0; i < data.Length; i++)
            {
                double residual = data[i] - mup;
                sumOfSquaredErrors += residual * residual;
            }
            double scale = variancePriorScale + sumOfSquaredErrors / 2d;
            double shape = variancePriorShape + n / 2d;
            conditionalVariance.SetParameters(new double[] { scale, shape });
            double sig2p = conditionalVariance.InverseCDF(random.NextDouble());

            return new double[] { mup, Math.Sqrt(sig2p) };
        }

        return (priors, LogLH, Proposal);
    }
    throw new Exception($"unknown MCMC model registry entry: {name}");
}

// `proposal_sigma` sentinel strings -- see fixtures/README.md's mcmc_sampler schema for why
// "identity" exists alongside the C# test's literal "zeros" (an all-zero proposal covariance
// is only safe when MAP initialization is expected to overwrite it before first use).
static Matrix ParseProposalSigma(JsonElement settings, int dimension)
{
    if (!settings.TryGetProperty("proposal_sigma", out var ps)) return new Matrix(dimension);
    return ps.GetString() switch
    {
        "zeros" => new Matrix(dimension),
        "identity" => Matrix.Identity(dimension),
        var s => throw new Exception($"unknown proposal_sigma sentinel: {s}")
    };
}

static MCMCSampler.InitializationType ParseInitialize(string s) => s switch
{
    "MAP" => MCMCSampler.InitializationType.MAP,
    "Randomize" => MCMCSampler.InitializationType.Randomize,
    "UserDefined" => MCMCSampler.InitializationType.UserDefined,
    _ => throw new Exception($"unknown initialize value: {s}")
};

// Builds + configures + samples() one sampler from a {"model": {...}, "settings": {...}}
// construct. `samplerTarget`: the fixture's file-level "target" (the sampler type, e.g.
// "RWMH"); a later task extends this with more cases as more samplers land.
static MCMCSampler BuildAndSampleMcmc(string samplerTarget, JsonElement construct,
                                       Dictionary<string, double[]> datasets)
{
    var modelSpec = construct.GetProperty("model");
    var data = datasets[modelSpec.GetProperty("dataset").GetString()!];
    var (priors, logLikelihood, proposal) = BuildMcmcModel(modelSpec.GetProperty("name").GetString()!,
        modelSpec.GetProperty("family").GetString()!, data);
    int d = priors.Count;

    bool hasSettings = construct.TryGetProperty("settings", out var settings);

    MCMCSampler sampler = samplerTarget switch
    {
        "RWMH" => new RWMH(priors, logLikelihood, hasSettings ? ParseProposalSigma(settings, d) : new Matrix(d)),
        "HMC" => new HMC(priors, logLikelihood,
            stepSize: hasSettings && settings.TryGetProperty("step_size", out var hss) ? hss.GetDouble() : 0.1,
            steps: hasSettings && settings.TryGetProperty("steps", out var hst) ? hst.GetInt32() : 50),
        "NUTS" => new NUTS(priors, logLikelihood,
            stepSize: hasSettings && settings.TryGetProperty("step_size", out var nss) ? nss.GetDouble() : 0.1,
            maxTreeDepth: hasSettings && settings.TryGetProperty("max_tree_depth", out var nmtd) ? nmtd.GetInt32() : 10),
        "ARWMH" => new ARWMH(priors, logLikelihood),
        "Gibbs" => new Gibbs(priors, logLikelihood,
            proposal ?? throw new Exception("Gibbs model has no proposal function")),
        "SNIS" => new SNIS(priors, logLikelihood),
        "DEMCz" => new DEMCz(priors, logLikelihood),
        "DEMCzs" => new DEMCzs(priors, logLikelihood),
        _ => throw new Exception($"unknown mcmc_sampler target: {samplerTarget}")
    };

    if (hasSettings)
    {
        if (settings.TryGetProperty("initialize", out var init)) sampler.Initialize = ParseInitialize(init.GetString()!);
        if (settings.TryGetProperty("prng_seed", out var seed)) sampler.PRNGSeed = seed.GetInt32();
        if (settings.TryGetProperty("initial_iterations", out var ii)) sampler.InitialIterations = ii.GetInt32();
        if (settings.TryGetProperty("warmup_iterations", out var wi)) sampler.WarmupIterations = wi.GetInt32();
        if (settings.TryGetProperty("iterations", out var it)) sampler.Iterations = it.GetInt32();
        if (settings.TryGetProperty("number_of_chains", out var nc)) sampler.NumberOfChains = nc.GetInt32();
        if (settings.TryGetProperty("thinning_interval", out var ti)) sampler.ThinningInterval = ti.GetInt32();
        if (settings.TryGetProperty("output_length", out var ol)) sampler.OutputLength = ol.GetInt32();
        if (sampler is ARWMH arwmh)
        {
            if (settings.TryGetProperty("scale", out var sc)) arwmh.Scale = sc.GetDouble();
            if (settings.TryGetProperty("beta", out var be)) arwmh.Beta = be.GetDouble();
        }
        if (sampler is DEMCz demcz)
        {
            if (settings.TryGetProperty("jump", out var jp)) demcz.Jump = jp.GetDouble();
            if (settings.TryGetProperty("jump_threshold", out var jt)) demcz.JumpThreshold = jt.GetDouble();
            if (settings.TryGetProperty("noise", out var ns)) demcz.Noise = ns.GetDouble();
        }
        if (sampler is DEMCzs demczs)
        {
            if (settings.TryGetProperty("jump", out var jp)) demczs.Jump = jp.GetDouble();
            if (settings.TryGetProperty("jump_threshold", out var jt)) demczs.JumpThreshold = jt.GetDouble();
            if (settings.TryGetProperty("snooker_threshold", out var st)) demczs.SnookerThreshold = st.GetDouble();
            if (settings.TryGetProperty("noise", out var ns)) demczs.Noise = ns.GetDouble();
        }
        if (sampler is NUTS nuts)
        {
            if (settings.TryGetProperty("adapt_mass_matrix", out var amm)) nuts.AdaptMassMatrix = amm.GetBoolean();
        }
    }

    sampler.Sample();
    return sampler;
}

static double DispatchMcmc(MCMCSampler sampler, MCMCResults results, string m, JsonElement[] a)
{
    int Idx(int i) => a[i].GetInt32();
    switch (m)
    {
        case "posterior_mean": return results.ParameterResults[Idx(0)].SummaryStatistics.Mean;
        case "posterior_sd": return results.ParameterResults[Idx(0)].SummaryStatistics.StandardDeviation;
        case "posterior_median": return results.ParameterResults[Idx(0)].SummaryStatistics.Median;
        case "posterior_lower_ci": return results.ParameterResults[Idx(0)].SummaryStatistics.LowerCI;
        case "posterior_upper_ci": return results.ParameterResults[Idx(0)].SummaryStatistics.UpperCI;
        case "chain_value": return sampler.MarkovChains[Idx(0)][Idx(1)].Values[Idx(2)];
        case "chain_fitness": return sampler.MarkovChains[Idx(0)][Idx(1)].Fitness;
        case "map_value": return results.MAP.Values[Idx(0)];
        case "map_fitness": return results.MAP.Fitness;
        case "acceptance_rate": return sampler.AcceptanceRates[Idx(0)];
        case "mean_log_likelihood": return sampler.MeanLogLikelihood[Idx(0)];
        case "rhat": return results.ParameterResults[Idx(0)].SummaryStatistics.Rhat;
        case "ess": return results.ParameterResults[Idx(0)].SummaryStatistics.ESS;
        default: throw new Exception($"unknown mcmc_sampler fixture method: {m}");
    }
}

// --- bootstrap helpers (Task P3.10) -------------------------------------------------------
//
// Mirrors model_registry.hpp's "normal_quantiles" entry EXACTLY -- see that header's comment
// for the sampleData/BCa semantics (transcribed from Test_Bootstrap.cs's private
// CreateNormalBootstrap() helper and Test_BCaCI()).
static Bootstrap<double[]> BuildBootstrapModel(string name, double mu, double sigma, int sampleSize,
    double[] probabilities, double[] sampleData)
{
    if (name != "normal_quantiles")
        throw new Exception($"unknown bootstrap model registry entry: {name}");

    double fitMu = mu, fitSigma = sigma;
    int resampleSize = sampleSize;
    double[]? originalData = null;

    if (sampleData.Length > 0)
    {
        var probe = new Normal();
        ((IEstimation)probe).Estimate(sampleData, ParameterEstimationMethod.MethodOfMoments);
        fitMu = probe.Mu;
        fitSigma = probe.Sigma;
        resampleSize = sampleData.Length;
        originalData = sampleData;
    }

    var parms = new ParameterSet(new double[] { fitMu, fitSigma }, double.NaN);
    var boot = new Bootstrap<double[]>(originalData!, parms);

    boot.ResampleFunction = (data, ps, rng) =>
    {
        var d = new Normal(ps.Values[0], ps.Values[1]);
        return d.GenerateRandomValues(resampleSize, rng.Next());
    };

    boot.FitFunction = (sample) =>
    {
        var d = new Normal();
        ((IEstimation)d).Estimate(sample, ParameterEstimationMethod.MethodOfMoments);
        if (!d.ParametersValid) throw new Exception("Invalid parameters.");
        return new ParameterSet(d.GetParameters, double.NaN);
    };

    boot.StatisticFunction = (ps) =>
    {
        var d = new Normal(ps.Values[0], ps.Values[1]);
        var result = new double[probabilities.Length];
        for (int i = 0; i < probabilities.Length; i++)
            result[i] = d.InverseCDF(probabilities[i]);
        return result;
    };

    if (sampleData.Length > 0)
    {
        boot.JackknifeFunction = (data, idx) =>
        {
            var list = new List<double>(data);
            list.RemoveAt(idx);
            return list.ToArray();
        };
        boot.SampleSizeFunction = (data) => data.Length;
    }

    return boot;
}

static BootstrapCIMethod ParseBootstrapCIMethod(string s) => s switch
{
    "Percentile" => BootstrapCIMethod.Percentile,
    "BiasCorrected" => BootstrapCIMethod.BiasCorrected,
    "BCa" => BootstrapCIMethod.BCa,
    "Normal" => BootstrapCIMethod.Normal,
    "BootstrapT" => BootstrapCIMethod.BootstrapT,
    _ => throw new Exception($"unknown bootstrap ci_method: {s}")
};

// Builds + configures + runs one Bootstrap<double[]> from a {"model": ..., ...} construct
// (see fixtures/README.md's bootstrap schema), then computes its confidence intervals once.
static (Bootstrap<double[]> boot, BootstrapResults results) BuildAndRunBootstrap(
    JsonElement construct, Dictionary<string, double[]> datasets)
{
    string modelName = construct.GetProperty("model").GetString()!;
    double mu = construct.TryGetProperty("mu", out var muEl) ? muEl.GetDouble() : 0.0;
    double sigma = construct.TryGetProperty("sigma", out var sigmaEl) ? sigmaEl.GetDouble() : 0.0;
    int sampleSize = construct.TryGetProperty("sample_size", out var ssEl) ? ssEl.GetInt32() : 0;
    double[] probabilities = construct.GetProperty("probabilities").EnumerateArray()
        .Select(x => x.GetDouble()).ToArray();
    double[] sampleData = construct.TryGetProperty("dataset", out var dsEl)
        ? datasets[dsEl.GetString()!] : Array.Empty<double>();

    var boot = BuildBootstrapModel(modelName, mu, sigma, sampleSize, probabilities, sampleData);
    if (construct.TryGetProperty("replicates", out var repEl)) boot.Replicates = repEl.GetInt32();
    if (construct.TryGetProperty("seed", out var seedEl)) boot.PRNGSeed = seedEl.GetInt32();
    if (construct.TryGetProperty("max_retries", out var mrEl)) boot.MaxRetries = mrEl.GetInt32();

    string run = construct.TryGetProperty("run", out var runEl) ? runEl.GetString()! : "regular";
    if (run == "regular") boot.Run();
    else if (run == "studentized") boot.RunWithStudentizedBootstrap();
    else throw new Exception($"unknown bootstrap run kind: {run}");

    var method = ParseBootstrapCIMethod(construct.GetProperty("ci_method").GetString()!);
    double alpha = construct.TryGetProperty("alpha", out var alphaEl) ? alphaEl.GetDouble() : 0.1;
    var results = boot.GetConfidenceIntervals(method, alpha);

    return (boot, results);
}

static double DispatchBootstrap(Bootstrap<double[]> boot, BootstrapResults results, string m, JsonElement[] a)
{
    int Idx(int i) => a[i].GetInt32();
    switch (m)
    {
        case "statistic_lower_ci": return results.StatisticResults[Idx(0)].LowerCI;
        case "statistic_upper_ci": return results.StatisticResults[Idx(0)].UpperCI;
        case "parameter_lower_ci": return results.ParameterResults[Idx(0)].LowerCI;
        case "parameter_upper_ci": return results.ParameterResults[Idx(0)].UpperCI;
        case "population_estimate": return results.ParameterResults[Idx(0)].PopulationEstimate;
        case "valid_count": return results.StatisticResults[Idx(0)].ValidCount;
        case "replicate_value": return boot.BootstrapParameterSets[Idx(0)].Values[Idx(1)];
        default: throw new Exception($"unknown bootstrap fixture method: {m}");
    }
}

// --- model_estimation helpers (Task T12) --------------------------------------------------
//
// Drives the REAL RMC.BestFit estimators (MaximumLikelihood / MaximumAPosteriori /
// BayesianAnalysis), subset-compiled in place (see OracleEmitter.csproj). One build+run per
// case (mirrors the mcmc_sampler/bootstrap single-stateful-glue-call contract); every assertion
// dispatches against that one cached estimator. Method-name strings match the C++/R/Python
// runners EXACTLY so the same fixture file drives all four.
static OptimizationMethod ParseOptimizationMethod(string s) => s switch
{
    "Brent" => OptimizationMethod.Brent,
    "BFGS" => OptimizationMethod.BFGS,
    "NelderMead" => OptimizationMethod.NelderMead,
    "Powell" => OptimizationMethod.Powell,
    "DifferentialEvolution" => OptimizationMethod.DifferentialEvolution,
    "MultilevelSingleLinkage" => OptimizationMethod.MultilevelSingleLinkage,
    _ => throw new Exception($"unknown model_estimation optimizer: {s}")
};

static BayesianAnalysis.SamplerType ParseSamplerType(string s) => s switch
{
    "DEMCz" => BayesianAnalysis.SamplerType.DEMCz,
    "DEMCzs" => BayesianAnalysis.SamplerType.DEMCzs,
    "ARWMH" => BayesianAnalysis.SamplerType.ARWMH,
    "NUTS" => BayesianAnalysis.SamplerType.NUTS,
    _ => throw new Exception($"unknown model_estimation sampler: {s}")
};

// --- GMM / Bulletin17C helpers (Task B12) -------------------------------------------------
//
// Drives the REAL RMC.BestFit GeneralizedMethodOfMoments over a concrete Bulletin17CDistribution
// (un-excluded from the subset by B12; see OracleEmitter.csproj). Mirrors the C++ runner's GMM
// path in core/tests/test_fixtures.cpp (build_and_run_estimation's GeneralizedMethodOfMoments
// arm) EXACTLY so the same fixture file drives all four harnesses: build the B17C model from the
// `construct.model` spec, construct GMM (default optimizer = BFGS, the C# GMM ctor default),
// apply the strategy/iterations knobs, Estimate() once, then PostProcess(sandwich: true,
// computeJstat: true) so the accessors return deterministic cached Sigma + J-statistic. The
// seeded-draw digest (optional `sample_size`/`seed`) pins the fitted best parameters into the
// B17C parent and takes one ISimulatable stream -- the same DRY choice the C++ runner makes.
static GeneralizedMethodOfMoments.GMMEstimationStrategy ParseGmmStrategy(string s) => s switch
{
    "OneStep" => GeneralizedMethodOfMoments.GMMEstimationStrategy.OneStep,
    "TwoStep" => GeneralizedMethodOfMoments.GMMEstimationStrategy.TwoStep,
    "Iterative" => GeneralizedMethodOfMoments.GMMEstimationStrategy.Iterative,
    _ => throw new Exception($"unknown GMM estimation strategy: {s}")
};

// A `type: "bulletin17c"` model spec -> a concrete Bulletin17CDistribution, mirroring
// model_spec.hpp's build_bulletin17c_model: family (default LogPearsonTypeIII) + the shared
// DataFrame builder + optional explicit parameter_values applied last.
static BestFitModels.Bulletin17CDistribution BuildBulletin17CModel(
    JsonElement modelSpec, Dictionary<string, double[]> datasets)
{
    var df = BuildModelDataFrame(modelSpec, datasets);
    var distType = modelSpec.TryGetProperty("family", out var fam)
        ? Enum.Parse<UnivariateDistributionType>(fam.GetString()!)
        : UnivariateDistributionType.LogPearsonTypeIII;
    var m = new BestFitModels.Bulletin17CDistribution(df, distType);
    // Bulletin17CDistribution is an IGMMModel, not an IModel, so it gets the parameter-list
    // overload directly; it has no UseDefaultFlatPriors property either.
    ApplyParameterOverrides(m.Parameters, modelSpec);
    if (modelSpec.TryGetProperty("parameter_values", out var pv))
        m.SetParameterValues(pv.EnumerateArray().Select(ParseNum).ToList());
    return m;
}

static (BestFitModels.Bulletin17CDistribution b17c, GeneralizedMethodOfMoments gmm, double[]? simulated)
    BuildGmm(JsonElement construct, Dictionary<string, double[]> datasets)
{
    var b17c = BuildBulletin17CModel(construct.GetProperty("model"), datasets);
    var method = construct.TryGetProperty("optimizer", out var o)
        ? ParseOptimizationMethod(o.GetString()!) : OptimizationMethod.BFGS;
    var gmm = new GeneralizedMethodOfMoments(b17c, method);
    if (construct.TryGetProperty("strategy", out var st))
        gmm.EstimationStrategy = ParseGmmStrategy(st.GetString()!);
    if (construct.TryGetProperty("max_gmm_iterations", out var mgi))
        gmm.MaxGMMIterations = mgi.GetInt32();
    if (!gmm.Estimate())
        throw new Exception("GeneralizedMethodOfMoments.Estimate() returned false");
    gmm.PostProcess(useSandwich: true, computeJstat: true);
    double[]? draws = null;
    if (construct.TryGetProperty("sample_size", out var ss))
    {
        b17c.SetParameterValues(gmm.BestParameterSet.Values);
        draws = b17c.GenerateRandomValues(ss.GetInt32(),
            construct.TryGetProperty("seed", out var se) ? se.GetInt32() : -1);
    }
    return (b17c, gmm, draws);
}

// GMM assertion surface: shares parameter/standard_error/covariance/correlation names with
// ML/MAP; adds j_stat/j_stat_pval and the B17C quantile_variance. quantile_variance lives on the
// B17C MODEL (not the estimator): args[0] is the annual EXCEEDANCE probability (AEP) and the C#
// QuantileVariance takes a NON-exceedance probability, so pass 1 - AEP, feeding it the fitted
// parameters + the estimator's covariance -- exactly the C++ runner's arm.
static double DispatchGmm(BestFitModels.Bulletin17CDistribution b17c, GeneralizedMethodOfMoments gmm,
                          double[]? simulated, string m, JsonElement[] a)
{
    int I(int i) => a[i].GetInt32();
    switch (m)
    {
        case "simulated_value":
            return (simulated ?? throw new Exception("simulated_value outside a seeded GMM case"))[I(0)];
        case "parameter": return gmm.BestParameterSet.Values[I(0)];
        case "standard_error": return gmm.GetStandardErrors()[I(0)];
        case "covariance": return gmm.GetCovarianceMatrix()[I(0), I(1)];
        case "correlation": return gmm.GetCorrelationMatrix()[I(0), I(1)];
        case "j_stat": return gmm.JStat;
        case "j_stat_pval": return gmm.JStatPval;
        // T13: GMMIterations/ConvergedWithinTolerance (off-by-one fix) and
        // OptimizerFallbackCount (sticky BFGS->NelderMead fallback; internal, accessible here
        // because Program.cs compiles into the same subset assembly as the real
        // GeneralizedMethodOfMoments.cs -- see OracleEmitter.csproj's B12 note).
        case "gmm_iterations": return gmm.GMMIterations;
        case "converged_within_tolerance": return gmm.ConvergedWithinTolerance ? 1.0 : 0.0;
        case "optimizer_fallback_count": return gmm.OptimizerFallbackCount;
        case "quantile_variance":
            return b17c.QuantileVariance(1.0 - a[0].GetDouble(), gmm.BestParameterSet.Values,
                gmm.GetCovarianceMatrix().ToArray());
        default: throw new Exception($"unknown GMM fixture method: {m}");
    }
}

// --- Phase 5 model-spec construction (Task M14) ---------------------------------------------
//
// Builds the SAME `construct.model` spec the three runners hand to the shared C++ builder
// (core/include/corehydro/models/model_spec.hpp; schema in fixtures/README.md's model_estimation
// section) against the REAL RMC.BestFit model classes, so one fixture file drives all four
// harnesses. `type` dispatch, the `data_frame` inline arrays, `trends`, and `parameter_values`
// mirror the C++ builder's semantics exactly. A spec without `type`/`data_frame`/`trends`
// builds exactly what the Phase 4 emitter built (DataFrame { ExactSeries } +
// UnivariateDistribution) -- byte-for-byte.

// A `{ "family": ..., "parameters": [...] }` distribution spec -> a parameterized
// distribution through the same factory every other fixture kind uses (mirrors
// model_spec.hpp's build_spec_distribution).
static UnivariateDistributionBase BuildSpecDistribution(JsonElement spec)
{
    var dist = UnivariateDistributionFactory.CreateDistribution(
        Enum.Parse<UnivariateDistributionType>(spec.GetProperty("family").GetString()!));
    if (spec.TryGetProperty("parameters", out var ps))
        dist.SetParameters(ps.EnumerateArray().Select(ParseNum).ToArray());
    return dist;
}

// A `data_frame` spec object -> a real RMC.BestFit DataFrame. Threshold processing happens at
// the model boundary (every model's DataFrame setter runs ProcessThresholdSeries itself),
// exactly like the C++ port. The optional `mgbt_low_outliers` flag (M14) triggers the PUBLIC
// SetLowOutliersFromMGBT() path at the frame boundary, before the model ctor -- flagging low
// outliers and setting LowOutlierThreshold, which left-censors the flagged values in the fit.
static BestFitModels.DataFrame BuildSpecDataFrame(JsonElement dfSpec)
{
    var df = new BestFitModels.DataFrame();
    if (dfSpec.TryGetProperty("exact", out var exactEl))
        df.ExactSeries = new ExactSeries(exactEl.EnumerateArray().Select(e => new ExactData(
            e.GetProperty("index").GetInt32(), e.GetProperty("value").GetDouble(), 0d,
            e.TryGetProperty("is_low_outlier", out var lo) && lo.GetBoolean())).ToList());
    if (dfSpec.TryGetProperty("interval", out var intervalEl))
        df.IntervalSeries = new IntervalSeries(intervalEl.EnumerateArray().Select(e =>
            new IntervalData(e.GetProperty("index").GetInt32(), e.GetProperty("lower").GetDouble(),
                e.GetProperty("value").GetDouble(), e.GetProperty("upper").GetDouble())).ToList());
    if (dfSpec.TryGetProperty("threshold", out var thresholdEl))
        df.ThresholdSeries = new ThresholdSeries(thresholdEl.EnumerateArray().Select(e =>
            new ThresholdData(e.GetProperty("start_index").GetInt32(),
                              e.GetProperty("end_index").GetInt32(),
                              e.GetProperty("value").GetDouble())
            { NumberAbove = e.GetProperty("number_above").GetInt32() }).ToList());
    if (dfSpec.TryGetProperty("uncertain", out var uncertainEl))
        df.UncertainSeries = new UncertainSeries(uncertainEl.EnumerateArray().Select(e =>
            new UncertainData(e.GetProperty("index").GetInt32(),
                              BuildSpecDistribution(e.GetProperty("distribution")))).ToList());
    if (dfSpec.TryGetProperty("low_outlier_threshold", out var lot))
        df.LowOutlierThreshold = lot.GetDouble();
    if (dfSpec.TryGetProperty("mgbt_low_outliers", out var mgbt) && mgbt.GetBoolean())
        df.SetLowOutliersFromMGBT();
    if (dfSpec.TryGetProperty("threshold_low_outliers", out var tlo) && tlo.GetBoolean())
        df.SetLowOutliersFromThreshold();
    // Mirrors model_spec.hpp's build_data_frame: a spec describes a finished frame, so leave it
    // fully computed. BestFit expects the caller (in the app, the INotifyPropertyChanged
    // cascade) to recompute the Hirsch-Stedinger positions after any edit; a headless caller
    // must do it explicitly, or the Bulletin17C ROS imputation reads positions that are still 0.
    df.CalculatePlottingPositions();
    return df;
}

// Resolves a model spec's data source: the inline `data_frame` object when present, otherwise
// an exact-only frame over the file-level `dataset` values (the Phase 4 path, byte-for-byte).
static BestFitModels.DataFrame BuildModelDataFrame(JsonElement modelSpec,
                                                   Dictionary<string, double[]> datasets)
{
    if (modelSpec.TryGetProperty("data_frame", out var dfSpec)) return BuildSpecDataFrame(dfSpec);
    if (modelSpec.TryGetProperty("dataset", out var ds))
        return new BestFitModels.DataFrame { ExactSeries = new ExactSeries(datasets[ds.GetString()!]) };
    throw new Exception("model spec requires either 'dataset' or 'data_frame'");
}

// `families` -> distribution types (mixture / competing_risks component lists).
static List<UnivariateDistributionType> ParseFamilies(JsonElement modelSpec) =>
    modelSpec.GetProperty("families").EnumerateArray()
        .Select(f => Enum.Parse<UnivariateDistributionType>(f.GetString()!)).ToList();

// The `construct.model` dispatch (mirrors model_spec.hpp's build_model): `type` defaults to
// "univariate_distribution" (the Phase 4 behavior). All four model types derive from
// UnivariateDistributionModelBase, which carries the DataFrame property the M14 data-frame
// assertion methods (plotting_position / number_of_low_outliers / low_outlier_threshold) read.
static BestFitModels.UnivariateDistributionModelBase BuildSpecModel(
    JsonElement modelSpec, Dictionary<string, double[]> datasets)
{
    string type = modelSpec.TryGetProperty("type", out var t)
        ? t.GetString()! : "univariate_distribution";
    BestFitModels.UnivariateDistributionModelBase model;
    if (type == "univariate_distribution")
    {
        var distType = Enum.Parse<UnivariateDistributionType>(modelSpec.GetProperty("family").GetString()!);
        var ud = new BestFitModels.UnivariateDistribution(
            BuildModelDataFrame(modelSpec, datasets), distType);
        if (modelSpec.TryGetProperty("trends", out var trendsEl))
        {
            var trends = trendsEl.EnumerateArray().ToArray();
            ud.IsNonstationary = true;

            // Pass 1: attach every trend (SetTrendModel supplies the data-driven defaults),
            // then override the anchor where the spec asks -- mirroring model_spec.hpp.
            foreach (var tr in trends)
            {
                int p = tr.GetProperty("parameter").GetInt32();
                ud.SetTrendModel(p, Enum.Parse<RMC.BestFit.Models.TrendFunctions.Support.TrendModelType>(
                    tr.GetProperty("type").GetString()!));
                if (tr.TryGetProperty("start_index", out var si))
                    ud.TrendModels[p].StartIndex = si.GetInt32();
            }

            // Pass 2 (after the parameter layout is final): explicit per-trend values
            // overwrite their slice of the full parameter vector, applied through ONE
            // SetParameterValues call (the sync-safe setter the model mandates).
            bool hasValues = false;
            var full = ud.Parameters.Select(mp => mp.Value).ToList();
            foreach (var tr in trends)
            {
                if (!tr.TryGetProperty("values", out var valuesEl)) continue;
                hasValues = true;
                int p = tr.GetProperty("parameter").GetInt32();
                int offset = 0;
                for (int j = 0; j < p; j++) offset += ud.TrendModels[j].NumberOfParameters;
                var values = valuesEl.EnumerateArray().Select(ParseNum).ToArray();
                if (values.Length != ud.TrendModels[p].NumberOfParameters)
                    throw new Exception("trend spec 'values' length does not match the trend's parameter count");
                for (int k = 0; k < values.Length; k++) full[offset + k] = values[k];
            }
            if (hasValues) ud.SetParameterValues(full);
        }
        model = ud;
    }
    else if (type == "mixture")
    {
        model = new BestFitModels.MixtureModel(BuildModelDataFrame(modelSpec, datasets),
            ParseFamilies(modelSpec),
            modelSpec.TryGetProperty("zero_inflated", out var zi) && zi.GetBoolean());
    }
    else if (type == "competing_risks")
    {
        model = new BestFitModels.CompetingRisksModel(BuildModelDataFrame(modelSpec, datasets),
            ParseFamilies(modelSpec));
    }
    else if (type == "point_process")
    {
        // Default-construct (non-seasonal GEV competing-risks distribution), assign the frame,
        // then the optional knobs in C#-property order: UseDefaults before the explicit
        // Threshold/TotalYears so an explicit value is never clobbered by the defaults cascade.
        var pp = new BestFitModels.PointProcessModel();
        pp.DataFrame = BuildModelDataFrame(modelSpec, datasets);
        if (modelSpec.TryGetProperty("use_defaults", out var udFlag)) pp.UseDefaults = udFlag.GetBoolean();
        if (modelSpec.TryGetProperty("threshold", out var th)) pp.Threshold = th.GetDouble();
        if (modelSpec.TryGetProperty("total_years", out var ty)) pp.TotalYears = ty.GetDouble();
        model = pp;
    }
    else
    {
        throw new Exception($"unknown model_estimation model type: {type}");
    }

    // Optional model-level `use_default_flat_priors` + `parameters` + `parameter_values`,
    // applied in that order (see model_spec.hpp's apply_parameter_values).
    if (modelSpec.TryGetProperty("use_default_flat_priors", out var udfp))
        model.UseDefaultFlatPriors = udfp.GetBoolean();
    ApplyParameterOverrides(model.Parameters, modelSpec);
    if (modelSpec.TryGetProperty("parameter_values", out var pv))
        model.SetParameterValues(pv.EnumerateArray().Select(ParseNum).ToList());
    return model;
}

// Builds the real BestFit model from a case's `construct.model` (see BuildSpecModel) and
// returns it together with the already-run estimator selected by `target` (null plus the
// cached seeded draw vector for the estimator-less `Simulation` target). For
// BayesianAnalysis, applies the fixture's `sampler` + `settings` knobs (UseSimulationDefaults /
// UseAdvancedSimulationDefaults set false FIRST so the explicit values aren't clobbered by the
// ctor's defaulting), then runs synchronously via `RunAsync(null, false, parallel: false)` --
// `parallel: false` sets `Sampler.ParallelizeChains = false` so the chain generation is serial,
// matching the C++ port (which has no ParallelizeChains) so the seeded chain digest reproduces
// bit-identically. (DIC/WAIC/LOOIC still use Parallel.For internally; see the fixture note.)
static (BestFitModels.UnivariateDistributionModelBase model, object? estimator, double[]? simulated)
    BuildEstimation(string target, JsonElement construct, Dictionary<string, double[]> datasets)
{
    var model = BuildSpecModel(construct.GetProperty("model"), datasets);

    if (target == "Simulation")
    {
        // No estimator: ONE seeded ISimulatable draw cached at build time (M13/M14); the
        // `simulated_value [i]` method asserts individual draws (the chain_value digest
        // precedent -- the C# stream is the oracle R/Python must reproduce bit-identically).
        if (model is not BestFitModels.ISimulatable<double[]> sim)
            throw new Exception("model_estimation Simulation target: model is not ISimulatable");
        var draws = sim.GenerateRandomValues(construct.GetProperty("sample_size").GetInt32(),
            construct.TryGetProperty("seed", out var se) ? se.GetInt32() : -1);
        return (model, null, draws);
    }
    // Validate (Task 16 on the IModel path; widened to this path in Task 21): builds the model
    // and stops -- no estimator, no seeded draw.
    if (target == "Validate")
    {
        return (model, null, null);
    }
    if (target == "MaximumLikelihood")
    {
        var method = construct.TryGetProperty("optimizer", out var o)
            ? ParseOptimizationMethod(o.GetString()!) : OptimizationMethod.DifferentialEvolution;
        var mle = new MaximumLikelihood(model, method);
        if (!mle.Estimate()) throw new Exception("MaximumLikelihood.Estimate() returned false");
        return (model, mle, null);
    }
    if (target == "MaximumAPosteriori")
    {
        var method = construct.TryGetProperty("optimizer", out var o)
            ? ParseOptimizationMethod(o.GetString()!) : OptimizationMethod.DifferentialEvolution;
        var map = new MaximumAPosteriori(model, method);
        if (!map.Estimate()) throw new Exception("MaximumAPosteriori.Estimate() returned false");
        return (model, map, null);
    }
    if (target == "BayesianAnalysis")
    {
        var samplerType = construct.TryGetProperty("sampler", out var s)
            ? ParseSamplerType(s.GetString()!) : BayesianAnalysis.SamplerType.DEMCzs;
        var ba = new BayesianAnalysis(model, samplerType)
        {
            UseSimulationDefaults = false,
            UseAdvancedSimulationDefaults = false,
        };
        if (construct.TryGetProperty("settings", out var settings))
        {
            if (settings.TryGetProperty("seed", out var seedEl)) ba.PRNGSeed = seedEl.GetInt32();
            if (settings.TryGetProperty("iterations", out var itEl)) ba.Iterations = itEl.GetInt32();
            if (settings.TryGetProperty("warmup_iterations", out var wiEl)) ba.WarmupIterations = wiEl.GetInt32();
            if (settings.TryGetProperty("number_of_chains", out var ncEl)) ba.NumberOfChains = ncEl.GetInt32();
            if (settings.TryGetProperty("thinning_interval", out var tiEl)) ba.ThinningInterval = tiEl.GetInt32();
            if (settings.TryGetProperty("initial_iterations", out var iiEl)) ba.InitialIterations = iiEl.GetInt32();
            if (settings.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        }
        ba.RunAsync(null, false, false).GetAwaiter().GetResult();
        return (model, ba, null);
    }
    throw new Exception($"unknown model_estimation target: {target}");
}

// --- Phase 7a model families (Task P4) -----------------------------------------------------
//
// The four remaining ModelBase families -- TimeSeries (AR/MA/ARIMA/ARIMAX), SpatialGEV,
// RatingCurve, BivariateDistribution -- derive from ModelBase/IModel, NOT
// UnivariateDistributionModelBase, so they take a SEPARATE build + estimation path from the
// Phase 4-6 univariate path above (BuildSpecModel / BuildEstimation stay byte-for-byte
// unchanged). This mirrors core/include/corehydro/models/model_spec.hpp field-for-field: the
// `construct.model.type` string selects the family and the schema is fixtures/README.md's
// model_estimation section. The three runners already build these families through their shared
// model_spec.hpp path; this emitter path is the fourth (oracle) leg.
//
// TimeSeries note: the C# TimeSeries index is a DateTime, but every model consumer touches the
// index only as a sequence position / inner-join key (never calendar arithmetic -- see the C++
// adapter header time_series.hpp). All series in a case are built from ONE fixed base date with
// the same interval, so their relative alignment (rating_curve stage<->discharge, ARIMAX
// covariate lags) is preserved exactly as the C++ integer-index adapter preserves it; the
// absolute `start_index` is therefore not modeled here (documented deviation, fit-invariant).
static DateTime EmitterSeriesEpoch() => new DateTime(2000, 1, 1);

static BestFitModels.Transform ParseTransform(string s) => s switch
{
    "None" => BestFitModels.Transform.None,
    "Logarithmic" => BestFitModels.Transform.Logarithmic,
    "BoxCox" => BestFitModels.Transform.BoxCox,
    "YeoJohnson" => BestFitModels.Transform.YeoJohnson,
    _ => throw new Exception($"unknown time_series transform: {s}")
};

static TimeInterval ParseTimeInterval(string s) => s switch
{
    "OneMinute" => TimeInterval.OneMinute,
    "FiveMinute" => TimeInterval.FiveMinute,
    "FifteenMinute" => TimeInterval.FifteenMinute,
    "ThirtyMinute" => TimeInterval.ThirtyMinute,
    "OneHour" => TimeInterval.OneHour,
    "SixHour" => TimeInterval.SixHour,
    "TwelveHour" => TimeInterval.TwelveHour,
    "OneDay" => TimeInterval.OneDay,
    "SevenDay" => TimeInterval.SevenDay,
    "OneMonth" => TimeInterval.OneMonth,
    "OneQuarter" => TimeInterval.OneQuarter,
    "OneYear" => TimeInterval.OneYear,
    "Irregular" => TimeInterval.Irregular,
    _ => throw new Exception($"unknown time_series time_interval: {s}")
};

// Wraps a flat value vector into a real Numerics.Data.TimeSeries (interval + start date + values).
// `time_interval` defaults OneDay (the C# field default); the start date is the fixed epoch (see
// the region note -- absolute start_index is fit-invariant given all series share it).
static TimeSeries BuildEmitterTimeSeries(JsonElement modelSpec, double[] values)
{
    TimeInterval interval = modelSpec.TryGetProperty("time_interval", out var ti)
        ? ParseTimeInterval(ti.GetString()!) : TimeInterval.OneDay;
    // Irregular is rejected by the (interval, startDate, data) ctor (it walks a REGULAR step), in
    // C# exactly as in the C++ port. Build it the way the C# regression tests do -- empty series
    // on the interval, then one Add per value -- matching models/model_spec.hpp's build_time_series
    // ordinate-for-ordinate. Index spacing is inert (the ARMA families index by position).
    if (interval == TimeInterval.Irregular)
    {
        var irregular = new TimeSeries(interval);
        for (int i = 0; i < values.Length; i++)
            irregular.Add(new SeriesOrdinate<DateTime, double>(
                EmitterSeriesEpoch().AddDays(i), values[i]));
        return irregular;
    }
    return new TimeSeries(interval, EmitterSeriesEpoch(), values);
}

// Resolves a time-series data source: inline `data` array, else the file-level dataset.
static double[] EmitterTimeSeriesValues(JsonElement spec, Dictionary<string, double[]> datasets)
{
    if (spec.TryGetProperty("data", out var d)) return d.EnumerateArray().Select(ParseNum).ToArray();
    if (spec.TryGetProperty("dataset", out var ds)) return datasets[ds.GetString()!];
    throw new Exception("time_series model requires either 'dataset' or 'data'");
}

// Optional model-level `parameters` block -> per-parameter bounds / fixed flag / prior
// distribution / starting value, written onto the ModelParameter elements the model built.
// Mirrors model_spec.hpp's apply_parameter_overrides; entries name their target by 0-based
// `index` into the model's own parameter list, so the block composes with `trends` and runs
// BEFORE the flat `parameter_values` vector.
static void ApplyParameterOverrides(List<BestFitModels.ModelParameter> parameters, JsonElement spec)
{
    if (!spec.TryGetProperty("parameters", out var block)) return;
    foreach (var entry in block.EnumerateArray())
    {
        int index = entry.GetProperty("index").GetInt32();
        if (index < 0 || index >= parameters.Count)
            throw new Exception("parameter spec 'index' is out of range for this model");
        var mp = parameters[index];
        if (entry.TryGetProperty("lower", out var lo)) mp.LowerBound = lo.GetDouble();
        if (entry.TryGetProperty("upper", out var hi)) mp.UpperBound = hi.GetDouble();
        if (entry.TryGetProperty("is_positive", out var ip)) mp.IsPositive = ip.GetBoolean();
        if (entry.TryGetProperty("is_fixed", out var isf)) mp.IsFixed = isf.GetBoolean();
        if (entry.TryGetProperty("name", out var nm)) mp.Name = nm.GetString()!;
        if (entry.TryGetProperty("prior", out var pr)) mp.PriorDistribution = BuildSpecDistribution(pr);
        if (entry.TryGetProperty("value", out var val)) mp.Value = val.GetDouble();
    }
}

// Optional model-level `parameter_values` -> ONE sync-safe SetParameterValues call (the setter
// every model mandates; poking Parameters directly desyncs trend / covariate copies).
static void ApplyGeneralParameterValues(BestFitModels.IModel m, JsonElement spec)
{
    if (spec.TryGetProperty("use_default_flat_priors", out var udfp))
        m.UseDefaultFlatPriors = udfp.GetBoolean();
    ApplyParameterOverrides(m.Parameters, spec);
    if (spec.TryGetProperty("parameter_values", out var pv))
        m.SetParameterValues(pv.EnumerateArray().Select(ParseNum).ToList());
}

static int OrderOf(JsonElement model, string key, int dflt) =>
    model.TryGetProperty("orders", out var o) && o.TryGetProperty(key, out var v) ? v.GetInt32() : dflt;

// `type: "time_series"` -- subtype selects AutoRegressive / MovingAverage / ARIMA / ARIMAX.
// Mirrors model_spec.hpp's build_time_series_model: ctor + transform, then (ARIMAX) the
// include-intercept / trend / seasonality / order / covariate setters, then parameter_values.
static BestFitModels.IModel BuildTimeSeriesModelGeneral(
    JsonElement model, Dictionary<string, double[]> datasets)
{
    string subtype = model.GetProperty("subtype").GetString()!;
    var ts = BuildEmitterTimeSeries(model, EmitterTimeSeriesValues(model, datasets));
    bool includeIntercept = !model.TryGetProperty("include_intercept", out var ii) || ii.GetBoolean();

    BestFitModels.IModel result;
    if (subtype == "ar")
    {
        var m = new BestFitModels.AutoRegressive(ts, OrderOf(model, "p", 1), includeIntercept);
        if (model.TryGetProperty("transform", out var tr)) m.TransformType = ParseTransform(tr.GetString()!);
        result = m;
    }
    else if (subtype == "ma")
    {
        var m = new BestFitModels.MovingAverage(ts, OrderOf(model, "q", 1), includeIntercept);
        if (model.TryGetProperty("transform", out var tr)) m.TransformType = ParseTransform(tr.GetString()!);
        result = m;
    }
    else if (subtype == "arima")
    {
        var m = new BestFitModels.ARIMA(ts, OrderOf(model, "p", 1), OrderOf(model, "d", 0),
                                        OrderOf(model, "q", 0), includeIntercept);
        if (model.TryGetProperty("transform", out var tr)) m.TransformType = ParseTransform(tr.GetString()!);
        result = m;
    }
    else if (subtype == "arimax")
    {
        var m = new BestFitModels.ARIMAX(ts);
        if (model.TryGetProperty("transform", out var tr)) m.TransformType = ParseTransform(tr.GetString()!);
        m.IncludeIntercept = includeIntercept;
        if (model.TryGetProperty("trend", out var trend))
            m.TrendType = trend.GetString()! switch
            {
                "Linear" => BestFitModels.ARIMAX.Trend.Linear,
                "Quadratic" => BestFitModels.ARIMAX.Trend.Quadratic,
                "Cubic" => BestFitModels.ARIMAX.Trend.Cubic,
                _ => BestFitModels.ARIMAX.Trend.None,
            };
        if (model.TryGetProperty("include_seasonality", out var seas)) m.IncludeSeasonality = seas.GetBoolean();
        m.AROrderP = OrderOf(model, "p", 1);
        m.DiffOrderD = OrderOf(model, "d", 0);
        m.MAOrderQ = OrderOf(model, "q", 0);
        m.XOrderB = OrderOf(model, "b", 0);
        if (model.TryGetProperty("covariates", out var covs))
        {
            var list = new List<TimeSeries>();
            foreach (var c in covs.EnumerateArray())
                list.Add(BuildEmitterTimeSeries(model, c.EnumerateArray().Select(ParseNum).ToArray()));
            m.SetCovariates(list);
        }
        result = m;
    }
    else
    {
        throw new Exception($"unknown time_series subtype: {subtype}");
    }
    // Optional `training_time_steps` (mirrors model_spec.hpp): a MODEL property whose
    // data-driven default exceeds the series length for any series shorter than 30.
    if (model.TryGetProperty("training_time_steps", out var tsSteps))
    {
        int steps = tsSteps.GetInt32();
        switch (result)
        {
            case BestFitModels.AutoRegressive arModel: arModel.TrainingTimeSteps = steps; break;
            case BestFitModels.MovingAverage maModel: maModel.TrainingTimeSteps = steps; break;
            case BestFitModels.ARIMAX arimaxModel: arimaxModel.TrainingTimeSteps = steps; break;
            case BestFitModels.ARIMA arimaModel: arimaModel.TrainingTimeSteps = steps; break;
        }
    }

    ApplyGeneralParameterValues(result, model);
    return result;
}

// `type: "spatial_gev"` -- SpatialGEV(atSiteData [obs,sites], coordinates [sites,2], three
// intercept-only GeneralLinearFunction level-2 trends). Optional gating flags applied after
// construction (their ctor defaults: link=true, errors/copula=false). Mirrors
// model_spec.hpp's build_spatial_gev_model.
static BestFitModels.IModel BuildSpatialGevModelGeneral(
    JsonElement model, Dictionary<string, double[]> datasets)
{
    var rows = model.GetProperty("at_site_data").EnumerateArray()
        .Select(r => r.EnumerateArray().Select(ParseNum).ToArray()).ToArray();
    int obs = rows.Length, sites = rows[0].Length;
    var atSite = new double[obs, sites];
    for (int i = 0; i < obs; i++)
        for (int j = 0; j < sites; j++) atSite[i, j] = rows[i][j];

    var coordRows = model.GetProperty("coordinates").EnumerateArray()
        .Select(r => r.EnumerateArray().Select(ParseNum).ToArray()).ToArray();
    var coords = new double[sites, 2];
    for (int i = 0; i < sites; i++)
        for (int j = 0; j < 2; j++) coords[i, j] = coordRows[i][j];

    var location = new BestFitModels.TrendFunctions.GeneralLinearFunction("Location");
    var scale = new BestFitModels.TrendFunctions.GeneralLinearFunction("Scale");
    var shape = new BestFitModels.TrendFunctions.GeneralLinearFunction("Shape");
    var m = new BestFitModels.SpatialExtremes.SpatialGEV(atSite, coords, location, scale, shape);
    if (model.TryGetProperty("use_copula_dependence", out var ucd)) m.UseCopulaDependence = ucd.GetBoolean();
    if (model.TryGetProperty("use_location_errors", out var ule)) m.UseLocationErrors = ule.GetBoolean();
    if (model.TryGetProperty("use_scale_errors", out var use)) m.UseScaleErrors = use.GetBoolean();
    if (model.TryGetProperty("use_shape_errors", out var ushp)) m.UseShapeErrors = ushp.GetBoolean();
    if (model.TryGetProperty("use_log_link_for_location", out var ull)) m.UseLogLinkForLocation = ull.GetBoolean();
    if (model.TryGetProperty("use_log_link_for_scale", out var ulls)) m.UseLogLinkForScale = ulls.GetBoolean();
    ApplyGeneralParameterValues(m, model);
    return m;
}

// `type: "rating_curve"` -- RatingCurve(stage, discharge, segments). Both series share the epoch
// + interval so the date inner-join aligns them 1:1 (>= MinimumAlignedObservations = 10).
static BestFitModels.IModel BuildRatingCurveModelGeneral(
    JsonElement model, Dictionary<string, double[]> datasets)
{
    var stage = BuildEmitterTimeSeries(model, model.GetProperty("stage").EnumerateArray().Select(ParseNum).ToArray());
    var discharge = BuildEmitterTimeSeries(model, model.GetProperty("discharge").EnumerateArray().Select(ParseNum).ToArray());
    int segments = model.TryGetProperty("segments", out var s) ? s.GetInt32() : 1;
    var m = new BestFitModels.RatingCurve(stage, discharge, segments);
    ApplyGeneralParameterValues(m, model);
    return m;
}

// A bivariate marginal spec -> a pre-fit UnivariateDistribution (an IUnivariateModel). Carries
// its own inline `data` (exact series) and pinned distribution `parameter_values` (marginals
// stay FIXED during the copula fit -- B1). Mirrors model_spec.hpp's build_bivariate_marginal.
static BestFitModels.IUnivariateModel BuildBivariateMarginalGeneral(JsonElement spec)
{
    var distType = Enum.Parse<UnivariateDistributionType>(spec.GetProperty("family").GetString()!);
    var df = new BestFitModels.DataFrame
    {
        ExactSeries = new ExactSeries(spec.GetProperty("data").EnumerateArray().Select(ParseNum).ToArray())
    };
    var m = new BestFitModels.UnivariateDistribution(df, distType);
    if (spec.TryGetProperty("parameter_values", out var pv))
        m.SetParameterValues(pv.EnumerateArray().Select(ParseNum).ToList());
    return m;
}

// `type: "bivariate"` -- a copula-coupled BivariateDistribution: two pre-fit IUnivariateModel
// marginals (held FIXED), a CopulaType, and a CopulaEstimationMethod (default
// InferenceFromMargins). Mirrors model_spec.hpp's build_bivariate_model.
static BestFitModels.IModel BuildBivariateModelGeneral(
    JsonElement model, Dictionary<string, double[]> datasets)
{
    var mx = BuildBivariateMarginalGeneral(model.GetProperty("marginal_x"));
    var my = BuildBivariateMarginalGeneral(model.GetProperty("marginal_y"));
    var copulaType = Enum.Parse<CopulaType>(
        model.TryGetProperty("copula", out var cp) ? cp.GetString()! : "Normal");
    var m = new BestFitModels.BivariateDistribution(mx, my, copulaType);
    if (model.TryGetProperty("estimation_method", out var em))
        m.CopulaEstimationMethod = Enum.Parse<CopulaEstimationMethod>(em.GetString()!);
    ApplyGeneralParameterValues(m, model);
    return m;
}

static BestFitModels.IModel BuildSpecModelGeneral(
    JsonElement modelSpec, Dictionary<string, double[]> datasets)
{
    string type = modelSpec.GetProperty("type").GetString()!;
    return type switch
    {
        "time_series" => BuildTimeSeriesModelGeneral(modelSpec, datasets),
        "spatial_gev" => BuildSpatialGevModelGeneral(modelSpec, datasets),
        "rating_curve" => BuildRatingCurveModelGeneral(modelSpec, datasets),
        "bivariate" => BuildBivariateModelGeneral(modelSpec, datasets),
        _ => throw new Exception($"unknown general model_estimation model type: {type}")
    };
}

// Seeded ISimulatable draw flattened to a 1-D vector so the `simulated_value [i]` digest works
// uniformly. Five families are ISimulatable<double[]>; BivariateDistribution is
// ISimulatable<double[,]> (n-row x 2-col) -- flattened ROW-MAJOR (i = row*cols + col), matching
// the C++/R/Python simulate_flat and the README schema.
static double[] SimulateFlatGeneral(BestFitModels.IModel model, int sampleSize, int seed)
{
    if (model is BestFitModels.ISimulatable<double[]> s1)
        return s1.GenerateRandomValues(sampleSize, seed);
    if (model is BestFitModels.ISimulatable<double[,]> s2)
    {
        var mat = s2.GenerateRandomValues(sampleSize, seed);
        int r = mat.GetLength(0), c = mat.GetLength(1);
        var flat = new double[r * c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++) flat[i * c + j] = mat[i, j];
        return flat;
    }
    throw new Exception("model is neither ISimulatable<double[]> nor ISimulatable<double[,]>");
}

// Builds a Phase 7a family model and runs the estimator selected by `target`. Mirrors
// BuildEstimation but over IModel (the four families are not UnivariateDistributionModelBase).
static (BestFitModels.IModel model, object? estimator, double[]? simulated)
    BuildEstimationGeneral(string target, JsonElement construct, Dictionary<string, double[]> datasets)
{
    var model = BuildSpecModelGeneral(construct.GetProperty("model"), datasets);

    if (target == "Simulation")
    {
        var draws = SimulateFlatGeneral(model, construct.GetProperty("sample_size").GetInt32(),
            construct.TryGetProperty("seed", out var se) ? se.GetInt32() : -1);
        return (model, null, draws);
    }
    // Validate (Task 16): builds the model and stops -- no estimator, no seeded draw. Lets a
    // case assert `is_valid`/`validation_message_contains` (below) against IModel.Validate()
    // without needing to fit or simulate, e.g. the TimeSeries transform-lambda-failure cases.
    if (target == "Validate")
    {
        return (model, null, null);
    }
    if (target == "MaximumLikelihood" || target == "MaximumAPosteriori")
    {
        var method = construct.TryGetProperty("optimizer", out var o)
            ? ParseOptimizationMethod(o.GetString()!) : OptimizationMethod.DifferentialEvolution;
        object est;
        ParameterSet best;
        if (target == "MaximumLikelihood")
        {
            var mle = new MaximumLikelihood(model, method);
            if (!mle.Estimate()) throw new Exception("MaximumLikelihood.Estimate() returned false");
            est = mle; best = mle.BestParameterSet;
        }
        else
        {
            var map = new MaximumAPosteriori(model, method);
            if (!map.Estimate()) throw new Exception("MaximumAPosteriori.Estimate() returned false");
            est = map; best = map.BestParameterSet;
        }
        double[]? draws = null;
        if (construct.TryGetProperty("sample_size", out var ss))
        {
            // Pin the fitted best parameters back into the model, then take one seeded draw --
            // the same shared `simulated_value` arm the Simulation target uses (P3 pattern).
            model.SetParameterValues(best.Values);
            draws = SimulateFlatGeneral(model, ss.GetInt32(),
                construct.TryGetProperty("seed", out var se) ? se.GetInt32() : -1);
        }
        return (model, est, draws);
    }
    if (target == "BayesianAnalysis")
    {
        var samplerType = construct.TryGetProperty("sampler", out var s)
            ? ParseSamplerType(s.GetString()!) : BayesianAnalysis.SamplerType.DEMCzs;
        var ba = new BayesianAnalysis(model, samplerType)
        {
            UseSimulationDefaults = false,
            UseAdvancedSimulationDefaults = false,
        };
        if (construct.TryGetProperty("settings", out var settings))
        {
            if (settings.TryGetProperty("seed", out var seedEl)) ba.PRNGSeed = seedEl.GetInt32();
            if (settings.TryGetProperty("iterations", out var itEl)) ba.Iterations = itEl.GetInt32();
            if (settings.TryGetProperty("warmup_iterations", out var wiEl)) ba.WarmupIterations = wiEl.GetInt32();
            if (settings.TryGetProperty("number_of_chains", out var ncEl)) ba.NumberOfChains = ncEl.GetInt32();
            if (settings.TryGetProperty("thinning_interval", out var tiEl)) ba.ThinningInterval = tiEl.GetInt32();
            if (settings.TryGetProperty("initial_iterations", out var iiEl)) ba.InitialIterations = iiEl.GetInt32();
            if (settings.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        }
        ba.RunAsync(null, false, false).GetAwaiter().GetResult();
        return (model, ba, null);
    }
    throw new Exception($"unknown model_estimation target: {target}");
}

// Task 9: the wider ML/MAP fit surface, shared by BOTH dispatch paths (DispatchMlMap over
// UnivariateDistributionModelBase and DispatchMlMapGeneral over IModel) so the two cannot drift.
// Returns null when `m` is not one of these methods, letting the caller fall through to its own
// switch. Every accessor is LAZY: ProfileLikelihood/ParameterConfidenceIntervals cost
// bins * n_params likelihood evaluations each, so a case that asserts none of them pays nothing.
//
// `profile_bins` and `alpha` come from the case CONSTRUCT (they are ProfileLikelihood(bins) /
// ParameterConfidenceIntervals(alpha) arguments, not per-assertion arguments), with the same
// defaults fit_runner.hpp's run_fit applies -- 100 bins and alpha 0.1, which are also the C#
// method defaults. `status_is [name]` compares OptimizationStatus.ToString() and returns
// 1.0/0.0 (the `is_valid`/`validation_message_contains` boolean-as-double precedent -- the
// fixture schema carries no string comparison).
//
// This function is called once per ASSERTION (the outer per-case loop calls DispatchMlMap/
// DispatchMlMapGeneral fresh for every assertion), so without memoization a case asserting
// several `profile_value`/`profile_lower`/`profile_upper` entries -- e.g. fit_profile.json's
// eleven profile_value + four profile_lower/upper assertions -- recomputes the full profile
// sweep or the confidence-interval solve from scratch on every one. `TotalFunctionEvaluations`
// is a snapshot property set once inside Estimate(), not a live counter, so the redundant calls
// were correct, just wasteful. Cache by `best` (the case's BestParameterSet instance, stable for
// the case's lifetime -- a fresh instance is built per case, and ParameterSet has no
// Equals/GetHashCode override, so reference identity is exactly "per case").
static List<double[,]> MemoizedProfileLikelihood(ParameterSet key, int bins, Func<int, List<double[,]>> compute)
{
    if (!ProfileFitCache.Profile.TryGetValue(key, out var byBins))
    {
        byBins = new Dictionary<int, List<double[,]>>();
        ProfileFitCache.Profile[key] = byBins;
    }
    if (!byBins.TryGetValue(bins, out var v))
    {
        v = compute(bins);
        byBins[bins] = v;
    }
    return v;
}

static double[,] MemoizedParameterCIs(ParameterSet key, double alpha, Func<double, double[,]> compute)
{
    if (!ProfileFitCache.Cis.TryGetValue(key, out var byAlpha))
    {
        byAlpha = new Dictionary<double, double[,]>();
        ProfileFitCache.Cis[key] = byAlpha;
    }
    if (!byAlpha.TryGetValue(alpha, out var v))
    {
        v = compute(alpha);
        byAlpha[alpha] = v;
    }
    return v;
}

static double? DispatchMlMapExtended(string m, JsonElement[] a, JsonElement construct,
    BestFitModels.IModel model, ParameterSet best,
    Func<int, List<double[,]>> profileLikelihood, Func<double, double[,]> parameterCIs,
    Func<int> totalFunctionEvaluations, Func<OptimizationStatus> status)
{
    int I(int i) => a[i].GetInt32();
    int bins = construct.TryGetProperty("profile_bins", out var pb) ? pb.GetInt32() : 100;
    double alpha = construct.TryGetProperty("alpha", out var al) ? al.GetDouble() : 0.1;
    switch (m)
    {
        case "profile_lower": return MemoizedParameterCIs(best, alpha, parameterCIs)[I(0), 0];
        case "profile_upper": return MemoizedParameterCIs(best, alpha, parameterCIs)[I(0), 1];
        // profile_value [param, bin, col]: ProfileLikelihood returns one [bins, 2] array per
        // parameter, col 0 = the bin midpoint parameter value, col 1 = the log-likelihood there.
        case "profile_value": return MemoizedProfileLikelihood(best, bins, profileLikelihood)[I(0)][I(1), I(2)];
        case "function_evaluations": return totalFunctionEvaluations();
        case "status_is": return status().ToString() == a[0].GetString() ? 1.0 : 0.0;
        case "nobs": return model.PointwiseDataLogLikelihood(best.Values).Length;
        case "prior_log_likelihood": return model.PriorLogLikelihood(best.Values);
        default: return null;
    }
}

// ML/MAP dispatch with LAZY accessors (covariance/SE/correlation are computed only when the
// method asks -- GetCovarianceMatrix throws for a 1-parameter model, e.g. the Normal copula).
static double DispatchMlMapGeneral(string m, JsonElement[] a, JsonElement construct,
    BestFitModels.IModel model, ParameterSet best,
    Func<int, double> param, Func<double> maxLL, Func<double> aic, Func<int, double> bic,
    Func<int, int, double> cov, Func<int, double> se, Func<int, int, double> corr,
    Func<int, List<double[,]>> profileLikelihood, Func<double, double[,]> parameterCIs,
    Func<int> totalFunctionEvaluations, Func<OptimizationStatus> status)
{
    int I(int i) => a[i].GetInt32();
    var extended = DispatchMlMapExtended(m, a, construct, model, best, profileLikelihood,
                                         parameterCIs, totalFunctionEvaluations, status);
    if (extended.HasValue) return extended.Value;
    switch (m)
    {
        case "parameter": return param(I(0));
        case "max_log_likelihood": return maxLL();
        case "aic": return aic();
        case "bic": return bic(I(0));
        case "covariance": return cov(I(0), I(1));
        case "standard_error": return se(I(0));
        case "correlation": return corr(I(0), I(1));
        default: throw new Exception($"unknown model_estimation method for ML/MAP: {m}");
    }
}

static double DispatchEstimationGeneral(
    (BestFitModels.IModel model, object? estimator, double[]? simulated) ec,
    string m, JsonElement[] a, JsonElement construct)
{
    if (m == "simulated_value")
        return (ec.simulated ?? throw new Exception("simulated_value outside a seeded case"))[a[0].GetInt32()];
    // Fixed-parameter model surface (reads the model at its CURRENT parameter values -- pinned by
    // the spec's parameter_values under a Simulation target). data_log_likelihood is generic on
    // IModel; residual[i] casts to the concrete family's Residuals(double[]) surface
    // (AR/MA/ARIMA/ARIMAX/RatingCurve). Deterministic (no fit) -> tight-tolerance oracles for the
    // C++-only "// P4 pending" ctest blocks (route b; not asserted through a committed fixture).
    if (m == "data_log_likelihood")
    {
        var pars = ec.model.Parameters.Select(p => p.Value).ToArray();
        return ec.model.DataLogLikelihood(pars);
    }
    if (m == "residual")
    {
        var pars = ec.model.Parameters.Select(p => p.Value).ToArray();
        double[] res = ec.model switch
        {
            BestFitModels.AutoRegressive ar => ar.Residuals(pars),
            BestFitModels.MovingAverage ma => ma.Residuals(pars),
            BestFitModels.ARIMA arima => arima.Residuals(pars),
            BestFitModels.ARIMAX arimax => arimax.Residuals(pars),
            BestFitModels.RatingCurve rc => rc.Residuals(pars),
            _ => throw new Exception($"residual not supported for {ec.model.GetType().Name}")
        };
        return res[a[0].GetInt32()];
    }
    // The Validate surface (Task 16): works under ANY target (it reads the model, not the
    // estimator). `is_valid` mirrors the `converged_within_tolerance` boolean-as-double
    // precedent; `validation_message_contains [substring]` is a structural substring check
    // (the fixture-checkable contract is "the failure is captured as a message", not a
    // byte-exact pin of the message text -- see the C++/R/Python dispatchers' identical note).
    if (m == "is_valid" || m == "validation_message_contains")
    {
        var (isValid, messages) = ec.model.Validate();
        if (m == "is_valid") return isValid ? 1.0 : 0.0;
        string needle = a[0].GetString()!;
        return messages.Any(msg => msg.Contains(needle)) ? 1.0 : 0.0;
    }
    switch (ec.estimator)
    {
        case MaximumLikelihood mle:
            return DispatchMlMapGeneral(m, a, construct, ec.model, mle.BestParameterSet,
                i => mle.BestParameterSet.Values[i],
                () => mle.MaximumLogLikelihood, () => mle.GetAIC(), n => mle.GetBIC(n),
                (i, j) => mle.GetCovarianceMatrix()[i, j], i => mle.GetStandardErrors()[i],
                (i, j) => mle.GetCorrelationMatrix()[i, j],
                bins => mle.ProfileLikelihood(bins), alpha => mle.ParameterConfidenceIntervals(alpha),
                () => mle.TotalFunctionEvaluations, () => mle.Status);
        case MaximumAPosteriori map:
            return DispatchMlMapGeneral(m, a, construct, ec.model, map.BestParameterSet,
                i => map.BestParameterSet.Values[i],
                () => map.MaximumLogLikelihood, () => map.GetAIC(), n => map.GetBIC(n),
                (i, j) => map.GetCovarianceMatrix()[i, j], i => map.GetStandardErrors()[i],
                (i, j) => map.GetCorrelationMatrix()[i, j],
                bins => map.ProfileLikelihood(bins), alpha => map.ParameterConfidenceIntervals(alpha),
                () => map.TotalFunctionEvaluations, () => map.Status);
        case BayesianAnalysis ba:
            return DispatchBayesian(ba, ec.model, m, a);
        case null:
            throw new Exception($"unknown Simulation fixture method: {m}");
        default:
            throw new Exception($"unknown estimator type: {ec.estimator.GetType().Name}");
    }
}

// Shared MaximumLikelihood/MaximumAPosteriori dispatch surface (identical member names on both
// C# classes). Passed the fit's already-computed pieces so each assertion is a cheap lookup.
static double DispatchMlMap(string m, JsonElement[] a, JsonElement construct,
                            BestFitModels.IModel model, ParameterSet best, double maxLL,
                            double aic, Func<int, double> bic, Matrix cov, double[] se, Matrix corr,
                            Func<int, List<double[,]>> profileLikelihood,
                            Func<double, double[,]> parameterCIs,
                            Func<int> totalFunctionEvaluations, Func<OptimizationStatus> status)
{
    int I(int i) => a[i].GetInt32();
    // Task 9: the wider fit surface, shared verbatim with DispatchMlMapGeneral.
    var extended = DispatchMlMapExtended(m, a, construct, model, best, profileLikelihood,
                                         parameterCIs, totalFunctionEvaluations, status);
    if (extended.HasValue) return extended.Value;
    switch (m)
    {
        case "parameter": return best.Values[I(0)];
        case "max_log_likelihood": return maxLL;
        case "aic": return aic;
        case "bic": return bic(I(0));  // args[0] is a sample size n, not an index
        case "covariance": return cov[I(0), I(1)];
        case "standard_error": return se[I(0)];
        case "correlation": return corr[I(0), I(1)];
        default: throw new Exception($"unknown model_estimation method for ML/MAP: {m}");
    }
}

static double DispatchBayesian(BayesianAnalysis ba, BestFitModels.IModel model, string m,
                               JsonElement[] a)
{
    int I(int i) => a[i].GetInt32();
    var results = ba.Results ?? throw new Exception("BayesianAnalysis.Results is null after RunAsync");
    // The point estimate the fit runner reports as the fit's `parameters` -- C# never centralizes
    // this ternary (see bayesian_analysis.hpp's point_estimate() note), so it is spelled out here
    // exactly as the ~15 C# Analysis classes each spell it inline.
    ParameterSet Point() => ba.PointEstimator == BayesianAnalysis.PointEstimateType.PosteriorMean
        ? results.PosteriorMean : results.MAP;
    switch (m)
    {
        case "dic": return ba.DIC;
        case "waic": return ba.WAIC;
        case "looic": return ba.LOOIC;
        case "posterior_mean": return results.PosteriorMean.Values[I(0)];
        case "chain_value":
            return (ba.Sampler ?? throw new Exception("BayesianAnalysis.Sampler is null"))
                .MarkovChains![I(0)][I(1)].Values[I(2)];
        // --- Task 9: the posterior summary + convergence diagnostics -------------------------
        // Rhat/ESS are set on ParameterStatistics by MCMCResults.ProcessParameterResults (the
        // Gelman-Rubin diagnostic over the post-warm-up MarkovChains, and the effective sample
        // size over Output); Median/StandardDeviation/LowerCI/UpperCI are the ParameterResults
        // summary over Output at the analysis's credible-interval alpha.
        case "rhat": return results.ParameterResults[I(0)].SummaryStatistics.Rhat;
        case "ess": return results.ParameterResults[I(0)].SummaryStatistics.ESS;
        case "acceptance_rate": return results.AcceptanceRates[I(0)];
        case "posterior_median": return results.ParameterResults[I(0)].SummaryStatistics.Median;
        case "posterior_sd":
            return results.ParameterResults[I(0)].SummaryStatistics.StandardDeviation;
        case "posterior_lower": return results.ParameterResults[I(0)].SummaryStatistics.LowerCI;
        case "posterior_upper": return results.ParameterResults[I(0)].SummaryStatistics.UpperCI;
        // The PSIS-LOO Pareto-k surface, read through ComputeInfluenceDiagnostics -- the same
        // wrapper fit_runner.hpp's run_fit_diagnostics reads, not the raw ba.ParetoK array, so
        // the four harnesses agree on the (NaN-guarded) MaxParetoK too.
        case "pareto_k": return ba.ComputeInfluenceDiagnostics().Observations[I(0)].ParetoK;
        case "max_pareto_k": return ba.ComputeInfluenceDiagnostics().MaxParetoK;
        case "nobs": return model.PointwiseDataLogLikelihood(Point().Values).Length;
        case "prior_log_likelihood": return model.PriorLogLikelihood(Point().Values);
        default: throw new Exception($"unknown model_estimation method for BayesianAnalysis: {m}");
    }
}

// The DataFrame assertion surface (M14): methods reachable from the model's DataFrame under
// ANY model_estimation target, corroborating the M1/M5 ctest oracles through the PUBLIC path.
// `plotting_position [kind, i]` reads item i's PlottingPosition from the named series
// ("exact" | "interval" | "uncertain", in spec order) after ONE CalculatePlottingPositions()
// pass (idempotent -- a pure function of the collections + PlottingParameter; threshold-series
// positions are NOT exposed because the C# assigns them to a sorted CLONE, so the original
// items never carry one). `number_of_low_outliers`/`low_outlier_threshold` read the frame's
// current state (set by the spec's `mgbt_low_outliers` MGBT trigger, or the explicit
// `low_outlier_threshold`).
static double DispatchModelDataFrame(BestFitModels.UnivariateDistributionModelBase model,
                                     string m, JsonElement[] a)
{
    var df = model.DataFrame;
    switch (m)
    {
        case "number_of_low_outliers": return df.NumberOfLowOutliers;
        case "low_outlier_threshold": return df.LowOutlierThreshold;
        case "plotting_position":
        {
            df.CalculatePlottingPositions();
            string seriesKind = a[0].GetString()!;
            int i = a[1].GetInt32();
            return seriesKind switch
            {
                "exact" => df.ExactSeries[i].PlottingPosition,
                "interval" => df.IntervalSeries[i].PlottingPosition,
                "uncertain" => df.UncertainSeries[i].PlottingPosition,
                var s => throw new Exception($"unknown plotting_position series kind: {s}")
            };
        }
        default: throw new Exception($"unknown model_estimation fixture method: {m}");
    }
}

static double DispatchEstimation(
    (BestFitModels.UnivariateDistributionModelBase model, object? estimator, double[]? simulated) ec,
    string m, JsonElement[] a, JsonElement construct)
{
    // The seeded-simulation digest (M13/M14): reads the vector cached at build time.
    if (m == "simulated_value")
        return (ec.simulated ?? throw new Exception("simulated_value outside a Simulation case"))[a[0].GetInt32()];
    // The M14 DataFrame surface works under any target (it reads the model, not the estimator).
    if (m == "plotting_position" || m == "number_of_low_outliers" || m == "low_outlier_threshold")
        return DispatchModelDataFrame(ec.model, m, a);
    // The Validate surface (Task 16, widened to this path in Task 21 so the older
    // UnivariateDistributionModelBase families reach the same oracle the IModel path already had
    // -- see DispatchEstimationGeneral's identical arm for the semantics).
    if (m == "is_valid" || m == "validation_message_contains")
    {
        var (isValid, messages) = ec.model.Validate();
        if (m == "is_valid") return isValid ? 1.0 : 0.0;
        string needle = a[0].GetString()!;
        return messages.Any(msg => msg.Contains(needle)) ? 1.0 : 0.0;
    }
    switch (ec.estimator)
    {
        case MaximumLikelihood mle:
            return DispatchMlMap(m, a, construct, ec.model, mle.BestParameterSet,
                mle.MaximumLogLikelihood, mle.GetAIC(),
                n => mle.GetBIC(n), mle.GetCovarianceMatrix(), mle.GetStandardErrors(),
                mle.GetCorrelationMatrix(),
                bins => mle.ProfileLikelihood(bins), alpha => mle.ParameterConfidenceIntervals(alpha),
                () => mle.TotalFunctionEvaluations, () => mle.Status);
        case MaximumAPosteriori map:
            return DispatchMlMap(m, a, construct, ec.model, map.BestParameterSet,
                map.MaximumLogLikelihood, map.GetAIC(),
                n => map.GetBIC(n), map.GetCovarianceMatrix(), map.GetStandardErrors(),
                map.GetCorrelationMatrix(),
                bins => map.ProfileLikelihood(bins), alpha => map.ParameterConfidenceIntervals(alpha),
                () => map.TotalFunctionEvaluations, () => map.Status);
        case BayesianAnalysis ba:
            return DispatchBayesian(ba, ec.model, m, a);
        case null:
            throw new Exception($"unknown Simulation fixture method: {m}");
        default:
            throw new Exception($"unknown estimator type: {ec.estimator.GetType().Name}");
    }
}

// --- analysis (Task A11: user-facing Analyses layer) --------------------------------------
//
// Drives the REAL RMC.BestFit UnivariateAnalysis / FittingAnalysis / Bulletin17CAnalysis
// (subset-compiled -- see OracleEmitter.csproj) so the tightened fixtures/analyses/*.json values
// are the exact C# oracles. Mirrors core/tests/test_fixtures.cpp's build_and_run_analysis +
// dispatch_analysis field-for-field: build the same analysis from the same `construct` spec, run
// it seeded, cache the flat result surface, then dispatch each assertion against that cache. One
// build+run per case; the same fixture file drives all four harnesses.
//
// The C# `async Task RunAsync` is driven synchronously via `.GetAwaiter().GetResult()`. The
// UnivariateAnalysis Bayesian run uses the seeded DEMCzs sampler, which draws from the shared
// history archive (NOT the current chain states), so C#'s default `ParallelizeChains = true` is
// order-invariant and reproduces the C++ serial `estimate()` bit-for-bit (proven by the tightened
// short_exact frequency-curve digest). The B17C GMM point estimate + Cohn CI are RNG-free
// (BFGS + nested Gaussian quadrature over the sandwich covariance), and FittingAnalysis fits each
// candidate by the seeded DifferentialEvolution -- all deterministic.
static BestFitAnalyses.UncertaintyMethod ParseUncertaintyMethod(string s) => s switch
{
    "MultivariateNormal" => BestFitAnalyses.UncertaintyMethod.MultivariateNormal,
    "Bootstrap" => BestFitAnalyses.UncertaintyMethod.Bootstrap,
    // Task X12: the two B17C uncertainty paths un-gated in the core by X8/X9 (LinkedMVN, pivot /
    // BiasCorrected bootstrap). Both draws are MersenneTwister-seeded -> reproducible.
    "LinkedMultivariateNormal" => BestFitAnalyses.UncertaintyMethod.LinkedMultivariateNormal,
    "BiasCorrectedBootstrap" => BestFitAnalyses.UncertaintyMethod.BiasCorrectedBootstrap,
    _ => throw new Exception($"unsupported/ deferred uncertainty method: {s}")
};

// Applies the shared Bayesian MCMC knobs from a construct object (D6; mirrors the C++
// test_fixtures.cpp::apply_analysis_bayes_knobs and the R/Python analysis glue). Used by the
// D5-authored per-family (Mixture/CompetingRisk/PointProcess) and time-series (AR/MA/ARIMA/ARIMAX)
// analysis drivers. Sets Type FIRST (its C# setter reruns SetDefaultSimulationOptions while
// UseSimulationDefaults is true), then the explicit overrides -- exactly as the C++ side does.
static void ApplyAnalysisBayesKnobs(BayesianAnalysis ba, JsonElement construct)
{
    ba.Type = ParseSamplerType(construct.TryGetProperty("sampler", out var s)
        ? s.GetString()! : "DEMCzs");
    if (construct.TryGetProperty("credible_level", out var clEl))
        ba.CredibleIntervalWidth = clEl.GetDouble();
    if (construct.TryGetProperty("seed", out var seEl)) ba.PRNGSeed = seEl.GetInt32();
    if (construct.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
    if (construct.TryGetProperty("iterations", out var itEl))
    {
        int it = itEl.GetInt32();
        ba.Iterations = it;
        ba.WarmupIterations = Math.Max(50, it / 2);
    }
    if (construct.TryGetProperty("thinning_interval", out var thEl)) ba.ThinningInterval = thEl.GetInt32();
    if (construct.TryGetProperty("number_of_chains", out var ncEl)) ba.NumberOfChains = ncEl.GetInt32();
    if (construct.TryGetProperty("initial_iterations", out var iiEl)) ba.InitialIterations = iiEl.GetInt32();
}

// Collects the UncertaintyAnalysisResults surface for a univariate-family analysis
// (Mixture/CompetingRisk/PointProcess). Mirrors test_fixtures.cpp::collect_univariate_family_results:
// parameters from the point-estimate distribution, and the compute-ctor's double[n,2]
// ConfidenceIntervals (col 0 = lower, col 1 = upper).
static void CollectFamilyResults(UncertaintyAnalysisResults? results,
                                 UnivariateDistributionBase? pe, AnalysisData r)
{
    if (results == null) return;
    if (pe != null) r.Parameters.AddRange(pe.GetParameters);
    if (results.ModeCurve != null) r.ModeCurve.AddRange(results.ModeCurve);
    if (results.MeanCurve != null) r.MeanCurve.AddRange(results.MeanCurve);
    if (results.ConfidenceIntervals != null)
    {
        int n = results.ConfidenceIntervals.GetLength(0);
        for (int i = 0; i < n; i++)
        {
            r.LowerCI.Add(results.ConfidenceIntervals[i, 0]);
            r.UpperCI.Add(results.ConfidenceIntervals[i, 1]);
        }
    }
    r.Aic = results.AIC;
    r.Bic = results.BIC;
    r.Dic = results.DIC;
    r.Rmse = results.RMSE;
}

// Collects the UncertaintyAnalysisResults surface for a time-series analysis (AR/MA/ARIMA/ARIMAX).
// Mirrors test_fixtures.cpp::run_time_series_analysis's collect: the time-series analyses expose no
// point-estimate distribution, so parameters come from the BayesianAnalysis posterior (posterior
// mean or MAP per the point estimator); the forecast ConfidenceIntervals are the plain-DTO
// double[n,3] (col 0 = time index, col 1 = lower, col 2 = upper).
static void CollectTimeSeriesResults(UncertaintyAnalysisResults? results, BayesianAnalysis ba,
                                     AnalysisData r)
{
    if (results == null) return;
    if (ba.Results != null)
    {
        var pe = ba.PointEstimator == BayesianAnalysis.PointEstimateType.PosteriorMean
            ? ba.Results.PosteriorMean.Values
            : ba.Results.MAP.Values;
        r.Parameters.AddRange(pe);
    }
    if (results.ModeCurve != null) r.ModeCurve.AddRange(results.ModeCurve);
    if (results.MeanCurve != null) r.MeanCurve.AddRange(results.MeanCurve);
    if (results.ConfidenceIntervals != null)
    {
        int n = results.ConfidenceIntervals.GetLength(0);
        for (int i = 0; i < n; i++)
        {
            r.LowerCI.Add(results.ConfidenceIntervals[i, 1]);
            r.UpperCI.Add(results.ConfidenceIntervals[i, 2]);
        }
    }
    r.Aic = results.AIC;
    r.Bic = results.BIC;
    r.Dic = results.DIC;
    r.Rmse = results.RMSE;
}

// Drives a time-series analysis's Bayesian fit, mirroring the C++ time_series run(): the
// sim-defaults guard (which reruns SetDefault{,Advanced}SimulationOptions while
// UseSimulationDefaults is true -- the C# model property-change cascade the port folds into run(),
// deviation 6), then the SERIAL (parallel:false) fit and, when estimated, the forecast assembly.
static void RunTimeSeriesAnalysis(BayesianAnalysis ba, Action createResults)
{
    if (ba.UseSimulationDefaults) ba.SetDefaultSimulationOptions();
    if (ba.UseAdvancedSimulationDefaults) ba.SetDefaultAdvancedSimulationOptions();
    ba.RunAsync(null, false, false).GetAwaiter().GetResult();
    if (ba.IsEstimated) createResults();
}

// Task X12: AnalysisBase.IsEstimated has a PROTECTED setter (only RunAsync sets it). The serial-BA
// drivers below bypass RunAsync, so the CompositeAnalysis child-validation gate (which requires each
// sub-analysis IsEstimated == true) needs the flag set out of band. Invoke the non-public setter via
// reflection -- dev-only emitter, mirrors what the real RunAsync would have set on a successful fit.
static void ForceIsEstimated(object analysis, bool value)
{
    var setter = analysis.GetType().GetProperty("IsEstimated")!.GetSetMethod(nonPublic: true)!;
    setter.Invoke(analysis, new object[] { value });
}

// Task X12: parameter-estimation-method parse for the BootstrapAnalysis fixture (mirrors
// analysis_runner.hpp::parse_estimation_method).
static ParameterEstimationMethod ParseAnalysisEstimationMethod(string s) => s switch
{
    "MethodOfMoments" => ParameterEstimationMethod.MethodOfMoments,
    "MethodOfLinearMoments" => ParameterEstimationMethod.MethodOfLinearMoments,
    "MaximumLikelihood" => ParameterEstimationMethod.MaximumLikelihood,
    _ => throw new Exception($"unknown parameter estimation method '{s}'")
};

// Task X12: build + fit a BivariateAnalysis SERIALLY, mirroring the C++ bivariate_analysis::run()
// (analysis_runner.hpp::build_and_fit_bivariate): construct the copula-coupled model, optionally set
// the XY-ordinate joint-exceedance grid, apply the Bayesian knobs, then SetSampleData ->
// BayesianAnalysis.RunAsync(null,false,false) [serial, matching the C++ estimate()] -> if estimated
// CreateFrequencyAnalysisResultsAsync -> mirror the inner IsEstimated. Driving BA directly (not
// analysis.RunAsync, which leaves parallel=true) keeps the seeded DEMCzs stream on the same serial
// path the C++/R/Python harnesses use.
static BestFitAnalyses.BivariateAnalysis BuildAndFitBivariate(
    JsonElement construct, Dictionary<string, double[]> datasets, bool setOrdinatesFromXY)
{
    var bd = (BestFitModels.BivariateDistribution)
        BuildBivariateModelGeneral(construct.GetProperty("model"), datasets);
    var analysis = new BestFitAnalyses.BivariateAnalysis(bd);
    if (setOrdinatesFromXY && construct.TryGetProperty("xy_x", out var xxEl)
        && construct.TryGetProperty("xy_y", out var yyEl))
    {
        var xs = xxEl.EnumerateArray().Select(ParseNum).ToArray();
        var ys = yyEl.EnumerateArray().Select(ParseNum).ToArray();
        var ord = new List<UncertainOrdinate>();
        for (int i = 0; i < xs.Length && i < ys.Length; i++)
            ord.Add(new UncertainOrdinate(xs[i], new Deterministic(ys[i])));
        if (ord.Count > 0)
            analysis.XYOrdinates = new UncertainOrderedPairedData(
                ord, false, SortOrder.Ascending, false, SortOrder.Ascending,
                UnivariateDistributionType.Deterministic);
    }
    ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
    bd.SetSampleData();
    analysis.BayesianAnalysis.RunAsync(null, false, false).GetAwaiter().GetResult();
    if (analysis.BayesianAnalysis.IsEstimated)
        analysis.CreateFrequencyAnalysisResultsAsync().GetAwaiter().GetResult();
    ForceIsEstimated(analysis, analysis.BayesianAnalysis.IsEstimated);
    return analysis;
}

// Task X12: build + fit ONE UnivariateAnalysis child SERIALLY, mirroring the C++ UnivariateAnalysis
// arm (used by the CompositeAnalysis driver, one per child family). Same drive idiom as the
// UnivariateAnalysis target above: apply ordinates + Bayesian knobs, prep the data frame, serial BA
// fit, then the frequency + chronology results.
static BestFitAnalyses.UnivariateAnalysis BuildAndFitUnivariateChild(
    UnivariateDistributionType distType, double[] data, JsonElement construct)
{
    var df = new BestFitModels.DataFrame { ExactSeries = new ExactSeries(data) };
    var ud = new BestFitModels.UnivariateDistribution(df, distType);
    var analysis = new BestFitAnalyses.UnivariateAnalysis(ud);
    if (construct.TryGetProperty("exceedance_probabilities", out var epEl))
    {
        analysis.ProbabilityOrdinates.Clear();
        foreach (var v in epEl.EnumerateArray()) analysis.ProbabilityOrdinates.Add(ParseNum(v));
    }
    ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
    ud.DataFrame.ProcessThresholdSeries();
    if (ud.IsNonstationary) ud.DataFrame.CreateFullTimeSeries();
    ud.ProcessQuantilePriors();
    analysis.BayesianAnalysis.RunAsync(null, false, false).GetAwaiter().GetResult();
    if (analysis.BayesianAnalysis.IsEstimated)
    {
        analysis.CreateFrequencyAnalysisResultsAsync().GetAwaiter().GetResult();
        analysis.CreateChronologyResultsAsync().GetAwaiter().GetResult();
    }
    ForceIsEstimated(analysis, analysis.BayesianAnalysis.IsEstimated);
    return analysis;
}

static AnalysisData BuildAndRunAnalysis(string target, JsonElement construct,
                                        Dictionary<string, double[]> datasets)
{
    var r = new AnalysisData();

    // Optional exceedance-probability override (mirrors the C++ apply_ordinates: replace the
    // default grid only when the case supplies one).
    void ApplyOrdinates(ProbabilityOrdinates po)
    {
        if (!construct.TryGetProperty("exceedance_probabilities", out var epEl)) return;
        po.Clear();
        foreach (var v in epEl.EnumerateArray()) po.Add(ParseNum(v));
    }

    if (target == "FittingAnalysis")
    {
        var data = datasets[construct.GetProperty("dataset").GetString()!];
        var df = new BestFitModels.DataFrame { ExactSeries = new ExactSeries(data) };
        df.CalculatePlottingPositions();
        var analysis = new BestFitAnalyses.FittingAnalysis(df);
        // The full C# DistributionList (15 candidates, GeneralizedNormal at index 4) is driven as
        // shipped. GeneralizedNormal used to be removed here because the C++ distribution factory
        // had no case for it; now that it is ported and factory-constructible, both sides fit the
        // SAME 15-candidate set and the fixture asserts C# indices directly.
        analysis.RunAsync().GetAwaiter().GetResult();
        var fitted = analysis.FittedDistributions;
        r.CandidateCount = fitted.Count;
        foreach (var fd in fitted)
        {
            r.CandAic.Add(fd.AIC);
            r.CandBic.Add(fd.BIC);
            r.CandRmse.Add(fd.RMSE);
            r.CandConverged.Add(fd.FitSucceeded ? 1.0 : 0.0);
        }
        return r;
    }

    var modelSpec = construct.GetProperty("model");

    if (target == "UnivariateAnalysis")
    {
        var baseModel = BuildSpecModel(modelSpec, datasets);
        if (baseModel is not BestFitModels.UnivariateDistribution ud)
            throw new Exception("UnivariateAnalysis requires a univariate_distribution model");
        var analysis = new BestFitAnalyses.UnivariateAnalysis(ud);
        ApplyOrdinates(analysis.ProbabilityOrdinates);
        // The BayesianAnalysis was built (with its simulation defaults) by the analysis ctor;
        // override only the fixture-named knobs, exactly as the C++ runner does. NumberOfChains /
        // ThinningInterval / InitialIterations keep the C#/C++ defaults so the two match.
        var ba = analysis.BayesianAnalysis;
        ba.Type = ParseSamplerType(construct.TryGetProperty("sampler", out var s)
            ? s.GetString()! : "DEMCzs");
        if (construct.TryGetProperty("credible_level", out var clEl))
            ba.CredibleIntervalWidth = clEl.GetDouble();
        if (construct.TryGetProperty("seed", out var seEl)) ba.PRNGSeed = seEl.GetInt32();
        if (construct.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        if (construct.TryGetProperty("iterations", out var itEl))
        {
            int it = itEl.GetInt32();
            ba.Iterations = it;
            ba.WarmupIterations = Math.Max(50, it / 2);
        }
        // Optional explicit MCMC knobs (A11). The default thinning_interval=20 that
        // SetDefaultSimulationOptions picks for a 2-parameter DEMCzs run exposes a C#-vs-C++
        // divergence in the THINNED population-sampler stream (a real port bug -- see
        // docs/upstream-csharp-issues.md, A11 finding). Setting thinning_interval=1 lands on the
        // bayes_normal-proven bit-identical path, so the fixture pins it explicitly and all four
        // runners honor it.
        if (construct.TryGetProperty("thinning_interval", out var thEl)) ba.ThinningInterval = thEl.GetInt32();
        if (construct.TryGetProperty("number_of_chains", out var ncEl)) ba.NumberOfChains = ncEl.GetInt32();
        if (construct.TryGetProperty("initial_iterations", out var iiEl)) ba.InitialIterations = iiEl.GetInt32();
        // Drive the analysis SERIALLY, mirroring the C++ UnivariateAnalysis::run() (which has no
        // ParallelizeChains -- scope decision 1 of the port). The C# UnivariateAnalysis.RunAsync
        // leaves BayesianAnalysis.RunAsync's `parallel` at its default `true`; DEMCzs draws from
        // the shared history archive (not the current chain states), so serial and parallel agree,
        // but passing parallel:false keeps this on the same serial path the C++/R/Python harnesses
        // and the model_estimation emitter path use.
        ud.DataFrame.ProcessThresholdSeries();
        if (ud.IsNonstationary) ud.DataFrame.CreateFullTimeSeries();
        ud.ProcessQuantilePriors();
        ba.RunAsync(null, false, false).GetAwaiter().GetResult();
        if (ba.IsEstimated)
        {
            analysis.CreateFrequencyAnalysisResultsAsync().GetAwaiter().GetResult();
            analysis.CreateChronologyResultsAsync().GetAwaiter().GetResult();
        }
        var results = analysis.AnalysisResults;
        if (results != null)
        {
            var pe = analysis.GetPointEstimateDistribution();
            if (pe is not null) r.Parameters.AddRange(pe.GetParameters);
            if (results.ModeCurve != null) r.ModeCurve.AddRange(results.ModeCurve);
            if (results.MeanCurve != null) r.MeanCurve.AddRange(results.MeanCurve);
            if (results.ConfidenceIntervals != null)
            {
                int n = results.ConfidenceIntervals.GetLength(0);
                for (int i = 0; i < n; i++)
                {
                    r.LowerCI.Add(results.ConfidenceIntervals[i, 0]);
                    r.UpperCI.Add(results.ConfidenceIntervals[i, 1]);
                }
            }
            r.Aic = results.AIC;
            r.Bic = results.BIC;
            r.Dic = results.DIC;
            r.Rmse = results.RMSE;
        }
        return r;
    }

    if (target == "Bulletin17CAnalysis")
    {
        var model = BuildBulletin17CModel(modelSpec, datasets);
        var analysis = new BestFitAnalyses.Bulletin17CAnalysis(model);
        analysis.UncertaintyMethod = ParseUncertaintyMethod(
            construct.TryGetProperty("uncertainty_method", out var um)
                ? um.GetString()! : "MultivariateNormal");
        ApplyOrdinates(analysis.ProbabilityOrdinates);
        var ba = analysis.BayesianAnalysis;
        if (construct.TryGetProperty("confidence_level", out var clEl))
            ba.CredibleIntervalWidth = clEl.GetDouble();
        if (construct.TryGetProperty("seed", out var seEl)) ba.PRNGSeed = seEl.GetInt32();
        if (construct.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        analysis.RunAsync().GetAwaiter().GetResult();
        var ci = analysis.ComputeCohnStyleConfidenceIntervals();
        if (ci != null)
        {
            r.Exceedance.AddRange(ci.ExceedanceProbabilities);
            r.PointEstimates.AddRange(ci.PointEstimates);
            r.LowerCI.AddRange(ci.LowerCI);
            r.UpperCI.AddRange(ci.UpperCI);
            r.Beta1.AddRange(ci.Beta1);
            r.Nu.AddRange(ci.Nu);
            r.QuantileVariance.AddRange(ci.QuantileVariance);
            r.ConfidenceLevel = ci.ConfidenceLevel;
        }
        if (analysis.GMM != null && analysis.GMM.IsEstimated)
            r.Parameters.AddRange(analysis.GMM.BestParameterSet.Values);
        // T19: the genuinely ensemble-derived UncertaintyAnalysisResults surface (distinct from
        // the RNG-free Cohn CI above) -- MeanCurve is built from the ACTUAL sampled parameter
        // sets (BayesianAnalysis.Results.Output), so unlike the Cohn CI it DOES depend on which
        // bootstrap replicates were drawn. Reuses the generic mode_curve/mean_curve dispatch
        // every other analysis target already shares.
        if (analysis.AnalysisResults != null)
        {
            if (analysis.AnalysisResults.ModeCurve != null) r.ModeCurve.AddRange(analysis.AnalysisResults.ModeCurve);
            if (analysis.AnalysisResults.MeanCurve != null) r.MeanCurve.AddRange(analysis.AnalysisResults.MeanCurve);
        }
        // T19: BootstrapDiagnostics, populated only when the uncertainty method actually ran a
        // bootstrap arm (Bootstrap / BiasCorrectedBootstrap).
        if (analysis.BootstrapResults != null)
        {
            var boot = analysis.BootstrapResults;
            r.BootHasResults = true;
            r.BootTotalReplicates = boot.TotalReplicates;
            r.BootAttemptedReplicates = boot.AttemptedReplicates;
            r.BootFailedReplicates = boot.FailedReplicates;
            r.BootValidReplicates = boot.ValidReplicates;
            r.BootRetainedReplicates = boot.RetainedReplicates;
            r.BootFailureRate = boot.FailureRate;
            r.BootTotalRetries = boot.TotalRetries;
            r.BootAverageRetries = boot.AverageRetries;
            r.BootPivotRejections = boot.PivotRejections;
            r.BootMahalanobisRejections = boot.MahalanobisRejections;
            r.BootTransformFailures = boot.TransformFailures;
            r.BootStatusSuccessCount = boot.StatusSuccessCount;
            r.BootStatusMaxIterationsCount = boot.StatusMaximumIterationsCount;
            r.BootStatusMaxFunctionEvaluationsCount = boot.StatusMaximumFunctionEvaluationsCount;
            r.BootStatusFailureCount = boot.StatusFailureCount;
            r.BootStatusNoneCount = boot.StatusNoneCount;
            r.BootOptimizerFallbacks = boot.OptimizerFallbacks;
        }
        return r;
    }

    if (target == "Diagnostics")
    {
        // Mirror test_fixtures.cpp::run_diagnostics_analysis: build the model, run a seeded
        // deterministic BayesianAnalysis (serial, ParallelizeChains=false), then compute all three
        // diagnostics off that single fit. The BayesianAnalysis knobs are applied in the same order
        // as BuildEstimation's BayesianAnalysis target + apply_analysis_bayes_knobs (C++), so the
        // C# and C++ seeded posteriors are the same stream.
        var model = BuildSpecModel(modelSpec, datasets);
        var ba = new BayesianAnalysis(model, ParseSamplerType(
            construct.TryGetProperty("sampler", out var s) ? s.GetString()! : "DEMCzs"))
        {
            UseSimulationDefaults = false,
            UseAdvancedSimulationDefaults = false,
        };
        if (construct.TryGetProperty("credible_level", out var clEl))
            ba.CredibleIntervalWidth = clEl.GetDouble();
        if (construct.TryGetProperty("seed", out var seEl)) ba.PRNGSeed = seEl.GetInt32();
        if (construct.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        if (construct.TryGetProperty("iterations", out var itEl))
        {
            int it = itEl.GetInt32();
            ba.Iterations = it;
            ba.WarmupIterations = Math.Max(50, it / 2);
        }
        if (construct.TryGetProperty("thinning_interval", out var thEl)) ba.ThinningInterval = thEl.GetInt32();
        if (construct.TryGetProperty("number_of_chains", out var ncEl)) ba.NumberOfChains = ncEl.GetInt32();
        if (construct.TryGetProperty("initial_iterations", out var iiEl)) ba.InitialIterations = iiEl.GetInt32();
        ba.RunAsync(null, false, false).GetAwaiter().GetResult();
        if (!ba.IsEstimated) return r;

        var lev = ba.ComputeLeverageDiagnostics();
        r.LevCount = lev.Count;
        r.LevPriorCount = lev.PriorComponents.Length;
        r.TotalLeverage = lev.TotalLeverage;
        r.TotalFitInfluence = lev.TotalFitInfluence;
        r.TotalVarianceInfluence = lev.TotalVarianceInfluence;
        foreach (var o in lev.Observations)
        {
            r.LevObsLeverage.Add(o.Leverage);
            r.LevObsFit.Add(o.FitInfluence);
            r.LevObsVar.Add(o.VarianceInfluence);
            r.LevObsValue.Add(o.Value);
        }

        var inf = ba.ComputeInfluenceDiagnostics();
        r.InfCount = inf.Count;
        r.MeanParetoK = inf.MeanParetoK;
        r.MaxParetoK = inf.MaxParetoK;
        r.CountParetoK05 = inf.CountParetoKAbove05;
        r.CountParetoK07 = inf.CountParetoKAbove07;
        r.CountParetoK10 = inf.CountParetoKAbove10;
        r.ProportionProblematic = inf.ProportionProblematic;
        r.IsReliable = inf.IsReliable ? 1.0 : 0.0;
        foreach (var o in inf.Observations)
        {
            r.InfParetoK.Add(o.ParetoK);
            r.InfElpdLoo.Add(o.ElpdLoo);
        }

        int thinEvery = construct.TryGetProperty("thin_every", out var teEl) ? teEl.GetInt32() : 10;
        var pri = ba.ComputePriorInfluenceDiagnostics(thinEvery);
        r.PriCount = pri.Count;
        r.TotalPriorLogLik = pri.TotalPriorLogLikelihood;
        r.TotalDataLogLik = pri.TotalDataLogLikelihood;
        r.PriorToDataRatio = pri.PriorToDataRatio;
        r.IsPriorInfluential = pri.IsPriorInfluential ? 1.0 : 0.0;
        r.MeanPriorPrecisionShare = pri.MeanPriorPrecisionShare;
        return r;
    }

    // --- D6: per-family analyses (Mixture/CompetingRisk/PointProcess). Mirror the C++
    // {mixture,competing_risk,point_process}_analysis::run(): build the model, apply ordinates +
    // Bayesian knobs, prep the data frame, then drive the Bayesian fit SERIALLY (parallel:false)
    // and assemble the frequency results. Driving BayesianAnalysis.RunAsync directly (rather than
    // analysis.RunAsync) skips the C# EM-seed initialization the C++ port omits, keeping the
    // seeded chain on the same standard-init path the C++ core uses. ---
    if (target == "MixtureAnalysis")
    {
        var model = (BestFitModels.MixtureModel)BuildSpecModel(modelSpec, datasets);
        var analysis = new BestFitAnalyses.MixtureAnalysis(model);
        ApplyOrdinates(analysis.ProbabilityOrdinates);
        ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
        model.DataFrame.ProcessThresholdSeries();
        model.ProcessQuantilePriors();
        analysis.BayesianAnalysis.RunAsync(null, false, false).GetAwaiter().GetResult();
        if (analysis.BayesianAnalysis.IsEstimated)
            analysis.CreateFrequencyAnalysisResultsAsync().GetAwaiter().GetResult();
        CollectFamilyResults(analysis.AnalysisResults, analysis.GetPointEstimateDistribution(), r);
        return r;
    }

    if (target == "CompetingRiskAnalysis")
    {
        var model = (BestFitModels.CompetingRisksModel)BuildSpecModel(modelSpec, datasets);
        var analysis = new BestFitAnalyses.CompetingRiskAnalysis(model);
        ApplyOrdinates(analysis.ProbabilityOrdinates);
        ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
        model.DataFrame.ProcessThresholdSeries();
        model.ProcessQuantilePriors();
        analysis.BayesianAnalysis.RunAsync(null, false, false).GetAwaiter().GetResult();
        if (analysis.BayesianAnalysis.IsEstimated)
            analysis.CreateFrequencyAnalysisResultsAsync().GetAwaiter().GetResult();
        CollectFamilyResults(analysis.AnalysisResults, analysis.GetPointEstimateDistribution(), r);
        return r;
    }

    if (target == "PointProcessAnalysis")
    {
        var model = (BestFitModels.PointProcessModel)BuildSpecModel(modelSpec, datasets);
        var analysis = new BestFitAnalyses.PointProcessAnalysis(model);
        ApplyOrdinates(analysis.ProbabilityOrdinates);
        ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
        model.DataFrame.ProcessThresholdSeries();
        model.ProcessQuantilePriors();
        analysis.BayesianAnalysis.RunAsync(null, false, false).GetAwaiter().GetResult();
        if (analysis.BayesianAnalysis.IsEstimated)
            analysis.CreateFrequencyAnalysisResultsAsync().GetAwaiter().GetResult();
        CollectFamilyResults(analysis.AnalysisResults, analysis.GetPointEstimateDistribution(), r);
        return r;
    }

    // --- D6: time-series analyses (AR/MA/ARIMA/ARIMAX). Mirror the C++ run_time_series_analysis:
    // build the concrete model, pin the training window (a fixed training_time_steps overriding the
    // 80%-of-data default), set the forecast horizon, apply knobs, then RunTimeSeriesAnalysis
    // (sim-defaults guard + serial fit + forecast assembly). The four model types share the
    // training-window property names but no common base, so the pin is repeated in each case. ---
    if (target == "ARAnalysis")
    {
        var model = (BestFitModels.AutoRegressive)BuildTimeSeriesModelGeneral(modelSpec, datasets);
        if (construct.TryGetProperty("training_time_steps", out var ttsEl))
        {
            model.UseDefaultTrainingSteps = false;
            model.TrainingTimeSteps = ttsEl.GetInt32();
        }
        var analysis = new BestFitAnalyses.ARAnalysis(model);
        if (construct.TryGetProperty("forecasting_time_steps", out var ftsEl))
            analysis.ForecastingTimeSteps = ftsEl.GetInt32();
        ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
        RunTimeSeriesAnalysis(analysis.BayesianAnalysis,
            () => analysis.CreateUncertaintyAnalysisResultsAsync().GetAwaiter().GetResult());
        CollectTimeSeriesResults(analysis.AnalysisResults, analysis.BayesianAnalysis, r);
        return r;
    }

    if (target == "MAAnalysis")
    {
        var model = (BestFitModels.MovingAverage)BuildTimeSeriesModelGeneral(modelSpec, datasets);
        if (construct.TryGetProperty("training_time_steps", out var ttsEl))
        {
            model.UseDefaultTrainingSteps = false;
            model.TrainingTimeSteps = ttsEl.GetInt32();
        }
        var analysis = new BestFitAnalyses.MAAnalysis(model);
        if (construct.TryGetProperty("forecasting_time_steps", out var ftsEl))
            analysis.ForecastingTimeSteps = ftsEl.GetInt32();
        ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
        RunTimeSeriesAnalysis(analysis.BayesianAnalysis,
            () => analysis.CreateUncertaintyAnalysisResultsAsync().GetAwaiter().GetResult());
        CollectTimeSeriesResults(analysis.AnalysisResults, analysis.BayesianAnalysis, r);
        return r;
    }

    if (target == "ARIMAAnalysis")
    {
        var model = (BestFitModels.ARIMA)BuildTimeSeriesModelGeneral(modelSpec, datasets);
        if (construct.TryGetProperty("training_time_steps", out var ttsEl))
        {
            model.UseDefaultTrainingSteps = false;
            model.TrainingTimeSteps = ttsEl.GetInt32();
        }
        var analysis = new BestFitAnalyses.ARIMAAnalysis(model);
        if (construct.TryGetProperty("forecasting_time_steps", out var ftsEl))
            analysis.ForecastingTimeSteps = ftsEl.GetInt32();
        ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
        RunTimeSeriesAnalysis(analysis.BayesianAnalysis,
            () => analysis.CreateUncertaintyAnalysisResultsAsync().GetAwaiter().GetResult());
        CollectTimeSeriesResults(analysis.AnalysisResults, analysis.BayesianAnalysis, r);
        return r;
    }

    if (target == "ARIMAXAnalysis")
    {
        var model = (BestFitModels.ARIMAX)BuildTimeSeriesModelGeneral(modelSpec, datasets);
        if (construct.TryGetProperty("training_time_steps", out var ttsEl))
        {
            model.UseDefaultTrainingSteps = false;
            model.TrainingTimeSteps = ttsEl.GetInt32();
        }
        var analysis = new BestFitAnalyses.ARIMAXAnalysis(model);
        if (construct.TryGetProperty("forecasting_time_steps", out var ftsEl))
            analysis.ForecastingTimeSteps = ftsEl.GetInt32();
        ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
        RunTimeSeriesAnalysis(analysis.BayesianAnalysis,
            () => analysis.CreateUncertaintyAnalysisResultsAsync().GetAwaiter().GetResult());
        CollectTimeSeriesResults(analysis.AnalysisResults, analysis.BayesianAnalysis, r);
        return r;
    }

    // --- X12: Phase-10 full-parity analysis drivers. Each mirrors the C++ analysis_runner.hpp
    // run_* builder against the REAL RMC.BestFit / Numerics classes so ONE fixture file drives all
    // four harnesses. The MCMC-driven arms fit the seeded DEMCzs BayesianAnalysis SERIALLY
    // (RunAsync(null,false,false)) to land on the same serial stream the C++ estimate() uses; the
    // deterministic post-processing (curve/CI aggregation) is order-invariant. ---

    if (target == "BivariateAnalysis")
    {
        var analysis = BuildAndFitBivariate(construct, datasets, /*setOrdinatesFromXY:*/ true);
        CollectFamilyResults(analysis.AnalysisResults, null, r);
        if (analysis.BayesianAnalysis.Results != null)
            r.Parameters.AddRange(analysis.BayesianAnalysis.Results.MAP.Values);
        return r;
    }

    if (target == "CoincidentFrequencyAnalysis")
    {
        // Fit the upstream bivariate WITHOUT the XY joint-exceedance grid (CFA consumes the fitted
        // marginals + copula and its own X/Y/response surface). Mirrors run_coincident.
        var biv = BuildAndFitBivariate(construct, datasets, /*setOrdinatesFromXY:*/ false);

        var xValues = construct.GetProperty("x_values").EnumerateArray().Select(ParseNum).ToArray();
        var yValues = construct.GetProperty("y_values").EnumerateArray().Select(ParseNum).ToArray();
        int rows = construct.GetProperty("response_rows").GetInt32();
        int cols = construct.GetProperty("response_cols").GetInt32();
        var flat = construct.GetProperty("response").EnumerateArray().Select(ParseNum).ToArray();
        var response = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                response[i, j] = flat[i * cols + j];

        var cfa = new BestFitAnalyses.CoincidentFrequencyAnalysis(biv, xValues, yValues, response);
        if (construct.TryGetProperty("number_of_bins", out var nbEl)) cfa.NumberOfBins = nbEl.GetInt32();
        if (construct.TryGetProperty("credible_level", out var clEl))
            cfa.BayesianAnalysis.CredibleIntervalWidth = clEl.GetDouble();
        // CFA.CreateFrequencyAnalysisResultsAsync reads the bivariate posterior directly and does not
        // gate on IsEstimated (there is no MCMC on CFA itself), so calling it is sufficient.
        cfa.CreateFrequencyAnalysisResultsAsync().GetAwaiter().GetResult();
        CollectFamilyResults(cfa.AnalysisResults, null, r);
        if (cfa.ZOutputValues != null) r.ZOutput.AddRange(cfa.ZOutputValues);
        return r;
    }

    if (target == "CompositeAnalysis")
    {
        // Fit one UnivariateAnalysis child per family over the shared dataset, then aggregate
        // deterministically (composite has no MCMC of its own). Mirrors run_composite.
        var data = datasets[modelSpec.GetProperty("dataset").GetString()!];
        var children = new List<BestFitAnalyses.UnivariateAnalysis>();
        foreach (var f in modelSpec.GetProperty("families").EnumerateArray())
            children.Add(BuildAndFitUnivariateChild(
                Enum.Parse<UnivariateDistributionType>(f.GetString()!), data, construct));

        var composite = new BestFitAnalyses.CompositeAnalysis();
        foreach (var c in children)
            composite.Analyses.Add(new BestFitAnalyses.WeightedUnivariateAnalysis(c, 0.0));

        composite.CompositeDistributionType = construct.TryGetProperty("composite_type", out var ctEl)
            ? Enum.Parse<BestFitAnalyses.CompositeType>(ctEl.GetString()!)
            : BestFitAnalyses.CompositeType.CompetingRisks;
        composite.ModelAverageMethod = construct.TryGetProperty("average_method", out var amEl)
            ? Enum.Parse<BestFitAnalyses.AverageMethod>(amEl.GetString()!)
            : BestFitAnalyses.AverageMethod.AIC;
        ApplyOrdinates(composite.ProbabilityOrdinates);
        if (construct.TryGetProperty("credible_level", out var clEl))
            composite.BayesianAnalysis.CredibleIntervalWidth = clEl.GetDouble();
        composite.RunAsync().GetAwaiter().GetResult();
        CollectFamilyResults(composite.AnalysisResults, composite.GetPointEstimateDistribution(), r);
        return r;
    }

    if (target == "RatingCurveAnalysis")
    {
        var rc = (BestFitModels.RatingCurve)BuildRatingCurveModelGeneral(modelSpec, datasets);
        var analysis = new BestFitAnalyses.RatingCurveAnalysis(rc);
        if (construct.TryGetProperty("stage_bins", out var sbEl))
        {
            analysis.UseDefaultStageBins = false;
            analysis.StageBins = sbEl.GetInt32();
        }
        if (construct.TryGetProperty("min_stage", out var minEl)) analysis.MinStage = minEl.GetDouble();
        if (construct.TryGetProperty("max_stage", out var maxEl)) analysis.MaxStage = maxEl.GetDouble();
        ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
        analysis.BayesianAnalysis.RunAsync(null, false, false).GetAwaiter().GetResult();
        if (analysis.BayesianAnalysis.IsEstimated)
        {
            analysis.CreateUncertaintyAnalysisResultsAsync().GetAwaiter().GetResult();
            analysis.UpdatePointEstimateResultsAsync().GetAwaiter().GetResult();
        }
        CollectFamilyResults(analysis.AnalysisResults, null, r);
        if (analysis.BayesianAnalysis.Results != null)
            r.Parameters.AddRange(analysis.BayesianAnalysis.Results.MAP.Values);
        return r;
    }

    if (target == "SpatialGEVAnalysis")
    {
        var sg = (BestFitModels.SpatialExtremes.SpatialGEV)BuildSpatialGevModelGeneral(modelSpec, datasets);
        var analysis = new BestFitAnalyses.SpatialGEVAnalysis(sg);
        ApplyOrdinates(analysis.ProbabilityOrdinates);
        ApplyAnalysisBayesKnobs(analysis.BayesianAnalysis, construct);
        // Site/uncertainty post-processing is PRIVATE (only RunAsync / ordinate-reprocess reach
        // it), so drive the analysis's own RunAsync. The DEMCzs population sampler draws from the
        // shared history archive, so its parallel default is order-invariant vs the C++ serial
        // estimate() (the same invariance the UnivariateAnalysis arm documents).
        analysis.RunAsync().GetAwaiter().GetResult();
        if (construct.TryGetProperty("cross_validation", out var cvEl) && cvEl.GetBoolean())
            analysis.RunCrossValidationAsync().GetAwaiter().GetResult();

        // Regional curve: CI is double[n,3] (col 0 = prob, 1 = lower, 2 = upper). Collect manually.
        var res = analysis.AnalysisResults;
        if (res != null)
        {
            if (res.ModeCurve != null) r.ModeCurve.AddRange(res.ModeCurve);
            if (res.MeanCurve != null) r.MeanCurve.AddRange(res.MeanCurve);
            if (res.ConfidenceIntervals != null)
            {
                int n = res.ConfidenceIntervals.GetLength(0);
                for (int i = 0; i < n; i++)
                {
                    r.LowerCI.Add(res.ConfidenceIntervals[i, 1]);
                    r.UpperCI.Add(res.ConfidenceIntervals[i, 2]);
                }
            }
            r.Aic = res.AIC; r.Bic = res.BIC; r.Dic = res.DIC; r.Rmse = res.RMSE;
        }
        if (analysis.BayesianAnalysis.Results != null)
            r.Parameters.AddRange(analysis.BayesianAnalysis.Results.MAP.Values);
        var sites = analysis.SiteResults;
        if (sites != null)
        {
            r.SiteCount = sites.Length;
            foreach (var sr in sites)
            {
                r.SiteLocationMean.Add(sr.LocationMean);
                r.SiteScaleMean.Add(sr.ScaleMean);
                r.SiteShapeMean.Add(sr.ShapeMean);
            }
            if (sites.Length > 0) r.Site0QuantileMean.AddRange(sites[0].QuantileMean);
        }
        var cv = analysis.CrossValidationResults;
        if (cv != null)
        {
            r.CvMae = cv.MeanAbsoluteError;
            r.CvRmse = cv.RootMeanSquareError;
            r.CvMeanBias = cv.MeanBias;
        }
        return r;
    }

    if (target == "BootstrapAnalysis")
    {
        var data = datasets[modelSpec.GetProperty("dataset").GetString()!];
        var dist = UnivariateDistributionFactory.CreateDistribution(
            Enum.Parse<UnivariateDistributionType>(modelSpec.GetProperty("family").GetString()!));
        var method = ParseAnalysisEstimationMethod(construct.TryGetProperty("estimation_method", out var emEl)
            ? emEl.GetString()! : "MaximumLikelihood");
        ((IEstimation)dist).Estimate(data, method);

        int sampleSize = construct.TryGetProperty("sample_size", out var ssEl) ? ssEl.GetInt32() : data.Length;
        int replications = construct.TryGetProperty("replications", out var rpEl) ? rpEl.GetInt32() : 1000;
        int seed = construct.TryGetProperty("seed", out var seEl) ? seEl.GetInt32() : 12345;
        double alpha = construct.TryGetProperty("alpha", out var alEl) ? alEl.GetDouble() : 0.1;
        var probs = construct.GetProperty("probabilities").EnumerateArray().Select(ParseNum).ToArray();

        var boot = new BootstrapAnalysis(dist, method, sampleSize, replications, seed);
        var res = boot.Estimate(probs, alpha, null, /*recordParameterSets:*/ false);
        CollectFamilyResults(res, null, r);
        r.Parameters.AddRange(dist.GetParameters);
        return r;
    }

    if (target == "PriorPredictiveCheck")
    {
        var model = BuildSpecModel(modelSpec, datasets);
        var check = new RMC.BestFit.Diagnostics.PriorPredictiveCheck(model);
        if (construct.TryGetProperty("seed", out var seEl)) check.Seed = seEl.GetInt32();
        if (construct.TryGetProperty("number_of_draws", out var ndEl)) check.NumberOfDraws = ndEl.GetInt32();
        int sampleSize = construct.TryGetProperty("sample_size", out var ssEl)
            ? ssEl.GetInt32() : datasets[modelSpec.GetProperty("dataset").GetString()!].Length;
        var summary = check.ComputeSummary(sampleSize);
        r.NumberOfValidDraws = summary.NumberOfValidDraws;
        r.SummaryMeanQuantile.AddRange(summary.MeanQuantiles);
        r.SummarySdQuantile.AddRange(summary.SDQuantiles);
        r.SummaryMinQuantile.AddRange(summary.MinQuantiles);
        r.SummaryMaxQuantile.AddRange(summary.MaxQuantiles);
        return r;
    }

    if (target == "PosteriorPredictiveCheck")
    {
        var model = BuildSpecModel(modelSpec, datasets);
        var data = datasets[modelSpec.GetProperty("dataset").GetString()!];
        var observed = construct.TryGetProperty("observed_dataset", out var odEl)
            ? datasets[odEl.GetString()!] : data;

        // Fit a quick seeded serial BayesianAnalysis for the posterior draws.
        var ba = new BayesianAnalysis(model, ParseSamplerType(
            construct.TryGetProperty("sampler", out var sEl) ? sEl.GetString()! : "DEMCzs"))
        {
            UseSimulationDefaults = false,
            UseAdvancedSimulationDefaults = false,
        };
        ApplyAnalysisBayesKnobs(ba, construct);
        ba.RunAsync(null, false, false).GetAwaiter().GetResult();
        if (!ba.IsEstimated || ba.Results == null)
            throw new Exception("PosteriorPredictiveCheck MCMC fit failed");

        var check = new RMC.BestFit.Diagnostics.PosteriorPredictiveCheck(model, ba.Results, observed);
        if (construct.TryGetProperty("check_seed", out var csEl)) check.Seed = csEl.GetInt32();
        else if (construct.TryGetProperty("seed", out var seEl)) check.Seed = seEl.GetInt32();
        int nRep = construct.TryGetProperty("number_of_replicates", out var nrEl) ? nrEl.GetInt32() : 1000;
        var results = check.ComputeCommonPValues(nRep);
        r.NumberOfReplicates = results.NumberOfReplicates;
        r.MeanPValue = results.MeanPValue;
        r.SdPValue = results.SDPValue;
        r.SkewnessPValue = results.SkewnessPValue;
        r.MinPValue = results.MinPValue;
        r.MaxPValue = results.MaxPValue;
        r.HasMisfit = results.HasPotentialMisfit() ? 1.0 : 0.0;
        return r;
    }

    throw new Exception($"unknown analysis target: {target}");
}

// Flat analysis-result dispatch, matching test_fixtures.cpp's dispatch_analysis method names.
static double DispatchAnalysis(AnalysisData r, string m, JsonElement[] a)
{
    int I(int i) => a[i].GetInt32();
    switch (m)
    {
        case "candidate_count": return r.CandidateCount;
        case "candidate_aic": return r.CandAic[I(0)];
        case "candidate_bic": return r.CandBic[I(0)];
        case "candidate_rmse": return r.CandRmse[I(0)];
        case "candidate_converged": return r.CandConverged[I(0)];
        case "parameter": return r.Parameters[I(0)];
        // curve_length == mode-curve length (test_fixtures.cpp: mode_curve.size()); used by the
        // D5 time-series forecast fixtures to assert the observed + forecast horizon length.
        case "curve_length": return r.ModeCurve.Count;
        case "mode_curve": return r.ModeCurve[I(0)];
        case "mean_curve": return r.MeanCurve[I(0)];
        case "lower_ci": return r.LowerCI[I(0)];
        case "upper_ci": return r.UpperCI[I(0)];
        case "exceedance_probability": return r.Exceedance[I(0)];
        case "point_estimate": return r.PointEstimates[I(0)];
        case "beta1": return r.Beta1[I(0)];
        case "nu": return r.Nu[I(0)];
        case "quantile_variance": return r.QuantileVariance[I(0)];
        case "aic": return r.Aic;
        case "bic": return r.Bic;
        case "dic": return r.Dic;
        case "rmse": return r.Rmse;
        case "confidence_level": return r.ConfidenceLevel;
        // --- D6 Diagnostics dispatch (names match diagnostics_smoke.json + test_fixtures.cpp). ---
        case "leverage_count": return r.LevCount;
        case "leverage_prior_count": return r.LevPriorCount;
        case "total_leverage": return r.TotalLeverage;
        case "total_fit_influence": return r.TotalFitInfluence;
        case "total_variance_influence": return r.TotalVarianceInfluence;
        case "obs_leverage": return r.LevObsLeverage[I(0)];
        case "obs_fit_influence": return r.LevObsFit[I(0)];
        case "obs_variance_influence": return r.LevObsVar[I(0)];
        case "obs_value": return r.LevObsValue[I(0)];
        case "influence_count": return r.InfCount;
        case "mean_pareto_k": return r.MeanParetoK;
        case "max_pareto_k": return r.MaxParetoK;
        case "count_pareto_k_above_05": return r.CountParetoK05;
        case "count_pareto_k_above_07": return r.CountParetoK07;
        case "count_pareto_k_above_10": return r.CountParetoK10;
        case "proportion_problematic": return r.ProportionProblematic;
        case "is_reliable": return r.IsReliable;
        case "pareto_k": return r.InfParetoK[I(0)];
        case "elpd_loo": return r.InfElpdLoo[I(0)];
        case "prior_influence_count": return r.PriCount;
        case "total_prior_log_likelihood": return r.TotalPriorLogLik;
        case "total_data_log_likelihood": return r.TotalDataLogLik;
        case "prior_to_data_ratio": return r.PriorToDataRatio;
        case "is_prior_influential": return r.IsPriorInfluential;
        case "mean_prior_precision_share": return r.MeanPriorPrecisionShare;
        // --- X12 extended-analysis dispatch (names match test_fixtures.cpp + the *_smoke.json). ---
        case "z_output": return r.ZOutput[I(0)];
        case "z_output_length": return r.ZOutput.Count;
        case "site_count": return r.SiteCount;
        case "site_location_mean": return r.SiteLocationMean[I(0)];
        case "site_scale_mean": return r.SiteScaleMean[I(0)];
        case "site_shape_mean": return r.SiteShapeMean[I(0)];
        case "site_quantile_mean": return r.Site0QuantileMean[I(0)];
        case "cv_mae": return r.CvMae;
        case "cv_rmse": return r.CvRmse;
        case "cv_mean_bias": return r.CvMeanBias;
        case "mean_p_value": return r.MeanPValue;
        case "sd_p_value": return r.SdPValue;
        case "skewness_p_value": return r.SkewnessPValue;
        case "min_p_value": return r.MinPValue;
        case "max_p_value": return r.MaxPValue;
        case "predictive_replicates": return r.NumberOfReplicates;
        case "has_misfit": return r.HasMisfit;
        case "number_of_valid_draws": return r.NumberOfValidDraws;
        case "summary_mean_quantile": return r.SummaryMeanQuantile[I(0)];
        case "summary_sd_quantile": return r.SummarySdQuantile[I(0)];
        case "summary_min_quantile": return r.SummaryMinQuantile[I(0)];
        case "summary_max_quantile": return r.SummaryMaxQuantile[I(0)];
        // --- T19: BootstrapDiagnostics dispatch (names match test_fixtures.cpp). ---
        case "boot_has_results": return r.BootHasResults ? 1.0 : 0.0;
        case "boot_total_replicates": return r.BootTotalReplicates;
        case "boot_attempted_replicates": return r.BootAttemptedReplicates;
        case "boot_failed_replicates": return r.BootFailedReplicates;
        case "boot_valid_replicates": return r.BootValidReplicates;
        case "boot_retained_replicates": return r.BootRetainedReplicates;
        case "boot_failure_rate": return r.BootFailureRate;
        case "boot_total_retries": return r.BootTotalRetries;
        case "boot_average_retries": return r.BootAverageRetries;
        case "boot_pivot_rejections": return r.BootPivotRejections;
        case "boot_mahalanobis_rejections": return r.BootMahalanobisRejections;
        case "boot_transform_failures": return r.BootTransformFailures;
        case "boot_status_success_count": return r.BootStatusSuccessCount;
        case "boot_status_max_iterations_count": return r.BootStatusMaxIterationsCount;
        case "boot_status_max_function_evaluations_count": return r.BootStatusMaxFunctionEvaluationsCount;
        case "boot_status_failure_count": return r.BootStatusFailureCount;
        case "boot_status_none_count": return r.BootStatusNoneCount;
        case "boot_optimizer_fallbacks": return r.BootOptimizerFallbacks;
        default: throw new Exception($"unknown analysis fixture method: {m}");
    }
}

// One arm per toolbox group. Later tasks extend this switch; the shape never changes. `options`
// carries whatever scalars/enum names/flags a group needs (mirrors toolbox_runner.hpp's
// options_json contract); correlation ignores it today but a group that needs it (e.g. goodness
// of fit's k/n/log_likelihood/threshold/distribution spec) does not have to change the signature.
// `asrt` is the fixture assertion driving this call: most methods return one scalar and ignore
// it, but a method that returns a named/ordered set (gof's "metrics"/"classification") or a bare
// array (gof's "aic_weights"/"rmse_weights") reads its `index`/`label` to pick one value, exactly
// as run_toolbox_kind's client-side select does in the C++/R/Python fixture runners.
static double ToolboxDispatch(string group, string method, List<double[]> data, JsonElement options,
                              JsonElement asrt)
{
    switch (group)
    {
        case "correlation":
            return method switch
            {
                "pearson"  => Correlation.Pearson(data[0], data[1]),
                "spearman" => Correlation.Spearman(data[0], data[1]),
                "kendall"  => Correlation.KendallsTau(data[0], data[1]),
                _ => throw new Exception($"unknown correlation method: {method}")
            };
        case "gof":
            return GofDispatch(method, data, options, asrt);
        case "spectra":
            return SpectraDispatch(method, data, options, asrt);
        case "statistics":
            return StatisticsDispatch(method, data, options, asrt);
        case "histogram":
            return HistogramToolboxDispatch(method, data, options, asrt);
        case "interpolation":
            return InterpolationDispatch(method, data, options, asrt);
        case "regression":
            return RegressionDispatch(method, data, options, asrt);
        case "sampling":
            return SamplingDispatch(method, data, options, asrt);
        case "probability":
            return ProbabilityDispatch(method, data, options, asrt);
        case "link":
            return LinkDispatch(method, data, options, asrt);
        case "trend":
            return TrendDispatch(method, data, options, asrt);
        default:
            throw new Exception($"unknown toolbox group: {group}");
    }
}

// Mirrors numerics/support/toolbox/sampling.hpp's run_sampling arm. The C# SobolSequence ctor
// takes no path (the direction numbers are a compiled resource -- see sobol.hpp's own
// divergence note), so a "path" key in options, if present, is simply not read here.
static double SamplingDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    if (method == "sobol")
    {
        int dimension = options.ValueKind == JsonValueKind.Object && options.TryGetProperty("dimension", out var d)
            ? d.GetInt32() : 1;
        int n = options.ValueKind == JsonValueKind.Object && options.TryGetProperty("n", out var nEl)
            ? nEl.GetInt32() : 1;
        int skip = options.ValueKind == JsonValueKind.Object && options.TryGetProperty("skip", out var sk)
            ? sk.GetInt32() : 0;
        var sobol = new SobolSequence(dimension);
        if (skip > 0) sobol.SkipTo(skip);
        var flat = new List<double>();
        for (int i = 0; i < n; i++) flat.AddRange(sobol.NextDouble());
        return ToolboxSelectFlat(asrt, flat.ToArray(), n, dimension);
    }
    if (method == "stratify")
    {
        double lower = options.GetProperty("lower").GetDouble();
        double upper = options.GetProperty("upper").GetDouble();
        int bins = options.GetProperty("bins").GetInt32();
        bool probability = OptBool(options, "probability", false);
        bool logarithmic = OptBool(options, "logarithmic", false);
        var stratOptions = new StratificationOptions(lower, upper, bins, probability);
        var strat = Stratify.XValues(stratOptions, logarithmic);
        var flat = new List<double>();
        foreach (var b in strat)
        {
            flat.Add(b.LowerBound);
            flat.Add(b.UpperBound);
            flat.Add(b.Midpoint);
            flat.Add(b.Weight);
        }
        return ToolboxSelectFlat(asrt, flat.ToArray(), strat.Count, 4);
    }
    throw new Exception($"unknown sampling method: {method}");
}

// Mirrors numerics/support/toolbox/probability.hpp's run_probability arm against the real
// Numerics.Data.Statistics.Probability.
static double ProbabilityDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    if (method != "joint") throw new Exception($"unknown probability method: {method}");
    double[] p = data[0];
    string dep = OptString(options, "dependency", "independent");
    var type = dep switch
    {
        "independent" => Probability.DependencyType.Independent,
        "positive" => Probability.DependencyType.PerfectlyPositive,
        "negative" => Probability.DependencyType.PerfectlyNegative,
        "correlation" => Probability.DependencyType.CorrelationMatrix,
        _ => throw new Exception($"unknown dependency '{dep}'; expected independent, positive, negative, or correlation")
    };
    if (data.Count < 2)
    {
        if (type == Probability.DependencyType.CorrelationMatrix)
            throw new Exception("dependency 'correlation' needs an indicator vector and a correlation matrix");
        return Probability.JointProbability(p, type);
    }
    var indicators = data[1].Select(v => (int)v).ToArray();
    if (data.Count < 3) return Probability.JointProbability(p, indicators, null, type);
    int n = p.Length;
    double[] flat = data[2];
    if (flat.Length != n * n) throw new Exception($"the correlation matrix must be {n} by {n}");
    var corr = new double[n, n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            corr[i, j] = flat[i * n + j];
    return Probability.JointProbability(p, indicators, corr, type);
}

// Mirrors numerics/support/toolbox/link.hpp's build_link against the real Numerics
// LinkFunctionFactory and the real RMC.BestFit link classes -- every constructor argument order
// and default matches the C++ port (verified against the same headers), so this is a faithful
// C# mirror rather than an independent re-derivation.
static double OptD(JsonElement? parameters, string key, double fallback)
{
    if (parameters is JsonElement p && p.ValueKind == JsonValueKind.Object && p.TryGetProperty(key, out var v))
        return v.GetDouble();
    return fallback;
}

static ILinkFunction BuildLink(JsonElement spec)
{
    string type = spec.GetProperty("type").GetString()!;
    JsonElement? parameters = spec.TryGetProperty("parameters", out var p) && p.ValueKind == JsonValueKind.Object
        ? p : (JsonElement?)null;
    double Opt(string key, double dflt) => OptD(parameters, key, dflt);

    switch (type)
    {
        case "Identity": return LinkFunctionFactory.Create(LinkFunctionType.Identity);
        case "Log": return LinkFunctionFactory.Create(LinkFunctionType.Log);
        case "Logit": return LinkFunctionFactory.Create(LinkFunctionType.Logit);
        case "Probit": return LinkFunctionFactory.Create(LinkFunctionType.Probit);
        case "ComplementaryLogLog": return LinkFunctionFactory.Create(LinkFunctionType.ComplementaryLogLog);
        case "FisherZ": return LinkFunctionFactory.Create(LinkFunctionType.FisherZ);
        case "YeoJohnson": return new YeoJohnsonLink(Opt("lambda", 1.0));
        case "ASinH":
            return new ASinHLink(Opt("gamma0", 0.0), Opt("scale", 1.0), Opt("epsilon", 0.0), Opt("delta", 1.0));
        case "SES": return new SESLink(Opt("a", 1.0));
        case "LogSES": return new LogSESLink(Opt("sigma0", 1.0), Opt("a", 1.0), Opt("lambda", 0.2));
        case "LogASinH":
            return new LogASinHLink(Opt("sigma0", 1.0), Opt("log_scale", 1.0), Opt("epsilon", 0.0), Opt("delta", 1.0));
        case "Centered":
            if (!spec.TryGetProperty("inner", out var innerSpec))
                throw new Exception("link type 'Centered' needs an 'inner' link spec");
            return new CenteredLink(BuildLink(innerSpec), Opt("mu0", 0.0), Opt("scale", 1.0));
        default:
            throw new Exception($"unknown link type: {type}");
    }
}

static double LinkDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    ILinkFunction link = BuildLink(options.GetProperty("link"));
    double[] x = data[0];
    var values = method switch
    {
        "link" => x.Select(link.Link).ToArray(),
        "inverse_link" => x.Select(link.InverseLink).ToArray(),
        "d_link" => x.Select(link.DLink).ToArray(),
        _ => throw new Exception($"unknown link method: {method}")
    };
    return ToolboxSelectFlat(asrt, values, values.Length, 1);
}

// Mirrors numerics/support/toolbox/trend.hpp's run_trend: builds the type's own class-defaulted
// trend (RMC.BestFit's own SetTrendModel type -> instance switch, mirrored in
// trend_model_factory.hpp -- GeneralLinear falls through to ConstantTrend, matching upstream),
// then applies StartIndex/SetParameterValues from the spec exactly as build_spec_trend does.
static ITrendModel BuildTrend(JsonElement spec)
{
    string type = spec.GetProperty("type").GetString()!;
    ITrendModel t = type switch
    {
        "Constant" => new ConstantTrend(),
        "Cubic" => new CubicTrend(),
        "Exponential" => new ExponentialTrend(),
        "Linear" => new LinearTrend(),
        "Logistic" => new LogisticTrend(),
        "Power" => new PowerTrend(),
        "Quadratic" => new QuadraticTrend(),
        "Reciprocal" => new ReciprocalTrend(),
        "Sinusoidal" => new SinusoidalTrend(),
        "StepFunction" => new StepFunction(),
        "GeneralLinear" => new ConstantTrend(),
        _ => throw new Exception($"unknown trend model type: {type}")
    };
    if (spec.TryGetProperty("start_index", out var si)) t.StartIndex = si.GetInt32();
    if (spec.TryGetProperty("values", out var vs))
        t.SetParameterValues(vs.EnumerateArray().Select(e => e.GetDouble()).ToList());
    return t;
}

static double TrendDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    ITrendModel t = BuildTrend(options.GetProperty("trend"));
    if (method == "predict")
    {
        double[] values = data[0].Select(i => (double)t.Predict((int)i)).ToArray();
        return ToolboxSelectFlat(asrt, values, values.Length, 1);
    }
    if (method == "parameters")
    {
        double[] values = t.Parameters.Select(mp => mp.Value).ToArray();
        string select = asrt.ValueKind == JsonValueKind.Object && asrt.TryGetProperty("select", out var s)
            ? s.GetString()! : "value";
        if (select == "length") return values.Length;
        if (asrt.TryGetProperty("label", out var lbl))
        {
            string label = lbl.GetString()!;
            for (int i = 0; i < t.Parameters.Count; i++)
                if (t.Parameters[i].Name == label) return t.Parameters[i].Value;
            throw new Exception($"trend result has no label '{label}'");
        }
        return ToolboxSelectFlat(asrt, values, values.Length, 1);
    }
    throw new Exception($"unknown trend method: {method}");
}

// Selects a value the way the fixture runners' client-side "select" logic does, generalized
// to a FLAT array with a known {rows, cols} shape (numerics.support.toolbox_runner.hpp's `dims`):
// "length" -> flat.Length, "rows"/"columns" -> the shape, else index into flat[index]. `label` is
// not answerable here -- unlike the C++/R/Python `toolbox_select` helpers, this function has no
// `names` array to look an assertion's `label` key up against, because every caller builds its
// flat array by hand from a positional (matrix-shaped) C# result. Two callers (sampling.stratify,
// regression.prediction_intervals) DO carry real names alongside their dims in the C++ result, so
// a `label` there is a real gap rather than a nonsensical request -- but threading a names array
// through every call site for a combination no fixture exercises is not worth doing blind, so this
// throws instead of silently falling through to `ToolboxSelectIndex`'s `"index"`-or-0 default,
// which used to read index 0 in the emitter while the C++/R/Python runners honored `label`
// correctly. Was a documented known limitation (CHANGELOG.md's v0.6.0 entry); now closed by making
// the mismatch loud instead of silent.
static double ToolboxSelectFlat(JsonElement asrt, double[] flat, int rows, int cols)
{
    if (asrt.ValueKind == JsonValueKind.Object && asrt.TryGetProperty("label", out var lbl))
        throw new Exception(
            $"toolbox select by label '{lbl.GetString()}' is not supported for a flat matrix " +
            "result (no names array); use index/rows/columns instead");
    string select = asrt.ValueKind == JsonValueKind.Object && asrt.TryGetProperty("select", out var s)
        ? s.GetString()! : "value";
    if (select == "length") return flat.Length;
    if (select == "rows") return rows;
    if (select == "columns") return cols;
    return flat[ToolboxSelectIndex(asrt)];
}

// Same selection grammar as ToolboxSelectFlat, for a method whose C++ ToolboxResult carries NO
// `dims` at all (numerics/support/toolbox/interpolation.hpp's `linear`/`bilinear` and
// numerics/support/toolbox/regression.hpp's `residuals`/`predict` never set `r.dims`): `select:
// "rows"`/`"columns"` throws here, matching what the C++/R/Python `toolbox_select` helpers do
// when `r.dims` is empty, instead of answering a fabricated `(1, flat.Length)` shape the way
// routing these through `ToolboxSelectFlat` used to (a `select: "rows"` on `interpolation.linear`
// silently returned `1`).
static double ToolboxSelectFlatNoDims(JsonElement asrt, double[] flat)
{
    if (asrt.ValueKind == JsonValueKind.Object && asrt.TryGetProperty("label", out var lbl))
        throw new Exception(
            $"toolbox select by label '{lbl.GetString()}' is not supported for a flat " +
            "result (no names array); use index instead");
    string select = asrt.ValueKind == JsonValueKind.Object && asrt.TryGetProperty("select", out var s)
        ? s.GetString()! : "value";
    if (select == "length") return flat.Length;
    if (select == "rows" || select == "columns")
        throw new Exception($"toolbox select '{select}' has no dims for this method");
    return flat[ToolboxSelectIndex(asrt)];
}

// Mirrors numerics/support/toolbox_runner.hpp's run_histogram arm. bins == 0 (the default)
// selects the Rice-Rule constructor `Histogram(data)`; bins > 0 selects `Histogram(data, bins)`
// -- there is no lower/upper-bound overload to expose. "bins" flattens {lower, upper, midpoint,
// frequency} row-major (dims = {NumberOfBins, 4}); "statistics" returns the named
// {mean, median, mode, sd, lower, upper, bin_width, bins} set.
static double HistogramToolboxDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    double[] x = data[0];
    int bins = options.ValueKind == JsonValueKind.Object && options.TryGetProperty("bins", out var b)
        ? b.GetInt32() : 0;
    var h = bins > 0 ? new Histogram(x, bins) : new Histogram(x);
    if (method == "bins")
    {
        var flat = new List<double>();
        for (int i = 0; i < h.NumberOfBins; i++)
        {
            var bin = h[i];
            flat.Add(bin.LowerBound);
            flat.Add(bin.UpperBound);
            flat.Add(bin.Midpoint);
            flat.Add(bin.Frequency);
        }
        return ToolboxSelectFlat(asrt, flat.ToArray(), h.NumberOfBins, 4);
    }
    if (method == "statistics")
    {
        var names = new[] { "mean", "median", "mode", "sd", "lower", "upper", "bin_width", "bins" };
        var values = new[]
        {
            h.Mean, h.Median, h.Mode, h.StandardDeviation, h.LowerBound, h.UpperBound,
            h.BinWidth, (double)h.NumberOfBins
        };
        return ToolboxSelectNamed(asrt, names, values);
    }
    throw new Exception($"unknown histogram method: {method}");
}

// Mirrors numerics/support/toolbox_runner.hpp's run_interpolation arm: Linear's separate
// Extrapolate() surface (vs. the clamp-to-end-knot Interpolate()) and Bilinear's three
// independent transforms.
static Numerics.Data.Transform ParseInterpolationTransform(string s) => s switch
{
    "none" => Numerics.Data.Transform.None,
    "log" => Numerics.Data.Transform.Logarithmic,
    "normal_z" => Numerics.Data.Transform.NormalZ,
    _ => throw new Exception($"unknown transform '{s}'; expected none, log, or normal_z")
};

static Numerics.Data.SortOrder ParseSortOrder(string s) => s switch
{
    "ascending" => Numerics.Data.SortOrder.Ascending,
    "descending" => Numerics.Data.SortOrder.Descending,
    _ => throw new Exception($"unknown sort order '{s}'; expected ascending or descending")
};

static string OptString(JsonElement options, string key, string fallback) =>
    options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out var v) ? v.GetString()! : fallback;

static bool OptBool(JsonElement options, string key, bool fallback) =>
    options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out var v) ? v.GetBoolean() : fallback;

static double InterpolationDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    var order = ParseSortOrder(OptString(options, "sort_order", "ascending"));
    if (method == "linear")
    {
        double[] x = data[0], y = data[1], xout = data[2];
        var interp = new Linear(x, y, order)
        {
            XTransform = ParseInterpolationTransform(OptString(options, "x_transform", "none")),
            YTransform = ParseInterpolationTransform(OptString(options, "y_transform", "none"))
        };
        bool extrapolate = OptBool(options, "extrapolate", false);
        var values = xout.Select(v => extrapolate ? interp.Extrapolate(v) : interp.Interpolate(v)).ToArray();
        return ToolboxSelectFlatNoDims(asrt, values);
    }
    if (method == "bilinear")
    {
        double[] x1 = data[0], x2 = data[1], flat = data[2], x1out = data[3], x2out = data[4];
        var y = new double[x1.Length, x2.Length];
        for (int i = 0; i < x1.Length; i++)
            for (int j = 0; j < x2.Length; j++)
                y[i, j] = flat[i * x2.Length + j];
        var interp = new Bilinear(x1, x2, y, order)
        {
            X1Transform = ParseInterpolationTransform(OptString(options, "x1_transform", "none")),
            X2Transform = ParseInterpolationTransform(OptString(options, "x2_transform", "none")),
            YTransform = ParseInterpolationTransform(OptString(options, "y_transform", "none"))
        };
        var values = new double[x1out.Length];
        for (int i = 0; i < x1out.Length; i++) values[i] = interp.Interpolate(x1out[i], x2out[i]);
        return ToolboxSelectFlatNoDims(asrt, values);
    }
    throw new Exception($"unknown interpolation method: {method}");
}

// Mirrors numerics/support/toolbox_runner.hpp's run_regression arm against the real
// Numerics.Data.LinearRegression: predictors cross as one flattened row-major vector plus
// `rows`/`columns` options (the binding layer has no matrix type common to R and Python), so the
// Matrix here is filled element-by-element from that flat array rather than built via a
// (rows, cols, flat) constructor overload, which LinearAlgebra.Matrix does not have.
static double RegressionDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    int GetInt(string key, int fallback) =>
        options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out var v) ? v.GetInt32() : fallback;
    double[] flat = data[0];
    double[] y = data[1];
    int rows = GetInt("rows", y.Length);
    int cols = GetInt("columns", 1);
    bool intercept = OptBool(options, "intercept", true);
    var x = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            x[i, j] = flat[i * cols + j];
    var lm = new LinearRegression(x, new Vector(y), intercept);

    if (method == "fit")
    {
        var names = new List<string>();
        var values = new List<double>();
        for (int i = 0; i < lm.Parameters.Count; i++) { names.Add($"beta_{i + 1}"); values.Add(lm.Parameters[i]); }
        for (int i = 0; i < lm.ParameterStandardErrors.Count; i++) { names.Add($"se_{i + 1}"); values.Add(lm.ParameterStandardErrors[i]); }
        names.Add("r_squared"); values.Add(lm.RSquared);
        names.Add("adj_r_squared"); values.Add(lm.AdjRSquared);
        names.Add("sigma"); values.Add(lm.StandardError);
        names.Add("df"); values.Add(lm.DegreesOfFreedom);
        names.Add("n"); values.Add(lm.SampleSize);
        return ToolboxSelectNamed(asrt, names.ToArray(), values.ToArray());
    }
    if (method == "covariance")
    {
        int m = lm.Covariance.NumberOfRows;
        var flatCov = new double[m * m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) flatCov[i * m + j] = lm.Covariance[i, j];
        return ToolboxSelectFlat(asrt, flatCov, m, m);
    }
    if (method == "residuals") return ToolboxSelectFlatNoDims(asrt, lm.Residuals);
    if (method == "predict" || method == "prediction_intervals")
    {
        double[] newFlat = data[2];
        int predictRows = options.GetProperty("predict_rows").GetInt32();
        var xp = new Matrix(predictRows, cols);
        for (int i = 0; i < predictRows; i++)
            for (int j = 0; j < cols; j++)
                xp[i, j] = newFlat[i * cols + j];
        if (method == "predict")
        {
            var values = lm.Predict(xp);
            return ToolboxSelectFlatNoDims(asrt, values);
        }
        double alpha = options.ValueKind == JsonValueKind.Object && options.TryGetProperty("alpha", out var a)
            ? a.GetDouble() : 0.1;
        var pi = lm.PredictionIntervals(xp, alpha);
        int piRows = pi.GetLength(0);
        var flatPi = new double[piRows * 3];
        for (int i = 0; i < piRows; i++)
            for (int j = 0; j < 3; j++) flatPi[i * 3 + j] = pi[i, j];
        return ToolboxSelectFlat(asrt, flatPi, piRows, 3);
    }
    throw new Exception($"unknown regression method: {method}");
}

// Mirrors numerics/support/toolbox_runner.hpp's run_spectra arm for the two methods
// fixtures/toolbox/autocorrelation.json pins: "autocorrelation" (covariance and partial types --
// the correlation type is already cross-checked against Fourier.Autocorrelation by
// fixtures/special_functions/fourier.json's Fourier.autocorrelation_at) and
// "autocorrelation_ci". cross_correlation/dft/dft_real reuse the values already pinned by
// Fourier.correlation_at/fft_at/real_fft_at in fourier.json, so they are not re-dispatched here.
// "autocorrelation" flattens the {lag, value} pairs row-major (dims = {n, 2}, matching the C++
// port), so asrt.index selects a FLAT position -- fixture cases use index = 2*lag + 1 to read the
// value column. correlation_confidence_interval's C++ names are {"lower","upper"}, matching
// Autocorrelation.CorrelationConfidenceInterval's {lo, hi} order.
static double SpectraDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    if (method == "autocorrelation")
    {
        double[] x = data[0];
        int lagMax = options.ValueKind == JsonValueKind.Object && options.TryGetProperty("lag_max", out var lm)
            ? lm.GetInt32() : -1;
        string typeName = options.ValueKind == JsonValueKind.Object && options.TryGetProperty("type", out var t)
            ? t.GetString()! : "correlation";
        var type = typeName switch
        {
            "covariance" => Autocorrelation.Type.Covariance,
            "partial" => Autocorrelation.Type.Partial,
            "correlation" => Autocorrelation.Type.Correlation,
            _ => throw new Exception($"unknown spectra.autocorrelation type: {typeName}")
        };
        var acf = Autocorrelation.Function(x, lagMax, type);
        if (acf is null) throw new Exception("spectra.autocorrelation: series too short for the requested lag");
        int flatIndex = ToolboxSelectIndex(asrt);
        return acf[flatIndex / 2, flatIndex % 2];
    }
    if (method == "autocorrelation_ci")
    {
        int sampleSize = options.GetProperty("sample_size").GetInt32();
        double interval = options.ValueKind == JsonValueKind.Object && options.TryGetProperty("confidence_level", out var cl)
            ? cl.GetDouble() : 0.95;
        var ci = Autocorrelation.CorrelationConfidenceInterval(sampleSize, interval);
        return ToolboxSelectNamed(asrt, new[] { "lower", "upper" }, ci);
    }
    throw new Exception($"spectra method '{method}' has no dumped oracle case wired in the emitter");
}

// Mirrors numerics/support/toolbox_runner.hpp's run_statistics arm for the three methods
// fixtures/toolbox/statistics.json pins (product_moments/l_moments/ranks -- the only statistics
// methods with no existing special_function pin). summary/running/percentile/running_covariance
// are exercised by the ctest/testthat/pytest suites directly, not by a dumped C# oracle here.
static double StatisticsDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    double[] x = data[0];
    if (method == "product_moments")
        return ToolboxSelectNamed(asrt, new[] { "mean", "sd", "skewness", "kurtosis" }, Statistics.ProductMoments(x));
    if (method == "l_moments")
        return ToolboxSelectNamed(asrt, new[] { "l1", "l2", "t3", "t4" }, Statistics.LinearMoments(x));
    if (method == "ranks")
    {
        RejectDimsSelect(asrt, "statistics.ranks");
        return Statistics.RanksInPlace(x)[ToolboxSelectIndex(asrt)];
    }
    if (method == "percentile") RejectDimsSelect(asrt, "statistics.percentile");
    throw new Exception($"statistics method '{method}' has no dumped oracle case wired in the emitter");
}

// Selects a single value out of an ordered/named result the way the fixture runners' select
// logic does: `label` (by name) wins over `index` (by position, default 0).
static int ToolboxSelectIndex(JsonElement asrt)
{
    if (asrt.ValueKind == JsonValueKind.Object && asrt.TryGetProperty("index", out var idx))
        return idx.GetInt32();
    return 0;
}

// Throws if an assertion asks for `select: "rows"` or `"columns"` against a method whose C++
// ToolboxResult never sets `r.dims` (every `gof` method; `statistics.ranks`/`percentile`),
// matching what the C++/R/Python `toolbox_select` helpers do when `r.dims` is empty. These
// methods route through ToolboxSelectIndex/ToolboxSelectNamed, neither of which reads `select`
// at all, so without this guard a `rows`/`columns` request would silently fall through to the
// index-0 (or named) value instead of failing the way it should.
static void RejectDimsSelect(JsonElement asrt, string context)
{
    if (asrt.ValueKind == JsonValueKind.Object && asrt.TryGetProperty("select", out var s))
    {
        string select = s.GetString()!;
        if (select == "rows" || select == "columns")
            throw new Exception($"toolbox select '{select}' has no dims for {context}");
    }
}

static double ToolboxSelectNamed(JsonElement asrt, string[] names, double[] values)
{
    if (asrt.ValueKind == JsonValueKind.Object && asrt.TryGetProperty("label", out var lbl))
    {
        string want = lbl.GetString()!;
        int i = Array.IndexOf(names, want);
        if (i < 0) throw new Exception($"unknown label: {want}");
        return values[i];
    }
    return values[ToolboxSelectIndex(asrt)];
}

// Mirrors numerics/support/toolbox_runner.hpp's run_gof arm against the real
// Numerics.Data.Statistics.GoodnessOfFit (verified method-for-method against
// upstream/Numerics/Numerics/Data/Statistics/GoodnessOfFit.cs -- the C++ port's names are
// lower_snake_case, the C# statics are PascalCase, and GoodnessOfFit itself has no standalone
// Pearson method, so "pearson" dispatches to the real Correlation.Pearson the C++ port mirrors).
static double GofDispatch(string method, List<double[]> data, JsonElement options, JsonElement asrt)
{
    // No `gof` method ever sets `r.dims` in C++ -- one guard up front covers all of them (the
    // scalar arms below don't consult `asrt` at all, so without this a `select: "rows"`/"columns"`
    // would silently return the computed metric instead of failing).
    RejectDimsSelect(asrt, "gof." + method);
    double[] o = data.Count > 0 ? data[0] : Array.Empty<double>();
    double[] m = data.Count > 1 ? data[1] : Array.Empty<double>();
    int GetInt(string key) => options.GetProperty(key).GetInt32();
    double GetDouble(string key) => options.GetProperty(key).GetDouble();
    int GetK() => options.ValueKind == JsonValueKind.Object && options.TryGetProperty("k", out var kEl)
        ? kEl.GetInt32() : 0;

    if (method == "aic") return GoodnessOfFit.AIC(GetInt("k"), GetDouble("log_likelihood"));
    if (method == "aicc") return GoodnessOfFit.AICc(GetInt("n"), GetInt("k"), GetDouble("log_likelihood"));
    if (method == "bic") return GoodnessOfFit.BIC(GetInt("n"), GetInt("k"), GetDouble("log_likelihood"));
    if (method == "aic_weights") return GoodnessOfFit.AICWeights(o)[ToolboxSelectIndex(asrt)];
    if (method == "rmse_weights") return GoodnessOfFit.RMSEWeights(o)[ToolboxSelectIndex(asrt)];

    if (method == "ks" || method == "ad" || method == "chi_squared" || method == "rmse_dist")
    {
        var obs = (double[])o.Clone();
        Array.Sort(obs);
        var model = BuildSpecDistribution(options.GetProperty("model"));
        if (method == "ks") return GoodnessOfFit.KolmogorovSmirnov(obs, model);
        if (method == "ad") return GoodnessOfFit.AndersonDarling(obs, model);
        if (method == "chi_squared") return GoodnessOfFit.ChiSquared(obs, model);
        return data.Count > 1 ? GoodnessOfFit.RMSE(obs, m, model) : GoodnessOfFit.RMSE(obs, model);
    }

    if (method == "classification")
    {
        // Two already-binary label vectors, compared elementwise -- no threshold, matching the
        // real GoodnessOfFit classification statics exactly (ConfusionMatrix stays private).
        var names = new[] { "accuracy", "precision", "recall", "f1", "specificity",
                            "balanced_accuracy" };
        var values = new[] {
            GoodnessOfFit.Accuracy(o, m), GoodnessOfFit.Precision(o, m), GoodnessOfFit.Recall(o, m),
            GoodnessOfFit.F1Score(o, m), GoodnessOfFit.Specificity(o, m),
            GoodnessOfFit.BalancedAccuracy(o, m)
        };
        return ToolboxSelectNamed(asrt, names, values);
    }

    var allNames = new[] { "rmse", "mse", "mae", "mape", "smape", "nse", "log_nse", "kge", "kge_mod",
                           "pbias", "rsr", "pearson", "r_squared", "d", "d_mod", "d_ref", "ve" };
    if (method == "metrics")
    {
        var allValues = new[] {
            GoodnessOfFit.RMSE(o, m, GetK()), GoodnessOfFit.MSE(o, m), GoodnessOfFit.MAE(o, m),
            GoodnessOfFit.MAPE(o, m), GoodnessOfFit.sMAPE(o, m),
            GoodnessOfFit.NashSutcliffeEfficiency(o, m), GoodnessOfFit.LogNashSutcliffeEfficiency(o, m),
            GoodnessOfFit.KlingGuptaEfficiency(o, m), GoodnessOfFit.KlingGuptaEfficiencyMod(o, m),
            GoodnessOfFit.PBIAS(o, m), GoodnessOfFit.RSR(o, m),
            Correlation.Pearson(o, m), GoodnessOfFit.RSquared(o, m),
            GoodnessOfFit.IndexOfAgreement(o, m), GoodnessOfFit.ModifiedIndexOfAgreement(o, m),
            GoodnessOfFit.RefinedIndexOfAgreement(o, m), GoodnessOfFit.VolumetricEfficiency(o, m)
        };
        return ToolboxSelectNamed(asrt, allNames, allValues);
    }
    return method switch
    {
        "rmse" => GoodnessOfFit.RMSE(o, m, GetK()),
        "mse" => GoodnessOfFit.MSE(o, m),
        "mae" => GoodnessOfFit.MAE(o, m),
        "mape" => GoodnessOfFit.MAPE(o, m),
        "smape" => GoodnessOfFit.sMAPE(o, m),
        "nse" => GoodnessOfFit.NashSutcliffeEfficiency(o, m),
        "log_nse" => GoodnessOfFit.LogNashSutcliffeEfficiency(o, m),
        "kge" => GoodnessOfFit.KlingGuptaEfficiency(o, m),
        "kge_mod" => GoodnessOfFit.KlingGuptaEfficiencyMod(o, m),
        "pbias" => GoodnessOfFit.PBIAS(o, m),
        "rsr" => GoodnessOfFit.RSR(o, m),
        "pearson" => Correlation.Pearson(o, m),
        "r_squared" => GoodnessOfFit.RSquared(o, m),
        "d" => GoodnessOfFit.IndexOfAgreement(o, m),
        "d_mod" => GoodnessOfFit.ModifiedIndexOfAgreement(o, m),
        "d_ref" => GoodnessOfFit.RefinedIndexOfAgreement(o, m),
        "ve" => GoodnessOfFit.VolumetricEfficiency(o, m),
        _ => throw new Exception($"unknown gof method: {method}")
    };
}

// --dump: the sanctioned curation path (see fixtures/README.md and the Task 5 brief).
// Author a fixture case with placeholder "expected" values, run
// `verify_oracles.py --dump` (threads this flag through to `dotnet run -- --dump`), paste
// the printed actuals into the fixture, then re-run without --dump to verify. Kept small:
// one helper that prints one JSON line per assertion instead of comparing.
static void DumpLine(string target, string caseName, string method, JsonElement[] args, Func<object> compute)
{
    object actualOrError;
    try { actualOrError = compute(); }
    catch (Exception ex) { actualOrError = "ERROR: " + ex.Message; }
    var line = new Dictionary<string, object?>
    {
        ["target"] = target,
        ["case"] = caseName,
        ["method"] = method,
        ["args"] = args,
        ["actual"] = actualOrError,
    };
    // PDF/CDF spot values can legitimately be +-Infinity (e.g. LogPDF of a non-finite
    // input); System.Text.Json refuses to write those without this option.
    var options = new JsonSerializerOptions
    {
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };
    Console.WriteLine(JsonSerializer.Serialize(line, options));
}

// --- main -------------------------------------------------------------------------------

bool dump = args.Contains("--dump");
string[] positional = args.Where(a => a != "--dump").ToArray();
string fixturesDir = positional.Length > 0 ? positional[0]
    : Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "fixtures");
fixturesDir = Path.GetFullPath(fixturesDir);
if (!Directory.Exists(fixturesDir))
{
    Console.Error.WriteLine($"fixtures directory not found: {fixturesDir}");
    return 2;
}

int pass = 0, fail = 0, skip = 0;
var failures = new List<string>();

foreach (var file in Directory.EnumerateFiles(fixturesDir, "*.json", SearchOption.AllDirectories))
{
    using var doc = JsonDocument.Parse(File.ReadAllText(file));
    var root = doc.RootElement;
    string? kindStr = root.TryGetProperty("kind", out var kind) ? kind.GetString() : null;

    // --- special_function branch --------------------------------------------------------
    // Most files dispatch every case through one file-level `target` (e.g. "Erf.function").
    // The Cholesky fixture groups several related dispatch keys in one file, so a case may
    // override `target`; a case without its own falls back to the file-level one, leaving
    // single-target files' behavior unchanged.
    if (kindStr == "special_function")
    {
        string fileTarget = root.GetProperty("target").GetString()!;
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string sfTarget = c.TryGetProperty("target", out var caseTarget)
                ? caseTarget.GetString()! : fileTarget;
            var fn = ResolveSpecialFunction(sfTarget);
            if (fn is null) { Console.Error.WriteLine($"  SKIP unknown special-function target: {sfTarget}"); continue; }
            string caseName = c.GetProperty("name").GetString()!;
            var caseArgsJson = c.GetProperty("args").EnumerateArray().ToArray();
            var argList = caseArgsJson.Select(ParseNum).ToArray();

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                string where = $"{sfTarget}/{caseName}/{method}";

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine(sfTarget, caseName, method, caseArgsJson, () => (object)fn(argList));
                    continue;
                }

                double actual;
                try { actual = fn(argList); }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); continue; }
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }
        }
        continue;
    }

    // --- goodness_of_fit branch ---------------------------------------------------------
    if (kindStr == "goodness_of_fit")
    {
        var dsSets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var dsNode))
            foreach (var kv in dsNode.EnumerateObject())
                dsSets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            string fn = c.GetProperty("function").GetString()!;

            // Resolve inputs: scalar args or observed/modeled dataset pairs
            double[] scalarArgs = c.TryGetProperty("args", out var argsNode)
                ? argsNode.EnumerateArray().Select(ParseNum).ToArray()
                : Array.Empty<double>();
            double[]? obs = c.TryGetProperty("observed_dataset", out var obsName)
                ? dsSets[obsName.GetString()!] : null;
            double[]? mod = c.TryGetProperty("modeled_dataset", out var modName)
                ? dsSets[modName.GetString()!] : null;

            double actual;
            try
            {
                actual = fn switch
                {
                    "AIC"  => GoodnessOfFit.AIC((int)scalarArgs[0], scalarArgs[1]),
                    "AICc" => GoodnessOfFit.AICc((int)scalarArgs[0], (int)scalarArgs[1], scalarArgs[2]),
                    "BIC"  => GoodnessOfFit.BIC((int)scalarArgs[0], (int)scalarArgs[1], scalarArgs[2]),
                    "MSE"  => GoodnessOfFit.MSE(obs!, mod!),
                    "MAE"  => GoodnessOfFit.MAE(obs!, mod!),
                    "NashSutcliffeEfficiency"    => GoodnessOfFit.NashSutcliffeEfficiency(obs!, mod!),
                    "KlingGuptaEfficiency"       => GoodnessOfFit.KlingGuptaEfficiency(obs!, mod!),
                    "KlingGuptaEfficiencyMod"    => GoodnessOfFit.KlingGuptaEfficiencyMod(obs!, mod!),
                    "PBIAS"                      => GoodnessOfFit.PBIAS(obs!, mod!),
                    "RSR"                        => GoodnessOfFit.RSR(obs!, mod!),
                    "IndexOfAgreement"           => GoodnessOfFit.IndexOfAgreement(obs!, mod!),
                    "ModifiedIndexOfAgreement"   => GoodnessOfFit.ModifiedIndexOfAgreement(obs!, mod!),
                    "RefinedIndexOfAgreement"    => GoodnessOfFit.RefinedIndexOfAgreement(obs!, mod!),
                    "VolumetricEfficiency"       => GoodnessOfFit.VolumetricEfficiency(obs!, mod!),
                    _ => throw new Exception($"unknown goodness_of_fit function: {fn}")
                };
            }
            catch (Exception ex) { fail++; failures.Add($"gof/{caseName}: {ex.Message}"); continue; }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string where = $"gof/{caseName}";
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }
        }
        continue;
    }

    // --- toolbox branch ------------------------------------------------------------------
    // Every Numerics utility group. Mirrors the GRAMMAR of numerics/support/toolbox_runner.hpp
    // against the real C# statics; the dispatch below is this file's own transcription, so an
    // oracle never runs the code under test. --dump supported for curation (see DumpLine()).
    if (kindStr == "toolbox")
    {
        var tbSets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var tbDs))
            foreach (var kv in tbDs.EnumerateObject())
                tbSets[kv.Name] = kv.Value.EnumerateArray().Select(ParseNum).ToArray();

        string group = root.GetProperty("group").GetString()!;
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            var data = new List<double[]>();
            if (c.TryGetProperty("data", out var dataNode))
                foreach (var d in dataNode.EnumerateArray())
                    data.Add(d.ValueKind == JsonValueKind.String
                        ? tbSets[d.GetString()!]
                        : d.EnumerateArray().Select(ParseNum).ToArray());
            JsonElement options = c.TryGetProperty("options", out var optionsEl) ? optionsEl : default;

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                string where = $"toolbox/{group}/{caseName}/{method}";

                if (dump)
                {
                    var dumpArgs = c.TryGetProperty("data", out var dn)
                        ? dn.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                    DumpLine($"toolbox/{group}", caseName, method, dumpArgs,
                        () => (object)ToolboxDispatch(group, method, data, options, asrt));
                    continue;
                }

                double actual;
                try { actual = ToolboxDispatch(group, method, data, options, asrt); }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); continue; }
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }
        }
        continue;
    }

    // --- optimizer branch (Task 8) --------------------------------------------------------
    // Runs the SIX real C# optimizers (all deriving from the real Optimizer base -- unlike this
    // port's NelderMead/BrentSearch, which are deliberately standalone; see optimizer.hpp's file
    // header) against the built-in objectives fixtures/toolbox/optimizers.json names by
    // `construct.objective` (OptimizerTestFunction() above). Mirrors optimizer_runner.hpp's
    // grammar: construct carries method/objective/lower/upper/initial/maximize/seed; assertions
    // carry value/parameter/status. "value" un-applies the real Optimizer's FunctionScale sign
    // convention (Fitness = FunctionScale * raw, FunctionScale = -1 under Maximize()) back to the
    // raw objective value, matching what the C++/R/Python runner reports.
    if (kindStr == "optimizer")
    {
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            var construct = c.GetProperty("construct");
            string method = construct.GetProperty("method").GetString()!;
            string objectiveName = construct.TryGetProperty("objective", out var objEl)
                ? objEl.GetString()! : "DeJong";
            Func<double[], double> objective = OptimizerTestFunction(objectiveName);
            double[] lower = construct.TryGetProperty("lower", out var lowerEl)
                ? lowerEl.EnumerateArray().Select(ParseNum).ToArray() : Array.Empty<double>();
            double[] upper = construct.TryGetProperty("upper", out var upperEl)
                ? upperEl.EnumerateArray().Select(ParseNum).ToArray() : Array.Empty<double>();
            double[] initial = construct.TryGetProperty("initial", out var initialEl)
                ? initialEl.EnumerateArray().Select(ParseNum).ToArray() : Array.Empty<double>();
            bool maximize = construct.TryGetProperty("maximize", out var maxEl) && maxEl.GetBoolean();
            int? seed = construct.TryGetProperty("seed", out var seedEl) ? seedEl.GetInt32() : null;

            Optimizer optimizer = method switch
            {
                "de" => new DifferentialEvolution(objective, lower.Length, lower, upper),
                "bfgs" => new BFGS(objective, initial.Length, initial, lower, upper),
                "powell" => new Powell(objective, initial.Length, initial, lower, upper),
                "mlsl" => new MLSL(objective, initial.Length, initial, lower, upper),
                "nelder_mead" => new NelderMead(objective, initial.Length, initial, lower, upper),
                "brent" => new BrentSearch(x => objective([x]), lower[0], upper[0]),
                _ => throw new Exception($"unknown optimizer method: {method}")
            };
            if (seed.HasValue)
            {
                if (optimizer is DifferentialEvolution deOptimizer) deOptimizer.PRNGSeed = seed.Value;
                else if (optimizer is MLSL mlslOptimizer) mlslOptimizer.PRNGSeed = seed.Value;
            }

            if (maximize) optimizer.Maximize(); else optimizer.Minimize();
            double[] parameters = optimizer.BestParameterSet.Values;
            double value = maximize ? -optimizer.BestParameterSet.Fitness : optimizer.BestParameterSet.Fitness;
            string status = optimizer.Status.ToString();

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string am = asrt.GetProperty("method").GetString()!;
                string where = $"optimizer/{caseName}/{am}";
                if (am == "status")
                {
                    string expected = asrt.GetProperty("expected").GetString()!;
                    if (status == expected) pass++;
                    else { fail++; failures.Add($"{where}: expected {expected} got {status}"); }
                    continue;
                }
                double actual = am switch
                {
                    "value" => value,
                    "parameter" => parameters[asrt.GetProperty("args")[0].GetInt32()],
                    _ => throw new Exception($"unknown optimizer fixture assertion method: {am}")
                };
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }
        }
        continue;
    }

    // --- callback branch (the callback surface, Task 1) -------------------------------------
    // Drives the REAL C# Brent / NumericalDerivative against the delegates
    // CallbackScalarFunction/CallbackVectorFunction above, mirroring callback/math.hpp's own
    // dispatch: construct carries group/method/callback/options; assertions carry
    // value (`args: [index]` into the flat result), dim (`args: [index]` into {rows, cols}), or
    // status. The hessian result is flattened row-major, exactly as the C++ runner flattens it.
    //
    // The same branch drives "callback_cross_language" (Task 8), each of whose cases nests an
    // "mcmc" and a "bootstrap" sub-block shaped exactly like a "callback"-kind case: CallbackSubCases
    // above flattens both kinds into the same triples, so the cross-language fixture reuses this
    // evaluation path verbatim rather than growing one of its own.
    if (kindStr == "callback" || kindStr == "callback_cross_language")
    {
        foreach (var (caseName, subLabel, c) in CallbackSubCases(root, kindStr))
        {
            string dumpTarget = subLabel.Length == 0 ? "callback" : $"callback_cross_language/{subLabel}";
            var construct = c.GetProperty("construct");
            string group = construct.GetProperty("group").GetString()!;
            if (group != "math" && group != "rng" && group != "mcmc" && group != "bootstrap" &&
                group != "gmm")
                throw new Exception($"unknown callback fixture group: {group}");
            string method = construct.GetProperty("method").GetString()!;
            string callbackName = construct.GetProperty("callback").GetString()!;
            var options = construct.TryGetProperty("options", out var optEl)
                ? optEl : default;
            double Opt(string key, double dflt) =>
                options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out var v)
                    ? ParseNum(v) : dflt;
            // P2 "math extras": root_find_newton's bracket presence check (BOTH lower and upper
            // present selects the robust variant, not a method sub-key) needs to distinguish
            // "absent" from "present with a default-looking value", which Opt's own dflt cannot.
            bool Has(string key) =>
                options.ValueKind == JsonValueKind.Object && options.TryGetProperty(key, out _);
            double[] OptVector(string key)
            {
                if (options.ValueKind != JsonValueKind.Object || !options.TryGetProperty(key, out var v))
                    throw new Exception($"callback/{method} requires the option '{key}'");
                return v.EnumerateArray().Select(ParseNum).ToArray();
            }

            double[] values;
            int[] dims;
            // Set by the mcmc arm, which is the one group whose result is partly NAMED (see
            // callback/mcmc.hpp's layout note); every other arm leaves it empty and its cases
            // assert by index.
            string[] valueNames = Array.Empty<string>();
            // Only quadrature has a C# class with a status to read (AdaptiveGaussKronrod over the
            // Integrator base). Brent and NumericalDerivative are static methods with none, so
            // their arms leave this at the "Success" the C++ runner unconditionally reports for
            // them -- see the status assertion below.
            string statusName = "Success";
            if (group == "mcmc")
            {
                if (method != "sample") throw new Exception($"unknown mcmc fixture method: {method}");
                var f = CallbackMcmcFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not an mcmc log-likelihood");
                // The two other delegates, each resolved out of the same catalog. An absent key
                // means no proposal (illegal for Gibbs, unused by everything else) and the C#
                // class's own default gradient.
                Gibbs.Proposal? proposalFn = null;
                if (construct.TryGetProperty("proposal", out var propEl))
                {
                    string pn = propEl.GetString()!;
                    proposalFn = CallbackProposalFunction(pn)
                        ?? throw new Exception($"callback '{pn}' is not a proposal function");
                }
                HMC.Gradient? gradientFn = null;
                if (construct.TryGetProperty("gradient", out var gradEl))
                {
                    string gn = gradEl.GetString()!;
                    gradientFn = CallbackGradientFunction(gn)
                        ?? throw new Exception($"callback '{gn}' is not a gradient function");
                }
                var sampler = BuildAndSampleCallbackMcmc(options, new LogLikelihood(f), proposalFn,
                                                          gradientFn);
                (values, valueNames, dims) = FlattenCallbackMcmc(sampler);
            }
            else if (group == "bootstrap")
            {
                if (method != "run") throw new Exception($"unknown bootstrap fixture method: {method}");
                // `callback` names the RESAMPLE delegate -- the one handed the generator, this
                // group's counterpart of the mcmc group's log-likelihood -- and the other four
                // have keys of their own. An absent `jackknife` means every method but BCa; `fit`
                // and `fit_with_covariance` are the two fitting delegates the run types take, and
                // a case supplies exactly the one its own `run_type` needs.
                var resampleFn = CallbackResampleFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not a resample function");
                string statName = construct.GetProperty("statistic").GetString()!;
                var statFn = CallbackStatisticFunction(statName)
                    ?? throw new Exception($"callback '{statName}' is not a statistic function");
                string runType = options.ValueKind == JsonValueKind.Object &&
                                 options.TryGetProperty("run_type", out var runTypeEl)
                    ? runTypeEl.GetString()! : "regular";
                if (runType == "pivotal")
                {
                    string fitCovName = construct.GetProperty("fit_with_covariance").GetString()!;
                    var fitCovFn = CallbackFitWithCovarianceFunction(fitCovName)
                        ?? throw new Exception($"callback '{fitCovName}' is not a covariance-aware fit function");
                    (values, valueNames, dims) =
                        RunCallbackBootstrapPivotal(options, resampleFn, fitCovFn, statFn);
                }
                else if (runType != "regular")
                {
                    throw new Exception($"unknown run_type '{runType}'; expected one of regular, pivotal");
                }
                else
                {
                    string fitName = construct.GetProperty("fit").GetString()!;
                    var fitFn = CallbackFitFunction(fitName)
                        ?? throw new Exception($"callback '{fitName}' is not a fit function");
                    Func<double[], int, double[]>? jackFn = null;
                    if (construct.TryGetProperty("jackknife", out var jackEl))
                    {
                        string jn = jackEl.GetString()!;
                        jackFn = CallbackJackknifeFunction(jn)
                            ?? throw new Exception($"callback '{jn}' is not a jackknife function");
                    }
                    (values, valueNames, dims) =
                        RunCallbackBootstrap(options, resampleFn, fitFn, statFn, jackFn);
                }
            }
            else if (group == "gmm")
            {
                if (method != "fit") throw new Exception($"unknown gmm fixture method: {method}");
                // `callback` names the MOMENT CONDITION function -- this group's required delegate,
                // its counterpart of the mcmc group's log-likelihood -- and the two optional ones
                // have keys of their own. An absent key means the C# class's own numerical Jacobian
                // and no penalty.
                var momentFn = CallbackMomentConditionFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not a moment condition function");
                JacobianFunction? jacobianFn = null;
                if (construct.TryGetProperty("jacobian", out var jacEl))
                {
                    string jn = jacEl.GetString()!;
                    jacobianFn = CallbackJacobianFunction(jn)
                        ?? throw new Exception($"callback '{jn}' is not a jacobian function");
                }
                PenaltyFunction? penaltyFn = null;
                if (construct.TryGetProperty("penalty", out var penEl))
                {
                    string pn2 = penEl.GetString()!;
                    penaltyFn = CallbackPenaltyFunction(pn2)
                        ?? throw new Exception($"callback '{pn2}' is not a penalty function");
                }
                (values, valueNames, dims) = RunCallbackGmm(options, momentFn, jacobianFn, penaltyFn);
            }
            else if (group == "rng")
            {
                if (method != "probe") throw new Exception($"unknown rng fixture method: {method}");
                var f = CallbackRngFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not an rng function");
                // Exactly as callback/rng.hpp: `seeds` (the C# int[] constructor, upstream's own
                // Test_MersenneTwister seeding) wins over `seed` (the int constructor every
                // sampler uses).
                MersenneTwister prng;
                if (options.ValueKind == JsonValueKind.Object &&
                    options.TryGetProperty("seeds", out var seedsEl))
                    prng = new MersenneTwister(seedsEl.EnumerateArray().Select(v => (int)ParseNum(v)).ToArray());
                else
                    prng = new MersenneTwister((int)Opt("seed", 0d));
                double[] parameters =
                    options.ValueKind == JsonValueKind.Object &&
                    options.TryGetProperty("parameters", out var parEl)
                        ? parEl.EnumerateArray().Select(ParseNum).ToArray()
                        : Array.Empty<double>();
                values = f(parameters, prng);
                dims = [values.Length];
            }
            else if (method == "root_find")
            {
                // P2 "math extras": `options.method` picks the ported root finder, absent meaning
                // "brent" -- preserving every fixture written before this key existed, exactly as
                // callback/math.hpp's own root_find arm does.
                var f = CallbackScalarFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not a scalar function");
                string rootMethod = options.ValueKind == JsonValueKind.Object &&
                                    options.TryGetProperty("method", out var rmEl)
                    ? rmEl.GetString()! : "brent";
                double tol = Opt("tolerance", 1E-8);
                int maxIter = (int)Opt("max_iterations", 1000);
                double rootValue = rootMethod switch
                {
                    "brent" => Numerics.Mathematics.RootFinding.Brent.Solve(
                        f, Opt("lower", 0d), Opt("upper", 0d), tol, maxIter),
                    "bisection" => Numerics.Mathematics.RootFinding.Bisection.Solve(
                        f, Opt("first_guess", 0d), Opt("lower", 0d), Opt("upper", 0d), tol, maxIter),
                    "secant" => Numerics.Mathematics.RootFinding.Secant.Solve(
                        f, Opt("lower", 0d), Opt("upper", 0d), tol, maxIter),
                    _ => throw new Exception($"math/root_find: unknown method '{rootMethod}'")
                };
                values = [rootValue];
                dims = [];
            }
            else if (method == "root_find_newton")
            {
                // The second callback, `df`, resolved out of the same catalog as `callback` --
                // math/root_find_newton's own second required delegate, mirroring the gmm group's
                // `jacobian` key.
                var f = CallbackScalarFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not a scalar function");
                string dfName = construct.GetProperty("df").GetString()!;
                var df = CallbackScalarFunction(dfName)
                    ?? throw new Exception($"callback '{dfName}' is not a scalar function");
                double firstGuess = Has("first_guess")
                    ? Opt("first_guess", 0d)
                    : throw new Exception("math/root_find_newton requires the option 'first_guess'");
                double tol = Opt("tolerance", 1E-8);
                int maxIter = (int)Opt("max_iterations", 1000);
                // Both present -- not a method sub-key -- selects the robust (bracketed) variant,
                // matching the ported class's own two static methods. See callback/math.hpp.
                double rootValue = Has("lower") && Has("upper")
                    ? Numerics.Mathematics.RootFinding.NewtonRaphson.RobustSolve(
                        f, df, firstGuess, Opt("lower", 0d), Opt("upper", 0d), tol, maxIter)
                    : Numerics.Mathematics.RootFinding.NewtonRaphson.Solve(
                        f, df, firstGuess, tol, maxIter);
                values = [rootValue];
                dims = [];
            }
            else if (method == "root_find_system")
            {
                var F = CallbackSystemFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not a system function");
                string jName = construct.GetProperty("jacobian").GetString()!;
                var Jraw = CallbackSystemJacobianFunction(jName)
                    ?? throw new Exception($"callback '{jName}' is not a system jacobian function");
                double[] firstGuess = OptVector("first_guess");
                double tol = Opt("tolerance", 1E-8);
                int maxIter = (int)Opt("max_iterations", 1000);
                Vector FVec(Vector v) => new Vector(F(v.ToArray()));
                Matrix JMat(Vector v) => new Matrix(Jraw(v.ToArray()));
                Vector rootVector = Numerics.Mathematics.RootFinding.NewtonRaphson.Solve(
                    FVec, JMat, new Vector(firstGuess), tol, maxIter);
                values = rootVector.ToArray();
                dims = [values.Length];
            }
            else if (method == "derivative")
            {
                var f = CallbackScalarFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not a scalar function");
                values = [NumericalDerivative.Derivative(f, Opt("point", 0d), Opt("step_size", -1d))];
                dims = [];
            }
            else if (method == "gradient")
            {
                var f = CallbackVectorFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not a vector function");
                values = NumericalDerivative.Gradient(f, OptVector("point"));
                dims = [values.Length];
            }
            else if (method == "hessian")
            {
                var f = CallbackVectorFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not a vector function");
                double[,] h = NumericalDerivative.Hessian(f, OptVector("point"));
                int n = h.GetLength(0), m = h.GetLength(1);
                values = new double[n * m];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < m; j++) values[i * m + j] = h[i, j];
                dims = [n, m];
            }
            else if (method == "quadrature")
            {
                var f = CallbackScalarFunction(callbackName)
                    ?? throw new Exception($"callback '{callbackName}' is not a scalar function");
                var agk = new Numerics.Mathematics.Integration.AdaptiveGaussKronrod(
                    f, Opt("lower", 0d), Opt("upper", 0d));
                // Written only when the fixture carries the key, exactly as callback/math.hpp
                // does, so an absent key exercises the C# class's OWN default.
                if (options.ValueKind == JsonValueKind.Object)
                {
                    if (options.TryGetProperty("absolute_tolerance", out var at))
                        agk.AbsoluteTolerance = ParseNum(at);
                    if (options.TryGetProperty("relative_tolerance", out var rt))
                        agk.RelativeTolerance = ParseNum(rt);
                    if (options.TryGetProperty("max_function_evaluations", out var mfe))
                        agk.MaxFunctionEvaluations = (int)ParseNum(mfe);
                }
                agk.Integrate();
                values = [agk.Result, agk.FunctionEvaluations, agk.StandardError];
                dims = [];
                statusName = agk.Status.ToString();
            }
            else
            {
                throw new Exception($"unknown callback fixture method: {method}");
            }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string am = asrt.GetProperty("method").GetString()!;
                string where = $"{dumpTarget}/{caseName}/{am}";
                int index = asrt.TryGetProperty("args", out var argsEl) ? argsEl[0].GetInt32() : 0;
                if (am == "status")
                {
                    // `statusName` is read off the driven C# object for every method whose class
                    // has one (today: quadrature's AdaptiveGaussKronrod.Status, an
                    // IntegrationStatus whose member names match the C++ status_name strings). For
                    // root_find/derivative/gradient/hessian it stays "Success", which is not a
                    // hardcoded oracle but the honest report that Brent and NumericalDerivative
                    // are static methods with no status object -- exactly what the C++ runner
                    // reports for them. A later group with a real status must set `statusName` in
                    // its own arm the same way the quadrature arm does.
                    string expected = asrt.GetProperty("expected").GetString()!;
                    if (dump) { DumpLine(dumpTarget, caseName, am, Array.Empty<JsonElement>(), () => (object)statusName); continue; }
                    if (expected == statusName) pass++;
                    else { fail++; failures.Add($"{where}: expected {expected} got {statusName}"); }
                    continue;
                }
                // `named` reads a value by the label the group gave it rather than by position,
                // because the mcmc group's summary block is long and its indices shift with the
                // chain and parameter counts -- "posterior_mean[0]" says what it pins; "12" does
                // not. Every runner resolves it the same way.
                double actual = am switch
                {
                    "value" => values[index],
                    "dim" => dims[index],
                    "named" => values[Array.IndexOf(valueNames, asrt.GetProperty("name").GetString()!) is int ni && ni >= 0
                        ? ni
                        : throw new Exception($"{where}: no result named '{asrt.GetProperty("name").GetString()}'")],
                    _ => throw new Exception($"unknown callback fixture assertion method: {am}")
                };
                if (dump)
                {
                    var dumpArgs = asrt.TryGetProperty("args", out var aEl)
                        ? aEl.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                    if (am == "named")
                        DumpLine(dumpTarget, caseName, asrt.GetProperty("name").GetString()!,
                                 Array.Empty<JsonElement>(), () => (object)actual);
                    else
                        DumpLine(dumpTarget, caseName, am, dumpArgs, () => (object)actual);
                    continue;
                }
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }
        }
        continue;
    }

    // --- toolbox_cross_language branch (Task 9) --------------------------------------------
    // The one fixture whose job is proving R and Python agree bit for bit: one seeded
    // DifferentialEvolution run over a built-in objective, plus one Sobol block and one
    // stratification, all reached through the SAME grammar the "optimizer"/"toolbox" branches
    // above already use (OptimizerTestFunction/ToolboxDispatch) -- just nested three ways under
    // one case instead of one kind per file, since the case's whole point is asserting all three
    // together as a single cross-language guarantee. --dump supported for curation.
    if (kindStr == "toolbox_cross_language")
    {
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;

            // --- optimizer sub-block (mirrors the "optimizer" kind branch above) -----------
            var optBlock = c.GetProperty("optimizer");
            var construct = optBlock.GetProperty("construct");
            string method = construct.GetProperty("method").GetString()!;
            string objectiveName = construct.TryGetProperty("objective", out var objEl)
                ? objEl.GetString()! : "DeJong";
            Func<double[], double> objective = OptimizerTestFunction(objectiveName);
            double[] lower = construct.TryGetProperty("lower", out var lowerEl)
                ? lowerEl.EnumerateArray().Select(ParseNum).ToArray() : Array.Empty<double>();
            double[] upper = construct.TryGetProperty("upper", out var upperEl)
                ? upperEl.EnumerateArray().Select(ParseNum).ToArray() : Array.Empty<double>();
            double[] initial = construct.TryGetProperty("initial", out var initialEl)
                ? initialEl.EnumerateArray().Select(ParseNum).ToArray() : Array.Empty<double>();
            bool maximize = construct.TryGetProperty("maximize", out var maxEl) && maxEl.GetBoolean();
            int? seed = construct.TryGetProperty("seed", out var seedEl) ? seedEl.GetInt32() : null;

            Optimizer optimizer = method switch
            {
                "de" => new DifferentialEvolution(objective, lower.Length, lower, upper),
                "bfgs" => new BFGS(objective, initial.Length, initial, lower, upper),
                "powell" => new Powell(objective, initial.Length, initial, lower, upper),
                "mlsl" => new MLSL(objective, initial.Length, initial, lower, upper),
                "nelder_mead" => new NelderMead(objective, initial.Length, initial, lower, upper),
                "brent" => new BrentSearch(x => objective([x]), lower[0], upper[0]),
                _ => throw new Exception($"unknown optimizer method: {method}")
            };
            if (seed.HasValue)
            {
                if (optimizer is DifferentialEvolution deOptimizer) deOptimizer.PRNGSeed = seed.Value;
                else if (optimizer is MLSL mlslOptimizer) mlslOptimizer.PRNGSeed = seed.Value;
            }
            if (maximize) optimizer.Maximize(); else optimizer.Minimize();
            double[] parameters = optimizer.BestParameterSet.Values;
            double optValue = maximize ? -optimizer.BestParameterSet.Fitness : optimizer.BestParameterSet.Fitness;
            string status = optimizer.Status.ToString();

            foreach (var asrt in optBlock.GetProperty("assertions").EnumerateArray())
            {
                string am = asrt.GetProperty("method").GetString()!;
                string where = $"toolbox_cross_language/{caseName}/optimizer/{am}";
                if (dump)
                {
                    var dumpArgs = asrt.TryGetProperty("args", out var aEl)
                        ? aEl.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                    DumpLine("toolbox_cross_language/optimizer", caseName, am, dumpArgs, () => am switch
                    {
                        "value" => (object)optValue,
                        "parameter" => (object)parameters[asrt.GetProperty("args")[0].GetInt32()],
                        "status" => (object)status,
                        _ => throw new Exception(
                            $"unknown toolbox_cross_language optimizer assertion method: {am}")
                    });
                    continue;
                }
                if (am == "status")
                {
                    string expected = asrt.GetProperty("expected").GetString()!;
                    if (status == expected) pass++;
                    else { fail++; failures.Add($"{where}: expected {expected} got {status}"); }
                    continue;
                }
                double actual = am switch
                {
                    "value" => optValue,
                    "parameter" => parameters[asrt.GetProperty("args")[0].GetInt32()],
                    _ => throw new Exception(
                        $"unknown toolbox_cross_language optimizer assertion method: {am}")
                };
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }

            // --- sobol / stratify sub-blocks, both routed through the same ToolboxDispatch the
            // "toolbox" kind branch above uses (group "sampling"; no positional data, options
            // only) ------------------------------------------------------------------------
            foreach (var (subKey, methodName) in new[] { ("sobol", "sobol"), ("stratify", "stratify") })
            {
                var block = c.GetProperty(subKey);
                JsonElement options = block.TryGetProperty("options", out var oEl) ? oEl : default;
                var data = new List<double[]>();
                foreach (var asrt in block.GetProperty("assertions").EnumerateArray())
                {
                    string where = $"toolbox_cross_language/{caseName}/{subKey}";
                    if (dump)
                    {
                        DumpLine($"toolbox_cross_language/{subKey}", caseName, methodName,
                            Array.Empty<JsonElement>(),
                            () => (object)ToolboxDispatch("sampling", methodName, data, options, asrt));
                        continue;
                    }
                    double actual;
                    try { actual = ToolboxDispatch("sampling", methodName, data, options, asrt); }
                    catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); continue; }
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
            }
        }
        continue;
    }

    // --- data_utility branch --------------------------------------------------------------
    // MGBT count, Box-Cox / Yeo-Johnson lambda + transform, plotting positions, Latin
    // hypercube. Same flat shape as goodness_of_fit; --dump supported for curation.
    if (kindStr == "data_utility")
    {
        var duSets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var duDs))
            foreach (var kv in duDs.EnumerateObject())
                duSets[kv.Name] = kv.Value.EnumerateArray().Select(ParseNum).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            string fn = c.GetProperty("function").GetString()!;
            double[] duArgs = c.TryGetProperty("args", out var duArgsNode)
                ? duArgsNode.EnumerateArray().Select(ParseNum).ToArray()
                : Array.Empty<double>();
            double[] duData = c.TryGetProperty("dataset", out var duName)
                ? duSets[duName.GetString()!] : Array.Empty<double>();

            Func<double> compute = fn switch
            {
                "MGBT" => () => MultipleGrubbsBeckTest.Function(duData),
                "BoxCoxLambda" => () => { BoxCox.FitLambda(duData, out double lam); return lam; },
                "BoxCoxTransform" => () => BoxCox.Transform(duData, duArgs[0])[(int)duArgs[1]],
                // Use the (values, out lambda) overload, matching Test_YeoJohnson.cs's
                // Test_FitLambda_InvalidSamples_ReturnsNaN and BoxCoxLambda above -- the
                // single-arg YeoJohnson.FitLambda(values) overload instead THROWS
                // ArgumentException for < 2 points, which is a different C# method (not
                // what the CanFitLambda-hardened NaN semantics being pinned here exercise).
                "YeoJohnsonLambda" => () => { YeoJohnson.FitLambda(duData, out double yjLam); return yjLam; },
                "YeoJohnsonTransform" => () => YeoJohnson.Transform(duData, duArgs[0])[(int)duArgs[1]],
                "PlottingPosition" => () => PlottingPositions.Function((int)duArgs[0], duArgs[1])[(int)duArgs[2]],
                // args: [sample_size, dimension, seed, row, col]
                "LHSRandom" => () => LatinHypercube.Random((int)duArgs[0], (int)duArgs[1], (int)duArgs[2])[(int)duArgs[3], (int)duArgs[4]],
                "LHSMedian" => () => LatinHypercube.Median((int)duArgs[0], (int)duArgs[1], (int)duArgs[2])[(int)duArgs[3], (int)duArgs[4]],
                // Threshold-selection diagnostics: args [u_min, u_max, n_thresholds,
                // confidence_level, point_index]; the function name picks the field.
                // `*PointCount` ignores the index and returns the surviving candidate count.
                _ when fn.StartsWith("MRL") => () =>
                {
                    var r = ThresholdDiagnostics.ComputeMeanResidualLife(
                        duData, duArgs[0], duArgs[1], (int)duArgs[2], duArgs[3]);
                    if (fn == "MRLPointCount") return r.Points.Count;
                    var pt = r.Points[(int)duArgs[4]];
                    return fn switch
                    {
                        "MRLThreshold" => pt.Threshold,
                        "MRLMeanExcess" => pt.MeanExcess,
                        "MRLLowerCI" => pt.LowerCI,
                        "MRLUpperCI" => pt.UpperCI,
                        "MRLCount" => pt.ExceedanceCount,
                        _ => throw new Exception($"unknown data_utility function: {fn}")
                    };
                },
                _ when fn.StartsWith("GPDStability") => () =>
                {
                    var r = ThresholdDiagnostics.ComputeParameterStability(
                        duData, duArgs[0], duArgs[1], (int)duArgs[2], duArgs[3]);
                    if (fn == "GPDStabilityPointCount") return r.Points.Count;
                    var pt = r.Points[(int)duArgs[4]];
                    return fn switch
                    {
                        "GPDStabilityThreshold" => pt.Threshold,
                        "GPDStabilityModifiedScale" => pt.ModifiedScale,
                        "GPDStabilityModifiedScaleLowerCI" => pt.ModifiedScaleLowerCI,
                        "GPDStabilityModifiedScaleUpperCI" => pt.ModifiedScaleUpperCI,
                        "GPDStabilityShape" => pt.Shape,
                        "GPDStabilityShapeLowerCI" => pt.ShapeLowerCI,
                        "GPDStabilityShapeUpperCI" => pt.ShapeUpperCI,
                        "GPDStabilityCount" => pt.ExceedanceCount,
                        _ => throw new Exception($"unknown data_utility function: {fn}")
                    };
                },
                _ => throw new Exception($"unknown data_utility function: {fn}")
            };

            if (dump)
            {
                var duArgsJson = c.TryGetProperty("args", out var duArgsNode2)
                    ? duArgsNode2.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                DumpLine("data_utility", caseName, fn, duArgsJson, () => (object)compute());
                continue;
            }

            double actual;
            try { actual = compute(); }
            catch (Exception ex) { fail++; failures.Add($"data_utility/{caseName}: {ex.Message}"); continue; }
            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string where = $"data_utility/{caseName}";
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }
        }
        continue;
    }

    // --- multivariate_distribution branch ------------------------------------------------
    if (kindStr == "multivariate_distribution")
    {
        string mvTarget = root.GetProperty("target").GetString()!;
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            MultivariateDistribution mvDist;
            try { mvDist = BuildMultivariate(mvTarget, c.GetProperty("construct")); }
            catch (Exception ex) { failures.Add($"{mvTarget}/{caseName}: build failed: {ex.Message}"); fail++; continue; }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{mvTarget}/{caseName}/{method}";
                string mode = asrt.GetProperty("mode").GetString()!;

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine(mvTarget, caseName, method, argList,
                             () => mode == "bool" ? (object)mvDist.ParametersValid
                                                   : (object)DispatchMultivariate(mvDist, mvTarget, method, argList));
                    continue;
                }

                try
                {
                    if (mode == "bool")
                    {
                        bool ok = mvDist.ParametersValid == asrt.GetProperty("expected").GetBoolean();
                        if (ok) pass++; else { fail++; failures.Add(where + ": bool mismatch"); }
                        continue;
                    }
                    double actual = DispatchMultivariate(mvDist, mvTarget, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- bivariate_copula branch ----------------------------------------------------------
    if (kindStr == "bivariate_copula")
    {
        string copTarget = root.GetProperty("target").GetString()!;
        var copDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var copDs))
            foreach (var kv in copDs.EnumerateObject())
                copDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            BivariateCopula copula;
            try { copula = BuildCopula(copTarget, c.GetProperty("construct"), copDatasets); }
            catch (Exception ex) { failures.Add($"{copTarget}/{caseName}: build failed: {ex.Message}"); fail++; continue; }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{copTarget}/{caseName}/{method}";
                string mode = asrt.GetProperty("mode").GetString()!;

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine(copTarget, caseName, method, argList,
                             () => mode == "bool" ? (object)copula.ParametersValid
                                                   : (object)DispatchCopula(copula, method, argList, copDatasets));
                    continue;
                }

                try
                {
                    if (mode == "bool")
                    {
                        bool ok = copula.ParametersValid == asrt.GetProperty("expected").GetBoolean();
                        if (ok) pass++; else { fail++; failures.Add(where + ": bool mismatch"); }
                        continue;
                    }
                    double actual = DispatchCopula(copula, method, argList, copDatasets);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- mcmc_sampler branch ---------------------------------------------------------------
    // One sampler run per case (see fixtures/README.md's mcmc_sampler schema): construct the
    // model + sampler, apply settings, sample() ONCE, post-process an MCMCResults, then
    // dispatch every assertion against that one cached (sampler, results) pair.
    if (kindStr == "mcmc_sampler")
    {
        string mcmcTarget = root.GetProperty("target").GetString()!;
        var mcmcDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var mcmcDs))
            foreach (var kv in mcmcDs.EnumerateObject())
                mcmcDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            MCMCSampler mcmcSampler;
            MCMCResults mcmcResults;
            try
            {
                mcmcSampler = BuildAndSampleMcmc(mcmcTarget, c.GetProperty("construct"), mcmcDatasets);
                mcmcResults = new MCMCResults(mcmcSampler);
            }
            catch (Exception ex)
            {
                failures.Add($"{mcmcTarget}/{caseName}: build/sample failed: {ex.Message}");
                fail++;
                continue;
            }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{mcmcTarget}/{caseName}/{method}";

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine(mcmcTarget, caseName, method, argList,
                             () => (object)DispatchMcmc(mcmcSampler, mcmcResults, method, argList));
                    continue;
                }

                try
                {
                    double actual = DispatchMcmc(mcmcSampler, mcmcResults, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- bootstrap branch --------------------------------------------------------------------
    // One bootstrap run per case (see fixtures/README.md's bootstrap schema): build the model
    // + configure + run() (or run_with_studentized_bootstrap()) ONCE, compute confidence
    // intervals ONCE, then dispatch every assertion against that one cached (boot, results)
    // pair.
    if (kindStr == "bootstrap")
    {
        var bsDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var bsDs))
            foreach (var kv in bsDs.EnumerateObject())
                bsDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            Bootstrap<double[]> boot;
            BootstrapResults results;
            try
            {
                (boot, results) = BuildAndRunBootstrap(c.GetProperty("construct"), bsDatasets);
            }
            catch (Exception ex)
            {
                failures.Add($"Bootstrap/{caseName}: build/run failed: {ex.Message}");
                fail++;
                continue;
            }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"Bootstrap/{caseName}/{method}";

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine("Bootstrap", caseName, method, argList,
                             () => (object)DispatchBootstrap(boot, results, method, argList));
                    continue;
                }

                try
                {
                    double actual = DispatchBootstrap(boot, results, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- analysis branch (Task A11: user-facing Analyses layer) ------------------------------
    // One analysis build+run per case; dispatch every assertion against the cached flat result.
    // Same "build once, dispatch many, --dump supported" shape as model_estimation.
    if (kindStr == "analysis")
    {
        string anTarget = root.GetProperty("target").GetString()!;
        var anDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var anDs))
            foreach (var kv in anDs.EnumerateObject())
                anDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            AnalysisData anData;
            try { anData = BuildAndRunAnalysis(anTarget, c.GetProperty("construct"), anDatasets); }
            catch (Exception ex)
            {
                failures.Add($"{anTarget}/{caseName}: build/run failed: {ex.Message}");
                fail++;
                continue;
            }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{anTarget}/{caseName}/{method}";

                if (dump)
                {
                    DumpLine(anTarget, caseName, method, argList,
                             () => (object)DispatchAnalysis(anData, method, argList));
                    continue;
                }

                // Oracle-exempt assertion (same treatment as the GEV std-err skips): a value the
                // shipped C++/R/Python harnesses check against the ported core, but which the real
                // C# library cannot reproduce because it rides an oracle-locked, documented port
                // deviation. D6: the three PriorInfluenceDiagnostics quantities collapse two Normal
                // parameter priors into one under the name-keyed dedup because the Phase-4 C++ model
                // deliberately leaves ModelParameter names empty (univariate_distribution_model.hpp
                // ~130) while C# keeps "Parameter Prior: Mean"/"Std Dev" distinct. Skipped here (not
                // failed) so the dev-only gate stays honest without papering the divergence into a
                // wide tolerance. See the fixture `source` + docs/upstream-csharp-issues.md.
                if (asrt.TryGetProperty("oracle_skip", out var osEl) && osEl.GetBoolean())
                {
                    skip++;
                    continue;
                }

                try
                {
                    double actual = DispatchAnalysis(anData, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- model_estimation branch (Task T12) --------------------------------------------------
    // One estimator build+run per case; dispatch every assertion against that cached estimator.
    // Same "build once, dispatch many, --dump supported" shape as mcmc_sampler/bootstrap.
    if (kindStr == "model_estimation")
    {
        string estTarget = root.GetProperty("target").GetString()!;
        var estDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var estDs))
            foreach (var kv in estDs.EnumerateObject())
                estDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;

            // GMM/B17C target (B12): the concrete Bulletin17CDistribution is NOT a ModelBase, so
            // it takes its own build+dispatch path (mirrors the C++ runner's separate GMM arm).
            if (estTarget == "GeneralizedMethodOfMoments")
            {
                (BestFitModels.Bulletin17CDistribution b17c, GeneralizedMethodOfMoments gmm, double[]? simulated) gmmCase;
                try { gmmCase = BuildGmm(c.GetProperty("construct"), estDatasets); }
                catch (Exception ex)
                {
                    failures.Add($"{estTarget}/{caseName}: build/run failed: {ex.Message}");
                    fail++;
                    continue;
                }

                foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
                {
                    string method = asrt.GetProperty("method").GetString()!;
                    var argList = asrt.TryGetProperty("args", out var av)
                        ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                    string where = $"{estTarget}/{caseName}/{method}";

                    if (dump)
                    {
                        DumpLine(estTarget, caseName, method, argList,
                                 () => (object)DispatchGmm(gmmCase.b17c, gmmCase.gmm, gmmCase.simulated, method, argList));
                        continue;
                    }

                    try
                    {
                        double actual = DispatchGmm(gmmCase.b17c, gmmCase.gmm, gmmCase.simulated, method, argList);
                        if (Compare(actual, asrt)) pass++;
                        else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                    }
                    catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
                }
                continue;
            }

            // Phase 7a families (P4): TimeSeries / SpatialGEV / RatingCurve / Bivariate derive
            // from ModelBase/IModel (not UnivariateDistributionModelBase), so they take the
            // separate general build + dispatch path. The `type` string selects the family.
            var modelSpecEl = c.GetProperty("construct").GetProperty("model");
            string modelType = modelSpecEl.TryGetProperty("type", out var mtEl)
                ? mtEl.GetString()! : "univariate_distribution";
            if (modelType is "time_series" or "spatial_gev" or "rating_curve" or "bivariate")
            {
                (BestFitModels.IModel model, object? estimator, double[]? simulated) gc;
                try { gc = BuildEstimationGeneral(estTarget, c.GetProperty("construct"), estDatasets); }
                catch (Exception ex)
                {
                    failures.Add($"{estTarget}/{caseName}: build/run failed: {ex.Message}");
                    fail++;
                    continue;
                }

                foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
                {
                    string method = asrt.GetProperty("method").GetString()!;
                    var argList = asrt.TryGetProperty("args", out var av2)
                        ? av2.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                    string where2 = $"{estTarget}/{caseName}/{method}";

                    if (dump)
                    {
                        DumpLine(estTarget, caseName, method, argList,
                                 () => (object)DispatchEstimationGeneral(gc, method, argList, c.GetProperty("construct")));
                        continue;
                    }

                    try
                    {
                        double actual = DispatchEstimationGeneral(gc, method, argList, c.GetProperty("construct"));
                        if (Compare(actual, asrt)) pass++;
                        else { fail++; failures.Add($"{where2}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                    }
                    catch (Exception ex) { fail++; failures.Add($"{where2}: {ex.Message}"); }
                }
                continue;
            }

            (BestFitModels.UnivariateDistributionModelBase model, object? estimator, double[]? simulated) estimator;
            try { estimator = BuildEstimation(estTarget, c.GetProperty("construct"), estDatasets); }
            catch (Exception ex)
            {
                failures.Add($"{estTarget}/{caseName}: build/run failed: {ex.Message}");
                fail++;
                continue;
            }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{estTarget}/{caseName}/{method}";

                if (dump)
                {
                    DumpLine(estTarget, caseName, method, argList,
                             () => (object)DispatchEstimation(estimator, method, argList, c.GetProperty("construct")));
                    continue;
                }

                try
                {
                    double actual = DispatchEstimation(estimator, method, argList, c.GetProperty("construct"));
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    if (kindStr != "univariate_distribution") continue;

    string target = root.GetProperty("target").GetString()!;

    var datasets = new Dictionary<string, double[]>();
    if (root.TryGetProperty("datasets", out var ds))
        foreach (var kv in ds.EnumerateObject())
            datasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    foreach (var c in root.GetProperty("cases").EnumerateArray())
    {
        string caseName = c.GetProperty("name").GetString()!;
        UnivariateDistributionBase dist;
        try { dist = Build(target, c.GetProperty("construct"), datasets); }
        catch (Exception ex) { failures.Add($"{target}/{caseName}: build failed: {ex.Message}"); fail++; continue; }

        foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
        {
            string method = asrt.GetProperty("method").GetString()!;
            var argList = asrt.TryGetProperty("args", out var av)
                ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
            string where = $"{target}/{caseName}/{method}";
            string mode = asrt.GetProperty("mode").GetString()!;

            // --dump: the curation path. Print target/case/method/args and the actual
            // C#-computed value as a JSON line instead of comparing against the
            // fixture's (possibly still-placeholder) "expected". See DumpLine().
            if (dump)
            {
                DumpLine(target, caseName, method, argList,
                         () =>
                         {
                             if (mode == "bool") return (object)dist.ParametersValid;
                             double? v = Dispatch(dist, method, argList);
                             return v is null ? (object)"SKIPPED" : (object)v.Value;
                         });
                continue;
            }

            try
            {
                if (mode == "bool")
                {
                    bool ok = dist.ParametersValid == asrt.GetProperty("expected").GetBoolean();
                    if (ok) pass++; else { fail++; failures.Add(where + ": bool mismatch"); }
                    continue;
                }
                double? actual = Dispatch(dist, method, argList);
                if (actual is null) { skip++; continue; }
                if (Compare(actual.Value, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual.Value:G17}"); }
            }
            catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
        }
    }
}

Console.WriteLine($"oracle verification: {pass} reproduced, {fail} failed, {skip} skipped (GEV std-err + oracle-exempt)");
foreach (var f in failures) Console.Error.WriteLine("  FAIL " + f);
return fail == 0 ? 0 : 1;

// Memoizes DispatchMlMapExtended's ProfileLikelihood/ParameterConfidenceIntervals calls, keyed by
// the case's BestParameterSet instance (stable for the case's lifetime; ParameterSet has no
// Equals/GetHashCode override, so reference identity is exactly "per case" -- see
// MemoizedProfileLikelihood/MemoizedParameterCIs above). Declared after the top-level statements
// (C# requires it), matching the AnalysisData precedent just below.
static class ProfileFitCache
{
    public static readonly Dictionary<ParameterSet, Dictionary<int, List<double[,]>>> Profile = new();
    public static readonly Dictionary<ParameterSet, Dictionary<double, double[,]>> Cis = new();
}

// Flat analysis-result surface (Task A11), mirroring test_fixtures.cpp's AnalysisResult. Only the
// fields a given target populates are filled; curve/CI vectors are indexed by the exceedance grid,
// the candidate_* vectors carry one entry per fitted candidate. Declared after the top-level
// statements (C# requires it), referenced by BuildAndRunAnalysis / DispatchAnalysis above.
class AnalysisData
{
    public List<double> Parameters = new(), ModeCurve = new(), MeanCurve = new(),
                        LowerCI = new(), UpperCI = new();
    public List<double> Exceedance = new(), PointEstimates = new(), Beta1 = new(), Nu = new(),
                        QuantileVariance = new();
    public List<double> CandAic = new(), CandBic = new(), CandRmse = new(), CandConverged = new();
    public double Aic = double.NaN, Bic = double.NaN, Dic = double.NaN, Rmse = double.NaN,
                 ConfidenceLevel = double.NaN;
    public int CandidateCount = 0;

    // --- D6 Diagnostics surface (target == "Diagnostics"). Mirrors test_fixtures.cpp's
    // AnalysisResult diagnostics slice field-for-field so the same fixture drives both. ---
    public int LevCount = 0, LevPriorCount = 0, InfCount = 0, PriCount = 0;
    public double TotalLeverage = double.NaN, TotalFitInfluence = double.NaN,
                 TotalVarianceInfluence = double.NaN;
    public List<double> LevObsLeverage = new(), LevObsFit = new(), LevObsVar = new(),
                        LevObsValue = new();
    public double MeanParetoK = double.NaN, MaxParetoK = double.NaN;
    public int CountParetoK05 = 0, CountParetoK07 = 0, CountParetoK10 = 0;
    public double ProportionProblematic = double.NaN, IsReliable = double.NaN;
    public List<double> InfParetoK = new(), InfElpdLoo = new();
    public double TotalPriorLogLik = double.NaN, TotalDataLogLik = double.NaN,
                 PriorToDataRatio = double.NaN, IsPriorInfluential = double.NaN,
                 MeanPriorPrecisionShare = double.NaN;

    // --- X12 extended-analysis surface (Phase 10 full parity). Mirrors test_fixtures.cpp's
    // ExtendedAnalysisResult slice field-for-field so the same fixtures drive both. ---
    // CoincidentFrequency Z output bins.
    public List<double> ZOutput = new();
    // SpatialGEV site results (per site) + site-0 quantile curve.
    public int SiteCount = 0;
    public List<double> SiteLocationMean = new(), SiteScaleMean = new(), SiteShapeMean = new();
    public List<double> Site0QuantileMean = new();
    // SpatialGEV cross-validation (populated only when construct.cross_validation is true).
    public double CvMae = double.NaN, CvRmse = double.NaN, CvMeanBias = double.NaN;
    // Predictive checks (posterior p-values / misfit; prior summary quantiles).
    public int NumberOfReplicates = 0, NumberOfValidDraws = 0;
    public double MeanPValue = double.NaN, SdPValue = double.NaN, SkewnessPValue = double.NaN,
                  MinPValue = double.NaN, MaxPValue = double.NaN, HasMisfit = double.NaN;
    public List<double> SummaryMeanQuantile = new(), SummarySdQuantile = new(),
                        SummaryMinQuantile = new(), SummaryMaxQuantile = new();

    // --- T19 BootstrapDiagnostics surface (target == "Bulletin17CAnalysis", Bootstrap /
    // BiasCorrectedBootstrap). Mirrors test_fixtures.cpp's AnalysisResult boot_* slice
    // field-for-field so the same fixture drives both. ---
    public bool BootHasResults = false;
    public int BootTotalReplicates = 0, BootAttemptedReplicates = 0, BootFailedReplicates = 0,
              BootValidReplicates = 0, BootRetainedReplicates = 0, BootTotalRetries = 0,
              BootPivotRejections = 0, BootMahalanobisRejections = 0, BootTransformFailures = 0,
              BootStatusSuccessCount = 0, BootStatusMaxIterationsCount = 0,
              BootStatusMaxFunctionEvaluationsCount = 0, BootStatusFailureCount = 0,
              BootStatusNoneCount = 0, BootOptimizerFallbacks = 0;
    public double BootFailureRate = double.NaN, BootAverageRetries = double.NaN;
}
