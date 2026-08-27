# P5 Machine Learning Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Work task-by-task;
> each task ends in a green build and a commit.

**Goal:** Port the whole `Numerics/Machine Learning/` subsystem -- the five supervised classes
(DecisionTree, RandomForest, KNearestNeighbors, NaiveBayes, GeneralizedLinearModel), the three
unsupervised ones (KMeans, GaussianMixtureModel, JenksNaturalBreaks) and their two support types
(DecisionNode, JenksCluster) -- validated by ctest transcriptions of the eight C# test files,
pinned by fixtures through all four runners with dotnet-gate reproduction, exposed in R and Python
behind a new `ml` toolbox group, documented with a worked example pair, and shipped as branch
`port-machine-learning` at v0.12.0. `GaussianMixtureModel` landing also un-gates the three members
P4 deferred to this phase: `HypothesisTests::unimodality_test`, `DataFrame::unimodality_test` and
`DataFrame::summary_hypothesis_test`.

**Architecture:** One new dispatch surface on a pattern the repo already established. `ml` is a new
one-header group under `core/include/corehydro/numerics/support/toolbox/`, dispatched by
`toolbox_runner.hpp` exactly as P3's `network` and P4's `hypothesis`/`paired_data` groups were,
taking the ONE shape the `regression` group already uses for a predictor matrix -- a flattened
row-major `data[0]`, a response `data[1]`, an optional new-data `data[2]`, with `rows`/`columns`/
`predict_rows` and every scalar knob in `options_json`. Nothing else is invented: the two P4-deferred
DataFrame members join the existing `models/data_frame_runner.hpp` behind the existing `data_frame`
fixture kind, and `unimodality` joins the existing `hypothesis` toolbox group as a thirteenth method.

**Tech Stack:** C++17 header-only core, cpp11 (R), pybind11 (Python), nlohmann/json fixture runner,
dotnet 10 oracle emitter.

## Global Constraints

- Every ported file carries `// ported from: <path> @ 2a0357a` (Numerics) or `@ c2e6192`
  (RMC.BestFit) and mirrors the C# class/method layout line-for-line where possible.
- Upstream paths contain spaces (`Numerics/Machine Learning/Supervised/`) -- quote them in every
  shell command. Numerics is valid UTF-8 and its working tree is clean, so reading and grepping it
  directly is fine.
- **RMC.BestFit is read with `git -C upstream/RMC-BestFit show c2e6192:'<path>'`, never grepped over
  the working tree.** The one measured exception recorded in the P4 plan still holds:
  `src/RMC.BestFit/Models/DataFrame/DataFrame.cs` at `c2e6192` IS valid UTF-8 (with a BOM) and must
  be read plain; piping it through `iconv` mangles the oracle-visible `λ` in a summary key.
- Oracle values live ONLY in `fixtures/*.json`; ctest suites transcribing C# test methods carry
  C#-test-literal assertions (the established second oracle class); never invent expected values.
  Where upstream has no numeric assertion at all (DecisionTree, RandomForest, the KMeans/GMM label
  counts beyond the ones asserted), curate with `python3 tools/verify_oracles.py --dump` against the
  real library and say so in the fixture's `source` string.
- Every new fixture case gets a dotnet emitter driver; `python3 tools/verify_oracles.py` must end
  `0 failed`. The `toolbox` branch of the emitter does NOT honor `oracle_skip` -- an assertion the
  C# cannot reproduce is OMITTED from the fixture, not masked.
- No `M_PI` (use `corehydro::numerics::kPi`); no namespace aliases named `gamma` or `stat`; no
  file-local `const` used implicitly in a capture-less lambda (MSVC C3493 -- use file-scope
  `constexpr`); no new external C++ dependencies; **no threads** (see scope decision 1).
- After editing any `corehydror/src/*.cpp`: `Rscript -e 'cpp11::cpp_register("corehydror")'`.
  Adding a toolbox GROUP does not require it (the glue is group-generic); adding a data-layer entry
  point does.
- Task 6 adds public methods to `DataFrame`, a class-layout change, so EVERY R rebuild from Task 6
  onward uses `R CMD INSTALL --preclean corehydror`.
- pytest reads fixtures materialized by pip; re-run
  `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy` after fixture edits.
  testthat reads `system.file("fixtures", package = "corehydror")`, so re-run `R CMD INSTALL`.
- Every new R export goes in BOTH `corehydror/_pkgdown.yml` and `site/_quarto.yml`
  `quartodoc.sections`, and in `corehydror/NAMESPACE` (hand-maintained -- do not let roxygen
  rewrite it). New Python exports join `corehydropy/src/corehydropy/<module>.py`'s `__all__` AND
  both the import block and `__all__` in `corehydropy/src/corehydropy/__init__.py`.
- R and Python error messages for the same condition must be textually identical (`stop(...,
  call. = FALSE)` vs `raise ValueError(...)`).
- Commits are GPG-signed, identity `Cam Bracken <cameron.bracken@pm.me>`, no Co-Authored-By
  trailers. Push only when Cam asks (the ship task pushes; that is the standing exception).
- **Branch base:** `port-machine-learning`, created off `fec9fe9` (`origin/main`, the merge of PR
  #30). Unlike P4, the stack HAS landed -- `main` is the base and the P5 PR targets `main`.
- **Baseline numbers, measured on this branch at creation (2026-08-27), not copied from prose:**
  ctest **108/108** (813 s wall). The P4 release records the rest: oracle gate 6128 reproduced / 0
  failed / 11 skipped; testthat 7161/0; pytest 1795. Re-measure R/Python/oracle on the first task
  that runs them; if any number differs, the fresh measurement governs (the standing process note:
  never reconcile against a stale census).

## Scope decisions taken during recon (do not re-litigate)

1. **`Parallel.For` becomes a serial loop everywhere, and that is exact, with one measured
   exception.** Every `Parallel.For` in this subsystem (RandomForest `Train`/`Predict`, kNN's three
   private helpers, KMeans `GetLabels`) writes only to its own index of a pre-allocated array and
   reads nothing another iteration writes, so serial execution gives bit-identical results. The seeds
   RandomForest and kNN hand their per-iteration workers are drawn SERIALLY before the loop
   (`Random.NextIntegers(NumberOfTrees)`), so the PRNG stream is order-independent too. The
   exception is `Statistics.ParallelMean`, whose `data.AsParallel().Sum()` splits the sum across
   `Environment.ProcessorCount` partitions and adds the partial sums, making its last bits
   machine-dependent in upstream itself. The port sums serially. It is reached from exactly two
   places: RandomForest `Predict`'s column 3 and kNN `PredictionIntervals`' column 3 (both the
   "mean" column). Task 3 MEASURES the divergence against the real library before anything is
   pinned; whatever it finds is written up in `docs/upstream-csharp-issues.md` and the affected
   fixture assertions are pinned at the measured tolerance with the measurement in the `source`
   string -- not masked with `oracle_skip`, not silently widened.
2. **The .NET comparison sort is load-bearing for kNN and is extracted, not duplicated.**
   `KNearestNeighbors` sorts its `kNNItem[]` with `Array.Sort(items, (a, b) =>
   a.Distance.CompareTo(b.Distance))`, an UNSTABLE introsort, and its `Test_GetNeighbors_MultiRow`
   query sits at (0.5, 0.5) in a symmetric cluster where four training points tie exactly. The tie
   permutation is therefore oracle-visible, and the repo already owns a faithful .NET
   `ArraySortHelper` introsort port -- inside `models/data_frame/data_frame_plotting.hpp`, one layer
   ABOVE `numerics/`. Task 1 moves those helpers verbatim into
   `numerics/utilities/dotnet_sort.hpp` and leaves `data_frame_plotting.hpp` including it, so the
   plotting-position oracles keep passing unchanged (a pure move: if any plotting fixture value
   shifts, the move was not verbatim). `Array.Sort(double[])` and `Array.Sort(int[])` (RandomForest
   `Predict`, kNN intervals, KMeans `Initialize`) stay `std::sort` -- for primitives, tied elements
   are indistinguishable, so no permutation is observable.
3. **LINQ ordering semantics are reproduced explicitly, not approximated.** Three call sites depend
   on them: `y.Distinct().ToArray()` (NaiveBayes `Classes`) and `x.Distinct().ToArray()`
   (DecisionTree classification thresholds) yield FIRST-APPEARANCE order, and
   `.GroupBy(i => i).OrderByDescending(grp => grp.Count()).Select(grp => grp.Key).First()`
   (DecisionTree's classification leaf, kNN's classification prediction) resolves a tie in favor of
   the group that appeared FIRST, because `GroupBy` yields groups in first-appearance order and
   LINQ's `OrderByDescending` is a stable sort. Both become named helpers in Task 1
   (`distinct_in_first_appearance_order`, `most_common_first_appearance_wins`) with the semantics in
   the header comment, so no call site re-derives them and none of them reaches for
   `std::unordered_set` (whose iteration order is unspecified).
4. **`GeneralizedLinearModel::Summary()` is severed, `ParameterNames` is not.** `Summary()` builds
   ~35 lines of formatted `StringBuilder` text with .NET-specific `N4`/`E2` numeric formatting and
   the `Y.Header` model name. It is presentation only, with no numeric surface, and joins the
   Bulletin17C GMM report text as a documented severance in the ported header and in
   `upstream/CLAUDE.md`. `ParameterNames` (which `Summary()` reads but which is a real public
   member) ships, including the C# `"β1"`-style default names -- write them as UTF-8 literals and
   verify the file stays valid UTF-8, exactly as P4 did for the `λ` summary key.
5. **The Jenks 7,889-value dataset stays in ctest and never enters a fixture.** `Test_Jenks_5/7/9
   Classes` all run against one 75 KB literal array, whose R BAMMtools breaks are the correctness
   oracle. `fixtures/` is symlinked into BOTH shipped packages and is already 2.1 MB against CRAN's
   5 MB guidance, so the array goes into the ctest (which no package ships) and the `ml` fixture
   gets a small curated Jenks case reproduced by the dotnet gate. This is the split the repo already
   uses: ctest for correctness against the literature, fixtures for cross-language reproduction.
6. **`RandomForest`'s 1000-tree default is kept in the ctest only if it runs in reasonable wall
   time.** DecisionTree's classification `Entropy` is O(n^2) per candidate threshold (its histogram
   lambda rescans `y` for every point), and the C# test hides that behind `Parallel.For` over 1000
   trees. Task 8 Step 1 MEASURES the serial C++ cost first. Rule: prefer the C# defaults; if the
   suite exceeds 60 s, lower `NumberOfTrees` in the ctest, mark the deviation in a numbered note
   naming the measured time at both settings, and keep the C# default as the class default. Never
   change the class default to make a test fast.
7. **No new user-facing "model object".** ML verbs train and answer in one call, the way
   `linear_regression()` already does. `ml_random_forest(x, y, newdata)` trains and returns the
   prediction intervals; there is no fitted handle to keep alive across the binding boundary. Where
   a class exposes several output blocks (KMeans's means/labels/iterations), the host verb makes one
   runner call per block and the runner re-trains -- deterministic under a seed, exactly as
   `regression`'s `fit`/`covariance`/`residuals` each rebuild their `LinearRegression`. Say so in
   both help pages for `ml_random_forest`, the one place the cost is user-visible.

---

### Task 1: Branch prerequisites -- the Statistics/Tools helpers, the Matrix accessors, and the
shared .NET sort and LINQ-order helpers

**Files:**
- Modify: `core/include/corehydro/numerics/data/statistics.hpp` (add `population_variance`,
  `population_standard_deviation`, `parallel_mean`, `five_number_summary`, `entropy`; rewrite the
  header notes that currently say each is omitted)
- Modify: `core/include/corehydro/numerics/tools.hpp` (add `standardize`)
- Modify: `core/include/corehydro/numerics/math/linalg/matrix.hpp` (add `row`, `column`, the
  column-list constructor, and the single-column constructor)
- Create: `core/include/corehydro/numerics/utilities/dotnet_sort.hpp` (the verbatim move)
- Modify: `core/include/corehydro/models/data_frame/data_frame_plotting.hpp` (delete the moved
  block, include the new header, keep the namespace alias so no call site changes)
- Create: `core/include/corehydro/numerics/machine_learning/support/linq_order.hpp`
- Test: `core/tests/test_ml_prerequisites.cpp` (new); Modify: `core/CMakeLists.txt`

**Interfaces:**
- Produces, in `namespace corehydro::numerics::data::statistics`:
  - `inline double population_variance(const std::vector<double>& data)` -- the N-normalizer
    running-difference recurrence at `Statistics.cs:218`, NOT `variance() * (n-1) / n`.
  - `inline double population_standard_deviation(const std::vector<double>& data)`
  - `inline double parallel_mean(const std::vector<double>& data)` -- a SERIAL sum (scope
    decision 1), with the divergence in a header note.
  - `inline std::vector<double> five_number_summary(const std::vector<double>& data)`
  - `inline double entropy(const std::vector<double>& data, const std::function<double(double)>& pdf)`
- Produces, in `namespace corehydro::numerics::tools`: `inline std::vector<double> standardize(const
  std::vector<double>& values)` -- returns an all-zero vector when `sd <= 0` or NaN, per
  `Tools.cs:351`.
- Produces, on `corehydro::numerics::math::linalg::Matrix`:
  - `std::vector<double> row(int i) const`, `std::vector<double> column(int j) const`
  - `explicit Matrix(const std::vector<double>& column_values)` -- an n-by-1 matrix (C#
    `Matrix(double[])`)
  - `explicit Matrix(const std::vector<std::vector<double>>& columns)` -- C#
    `Matrix(List<double[]>)`, where each entry is a COLUMN. Guard on ragged input with the C#
    message.
- Produces, in `namespace corehydro::numerics::utilities`: the six `dn_*` templates and
  `dotnet_list_sort<T, Compare>` moved verbatim from `data_frame_plotting.hpp:99-200`.
- Produces, in `namespace corehydro::numerics::machine_learning::support`:
  - `std::vector<double> distinct_in_first_appearance_order(const std::vector<double>& v)`
  - `double most_common_first_appearance_wins(const std::vector<double>& v)`

- [ ] **Step 1: Write the failing ctest**

Create `core/tests/test_ml_prerequisites.cpp`:

1. `population_variance` on a 10-value vector equals `variance(v) * (n - 1) / n` to `1e-12` (an
   identity check, not a new oracle) and returns NaN on an empty vector.
2. `parallel_mean` equals `mean` exactly on the same vector.
3. `five_number_summary` on an unsorted vector equals `{minimum, percentile(.25), percentile(.5),
   percentile(.75), maximum}` element for element.
4. `entropy(v, pdf)` with `pdf = [](double x){ return x; }` over `{0.25, 0.25, 0.5}` equals
   `-(0.25*log(0.25) + 0.25*log(0.25) + 0.5*log(0.5))`, and a `pdf` returning 0 for one point skips
   that point rather than producing `-inf`.
5. `standardize` on a vector with positive spread gives mean 0 / sd 1 to `1e-12`; on ten identical
   values it returns all zeros (the degenerate branch), NOT NaNs.
6. `Matrix` from a column list: three columns of five values give `number_of_rows() == 5`,
   `number_of_columns() == 3`, and `m(i, j) == columns[j][i]`; `row(2)` and `column(1)` return the
   expected vectors; the single-vector ctor gives an n-by-1.
7. `dotnet_list_sort` on a hand-built vector of `{value, tag}` pairs with a comparator on `value`
   alone, containing three tie runs, reproduces the SAME permutation as the pre-move code path.
   Assert the exact output tag order (record it by running the existing
   `data_frame_plotting.hpp` helper before the move -- this is a regression pin on a verbatim move,
   not a new oracle).
8. `distinct_in_first_appearance_order({3, 1, 3, 2, 1})` is `{3, 1, 2}`;
   `most_common_first_appearance_wins({2, 1, 2, 1, 3})` is `2` (both 1 and 2 appear twice; 2
   appeared first).

Wire the file into `core/CMakeLists.txt` beside the other `test_*` entries.

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake -S core -B core/build && cmake --build core/build -j8 && ctest --test-dir core/build -R test_ml_prerequisites
```
Expected: compile FAILS -- none of the helpers exist.

- [ ] **Step 3: Implement**

Transcribe the five statistics methods from `Numerics/Data/Statistics/Statistics.cs` (lines 143,
218, 253, 590, 736) and `Tools.Standardize` from `Numerics/Utilities/Tools.cs:351`. Rewrite the
three header notes in `statistics.hpp` that currently say `PopulationVariance` /
`PopulationStandardDeviation` / `ParallelMean` / `FiveNumberSummary` are "not ported" -- they now
are, and `parallel_mean` carries the serial-sum note from scope decision 1.

The `dotnet_sort.hpp` move must be VERBATIM: cut lines 99-200 of `data_frame_plotting.hpp`, paste
into the new header under `namespace corehydro::numerics::utilities`, and leave a
`namespace ... = corehydro::numerics::utilities;` alias (or `using` declarations) where the block
was, so every existing call site in that file compiles untouched. Do not reformat, rename, or
"improve" a single line.

`linq_order.hpp` carries the semantics from scope decision 3 in its file header, naming both C#
call sites for each helper.

- [ ] **Step 4: Run the ctest and the plotting-position regression**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R "test_ml_prerequisites|test_data_frame|test_fixtures"
```
Expected: PASS. `test_fixtures` passing is the proof the introsort move was verbatim.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add the machine-learning prerequisites and share the .NET sort port"
```

---

### Task 2: `JenksCluster` and `JenksNaturalBreaks`

**Files:**
- Create: `core/include/corehydro/numerics/machine_learning/support/jenks_cluster.hpp`
- Create: `core/include/corehydro/numerics/machine_learning/unsupervised/jenks_natural_breaks.hpp`
- Create: `core/tests/test_jenks_natural_breaks.cpp` and `core/tests/data/jenks_dataset.hpp`;
  Modify: `core/CMakeLists.txt`

**Interfaces:**
- Produces, in `namespace corehydro::numerics::machine_learning`:
  - `class JenksCluster` -- ctor `(const std::vector<double>& data, int start_index, int end_index)`
    plus `start_index()`, `end_index()`, `count()`, `min_value()`, `max_value()`, `sum()`,
    `average()`, `variance()`, `sum_of_squared_deviations()`.
  - `class JenksNaturalBreaks` -- ctor `(const std::vector<double>& data, int number_of_clusters,
    bool is_data_sorted = false)` (which estimates in the constructor, as C# does), plus
    `sorted_data()`, `number_of_clusters()`, `clusters()`, `breaks()`,
    `goodness_of_variance_fit()`.

Start here rather than with a supervised class: Jenks has no PRNG, no matrix, and no dependency on
anything Task 1 added except nothing at all, so it is the cheapest place to establish the
directory layout and the provenance-header shape for the subsystem.

- [ ] **Step 1: Write the failing ctest**

Put the 7,889-value `_dataset3` literal in `core/tests/data/jenks_dataset.hpp` as
`inline const std::vector<double> kJenksDataset3 = {...};`. Extract it mechanically from
`upstream/Numerics/Test_Numerics/Machine Learning/Unsupervised/Test_JenksNaturalBreaks.cs` (it is
one 75 KB line) with a throwaway script -- do NOT retype it and do not re-format the values.

`core/tests/test_jenks_natural_breaks.cpp` transcribes all three `[TestMethod]`s:

| C# test | k | expected breaks | tol |
|---|---|---|---|
| `Test_Jenks_5Classes` | 5 | `4.141846, 8.523254, 12.64075, 20.13635, 37.00143` | `1e-5` |
| `Test_Jenks_7Classes` | 7 | `2.769867, 6.317200, 8.810181, 11.378660, 15.062380, 22.131710, 37.001430` | `1e-5` |
| `Test_Jenks_9Classes` | 9 | `2.426910, 5.687012, 7.817596, 9.682861, 11.828340, 14.905240, 19.065770, 25.039920, 37.001430` | `1e-5` |

Add a supplement, clearly marked as a corehydro addition beyond the C# assertions: each of the four
constructor guards throws with upstream's exact message (empty data; `number_of_clusters <= 0`;
`number_of_clusters > data.size()`); `goodness_of_variance_fit()` on the 5-class fit is in `(0, 1)`;
`clusters()` partition the sorted data with no gap and no overlap (`clusters[i].end_index() + 1 ==
clusters[i+1].start_index()`, first start 0, last end `n - 1`); and a `JenksCluster` over a
hand-written 6-value run reproduces `Sum`, `Average`, `Variance` and `SumOfSquaredDeviations`
computed by hand -- the Welford recurrence, not the two-pass formula.

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Transcribe both files literally. Two fidelity points the headers must record:

1. `JenksCluster` computes `Variance` from `Count`, a property derived from the indices, in the
   SAME constructor that sets them -- and `Count <= 1 ? 0 : m2 / (Count - 1)` reads `Count` after
   `StartIndex`/`EndIndex` are set, so C++ must set the index members first. It is the
   sample (N-1) variance while `SumOfSquaredDeviations` is the raw `m2`; `GoodnessOfVarianceFit`
   sums the latter.
2. `JenksNaturalBreaks::Estimate` is a 1-BASED dynamic program over `[n+1, k+1]` arrays with a
   `double.MaxValue` sentinel and a `>=` (not `>`) comparison in the inner update, so a tie moves
   the lower class limit to the LATER `i3`. Transcribe the loop bounds and the comparison exactly;
   an off-by-one or a `>` silently changes every break.

- [ ] **Step 4: Run the ctest and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port Jenks natural breaks"
```

---

### Task 3: `KMeans`

**Files:**
- Create: `core/include/corehydro/numerics/machine_learning/unsupervised/k_means.hpp`
- Create: `core/tests/test_k_means.cpp` and `core/tests/data/iris_dataset.hpp`;
  Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1 (`Matrix` column-list ctor and `row`, `tools::distance`, the ported
  `next_integers(rng, min, max, length, replace)` and `MersenneTwister::next`/`next_double`).
- Produces, in `namespace corehydro::numerics::machine_learning`:
  `class KMeans` -- ctors `(const std::vector<double>& x, int k)` and `(const Matrix& x, int k)`,
  members `k()`, `x()`, `dimension()`, `means()`, `labels()`, `max_iterations()` /
  `set_max_iterations()`, `iterations()`, `train(int seed = -1, bool k_means_plus_plus = true)`, and
  the public static `initialize(const Matrix& x, int k, int seed, bool k_means_plus_plus)`.

`core/tests/data/iris_dataset.hpp` holds the four 150-value iris columns AND the 70/30 train/test
split the five supervised test files share (`sepalLengthTrain` ... `speciesTest`), extracted
mechanically from the C# test files. Every later task reads it; nothing re-types it.

- [ ] **Step 1: Write the failing ctest**

`core/tests/test_k_means.cpp` transcribes `Test_KMeans_Iris`: build the 150-by-4 iris matrix from
the four columns, `KMeans kmeans(features, 3); kmeans.train(12345);` then assert the three label
counts exactly (`62`, `38`, `50`) and all twelve cluster means at `1e-6`:

```
mean[0] = {5.901613, 2.748387, 4.393548, 1.433871}
mean[1] = {6.850000, 3.073684, 5.742105, 2.071053}
mean[2] = {5.006000, 3.428000, 1.462000, 0.246000}
```

Supplement (corehydro addition): `iterations()` is at least 2 and at most `max_iterations()`;
`train(12345, false)` (the random-initialization branch, which no C# test reaches) converges and
gives 150 labels all in `[0, 3)`; running `train(12345)` twice gives identical means (the seeded
determinism contract).

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Transcribe `KMeans.cs`. Four fidelity points for the header:

1. The k-means++ CDF loop divides `D[i] /= sum` IN PLACE while building `cdf[i]` and `break`s on
   the first `u <= cdf[i]`; if floating-point drift leaves `u` above the last CDF entry, `idx`
   keeps its PREVIOUS value (the last chosen center's index, or the uniform draw from the first
   step). That is upstream's behavior and it is reachable; do not add a fallback.
2. `Train` compares `oldLabels` to the new labels and breaks BEFORE the M-step, so `Iterations`
   counts E-steps and the returned `Means` are the ones the last E-step assigned against -- i.e.
   the loop exits with means one M-step behind the labels. This is why the C# test's means match
   the label counts.
3. `Iterations` is the loop variable and is left at `MaxIterations + 1` if the loop never converges.
4. `GetCentroids` divides by `count[k] > 0 ? count[k] : 1`, so an empty cluster keeps a zero
   centroid rather than producing NaN.

The `Parallel.For` in `GetLabels` becomes a serial loop (scope decision 1).

- [ ] **Step 4: Run the ctest and watch it pass**

If the means miss at `1e-6`, the divergence is in the PRNG stream, not the arithmetic -- check
`initialize`'s draw order first (`rnd.next(0, rows)` then, per additional center, `rnd.next_double()`).

- [ ] **Step 5: Measure the `parallel_mean` divergence (scope decision 1) and record it**

This is the first task with a real C# comparison surface, so settle the open question now rather
than at fixture time. Write a throwaway emitter probe (or a small `dotnet script` against the local
Numerics build) computing `Statistics.ParallelMean` and `Statistics.Mean` over (a) a 1,000-value
vector, (b) a 100,000-value vector of the same magnitude, and compare each with the serial sum.
Record the measured ULP difference (or exact agreement) in `docs/upstream-csharp-issues.md` as a new
entry, including this machine's `Environment.ProcessorCount`, and state the consequence for the
RandomForest/kNN mean columns. If they agree exactly at the sizes this phase pins, say that too --
a measured "no divergence at these sizes" is a useful result and stops the next reader re-deriving it.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: port k-means clustering"
```

---

### Task 4: `GaussianMixtureModel`

**Files:**
- Create: `core/include/corehydro/numerics/machine_learning/unsupervised/gaussian_mixture_model.hpp`
- Create: `core/tests/test_gaussian_mixture_model.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 3 (`KMeans`), `CholeskyDecomposition` (`log_determinant`, `forward`),
  `MatrixRegularization::make_symmetric_positive_definite`.
- Produces, in `namespace corehydro::numerics::machine_learning`:
  `class GaussianMixtureModel` -- ctors `(const std::vector<double>& x, int k)` and
  `(const Matrix& x, int k)`, members `k()`, `x()`, `dimension()`, `means()`, `sigmas()`,
  `labels()`, `weights()`, `likelihood_matrix()`, `log_likelihood()`, `max_iterations()` /
  `set_max_iterations()`, `tolerance()` / `set_tolerance()`, `iterations()`,
  `train(int seed = -1, bool k_means_plus_plus = true)`.

- [ ] **Step 1: Write the failing ctest**

Transcribe `Test_GMM_Iris` (same 150-by-4 matrix, `k = 3`, `train(12345)`), asserting the three
weights at `1e-2` (`0.3005423`, `0.3661243`, `0.3333333`) and the twelve means at `1e-2`:

```
mean[0] = {5.915044, 2.777451, 4.204002, 1.298935}
mean[1] = {6.546807, 2.949613, 5.482252, 1.985523}
mean[2] = {5.006000, 3.428000, 1.462000, 0.246000}
```

Supplement (corehydro addition, and the part that actually pins the port -- the C# tolerances above
are loose because they compare against R's `mclust`, not against C#): assert `weights()` sums to 1
at `1e-12`; every `sigmas()[k]` is symmetric and has a positive diagonal; `log_likelihood()` is
finite and negative; `labels()` has 150 entries all in `[0, 3)`; and a `k = 1` fit on the
`Test_UnimodalityTest` unimodal sample gives `mean()[0][0]` equal to the sample mean at `1e-8` and
`sigmas()[0](0, 0)` equal to the population variance at `1e-8` (a closed-form identity for a
single-component GMM -- the one exact check available without a C# literal).

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Transcribe `GaussianMixtureModel.cs`. Five fidelity points for the header:

1. `Train`'s convergence test is `Math.Abs((oldLogLH - newLogLH) / oldLogLH) < Tolerance` with
   `oldLogLH` initialized to `double.MinValue` -- NOT `-infinity`. The first iteration therefore
   divides by `-1.7976931348623157e308`, giving a ratio of essentially 1, so the test cannot fire
   on iteration 1. Use `std::numeric_limits<double>::lowest()`, not `-INFINITY`, or the first
   comparison becomes NaN and the loop never breaks.
2. `LogLikelihood` is assigned ONLY inside the convergence branch, so a run that exhausts
   `MaxIterations` leaves it at 0 (the default). Mirror that; do not "fix" it.
3. The E-step breaks the log-sum-exp normalization into `max` + `log(sum)` and OVERWRITES
   `LikelihoodMatrix` with the normalized responsibilities in place; the M-step then reads them.
   `Labels[i]` is set from the argmax of the UNNORMALIZED log values, with `idx` starting at `-1`,
   so an all-NaN row would leave a label of `-1`.
4. Both the initialization and the M-step floor the covariance diagonal at `1e-6 * colVar`, where
   `colVar` is the POPULATION variance of the whole column (recomputed inside the `d` loop each
   time -- an O(K*D*N) redundancy that is not worth "optimizing", since the sum order is what the
   oracle sees).
5. `Sigmas[k] = new Matrix(Dimension)` starts as all zeros, and the initialization only fills it
   when `clusterCount > 1`; a singleton cluster therefore reaches the E-step with only the floored
   diagonal.

- [ ] **Step 4: Run the ctest and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port the Gaussian mixture model"
```

---

### Task 5: Un-gate `HypothesisTests::unimodality_test` and the `hypothesis` group's thirteenth method

**Files:**
- Modify: `core/include/corehydro/numerics/data/hypothesis_tests.hpp` (add `unimodality_test`;
  REPLACE the P4 severance note at lines ~10-16)
- Modify: `core/include/corehydro/numerics/support/toolbox/hypothesis.hpp` (a thirteenth arm and
  the guard list in its header comment)
- Modify: `core/tests/test_hypothesis_tests.cpp` (un-skip `Test_Unimodality`)
- Modify: `fixtures/toolbox/hypothesis.json`, `tools/oracle_emitter/Program.cs`
  (`HypothesisDispatch`)
- Modify: `corehydror/R/toolbox.R`, `corehydropy/src/corehydropy/toolbox.py` (the `method` list and
  both help pages), `corehydror/tests/testthat/test-toolbox.R`, `corehydropy/tests/test_toolbox.py`

**Interfaces:**
- Produces: `double unimodality_test(const std::vector<double>& sample)` in
  `corehydro::numerics::data::hypothesis_tests`, and `hypothesis_test(x, method = "unimodality")`
  in both packages.

- [ ] **Step 1: Write the failing tests**

In `test_hypothesis_tests.cpp`, transcribe `Test_UnimodalityTest` from
`upstream/Numerics/Test_Numerics/Data/Statistics/Test_HypothesisTests.cs:319` -- the 30-value
unimodal sample against `0.4142441` at `1e-4`, and the 30-value bimodal sample against
`2.55425752131444e-05` (read the exact literal and tolerance from the file; do not copy them from
this plan). Add the C# guard check (`n < 10` throws). In the binding tests, assert
`hypothesis_test(unimodal, method = "unimodality")` reproduces the same p-value in both languages.
Expected: FAIL.

- [ ] **Step 2: Implement**

Transcribe `HypothesisTests.cs:317-350`. Three fidelity points:

1. The whole body is wrapped in `try { ... } catch { return double.NaN; }`. Mirror it with
   `catch (...) { return kNaN; }` -- a Cholesky failure inside the GMM must surface as NaN, not as
   an exception crossing into R or Python.
2. Both GMMs are trained with the HARD-CODED seed `12345` and `kMeansPlusPlus = true`. The method
   takes no seed argument; do not add one.
3. `df = 3` is hard-coded regardless of the sample's dimension.

Replace the P4 severance note in `hypothesis_tests.hpp` with a note recording that P5 landed it and
naming `gaussian_mixture_model.hpp` as the dependency. Add the `unimodality` arm to
`toolbox/hypothesis.hpp` and extend that header's guard list with `unimodality  sample.size() >= 10`.

- [ ] **Step 3: Fixture and emitter parity**

Add a `unimodality` case to `fixtures/toolbox/hypothesis.json` (both samples, both C# literals, the
C# tolerances, `source` naming `Test_UnimodalityTest`) and the emitter arm. Run:

```bash
cmake --build core/build -j && ctest --test-dir core/build -R "test_hypothesis_tests|test_fixtures"
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`. **If the p-value does not reproduce against the real C# library, the
GMM port diverged** -- fix Task 4, do not widen the tolerance.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: un-gate the GMM unimodality hypothesis test"
```

---

### Task 6: Un-gate `DataFrame::unimodality_test` and `DataFrame::summary_hypothesis_test`

**Files:**
- Modify: `core/include/corehydro/models/data_frame/data_frame.hpp` (add both members; REPLACE the
  P4 severance note at lines ~53-65 and the in-region note at ~521)
- Modify: `core/include/corehydro/models/data_frame_runner.hpp` (two new methods)
- Modify: `core/tests/test_data_frame_facades.cpp`
- Modify: `fixtures/data/data_frame_facades.json`, `tools/oracle_emitter/Program.cs`
  (`DataFrameDispatch`)
- Modify: `corehydror/R/data.R`, `corehydropy/src/corehydropy/data.py` (the `method` list and both
  help pages), and both packages' data tests
- Modify: `upstream/CLAUDE.md` (the P5-deferral paragraph at ~lines 125-135)

**Interfaces:**
- Consumes: Task 5.
- Produces, public on `corehydro::models::DataFrame`:
  - `double unimodality_test(bool use_log10 = false) const`
  - `std::vector<std::pair<std::string, double>> summary_hypothesis_test(int index = -1, bool
    use_log10 = false) const` -- an ORDERED key/value list, as the twelve P4 facades already
    return, because the C# `Dictionary<string, double>` insertion order is what the fixture selects
    by `label`.
- Produces, user-facing: `analysis_data_hypothesis_test(method = "unimodality")` and
  `method = "summary_hypothesis"` (the latter returning the ten named p-values).

- [ ] **Step 1: Write the failing ctest**

Extend `test_data_frame_facades.cpp` with `Test_UnimodalityTest` from
`src/RMC.BestFit.Tests/DataFrame/ExactDataHypothesisTests.cs` (read it with `git -C
upstream/RMC-BestFit show`; P4 deliberately skipped exactly this method). Add a
`summary_hypothesis_test` case asserting the ten keys in insertion order and that each value equals
the corresponding standalone facade on the same frame -- the point of the method is that it is a
facade over the other ten, so an identity check is the right oracle and needs no new literal.
Expected: FAIL.

- [ ] **Step 2: Implement**

Transcribe `DataFrame.cs:1015-1027` (`UnimodalityTest`) and `1077-1145`
(`SummaryHypothesisTest`). Five fidelity points the header must carry, each an upstream oddity a
"cleanup" would change:

1. The whole ten-call block sits in ONE `try`/`catch` that CLEARS the dictionary and refills all ten
   keys with NaN if any single call throws. Mirror it exactly. (This is precisely why P4 could not
   ship the method with a throwing unimodality arm.)
2. The split index is clamped with `if (index < 0 || index <= minIndex || index > maxIndex) index =
   ExactSeries[(int)((double)values.Length / 2)].Index;` -- the clamp reads `values.Length`, which
   under `useLog10` is the count of POSITIVE values, but indexes `ExactSeries`, which is not
   filtered. Reproduce the mismatch.
3. Under `useLog10` the `indexes`/`values` arrays and the `v1`/`v2` splits filter on
   `x.Value > 0`, but the real-space branch does not. The two branches are not symmetric.
4. `LjungBoxTest(values)` is called with the DEFAULT `lagMax`, unlike the standalone
   `LjungBoxTest` facade, which takes one.
5. The Mann-Whitney call passes `v1.Count <= v2.Count ? v1 : v2` and `v1.Count > v2.Count ? v1 :
   v2`, which is the correct shorter-first ordering; `docs/upstream-csharp-issues.md` already
   records that this reads oddly but is not a bug. Point the header at that entry rather than
   re-litigating it.

The ten dictionary keys, in insertion order, are exact strings: `"Jarque-Bera test for normality"`,
`"Ljung-Box test for independence"`, `"Wald-Wolfowitz test for independence and stationarity
(trend)"`, `"Mann-Whitney test for homogeneity and stationarity (jump)"`, `"Mann-Kendall test for
homogeneity and stationarity (trend)"`, `"Linear trend test for stationarity (trend)"`, `"Equal
variance t-test for differences in the means of two samples"`, `"Unequal variance t-test for
differences in the means of two samples"`, `"F-test for differences in the variances of two
samples"`, `"Mixture model test for unimodality"`. Read them from the source, not from this plan.

- [ ] **Step 3: Fixture, emitter, and the two verbs**

Add `unimodality` and `summary_hypothesis` cases to `fixtures/data/data_frame_facades.json`, the
matching `DataFrameDispatch` arms, and the two method names to both host verbs' documented lists.

```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
cmake --build core/build -j && ctest --test-dir core/build && python3 tools/verify_oracles.py
```
Expected: all green; gate `0 failed`.

- [ ] **Step 4: Retire the severance notes and commit**

Rewrite the `data_frame.hpp` note, the in-region note, and the `upstream/CLAUDE.md` paragraph to
record that P5 landed both members. `models/data_frame/` now has NO deferred facade.

```bash
git add -A && git commit -m "feat: un-gate the DataFrame unimodality and summary hypothesis tests"
```

---

### Task 7: `DecisionNode` and `DecisionTree`

**Files:**
- Create: `core/include/corehydro/numerics/machine_learning/support/decision_node.hpp`
- Create: `core/include/corehydro/numerics/machine_learning/supervised/decision_tree.hpp`
- Create: `core/tests/test_decision_tree.cpp` and `core/tests/data/fpp3_dataset.hpp`;
  Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1 (`linq_order`, `Matrix` ctors/`row`/`column`, `statistics::population_variance`,
  `statistics::entropy`), the ported `KernelDensity` distribution, `next_integers(rng, 0, dim,
  features, false)`.
- Produces, in `namespace corehydro::numerics::machine_learning`:
  - `struct DecisionNode { int feature_index = -1; double threshold = kNaN; std::shared_ptr<DecisionNode>
    left, right; double value = kNaN; bool is_leaf_node = false; };` (C# uses a nullable reference;
    `shared_ptr` is the faithful and safe C++ spelling -- say so in the header).
  - `class DecisionTree` -- ctors `(const std::vector<double>& x, const std::vector<double>& y, int
    seed = -1)` and `(const Matrix& x, const Vector& y, int seed = -1)`, members
    `minimum_split_size()` / setter, `max_depth()` / setter, `dimensions()`, `features()` / setter,
    `random()`, `root()`, `y()`, `x()`, `is_regression()` / setter, `is_trained()`, `train()`, and
    `std::optional<std::vector<double>> predict(const Matrix& x) const`.

`core/tests/data/fpp3_dataset.hpp` holds the five 198-value `fpp3` series
(`consumption`/`income`/`production`/`savings`/`unemployment`) the DecisionTree, RandomForest, kNN
and GLM regression tests share, extracted mechanically from the C# test files.

- [ ] **Step 1: Write the failing ctest**

Transcribe both `[TestMethod]`s:

- `Test_DecisionTree_Iris`: the 90-row training split, `DecisionTree tree(x, y, 12345)` with
  `set_is_regression(false)` and `set_features(4)`, `train()`, predict the 60-row test split, and
  assert `GoodnessOfFit::accuracy(y_test, prediction) >= 90`.
- `Test_DecisionTree_Regression`: the fpp3 series, `tIdx = 118`, train on rows `[0, 118]` and test
  on `[119, end]` (note `Subset(0, tIdx)` is INCLUSIVE of `tIdx` and `Subset(tIdx + 1)` runs to the
  end -- check the ported `subset` helper's convention against `ExtensionMethods.cs` before
  transcribing, and if it is not ported, slice explicitly and say so), then assert the tree's
  R-squared is LESS than the linear-regression R-squared, as C# does.

Supplement (corehydro addition -- both C# assertions are inequalities and would pass on a badly
broken port): `predict()` on a matrix whose column count differs from `dimensions()` returns
`std::nullopt`; `predict()` before `train()` returns `std::nullopt`; the ctor guards throw on
mismatched lengths and on fewer than ten points; a tree trained twice with seed 12345 gives
bit-identical predictions; and a hand-built 12-point 1-D regression problem with a clean split at
`x = 5` produces a root node whose `threshold` and two leaf `value`s are the hand-computed means.

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Transcribe `DecisionTree.cs`. Six fidelity points for the header:

1. `GrowTree` calls `BestSplit` BEFORE testing the stopping criteria, so the PRNG advances by one
   `next_integers(0, Dimensions, Features, false)` draw at EVERY node including leaves. Reordering
   for efficiency changes the stream and every seeded oracle.
2. `numberOfLabels` is `yTrain.Length` for regression (so the `<= 1` stopping test only fires on a
   single-row node) and the distinct count for classification.
3. `BestSplit` seeds `best` with `double.MinValue` and uses a strict `>`; `VarianceReduction` and
   `InformationGain` also RETURN `double.MinValue` for a degenerate split. So when every candidate
   is degenerate, `bestFeatureIndex` stays `-1` and the node becomes a leaf -- but when the FIRST
   candidate is non-degenerate and equals a later one, the earlier wins. Use
   `std::numeric_limits<double>::lowest()`, not `-INFINITY`.
4. Regression thresholds are the raw column (duplicates included); classification thresholds are
   `x.Distinct()` in first-appearance order (Task 1's helper). The regression path therefore
   evaluates duplicate thresholds repeatedly, which is redundant but stream-neutral -- keep it.
5. The classification leaf value is the first-appearance-wins mode (Task 1's helper), and the
   classification `Entropy` uses an O(n) relative-frequency lambda inside `Statistics.Entropy`,
   making it O(n^2) per threshold. The regression `Entropy` builds a `KernelDensity` over `y` --
   dead code in practice, since regression never calls `Entropy`, but port it (it is reachable if
   a caller flips `IsRegression` between `Train` calls).
6. `TraverseTree` falls back to `node.Value` (NaN for an internal node) when a child is null, rather
   than throwing.

- [ ] **Step 4: Run the ctest and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port the decision tree learner"
```

---

### Task 8: `RandomForest`

**Files:**
- Create: `core/include/corehydro/numerics/machine_learning/supervised/random_forest.hpp`
- Create: `core/tests/test_random_forest.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 7, `statistics::percentile(values, p, true)`, `statistics::parallel_mean`.
- Produces `class RandomForest` in the same namespace: the same three ctors, plus
  `number_of_trees()` / setter, `minimum_split_size()`, `max_depth()`, `dimensions()`,
  `features()`, `random()`, `y()`, `x()`, `decision_trees()`, `is_regression()` / setter,
  `is_trained()`, `train()`, and
  `std::optional<Matrix> predict(const Matrix& x, double alpha = 0.1) const` returning the
  n-by-4 lower/median/upper/mean matrix.

- [ ] **Step 1: Measure the serial cost, THEN write the ctest**

Before transcribing, time a single classification `DecisionTree::train()` on the 90-row iris split
(Task 7's ctest already builds it). Multiply by 1000. If the projected `Test_RandomForest_Iris` wall
time exceeds 60 s, apply scope decision 6: lower `NumberOfTrees` in the ctest ONLY, and record both
the measured single-tree time and the chosen tree count in a numbered note at the top of the test
file. Then transcribe both `[TestMethod]`s (`Test_RandomForest_Iris`: accuracy of
`prediction.column(1)` -- the MEDIAN column, not the mean -- at least 90; `Test_RandomForest_Regression`:
R-squared of `rfPredict.column(3)` GREATER than the linear-regression R-squared).

Supplement (corehydro addition): the four columns are ordered and finite
(`lower <= median <= upper` row-wise); `predict()` before `train()` returns `std::nullopt`; two
seeded runs give bit-identical output; and for classification every one of the four columns is an
integer (the `Math.Floor` branch).

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Transcribe `RandomForest.cs`. Four fidelity points:

1. `Train` draws ALL `NumberOfTrees` seeds up front with `Random.NextIntegers(NumberOfTrees)`, then
   builds each tree from its own `MersenneTwister(seed)`. That is what makes the parallel loop
   reproducible and what lets the port run serially with identical results (scope decision 1).
2. `BootstrapDecisionTree` passes the SAME `seed` both to its local resampling generator and to the
   `DecisionTree` constructor, so the tree's feature draws start from a generator in its initial
   state while the resample already consumed `X.NumberOfRows` draws from a separate instance. Two
   generators, one seed. Do not share one.
3. `Predict`'s classification branch applies `Math.Floor` to each percentile AND to the mean.
4. `Predict` computes percentiles with `Statistics.Percentile(values, p, true)` on an
   already-sorted row and the mean with `Statistics.ParallelMean` -- see scope decision 1 and the
   Task 3 Step 5 measurement.

- [ ] **Step 4: Run the ctest and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port the random forest learner"
```

---

### Task 9: `KNearestNeighbors`

**Files:**
- Create: `core/include/corehydro/numerics/machine_learning/supervised/k_nearest_neighbors.hpp`
- Create: `core/tests/test_k_nearest_neighbors.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1 (`dotnet_list_sort`, `linq_order`, `tools::standardize` for the ctest only).
- Produces `class KNearestNeighbors`: the three ctors `(x, y, k)`, members `k()`, `y()`, `x()`,
  `number_of_features()`, `is_regression()` / setter, and the four public verbs
  `get_neighbors(const Matrix&)`, `predict(const Matrix&)`,
  `bootstrap_predict(const Matrix&, int seed = -1)`,
  `prediction_intervals(const Matrix&, int seed = -1, int realizations = 1000, double alpha = 0.1)`.
  The first three return `std::optional`; `prediction_intervals` returns a `Matrix`.

- [ ] **Step 1: Write the failing ctest**

Transcribe all four `[TestMethod]`s:

- `Test_kNN_Iris`: `k = 5`, classification, and assert the FULL 60-value prediction vector against
  the C# literal (a real oracle, unlike DecisionTree's).
- `Test_kNN_Classification`: the 17-point 2-D example, `k = 3`, `predict({{2.5, 7}})[0] == 0`.
- `Test_kNN_Regression`: the standardized fpp3 predictors, `k = 5`, R-squared greater than the
  linear-regression R-squared.
- `Test_GetNeighbors_MultiRow`: the two symmetric clusters, `k = 2`, the two queries, and the four
  index-range assertions. **This is the tie-permutation test** (scope decision 2) -- add a
  corehydro supplement asserting the EXACT four indices this port returns, and cross-check them
  against the real library in Task 12's fixture rather than inventing them here.

Supplement: `get_neighbors`/`predict` on a matrix with the wrong column count returns
`std::nullopt`; a zero distance takes the `w = 1` branch of the inverse-distance weighting (query a
training row exactly and assert the prediction is that row's `y` when `k = 1`); both ctor guards
throw.

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Transcribe `KNearestNeighbors.cs`. Five fidelity points:

1. `kNN` (the neighbor-index helper) guards on `NumberOfFeatures != xTrain.NumberOfColumns`, which
   is always false since `NumberOfFeatures` IS `X.NumberOfColumns` -- it never rejects a mismatched
   TEST matrix, unlike `kNNPredict`, which checks `xTest.NumberOfColumns != xTrain.NumberOfColumns`
   correctly. Mirror both, and note that `get_neighbors` therefore reads past the end of a
   too-narrow query row in C#; the port must keep the guard as written but is free to let the
   underlying `row()` accessor throw rather than reading out of bounds. Record this as an upstream
   defect in `docs/upstream-csharp-issues.md`.
2. The distance sort is `dotnet_list_sort` with `a.Distance.CompareTo(b.Distance)` -- `CompareTo`
   for `double` orders NaN BELOW every number and treats `-0.0` and `0.0` as equal, which is not
   `operator<`. Use the existing `dn_compare_double` helper (the same one
   `data_frame_plotting.hpp` uses), not a raw `<`.
3. The regression prediction is an inverse-SQUARED-distance weighted average with `w = 1` when the
   distance is exactly 0, and it accumulates `knn[j] = y * w` then divides each term by `sum`
   separately (`avg += knn[j] / sum`) rather than dividing once at the end. The division order is
   oracle-visible; keep it.
4. `kNNBootstrapPredict` resamples the TRAINING set with replacement and predicts from the
   resampled set; `kNNPredictionIntervals` draws `realizations` seeds up front, exactly as
   RandomForest does.
5. `PredictionIntervals` has NO classification branch -- it always takes percentiles and the mean,
   even when `IsRegression` is false. Mirror it.

- [ ] **Step 4: Run the ctest and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port k-nearest neighbors"
```

---

### Task 10: `NaiveBayes`

**Files:**
- Create: `core/include/corehydro/numerics/machine_learning/supervised/naive_bayes.hpp`
- Create: `core/tests/test_naive_bayes.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1 (`linq_order::distinct_in_first_appearance_order`), the ported `Normal`
  distribution's `log_pdf`.
- Produces `class NaiveBayes`: the three ctors, members `y()`, `x()`, `classes()`, `means()`,
  `standard_deviations()`, `priors()`, `is_trained()`, `train()`, and
  `std::optional<std::vector<double>> predict(const Matrix&) const`.

- [ ] **Step 1: Write the failing ctest**

Transcribe `Test_NaiveBayes_Iris` in full -- it is the richest oracle in the subsystem: twelve
means at `1e-6`, twelve standard deviations at `1e-6`, three priors at `1e-6`, and the exact
60-value prediction vector (which contains two deliberate misclassifications at positions 42 and
53). Read every literal from the C# file.

Supplement: the ctor's three guards throw; `predict` before `train` returns `std::nullopt`; a class
with a single member gets the `1e-6` standard-deviation floor.

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Transcribe `NaiveBayes.cs`. Three fidelity points:

1. `Classes` is `y.Distinct()` in FIRST-APPEARANCE order, so `Means[i]` indexes classes in the
   order they appear in the training response, not in sorted order. The iris test happens to be
   sorted, which hides it; a shuffled response would not.
2. The variance is the naive `(u2 - u1^2) * n/(n-1)` two-moment form clamped at 0 by
   `Math.Max(0, ...)`, NOT `Statistics.Variance`. It is catastrophically cancelling for
   large-magnitude features, and it is what the oracle sees. Do not substitute the stable
   recurrence.
3. `MAP` seeds `max` with `double.MinValue` and uses a strict `>`, so ties go to the FIRST class in
   `Classes` order; and it constructs a fresh `Normal(mean, sd)` per feature per class per point
   rather than caching. Keep the construction (it validates parameters, and a zero sd would throw).

- [ ] **Step 4: Run the ctest and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port the Gaussian naive Bayes classifier"
```

---

### Task 11: `GeneralizedLinearModel`

**Files:**
- Create: `core/include/corehydro/numerics/machine_learning/supervised/generalized_linear_model.hpp`
- Create: `core/tests/test_generalized_linear_model.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: the ported link layer (`ILinkFunction`, `LinkFunctionFactory`, `LinkFunctionType`), the
  five ported local optimizers plus `LocalMethod`, `GoodnessOfFit::aic`/`aicc`/`bic`,
  `factorial::log_factorial`, `Normal::standard_cdf`/`standard_z`, `Matrix::inverse`/`transpose`/
  `multiply`/`diagonal`, and Task 1's `five_number_summary` (for the severed `Summary()` only --
  which means it is NOT needed here; it is needed by the P4-era statistics surface and was added
  for completeness).
- Produces `class GeneralizedLinearModel`: the two ctors (enum-typed and custom-`ILinkFunction`),
  members `has_intercept()`, `y()`, `x()`, `parameters()`, `parameter_names()`,
  `parameter_standard_errors()`, `parameter_z_scores()`, `parameter_p_values()`, `covariance()`,
  `residuals()`, `standard_error()`, `sample_size()`, `degrees_of_freedom()`, `aic()`, `aicc()`,
  `bic()`, `use_robust_se()` / setter, `optimizer()`, `link_type()`, `link_function()`,
  `set_optimizer(LocalMethod method = LocalMethod::NelderMead)`, `train()`, `predict(const Matrix&)`,
  and `predict_intervals(const Matrix&, double alpha = 0.1)`.

- [ ] **Step 1: Write the failing ctest**

Transcribe all six `[TestMethod]`s. Every expected value is R's `glm` output, so these are strong
oracles:

| C# test | link | asserted | tol |
|---|---|---|---|
| `Test_SimpleLinearRegression` | Identity | `params {0.54510, 0.28060}`, `se {0.05569, 0.04744}`, `sigma 0.6026`, `df 185` | `1e-3` |
| `Test_MultipleLinearRegression` | Identity | `params {0.26729, 0.71449, 0.04589, -0.04527, -0.20477}`, `se {0.03721, 0.04219, 0.02588, 0.00278, 0.10550}`, `sigma 0.3286`, `df 182` | `1e-3` |
| `Test_Log` | Log | `params {6.322, 2.405e-3, -1.480e-4}`, `se {9.382e-3, 1.885e-5, 8.163e-6}`, `aic 4069.4`, `df 23` | `1e-2` |
| `Test_Logistic` | Logit | `params {-3.449548, 0.002294, 0.777014, -0.560031}`, `aic 467.44`, `df 396` | `1e-2` (`se` at `1e-1`) |
| `Test_Probit` | Probit | `params {-2.0915037, 0.0013982, 0.4643598, -0.3317117}`, `aic 467.48`, `df 396` | `1e-2` |
| `Test_LogLog` | ComplementaryLogLog | `params {-3.0603413, 0.0017513, 0.6213458, -0.4587797}`, `aic 467.5`, `df 396` | `1e-2` |

Read every literal and tolerance from the C# file; note that `Test_Logistic` loosens the standard
errors to `1e-1` while the parameters stay at `1e-2`, and reproduce that split rather than
flattening it. The `gre`/`gpa`/`rank`/`admits` arrays for the last three go in
`core/tests/data/admissions_dataset.hpp`, extracted mechanically.

Supplement: `parameter_names()` is `{"Intercept", "β1", ...}` when no header is supplied;
`predict_intervals` gives `lower <= mean <= upper` row-wise; `PrepareDesignMatrix` accepts both a
`p`-column and a `p-1`-column matrix when there is an intercept and throws with the C# message
otherwise; `set_optimizer(LocalMethod::BFGS)` followed by `train()` reaches the same identity-link
answer to `1e-3` (a cross-optimizer check the C# suite never runs).

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Transcribe `GeneralizedLinearModel.cs` EXCEPT `Summary()` (scope decision 4). Six fidelity points:

1. The objective returns `double.MaxValue` (not infinity) for a non-finite log-likelihood, and it
   returns the NEGATED log-likelihood because the optimizers minimize.
2. The identity-family per-observation term is `-0.5 * resid^2` -- no `sigma`, no constant. The
   constant is added back only in `ComputeDiagnostics`, which recomputes the log-likelihood a
   SECOND time with a different formula per family. The two do not agree, and only the second one
   feeds AIC/AICc/BIC.
3. The parameter bounds are family-specific and asymmetric (`Identity`/`Log`: `+/- slope * 100`
   with the intercept bounded by `min(init/100, init*100)` and `max(...)` -- which for a NEGATIVE
   intercept gives `[init*100, init/100]`, a correctly-ordered but very wide box; `Logit`/`CLogLog`:
   `+/- 10`; `Probit`: `+/- 6`). Transcribe the expressions, not their intent.
4. `ParameterStandardErrors` multiplies by `StandardError` for EVERY link except `Log`. The
   asymmetry is upstream's and it is what makes the `Test_Log` standard errors reproduce.
5. `Covariance` is `(J'J)^-1` from the delta-method Jacobian, NOT the observed information; the
   robust form is the sandwich `(J'J)^-1 J' diag(resid^2) J (J'J)^-1`.
6. `ComputeDiagnostics`'s Poisson branch calls `Factorial.LogFactorial((int)Math.Round(y))`, which
   silently rounds a non-integer response and throws for a negative one.

Record the `Summary()` severance in the file header and in `upstream/CLAUDE.md`'s
"What is deliberately not ported" list, beside the Bulletin17C GMM report text.

- [ ] **Step 4: Run the ctest and watch it pass**

If a parameter misses at the C# tolerance, the divergence is almost certainly in the optimizer
bounds or the initial value, not in the likelihood -- print `optimizer().best_parameter_set()` and
compare against a C# run before touching anything else.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port the generalized linear model"
```

---

### Task 12: The `ml` toolbox group, the fixtures, and the emitter driver

**Files:**
- Create: `core/include/corehydro/numerics/support/toolbox/ml.hpp`
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (include + dispatch line +
  the file-header group count: seventeen groups become eighteen, and the list gains `ml`)
- Create: `fixtures/ml/machine_learning.json`; Modify: `fixtures/README.md`
- Modify: `tools/oracle_emitter/Program.cs` (`ToolboxDispatch` `case "ml"` + `MLDispatch`)

**Interfaces:**
- Produces, toolbox group `ml`, in `corehydro::numerics::support::detail`:
  `inline ToolboxResult run_ml(const std::string& method, const std::vector<std::vector<double>>&
  data, const JsonValue& options)`.

  **Data layout, identical to the `regression` group's and documented once in the header:**
  `data[0]` is the training predictor matrix flattened ROW-MAJOR, `data[1]` is the response (empty
  for the three unsupervised methods), `data[2]` is the optional new-data matrix flattened
  row-major. `options` carries `rows`, `columns`, `predict_rows`, and every scalar knob. The
  binding layer transposes on the way in; this is the one place the layout is assumed.

  Methods, grouped by class:
  - `kmeans_means` (dims `{k, p}`), `kmeans_labels` (n), `kmeans_iterations` (scalar).
    Options: `k`, `seed`, `kmeans_plus_plus` (default `true`), `max_iterations` (default 1000).
  - `gmm_means` (dims `{k, p}`), `gmm_weights` (k), `gmm_labels` (n),
    `gmm_sigmas` (dims `{k * p, p}`), `gmm_log_likelihood`, `gmm_iterations`.
    Options: the KMeans set plus `tolerance` (default 1e-8).
  - `jenks_breaks` (k), `jenks_gvf` (scalar), `jenks_clusters` (dims `{k, 8}`, columns
    `start_index, end_index, count, min, max, sum, average, variance` -- `names` labels them).
    Options: `n_clusters`, `is_data_sorted` (default `false`). Reads `data[0]` as a plain vector,
    not a matrix.
  - `decision_tree_predict` (n_test). Options: `seed`, `is_regression` (default `true`),
    `features`, `minimum_split_size`, `max_depth`.
  - `random_forest_predict` (dims `{n_test, 4}`, `names = {"lower", "median", "upper", "mean"}`).
    Options: the DecisionTree set plus `number_of_trees` (default 1000) and `alpha` (default 0.1).
  - `knn_predict` (n_test), `knn_neighbors` (dims `{n_test, k}`),
    `knn_bootstrap_predict` (n_test), `knn_prediction_intervals` (dims `{n_test, 4}`, same names).
    Options: `k`, `is_regression`, `seed`, `realizations`, `alpha`.
  - `naive_bayes_means` (dims `{c, p}`), `naive_bayes_sds` (dims `{c, p}`),
    `naive_bayes_priors` (c), `naive_bayes_classes` (c), `naive_bayes_predict` (n_test).
  - `glm_fit` (a NAMED flat result mirroring `regression`'s `fit`: `beta_1..p`, `se_1..p`,
    `z_1..p`, `p_1..p`, then `sigma`, `df`, `n`, `aic`, `aicc`, `bic`), `glm_covariance`
    (dims `{p, p}`), `glm_residuals` (n), `glm_predict` (n_test),
    `glm_predict_intervals` (dims `{n_test, 3}`, `names = {"lower", "mean", "upper"}`).
    Options: `intercept` (default `true`), `link` (one of the seven `LinkFunctionType` names,
    default `"identity"`), `local_method` (default `"nelder_mead"`), `robust_se` (default `false`),
    `alpha`.

- [ ] **Step 1: Write the fixture first**

`fixtures/ml/machine_learning.json` (kind `toolbox`, group `ml`). Cases, with the iris columns and
the fpp3 series as `datasets` entries so each is written once:

1. `kmeans_iris` -- the three C#-test label counts (as `select`ed values off `kmeans_labels`) and
   the twelve means at `1e-6`, `source` naming `Test_KMeans_Iris`.
2. `gmm_iris` -- the three weights and twelve means at the C# tolerances, plus `gmm_log_likelihood`
   and `gmm_iterations` curated with `--dump` (say so in those assertions' `source`).
3. `jenks_small` -- a 30-value curated dataset (scope decision 5), breaks and GVF from `--dump`.
4. `naive_bayes_iris` -- the twelve means, twelve sds, three priors and the 60 predictions, all C#
   literals.
5. `knn_iris` -- the 60 classification predictions (C# literals) and the
   `Test_GetNeighbors_MultiRow` neighbor indices (curated -- this is the tie-permutation proof of
   scope decision 2, and the emitter running the REAL `Array.Sort` is what makes it an oracle).
6. `knn_intervals` -- a seeded `knn_prediction_intervals` on a small regression problem with
   `realizations = 50`, pinning all four columns. **The mean column is the `parallel_mean` surface
   from scope decision 1** -- pin it at the tolerance Task 3 Step 5 measured, with the measurement
   in the `source` string.
7. `decision_tree_fpp3` and `random_forest_fpp3` -- a small seeded regression case
   (`number_of_trees` kept small enough that the emitter and the three runners finish quickly; say
   the chosen count in the case name) with the predictions curated via `--dump`.
8. `glm_identity`, `glm_log`, `glm_logit`, `glm_probit`, `glm_cloglog` -- the C# test literals at
   the C# tolerances, plus a `glm_covariance` and a `glm_predict_intervals` case curated via
   `--dump`.

- [ ] **Step 2: Implement the group and the dispatch line**

`toolbox/ml.hpp` follows `toolbox/regression.hpp`'s shape: a `build_ml_matrix` helper reading
`data[0]` with `rows`/`columns`, one arm per method, `throw std::runtime_error("unknown ml method: "
+ method)` at the end, and a file-header block comment naming each method's C# counterpart, its
options and its result shape. Wire `if (group == "ml") return detail::run_ml(method, data,
options);` into `toolbox_runner.hpp` and update its header's group count and list.

- [ ] **Step 3: Emitter driver and the gate**

`ToolboxDispatch` gains `case "ml": return MLDispatch(...)`, and `MLDispatch` drives the real
`Numerics.MachineLearning` classes, reading the same options keys.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`. Any value that will not reproduce is OMITTED from the fixture with the
reason in the case comment -- never masked with `oracle_skip`, never given a loosened tolerance.

- [ ] **Step 4: Fixture README and commit**

Append an `### ml` section to `fixtures/README.md` documenting the data layout and every method.

```bash
git add -A && git commit -m "feat: add the machine-learning toolbox group and its oracles"
```

---

### Task 13: The eight `ml_*` verbs in R and Python

**Files:**
- Create: `corehydror/R/ml.R`, `corehydropy/src/corehydropy/ml.py`
- Modify: `corehydror/NAMESPACE`, `corehydror/_pkgdown.yml`, `site/_quarto.yml`,
  `corehydropy/src/corehydropy/__init__.py`
- Create: `corehydror/tests/testthat/test-ml.R`, `corehydropy/tests/test_ml.py`
- Modify: `core/tests/test_fixtures.cpp` / `corehydror/tests/testthat/test-fixtures.R` /
  `corehydropy/tests/test_fixtures.py` only if the new group needs a dispatch entry (it should
  not -- the `toolbox` kind is group-generic; confirm by running the suites before editing them)

**Interfaces:** eight verbs, argument-for-argument identical in both languages:

- `ml_kmeans(x, k, seed = NULL, kmeans_plus_plus = TRUE, max_iterations = 1000)` ->
  `list(means, labels, iterations)`; `labels` are 0-BASED in BOTH languages (the same choice
  `shortest_path()` made for node indices in P3 -- say so in both help pages).
- `ml_gaussian_mixture(x, k, seed = NULL, kmeans_plus_plus = TRUE, max_iterations = 1000,
  tolerance = 1e-8)` -> `list(means, sigmas, weights, labels, log_likelihood, iterations)`, where
  `sigmas` is a length-k list of p-by-p matrices.
- `ml_jenks_breaks(x, n_clusters, is_data_sorted = FALSE)` -> `list(breaks, clusters, gvf)`.
- `ml_decision_tree(x, y, newdata, seed = NULL, regression = TRUE, features = NULL,
  minimum_split_size = 2, max_depth = 100)` -> a numeric vector of predictions.
- `ml_random_forest(..., number_of_trees = 1000, alpha = 0.1)` -> an n-by-4 matrix with
  `lower`/`median`/`upper`/`mean` columns. Both help pages state the training cost and that
  `number_of_trees` is the knob.
- `ml_knn(x, y, newdata, k, regression = TRUE, what = c("prediction", "neighbors",
  "prediction_intervals"), seed = NULL, realizations = 1000, alpha = 0.1)`.
- `ml_naive_bayes(x, y, newdata = NULL)` -> `list(classes, means, standard_deviations, priors,
  prediction)`, with `prediction` absent when `newdata` is `NULL`.
- `ml_glm(x, y, intercept = TRUE, link = "identity", local_method = "nelder_mead",
  robust_se = FALSE)` -> a list mirroring `linear_regression()`'s shape (`coefficients`,
  `standard_errors`, `z_values`, `p_values`, `sigma`, `df`, `n`, `aic`, `aicc`, `bic`, `vcov`,
  `residuals`) plus `predict`/`confint` helpers if `linear_regression()` has them -- read
  `corehydror/R/regression.R` first and match it rather than inventing a second shape.

- [ ] **Step 1: Write failing R and Python binding tests**

In `test-ml.R` and `test_ml.py`, mirror each other case for case: the iris k-means means and label
counts; the naive-Bayes iris means and the 60 predictions; a seeded `ml_random_forest` with a small
`number_of_trees` returning a 4-column matrix with ordered rows; `ml_glm` on the fpp3 simple
regression reproducing `0.54510` / `0.28060`; `ml_jenks_breaks` on a small vector; and the argument
validation (a non-numeric `x`; `y` of the wrong length; `k` above `nrow(x)`; an unknown `link`; an
unknown `what`) with identical message text in both languages.
Expected: FAIL (functions not found).

- [ ] **Step 2: Implement both verb sets**

Full roxygen / numpydoc on every verb, each with a runnable example inside CRAN's time limits (use
small `number_of_trees` and `realizations` in examples). Every help page states the 0-based label
convention where it applies and, for `ml_knn` and `ml_random_forest`, that a seeded run is
bit-identical between R and Python because the whole computation lives in the shared core.

```bash
R CMD INSTALL --preclean corehydror && Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```

- [ ] **Step 3: Documentation indexes and commit**

All eight into `NAMESPACE`, a new `Machine learning` group in `_pkgdown.yml` and a matching
`quartodoc.sections` entry in `site/_quarto.yml`, plus Python's two `__all__`s and the
`__init__.py` import block. Regenerate the eight `.Rd` files.

```bash
Rscript -e 'roxygen2::roxygenise("corehydror", roclets = "rd")'
git add -A && git commit -m "feat: expose the machine-learning layer in R and Python"
```

---

### Task 14: The cross-language digest fixture

**Files:**
- Create: `fixtures/ml/ml_cross_language.json`; Modify: `fixtures/README.md`
- Modify: the three fixture runners only if the new kind needs a dispatch entry (model it on
  `fixtures/toolbox/toolbox_cross_language.json`, which nests `optimizer`- and `toolbox`-kind cases
  under one case name -- if that kind's machinery already covers `ml`, reuse it verbatim)

**Interfaces:** one `toolbox_cross_language`-kind file nesting three seeded `ml`-group cases under
one case name at ZERO tolerance: a seeded `kmeans_means` on the iris matrix, a seeded
`random_forest_predict` with a small tree count, and a seeded `knn_prediction_intervals`. The
promise being proven is the one the estimation, toolbox and callback layers each proved for their
own surface: every operation lives in the shared core, so a seeded run is bit-identical across C++,
R and Python.

- [ ] **Step 1: Write it, run all three runners, and confirm zero tolerance holds**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
R CMD INSTALL corehydror && Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy && pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
```

If any value fails at zero tolerance between R and Python, that is a real bug in the shared path --
find it; do not loosen the tolerance. If a value fails only against C#, it is the `parallel_mean`
surface or an optimizer ULP drift: record the measurement, drop that assertion from the C# side per
the standing rule, and say so in the case `source`.

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "test: prove the machine-learning layer reproduces across languages"
```

---

### Task 15: Worked example 21, documentation, and the version bump

**Files:**
- Create: `site/examples/21-machine-learning/{python.ipynb, r.qmd}`
- Modify: `site/_quarto.yml` (the example listing), `site/status.qmd`, `site/index.qmd`,
  `README.md`, `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`,
  `corehydropy/src/corehydropy/__init__.py` (`__version__`), `NEWS.md`, `CHANGELOG.md`,
  `.claude/CLAUDE.md` (the Layout and Status sections), `upstream/CLAUDE.md`
- Modify: `docs/upstream-csharp-issues.md` (the entries this phase found)

- [ ] **Step 1: Write the example pair**

Following the established shape: a Python notebook committed WITH outputs and an R Quarto twin with
`freeze: auto`, both ending in an executable reproduction check comparing an R-computed value with
the Python one. Cover clustering (k-means and the GMM on iris), classification (naive Bayes and
kNN), regression (random forest against a linear model), the natural-breaks classifier, and the GLM
with a non-identity link. State the honest limit in prose, not a footnote: a seeded ML run is
bit-identical R-to-Python because the whole computation is in the shared core, and where the C#
comparison is looser, say exactly why (the `parallel_mean` measurement from Task 3 Step 5).

- [ ] **Step 2: Version bump to 0.12.0 and the prose sweep**

Bump both packages. Bring `site/status.qmd`, `site/index.qmd` and `README.md` to the new surface
(the Machine Learning row moves from unported to shipped). Add a Status paragraph and a Layout
paragraph to `.claude/CLAUDE.md` in the voice of the existing ones. Retire the ML entries in
`upstream/CLAUDE.md`'s port-history notes.

- [ ] **Step 3: Full green run**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
pixi run docs
R CMD check --as-cran corehydror_0.12.0.tar.gz
```
Expected: ctest all-pass; gate `0 failed`; `R CMD check` at the SAME three NOTEs as P4 with no
WARNING. **Run `R CMD check --as-cran` here, not at ship time** -- the P4 lesson.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs: worked example 21 and the v0.12.0 release notes"
```

---

### Task 16: Whole-branch review and ship

- [ ] **Step 1: Whole-branch review**

Read the complete diff against `main` in one pass. The classes of finding prior phases turned up
that a task-by-task review cannot: a default that disagrees between R and Python; an error message
that differs textually; a new export missing from one of the two documentation indexes; prose in a
help page that describes an argument the function does not have; a fixture assertion that pins
nothing (a value that would pass if the feature were ignored entirely); a supplement test that
asserts an inequality where an equality was available.

- [ ] **Step 2: Fix, re-run the full suite, and push**

```bash
git push -u origin port-machine-learning
gh pr create --base main --title "P5: machine learning" --body "..."
gh run watch <id> --exit-status
```

Report the final numbers (ctest, oracle gate, testthat, pytest, `R CMD check`) measured on the
final commit, not copied from any earlier step.
