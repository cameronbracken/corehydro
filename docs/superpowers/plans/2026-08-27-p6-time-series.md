# P6 TimeSeries Container Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Work task-by-task;
> each task ends in a green build and a commit.

**Goal:** Port the heavy Numerics `TimeSeries` container (2,334 lines) and its `Series` base, then
un-gate everything that has been waiting on it -- the seasonal `PointProcessModel` data path,
`DataFrame.CreateBlockSeries` / `CreatePeaksOverThresholdSeries`, `PointProcessModel.
GeneratePOTTimeSeries`, the ARIMAX covariate forecast-tail extension, and the `Autocorrelation`
TimeSeries overloads. Expose the container in R and Python as a `time_series()` / `TimeSeries`
object with verbs over it, validated by ctest transcriptions of the 58 upstream `[TestMethod]`s,
pinned by fixtures through all four runners with dotnet-gate reproduction, documented with a worked
example pair, and shipped as branch `port-time-series` at v0.13.0. This is the last of the six port
phases in `docs/superpowers/specs/2026-08-20-remaining-port-and-v1-release-design.md`; when it
lands, every portable file in Numerics and RMC.BestFit is ported.

**Architecture:** One new value type, one new dispatch group, one new host object.

- `DateTime` (`numerics/data/time_series/support/date_time.hpp`) is a **corehydro addition with no
  upstream file**: a ticks-based mirror of the `System.DateTime` subset this container uses. It has
  to exist because the container is calendar-driven (`AddMonths`, water years, `Month`/`Year`/
  `DayOfYear` grouping) and the core takes no external dependency. Everything ported here reads
  dates through it.
- The container itself replaces the P2 thin adapter in place, on a real `Series<TIndex, TValue>`
  base ported beside it. The adapter's `IndexType` goes from `long` to `DateTime`; the existing
  fixture suite staying green is the proof that swap is behaviour-neutral.
- `timeseries` is the nineteenth toolbox group under `numerics/support/toolbox/`, dispatched by
  `toolbox_runner.hpp` exactly as P5's `ml` and P4's `hypothesis` / `paired_data` were. Dates
  travel as epoch seconds in a `data[]` vector (see scope decision 4); a method that returns a
  series returns an n-by-2 matrix through `ToolboxResult::dims`, the shape the `regression` group
  already established.
- The host surface is an object plus verbs, mirroring the `distribution()` / `Distribution`
  precedent: R gets `time_series()` and `ts_*()` functions, Python gets a `TimeSeries` class with
  methods. No new runner and no per-method glue.

**Tech Stack:** C++17 header-only core, cpp11 (R), pybind11 (Python), nlohmann/json fixture runner,
dotnet 10 oracle emitter.

## Global Constraints

- Every ported file carries `// ported from: <path> @ 2a0357a` (Numerics) or `@ c2e6192`
  (RMC.BestFit) and mirrors the C# class/method layout line-for-line where possible. `date_time.hpp`
  is the exception: it has no upstream file, so it carries a `// corehydro ADDITION` header naming
  the .NET behaviours it reproduces and where each is used.
- Upstream paths contain spaces (`Numerics/Data/Time Series/Support/`) -- quote them in every shell
  command. Numerics is valid UTF-8; reading and grepping it directly is fine.
- **RMC.BestFit is read with `git -C upstream/RMC-BestFit show c2e6192:'<path>'`, never grepped over
  the working tree.** The measured exception from P4/P5 still holds:
  `src/RMC.BestFit/Models/DataFrame/DataFrame.cs` IS valid UTF-8 (with a BOM) and must be read
  plain; piping it through `iconv` mangles the oracle-visible `λ` in a summary key.
- Oracle values live ONLY in `fixtures/*.json`; ctest suites transcribing C# test methods carry
  C#-test-literal assertions (the established second oracle class); never invent expected values.
  Where upstream has no numeric assertion, curate with `python3 tools/verify_oracles.py --dump`
  against the real library and say so in the fixture's `source` string.
- Every new fixture case gets a dotnet emitter driver; `python3 tools/verify_oracles.py` must end
  `0 failed`. The `toolbox` branch of the emitter does NOT honor `oracle_skip` -- an assertion the
  C# cannot reproduce is OMITTED from the fixture, not masked.
- No `M_PI` (use `corehydro::numerics::kPi`); no namespace aliases named `gamma` or `stat`; no
  file-local `const` used implicitly in a capture-less lambda (MSVC C3493 -- use file-scope
  `constexpr`); no new external C++ dependencies; no threads; no `<chrono>` calendar types
  (`std::chrono::year_month_day` is C++20 and the core is C++17).
- After editing any `corehydror/src/*.cpp`: `Rscript -e 'cpp11::cpp_register("corehydror")'`.
  Adding a toolbox GROUP does not require it; adding an entry point does.
- **Task 2 changes `TimeSeries`'s member layout and every model that owns one, so EVERY R rebuild
  from Task 2 onward uses `R CMD INSTALL --preclean corehydror`.** A stale `.o` from the `long`-index
  ABI will return garbage or abort R.
- pytest reads fixtures materialized by pip; re-run
  `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy` after fixture edits.
  testthat reads the source tree through pkgload's `system.file()` shim, so a fixture edit is live
  with no reinstall.
- Every new R export goes in BOTH `corehydror/_pkgdown.yml` and `site/_quarto.yml`
  `quartodoc.sections`, and in `corehydror/NAMESPACE` (hand-maintained -- do not let roxygen
  rewrite it). New Python exports join the module's `__all__` AND both the import block and
  `__all__` in `corehydropy/src/corehydropy/__init__.py`.
- R and Python error messages for the same condition must be textually identical (`stop(...,
  call. = FALSE)` vs `raise ValueError(...)`).
- Commits are GPG-signed, identity `Cam Bracken <cameron.bracken@pm.me>`, no Co-Authored-By
  trailers. Push only when Cam asks (the ship task pushes; that is the standing exception).
- **Branch base:** `port-time-series`, created off `45bec0e` (`origin/main`, the merge of PR #31).
- **Baseline numbers, measured on this branch at creation (2026-08-27):** ctest run recorded in
  Task 1 Step 0. The P5 release records the rest: oracle gate 6383 reproduced / 0 failed / 11
  skipped; testthat 7529/0; pytest 1829. Re-measure R/Python/oracle on the first task that runs
  them; if any number differs, the fresh measurement governs -- never reconcile against a stale
  census.

## Scope decisions taken during recon (do not re-litigate)

1. **A `DateTime` value type is unavoidable, and it mirrors .NET exactly rather than approximating.**
   The container groups by `Index.Year` / `.Month` / `.DayOfYear`, walks by `AddMonths(1)` /
   `AddYears(1)`, and `PointProcessModel.GeneratePOTTimeSeries` calls
   `startDate.AddDays(offsetDays)` with a FRACTIONAL double whose result feeds `DayOfYear`. So the
   port reproduces .NET's documented algorithms, not a convenient rewrite:
   - Ticks are 100 ns since 0001-01-01 00:00:00, stored in an `int64_t`; `default(DateTime)` is
     tick 0, which the seasonal PointProcess path tests for with `dt == default`.
   - `AddDays`/`AddHours`/`AddMinutes(double)` round to whole MILLISECONDS the way .NET's private
     `Add(double value, int scale)` does (`(long)(value * scale + (value >= 0 ? 0.5 : -0.5))`,
     then `* TicksPerMillisecond`), and throw when the result leaves `[MinValue, MaxValue]`.
   - `AddMonths(int)` clamps the day to the target month's length and preserves the time-of-day
     ticks; `AddYears` is `AddMonths(value * 12)` under .NET's range check.
   - `ToOADate()` (used by both `InterpolateMissingData` overloads) reproduces the .NET conversion
     including its `< TicksPerDay` and negative-value special cases.
   - The calendar is proleptic Gregorian from year 1, as .NET's is. There is NO time zone and no
     `DateTimeKind`: every date in this container is `Unspecified` in C#, and adding a zone would
     invent behaviour upstream does not have.
   Correctness is not asserted from memory: Task 1 pins it against the real `System.DateTime` with
   an emitter probe, and Task 5 puts a subset of that surface behind the oracle gate permanently.
2. **The index-type swap is a separate task from any new behaviour.** Task 2 changes
   `TimeSeries::IndexType` from `long` to `DateTime` and touches its five consumers
   (`model_spec.hpp`'s `build_time_series`, `rating_curve.hpp`'s date join, AR/MA/ARIMA/ARIMAX, and
   the ctests) WITHOUT adding a single new method. The existing suites -- ctest, the fixture runner,
   testthat, pytest -- must stay green unchanged. If one value moves, the mapping from the old
   integer offset to a date was not order- and equality-preserving and the task is wrong. Fixture
   `start_index` keeps working: it maps to `DateTime(1900, 1, 1) + i` steps of the series interval,
   and a new optional `start_date` key takes an ISO string for cases that care about the calendar.
3. **`Parallel.For` becomes a serial loop, exactly as P5 decided, with the same one exception.**
   `MonthlyPercentiles` and `MonthlySummaryStatistics` each write only to their own row of a
   pre-allocated array and read nothing another iteration writes, so serial execution is
   bit-identical. `MonthlySummaryStatistics` column 7 is `Statistics.ParallelMean`, whose PLINQ
   partitioned sum is not reproducible against itself across machines -- the entry P5 added to
   `docs/upstream-csharp-issues.md` covers it. That column is therefore OMITTED from the C#
   comparison in the fixture (not masked with `oracle_skip`, not given a loose tolerance), with a
   pointer to the existing entry, and the ctest asserts it against the serial mean instead.
4. **Dates cross into R and Python as epoch seconds (a double), never as ticks.** .NET ticks reach
   ~6.3e17 and a double carries 53 bits (~9e15), so a tick count is not exactly representable;
   epoch seconds for any date this century is ~1.7e9 and every interval down to `OneMinute` lands
   on a whole second. R takes and returns `POSIXct` in UTC (which IS epoch seconds); Python takes
   `datetime`, `numpy.datetime64`, pandas `DatetimeIndex`, ISO strings or floats and returns
   `numpy.datetime64[s]`. The conversion lives in the two host layers, not in the core, and the
   core's `DateTime` gains `from_unix_seconds` / `to_unix_seconds` as clearly-marked corehydro
   additions.
5. **`List<T>.Sort(Comparison<T>)` is the .NET introsort and its tie permutation is oracle-visible
   here.** `SortByValue` on a series with repeated values, and `SortByTime` on an irregular series
   with duplicate dates, both expose it. Use `numerics/utilities/dotnet_sort.hpp`'s
   `dotnet_list_sort` (extracted in P5) with a comparator matching `double.CompareTo` /
   `DateTime.CompareTo` semantics -- not `std::sort`, and not a raw `<`.
6. **`Clone()` is a direct deep copy, and that is exactly equivalent to upstream's XML round-trip.**
   C# `Clone()` is `new TimeSeries(ToXElement())`, and `ToXElement` writes the index with the "o"
   round-trip format and the value with "G17"; both are round-trip formats, and `NaN` survives
   `double.TryParse(..., NumberStyles.Any, ...)`. XML is a project-wide severance, so the port
   copies directly and the ctest pins the equivalence by asserting a clone equals its source
   bit-for-bit, NaNs included.
7. **`TimeSeries.SummaryHypothesisTest` is NOT the `DataFrame` facade of the same name.** The
   container's has SEVEN keys and splits on a value index; the DataFrame's (shipped in P5) has TEN
   and splits on an ordinate index. They are different methods with different key strings. Port
   this one as written and do not share an implementation between them.
8. **The severances stay severed.** `TimeSeriesDownload.cs` (network retrieval), XML persistence,
   `INotifyPropertyChanged` / `CollectionChanged` / `SuppressCollectionChanged` (ported as no-ops,
   the project-wide precedent), and the `XElement` constructors. `Series.cs`'s `Remove` / `RemoveAt`
   / `Clear` DO get ported -- v2.1.4 changed their semantics and upstream has three tests
   (`Test_Clear_*`, `Test_RemoveAt_WithDuplicateOrdinates_*`) asserting them, so the current
   "nothing to land on" note in `upstream/CLAUDE.md` retires with this phase.
9. **The host surface is an object plus verbs, and the object is inert data.** `time_series()` /
   `TimeSeries` carries dates, values and interval; every verb serializes them to the runner and
   gets a fresh answer. There is no live C++ handle across the boundary and no mutation surface: the
   C# in-place math methods (`Add`, `LogTransform`, `Standardize`, `ReplaceMissingData`, ...) are
   exposed as pure verbs that RETURN a new series, which is the idiom both host languages expect.
   The header for the group says so, naming each C# mutator it wraps.
10. **Upstream oddities found during recon are mirrored and measured, not fixed.** Four are already
    on the list to check and write up in `docs/upstream-csharp-issues.md` if they hold up against
    the real library: `PeaksOverThresholdSeries` resumes its outer loop at `idx + 1`, skipping the
    observation immediately after a cluster; `ConvertTimeInterval` computes its loop bound `N` as a
    double from `EndDate - StartDate` and can emit a trailing partial block; `MovingAverage` /
    `MovingSum` throw when `period >= Count` rather than at `period > Count`; and the `LogTransform`
    indexed overload's `else` branch NaNs any listed index whose value is non-positive OR
    out of range -- including writing through an out-of-range index, which the non-indexed overload
    cannot do. Measure each before writing it up; a suspicion is not a finding.

---

### Task 1: The `DateTime` value type, the three enums, and the sort-direction / subset prerequisites

**Files:**
- Create: `core/include/corehydro/numerics/data/time_series/support/date_time.hpp`
- Create: `core/include/corehydro/numerics/data/time_series/support/block_function_type.hpp`,
  `math_function_type.hpp`, `smoothing_function_type.hpp`, `list_sort_direction.hpp`
- Modify: `core/include/corehydro/numerics/utilities/extension_methods.hpp` (add `subset`)
- Create: `core/tests/test_date_time.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Produces, in `namespace corehydro::numerics::data`:
  - `class DateTime` -- `DateTime()` (tick 0), `DateTime(int64_t ticks)`,
    `DateTime(int year, int month, int day, int hour = 0, int minute = 0, int second = 0,
    int millisecond = 0)`, statics `min_value()`, `max_value()`, `days_in_month(int, int)`,
    `is_leap_year(int)`; accessors `ticks()`, `year()`, `month()`, `day()`, `hour()`, `minute()`,
    `second()`, `millisecond()`, `day_of_year()`, `day_of_week()`, `date()`, `time_of_day_ticks()`;
    arithmetic `add_ticks`, `add_milliseconds`, `add_seconds`, `add_minutes`, `add_hours`,
    `add_days`, `add_months`, `add_years`, `subtract_total_hours(const DateTime&)`; comparisons
    (`==`, `!=`, `<`, `<=`, `>`, `>=`, and `compare_to` for the sort comparator); `to_oa_date()`;
    and the corehydro additions `from_unix_seconds(double)`, `to_unix_seconds()`,
    `to_iso_string()`, `parse_iso(const std::string&)`.
  - `enum class BlockFunctionType { Minimum, Maximum, Average, Sum }`,
    `enum class MathFunctionType { Add, Subtract, Multiply, Divide, Logarithm, Exponentiate,
    Inverse, Replace, Interpolate }`,
    `enum class SmoothingFunctionType { Difference, MovingAverage, MovingSum, None }` -- all in C#
    declaration order.
  - `enum class ListSortDirection { Ascending, Descending }` (the `System.ComponentModel` enum the
    two sort methods take).
- Produces, in `namespace corehydro::numerics::utilities`:
  `std::vector<double> subset(const std::vector<double>& v, int start_index)` and
  `subset(const std::vector<double>& v, int start_index, int end_index)` -- the two
  `ExtensionMethods.Subset` overloads (INCLUSIVE of `end_index`; verify against
  `Numerics/Utilities/ExtensionMethods.cs` before transcribing, and rewrite the header's "omitted"
  note for `Subset` only).

- [ ] **Step 0: Record the baseline**

`cmake -S core -B core/build && cmake --build core/build -j8 && ctest --test-dir core/build` on the
freshly created branch. Write the pass count and wall time into this plan's Global Constraints
block. Do not proceed on a red baseline.

- [ ] **Step 1: Write the failing ctest**

`core/tests/test_date_time.cpp`. Every expected value in this file comes from the REAL
`System.DateTime`, produced by the probe in Step 3 -- write the test first with the values you
expect, then correct any that the probe contradicts and say in a comment that the probe governs.
Cover:

1. Construction and components: `DateTime(2023, 1, 1)` has `year() == 2023`, `month() == 1`,
   `day() == 1`, `day_of_year() == 1`, `ticks() == 638064864000000000`; `DateTime(2024, 12, 31)`
   has `day_of_year() == 366` (leap) and `DateTime(2023, 12, 31)` has `365`.
2. Range guards: month 0, month 13, day 0, day 32, day 30 of February, and year 0 each throw with
   .NET's message; `DateTime(2024, 2, 29)` constructs.
3. `add_months` clamping: `DateTime(2023, 1, 31).add_months(1)` is 2023-02-28;
   `DateTime(2024, 1, 31).add_months(1)` is 2024-02-29; `add_months(-1)` from 2023-03-31 is
   2023-02-28; a time-of-day of 13:45:30 survives all three.
4. `add_years`: 2024-02-29 plus one year is 2025-02-28.
5. `add_days` with a fraction: `DateTime(2000, 1, 1).add_days(1.5)` is 2000-01-02T12:00:00;
   `add_days(0.0000001)` rounds to the nearest millisecond, not to a tick.
6. Interval walking: 12 successive `add_months(1)` from 2023-01-31 land on the last day of each
   month and end at 2024-01-31 (the clamp is NOT sticky -- each step re-reads the original day
   only in so far as the previous result carries it; assert the exact 12 dates the probe gives).
7. `to_oa_date()`: 1899-12-30 is 0.0; 1900-01-01 is 2.0; 2023-01-01 is 44927.0; a date with a
   half-day time-of-day is `+0.5`; a pre-1899 date takes the negative branch.
8. Comparison and `compare_to`: strict ordering, equality on equal ticks, and
   `min_value() < max_value()`.
9. The corehydro additions round-trip: `from_unix_seconds(to_unix_seconds())` is identity for a
   whole-second date; `parse_iso(to_iso_string())` is identity; `to_unix_seconds()` of
   1970-01-01 is 0.
10. `subset({1..10}, 3)` is `{4,...,10}` and `subset({1..10}, 2, 5)` is `{3,4,5,6}` (INCLUSIVE end
    -- confirm against the C# source, not this plan).

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Probe the real `System.DateTime`, then implement**

Write a throwaway dotnet probe (a `dotnet script`, or a temporary `Main` in
`tools/oracle_emitter/`) that prints every value asserted in Step 1 from the real BCL, and diff it
against the test file. Any disagreement is the port's problem, not the BCL's. Record in the test
file's header comment that its literals came from the probe and on what date.

Then implement `date_time.hpp`. The header comment must state, per method, which .NET behaviour it
reproduces and which call site in this phase needs it (`add_months` -> `AddTimeInterval` and
`ShiftDatesByMonth`; `to_oa_date` -> `InterpolateMissingData`; `day_of_year` -> the seasonal
PointProcess path and `GeneratePOTTimeSeries`; and so on). Implement the civil-date conversion with
the standard days-from-civil algorithm over `int64_t` -- no `<ctime>`, no `<chrono>` calendar, no
`double` intermediate for the tick math.

- [ ] **Step 4: Run the ctest and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add a .NET-compatible date-time value type for the time-series port"
```

---

### Task 2: The `Series` base, and the index-type swap (no new behaviour)

**Files:**
- Create: `core/include/corehydro/numerics/data/time_series/support/series.hpp`
- Modify: `core/include/corehydro/numerics/data/time_series/time_series.hpp` (re-seat on `Series`,
  `IndexType` becomes `DateTime`)
- Modify: `core/include/corehydro/models/model_spec.hpp` (`build_time_series`),
  `core/include/corehydro/models/rating_curve/rating_curve.hpp` (the date join key),
  `core/include/corehydro/models/time_series/{auto_regressive,moving_average,arima,arimax}.hpp`
  (only where they name the index type)
- Modify: the ctests that construct a `TimeSeries` (`test_time_series.cpp`, `test_time_series_arma.cpp`,
  `test_arima.cpp`, `test_arimax.cpp`, `test_rating_curve.cpp`, `test_rating_curve_analysis.cpp`,
  `test_time_series_analyses.cpp`)

**Interfaces:**
- Produces, in `namespace corehydro::numerics::data`: `template <typename TIndex, typename TValue>
  class Series` -- the protected `series_ordinates_` vector, `count()`, `operator[]`, `add`,
  `insert`, `remove`, `remove_at`, `clear`, `contains`, `index_of`, `begin`/`end`,
  `values_to_list`/`values_to_array`, `indexes_to_list`/`indexes_to_array`, and
  `suppress_collection_changed` as an inert flag (the event is severed; the flag is ported because
  callers set it and the C# `RemoveAt` reads it).
- Changes: `TimeSeries::IndexType` from `long` to `DateTime`; the three surviving constructors take
  `DateTime` where they took `long`.
- `build_time_series` keeps `start_index` (mapped to `DateTime(1900, 1, 1)` advanced `i` intervals)
  and gains an optional `start_date` ISO-string key.

- [ ] **Step 1: Port `Series` and re-seat the container**

Transcribe `Support/Series.cs`. Three fidelity points for the header:

1. v2.1.4's `Clear()` raises ONE reset notification rather than one per element, and `RemoveAt`
   removes the requested INDEX rather than the first equal ordinate -- the two behaviours
   `Test_Clear_EmptiesSeriesAndRaisesSingleReset` and
   `Test_RemoveAt_WithDuplicateOrdinates_RemovesRequestedIndex` pin. The notification itself is
   severed; the removal semantics are not.
2. `Remove(item)` removes the first ordinate EQUAL to the argument (`SeriesOrdinate::operator==`
   compares index and value), which is not the same as removing by identity.
3. The `IList`/`ICollection`/`IEnumerable` non-generic implementations, `SyncRoot`,
   `IsSynchronized`, `CopyTo(Array, int)` and the `object`-typed overloads are .NET collection
   plumbing with no C++ analogue and are severed by name in the header.

- [ ] **Step 2: Swap the index type and fix every consumer**

`rating_curve.hpp`'s join is the only place the index is READ for meaning: it keys a lookup by date
to inner-join stage against discharge. A `std::map<DateTime, double>` keyed on ticks preserves the
existing behaviour exactly. Delete the P2 header note that says a plain integer index suffices and
replace it with what is now true.

- [ ] **Step 3: Prove the swap changed nothing**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy && pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
```
Expected: every count identical to the Task 1 Step 0 baseline, gate `0 failed`. **A single moved
value means the date mapping is not equality- and order-preserving; fix the mapping, do not re-pin
a fixture.**

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: port the Series base and move the time series onto real dates"
```

---

### Task 3: The container, part 1 -- construction, math, missing data, intervals, moving windows

**Files:**
- Modify: `core/include/corehydro/numerics/data/time_series/time_series.hpp` (the P2 adapter grows
  into the full container; its DEFERRED list shrinks to the severances in scope decision 8)
- Create: `core/tests/test_time_series_container.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:** everything in `TimeSeries.cs` lines 32-1331 -- the five non-XML constructors,
`time_interval()`, `has_missing_values()`, `start_date()`, `end_date()`, `sort_by_time`,
`sort_by_value`, the ten in-place math methods and their indexed overloads, `standardize`,
`cumulative_sum`, `difference`, `number_of_missing_values`, `replace_missing_data` x2,
`interpolate_missing_data` x2, the static `fill_missing_dates`, the statics `add_time_interval`,
`subtract_time_interval`, `time_interval_in_hours`, the private `check_if_min_steps_exceeded`,
`moving_average`, `moving_sum`, `shift_all_dates`, `shift_dates_by_day/month/year`,
`clip_time_series`, and `convert_time_interval`.

- [ ] **Step 1: Write the failing ctest**

Transcribe these upstream `[TestMethod]`s 1:1, reading every literal from
`Test_TimeSeries.cs`: `Test_Construction`, `Test_Getters`, `Test_Clone`, `Test_Sort`, `Test_Math`,
`Test_Cumulative`, `Test_Difference`, `Test_Difference_NaN`, `Test_Missing`, `Test_AddInterval`,
`Test_SubtractInterval`, `Test_TimeIntervalInHours`, `Test_MovingAverage`, `Test_MovingSum`,
`Test_MovingSum_NaN_Strict`, `Test_MovingSum_NaN_SkipMode`, `Test_MovingAverage_NaN_Strict`,
`Test_MovingAverage_NaN_SkipMode`, `Test_ShiftAllDates`, `Test_ShiftDatesByDays`,
`Test_ShiftDatesByMonth`, `Test_ShiftDatesByYear`, `Test_ClipTimeSeries`, the four
`Test_ConvertTimeInterval_*`, `Test_Clear_EmptiesSeriesAndRaisesSingleReset`,
`Test_Clear_LargeSeries_CompletesQuickly` (assert the result, drop the timing assertion and say
so), and `Test_RemoveAt_WithDuplicateOrdinates_RemovesRequestedIndex`. `Test_ToXElement` is skipped
by the XML severance -- record that in a numbered note rather than silently omitting it.

Supplement (corehydro additions, clearly marked): the `Clone` bit-for-bit equivalence from scope
decision 6 including a NaN; `sort_by_value` on a series with three tied values reproduces the
introsort permutation (assert the exact index order, cross-checked against the real library in Task
5's fixture); `divide(0)` throws with the C# message; and the four constructor guards throw.

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Transcribe lines 32-1331. Fidelity points the header must carry:

1. The in-place math methods SKIP `NaN` (missing stays missing) except `log_transform`, which
   WRITES `NaN` for any non-positive value, and `standardize`, which does not check at all and
   propagates `NaN` through the whole series.
2. `standardize` throws only when the standard deviation is exactly 0, and it uses the container's
   own NaN-skipping `mean_value()` / `standard_deviation()` -- not `statistics::mean`.
3. `interpolate_missing_data` calls `sort_by_time()` FIRST, interpolates in OA-date space, and its
   extrapolation branch reads `this[i - 2]`, so the indexed overload can read index `-1` when an
   early index is passed. Mirror it; the guard `i >= 2` exists only in the non-indexed overload.
4. `moving_average` / `moving_sum` keep a running sum and a running NaN count, so a NaN inside a
   window is subtracted out again when it leaves -- the result is NOT recomputed per window, and
   the accumulation order is oracle-visible.
5. `convert_time_interval` returns a null result for the unreachable final branch (map to
   `std::optional`), and its loop bound `N` is a double computed from the total hours between the
   end and start dates. Do not "tidy" it into an integer count.
6. `shift_all_dates` re-walks the index by `add_time_interval` from the new start, so it CHANGES
   the spacing of an irregular series to regular unless the interval is `Irregular`, where it moves
   only the first ordinate.
7. `difference` builds the result from `start_date()`, so a differenced series keeps the ORIGINAL
   start date rather than shifting forward by the lag.

- [ ] **Step 4: Run the ctest and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port the time-series container's construction, math, and interval methods"
```

---

### Task 4: The container, part 2 -- statistics, block series, decomposition, resampling

**Files:**
- Modify: `core/include/corehydro/numerics/data/time_series/time_series.hpp`
- Modify: `core/tests/test_time_series_container.cpp`

**Interfaces:** `TimeSeries.cs` lines 1333-2332 -- `min_value`, `max_value`, `mean_value`,
`standard_deviation`, `summary_percentiles`, `percentiles`, `duration`, `monthly_percentiles`,
`monthly_summary_statistics`, `summary_statistics`, `summary_hypothesis_test`,
`monthly_frequency`, `calendar_year_series`, `custom_year_series` (both overloads),
`monthly_series`, `quarterly_series`, `peaks_over_threshold_series`, `seasonal_decompose`,
`resample_with_knn`, `resample_with_block_bootstrap`, `clone`.

- [ ] **Step 1: Write the failing ctest**

Transcribe `Test_Stats`, `Test_SummaryStats`, `Test_SummaryHypothesisTest`, `Test_MonthlyStats`,
`Test_MonthlyFrequency`, `Test_CalendarYearSeries`, `Test_CalendarYearSeries_NaN`,
`Test_WaterYearSeries`, `Test_CustomYearSeries`, `Test_MonthlySeries`, `Test_MonthlySeries_NaN`,
`Test_QuarterlySeries`, `Test_PeaksOverThreshold`, `Test_PeaksOverThreshold_MovingSum_NaN`,
`Test_SeasonalDecompose`, `Test_SeasonalDecompose_InvalidInputs`, and all eleven
`Test_ResampleWith*` methods (including the statistical ones -- they use the shared `MakeAR1`
helper, so port that helper too, from the same file).

`Test_MonthlyStats` asserts the `MonthlySummaryStatistics` mean column, which is `ParallelMean`;
assert it against the serial mean and note scope decision 3.

- [ ] **Step 2: Run it and watch it fail**

- [ ] **Step 3: Implement**

Fidelity points for the header:

1. The five block-series methods share one hand-written block-function body that is COPIED four
   times upstream, with `Minimum` seeded at `double.MaxValue` and `Maximum` at `double.MinValue`
   and a strict comparison, so a block of all-NaN values yields an ordinate with a NaN value that
   is then dropped by the `if (!IsNaN)` guard. `Sum` and `Average` stamp the LAST ordinate's date;
   `Minimum` and `Maximum` stamp the extremum's own date. Port the repetition as repetition -- a
   shared helper is fine only if it is byte-for-byte the same computation, and the header must say
   which it is.
2. `custom_year_series(startMonth)` shifts the whole series FORWARD by `12 - startMonth + 1` months,
   groups by the shifted year, and shifts the RESULT back. That is what makes an October start a
   water year, and it is why the returned dates are the original event dates.
3. `peaks_over_threshold_series` resumes the outer loop at `idx + 1` (scope decision 10 --
   measure it, then write it up), and its cluster-maximum tie rule is `>=`, so the LAST tied peak
   in a cluster wins.
4. `seasonal_decompose` pads to a power of two, keeps only the harmonic bins of the seasonal
   frequency, and scales the inverse transform by `2 / fftLength`; the trend is
   `moving_average(period)` and the residual is only defined where the trend is. Use the ported
   `math::fourier::real_fft`.
5. `resample_with_knn` builds its `KNearestNeighbors` (P5) over the STANDARDIZED values with the
   LAST observation excluded, draws `prng.next(Count - 1)` for the start index, and advances to
   `this[selectedIdx + 1]` -- the conditional Lall-Sharma step. The PRNG draw order is
   start-index, then one `next(kNearest.Length)` per step; any reordering breaks the seeded oracle.
6. `resample_with_block_bootstrap` collects EVERY contiguous block (overlapping, despite the
   "non-overlapping" wording in the doc comment), samples with replacement until the length is
   reached, then trims.
7. `percentiles` sorts a NaN-filtered copy and calls `statistics::percentile(..., true)`;
   `duration` sorts ASCENDING then reverses, and pairs Weibull plotting positions with the
   descending values.
8. `summary_statistics` guards on `Count <= 2` (the ordinate count, not the non-NaN count) before
   computing moments and percentiles, so a series with three ordinates one of which is NaN still
   takes the compute branch.

- [ ] **Step 4: Run the ctest, then measure the four suspected upstream oddities**

Drive each of scope decision 10's four cases against the real library through a throwaway emitter
probe, and write up whichever reproduce in `docs/upstream-csharp-issues.md` with the measurement.
A suspicion that does not reproduce gets deleted from the list, not softened.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port the time-series container's statistics, block series, and resampling"
```

---

### Task 5: The `timeseries` toolbox group, its fixtures, and the emitter drivers

**Files:**
- Create: `core/include/corehydro/numerics/support/toolbox/timeseries.hpp`
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (the include block, the
  dispatch table, and the group count in its header comment: eighteen becomes nineteen)
- Create: `fixtures/timeseries/date_time.json`, `fixtures/timeseries/time_series.json`,
  `fixtures/timeseries/block_series.json`, `fixtures/timeseries/resampling.json`
- Modify: `tools/oracle_emitter/Program.cs` (a `TimeSeriesDispatch` arm), `fixtures/README.md`
- Modify: `core/tests/test_fixtures.cpp` only if the group needs a new data shape (it should not)

**Interfaces:** `detail::run_timeseries(method, data, options)` over `data[0]` = dates as epoch
seconds, `data[1]` = values, optional `data[2]` = a second date vector or an index list; every
scalar, enum name and flag in `options_json`. A method returning a series returns an n-by-2 matrix
(`dims == {n, 2}`, column 0 dates as epoch seconds, column 1 values). A method returning a named set
uses `names`, as the `statistics` group does.

- [ ] **Step 1: Write the group header and the fixtures together**

Methods, one per public container verb, named for the verb: `moving_average`, `moving_sum`,
`cumulative_sum`, `difference`, `seasonal_decompose`, `convert_interval`, `shift_days`,
`shift_months`, `shift_years`, `shift_all_dates`, `clip`, `fill_missing_dates`,
`replace_missing`, `interpolate_missing`, `math` (the `MathFunctionType` verbs behind one method
name, per scope decision 9), `standardize`, `sort`, `summary_statistics`, `summary_hypothesis_test`,
`percentiles`, `duration`, `monthly_percentiles`, `monthly_summary_statistics`,
`monthly_frequency`, `block_series` (the `TimeBlockWindow` selector over the five block methods),
`peaks_over_threshold`, `resample_knn`, `resample_block_bootstrap`, plus `date_add`,
`date_subtract`, `date_components` and `interval_in_hours` for the DateTime surface (scope
decision 1 -- this is what puts it permanently behind the oracle gate).

`fixtures/timeseries/date_time.json` pins the `add_months` clamping table, the fractional
`add_days` rounding, `day_of_year` across a leap boundary, and `to_oa_date` -- every value
reproduced by the emitter from the real `System.DateTime`.

- [ ] **Step 2: Wire the emitter and run the gate**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: `0 failed`. Omit, with a comment naming scope decision 3, the `monthly_summary_statistics`
mean column from the C# comparison.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add the time-series toolbox group and its oracles"
```

---

### Task 6: The R and Python surface

**Files:**
- Create: `corehydror/R/timeseries.R`; Modify: `corehydror/NAMESPACE`, `corehydror/_pkgdown.yml`
- Create: `corehydropy/src/corehydropy/timeseries.py`; Modify:
  `corehydropy/src/corehydropy/__init__.py`, `site/_quarto.yml`
- Create: `corehydror/tests/testthat/test-timeseries.R`, `corehydropy/tests/test_timeseries.py`

**Interfaces:**
- R: `time_series(dates, values, interval = "one_day")` returning a `corehydro_ts` classed list,
  with `print`, `format`, `as.data.frame`, `length` and `[` methods, and the verbs
  `ts_moving_average`, `ts_moving_sum`, `ts_cumulative_sum`, `ts_difference`,
  `ts_seasonal_decompose`, `ts_convert_interval`, `ts_shift`, `ts_clip`, `ts_fill_missing_dates`,
  `ts_replace_missing`, `ts_interpolate_missing`, `ts_math`, `ts_standardize`, `ts_sort`,
  `ts_statistics`, `ts_hypothesis_test`, `ts_percentiles`, `ts_duration`, `ts_monthly_statistics`,
  `ts_monthly_frequency`, `ts_block_series`, `ts_water_year`, `ts_calendar_year`,
  `ts_peaks_over_threshold`, `ts_resample_knn`, `ts_resample_block_bootstrap`.
- Python: a `TimeSeries` class with the same operations as methods (`ts.moving_average(7)`,
  `ts.water_year(block="maximum")`, ...), `dates` / `values` / `interval` properties, `__len__`,
  `__repr__`, `to_frame()` (pandas if available, else a dict of arrays -- pandas must NOT become a
  hard dependency), and `from_frame()`.
- Dates in: R `POSIXct` (UTC), `Date`, or ISO character; Python `datetime`, `numpy.datetime64`,
  pandas `DatetimeIndex`, ISO `str`, or float epoch seconds. Dates out: R `POSIXct` in UTC, Python
  `numpy.datetime64[s]`.

- [ ] **Step 1: Write the failing binding tests**

Both suites assert the same things in the same order: construction from each accepted date type;
that a round-trip through the object preserves dates exactly; the C# literal from
`Test_CalendarYearSeries` reached through `ts_block_series` / `ts.calendar_year()`; a seeded
`resample_knn` reproducing the fixture; and that R and Python agree bit-for-bit on the seeded
resampling (the digest that Task 9 promotes into a cross-language fixture).

Add the error-message parity check: an unknown `interval`, an unknown `block`, and a `period`
larger than the series each raise the SAME message string in both languages.

- [ ] **Step 2: Implement both surfaces, then the docs indexes**

Every new export lands in `_pkgdown.yml` and `site/_quarto.yml` in the same commit. P1's
`type_name` lesson applies: a name or label that no fixture pins is exactly what silently breaks,
so the tests assert the printed interval name and the returned column names, not just the numbers.

- [ ] **Step 3: Run both suites**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: expose the time-series container in R and Python"
```

---

### Task 7: Un-gate the BestFit data paths -- `ExactData.DateTime`, the two DataFrame builders, and the seasonal PointProcess model

**Files:**
- Modify: `core/include/corehydro/models/data_frame/data_types/exact_data.hpp` (add the DateTime
  member, the `(DateTime, double)` constructor, the accessor, the clone carry-over; retire the
  severance note)
- Modify: `core/include/corehydro/models/data_frame/data_frame.hpp` (`create_block_series`,
  `create_peaks_over_threshold_series`; retire the note at ~line 65)
- Modify: `core/include/corehydro/models/univariate_distribution/point_process_model.hpp` (the
  seasonal branch of `set_ams_data`, `generate_pot_time_series`; retire the DEFERRED block at the
  top)
- Modify: `core/include/corehydro/models/data_frame_runner.hpp`, `core/include/corehydro/models/model_spec.hpp`
- Modify: `fixtures/data/*.json` and the point-process fixture; `tools/oracle_emitter/Program.cs`
- Modify: `core/tests/test_data_frame.cpp`, `core/tests/test_point_process_model.cpp`

**Interfaces:**
- `ExactData(DateTime date_time, double value)` sets `index = date_time.year()` and keeps the date
  (C# `ExactData.cs:31`); `date_time()` accessor; `clone()` carries it.
- `DataFrame::create_block_series(const TimeSeries&, TimeBlockWindow = WaterYear,
  BlockFunctionType = Maximum, SmoothingFunctionType = None, int start_month = 10,
  int end_month = 9, int period = 1)` and
  `create_peaks_over_threshold_series(const TimeSeries&, double threshold,
  int min_steps_between_peaks = 1, SmoothingFunctionType = None, int period = 1)`.
- `PointProcessModel::set_ams_data()`'s seasonal branch, and
  `generate_pot_time_series(DateTime start_date, double duration_years, int seed = -1)`.

- [ ] **Step 1: Write the failing tests**

Transcribe the upstream BestFit tests that cover these (read them with `git -C upstream/RMC-BestFit
show`; find them under `src/RMC.BestFit.Tests/` for `DataFrame` and `PointProcessModel`). Where a
seasonal test was previously skipped in our ctest because the path was a no-op, un-skip it rather
than writing a new one.

- [ ] **Step 2: Implement**

Fidelity points:

1. `create_block_series` sets `lambda = 1` unconditionally;
   `create_peaks_over_threshold_series` sets `lambda = events / (endYear - startYear + 1)`.
2. The seasonal `set_ams_data` branch substitutes `DateTime(exact.index(), 1, 1)` when the ordinate
   carries no date (`dt == default`), shifts by `12 - StartMonth + 1` months for a water year, and
   fills `pot_days` from `DayOfYear` AFTER the shift. The whole body stays inside the C#
   swallow-all `catch` -- keep it, and keep the debug message's intent in a comment.
3. `generate_pot_time_series` draws `SamplePoisson(lambda * durationYears, rng)` first, then one
   `next_double()` per event scaled by `durationYears * 365.25`, and assigns the season from
   `eventDate.day_of_year()` against `k1`/`k2`. The already-ported `sample_poisson` helper is the
   one it calls -- do not write a second one.

- [ ] **Step 3: Fixtures, emitter, gate**

The seasonal path is now reachable, so the seasonal PointProcess fixtures that previously asserted
the empty-AMS consequence must be re-pinned against the real C#. That is expected and is the point
of the task: state in each fixture's `source` that the value moved because the path un-gated.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: un-gate the seasonal point-process and data-frame block series"
```

---

### Task 8: Un-gate the ARIMAX covariate forecast tail and the Autocorrelation TimeSeries overloads

**Files:**
- Modify: `core/include/corehydro/models/time_series/arimax.hpp` (replace the two throwing guards)
- Modify: `core/include/corehydro/numerics/data/autocorrelation.hpp` (the four TimeSeries
  overloads; retire the severance note in that header and in `upstream/CLAUDE.md`)
- Modify: `core/include/corehydro/numerics/support/toolbox/spectra.hpp` (accept a dated series)
- Modify: the ARIMAX and spectra fixtures, `tools/oracle_emitter/Program.cs`
- Modify: `core/tests/test_arimax.cpp`, `core/tests/test_autocorrelation.cpp`

**Interfaces:** `CovariateExtensionMethod::{None, BlockBootstrap, KNN}` becomes live in
`predict_components` and `generate_random_values`; `autocorrelation::function(const TimeSeries&,
int lag_max = -1, Type = Correlation)` plus its three private helpers.

- [ ] **Step 1: Write the failing tests**

Transcribe the ARIMAX forecast tests that exercise each extension method (C# `ARIMAX.cs:1595-1650`
and `2342-2360` name them) and the `Autocorrelation` TimeSeries overload tests.

- [ ] **Step 2: Implement**

The `None` arm still throws -- that is upstream's behaviour, not a deferral, and the message must
now say so rather than saying "deferred". Every throwing string in these two files that mentions
the unported container is rewritten in this task; grep for "deferred" in both files afterwards and
confirm nothing stale is left.

- [ ] **Step 3: Gate and commit**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build && python3 tools/verify_oracles.py
git add -A && git commit -m "feat: un-gate the ARIMAX covariate extension and autocorrelation over a time series"
```

---

### Task 9: Cross-language proof, the worked example, and the release collateral

**Files:**
- Create: `fixtures/timeseries/timeseries_cross_language.json`
- Create: `site/examples/30-time-series/{python.ipynb, r.qmd}`; Modify: `site/_quarto.yml`,
  `site/examples/index.qmd`
- Modify: `site/status.qmd`, `site/index.qmd`, `README.md`, `corehydropy/README.md`,
  `.claude/CLAUDE.md`, `.claude/PLAN.md`, `upstream/CLAUDE.md`, `docs/upstream-csharp-issues.md`
- Modify: `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`, `NEWS.md`, `CHANGELOG.md`
  (version 0.13.0)

- [ ] **Step 1: The cross-language fixture**

One case name nesting a seeded `resample_knn`, a seeded `resample_block_bootstrap`, and a
deterministic `block_series` -- the pattern `fixtures/toolbox/toolbox_cross_language.json` and
`fixtures/callback/callback_cross_language.json` set. Assert at ZERO tolerance across all four
runners. The whole computation lives in the shared core (no host callback), so R-to-Python
bit-identity is a real guarantee here, unlike the callback layer's; say exactly that in the fixture
`source` and in the example.

- [ ] **Step 2: The worked example**

Example 30, one hydrologic problem end to end: a daily gauge record in, missing data filled,
converted to a water-year annual maximum series, fitted, and a POT series extracted with an
independence criterion -- then a seeded KNN resampling with the R and Python results compared, and
an executable reproduction check at the end of both pages. Python is a committed notebook WITH
outputs; R is Quarto with `freeze: auto` and the updated `site/_freeze/` committed.

- [ ] **Step 3: The status sweep**

`site/status.qmd`'s "Data (TimeSeries container) | Not ported" row retires. The `upstream/CLAUDE.md`
"what is deliberately not ported" entries for `Series.cs` and the heavy container retire; the
`TimeSeriesDownload.cs` entry stays. `.claude/CLAUDE.md` gets the P6 paragraph. Every number in the
prose is the number this branch measured, not a copied one.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs: worked example 30, the v0.13.0 release notes, and the status sweep"
```

---

### Task 10: Ship

- [ ] **Step 1: The full sweep, from clean**

```bash
rm -rf core/build && cmake -S core -B core/build && cmake --build core/build -j8 && ctest --test-dir core/build
python3 tools/verify_oracles.py
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy && pixi run python -m pytest corehydropy/tests -q
R CMD build corehydror && R CMD check --as-cran corehydror_0.13.0.tar.gz
make docs
```
`R CMD check --as-cran` must hold at the same three known NOTEs with no WARNING. Run it EARLY
enough in the phase to fix what it finds (the P4 lesson), not for the first time here.

- [ ] **Step 2: Whole-branch review, then push and open the PR**

Review the full branch diff before pushing -- every prior phase found something no single task's
review could see. Then push and open the PR against `main` with the measured numbers.
