// ported from: Numerics/Data/Paired Data/OrderedPairedData.cs @ 2a0357a
//
// P4 Task 8: the centrepiece of the Paired Data subsystem -- an x-y curve container that keeps
// itself sorted/validated against a caller-chosen monotonicity contract (StrictX/OrderX,
// StrictY/OrderY), plus linear interpolation with optional per-axis transforms, trapezoidal-rule
// area, six search algorithms ({sequential, bisection, hunt} x {x, y}), and three
// curve-simplification algorithms (Douglas-Peucker, Visvaligam-Whyatt, Lang). Builds on
// Task 7's Ordinate (ordinate.hpp) and LineSimplification (line_simplification.hpp, whose free
// `ramer_douglas_peucker` is a DIFFERENT algorithm implementation from this class's own
// `douglas_peucker_simplify` -- see ordinate.hpp's transcription note 2 and this file's own note
// below for the two independent PerpendicularDistance formulas).
//
// Severed (observable-collection / XML plumbing, project-wide precedent -- see e.g.
// ordinate.hpp's own header and models/data_frame/data_frame.hpp's): the XElement constructor,
// SaveToXElement(), the CollectionChanged event, SuppressCollectionChanged, and
// RaiseCollectionChangedReset. Every mutator below (add/insert/remove/remove_at/remove_range/
// clear/set) is therefore a straight port of the underlying List<Ordinate> mutation plus the
// IsValid bookkeeping, with no event-raising step. GetHashCode/Equals(object) are likewise not
// ported (no untyped-equality/hashing consumer in this port's scope; operator== is the one
// equality entry point, matching Ordinate's own precedent).
//
// STALE DOC (do not implement): the C# class <remarks> advertises a `Transform()` method "which
// allows to transform the OrderedPairedData class with another to create a new class" (e.g. flow
// -> stage via a rating curve). No such method exists anywhere in the v2.1.4 source -- grep
// confirms only Ordinate.Transform(xTransform, yTransform) (ported in ordinate.hpp) is real. The
// doc comment is aspirational/stale; nothing is invented here to match it.
//
// Five numbered transcription notes -- each an upstream defect or aliasing that a "cleanup"
// would silently change, reproduced on purpose. All five are new entries for
// docs/upstream-csharp-issues.md (Task 11 writes them up):
//
// 1. `remove_range(index, count)` (C# RemoveRange, lines ~453-462) has TWO bugs in five lines.
//    First, the guard is `index < 0 || (index + count) >= Count` where `>` is the correct
//    off-by-one-free test, so calling RemoveRange such that `index + count == Count` (i.e.
//    removing a trailing run of elements including the very last one) is a silent no-op instead
//    of performing the removal. Second, the loop that builds the `items` list handed to the
//    CollectionChanged event runs `for (i = index; i < count; i++)` rather than
//    `i < index + count` -- but since CollectionChanged is severed in this port (see above),
//    that `items` list has no consumer and no observable effect either way, so only the first
//    bug (the guard) is transcribed below; the second is recorded here for the documentation
//    task but has no C++ counterpart to reproduce. The Uncertain twin (Task 9,
//    UncertainOrderedPairedData) has NEITHER bug. `Test_Indexing` below calls
//    `remove_range(0, 3)` on 13 elements (3 < 13, well clear of the boundary), which sidesteps
//    both.
// 2. `sequential_search_y` (C# SequentialSearchY, line ~1114) reads `_ordinates[XSearchStart].Y`
//    where `_ordinates[YSearchStart].Y` is meant -- an X/Y search-state mix-up. With both
//    search-start fields starting at 0 and this port's ctest never diverging them before calling
//    sequential_search_y, the results agree with what the correct code would produce, which is
//    exactly why upstream's own `Test_Sequential` (transcribed below) passes despite the bug.
// 3. `x_delta_start_`/`y_delta_start_` (C# XdeltaStart/YdeltaStart) are declared, initialized to
//    0, and NEVER assigned anywhere in the class, so `x_correlated_`/`y_correlated_` (C#
//    Xcorrelated/Ycorrelated) can only become true when a search lands EXACTLY on the previous
//    search-start index (`abs(start - search_start) > 0` is false only for an exact match), and
//    `use_smart_search()`'s Hunt branch is therefore almost never taken -- `search_x`/`search_y`
//    fall through to Bisection on nearly every call. Contrast Interpolater.cs:46 (ported at
//    interpolater.hpp), which sets `delta_start = min(1, (int)pow(Count, 0.25))`, giving that
//    sibling class's correlated-search machinery a real (if degenerate, see that file's own
//    note) chance of firing.
// 4. `add` (C# Add, lines ~468-474) assigns `IsValid = OrdinateValid(Count - 1)`
//    UNCONDITIONALLY, so appending a single well-ordered point can flip an already-invalid
//    collection's IsValid flag back to true, even though earlier ordinates still violate the
//    monotonicity contract -- OrdinateValid only inspects the newly-added point's immediate
//    neighbor, not the whole series. `insert` (C# Insert, lines ~481-488) does not have this
//    bug: `if (IsValid) IsValid = OrdinateValid(index);` only ever NARROWS validity (true ->
//    possibly false), never widens it (false -> true). The Uncertain twin's Add only ever
//    assigns false, i.e. it can only narrow. The private `ordinate_valid(int)` (C# OrdinateValid)
//    itself mirrors this asymmetry at the boundary: it returns `true` for an out-of-range index
//    in THIS class, where the Uncertain twin's equivalent returns `false`.
// 5. `lang_simplify` (C# LangSimplify, lines ~1445-1448) returns `this` -- an ALIAS of the
//    receiver, not a copy -- when its guard `lookAhead <= 1 || tolerance <= 0` trips, while
//    DouglasPeuckerSimplify and VisvaligamWhyattSimplify both always return a freshly
//    constructed object. DECISION (this port, no C# analogue is possible): `lang_simplify`
//    below has a value-returning signature (`OrderedPairedData`, not a reference or pointer), so
//    there is no C++ construct that aliases `*this` the way a C# reference-type return can --
//    returning `*this` by value already copies it. This port therefore returns `clone()`
//    (construction-visible not-the-same-object semantics that match every OTHER simplifier's
//    contract) rather than trying to manufacture reference-return aliasing that no other method
//    here has and that callers of a value-typed API would not expect. The ctest below
//    (`test_lang_simplify_guard`) asserts this explicitly: the guarded return is
//    content-equal to the original AND independently mutable (proving it is a distinct object,
//    not merely equal by value) -- the only choice a value-returning signature leaves open.
//
// Sixth finding, NOT in the brief's five, discovered while writing the ctest and CONFIRMED
// against the real C# library (`dotnet run` against upstream/Numerics @ 2a0357a, not just this
// port's own transcription): unlike `douglas_peucker_simplify` and
// `visvaligam_whyatt_simplify` -- both of which explicitly force-keep the first AND last
// ordinate -- `lang_simplify` has NO such guarantee. CORRECTED MECHANISM (P4 whole-branch-review
// finding M7b -- the paragraph below used to misdescribe WHERE the point is lost; the loss
// itself is real and unchanged, only the explanation was wrong): the loop's own look-ahead clamp
// (`if (i + la > n) la = n - i - 1;`) uses a STRICT `>`, so at the exact-equality tail boundary
// (`i + la == n`) it does NOT fire and `la` stays at its full, un-clamped value; on this curve
// that happens at `i = 3` with `la` still 2. `recursive_tolerance`'s own inner guard
// (`if (i + n < count())`, also strict `<`) then ALSO fails at that SAME exact equality
// (`i + n == count()`) and is skipped entirely, so the call falls through to `return n;`
// UNCHANGED rather than ever testing whether the angle condition should reduce it. Back in the
// caller, this unreduced `offset` (2) points one past the last valid ordinate
// (`i + offset == count()`), so the caller's own `(i + offset) < count()` check -- correctly,
// given that oversized offset -- rejects the append, and the loop walks `i` to `count` and exits
// without ever revisiting the final point. (Verified: changing the loop clamp's `>` to `>=`
// -- so `la` clamps down to 1 at `i = 3`, `recursive_tolerance`'s guard then fires at `4 < 5`,
// and the resulting smaller `offset` of 1 correctly targets the real last ordinate -- reproduces
// the expected 4-point result against the real C# library; this is a candidate upstream fix, not
// applied here per the "reproduce upstream, do not silently fix" rule.) This is exactly what
// happens on the shared five-point sin curve at tolerance=0.01, look_ahead=2 (the very case
// upstream's own Test_LangSimplify exercises):
// `LangSimplify` returns 3 points, `{(0,0), (1.57,1), (4.71,-1)}`, dropping `(6.28,0)`
// entirely -- verified directly against the real C# library, not inferred. Upstream's own test
// NEVER catches this because its assertion loop is bounded by `test.Count` (the actual,
// possibly-short result), not `valid.Count`: with `test.Count == 3`, the loop only ever compares
// indices 0-2, and `valid[3] == (6.28,0)` is silently never checked -- the exact class of
// weak assertion the brief flagged for the simplification tests generally, caught here by this
// port's own length-first supplement (`test_lang_simplify` below asserts the CORRECT verified
// 3-point result, not the brief's claimed 4-point one, which does not match either this port's
// output or the real C# library's).
//
// Seventh finding (P4 whole-branch-review finding M1, CRITICAL): `visvaligam_whyatt_simplify`
// (C# VisvaligamWhyattSimplify, lines ~1398-1424) reads `ords[j + 1]` up to `j == Count - 2` and
// `ords[2]` unconditionally on every outer-loop iteration, exactly like every other C# List<T>
// indexer in this file -- but C#'s indexer THROWS `ArgumentOutOfRangeException` the moment `Count`
// drops below what the read needs, while the un-guarded C++ `std::vector::operator[]` this was
// transcribed to just reads past the end: undefined behavior, not a deterministic exception. A
// small `num_to_keep` (0, -1, ...) removes ordinates down past 3 while `remove_limit` is still
// unclamped, so the very next iteration's `ords[2]` read is out of bounds; measured, this crashed
// both R and Python outright (a bus error / SIGSEGV) for `num_to_keep` of 0 or -1, and for
// `num_to_keep = 1` it did NOT crash -- it silently returned a value read from freed/adjacent
// memory, which is worse than a crash because nothing signals the result is wrong. The fix below
// throws `std::out_of_range` at the exact point C#'s indexer would (`Count < 3`, checked at the
// top of the loop body before the `ords[2]` read), rather than clamping `remove_limit` or
// otherwise changing what the algorithm computes for any `num_to_keep` that does NOT hit this
// bound -- that part of the port is unaffected and unchanged.
//
// Also noted (not bugs, harmless-but-latent): `douglas_peucker_reduction`'s recursive helper
// (C# private DouglasPeuckerReduction(int,int,double,ref List<int>)) guards its recursive split
// with `maxDistance > tolerance & indexFarthest != 0` -- the `indexFarthest != 0` sentinel exists
// because `indexFarthest` is initialized to 0 and never reassigned if the `for` loop's body never
// improves on `dmax = 0`; testing against the loop's own starting index (`firstPoint`, whose
// self-distance is always exactly 0 and can therefore never win the `distance > maxDistance`
// comparison) would have been an equally correct guard, but the `!= 0` literal is what upstream
// wrote and it is transcribed as-is. `perpendicular_distance` below (the triangle-area-over-base
// formula, DIFFERENT from line_simplification.hpp's normalized-direction formula -- see
// ordinate.hpp's transcription note 2 for that split) divides by `triangle_base`, which is exactly
// 0 when `firstPoint`/`lastPoint` coincide (i.e. a degenerate segment); unlike
// line_simplification.hpp's PerpendicularDistance, THIS formula has no `mag > 0.0` guard, so a
// degenerate segment produces `0.0/0.0 = NaN`. No test in scope (this port's or upstream's)
// constructs a first/last pair that coincide, so this is unexercised in both languages; it is
// transcribed exactly as upstream wrote it, matching the file-wide "cleanup would change
// behavior" rule.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/data/paired_data/ordinate.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data::paired_data {

class OrderedPairedData {
   public:
    // C# constructor 1 (lines ~186-201): empty instance with the monotonicity contract set up
    // front; `capacity` only pre-reserves storage, mirroring List<Ordinate>(capacity).
    OrderedPairedData(bool strict_on_x, SortOrder x_order, bool strict_on_y, SortOrder y_order,
                       int capacity = 0) {
        if (capacity > 0) ordinates_.reserve(static_cast<std::size_t>(capacity));
        set_strict_x(strict_on_x);
        set_strict_y(strict_on_y);
        set_order_x(x_order);
        set_order_y(y_order);
    }

    // C# constructor 2 (lines ~212-221): parallel x/y arrays.
    OrderedPairedData(const std::vector<double>& x_data, const std::vector<double>& y_data,
                       bool strict_on_x, SortOrder x_order, bool strict_on_y, SortOrder y_order) {
        set_strict_x(strict_on_x);
        set_strict_y(strict_on_y);
        set_order_x(x_order);
        set_order_y(y_order);
        ordinates_.reserve(x_data.size());
        for (std::size_t i = 0; i < x_data.size(); ++i) ordinates_.emplace_back(x_data[i], y_data[i]);
        validate();
    }

    // C# constructor 3 (lines ~231-240): a list of ordinates. C# rebuilds each entry as
    // `new Ordinate(data[i].X, data[i].Y)`, which recomputes IsValid from X/Y -- equivalent to
    // just copying the Ordinate, since Ordinate's own IsValid is always exactly that computation.
    OrderedPairedData(const std::vector<Ordinate>& data, bool strict_on_x, SortOrder x_order,
                       bool strict_on_y, SortOrder y_order) {
        set_strict_x(strict_on_x);
        set_strict_y(strict_on_y);
        set_order_x(x_order);
        set_order_y(y_order);
        ordinates_.reserve(data.size());
        for (const auto& o : data) ordinates_.emplace_back(o.x, o.y);
        validate();
    }

    int count() const { return static_cast<int>(ordinates_.size()); }

    const Ordinate& operator[](int index) const { return ordinates_[static_cast<std::size_t>(index)]; }

    // C# indexer setter (lines ~309-332), the C++ analogue: operator[] has no set-through-index
    // idiom without a proxy object, so the setter's revalidation logic lives in this named
    // method (no CollectionChanged event to raise -- severed, see file header).
    void set(int index, const Ordinate& value) {
        std::size_t ui = static_cast<std::size_t>(index);
        if (ordinates_[ui] != value) {
            ordinates_[ui] = value;
            if (is_valid_) {
                if (!ordinate_valid(index)) is_valid_ = false;
            } else {
                if (ordinate_valid(index)) validate();
            }
        }
    }

    bool is_valid() const { return is_valid_; }

    bool strict_x() const { return strict_x_; }
    void set_strict_x(bool value) {
        if (strict_x_ != value) {
            strict_x_ = value;
            validate();
        }
    }
    bool strict_y() const { return strict_y_; }
    void set_strict_y(bool value) {
        if (strict_y_ != value) {
            strict_y_ = value;
            validate();
        }
    }
    SortOrder order_x() const { return order_x_; }
    void set_order_x(SortOrder value) {
        if (order_x_ != value) {
            order_x_ = value;
            validate();
        }
    }
    SortOrder order_y() const { return order_y_; }
    void set_order_y(SortOrder value) {
        if (order_y_ != value) {
            order_y_ = value;
            validate();
        }
    }

    int x_search_start() const { return x_search_start_; }
    void set_x_search_start(int value) { x_search_start_ = value; }
    int y_search_start() const { return y_search_start_; }
    void set_y_search_start(int value) { y_search_start_ = value; }
    bool use_smart_search() const { return use_smart_search_; }
    void set_use_smart_search(bool value) { use_smart_search_ = value; }

    // C# IndexOf(Ordinate) (lines ~339-342): List<T>.IndexOf, which for this struct resolves to
    // Ordinate.Equals(object) -> operator== (see ordinate.hpp) -- a linear scan.
    int index_of(const Ordinate& item) const {
        for (int i = 0; i < count(); ++i)
            if (ordinates_[static_cast<std::size_t>(i)] == item) return i;
        return -1;
    }

    // C# IndexOf(double, double) (lines ~350-358): 1e-9 absolute tolerance on both coordinates.
    int index_of(double x_value, double y_value) const {
        for (int i = 0; i < count(); ++i) {
            const Ordinate& o = ordinates_[static_cast<std::size_t>(i)];
            if (std::fabs(o.x - x_value) <= 0.000000001 && std::fabs(o.y - y_value) <= 0.000000001)
                return i;
        }
        return -1;
    }

    // C# GetErrors (lines ~407-416).
    std::vector<std::string> get_errors() const {
        std::vector<std::string> result;
        if (is_valid_) return result;
        if (ordinates_.empty()) {
            // Faithful analogue of C#'s `_ordinates.Last()` on an empty list (LINQ throws
            // InvalidOperationException: "Sequence contains no elements.") -- mirrored as a
            // throw rather than the undefined behavior `.back()` on an empty vector would be.
            throw std::runtime_error("Sequence contains no elements.");
        }
        for (int i = 0; i < count() - 1; ++i) {
            auto errs = ordinates_[static_cast<std::size_t>(i)].ordinate_errors(
                ordinates_[static_cast<std::size_t>(i + 1)], strict_x_, strict_y_, order_x_, order_y_,
                true);
            result.insert(result.end(), errs.begin(), errs.end());
        }
        auto last_errs = ordinates_.back().ordinate_errors();
        result.insert(result.end(), last_errs.begin(), last_errs.end());
        return result;
    }

    // C# Remove (lines ~423-432).
    bool remove(const Ordinate& item) {
        int idx = index_of(item);
        if (idx == -1) return false;
        ordinates_.erase(ordinates_.begin() + idx);
        validate();
        return true;
    }

    // C# RemoveAt (lines ~438-446).
    void remove_at(int index) {
        if (index < 0 || index >= count()) return;
        ordinates_.erase(ordinates_.begin() + index);
        validate();
    }

    // C# RemoveRange (lines ~453-462). See transcription note 1 above: the guard uses `>=`
    // where `>` is correct, so a removal reaching the trailing element is a silent no-op.
    void remove_range(int index, int cnt) {
        if (index < 0 || (index + cnt) >= count()) return;
        ordinates_.erase(ordinates_.begin() + index, ordinates_.begin() + index + cnt);
        validate();
    }

    // C# Add (lines ~468-474). See transcription note 4 above: the unconditional assignment can
    // widen IsValid from false back to true.
    void add(const Ordinate& item) {
        ordinates_.push_back(item);
        is_valid_ = ordinate_valid(count() - 1);
    }

    // C# Insert (lines ~481-488): only narrows IsValid (true -> possibly false).
    void insert(int index, const Ordinate& item) {
        ordinates_.insert(ordinates_.begin() + index, item);
        if (is_valid_) is_valid_ = ordinate_valid(index);
    }

    // C# Clear (lines ~493-506) minus the SuppressCollectionChanged save/restore dance (no
    // event to suppress in this port).
    void clear() {
        if (ordinates_.empty()) return;
        ordinates_.clear();
        is_valid_ = true;
    }

    bool contains(const Ordinate& item) const { return index_of(item) != -1; }

    // C# Clone (lines ~533-536).
    OrderedPairedData clone() const {
        return OrderedPairedData(ordinates_, strict_x_, order_x_, strict_y_, order_y_);
    }

    // C# Invert (lines ~541-547).
    OrderedPairedData invert() const {
        std::vector<Ordinate> inverted;
        inverted.reserve(ordinates_.size());
        for (const auto& o : ordinates_) inverted.emplace_back(o.y, o.x);
        return OrderedPairedData(inverted, strict_y_, order_y_, strict_x_, order_x_);
    }

    // C# TrapezoidalAreaUnderY (lines ~661-679).
    double trapezoidal_area_under_y() const {
        double sum = 0.0;
        if (order_x_ == SortOrder::Ascending) {
            for (int i = 1; i < count(); ++i) {
                std::size_t ui = static_cast<std::size_t>(i);
                sum += 0.5 * (ordinates_[ui].x - ordinates_[ui - 1].x) *
                       (ordinates_[ui - 1].y + ordinates_[ui].y);
            }
        } else if (order_x_ == SortOrder::Descending) {
            for (int i = 1; i < count(); ++i) {
                std::size_t ui = static_cast<std::size_t>(i);
                sum += 0.5 * (ordinates_[ui - 1].x - ordinates_[ui].x) *
                       (ordinates_[ui - 1].y + ordinates_[ui].y);
            }
        } else {
            throw std::runtime_error(
                "Unable to calculate area under y with no guarantee of sort order on x.");
        }
        return sum;
    }

    // C# TrapezoidalAreaUnderX (lines ~685-703).
    double trapezoidal_area_under_x() const {
        double sum = 0.0;
        if (order_y_ == SortOrder::Ascending) {
            for (int i = 1; i < count(); ++i) {
                std::size_t ui = static_cast<std::size_t>(i);
                sum += 0.5 * (ordinates_[ui].y - ordinates_[ui - 1].y) *
                       (ordinates_[ui - 1].x + ordinates_[ui].x);
            }
        } else if (order_y_ == SortOrder::Descending) {
            for (int i = 1; i < count(); ++i) {
                std::size_t ui = static_cast<std::size_t>(i);
                sum += 0.5 * (ordinates_[ui - 1].y - ordinates_[ui].y) *
                       (ordinates_[ui - 1].x + ordinates_[ui].x);
            }
        } else {
            throw std::runtime_error(
                "Unable to calculate area under x with no guarantee of sort order on y.");
        }
        return sum;
    }

    // C# GetYFromX(double, Transform, Transform) (lines ~848-861).
    double get_y_from_x(double x, Transform x_transform = Transform::None,
                         Transform y_transform = Transform::None) const {
        if (count() == 0) return std::numeric_limits<double>::quiet_NaN();
        if (order_x_ == SortOrder::None)
            throw std::runtime_error("Interpolation requires the x-values to be ascending or descending.");
        if (count() == 1) return ordinates_[0].y;
        if ((order_x_ == SortOrder::Ascending && x <= ordinates_[0].x) ||
            (order_x_ == SortOrder::Descending && x >= ordinates_[0].x))
            return ordinates_[0].y;
        std::size_t last = static_cast<std::size_t>(count() - 1);
        if ((order_x_ == SortOrder::Ascending && x >= ordinates_[last].x) ||
            (order_x_ == SortOrder::Descending && x <= ordinates_[last].x))
            return ordinates_[last].y;
        return base_interpolate(x, search_x(x), true, x_transform, y_transform);
    }

    // C# GetXFromY(double, Transform, Transform) (lines ~870-882).
    double get_x_from_y(double y, Transform x_transform = Transform::None,
                         Transform y_transform = Transform::None) const {
        if (count() == 0) return std::numeric_limits<double>::quiet_NaN();
        if (order_y_ == SortOrder::None)
            throw std::runtime_error("Interpolation requires the y-values to be ascending or descending.");
        if (count() == 1) return ordinates_[0].x;
        if ((order_y_ == SortOrder::Ascending && y <= ordinates_[0].y) ||
            (order_y_ == SortOrder::Descending && y >= ordinates_[0].y))
            return ordinates_[0].x;
        std::size_t last = static_cast<std::size_t>(count() - 1);
        if ((order_y_ == SortOrder::Ascending && y >= ordinates_[last].y) ||
            (order_y_ == SortOrder::Descending && y <= ordinates_[last].y))
            return ordinates_[last].x;
        return base_interpolate(y, search_y(y), false, x_transform, y_transform);
    }

    // C# GetYFromX(IList<double>, Transform, Transform) (lines ~891-897).
    std::vector<double> get_y_from_x(const std::vector<double>& x_values,
                                      Transform x_transform = Transform::None,
                                      Transform y_transform = Transform::None) const {
        std::vector<double> result(x_values.size());
        for (std::size_t i = 0; i < x_values.size(); ++i)
            result[i] = get_y_from_x(x_values[i], x_transform, y_transform);
        return result;
    }

    // C# GetXFromY(IList<double>, Transform, Transform) (lines ~906-912).
    std::vector<double> get_x_from_y(const std::vector<double>& y_values,
                                      Transform x_transform = Transform::None,
                                      Transform y_transform = Transform::None) const {
        std::vector<double> result(y_values.size());
        for (std::size_t i = 0; i < y_values.size(); ++i)
            result[i] = get_x_from_y(y_values[i], x_transform, y_transform);
        return result;
    }

    // C# SearchX (lines ~923-937). Mutable cache fields (search-start memory), matching
    // Interpolater's own const-search-with-mutable-cache pattern (interpolater.hpp).
    int search_x(double x) const {
        int start = use_smart_search_ ? (x_correlated_ ? hunt_search_x(x) : bisection_search_x(x))
                                       : sequential_search_x(x);
        x_correlated_ = std::abs(start - x_search_start_) > x_delta_start_ ? false : true;
        x_search_start_ = (start < 0 || start >= count()) ? 0 : start;
        return start;
    }

    // C# SearchY (lines ~944-958).
    int search_y(double y) const {
        int start = use_smart_search_ ? (y_correlated_ ? hunt_search_y(y) : bisection_search_y(y))
                                       : sequential_search_y(y);
        y_correlated_ = std::abs(start - y_search_start_) > y_delta_start_ ? false : true;
        y_search_start_ = (start < 0 || start >= count()) ? 0 : start;
        return start;
    }

    // C# BinarySearchX (lines ~968-1010): .NET List<T>.BinarySearch convention -- the bitwise
    // complement (`~low`) of the insertion point when not found.
    int binary_search_x(double value) const {
        int low = 0, high = count() - 1;
        bool ascending = order_x_ == SortOrder::Ascending;
        while (low <= high) {
            int median = low + ((high - low) >> 1);
            double mv = ordinates_[static_cast<std::size_t>(median)].x;
            if (mv == value) return median;
            if (mv < value) {
                if (ascending) low = median + 1; else high = median - 1;
            } else {
                if (ascending) high = median - 1; else low = median + 1;
            }
        }
        return ~low;
    }

    // C# BinarySearchY (lines ~1020-1062).
    int binary_search_y(double value) const {
        int low = 0, high = count() - 1;
        bool ascending = order_y_ == SortOrder::Ascending;
        while (low <= high) {
            int median = low + ((high - low) >> 1);
            double mv = ordinates_[static_cast<std::size_t>(median)].y;
            if (mv == value) return median;
            if (mv < value) {
                if (ascending) low = median + 1; else high = median - 1;
            } else {
                if (ascending) high = median - 1; else low = median + 1;
            }
        }
        return ~low;
    }

    // C# SequentialSearchX (lines ~1067-1095).
    int sequential_search_x(double x) const {
        int jl = x_search_start_;
        std::size_t last = static_cast<std::size_t>(count() - 1);
        if ((order_x_ == SortOrder::Ascending && x < ordinates_[0].x) ||
            (order_x_ == SortOrder::Descending && x > ordinates_[0].x)) {
            return 0;
        } else if ((order_x_ == SortOrder::Ascending && x > ordinates_[last].x) ||
                   (order_x_ == SortOrder::Descending && x < ordinates_[last].x)) {
            return count() - 2;
        } else if ((order_x_ == SortOrder::Ascending &&
                    x < ordinates_[static_cast<std::size_t>(x_search_start_)].x) ||
                   (order_x_ == SortOrder::Descending &&
                    x > ordinates_[static_cast<std::size_t>(x_search_start_)].x)) {
            jl = 0;
        }
        for (int i = jl; i < count(); ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            if ((order_x_ == SortOrder::Ascending && x <= ordinates_[ui].x) ||
                (order_x_ == SortOrder::Descending && x >= ordinates_[ui].x)) {
                jl = i - 1;
                break;
            }
        }
        return jl;
    }

    // C# SequentialSearchY (lines ~1101-1129). See transcription note 2 above: the third branch
    // reads `_ordinates[XSearchStart].Y`, not `_ordinates[YSearchStart].Y` -- transcribed as-is.
    int sequential_search_y(double y) const {
        int jl = y_search_start_;
        std::size_t last = static_cast<std::size_t>(count() - 1);
        if ((order_y_ == SortOrder::Ascending && y < ordinates_[0].y) ||
            (order_y_ == SortOrder::Descending && y > ordinates_[0].y)) {
            return 0;
        } else if ((order_y_ == SortOrder::Ascending && y > ordinates_[last].y) ||
                   (order_y_ == SortOrder::Descending && y < ordinates_[last].y)) {
            return count() - 2;
        } else if ((order_y_ == SortOrder::Ascending &&
                    y < ordinates_[static_cast<std::size_t>(x_search_start_)].y) ||
                   (order_y_ == SortOrder::Descending &&
                    y > ordinates_[static_cast<std::size_t>(x_search_start_)].y)) {
            // Bug transcribed verbatim (note 2): x_search_start_, not y_search_start_.
            jl = 0;
        }
        for (int i = jl; i < count(); ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            if ((order_y_ == SortOrder::Ascending && y <= ordinates_[ui].y) ||
                (order_y_ == SortOrder::Descending && y >= ordinates_[ui].y)) {
                jl = i - 1;
                break;
            }
        }
        return jl;
    }

    // C# BisectionSearchX (lines ~1135-1148).
    int bisection_search_x(double x) const {
        int ju = count() - 1, jl = 0;
        bool ascnd = order_x_ == SortOrder::Ascending;
        while (ju - jl > 1) {
            int jm = (ju + jl) >> 1;
            if ((x >= ordinates_[static_cast<std::size_t>(jm)].x) == ascnd)
                jl = jm;
            else
                ju = jm;
        }
        return jl;
    }

    // C# BisectionSearchY (lines ~1154-1167).
    int bisection_search_y(double y) const {
        int ju = count() - 1, jl = 0;
        bool ascnd = order_y_ == SortOrder::Ascending;
        while (ju - jl > 1) {
            int jm = (ju + jl) >> 1;
            if ((y >= ordinates_[static_cast<std::size_t>(jm)].y) == ascnd)
                jl = jm;
            else
                ju = jm;
        }
        return jl;
    }

    // C# HuntSearchX (lines ~1173-1223).
    int hunt_search_x(double x) const {
        int jl = x_search_start_, ju, inc = 1;
        bool ascnd = order_x_ == SortOrder::Ascending;
        if (jl < 0 || jl > count() - 1) {
            jl = 0;
            ju = count() - 1;
        } else {
            if ((x >= ordinates_[static_cast<std::size_t>(jl)].x) == ascnd) {
                for (;;) {
                    ju = jl + inc;
                    if (ju >= count() - 1) {
                        ju = count() - 1;
                        break;
                    } else if ((x < ordinates_[static_cast<std::size_t>(ju)].x) == ascnd) {
                        break;
                    } else {
                        jl = ju;
                        inc += inc;
                    }
                }
            } else {
                ju = jl;
                for (;;) {
                    jl = jl - inc;
                    if (jl <= 0) {
                        jl = 0;
                        break;
                    } else if ((x >= ordinates_[static_cast<std::size_t>(jl)].x) == ascnd) {
                        break;
                    } else {
                        ju = jl;
                        inc += inc;
                    }
                }
            }
        }
        while (ju - jl > 1) {
            int jm = (ju + jl) >> 1;
            if ((x >= ordinates_[static_cast<std::size_t>(jm)].x) == ascnd)
                jl = jm;
            else
                ju = jm;
        }
        return jl;
    }

    // C# HuntSearchY (lines ~1229-1279).
    int hunt_search_y(double y) const {
        int jl = y_search_start_, ju, inc = 1;
        bool ascnd = order_y_ == SortOrder::Ascending;
        if (jl < 0 || jl > count() - 1) {
            jl = 0;
            ju = count() - 1;
        } else {
            if ((y >= ordinates_[static_cast<std::size_t>(jl)].y) == ascnd) {
                for (;;) {
                    ju = jl + inc;
                    if (ju >= count() - 1) {
                        ju = count() - 1;
                        break;
                    } else if ((y < ordinates_[static_cast<std::size_t>(ju)].y) == ascnd) {
                        break;
                    } else {
                        jl = ju;
                        inc += inc;
                    }
                }
            } else {
                ju = jl;
                for (;;) {
                    jl = jl - inc;
                    if (jl <= 0) {
                        jl = 0;
                        break;
                    } else if ((y >= ordinates_[static_cast<std::size_t>(jl)].y) == ascnd) {
                        break;
                    } else {
                        ju = jl;
                        inc += inc;
                    }
                }
            }
        }
        while (ju - jl > 1) {
            int jm = (ju + jl) >> 1;
            if ((y >= ordinates_[static_cast<std::size_t>(jm)].y) == ascnd)
                jl = jm;
            else
                ju = jm;
        }
        return jl;
    }

    // C# DouglasPeuckerSimplify (lines ~1290-1298).
    OrderedPairedData douglas_peucker_simplify(double tolerance) const {
        std::vector<Ordinate> ords;
        auto indexes = douglas_peucker_reduction(tolerance);
        ords.reserve(indexes.size());
        for (int idx : indexes) {
            const Ordinate& o = ordinates_[static_cast<std::size_t>(idx)];
            ords.emplace_back(o.x, o.y);
        }
        return OrderedPairedData(ords, strict_x_, order_x_, strict_y_, order_y_);
    }

    // C# VisvaligamWhyattSimplify (lines ~1398-1424). See transcription note 7 above: the guard
    // below throws where C#'s `List<T>` indexer would (`Count < 3`), rather than reading past the
    // end of `ords` the way the un-guarded translation did.
    OrderedPairedData visvaligam_whyatt_simplify(int num_to_keep) const {
        std::vector<Ordinate> ords(ordinates_);
        int remove_limit = static_cast<int>(ords.size()) - num_to_keep;
        for (int i = 0; i < remove_limit; ++i) {
            if (ords.size() < 3)
                throw std::out_of_range(
                    "visvaligam_whyatt_simplify: num_to_keep leaves fewer than 3 ordinates to "
                    "triangulate (C# List<T> indexer throws ArgumentOutOfRangeException here)");
            int min_index = 1;
            double min_area = triangle_area(ords[0], ords[1], ords[2]);
            for (int j = 2; j <= static_cast<int>(ords.size()) - 2; ++j) {
                double tmp_area = triangle_area(ords[static_cast<std::size_t>(j - 1)],
                                                 ords[static_cast<std::size_t>(j)],
                                                 ords[static_cast<std::size_t>(j + 1)]);
                if (tmp_area < min_area) {
                    min_index = j;
                    min_area = tmp_area;
                }
            }
            ords.erase(ords.begin() + min_index);
        }
        return OrderedPairedData(ords, strict_x_, order_x_, strict_y_, order_y_);
    }

    // C# LangSimplify (lines ~1445-1473). See transcription note 5 above for the guarded-return
    // decision: this port returns clone() rather than trying to alias `*this`.
    OrderedPairedData lang_simplify(double tolerance, int look_ahead) const {
        if (look_ahead <= 1 || tolerance <= 0) return clone();

        std::vector<Ordinate> ords;
        int n = count();
        int la = look_ahead;
        if (la > n - 1) la = n - 1;
        ords.push_back(ordinates_[0]);

        for (int i = 0; i < n; ++i) {
            if (i + la > n) la = n - i - 1;
            int offset = recursive_tolerance(i, la, tolerance);
            if (offset > 0 && (i + offset) < count()) {
                const Ordinate& o = ordinates_[static_cast<std::size_t>(i + offset)];
                ords.emplace_back(o.x, o.y);
                i += offset - 1;
            }
        }
        return OrderedPairedData(ords, strict_x_, order_x_, strict_y_, order_y_);
    }

   private:
    // C# private OrdinateValid(int, bool) (lines ~383-401).
    bool ordinate_valid(int index, bool look_backward = true) const {
        int n = count();
        if (index < 0 || index > n - 1) return true;
        if (look_backward && index > 0) {
            if (!ordinates_[static_cast<std::size_t>(index)].ordinate_valid(
                    ordinates_[static_cast<std::size_t>(index - 1)], strict_x_, strict_y_, order_x_,
                    order_y_, false))
                return false;
        }
        if (index < n - 1) {
            if (!ordinates_[static_cast<std::size_t>(index)].ordinate_valid(
                    ordinates_[static_cast<std::size_t>(index + 1)], strict_x_, strict_y_, order_x_,
                    order_y_, true))
                return false;
        }
        return true;
    }

    // C# private Validate() (lines ~363-375). Called with lookBackward=false: a single forward
    // sweep, i.e. each i is checked only against i+1.
    void validate() {
        is_valid_ = true;
        for (int i = 0; i < count(); ++i) {
            if (!ordinate_valid(i, false)) {
                is_valid_ = false;
                return;
            }
        }
    }

    // C# private BaseInterpolate (lines ~740-839).
    double base_interpolate(double value, int index, bool given_x, Transform x_transform,
                             Transform y_transform) const {
        double x = given_x ? value : 0.0;
        double y = given_x ? 0.0 : value;
        double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
        std::size_t lo = static_cast<std::size_t>(index);
        std::size_t hi = static_cast<std::size_t>(index + 1);

        if (x_transform == Transform::None) {
            x1 = ordinates_[lo].x;
            x2 = ordinates_[hi].x;
        } else if (x_transform == Transform::Logarithmic) {
            x = corehydro::numerics::clamped_log10(x);
            x1 = corehydro::numerics::clamped_log10(ordinates_[lo].x);
            x2 = corehydro::numerics::clamped_log10(ordinates_[hi].x);
        } else if (x_transform == Transform::NormalZ) {
            x = distributions::Normal::standard_z(x);
            x1 = distributions::Normal::standard_z(ordinates_[lo].x);
            x2 = distributions::Normal::standard_z(ordinates_[hi].x);
        }

        if (y_transform == Transform::None) {
            y1 = ordinates_[lo].y;
            y2 = ordinates_[hi].y;
        } else if (y_transform == Transform::Logarithmic) {
            y = corehydro::numerics::clamped_log10(y);
            y1 = corehydro::numerics::clamped_log10(ordinates_[lo].y);
            y2 = corehydro::numerics::clamped_log10(ordinates_[hi].y);
        } else if (y_transform == Transform::NormalZ) {
            y = distributions::Normal::standard_z(y);
            y1 = distributions::Normal::standard_z(ordinates_[lo].y);
            y2 = distributions::Normal::standard_z(ordinates_[hi].y);
        }

        if (given_x) {
            if ((x2 - x1) == 0) {
                y = y1;
            } else {
                y = y1 + (x - x1) / (x2 - x1) * (y2 - y1);
            }
            if (y_transform == Transform::None) return y;
            if (y_transform == Transform::Logarithmic) return std::pow(10.0, y);
            if (y_transform == Transform::NormalZ) return distributions::Normal::standard_cdf(y);
        } else {
            if ((y2 - y1) == 0) {
                x = x1;
            } else {
                x = x1 + (y - y1) / (y2 - y1) * (x2 - x1);
            }
            if (x_transform == Transform::None) return x;
            if (x_transform == Transform::Logarithmic) return std::pow(10.0, x);
            if (x_transform == Transform::NormalZ) return distributions::Normal::standard_cdf(x);
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    // C# private DouglasPeuckerReduction() (lines ~1311-1328): the no-args public-facing helper
    // that seeds the recursive reduction and sorts the kept-index list.
    std::vector<int> douglas_peucker_reduction(double tolerance) const {
        std::vector<int> reduced;
        int n = count();
        if (n < 3 || tolerance <= 0) {
            for (int i = 0; i <= n - 1; ++i) reduced.push_back(i);
            return reduced;
        }
        int first_point = 0;
        int last_point = n - 1;
        reduced = {first_point, last_point};
        douglas_peucker_reduction(first_point, last_point, tolerance, reduced);
        std::sort(reduced.begin(), reduced.end());
        return reduced;
    }

    // C# private DouglasPeuckerReduction(int, int, double, ref List<int>) (lines ~1337-1364).
    void douglas_peucker_reduction(int first_point, int last_point, double tolerance,
                                    std::vector<int>& indexes_to_keep) const {
        double max_distance = 0.0;
        int index_farthest = 0;
        if (first_point != (last_point - 1)) {
            for (int index = first_point; index <= last_point - 1; ++index) {
                double distance = perpendicular_distance(
                    ordinates_[static_cast<std::size_t>(first_point)].x,
                    ordinates_[static_cast<std::size_t>(first_point)].y,
                    ordinates_[static_cast<std::size_t>(last_point)].x,
                    ordinates_[static_cast<std::size_t>(last_point)].y,
                    ordinates_[static_cast<std::size_t>(index)].x,
                    ordinates_[static_cast<std::size_t>(index)].y);
                if (distance > max_distance) {
                    max_distance = distance;
                    index_farthest = index;
                }
            }
        }
        if (max_distance > tolerance && index_farthest != 0) {
            indexes_to_keep.push_back(index_farthest);
            douglas_peucker_reduction(first_point, index_farthest, tolerance, indexes_to_keep);
            douglas_peucker_reduction(index_farthest, last_point, tolerance, indexes_to_keep);
        }
    }

    // C# private PerpendicularDistance (lines ~1376-1386): triangle-area-over-base. See the
    // file header's "Also noted" paragraph for the (unexercised) degenerate-segment 0/0 case.
    static double perpendicular_distance(double a_x, double a_y, double b_x, double b_y, double c_x,
                                          double c_y) {
        double area = std::fabs((a_x * b_y + b_x * c_y + c_x * a_y - b_x * a_y - c_x * b_y - a_x * c_y) *
                                 0.5);
        double base_len = std::sqrt(std::pow(a_x - b_x, 2) + std::pow(a_y - b_y, 2));
        return area * 2 / base_len;
    }

    // C# private TriangleArea (lines ~1433-1436).
    static double triangle_area(const Ordinate& p1, const Ordinate& p2, const Ordinate& p3) {
        return std::fabs((p1.x * p2.y + p2.x * p3.y + p3.x * p1.y - p2.x * p1.y - p3.x * p2.y -
                           p1.x * p3.y) *
                          0.5);
    }

    // C# private RecursiveTolerance (lines ~1482-1518).
    int recursive_tolerance(int i, int look_ahead, double tolerance) const {
        int n = look_ahead;
        const Ordinate& cp = ordinates_[static_cast<std::size_t>(i)];
        if (i + n < count()) {
            Ordinate v1(ordinates_[static_cast<std::size_t>(i + n)].x - cp.x,
                        ordinates_[static_cast<std::size_t>(i + n)].y - cp.y);
            for (int j = 1; j <= n; ++j) {
                const Ordinate& clp = ordinates_[static_cast<std::size_t>(i + j)];
                Ordinate v2(clp.x - cp.x, clp.y - cp.y);
                double angle = std::acos((v1.x * v2.x + v1.y * v2.y) /
                                          (std::sqrt(v1.y * v1.y + v1.x * v1.x) *
                                           std::sqrt(v2.y * v2.y + v2.x * v2.x)));
                if (std::isnan(angle) || std::isinf(angle)) angle = 0.0;
                double lh = std::sqrt((clp.x - cp.x) * (clp.x - cp.x) + (clp.y - cp.y) * (clp.y - cp.y));
                if (std::sin(angle) * lh >= tolerance) {
                    n -= 1;
                    if (n > 0) return recursive_tolerance(i, n, tolerance);
                    return 0;
                }
            }
        }
        return n;
    }

    std::vector<Ordinate> ordinates_;
    bool strict_x_ = false;
    bool strict_y_ = false;
    SortOrder order_x_ = SortOrder::Ascending;
    SortOrder order_y_ = SortOrder::Ascending;
    bool is_valid_ = false;

    mutable int x_search_start_ = 0;
    mutable int y_search_start_ = 0;
    bool use_smart_search_ = true;
    // Transcription note 3 (see file header): never assigned past 0, matching C#.
    int x_delta_start_ = 0;
    int y_delta_start_ = 0;
    mutable bool x_correlated_ = false;
    mutable bool y_correlated_ = false;
};

// C# operator== (lines ~573-609): the null-reference branches have no C++ analogue (values, not
// nullable references, are compared here) and are omitted; the numerical comparison (1e-13
// absolute tolerance per coordinate, DIFFERENT from Ordinate's own kDoubleMachineEpsilon --
// this is the literal `0.0000000000001d` upstream wrote for this class specifically) is
// transcribed as-is.
inline bool operator==(const OrderedPairedData& left, const OrderedPairedData& right) {
    if (left.count() != right.count()) return false;
    for (int i = 0; i < left.count(); ++i) {
        if (std::fabs(left[i].x - right[i].x) > 0.0000000000001) return false;
        if (std::fabs(left[i].y - right[i].y) > 0.0000000000001) return false;
    }
    return true;
}

// C# operator!= (lines ~617-620).
inline bool operator!=(const OrderedPairedData& left, const OrderedPairedData& right) {
    return !(left == right);
}

}  // namespace corehydro::numerics::data::paired_data
