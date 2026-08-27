// ported from: Numerics/Data/Paired Data/UncertainOrderedPairedData.cs @ 2a0357a
//
// P4 Task 9: the uncertain twin of Task 8's OrderedPairedData (ordered_paired_data.hpp) -- an
// x-y curve container whose Y coordinate is a whole continuous distribution (UncertainOrdinate,
// uncertain_ordinate.hpp) rather than a number, kept sorted/validated against the same caller-
// chosen monotonicity contract (StrictX/OrderX, StrictY/OrderY), plus CurveSample (probability or
// mean) to collapse it down to a plain OrderedPairedData. TabularFunction (Task 9,
// tabular_function.hpp) is this class's only consumer.
//
// Severed (observable-collection / XML plumbing, project-wide precedent -- see
// ordered_paired_data.hpp's own header): the three XElement constructors, SaveToXElement(), the
// CollectionChanged event, and RaiseCollectionChangedReset. GetHashCode/Equals(object) are
// likewise not ported (operator== is the one equality entry point, matching Ordinate/
// OrderedPairedData's own precedent). CopyTo(UncertainOrdinate[], int) is plain ICollection<T>
// boilerplate, excluded on the same "not in this port's member surface" grounds
// ordered_paired_data.hpp's header documents for its own CopyTo -- Test_IList's CopyTo segment is
// skipped in the ctest for the same reason.
//
// THE TWO PAIRED-DATA CLASSES ARE DELIBERATELY ASYMMETRIC. Cross-referencing
// ordered_paired_data.hpp's five numbered notes against this class's actual C# source (not
// assumed parity):
//   - `add()` here only ever NARROWS IsValid (assigns false, never widens it back to true), unlike
//     the twin's Add, which unconditionally assigns `OrdinateValid(Count-1)` and can flip an
//     already-invalid collection back to valid.
//   - the private `ordinate_valid(int)` here returns FALSE for an out-of-range index, where the
//     twin's equivalent returns TRUE -- the twin's header calls this out explicitly as the
//     asymmetry's boundary case.
//   - `remove_at(int)` here has NO bounds guard at all (C# `UncertainOrderedPairedData.RemoveAt`,
//     lines ~596-603) -- it indexes/erases directly, matching plain `List<T>.RemoveAt`, which
//     throws `ArgumentOutOfRangeException` on a bad index. The twin's `remove_at` (Task 8) DOES
//     guard (`if (index < 0 || index >= count()) return;`) because the C# `OrderedPairedData.
//     RemoveAt` (lines 438-440) has that exact guard in its own source. Ported here as a
//     std::out_of_range throw (this port's ArgumentOutOfRangeException convention) rather than
//     the silent no-op the twin has, and rather than leaving std::vector::erase to hit undefined
//     behavior on a bad iterator.
//   - `remove_range(index, count)` here (C# lines ~610-624) has NEITHER of the twin's two bugs:
//     there is no off-by-one guard at all (again, relying on `List<T>.RemoveRange`'s own correct
//     ArgumentOutOfRangeException/ArgumentException behavior), so it is ported with the CORRECT
//     `index + count > Count` bounds test, throwing std::out_of_range on violation -- not the
//     twin's buggy `>=` silent-no-op guard. The `itemsRemoved` array the C# builds purely to feed
//     the severed CollectionChanged event is dropped, matching the twin's own precedent for
//     inert event-only bookkeeping.
//   - `remove_range(int[])` (C# lines ~630-658) has NO counterpart on the twin at all (the twin's
//     OrderedPairedData has only the one RemoveRange(int,int) overload) -- this is the "both
//     overloads" the task brief refers to, both unique to this class. Ported below with the
//     `isContinuous`/`removedItems` contiguity bookkeeping dropped (it exists solely to pick
//     which CollectionChanged event-args overload to raise, per the project-wide plumbing cull).
//
// CONSTRUCTION corehydro addition: `UncertainOrdinate` is move-only-but-copyable (its copy
// constructor deep-clones Y; see uncertain_ordinate.hpp's header note). `std::vector<
// UncertainOrdinate>`'s own copy constructor therefore ALREADY deep-clones every element, so this
// class's implicit copy constructor (never explicitly declared below) does exactly what the C#
// data constructors do by hand with `.Clone()` calls, for free. The one place this port still
// clones by hand is where the C# clones an EXTERNAL, non-owned distribution
// (`IList<UnivariateDistributionBase> yData`, constructor 2 below).
//
// Four numbered transcription notes required by the task brief:
//
// 1. `Validate()` (C# lines ~390-402) EARLY-RETURNS when `SuppressCollectionChanged` is true.
//    Every OTHER use of SuppressCollectionChanged in this class (guarding the
//    `CollectionChanged?.Invoke` calls) is pure observable-collection plumbing and is severed
//    project-wide, same as the twin. This ONE read of the flag is different: it changes WHEN
//    `_isValid` gets recomputed, a real, externally-observable effect (`IsValid` can go stale
//    while the flag is set), not merely whether an event fires. That is why it survives the
//    plumbing cull here as a plain bool (`suppress_collection_changed_`, with public
//    accessors) with the same early-return in `validate()`, rather than being dropped like
//    the rest of the flag's uses.
// 2/3. See uncertain_ordinate.hpp's own transcription notes 2 (the OrdinateValid/OrdinateErrors
//    mean-vs-median probe mismatch) and 3 (operator== exact-X-equality) -- both live on
//    UncertainOrdinate, not here, but are exercised through this class's own Validate/GetErrors/
//    operator==.
// 4. `AddRange` (C# lines ~677-690) computes `startIndex = Count - 1` BEFORE adding anything (an
//    off-by-one -- the correct pre-addition insertion point is `Count`, not `Count - 1`), but that
//    value is used ONLY as an argument to the severed CollectionChanged event, so it is genuinely
//    inert in this port -- ported as `add_range` below with no `startIndex` computation at all
//    (there is nothing left for it to feed). `InsertRange` (C# lines ~716-731) is DIFFERENT: its
//    `_isValid`-narrowing loop calls `OrdinateValid(index)` (the constant, FIRST inserted
//    position) on every one of its `items.Count` iterations, instead of `OrdinateValid(i)` (each
//    newly-inserted position in turn). Unlike AddRange's bug, this one is NOT inert -- it directly
//    computes `_isValid`, a real, reachable, externally-observable value, so mischaracterizing it
//    as "only feeds the severed event args" would be wrong. It is ported exactly as C# wrote it
//    (`ordinate_valid(index)`, not `ordinate_valid(i)`, on every pass of the loop below) because
//    this is a faithful port of an upstream defect, not a fix -- but the header comment on
//    `insert_range` says plainly that the consequence is real, not severed.
#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "corehydro/numerics/data/paired_data/ordered_paired_data.hpp"
#include "corehydro/numerics/data/paired_data/uncertain_ordinate.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"

namespace corehydro::numerics::data::paired_data {

class UncertainOrderedPairedData {
   public:
    // C# constructor 1 (lines ~159-171): empty instance with the monotonicity contract and
    // distribution type set up front; fields are assigned directly (not through the property
    // setters), so no validate() runs here, matching C#.
    UncertainOrderedPairedData(bool strict_on_x, SortOrder x_order, bool strict_on_y, SortOrder y_order,
                                distributions::UnivariateDistributionType distribution_type,
                                int capacity = 0) {
        if (capacity > 0) ordinates_.reserve(static_cast<std::size_t>(capacity));
        distribution_ = distribution_type;
        strict_x_ = strict_on_x;
        strict_y_ = strict_on_y;
        order_x_ = x_order;
        order_y_ = y_order;
    }

    // C# constructor 2 (lines ~183-196): parallel x-values/y-distributions arrays. `y_data` is
    // borrowed (not owned) -- each entry is deep-cloned into the new UncertainOrdinate, exactly
    // mirroring C#'s `yData[i].Clone()`.
    UncertainOrderedPairedData(const std::vector<double>& x_data,
                                const std::vector<const distributions::UnivariateDistributionBase*>& y_data,
                                bool strict_on_x, SortOrder x_order, bool strict_on_y, SortOrder y_order,
                                distributions::UnivariateDistributionType distribution_type) {
        if (x_data.size() != y_data.size())
            throw std::runtime_error("Number of X values and Y values must be the same.");
        distribution_ = distribution_type;
        strict_x_ = strict_on_x;
        strict_y_ = strict_on_y;
        order_x_ = x_order;
        order_y_ = y_order;
        ordinates_.reserve(x_data.size());
        for (std::size_t i = 0; i < x_data.size(); ++i)
            ordinates_.emplace_back(x_data[i], y_data[i]->clone());
        validate();
    }

    // C# constructor 3 (lines ~207-224): built from another uncertain-ordinate collection
    // (`IList<UncertainOrdinate> data` in C#; TabularFunction's IsDeterministic setter passes
    // itself). NOTE, an upstream quirk preserved as-is: `distribution_type` is recorded purely as
    // this collection's own `Distribution` metadata -- it does NOT convert `data`'s Y
    // distributions to the given type. Calling this with `Deterministic` while `data` holds
    // Triangular ordinates yields a collection labelled Deterministic whose ordinates are still
    // Triangular. This matches upstream exactly (see TabularFunction.cs's IsDeterministic setter,
    // ported at tabular_function.hpp) and is not "fixed" here.
    UncertainOrderedPairedData(const UncertainOrderedPairedData& data, bool strict_on_x, SortOrder x_order,
                                bool strict_on_y, SortOrder y_order,
                                distributions::UnivariateDistributionType distribution_type) {
        distribution_ = distribution_type;
        strict_x_ = strict_on_x;
        strict_y_ = strict_on_y;
        order_x_ = x_order;
        order_y_ = y_order;
        ordinates_.reserve(static_cast<std::size_t>(data.count()));
        for (int i = 0; i < data.count(); ++i) ordinates_.push_back(data[i]);  // deep-clones
        validate();
    }

    bool allow_different_distribution_types() const { return allow_different_distribution_types_; }
    void set_allow_different_distribution_types(bool value) {
        if (allow_different_distribution_types_ != value) {
            allow_different_distribution_types_ = value;
            validate();
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

    distributions::UnivariateDistributionType distribution() const { return distribution_; }

    int count() const { return static_cast<int>(ordinates_.size()); }

    // See transcription note 1 above.
    bool suppress_collection_changed() const { return suppress_collection_changed_; }
    void set_suppress_collection_changed(bool value) { suppress_collection_changed_ = value; }

    const UncertainOrdinate& operator[](int index) const {
        return ordinates_[static_cast<std::size_t>(index)];
    }

    // C# indexer setter (lines ~540-563), the C++ analogue (see ordered_paired_data.hpp's own
    // `set()` for the same pattern): operator[] has no set-through-index idiom without a proxy
    // object, so the setter's revalidation logic lives in this named method.
    void set(int index, UncertainOrdinate value) {
        std::size_t ui = static_cast<std::size_t>(index);
        if (ordinates_[ui] != value) {
            ordinates_[ui] = std::move(value);
            if (is_valid_) {
                if (!ordinate_valid(index)) is_valid_ = false;
            } else {
                if (ordinate_valid(index)) validate();
            }
        }
    }

    // C# CurveSample(double) (lines ~354-364).
    OrderedPairedData curve_sample(double probability) const {
        if (probability < 0.0) probability = 0.0;
        if (probability > 1.0) probability = 1.0;
        std::vector<Ordinate> result;
        result.reserve(ordinates_.size());
        for (const auto& o : ordinates_) result.push_back(o.get_ordinate(probability));
        return OrderedPairedData(result, strict_x_, order_x_, strict_y_, order_y_);
    }

    // C# CurveSample() (lines ~370-376).
    OrderedPairedData curve_sample() const {
        std::vector<Ordinate> result;
        result.reserve(ordinates_.size());
        for (const auto& o : ordinates_) result.push_back(o.get_ordinate());
        return OrderedPairedData(result, strict_x_, order_x_, strict_y_, order_y_);
    }

    // C# Clone() (lines ~382-385).
    UncertainOrderedPairedData clone() const {
        return UncertainOrderedPairedData(ordinates_, strict_x_, order_x_, strict_y_, order_y_,
                                           distribution_, is_valid_);
    }

    // C# Validate() (lines ~390-402). PUBLIC here, unlike the twin's private Validate() (matching
    // the C# source on both sides exactly). See transcription note 1 above for the
    // SuppressCollectionChanged early return.
    void validate() {
        if (suppress_collection_changed_) return;
        is_valid_ = true;
        for (int i = 0; i < count(); ++i) {
            if (!ordinate_valid(i, false)) {
                is_valid_ = false;
                return;
            }
        }
    }

    // C# GetErrors() (lines ~435-445).
    std::vector<std::string> get_errors() const {
        std::vector<std::string> result;
        if (is_valid_) return result;
        if (ordinates_.empty()) {
            // Faithful analogue of C#'s `_uncertainOrdinates.Last()` on an empty list (LINQ
            // throws InvalidOperationException: "Sequence contains no elements."), matching
            // ordered_paired_data.hpp's own precedent for this exact situation.
            throw std::runtime_error("Sequence contains no elements.");
        }
        for (int i = 0; i < count() - 1; ++i) {
            auto errs = ordinates_[static_cast<std::size_t>(i)].ordinate_errors(
                ordinates_[static_cast<std::size_t>(i + 1)], strict_x_, strict_y_, order_x_, order_y_,
                true, allow_different_distribution_types_);
            result.insert(result.end(), errs.begin(), errs.end());
        }
        auto last_errs = ordinates_.back().ordinate_errors();
        result.insert(result.end(), last_errs.begin(), last_errs.end());
        return result;
    }

    // C# IndexOf(UncertainOrdinate) (lines ~570-573).
    int index_of(const UncertainOrdinate& item) const {
        for (int i = 0; i < count(); ++i)
            if (ordinates_[static_cast<std::size_t>(i)] == item) return i;
        return -1;
    }

    // C# Remove(UncertainOrdinate) (lines ~580-590).
    bool remove(const UncertainOrdinate& item) {
        int idx = index_of(item);
        if (idx == -1) return false;
        ordinates_.erase(ordinates_.begin() + idx);
        validate();
        return true;
    }

    // C# RemoveAt(int) (lines ~596-603). See the class-level asymmetry note above: NO bounds
    // guard in C# (relies on List<T>.RemoveAt's own ArgumentOutOfRangeException), ported here as
    // an explicit std::out_of_range throw rather than undefined std::vector::erase behavior.
    void remove_at(int index) {
        if (index < 0 || index >= count()) throw std::out_of_range("Index was out of range.");
        ordinates_.erase(ordinates_.begin() + index);
        validate();
    }

    // C# RemoveRange(int, int) (lines ~610-624). See the class-level asymmetry note above: no
    // off-by-one guard exists in C# at all (unlike the twin's buggy `>=`), so this is ported with
    // the CORRECT `>` bound. The `itemsRemoved` array C# builds is dropped (fed only the severed
    // CollectionChanged event).
    void remove_range(int index, int cnt) {
        if (index < 0 || cnt < 0 || index + cnt > count())
            throw std::out_of_range("Index and count do not denote a valid range.");
        ordinates_.erase(ordinates_.begin() + index, ordinates_.begin() + index + cnt);
        validate();
    }

    // C# RemoveRange(int[]) (lines ~630-658). See the class-level asymmetry note above: this
    // overload has no counterpart on the twin at all. The `isContinuous`/`removedItems`
    // contiguity bookkeeping is dropped (it exists solely to pick a CollectionChanged event-args
    // overload).
    void remove_range(const std::vector<int>& row_indices_to_remove) {
        if (row_indices_to_remove.empty()) return;
        std::vector<int> sorted_indices = row_indices_to_remove;
        std::sort(sorted_indices.begin(), sorted_indices.end());
        for (int i = static_cast<int>(sorted_indices.size()) - 1; i >= 0; --i) {
            ordinates_.erase(ordinates_.begin() + sorted_indices[static_cast<std::size_t>(i)]);
        }
        validate();
    }

    // C# Add(UncertainOrdinate) (lines ~664-671). Only ever NARROWS IsValid -- see the
    // class-level asymmetry note above contrasting this with the twin's Add.
    void add(UncertainOrdinate item) {
        ordinates_.push_back(std::move(item));
        if (!ordinate_valid(count() - 1)) is_valid_ = false;
    }

    // C# AddRange(IList<UncertainOrdinate>) (lines ~677-690). See transcription note 4 above:
    // the C# `startIndex` computation is dropped entirely (genuinely inert -- it feeds only the
    // severed CollectionChanged event).
    void add_range(const std::vector<UncertainOrdinate>& items) {
        if (items.empty()) return;
        for (const auto& item : items) {
            ordinates_.push_back(item);  // deep-clones
            if (!ordinate_valid(count() - 1)) is_valid_ = false;
        }
    }

    // C# Insert(int, UncertainOrdinate) (lines ~697-709): only narrows IsValid (true -> possibly
    // false), matching Add's asymmetry against the twin.
    void insert(int index, UncertainOrdinate item) {
        ordinates_.insert(ordinates_.begin() + index, std::move(item));
        if (is_valid_) {
            if (!ordinate_valid(index)) is_valid_ = false;
        }
    }

    // C# InsertRange(int, IList<UncertainOrdinate>) (lines ~716-731). See transcription note 4
    // above: the loop below calls `ordinate_valid(index)` (the constant first-inserted position)
    // on every pass, NOT `ordinate_valid(i)` -- an upstream defect with a REAL consequence
    // (unlike AddRange's inert startIndex bug), ported exactly as C# wrote it.
    void insert_range(int index, const std::vector<UncertainOrdinate>& items) {
        if (items.empty()) return;
        ordinates_.insert(ordinates_.begin() + index, items.begin(), items.end());  // deep-clones
        for (int i = index; i <= index + static_cast<int>(items.size()) - 1; ++i) {
            if (is_valid_) {
                if (!ordinate_valid(index)) is_valid_ = false;
            }
        }
    }

    // C# Clear() (lines ~736-749) minus the SuppressCollectionChanged save/restore dance (no
    // event to suppress in this port).
    void clear() {
        if (ordinates_.empty()) return;
        ordinates_.clear();
        is_valid_ = true;
    }

    bool contains(const UncertainOrdinate& item) const { return index_of(item) != -1; }

   private:
    // C# private constructor (lines ~236-254), the Clone() helper: sets `_isValid` directly
    // rather than recomputing via Validate().
    UncertainOrderedPairedData(const std::vector<UncertainOrdinate>& data, bool strict_on_x,
                                SortOrder x_order, bool strict_on_y, SortOrder y_order,
                                distributions::UnivariateDistributionType distribution_type,
                                bool data_valid) {
        distribution_ = distribution_type;
        strict_x_ = strict_on_x;
        strict_y_ = strict_on_y;
        order_x_ = x_order;
        order_y_ = y_order;
        ordinates_.reserve(data.size());
        for (const auto& o : data) ordinates_.push_back(o);  // deep-clones
        is_valid_ = data_valid;
    }

    // C# private OrdinateValid(int, bool) (lines ~411-429). Returns FALSE for an out-of-range
    // index -- see the class-level asymmetry note above contrasting this with the twin.
    bool ordinate_valid(int index, bool look_backward = true) const {
        int n = count();
        if (index < 0 || index > n - 1) return false;
        if (look_backward && index > 0) {
            if (!ordinates_[static_cast<std::size_t>(index)].ordinate_valid(
                    ordinates_[static_cast<std::size_t>(index - 1)], strict_x_, strict_y_, order_x_,
                    order_y_, false, allow_different_distribution_types_))
                return false;
        }
        if (index < n - 1) {
            if (!ordinates_[static_cast<std::size_t>(index)].ordinate_valid(
                    ordinates_[static_cast<std::size_t>(index + 1)], strict_x_, strict_y_, order_x_,
                    order_y_, true, allow_different_distribution_types_))
                return false;
        }
        return true;
    }

    std::vector<UncertainOrdinate> ordinates_;
    bool strict_x_ = false;
    bool strict_y_ = false;
    SortOrder order_x_ = SortOrder::None;
    SortOrder order_y_ = SortOrder::None;
    bool is_valid_ = true;
    bool allow_different_distribution_types_ = false;
    bool suppress_collection_changed_ = false;
    distributions::UnivariateDistributionType distribution_ =
        distributions::UnivariateDistributionType::Deterministic;
};

// C# operator== (lines ~453-488), simplified for value semantics (no C# `null` to check).
inline bool operator==(const UncertainOrderedPairedData& l, const UncertainOrderedPairedData& r) {
    if (l.count() != r.count()) return false;
    for (int i = 0; i < l.count(); ++i) {
        if (l[i] != r[i]) return false;
    }
    return true;
}

// C# operator!= (lines ~496-499).
inline bool operator!=(const UncertainOrderedPairedData& l, const UncertainOrderedPairedData& r) {
    return !(l == r);
}

}  // namespace corehydro::numerics::data::paired_data
