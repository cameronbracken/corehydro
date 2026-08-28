// ported from: Numerics/Data/Time Series/Support/Series.cs @ 2a0357a
//
// The abstract collection of `SeriesOrdinate<TIndex, TValue>` that `TimeSeries` derives from.
// Ported in P6 with the container itself; through P5 the repo carried a thin adapter with no
// mutation surface of this shape, so `upstream/CLAUDE.md` listed this file as "nothing to land
// on". That severance retires here -- v2.1.4 changed `Clear`/`RemoveAt` semantics and upstream
// has tests asserting them.
//
// Storage is BY VALUE (`std::vector<SeriesOrdinate<...>>`), where C# stores references to class
// instances. Every ported call site was checked for aliasing before that choice: each one either
// mutates through the indexer (`this[i].Value = ...`, `ShiftAllDates`'s `this[i].Index = ...`),
// which a `T&` from `operator[]` reproduces, or adds a `Clone()` (`CumulativeSum`,
// `ClipTimeSeries`, `PeaksOverThresholdSeries`) or a freshly constructed ordinate (the five
// block-series methods). No upstream path adds one ordinate instance to two series and then
// mutates it, so no aliasing is observable.
//
// Deliberately NOT ported (project-wide precedent, each a .NET collection or WPF concern with no
// numeric surface): `INotifyCollectionChanged` / the `CollectionChanged` event /
// `RaiseCollectionChanged` / `RaiseCollectionChangedReset` (the raisers are inert no-ops here,
// and `SuppressCollectionChanged` survives as a plain flag because callers set it and the C#
// reads it); the non-generic `IList` / `ICollection` / `IEnumerable` implementations and their
// `object?`-typed `Add`/`Insert`/`Remove`/`Contains`/`IndexOf`/`this[]` overloads; `SyncRoot` /
// `IsSynchronized` / `IsFixedSize` / `IsReadOnly`; and `CopyTo(Array, int)`.
#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

#include "corehydro/numerics/data/series_ordinate.hpp"

namespace corehydro::numerics::data {

template <typename TIndex, typename TValue>
class Series {
   public:
    using Ordinate = SeriesOrdinate<TIndex, TValue>;

    virtual ~Series() = default;

    // C# `Count`.
    int count() const { return static_cast<int>(series_ordinates_.size()); }

    // C# `this[int]`. The non-const overload returns a mutable reference because the container's
    // math region writes through it (`this[i].Value += constant`).
    Ordinate& operator[](int index) { return series_ordinates_[static_cast<std::size_t>(index)]; }
    const Ordinate& operator[](int index) const {
        return series_ordinates_[static_cast<std::size_t>(index)];
    }

    // C# `SuppressCollectionChanged`. The event is severed, so this only records intent; it is
    // ported because the container sets and clears it around every bulk mutation.
    bool suppress_collection_changed() const { return suppress_collection_changed_; }
    void set_suppress_collection_changed(bool value) { suppress_collection_changed_ = value; }

    // C# `Add(SeriesOrdinate)`.
    virtual void add(const Ordinate& item) { series_ordinates_.push_back(item); }

    // C# `Insert(int, SeriesOrdinate)`.
    virtual void insert(int index, const Ordinate& item) {
        series_ordinates_.insert(series_ordinates_.begin() + index, item);
    }

    // C# `Remove(SeriesOrdinate)`: removes the FIRST ordinate EQUAL to the argument (index and
    // value both), not the one with that identity, and reports whether it found one.
    virtual bool remove(const Ordinate& item) {
        int index = index_of(item);
        if (index < 0) return false;
        series_ordinates_.erase(series_ordinates_.begin() + index);
        return true;
    }

    // C# `RemoveAt(int)`. v2.1.4 made this remove the ordinate at the requested POSITION rather
    // than the first equal one, which is what `Test_RemoveAt_WithDuplicateOrdinates_
    // RemovesRequestedIndex` pins.
    virtual void remove_at(int index) {
        series_ordinates_.erase(series_ordinates_.begin() + index);
    }

    // C# `Clear()`. v2.1.4 clears the backing list in one operation (the previous
    // element-by-element removal was O(n^2) in the equality scans).
    void clear() { series_ordinates_.clear(); }

    // C# `Contains` / `IndexOf`, both by ordinate EQUALITY.
    bool contains(const Ordinate& item) const { return index_of(item) >= 0; }

    int index_of(const Ordinate& item) const {
        for (std::size_t i = 0; i < series_ordinates_.size(); ++i)
            if (series_ordinates_[i] == item) return static_cast<int>(i);
        return -1;
    }

    // Range-for support (C# `GetEnumerator`).
    typename std::vector<Ordinate>::iterator begin() { return series_ordinates_.begin(); }
    typename std::vector<Ordinate>::iterator end() { return series_ordinates_.end(); }
    typename std::vector<Ordinate>::const_iterator begin() const {
        return series_ordinates_.begin();
    }
    typename std::vector<Ordinate>::const_iterator end() const { return series_ordinates_.end(); }

    // First / last ordinate (C# `_seriesOrdinates.First()` / `.Last()`, which several container
    // methods call directly).
    Ordinate& first() { return series_ordinates_.front(); }
    const Ordinate& first() const { return series_ordinates_.front(); }
    Ordinate& last() { return series_ordinates_.back(); }
    const Ordinate& last() const { return series_ordinates_.back(); }

    // C# `ValuesToList` / `ValuesToArray`, one body here since C++ has a single vector type.
    std::vector<TValue> values_to_list() const {
        std::vector<TValue> out;
        out.reserve(series_ordinates_.size());
        for (const auto& o : series_ordinates_) out.push_back(o.value());
        return out;
    }
    std::vector<TValue> values_to_array() const { return values_to_list(); }

    // C# `IndexesToList` / `IndexesToArray`.
    std::vector<TIndex> indexes_to_list() const {
        std::vector<TIndex> out;
        out.reserve(series_ordinates_.size());
        for (const auto& o : series_ordinates_) out.push_back(o.index());
        return out;
    }
    std::vector<TIndex> indexes_to_array() const { return indexes_to_list(); }

   protected:
    // C# `protected List<SeriesOrdinate<TIndex, TValue>> _seriesOrdinates`.
    std::vector<Ordinate> series_ordinates_;
    bool suppress_collection_changed_ = false;
};

}  // namespace corehydro::numerics::data
