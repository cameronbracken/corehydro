// ported from: Numerics/Data/Interpolation/Support/Search.cs @ 2a0357a
//
// Two families of free functions live here. The `values` (plain double array) overloads --
// `sequential`/`bisection` below -- were ported early for the two callers this port already
// had: SNIS.cs's `Search.Sequential(rndOut[i], cdf, idx)` posterior-weight lookup during output
// resampling, and Histogram.cs's `GetBinIndexOf` bin lookup (see histogram.hpp). The
// OrderedPairedData and `vector<Ordinate>` overloads of `sequential`/`bisection`/`hunt` (C#
// lines 167, 254, 444, 545, 782, 925) were unported through that point because Paired Data
// (OrderedPairedData itself) did not exist yet in this port -- "no caller in this port's scope
// needs them" was true only because there was no OrderedPairedData to search. P4 Task 8 landed
// OrderedPairedData (ordered_paired_data.hpp) and ports all six of those overloads below,
// cross-checked in test_ordered_paired_data.cpp against OrderedPairedData's own
// search_x/search_y and *_search_x/*_search_y member methods on the same 1000-point curve. The
// `IList<double>` overload of Hunt (C# line 650) remains unported: neither SNIS nor Histogram
// needs a Hunt search over a plain double array, and OrderedPairedData's own hunt_search_x/y
// (member methods, ordered_paired_data.hpp) cover the Hunt-over-a-curve case this phase needed.
//
// Transcription note: the OrderedPairedData/Ordinate overloads of `hunt` below (C# lines 782,
// 925) use a COMPLETELY DIFFERENT indexing convention from every other search algorithm in this
// port (this file's own `values` overloads, OrderedPairedData's own hunt_search_x/y, and
// Interpolater's hunt_search below) -- `N = Count - 1` rather than `Count`, boundary sentinels
// at `N`/`N + 1` rather than `N - 1`/`N`, and the hunt-up/hunt-down loops re-check
// `data[xlo].x`/`data[xhi]` against the CURRENT (mutating) index each pass rather than the
// ju/jl-doubling shape the other Hunt variants share. This is transcribed exactly as upstream
// wrote it, not reconciled with the other two Hunt shapes in this port.
//
// This is a DIFFERENT algorithm from Interpolater's own sequential_search/
// bisection_search/hunt_search (ported in Phase 2 for the Linear/Bilinear path):
// Interpolater's variants are member functions with correlated-call search-start memory
// and no boundary-sentinel returns, while Search's are free functions returning
// -1/0/(N-1)/N sentinels for out-of-range/exact-endpoint queries. Phase 2 deliberately
// skipped this file entirely (see interpolater.hpp's header comment); this header ports
// the SNIS/Histogram-required subset of `values` overloads, plus (P4 Task 8) the full
// OrderedPairedData/Ordinate overload set.
//
// v2.1.4 sync (Numerics 33dc1af): FIXED, not mirrored -- bisection()'s loop condition used
// to be `x >= values[xm] && order == Ascending` (a logical AND against the order flag), not
// `(x >= values[xm]) == ascending` the way Interpolater's own bisection_search phrases the
// equivalent test. For order == Descending that condition was always false, so the loop
// only ever shrank `xhi`; `xlo` never advanced past `start`, so bisection() in descending
// order always returned `start` instead of the correct bracketing index. Upstream's fix
// splits the loop into separate ascending/descending branches (`x >= values[xm]` for
// ascending, `x < values[xm]` for descending) rather than adopting the equality-test
// phrasing Interpolater/Hunt already used; this port mirrors that same split. Previously
// dead code for every caller in this port's scope (Histogram::get_bin_index_of and SNIS
// both only ever call with the default Ascending order), now oracle-covered in the
// descending direction too via fixtures/special_functions/search.json's
// `*_descending_*` cases. See docs/upstream-csharp-issues.md (marked RESOLVED).
#pragma once
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "corehydro/numerics/data/paired_data/ordered_paired_data.hpp"
#include "corehydro/numerics/data/paired_data/ordinate.hpp"

namespace corehydro::numerics::data::search {

// Sequential (linear) search for the lower-bound index of x within `values`, scanning
// forward from `start`. Returns -1 if x is below the first value (above the first, for
// descending order); returns `values.size()` if x is beyond the far endpoint; returns the
// exact endpoint index (0 or N-1) on an exact match; otherwise the lower-bound index found
// by scanning forward from `start`.
inline int sequential(double x, const std::vector<double>& values, int start = 0,
                       SortOrder order = SortOrder::Ascending) {
    int n = static_cast<int>(values.size());
    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start >= n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (order == SortOrder::Ascending) {
        if (x < values[0]) return -1;
        if (x == values[0]) return 0;
        if (x == values[static_cast<std::size_t>(n - 1)]) return n - 1;
        if (x > values[static_cast<std::size_t>(n - 1)]) return n;
        for (int i = start; i < n; ++i)
            if (x <= values[static_cast<std::size_t>(i)]) return i - 1;
    } else {
        if (x > values[0]) return -1;
        if (x == values[0]) return 0;
        if (x == values[static_cast<std::size_t>(n - 1)]) return n - 1;
        if (x < values[static_cast<std::size_t>(n - 1)]) return n;
        for (int i = start; i < n; ++i)
            if (x >= values[static_cast<std::size_t>(i)]) return i - 1;
    }
    return 0;
}

// Bisection search for the lower-bound index of x within `values`; same boundary-sentinel
// conventions as sequential() above, halving the search bracket instead of scanning
// linearly. See the file header for a transcribed-verbatim quirk in descending order.
inline int bisection(double x, const std::vector<double>& values, int start = 0,
                      SortOrder order = SortOrder::Ascending) {
    int n = static_cast<int>(values.size());
    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start >= n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (order == SortOrder::Ascending) {
        if (x < values[0]) return -1;
        if (x == values[0]) return 0;
        if (x == values[static_cast<std::size_t>(n - 1)]) return n - 1;
        if (x > values[static_cast<std::size_t>(n - 1)]) return n;
    } else {
        if (x > values[0]) return -1;
        if (x == values[0]) return 0;
        if (x == values[static_cast<std::size_t>(n - 1)]) return n - 1;
        if (x < values[static_cast<std::size_t>(n - 1)]) return n;
    }

    int xlo = start, xhi = n;
    if (order == SortOrder::Ascending) {
        while (xhi - xlo > 1) {
            int xm = xlo + ((xhi - xlo) >> 1);
            if (x >= values[static_cast<std::size_t>(xm)])
                xlo = xm;
            else
                xhi = xm;
        }
    } else {
        while (xhi - xlo > 1) {
            int xm = xlo + ((xhi - xlo) >> 1);
            if (x < values[static_cast<std::size_t>(xm)])
                xlo = xm;
            else
                xhi = xm;
        }
    }
    return xlo;
}

// ---------------------------------------------------------------------------------------------
// P4 Task 8: the OrderedPairedData/Ordinate overloads (C# lines 167, 254, 444, 545, 782, 925).
// See the file header for the Hunt-indexing-convention transcription note.
// ---------------------------------------------------------------------------------------------

// C# Search.Sequential(double, OrderedPairedData, int) (C# line 167).
inline int sequential(double x, const paired_data::OrderedPairedData& data, int start = 0) {
    int n = data.count();
    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start >= n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (data.order_x() == SortOrder::Ascending) {
        if (x < data[0].x) return -1;
        if (x == data[0].x) return 0;
        if (x == data[n - 1].x) return n - 1;
        if (x > data[n - 1].x) return n;
        for (int i = start; i < n; ++i)
            if (x <= data[i].x) return i - 1;
    } else {
        if (x > data[0].x) return -1;
        if (x == data[0].x) return 0;
        if (x == data[n - 1].x) return n - 1;
        if (x < data[n - 1].x) return n;
        for (int i = start; i < n; ++i)
            if (x >= data[i].x) return i - 1;
    }
    return 0;
}

// C# Search.Sequential(double, IList<Ordinate>, int, SortOrder) (C# line 254).
inline int sequential(double x, const std::vector<paired_data::Ordinate>& ordinates, int start = 0,
                       SortOrder order = SortOrder::Ascending) {
    int n = static_cast<int>(ordinates.size());
    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start >= n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (order == SortOrder::Ascending) {
        if (x < ordinates[0].x) return -1;
        if (x == ordinates[0].x) return 0;
        if (x == ordinates[static_cast<std::size_t>(n - 1)].x) return n - 1;
        if (x > ordinates[static_cast<std::size_t>(n - 1)].x) return n;
        for (int i = start; i < n; ++i)
            if (x <= ordinates[static_cast<std::size_t>(i)].x) return i - 1;
    } else {
        if (x > ordinates[0].x) return -1;
        if (x == ordinates[0].x) return 0;
        if (x == ordinates[static_cast<std::size_t>(n - 1)].x) return n - 1;
        if (x < ordinates[static_cast<std::size_t>(n - 1)].x) return n;
        for (int i = start; i < n; ++i)
            if (x >= ordinates[static_cast<std::size_t>(i)].x) return i - 1;
    }
    return 0;
}

// C# Search.Bisection(double, OrderedPairedData, int) (C# line 444).
inline int bisection(double x, const paired_data::OrderedPairedData& data, int start = 0) {
    int n = data.count();
    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start >= n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (data.order_x() == SortOrder::Ascending) {
        if (x < data[0].x) return -1;
        if (x == data[0].x) return 0;
        if (x == data[n - 1].x) return n - 1;
        if (x > data[n - 1].x) return n;
    } else {
        if (x > data[0].x) return -1;
        if (x == data[0].x) return 0;
        if (x == data[n - 1].x) return n - 1;
        if (x < data[n - 1].x) return n;
    }

    int xlo = start, xhi = n;
    if (data.order_x() == SortOrder::Ascending) {
        while (xhi - xlo > 1) {
            int xm = xlo + ((xhi - xlo) >> 1);
            if (x >= data[xm].x)
                xlo = xm;
            else
                xhi = xm;
        }
    } else {
        while (xhi - xlo > 1) {
            int xm = xlo + ((xhi - xlo) >> 1);
            if (x < data[xm].x)
                xlo = xm;
            else
                xhi = xm;
        }
    }
    return xlo;
}

// C# Search.Bisection(double, IList<Ordinate>, int, SortOrder) (C# line 545).
inline int bisection(double x, const std::vector<paired_data::Ordinate>& ordinates, int start = 0,
                      SortOrder order = SortOrder::Ascending) {
    int n = static_cast<int>(ordinates.size());
    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start >= n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (order == SortOrder::Ascending) {
        if (x < ordinates[0].x) return -1;
        if (x == ordinates[0].x) return 0;
        if (x == ordinates[static_cast<std::size_t>(n - 1)].x) return n - 1;
        if (x > ordinates[static_cast<std::size_t>(n - 1)].x) return n;
    } else {
        if (x > ordinates[0].x) return -1;
        if (x == ordinates[0].x) return 0;
        if (x == ordinates[static_cast<std::size_t>(n - 1)].x) return n - 1;
        if (x < ordinates[static_cast<std::size_t>(n - 1)].x) return n;
    }

    int xlo = start, xhi = n;
    if (order == SortOrder::Ascending) {
        while (xhi - xlo > 1) {
            int xm = xlo + ((xhi - xlo) >> 1);
            if (x >= ordinates[static_cast<std::size_t>(xm)].x)
                xlo = xm;
            else
                xhi = xm;
        }
    } else {
        while (xhi - xlo > 1) {
            int xm = xlo + ((xhi - xlo) >> 1);
            if (x < ordinates[static_cast<std::size_t>(xm)].x)
                xlo = xm;
            else
                xhi = xm;
        }
    }
    return xlo;
}

// C# Search.Hunt(double, OrderedPairedData, int) (C# line 782). See the file header's
// transcription note: this uses the N = Count - 1 indexing convention, DIFFERENT from every
// other search algorithm in this port.
inline int hunt(double x_value, const paired_data::OrderedPairedData& data, int start = 0) {
    int n = data.count() - 1;
    int xlo, xhi;
    bool ascnd = data.order_x() == SortOrder::Ascending;

    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start > n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (data.order_x() == SortOrder::Ascending) {
        if (x_value < data[0].x) return -1;
        if (x_value == data[0].x) return 0;
        if (x_value == data[n].x) return n;
        if (x_value > data[n].x) return n + 1;
    } else {
        if (x_value > data[0].x) return -1;
        if (x_value == data[0].x) return 0;
        if (x_value == data[n].x) return n;
        if (x_value < data[n].x) return n + 1;
    }

    xlo = start;
    if (xlo <= 0 || xlo > n) {
        xlo = 0;
        xhi = n + 1;
    } else {
        int inc = 1;
        if ((x_value >= data[xlo].x) == ascnd) {
            xhi = xlo + 1;
            while ((x_value >= data[xhi].x) == ascnd) {
                xlo = xhi;
                inc += inc;
                xhi = xlo + inc;
                if (xhi > n) {
                    xhi = n + 1;
                    break;
                }
            }
        } else {
            xhi = xlo - 1;
            while ((x_value < data[xlo].x) == ascnd) {
                xhi = xlo;
                inc += inc;
                xlo = xhi - inc;
                if (xlo < 1) {
                    xlo = 0;
                    break;
                }
            }
        }
    }
    while (xhi - xlo > 1) {
        int xm = xlo + ((xhi - xlo) >> 1);
        if ((x_value >= data[xm].x) == ascnd)
            xlo = xm;
        else
            xhi = xm;
    }
    return xlo;
}

// C# Search.Hunt(double, IList<Ordinate>, int, SortOrder) (C# line 925). Same N = Count - 1
// indexing convention as the OrderedPairedData overload above.
inline int hunt(double x_value, const std::vector<paired_data::Ordinate>& ordinates, int start = 0,
                 SortOrder order = SortOrder::Ascending) {
    int n = static_cast<int>(ordinates.size()) - 1;
    int xlo, xhi;
    bool ascnd = order == SortOrder::Ascending;

    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start > n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (order == SortOrder::Ascending) {
        if (x_value < ordinates[0].x) return -1;
        if (x_value == ordinates[0].x) return 0;
        if (x_value == ordinates[static_cast<std::size_t>(n)].x) return n;
        if (x_value > ordinates[static_cast<std::size_t>(n)].x) return n + 1;
    } else {
        if (x_value > ordinates[0].x) return -1;
        if (x_value == ordinates[0].x) return 0;
        if (x_value == ordinates[static_cast<std::size_t>(n)].x) return n;
        if (x_value < ordinates[static_cast<std::size_t>(n)].x) return n + 1;
    }

    xlo = start;
    if (xlo <= 0 || xlo > n) {
        xlo = 0;
        xhi = n + 1;
    } else {
        int inc = 1;
        if ((x_value >= ordinates[static_cast<std::size_t>(xlo)].x) == ascnd) {
            xhi = xlo + 1;
            while ((x_value >= ordinates[static_cast<std::size_t>(xhi)].x) == ascnd) {
                xlo = xhi;
                inc += inc;
                xhi = xlo + inc;
                if (xhi > n) {
                    xhi = n + 1;
                    break;
                }
            }
        } else {
            xhi = xlo - 1;
            while ((x_value < ordinates[static_cast<std::size_t>(xlo)].x) == ascnd) {
                xhi = xlo;
                inc += inc;
                xlo = xhi - inc;
                if (xlo < 1) {
                    xlo = 0;
                    break;
                }
            }
        }
    }
    while (xhi - xlo > 1) {
        int xm = xlo + ((xhi - xlo) >> 1);
        if ((x_value >= ordinates[static_cast<std::size_t>(xm)].x) == ascnd)
            xlo = xm;
        else
            xhi = xm;
    }
    return xlo;
}

}  // namespace corehydro::numerics::data::search
