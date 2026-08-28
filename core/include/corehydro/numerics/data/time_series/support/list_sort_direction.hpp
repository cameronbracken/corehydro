// corehydro ADDITION -- stands in for `System.ComponentModel.ListSortDirection`, the BCL enum
// `TimeSeries.SortByTime` / `SortByValue` take as their one argument. There is no upstream file
// to port; the member order and names are the BCL's. It lives beside the ported TimeSeries
// support enums because that is the only surface that uses it.
#pragma once

namespace corehydro::numerics::data {

enum class ListSortDirection {
    Ascending,
    Descending,
};

}  // namespace corehydro::numerics::data
