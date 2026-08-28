// P2 support ctest (C++-only): the THIN TimeSeries adapter. No upstream Test_TimeSeries
// value oracle is in scope (the 2,334-line container's heavy tests are out of scope);
// these are hand-computed oracles pinning the ported model-consumed surface.
//
// Ported member bodies governed by
//   upstream/Numerics/Numerics/Data/Time Series/TimeSeries.cs @ 2a0357a
//   (Difference 496-515, MinValue 1338, MeanValue 1370, StandardDeviation 1389) and its
//   Series base (ValuesToArray / ValuesToList 248-259).
#include <cmath>
#include <vector>

#include "corehydro/numerics/data/time_series/support/time_interval.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "check.hpp"

using corehydro::numerics::data::TimeInterval;
using corehydro::numerics::data::TimeSeries;

namespace {
// P6 index swap: the TimeSeries container indexes by DateTime rather than by an integer offset.
// Nothing in this suite reads the index for meaning (the ARMA families index by POSITION, the
// rating curve by date EQUALITY), so every series here is anchored at one arbitrary date; the
// mapping is monotone, so every oracle below is unchanged by the swap.
const corehydro::numerics::data::DateTime kT0(1900, 1, 1);


// Round-trip: build from a known vector, read values back in order.
void test_round_trip() {
    std::vector<double> data = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0};
    TimeSeries ts(TimeInterval::OneDay, kT0, data);
    CHECK_EQ(ts.count(), static_cast<int>(data.size()));

    std::vector<double> arr = ts.values_to_array();
    std::vector<double> lst = ts.values_to_list();
    CHECK_EQ(arr.size(), data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        CHECK_NEAR(arr[i], data[i], 1e-12);
        CHECK_NEAR(lst[i], data[i], 1e-12);
        CHECK_NEAR(ts[static_cast<int>(i)].value(), data[i], 1e-12);
    }
    // The index advances by one TIME INTERVAL per ordinate (real calendar math since P6).
    CHECK_TRUE(ts[0].index() == kT0);
    CHECK_TRUE(ts[5].index() == kT0.add_days(5.0));
}

// Difference(1,1): successive first differences; Difference(1,2): second differences.
void test_difference() {
    std::vector<double> data = {1.0, 4.0, 9.0, 16.0, 25.0};
    TimeSeries ts(TimeInterval::OneDay, kT0, data);

    TimeSeries d1 = ts.difference(1, 1);
    std::vector<double> e1 = {3.0, 5.0, 7.0, 9.0};  // successive diffs
    std::vector<double> a1 = d1.values_to_array();
    CHECK_EQ(a1.size(), e1.size());
    for (std::size_t i = 0; i < e1.size(); ++i) CHECK_NEAR(a1[i], e1[i], 1e-12);

    TimeSeries d2 = ts.difference(1, 2);
    std::vector<double> e2 = {2.0, 2.0, 2.0};  // second diffs
    std::vector<double> a2 = d2.values_to_array();
    CHECK_EQ(a2.size(), e2.size());
    for (std::size_t i = 0; i < e2.size(); ++i) CHECK_NEAR(a2[i], e2[i], 1e-12);

    // lag with lag=2, differences=1.
    TimeSeries dl = ts.difference(2, 1);
    std::vector<double> el = {8.0, 12.0, 16.0};  // data[i+2]-data[i]
    std::vector<double> al = dl.values_to_array();
    CHECK_EQ(al.size(), el.size());
    for (std::size_t i = 0; i < el.size(); ++i) CHECK_NEAR(al[i], el[i], 1e-12);
}

// Difference throws when length <= lag (TimeSeries.cs:501).
void test_difference_throws() {
    std::vector<double> data = {1.0, 2.0, 3.0};
    TimeSeries ts(TimeInterval::OneDay, kT0, data);
    CHECK_THROWS(ts.difference(3, 1));  // length (3) <= lag (3)
    CHECK_THROWS(ts.difference(1, 3));  // second pass shrinks below lag
}

// Summary stats on a known vector (population-consistent MeanValue; sample-denominator SD).
void test_summary_stats() {
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    TimeSeries ts(TimeInterval::OneDay, kT0, data);

    // Mean = 40/8 = 5.
    CHECK_NEAR(ts.mean_value(), 5.0, 1e-12);
    // Sample SD: sum sq dev = 32, /(n-1=7) = 4.571428..., sqrt = 2.13808993...
    double var = 32.0 / 7.0;
    CHECK_NEAR(ts.standard_deviation(), std::sqrt(var), 1e-9);
    // Min.
    CHECK_NEAR(ts.min_value(), 2.0, 1e-12);
}

// NaN handling: MeanValue / MinValue skip NaN (TimeSeries.cs:1338/1370).
void test_summary_stats_nan() {
    double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> data = {2.0, nan, 6.0};
    TimeSeries ts(TimeInterval::OneDay, kT0, data);
    CHECK_NEAR(ts.mean_value(), 4.0, 1e-12);  // (2+6)/2
    CHECK_NEAR(ts.min_value(), 2.0, 1e-12);
}

// Clone() is a deep copy: mutating a clone's ordinate does not touch the original.
void test_clone_deep_copy() {
    std::vector<double> data = {1.0, 2.0, 3.0};
    TimeSeries ts(TimeInterval::OneDay, kT0, data);
    TimeSeries cl = ts.clone();
    cl[1].set_value(99.0);
    CHECK_NEAR(cl[1].value(), 99.0, 1e-12);
    CHECK_NEAR(ts[1].value(), 2.0, 1e-12);  // original unchanged
    CHECK_EQ(cl.count(), ts.count());
}

}  // namespace

int main() {
    test_round_trip();
    test_difference();
    test_difference_throws();
    test_summary_stats();
    test_summary_stats_nan();
    test_clone_deep_copy();
    return chtest::summary("time_series");
}
