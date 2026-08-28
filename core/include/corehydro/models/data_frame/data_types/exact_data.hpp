// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataTypes/ExactData.cs @ fc28c0c
//
// Exact data ordinate: a precisely measured observation (systematic record).
//
// The DateTime member, its constructor overload and its property were deferred project-wide
// through P5 (they are reached only by the seasonal PointProcess path and by DataFrame's block
// series, both of which needed the unported TimeSeries container). P6 ported that container, so
// they land here.
//
// Deliberately NOT ported:
//   - ToXElement() and the XElement constructor (XML serialization)
//   - INotifyPropertyChanged / PropertyChanged (the C# IsLowOutlier setter raises it)
#pragma once
#include "corehydro/models/data_frame/data_types/data.hpp"
#include "corehydro/numerics/data/time_series/support/date_time.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class ExactData : public Data {
   public:
    // Construct an empty exact data ordinate (C# line 25).
    ExactData() = default;

    // Constructs a new exact data ordinate from a DATE (C# line 31): the ordinate's index is the
    // date's YEAR, and the date itself is carried alongside it for the seasonal path, which reads
    // its day-of-year.
    ExactData(const numerics::data::DateTime& date_time, double value)
        : Data(date_time.year(), value, 0.0), date_time_(date_time) {}

    // Constructs a new exact data ordinate (C# line 48).
    ExactData(int index, double value, double plotting_position = 0.0,
              bool is_low_outlier = false)
        : Data(index, value, plotting_position), is_low_outlier_(is_low_outlier) {}

    // --- DateTime: the date of the exact data value (C# line 87). Default-constructed (tick 0)
    // when the ordinate was built from a bare index, which is exactly the state
    // PointProcessModel::set_ams_data() tests for before substituting January 1 of the index
    // year. ---
    const numerics::data::DateTime& date_time() const { return date_time_; }

    // --- IsLowOutlier: whether the data value is a low outlier (C# line 93). ---
    bool is_low_outlier() const { return is_low_outlier_; }
    void set_is_low_outlier(bool is_low_outlier) { is_low_outlier_ = is_low_outlier; }

    // Validates the current state of the exact data and reports any issues found
    // (C# line 120).
    ValidationResult validate() const {
        ValidationResult result;

        if (index() < -100000 || index() > 100000) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The index must be between -100,000 and +100,000.");
        }

        if (!numerics::is_finite(value())) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The value must be a number.");
        }

        return result;
    }

    // Returns a copy of the data ordinate (C# line 143), carrying the date over exactly as the
    // C# object initializer does. Hides SeriesOrdinate::clone(), like the C# `new virtual`.
    ExactData clone() const {
        ExactData copy(index(), value(), plotting_position(), is_low_outlier());
        copy.date_time_ = date_time_;
        return copy;
    }

   private:
    bool is_low_outlier_ = false;
    numerics::data::DateTime date_time_;
};

}  // namespace corehydro::models
