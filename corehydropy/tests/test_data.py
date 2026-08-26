"""Behavioural tests for the data layer: AnalysisData, its summary, and threshold diagnostics.

Numeric oracles live in ``fixtures/`` (threshold_diagnostics.json and the censored estimation
fixtures); what is checked here is the Python surface -- spec assembly, argument validation, the
invariants a caller relies on, and agreement with the R package's identical surface.
"""

from __future__ import annotations

import math

import pytest

import corehydropy as ch

PEAKS = [12500.0, 15300.0, 8900.0, 22100.0, 18700.0, 14200.0, 9800.0, 28500.0, 17400.0,
         11600.0, 19200.0, 13800.0, 25600.0, 10500.0, 16900.0]


def test_analysis_data_assembles_each_series_indexed_from_zero():
    d = ch.analysis_data(
        exact=PEAKS,
        interval={"index": [15], "lower": [30000.0], "value": [35000.0], "upper": [40000.0]},
        threshold={"start_index": 15, "end_index": 54, "value": 28000.0, "number_above": 1},
        uncertain=[ch.Distribution("Normal", [14000, 2000])],
    )
    assert isinstance(d, ch.AnalysisData)
    assert len(d.spec["exact"]) == len(PEAKS)
    assert d.spec["exact"][0]["index"] == 0
    assert d.spec["exact"][-1]["index"] == len(PEAKS) - 1
    assert d.spec["exact"][0]["value"] == PEAKS[0]
    assert d.spec["interval"][0]["index"] == 15
    assert d.spec["threshold"][0]["number_above"] == 1
    assert d.spec["uncertain"][0]["distribution"]["family"] == "Normal"


def test_analysis_data_accepts_a_sequence_or_a_mapping_for_exact():
    assert ch.analysis_data(exact=PEAKS).spec == ch.analysis_data(exact={"value": PEAKS}).spec


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({}, "at least one of"),
        ({"interval": {"lower": [1], "upper": [3]}}, "`value` column"),
        ({"interval": {"lower": [5], "value": [1], "upper": [9]}}, "lower <= value <= upper"),
        (
            {"threshold": {"start_index": 5, "end_index": 1, "value": 1, "number_above": 0}},
            "end_index >= start_index",
        ),
        ({"exact": {"value": PEAKS, "index": [-1] * len(PEAKS)}}, "non-negative"),
        ({"exact": PEAKS, "low_outlier_threshold": 100, "mgbt_low_outliers": True},
         "do not also supply"),
    ],
)
def test_analysis_data_rejects_malformed_input(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        ch.analysis_data(**kwargs)


def test_analysis_data_rejects_a_non_distribution_uncertain_series():
    with pytest.raises(TypeError, match="Distribution"):
        ch.analysis_data(exact=PEAKS, uncertain=[1, 2])


def test_summary_reports_the_record_diagnostics():
    s = ch.analysis_data_summary(ch.analysis_data(PEAKS))
    assert s["value"] == PEAKS
    assert s["index"] == list(range(len(PEAKS)))
    assert s["exact_count"] == len(PEAKS)
    assert s["total_record_length"] == len(PEAKS)
    assert s["number_of_low_outliers"] == 0
    assert all(0 < p < 1 for p in s["plotting_position"])
    # Weibull positions on an uncensored record are rank / (n + 1).
    n = len(PEAKS)
    assert sorted(s["plotting_position"]) == pytest.approx([(i + 1) / (n + 1) for i in range(n)])


def test_a_plain_sequence_and_an_equivalent_frame_summarize_identically():
    assert ch.analysis_data_summary(PEAKS) == ch.analysis_data_summary(ch.analysis_data(PEAKS))


def test_censored_series_lengthen_the_record_and_shift_the_positions():
    plain = ch.analysis_data_summary(ch.analysis_data(PEAKS))
    censored = ch.analysis_data_summary(ch.analysis_data(
        exact=PEAKS,
        threshold={"start_index": 15, "end_index": 54, "value": 28000.0, "number_above": 1},
    ))
    assert censored["total_record_length"] == 55
    assert censored["threshold_count"] == 1
    assert censored["plotting_position"] != plain["plotting_position"]


def test_plotting_parameter_selects_the_convention_and_is_range_checked():
    weibull = ch.analysis_data_summary(ch.analysis_data(PEAKS))
    cunnane = ch.analysis_data_summary(ch.analysis_data(PEAKS), plotting_parameter=0.40)
    assert weibull["plotting_parameter"] == 0.0
    assert cunnane["plotting_parameter"] == 0.40
    assert weibull["plotting_position"] != cunnane["plotting_position"]
    for bad in (1.0, -0.1, math.nan):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            ch.analysis_data_summary(ch.analysis_data(PEAKS), plotting_parameter=bad)


def test_mgbt_censoring_flags_the_low_floods_and_matches_an_explicit_threshold():
    low = PEAKS + [120.0, 95.0]
    assert ch.mgbt_test(low) == 2

    s = ch.analysis_data_summary(ch.analysis_data(low, mgbt_low_outliers=True))
    assert s["number_of_low_outliers"] == 2
    assert s["low_outlier_threshold"] == 8900.0
    assert sum(s["is_low_outlier"]) == 2
    assert all(v < 8900.0 for v, f in zip(s["value"], s["is_low_outlier"]) if f)

    # An explicit threshold at the value MGBT chose must censor the same points.
    explicit = ch.analysis_data_summary(ch.analysis_data(low, low_outlier_threshold=8900.0))
    assert explicit["number_of_low_outliers"] == 2
    assert explicit["is_low_outlier"] == s["is_low_outlier"]


def test_a_threshold_censoring_more_than_half_the_record_is_rejected():
    with pytest.raises(Exception, match="50 percent"):
        ch.analysis_data_summary(ch.analysis_data(PEAKS, low_outlier_threshold=20000.0))


def test_threshold_diagnostics_returns_parallel_lists_per_method():
    x = ch.Distribution("GeneralizedPareto", [0, 100, 0.1]).random(300, seed=4242)

    mrl = ch.threshold_diagnostics(x, u_min=0, u_max=200, n_thresholds=8)
    n = len(mrl["threshold"])
    assert 0 < n <= 8
    assert len(mrl["mean_excess"]) == n
    assert len(mrl["exceedance_count"]) == n
    assert mrl["modified_scale"] == []
    assert all(lo < m for lo, m in zip(mrl["lower_ci"], mrl["mean_excess"]))
    assert all(hi > m for hi, m in zip(mrl["upper_ci"], mrl["mean_excess"]))
    counts = mrl["exceedance_count"]
    assert all(b <= a for a, b in zip(counts, counts[1:]))

    st = ch.threshold_diagnostics(x, u_min=0, u_max=200, n_thresholds=8,
                                  method="parameter_stability")
    assert len(st["shape"]) == len(st["threshold"])
    assert st["mean_excess"] == []
    assert all(lo < s for lo, s in zip(st["shape_lower_ci"], st["shape"]))
    # The sample is GPD(0, 100, 0.1), so the fitted shape should sit near 0.1.
    assert abs(st["shape"][0] - 0.1) < 0.15


def test_threshold_diagnostics_validates_its_method():
    with pytest.raises(ValueError, match="unknown method"):
        ch.threshold_diagnostics(PEAKS, 0, 1, method="nope")


def test_repr_shows_the_series_composition():
    assert "15 exact" in repr(ch.analysis_data(PEAKS))
    assert "MGBT" in repr(ch.analysis_data(PEAKS, mgbt_low_outliers=True))
    assert "9000" in repr(ch.analysis_data(PEAKS, low_outlier_threshold=9000.0))


def test_the_frame_round_trips_through_its_json_spec():
    import json

    d = ch.analysis_data(
        exact=PEAKS,
        interval={"index": [15], "lower": [30000.0], "value": [35000.0], "upper": [40000.0]},
    )
    assert json.loads(d.to_json()) == d.spec


# Harricana River annual peaks (Bobee & Ashkar 1991, Table 1.2, page 5); the oracle for
# mann_kendall (0.7757) is ExactDataHypothesisTests.Test_MannKendall, also transcribed into
# fixtures/data/data_frame_facades.json and core/tests/test_data_frame_facades.cpp.
HARRICANA = [
    122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225,
    174, 98.8, 149, 238, 262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172,
    153, 142, 317, 161, 201, 204, 194, 164, 183, 161, 167, 179, 185, 117, 192,
    337, 125, 166, 99.1, 202, 230, 158, 262, 154, 164, 182, 164, 183, 171, 250,
    184, 205, 237, 177, 239, 187, 180, 173, 174,
]


def test_analysis_data_hypothesis_test_reproduces_the_harricana_mann_kendall_oracle():
    r = ch.analysis_data_hypothesis_test(HARRICANA, "mann_kendall")
    assert r == {"mann_kendall": pytest.approx(0.7757, abs=1e-3)}


def test_analysis_data_hypothesis_test_accepts_an_analysis_data_frame():
    r_seq = ch.analysis_data_hypothesis_test(HARRICANA, "mann_kendall")
    r_obj = ch.analysis_data_hypothesis_test(ch.analysis_data(HARRICANA), "mann_kendall")
    assert r_seq == r_obj


def test_analysis_data_hypothesis_test_runs_a_two_sample_test_given_a_split_index():
    r = ch.analysis_data_hypothesis_test(HARRICANA, "mann_whitney", split_index=50)
    assert r["mann_whitney"] == pytest.approx(0.5892, abs=1e-2)


def test_analysis_data_hypothesis_test_validates_method_and_the_two_sample_split_index():
    with pytest.raises(ValueError, match="unknown method"):
        ch.analysis_data_hypothesis_test(HARRICANA, "not_a_method")
    with pytest.raises(ValueError, match="requires .split_index."):
        ch.analysis_data_hypothesis_test(HARRICANA, "equal_variance_t")


def test_analysis_data_statistics_returns_the_twenty_named_summary_statistics():
    s = ch.analysis_data_statistics(PEAKS)
    assert len(s["value"]) == 20
    assert list(s["value"].keys()) == [
        "Record Length", "Events Per Index (λ)", "Low Outliers", "Minimum", "Maximum",
        "Mean", "Std Dev", "Skewness", "Kurtosis", "Mean (of log)", "Std Dev (of log)",
        "Skewness (of log)", "Kurtosis (of log)", "1%", "5%", "25%", "50%", "75%", "95%", "99%",
    ]
    assert math.isfinite(s["value"]["Mean"])  # PEAKS has 15 >= 10 exact points


def test_analysis_data_statistics_all_data_uses_the_plotting_position_fit():
    exact_only = ch.analysis_data_statistics(HARRICANA)
    all_data = ch.analysis_data_statistics(HARRICANA, all_data=True)
    assert exact_only["value"]["Record Length"] == 69
    assert exact_only["value"]["Minimum"] == 98.8
    assert all_data["value"]["Minimum"] == 98.8
    # The two methods use different estimators, so their means need not agree even with no
    # censored data in the frame.
    assert exact_only["value"]["Mean"] != pytest.approx(all_data["value"]["Mean"])


def test_analysis_data_statistics_standardized_adds_parallel_columns():
    s = ch.analysis_data_statistics(HARRICANA, standardized=True)
    assert len(s["standardized_value"]) == len(HARRICANA)
    assert len(s["standardized_log10_value"]) == len(HARRICANA)
    assert all(math.isfinite(v) for v in s["standardized_value"])
    assert all(math.isfinite(v) for v in s["standardized_log10_value"])


def test_analysis_data_statistics_reports_nan_for_a_short_record():
    s = ch.analysis_data_statistics(HARRICANA[:9])
    assert all(math.isnan(v) for v in s["value"].values())
