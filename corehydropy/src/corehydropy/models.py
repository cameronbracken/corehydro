"""Constructors for the nine RMC.BestFit model families, and the verbs that evaluate them.

Mirrors :mod:`corehydror`'s ``model_*()`` surface exactly. A :class:`Model` is a plain spec
object holding the JSON that ``core/include/corehydro/models/model_spec.hpp`` parses; the core
rebuilds the model on every call, so nothing here holds a C++ object and a model can be printed,
pickled, and compared against the R package unchanged.
"""

from __future__ import annotations

import json

import numpy as np

from . import _core
from .data import AnalysisData
from .distributions import Distribution

__all__ = [
    "Model",
    "Trend",
    "ModelParameter",
    "trend",
    "model_parameter",
    "model_univariate",
    "model_mixture",
    "model_competing_risks",
    "model_point_process",
    "model_bulletin17c",
    "model_ar",
    "model_ma",
    "model_arima",
    "model_arimax",
    "model_spatial_gev",
    "model_rating_curve",
    "model_bivariate",
    "model_validate",
    "model_simulate",
    "model_log_likelihood",
    "model_parameters",
]

TREND_TYPES = (
    "Constant",
    "Cubic",
    "Exponential",
    "Linear",
    "Logistic",
    "Power",
    "Quadratic",
    "Reciprocal",
    "Sinusoidal",
    "StepFunction",
    "GeneralLinear",
)

# A handful of names a user is likelier to reach for than the library's own wording.
_PARAMETER_ALIASES = {
    "sd": "std dev",
    "stdev": "std dev",
    "std": "std dev",
    "sigma": "std dev",
    "mu": "mean",
    "alpha": "scale",
    "kappa": "shape",
    "xi": "location",
}


def _parameter_labels(family: str) -> list[str]:
    """Plain-word labels from the full names the core reports ("Mean (mu)" -> "mean")."""
    full = _core.dist_parameter_names(family)["full"]
    return [name.split("(")[0].strip().lower() for name in full]


def _resolve_parameter(parameter, family=None, what="parameter") -> int:
    """A name or 1-based position to the 0-based index the spec wants."""
    if isinstance(parameter, (int, np.integer)) and not isinstance(parameter, bool):
        if int(parameter) < 1:
            raise ValueError(f"`{what}` position must be a positive integer (1-based)")
        return int(parameter) - 1
    if not isinstance(parameter, str):
        raise TypeError(f"`{what}` must be a parameter name or a 1-based position")
    if family is None:
        raise ValueError(
            f"`{what}` was given as the name '{parameter}', but this model's parameter names "
            "are not known here (a trend widens the parameter vector); pass a 1-based "
            "position instead"
        )

    names = _core.dist_parameter_names(family)
    labels = _parameter_labels(family)
    wanted = _PARAMETER_ALIASES.get(parameter.strip().lower(), parameter.strip().lower())

    for candidates in ([s.lower() for s in names["short"]], [s.lower() for s in names["full"]], labels):
        if wanted in candidates:
            return candidates.index(wanted)
    raise ValueError(
        f"unknown {what} '{parameter}' for '{family}'; expected one of "
        f"{', '.join(labels)} (or a 1-based position)"
    )


class Trend:
    """A nonstationary trend on one distribution parameter.

    Attaching any trend makes a model nonstationary: the trended parameter is replaced in the
    parameter vector by the trend's own coefficients, so a linear trend on the location turns
    one location parameter into an intercept and a slope.

    Parameters
    ----------
    parameter : str or int
        The distribution parameter to trend, as a name (``"location"``) or a 1-based position.
    type : str
        One of ``Constant``, ``Linear``, ``Quadratic``, ``Cubic``, ``Exponential``,
        ``Logistic``, ``Power``, ``Reciprocal``, ``Sinusoidal``, ``StepFunction``, or
        ``GeneralLinear``.
    start_index : int, optional
        0-based index the trend is anchored at; the data-driven default is used when omitted.
    values : array_like, optional
        Trend coefficients, length matching the trend's own parameter count.

    Examples
    --------
    >>> trend("location", "Linear")
    <Trend Linear on parameter location>
    """

    def __init__(self, parameter, type: str, start_index=None, values=None) -> None:
        match = [t for t in TREND_TYPES if t.lower() == str(type).lower()]
        if not match:
            raise ValueError(
                f"unknown trend type '{type}'; expected one of {', '.join(TREND_TYPES)}"
            )
        self.parameter = parameter
        self.type = match[0]
        self.start_index = None if start_index is None else int(start_index)
        self.values = (
            None if values is None else [float(v) for v in np.asarray(values, float).ravel()]
        )

    def __repr__(self) -> str:
        return f"<Trend {self.type} on parameter {self.parameter}>"


class ModelParameter:
    """Bounds, a fixed flag, a prior, or a starting value for one model parameter.

    Supplying a prior replaces the model's default. Do that together with
    ``use_default_flat_priors=False`` on the model, so a later data assignment cannot overwrite
    what you set.

    Parameters
    ----------
    parameter : str or int
        The parameter to constrain, as a name (``"scale"``) or a 1-based position. In a model
        carrying trends the parameter vector is wider than the distribution's, so a position is
        required there.
    value : float, optional
        Starting value. A ``parameter_values`` vector on the model wins over this.
    lower, upper : float, optional
        Bounds used by the estimators and samplers.
    fixed : bool, optional
        Hold the parameter at its value instead of estimating it.
    prior : Distribution, optional
        The parameter's prior distribution.

    Examples
    --------
    >>> model_parameter("scale", lower=0.5, upper=40)
    <ModelParameter scale: lower = 0.5, upper = 40>
    """

    def __init__(self, parameter, value=None, lower=None, upper=None, fixed=None, prior=None):
        if prior is not None and not isinstance(prior, Distribution):
            raise TypeError("prior must be a Distribution object")
        if lower is not None and upper is not None and float(lower) > float(upper):
            raise ValueError("lower must not exceed upper")
        self.parameter = parameter
        self.value = None if value is None else float(value)
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)
        self.is_fixed = None if fixed is None else bool(fixed)
        self.prior = prior

    def __repr__(self) -> str:
        bits = []
        if self.value is not None:
            bits.append(f"value = {self.value:g}")
        if self.lower is not None:
            bits.append(f"lower = {self.lower:g}")
        if self.upper is not None:
            bits.append(f"upper = {self.upper:g}")
        if self.is_fixed:
            bits.append("fixed")
        if self.prior is not None:
            bits.append(f"prior = {self.prior.family}")
        detail = f": {', '.join(bits)}" if bits else ""
        return f"<ModelParameter {self.parameter}{detail}>"


def trend(parameter, type: str, start_index=None, values=None) -> Trend:
    """Construct a :class:`Trend`. See :class:`Trend` for the arguments."""
    return Trend(parameter, type, start_index, values)


def model_parameter(parameter, value=None, lower=None, upper=None, fixed=None, prior=None):
    """Construct a :class:`ModelParameter`. See :class:`ModelParameter` for the arguments."""
    return ModelParameter(parameter, value, lower, upper, fixed, prior)


class Model:
    """A model spec: the family, its data, and any trends, constraints, or fixed values.

    Build one with a ``model_*()`` constructor rather than directly. Pass it to any analysis in
    place of a plain data vector, or evaluate it with :func:`model_validate`,
    :func:`model_simulate`, :func:`model_log_likelihood`, and :func:`model_parameters`.
    """

    def __init__(self, spec: dict, dataset=None) -> None:
        self.spec = spec
        self.dataset = [] if dataset is None else list(dataset)

    def to_json(self) -> str:
        """The model as the JSON spec the shared C++ core parses."""
        return json.dumps(self.spec)

    def __repr__(self) -> str:
        label = self.spec.get("type", "univariate_distribution")
        if label == "time_series":
            label = f"time_series/{self.spec['subtype']}"
        detail = self.spec.get("family")
        if detail is None and "families" in self.spec:
            detail = " + ".join(self.spec["families"])
        head = f"<Model {label}" + (f": {detail}" if detail else "")
        bits = []
        if "data_frame" in self.spec:
            counts = [
                f"{len(self.spec['data_frame'][k])} {k}"
                for k in ("exact", "interval", "threshold", "uncertain")
                if k in self.spec["data_frame"]
            ]
            bits.append(", ".join(counts))
        elif self.dataset:
            bits.append(f"{len(self.dataset)} exact")
        if "trends" in self.spec:
            bits.append(f"{len(self.spec['trends'])} trends")
        if "parameters" in self.spec:
            bits.append(f"{len(self.spec['parameters'])} constraints")
        return head + (f" ({'; '.join(bits)})" if bits else "") + ">"


def _data_block(data) -> tuple[dict, list]:
    """The shared data block.

    A plain sequence emits ``dataset``, taking the vector-constructor path the analyses have
    always used; an :class:`AnalysisData` frame emits the full ``data_frame`` object. The two are
    equivalent for an uncensored record, and only the frame can carry censored series.
    """
    if isinstance(data, AnalysisData):
        return {"data_frame": data.spec}, []
    arr = np.asarray(data, dtype=float).ravel()
    return {"dataset": "data"}, [float(v) for v in arr]


def _build(
    spec: dict,
    dataset,
    trends=None,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
    family=None,
) -> Model:
    if trends is not None:
        if isinstance(trends, Trend):
            trends = [trends]
        if not all(isinstance(t, Trend) for t in trends):
            raise TypeError("trends must be Trend objects")
        spec["trends"] = []
        for t in trends:
            entry = {
                "parameter": _resolve_parameter(t.parameter, family, "trend parameter"),
                "type": t.type,
            }
            if t.start_index is not None:
                entry["start_index"] = t.start_index
            if t.values is not None:
                entry["values"] = t.values
            spec["trends"].append(entry)

    if parameters is not None:
        if isinstance(parameters, ModelParameter):
            parameters = [parameters]
        if not all(isinstance(p, ModelParameter) for p in parameters):
            raise TypeError("parameters must be ModelParameter objects")
        # A trended model's vector is wider than the distribution's, so a name cannot be
        # resolved from the family alone; require a position there rather than mis-targeting.
        resolve_family = family if "trends" not in spec else None
        spec["parameters"] = []
        for p in parameters:
            entry = {"index": _resolve_parameter(p.parameter, resolve_family, "parameter")}
            if p.value is not None:
                entry["value"] = p.value
            if p.lower is not None:
                entry["lower"] = p.lower
            if p.upper is not None:
                entry["upper"] = p.upper
            if p.is_fixed is not None:
                entry["is_fixed"] = p.is_fixed
            if p.prior is not None:
                entry["prior"] = {
                    "family": p.prior.family,
                    "parameters": list(p.prior.params),
                }
            spec["parameters"].append(entry)

    if use_default_flat_priors is not None:
        spec["use_default_flat_priors"] = bool(use_default_flat_priors)
    if parameter_values is not None:
        spec["parameter_values"] = [
            float(v) for v in np.asarray(parameter_values, float).ravel()
        ]
    return Model(spec, dataset)


def _check_model(model) -> Model:
    if not isinstance(model, Model):
        raise TypeError("model must be a Model from one of the model_*() constructors")
    return model


# --- The nine model families ------------------------------------------------------------------


def model_univariate(
    family: str,
    data,
    trends=None,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """A single distribution fit to a record.

    Optionally censored (via :class:`~corehydropy.data.AnalysisData`) and optionally
    nonstationary (via :func:`trend`). This is the workhorse model behind
    :func:`~corehydropy.univariate_analysis`.

    Parameters
    ----------
    family : str
        Distribution family name; see :func:`~corehydropy.distribution_names`.
    data : AnalysisData or array_like
        Observations, or a frame carrying censored observations.
    trends : Trend or sequence of Trend, optional
        Makes the model nonstationary.
    parameters : ModelParameter or sequence of ModelParameter, optional
        Bounds, fixed flags, priors, or starting values.
    parameter_values : array_like, optional
        All parameter values, applied last.
    use_default_flat_priors : bool, optional
        Set ``False`` alongside a custom prior.

    Returns
    -------
    Model
        The assembled model spec.

    Examples
    --------
    >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600]
    >>> model_univariate("LogPearsonTypeIII", peaks)
    <Model univariate_distribution: LogPearsonTypeIII (10 exact)>
    >>> model_univariate("Normal", peaks, trends=trend("mean", "Linear"))
    <Model univariate_distribution: Normal (10 exact; 1 trends)>
    """
    block, dataset = _data_block(data)
    spec = {"type": "univariate_distribution", "family": str(family), **block}
    return _build(
        spec, dataset, trends, parameters, parameter_values, use_default_flat_priors,
        family=str(family),
    )


def model_mixture(
    families,
    data,
    zero_inflated: bool = False,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """A weighted mixture of two or three component distributions.

    For records generated by more than one flood mechanism. Optionally zero-inflated.

    Parameters
    ----------
    families : sequence of str
        Component distribution family names.
    data : AnalysisData or array_like
        Observations.
    zero_inflated : bool, default False
        Add a zero-inflation component.
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """
    block, dataset = _data_block(data)
    spec = {
        "type": "mixture",
        "families": [str(f) for f in families],
        "zero_inflated": bool(zero_inflated),
        **block,
    }
    return _build(spec, dataset, None, parameters, parameter_values, use_default_flat_priors)


def model_competing_risks(
    families, data, parameters=None, parameter_values=None, use_default_flat_priors=None
) -> Model:
    """The maximum of several independent flood-generating processes.

    Parameters
    ----------
    families : sequence of str
        Component distribution family names.
    data : AnalysisData or array_like
        Observations.
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """
    block, dataset = _data_block(data)
    spec = {"type": "competing_risks", "families": [str(f) for f in families], **block}
    return _build(spec, dataset, None, parameters, parameter_values, use_default_flat_priors)


def model_point_process(
    data,
    threshold=None,
    total_years=None,
    use_defaults: bool = True,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """A non-seasonal point process over exceedances of a threshold.

    Combines an arrival rate with a magnitude distribution.

    Parameters
    ----------
    data : AnalysisData or array_like
        Observations.
    threshold : float, optional
        Exceedance threshold; derived from the data when omitted. See
        :func:`~corehydropy.data.threshold_diagnostics` for choosing one.
    total_years : float, optional
        Record length in years, used for the arrival rate.
    use_defaults : bool, default True
        Let the model derive its threshold and record length from the data. Applied before an
        explicit ``threshold`` or ``total_years``, so an explicit value wins.
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """
    block, dataset = _data_block(data)
    spec = {"type": "point_process", "use_defaults": bool(use_defaults), **block}
    if threshold is not None:
        spec["threshold"] = float(threshold)
    if total_years is not None:
        spec["total_years"] = float(total_years)
    return _build(spec, dataset, None, parameters, parameter_values, use_default_flat_priors)


def model_bulletin17c(
    data, family: str = "LogPearsonTypeIII", parameters=None, parameter_values=None
) -> Model:
    """The Bulletin 17C log-Pearson Type III flood-frequency model.

    Fit by the generalized method of moments. Handles the censored observation types directly,
    which is the point of using it over a plain :func:`model_univariate`.

    Parameters
    ----------
    data : AnalysisData or array_like
        Annual peaks.
    family : str, default "LogPearsonTypeIII"
        One of ``LogPearsonTypeIII``, ``Exponential``, ``Gamma``, ``LogNormal``, ``Normal``,
        or ``PearsonTypeIII``.
    parameters, parameter_values
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """
    block, dataset = _data_block(data)
    spec = {"type": "bulletin17c", "family": str(family), **block}
    return _build(spec, dataset, None, parameters, parameter_values, None, family=str(family))


def _time_series(
    subtype,
    data,
    orders,
    include_intercept,
    transform,
    extra=None,
    training_time_steps=None,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    if isinstance(data, AnalysisData):
        raise TypeError(
            "time-series models take a plain sequence in order, not an AnalysisData frame"
        )
    spec = {
        "type": "time_series",
        "subtype": subtype,
        "data": [float(v) for v in np.asarray(data, float).ravel()],
        "orders": orders,
        "include_intercept": bool(include_intercept),
    }
    if transform is not None:
        spec["transform"] = str(transform)
    if training_time_steps is not None:
        spec["training_time_steps"] = int(training_time_steps)
    if extra:
        spec.update(extra)
    return _build(spec, [], None, parameters, parameter_values, use_default_flat_priors)


def model_ar(
    data,
    p: int = 1,
    include_intercept: bool = True,
    transform=None,
    training_time_steps=None,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """An autoregressive AR(p) model.

    Parameters
    ----------
    data : array_like
        The observed series, in sequence order.
    p : int, default 1
        Autoregressive order.
    include_intercept : bool, default True
        Include an intercept term.
    transform : {"None", "Logarithmic", "BoxCox", "YeoJohnson"}, optional
        Variance-stabilizing transform.
    training_time_steps : int, optional
        Leading steps used for calibration, the rest held back for validation. The model
        default is ``max(30, floor(0.8 * n))``, which exceeds the series length for any series
        shorter than 30 and then fails :func:`model_validate`; set it explicitly for a short
        series.
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """
    return _time_series(
        "ar", data, {"p": int(p)}, include_intercept, transform,
        training_time_steps=training_time_steps,
        parameters=parameters, parameter_values=parameter_values,
        use_default_flat_priors=use_default_flat_priors,
    )


def model_ma(
    data,
    q: int = 1,
    include_intercept: bool = True,
    transform=None,
    training_time_steps=None,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """A moving-average MA(q) model.

    Parameters
    ----------
    data : array_like
        The observed series, in sequence order.
    q : int, default 1
        Moving-average order.
    include_intercept : bool, default True
        Include an intercept term.
    transform : {"None", "Logarithmic", "BoxCox", "YeoJohnson"}, optional
        Variance-stabilizing transform.
    training_time_steps : int, optional
        Leading steps used for calibration, the rest held back for validation. The model
        default is ``max(30, floor(0.8 * n))``, which exceeds the series length for any series
        shorter than 30 and then fails :func:`model_validate`; set it explicitly for a short
        series.
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """
    return _time_series(
        "ma", data, {"q": int(q)}, include_intercept, transform,
        training_time_steps=training_time_steps,
        parameters=parameters, parameter_values=parameter_values,
        use_default_flat_priors=use_default_flat_priors,
    )


def model_arima(
    data,
    p: int = 1,
    d: int = 0,
    q: int = 1,
    include_intercept: bool = True,
    transform=None,
    training_time_steps=None,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """An ARIMA(p, d, q) model.

    Parameters
    ----------
    data : array_like
        The observed series, in sequence order.
    p, d, q : int
        Autoregressive, differencing, and moving-average orders.
    include_intercept : bool, default True
        Include an intercept term.
    transform : {"None", "Logarithmic", "BoxCox", "YeoJohnson"}, optional
        Variance-stabilizing transform.
    training_time_steps : int, optional
        Leading steps used for calibration, the rest held back for validation. The model
        default is ``max(30, floor(0.8 * n))``, which exceeds the series length for any series
        shorter than 30 and then fails :func:`model_validate`; set it explicitly for a short
        series.
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """
    return _time_series(
        "arima", data, {"p": int(p), "d": int(d), "q": int(q)}, include_intercept, transform,
        training_time_steps=training_time_steps,
        parameters=parameters, parameter_values=parameter_values,
        use_default_flat_priors=use_default_flat_priors,
    )


def model_arimax(
    data,
    covariates,
    p: int = 1,
    d: int = 0,
    q: int = 0,
    b: int = 0,
    include_intercept: bool = True,
    transform=None,
    trend_type=None,
    include_seasonality=None,
    training_time_steps=None,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """An ARIMAX(p, d, q, b) model with covariates.

    Parameters
    ----------
    data : array_like
        The observed series, in sequence order.
    covariates : sequence of array_like or ndarray
        One covariate series per element, or a 2-D array with one covariate per column.
    p, d, q, b : int
        Autoregressive, differencing, moving-average, and covariate-lag orders.
    include_intercept : bool, default True
        Include an intercept term.
    transform : {"None", "Logarithmic", "BoxCox", "YeoJohnson"}, optional
        Variance-stabilizing transform.
    trend_type : {"None", "Linear", "Quadratic", "Cubic"}, optional
        Deterministic trend.
    include_seasonality : bool, optional
        Include seasonal terms.
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """
    arr = np.asarray(covariates, dtype=float)
    if arr.ndim == 2 and not isinstance(covariates, (list, tuple)):
        series = [arr[:, j].tolist() for j in range(arr.shape[1])]
    else:
        series = [
            [float(v) for v in np.asarray(z, float).ravel()] for z in covariates
        ]
    if not series:
        raise ValueError("covariates must be a non-empty sequence of series, or a 2-D array")
    extra: dict = {"covariates": series}
    if trend_type is not None:
        extra["trend"] = str(trend_type)
    if include_seasonality is not None:
        extra["include_seasonality"] = bool(include_seasonality)
    return _time_series(
        "arimax", data, {"p": int(p), "d": int(d), "q": int(q), "b": int(b)},
        include_intercept, transform, extra=extra,
        training_time_steps=training_time_steps,
        parameters=parameters, parameter_values=parameter_values,
        use_default_flat_priors=use_default_flat_priors,
    )


def model_spatial_gev(
    coordinates,
    at_site_data,
    use_copula_dependence=None,
    use_location_errors=None,
    use_scale_errors=None,
    use_shape_errors=None,
    use_log_link_for_location=None,
    use_log_link_for_scale=None,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """The Renard hierarchical spatial GEV model.

    At-site GEV distributions whose parameters are themselves regressions over site coordinates.

    Parameters
    ----------
    coordinates : array_like
        A sites-by-2 array of site coordinates.
    at_site_data : array_like
        An observations-by-sites array of the at-site records.
    use_copula_dependence, use_location_errors, use_scale_errors, use_shape_errors : bool, optional
        Gate the spatial-dependence and regression-error terms.
    use_log_link_for_location, use_log_link_for_scale : bool, optional
        Select a log link for the corresponding regression.
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """

    def rows(m, what):
        arr = np.asarray(m, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"`{what}` must be a 2-D array")
        return [arr[i, :].tolist() for i in range(arr.shape[0])]

    spec = {
        "type": "spatial_gev",
        "coordinates": rows(coordinates, "coordinates"),
        "at_site_data": rows(at_site_data, "at_site_data"),
    }
    for key, value in (
        ("use_copula_dependence", use_copula_dependence),
        ("use_location_errors", use_location_errors),
        ("use_scale_errors", use_scale_errors),
        ("use_shape_errors", use_shape_errors),
        ("use_log_link_for_location", use_log_link_for_location),
        ("use_log_link_for_scale", use_log_link_for_scale),
    ):
        if value is not None:
            spec[key] = bool(value)
    return _build(spec, [], None, parameters, parameter_values, use_default_flat_priors)


def model_rating_curve(
    stage,
    discharge,
    segments: int = 1,
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """A BaRatin stage-discharge rating curve.

    A hydraulically-controlled stage-discharge relationship of one to three segments, with a
    log10-space normal residual likelihood.

    Parameters
    ----------
    stage, discharge : array_like
        Aligned stage and discharge observations.
    segments : int, default 1
        Number of hydraulic control segments (1 to 3).
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """
    stage = [float(v) for v in np.asarray(stage, float).ravel()]
    discharge = [float(v) for v in np.asarray(discharge, float).ravel()]
    if len(stage) != len(discharge):
        raise ValueError("stage and discharge must be the same length")
    spec = {
        "type": "rating_curve",
        "segments": int(segments),
        "stage": stage,
        "discharge": discharge,
    }
    return _build(spec, [], None, parameters, parameter_values, use_default_flat_priors)


def model_bivariate(
    marginal_x,
    marginal_y,
    copula: str = "Normal",
    estimation_method: str = "InferenceFromMargins",
    parameters=None,
    parameter_values=None,
    use_default_flat_priors=None,
) -> Model:
    """Two fixed univariate marginals coupled by a bivariate copula.

    Parameters
    ----------
    marginal_x, marginal_y : dict
        Each a mapping with ``"family"``, ``"data"``, and ``"parameter_values"`` keys
        describing a fixed marginal.
    copula : str, default "Normal"
        One of ``Normal``, ``StudentT``, ``Clayton``, ``Frank``, ``Gumbel``, ``Joe``, or
        ``AliMikhailHaq``.
    estimation_method : str, default "InferenceFromMargins"
        One of ``InferenceFromMargins``, ``PseudoLikelihood``, or ``FullLikelihood``.
    parameters, parameter_values, use_default_flat_priors
        As in :func:`model_univariate`.

    Returns
    -------
    Model
        The assembled model spec.
    """

    def as_marginal(m, what):
        if not isinstance(m, dict) or "family" not in m or "data" not in m:
            raise ValueError(f"`{what}` must be a mapping with `family` and `data` keys")
        out = {
            "family": str(m["family"]),
            "data": [float(v) for v in np.asarray(m["data"], float).ravel()],
        }
        if m.get("parameter_values") is not None:
            out["parameter_values"] = [
                float(v) for v in np.asarray(m["parameter_values"], float).ravel()
            ]
        return out

    spec = {
        "type": "bivariate",
        "copula": str(copula),
        "estimation_method": str(estimation_method),
        "marginal_x": as_marginal(marginal_x, "marginal_x"),
        "marginal_y": as_marginal(marginal_y, "marginal_y"),
    }
    return _build(spec, [], None, parameters, parameter_values, use_default_flat_priors)


# --- Model verbs ------------------------------------------------------------------------------


def model_validate(model: Model) -> dict:
    """Check a model's data and parameters and report anything that would stop it being fit.

    Parameters
    ----------
    model : Model
        The model to validate.

    Returns
    -------
    dict
        ``is_valid`` (bool) and ``messages`` (list of str, empty when valid).
    """
    model = _check_model(model)
    return _core.model_validate(model.to_json(), model.dataset)


def model_simulate(model: Model, n: int, seed: int = 12345):
    """Draw a seeded random sample from a model at its current parameter values.

    The draws come from the same Mersenne Twister stream as the upstream C# library, so a given
    seed reproduces the C# values bit-for-bit and matches ``corehydror`` exactly.

    Parameters
    ----------
    model : Model
        The model to draw from.
    n : int
        Number of values to draw.
    seed : int, default 12345
        PRNG seed.

    Returns
    -------
    numpy.ndarray
        Length ``n``, or an ``n`` by 2 array for a bivariate model.
    """
    model = _check_model(model)
    out = np.asarray(_core.model_simulate(model.to_json(), model.dataset, int(n), int(seed)))
    if model.spec.get("type") == "bivariate":
        return out.reshape(int(n), 2)
    return out


def model_log_likelihood(model: Model, params=None) -> dict:
    """Evaluate a model's log-likelihood, decomposed into its data and prior halves.

    Parameters
    ----------
    model : Model
        The model to evaluate.
    params : array_like, optional
        Parameter values; the model's own current values are used when omitted.

    Returns
    -------
    dict
        ``log_likelihood``, ``data_log_likelihood``, ``prior_log_likelihood``, and
        ``parameters`` (the vector the model actually evaluated, which a mixture model rewrites
        in place to normalize its weights).
    """
    model = _check_model(model)
    values = [] if params is None else [float(v) for v in np.asarray(params, float).ravel()]
    return _core.model_log_likelihood(model.to_json(), model.dataset, values)


def model_parameters(model: Model) -> dict:
    """The model's current parameter vector, with names where the model exposes them.

    Parameters
    ----------
    model : Model
        The model to inspect.

    Returns
    -------
    dict
        ``values`` (list of float), ``names`` (plain-word parameter labels), and ``symbols``
        (the library's own symbols). The two name lists are empty for any model whose parameter
        vector is not a single distribution's, which is every family other than a stationary
        univariate one.
    """
    model = _check_model(model)
    values = model_log_likelihood(model)["parameters"]
    names: list[str] = []
    symbols: list[str] = []
    family = model.spec.get("family")
    if family is not None and "trends" not in model.spec:
        candidate = _parameter_labels(family)
        if len(candidate) == len(values):
            names = candidate
            symbols = list(_core.dist_parameter_names(family)["short"])
    return {"values": values, "names": names, "symbols": symbols}
