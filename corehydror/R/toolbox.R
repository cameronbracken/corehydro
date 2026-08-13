# The Numerics toolbox surface. Every verb serializes its options to the toolbox_runner.hpp
# grammar and runs one method through ch_toolbox_run_; bulk data goes across as numeric vectors,
# not JSON. Nothing holds C++ state.

# Internal: one call into the shared runner. `data` is a list of numeric vectors, `options` a
# named list serialized with the same emitter every other spec uses.
toolbox_run <- function(group, method, data = list(), options = list()) {
  opts <- if (length(options) == 0L) "{}" else to_spec_json(options)
  ch_toolbox_run_(group, method, lapply(data, as.double), opts)
}

# Internal: reject the two mistakes every paired-series verb can make, naming the argument.
check_pair <- function(x, y, x_name = "x", y_name = "y") {
  if (!is.numeric(x) || !is.numeric(y)) {
    stop(sprintf("`%s` and `%s` must be numeric vectors", x_name, y_name), call. = FALSE)
  }
  if (length(x) != length(y)) {
    stop(sprintf("`%s` and `%s` must have the same length; got %d and %d",
                 x_name, y_name, length(x), length(y)), call. = FALSE)
  }
  if (length(x) < 2L) {
    stop(sprintf("`%s` and `%s` must have at least two elements", x_name, y_name), call. = FALSE)
  }
  invisible(NULL)
}

#' Correlation between two samples
#'
#' Mirrors the C# `Correlation` class of the Numerics library. Upstream's matrix overloads
#' (`Pearson(double[,])`, `Spearman(double[,])`) are not ported, so only the paired-vector forms
#' are available here.
#'
#' @param x,y numeric vectors of equal length, at least two elements.
#' @param method one of `"pearson"` (the default), `"spearman"`, or `"kendall"`.
#' @return a single numeric correlation coefficient.
#' @examples
#' x <- c(14, 8, 32, 7, 3, 15)
#' y <- c(10, 5, 7, 4, 3, 8)
#' correlation(x, y)
#' correlation(x, y, method = "kendall")
#' @export
correlation <- function(x, y, method = c("pearson", "spearman", "kendall")) {
  method <- match.arg(method)
  check_pair(x, y)
  toolbox_run("correlation", method, list(x, y))$values[[1]]
}
