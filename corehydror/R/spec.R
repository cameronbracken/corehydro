# Internal: the model/data spec serializer.
#
# Specs cross into C++ as JSON strings and are parsed by core/include/corehydro/models/
# json_lite.hpp. The grammar is small (objects, arrays, numbers, strings, booleans), so this
# emits it directly rather than depending on jsonlite: the package then has no runtime
# dependency at all, and double formatting stays under our control. That matters because the
# Python package serializes the identical spec with json.dumps() and both must reach the C++
# parser as the same value -- 17 significant digits round-trips an IEEE double exactly, which is
# what `%.17g` gives here and what repr() gives on the Python side.
#
# Emission rules:
#   NULL                     -> the key is dropped entirely (absent means "use the default")
#   a corehydro_dist         -> {"family": ..., "parameters": [...]}
#   an unnamed list / vector -> a JSON array
#   a named list             -> a JSON object
# A length-1 vector still emits as a scalar unless wrapped in I(), because every scalar key in
# the spec grammar (family, dataset, threshold, ...) would otherwise become a one-element array.

# Marks a value that must emit as a JSON array even when it has length 1.
spec_array <- function(x) {
  structure(x, class = c("corehydro_spec_array", class(x)))
}

spec_number <- function(x) {
  if (!is.finite(x)) {
    stop("spec values must be finite; got ", format(x), call. = FALSE)
  }
  # %.17g round-trips an IEEE double; trim the exponent form R and C++ both accept.
  sprintf("%.17g", as.double(x))
}

spec_string <- function(x) {
  x <- gsub("\\", "\\\\", as.character(x), fixed = TRUE)
  x <- gsub("\"", "\\\"", x, fixed = TRUE)
  x <- gsub("\n", "\\n", x, fixed = TRUE)
  x <- gsub("\t", "\\t", x, fixed = TRUE)
  x <- gsub("\r", "\\r", x, fixed = TRUE)
  paste0("\"", x, "\"")
}

to_spec_json <- function(x) {
  if (is.null(x)) {
    return("null")
  }
  if (inherits(x, "corehydro_dist")) {
    return(to_spec_json(list(family = x$family, parameters = spec_array(x$params))))
  }

  forced_array <- inherits(x, "corehydro_spec_array")

  if (is.list(x)) {
    nms <- names(x)
    if (!is.null(nms) && all(nzchar(nms)) && !forced_array) {
      keep <- !vapply(x, is.null, logical(1))
      x <- x[keep]
      if (length(x) == 0L) {
        return("{}")
      }
      parts <- vapply(
        seq_along(x),
        function(i) paste0(spec_string(names(x)[i]), ":", to_spec_json(x[[i]])),
        character(1)
      )
      return(paste0("{", paste(parts, collapse = ","), "}"))
    }
    parts <- vapply(x, to_spec_json, character(1))
    return(paste0("[", paste(parts, collapse = ","), "]"))
  }

  if (is.logical(x)) {
    out <- ifelse(is.na(x), "null", ifelse(x, "true", "false"))
  } else if (is.character(x)) {
    out <- vapply(x, spec_string, character(1), USE.NAMES = FALSE)
  } else if (is.numeric(x)) {
    out <- vapply(x, spec_number, character(1), USE.NAMES = FALSE)
  } else {
    stop("cannot serialize an object of class ", class(x)[1], " into a spec", call. = FALSE)
  }

  if (length(out) == 1L && !forced_array) {
    return(out)
  }
  paste0("[", paste(out, collapse = ","), "]")
}
