// Shared-core version stamp. Single source of truth for the C++ core version;
// bindings may record it to report which core they were built against, and a future
// standalone release uses it. Not included by any other core header -- it exists for
// consumers.
#pragma once

#define BESTFIT_CORE_VERSION_MAJOR 0
#define BESTFIT_CORE_VERSION_MINOR 3
#define BESTFIT_CORE_VERSION_PATCH 0
#define BESTFIT_CORE_VERSION "0.3.0"
