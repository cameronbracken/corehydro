#!/usr/bin/env python3
"""Verify the oracle fixtures reproduce against the upstream Numerics C# library.

This is the "run dotnet" half of the hybrid oracle workflow. The fixture expected values
are scraped/curated from the C# unit tests; this gate rebuilds each case against the real
``Numerics`` library (via ``tools/oracle_emitter``) and checks every value reproduces to its
stated tolerance. It is a DEV-ONLY check: it needs ``dotnet`` and the ``upstream/Numerics``
submodule, neither of which is required to build or ship the packages.

Usage:  python3 tools/verify_oracles.py [--fixtures DIR] [--dump]
  exits 0 if every supported assertion reproduces, 1 on any mismatch, 2 if the toolchain
  is unavailable (no dotnet / submodule not checked out).

  --dump: curation path for multivariate_distribution fixtures (see the emitter's
  DumpLine()). Prints target/case/method/args and the actual C#-computed value as JSON
  lines instead of comparing, so a case can be authored with placeholder "expected"
  values, run once with --dump to harvest the real values, then re-run without --dump to
  verify.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMITTER = ROOT / "tools" / "oracle_emitter"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", default=str(ROOT / "fixtures"), help="fixtures directory")
    ap.add_argument("--dump", action="store_true", help="curation path; see module docstring")
    args = ap.parse_args()

    if shutil.which("dotnet") is None:
        print("dotnet not found on PATH; skipping oracle reproduction check.", file=sys.stderr)
        return 2
    if not (ROOT / "upstream" / "Numerics" / "Numerics" / "Numerics.csproj").exists():
        print("upstream/Numerics submodule not checked out; run: git submodule update --init",
              file=sys.stderr)
        return 2

    # Build in one MSBuild process. Multi-node builds can wait indefinitely on the local node IPC
    # connection in restricted runners even though the same compile finishes in a few seconds
    # in-process. Disable NuGet auditing because this dev-only project has no package dependencies
    # and oracle verification must not depend on vulnerability-service availability.
    restore = ["dotnet", "restore", str(EMITTER), "--ignore-failed-sources",
               "-p:NuGetAudit=false", "--disable-parallel"]
    if subprocess.run(restore, cwd=ROOT).returncode != 0:
        return 1
    build = ["dotnet", "build", str(EMITTER), "-c", "Release", "--no-restore", "--nologo",
             "-m:1", "-nodeReuse:false", "-p:UseSharedCompilation=false",
             "-p:NuGetAudit=false"]
    if subprocess.run(build, cwd=ROOT).returncode != 0:
        return 1

    emitter_dll = EMITTER / "bin" / "Release" / "net10.0" / "oracle_emitter.dll"
    cmd = ["dotnet", str(emitter_dll), str(Path(args.fixtures).resolve())]
    if args.dump:
        cmd.append("--dump")
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
