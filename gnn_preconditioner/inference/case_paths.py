"""Resolve OpenFOAM case directories from current repo roots.

Some older HDF5 datasets preserve their original ``source_case_path`` metadata
from a pre-flattened workspace layout. Runtime tools should treat that metadata
as provenance and resolve the case by name/case id under the active repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPO_ROOT / "gnn_preconditioner" / "data_generation" / "cases"

CASE_SUBROOTS = (
    "pilot_v1",
    "full_surface_coarse_v1_t3p0",
    "pilot_fine_legacy_x1p15",
    "full_surface_coarse_v1_steady",
)


def _required_files_exist(case_dir: Path, required_files: Iterable[str]) -> bool:
    return all((case_dir / rel).is_file() for rel in required_files)


def resolve_case_dir(
    *,
    source_case_path: str | Path | None = None,
    case_id: int | None = None,
    required_files: Iterable[str] = (),
) -> Path | None:
    """Return the active in-repo case directory for stale or current metadata."""

    candidates: list[Path] = []
    raw_path: Path | None = None
    if source_case_path:
        raw_path = Path(str(source_case_path))
        candidates.append(raw_path)

    active_roots = [CASES_ROOT / name for name in CASE_SUBROOTS]
    if raw_path is not None:
        for root in active_roots:
            candidates.append(root / raw_path.name)

    if case_id is not None:
        for root in active_roots:
            candidates.extend(sorted(root.glob(f"case_{int(case_id):03d}_*")))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.is_dir() and _required_files_exist(candidate, required_files):
                return candidate
        except OSError:
            continue
    return None
