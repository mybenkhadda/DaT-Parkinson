"""Build submission.zip from submission_src/ with main.py at the archive root.

Usage:
    python scripts/build_submission.py [--output submission.zip]

Deliberately zips the *contents* of submission_src/ rather than the folder
itself -- DrivenData's runtime expects main.py at the archive root, not
nested under a submission_src/ prefix.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_SRC = REPO_ROOT / "submission_src"

# Files/directories we never want inside the archive even if present locally
# (caches, bytecode, editor artifacts).
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".DS_Store"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_NAMES for part in path.relative_to(root).parts):
            continue
        if path.suffix in EXCLUDE_SUFFIXES:
            continue
        yield path


def build(output_path: Path) -> None:
    if not SUBMISSION_SRC.is_dir():
        raise FileNotFoundError(f"submission_src/ not found at {SUBMISSION_SRC}")
    if not (SUBMISSION_SRC / "main.py").is_file():
        raise FileNotFoundError(f"main.py not found directly under {SUBMISSION_SRC}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in iter_files(SUBMISSION_SRC):
            arcname = file_path.relative_to(SUBMISSION_SRC)
            zf.write(file_path, arcname=str(arcname))

    _verify(output_path)


def _verify(output_path: Path) -> None:
    with zipfile.ZipFile(output_path) as zf:
        names = zf.namelist()
    if "main.py" not in names:
        raise AssertionError(
            "main.py is not at the archive root -- packaging is broken. "
            f"Archive top-level entries: {sorted({n.split('/')[0] for n in names})}"
        )
    nested_main = [n for n in names if n != "main.py" and n.endswith("main.py")]
    if nested_main:
        raise AssertionError(f"Found nested main.py copies that shouldn't exist: {nested_main}")
    print(f"OK: {output_path} contains {len(names)} files, main.py is at archive root.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / "submission.zip"),
                         help="Output zip path (default: submission.zip at repo root)")
    args = parser.parse_args()
    build(Path(args.output))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
