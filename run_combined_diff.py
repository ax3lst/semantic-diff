#!/usr/bin/env python3
"""
Run combined_diff.diff_documents the same way app.py does for local dev.

Usage:
  python run_combined_diff.py path/to/old.json path/to/new.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

from combined_diff import diff_documents


def _venv_python(root: Path) -> Path | None:
    unix = root / "venv" / "bin" / "python"
    if unix.is_file():
        return unix
    win = root / "venv" / "Scripts" / "python.exe"
    if win.is_file():
        return win
    return None


def _ensure_local_venv() -> None:
    root = Path(__file__).resolve().parent
    venv_python = _venv_python(root)
    if not venv_python:
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if os.environ.get("VIRTUAL_ENV"):
        return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run combined_diff the same way as app.py /diff."
    )
    parser.add_argument(
        "old",
        type=Path,
        nargs="?",
        default=Path("data_old.json"),
        help="Path to the old JSON export (default: data_old.json)",
    )
    parser.add_argument(
        "new",
        type=Path,
        nargs="?",
        default=Path("data_new.json"),
        help="Path to the new JSON export (default: data_new.json)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write the final HTML report (default: reports)",
    )
    return parser.parse_args()


def main() -> int:
    _ensure_local_venv()
    load_dotenv()
    args = parse_args()

    if not args.old.is_file():
        raise SystemExit(f"old file not found: {args.old}")
    if not args.new.is_file():
        raise SystemExit(f"new file not found: {args.new}")

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        old_path = workdir / "old.json"
        new_path = workdir / "new.json"
        shutil.copyfile(args.old, old_path)
        shutil.copyfile(args.new, new_path)

        report_local = diff_documents(
            old=old_path,
            new=new_path,
            out=workdir / "combined_report.html",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        final_name = f"diff_report_{uuid.uuid4().hex}.html"
        final_path = report_dir / final_name
        os.replace(report_local, final_path)

    print(f"Report written: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
