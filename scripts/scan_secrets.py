from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys


@dataclass(frozen=True)
class Finding:
    location: str
    kind: str


TEXT_PATTERNS = [
    ("KAGGLE_KEY assignment", re.compile(r"(?m)^\s*KAGGLE_KEY\s*=\s*['\"][^'\"]+['\"]\s*$")),
    ("KAGGLE_USERNAME assignment", re.compile(r"(?m)^\s*KAGGLE_USERNAME\s*=\s*['\"][^'\"]+['\"]\s*$")),
    ("kaggle.json key", re.compile(r'"key"\s*:\s*"[^\"]{8,}"')),
]


def git_ls_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def git_rev_list_all() -> list[str]:
    out = subprocess.check_output(["git", "rev-list", "--all"], text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def git_show(commit: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True, errors="ignore")


def scan_text(content: str, *, location: str) -> list[Finding]:
    findings: list[Finding] = []
    for kind, pattern in TEXT_PATTERNS:
        if pattern.search(content):
            findings.append(Finding(location=location, kind=kind))
    return findings


def scan_working_tree() -> list[Finding]:
    findings: list[Finding] = []
    for rel in git_ls_files():
        # Hard fail if these are tracked at all.
        if rel.endswith("kaggle.json") or rel.endswith(".env"):
            findings.append(Finding(location=rel, kind="tracked secret/config file"))
            continue

        path = Path(rel)
        try:
            if path.is_dir() or not path.exists():
                continue
            if path.stat().st_size > 3_000_000:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        findings.extend(scan_text(content, location=rel))
    return findings


def scan_history() -> list[Finding]:
    target_paths = {
        "notebooks/Hull Tactical.py",
        "notebooks/Hull Tactical.ipynb",
        "kaggle.json",
        ".env",
        ".envrc",
    }
    findings: list[Finding] = []
    for commit in git_rev_list_all():
        files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], text=True).splitlines()
        for rel in files:
            if rel not in target_paths:
                continue
            try:
                content = git_show(commit, rel)
            except subprocess.CalledProcessError:
                continue
            for f in scan_text(content, location=f"{commit[:12]}:{rel}"):
                findings.append(f)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight secret scan (focused on Kaggle creds).")
    parser.add_argument("--history", action="store_true", help="Scan git history (slower).")
    args = parser.parse_args()

    findings = scan_history() if args.history else scan_working_tree()
    if not findings:
        print("OK: no obvious secrets found.")
        return 0

    print("ERROR: potential secrets found:")
    for f in findings:
        print(f"- {f.location} :: {f.kind}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

