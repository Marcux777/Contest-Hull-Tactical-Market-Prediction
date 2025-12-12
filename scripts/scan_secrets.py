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
    ("KAGGLE_KEY assignment", re.compile(r"(?m)^\s*KAGGLE_KEY\s*=\s*['\"][^'\"]+['\"]\s*(?:#.*)?$")),
    ("KAGGLE_USERNAME assignment", re.compile(r"(?m)^\s*KAGGLE_USERNAME\s*=\s*['\"][^'\"]+['\"]\s*(?:#.*)?$")),
    ("os.environ[KAGGLE_KEY] assignment", re.compile(r"os\.environ\[\s*['\"]KAGGLE_KEY['\"]\s*\]\s*=\s*['\"][^'\"]{8,}['\"]")),
    ("os.environ[KAGGLE_USERNAME] assignment", re.compile(r"os\.environ\[\s*['\"]KAGGLE_USERNAME['\"]\s*\]\s*=\s*['\"][^'\"]{2,}['\"]")),
    ("os.environ.setdefault(KAGGLE_KEY, ...)", re.compile(r"os\.environ\.setdefault\(\s*['\"]KAGGLE_KEY['\"]\s*,\s*['\"][^'\"]{8,}['\"]\s*\)")),
    ("os.environ.setdefault(KAGGLE_USERNAME, ...)", re.compile(r"os\.environ\.setdefault\(\s*['\"]KAGGLE_USERNAME['\"]\s*,\s*['\"][^'\"]{2,}['\"]\s*\)")),
    ("export KAGGLE_KEY=...", re.compile(r"(?m)^\s*export\s+KAGGLE_KEY\s*=\s*(['\"][^'\"]{8,}['\"]|[^#\s]{8,})\s*(?:#.*)?$")),
    ("export KAGGLE_USERNAME=...", re.compile(r"(?m)^\s*export\s+KAGGLE_USERNAME\s*=\s*(['\"][^'\"]{2,}['\"]|[^#\s]{2,})\s*(?:#.*)?$")),
    ("YAML KAGGLE_KEY: ...", re.compile(r"(?m)^\s*KAGGLE_KEY\s*:\s*(['\"][^'\"]{8,}['\"]|[^#\s]{8,})\s*(?:#.*)?$")),
    ("YAML KAGGLE_USERNAME: ...", re.compile(r"(?m)^\s*KAGGLE_USERNAME\s*:\s*(['\"][^'\"]{2,}['\"]|[^#\s]{2,})\s*(?:#.*)?$")),
    ("KAGGLE_KEY assignment (ipynb escaped)", re.compile(r"KAGGLE_KEY\s*=\s*\\\"[^\\\"]+\\\"\s*(?:#.*)?")),
    ("KAGGLE_USERNAME assignment (ipynb escaped)", re.compile(r"KAGGLE_USERNAME\s*=\s*\\\"[^\\\"]+\\\"\s*(?:#.*)?")),
    ("kaggle.json key", re.compile(r'"key"\s*:\s*"[^\"]{8,}"')),
    ("kaggle.json key (ipynb escaped)", re.compile(r"\\\"key\\\"\\s*:\\s*\\\"[^\\\"]{8,}\\\"")),
]

HISTORY_EXTS = {
    ".py",
    ".ipynb",
    ".md",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".bat",
}

MAX_SCAN_BYTES = 3_000_000


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
            if path.stat().st_size > MAX_SCAN_BYTES:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        findings.extend(scan_text(content, location=rel))
    return findings


def scan_local_sensitive_paths() -> list[Finding]:
    """Scans common local secret files even if they are untracked/ignored.

    This prevents accidental leakage when zipping/sharing the repo folder.
    """
    findings: list[Finding] = []
    # Kaggle token file should live in ~/.kaggle/kaggle.json, never inside the repo.
    kaggle_token_paths = [Path("kaggle.json"), Path(".kaggle") / "kaggle.json"]
    for p in kaggle_token_paths:
        try:
            if not p.exists() or p.is_dir():
                continue
            findings.append(Finding(location=str(p), kind="Kaggle token file present in repo working tree"))
            if p.stat().st_size > MAX_SCAN_BYTES:
                continue
            content = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        findings.extend(scan_text(content, location=str(p)))

    # Optional local env files: only flag if they contain Kaggle patterns.
    env_paths = [Path(".env"), Path(".envrc")]
    for p in env_paths:
        try:
            if not p.exists() or p.is_dir():
                continue
            if p.stat().st_size > MAX_SCAN_BYTES:
                continue
            content = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        findings.extend(scan_text(content, location=str(p)))

    return findings


def _should_scan_history_path(path: str) -> bool:
    p = Path(path)
    if p.name in {"kaggle.json", ".env", ".envrc"}:
        return True
    if p.suffix.lower() in HISTORY_EXTS:
        return True
    return False


def scan_history() -> list[Finding]:
    findings: list[Finding] = []
    for commit in git_rev_list_all():
        tree = subprocess.check_output(["git", "ls-tree", "-r", "-l", commit], text=True).splitlines()
        for line in tree:
            try:
                meta, rel = line.split("\t", 1)
            except ValueError:
                continue
            parts = meta.split()
            if len(parts) < 4:
                continue
            size_raw = parts[3]
            if size_raw.isdigit() and int(size_raw) > MAX_SCAN_BYTES:
                continue
            if not _should_scan_history_path(rel):
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
    findings.extend(scan_local_sensitive_paths())
    if not findings:
        print("OK: no obvious secrets found.")
        return 0

    print("ERROR: potential secrets found:")
    for f in findings:
        print(f"- {f.location} :: {f.kind}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
