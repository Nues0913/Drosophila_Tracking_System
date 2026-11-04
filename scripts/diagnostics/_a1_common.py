"""File provenance and report writers shared by the read-only A1 diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(paths: list[Path]) -> dict:
    root = Path(__file__).resolve().parents[2]
    programs = sorted((root / "python/flytrack_core/model").rglob("*.py")) + [
        Path(__file__),
        Path(__file__).with_name("a1_summary.py"),
        Path(__file__).with_name("episode_trace.py"),
        root / "python/flytrack_core/metrics.py",
        root / "python/flytrack_train/evaluation.py",
        root / "python/flytrack_train/worlds.py",
        root / "python/flytrack_train/config.py",
        root / "python/flytrack_sim/teacher.py",
        root / "python/flytrack_sim/env.py",
        root / "scripts/lesion_analysis.py",
    ]
    head = "NOT AVAILABLE"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            head = result.stdout.strip()
    except OSError:
        pass
    return {
        "inputs_sha256": {str(path): sha256(path) for path in sorted(set(paths))},
        "program_version": {
            "method": "file_sha256",
            "git_head": head,
            "files": {str(path.relative_to(root)): sha256(path) for path in programs},
        },
    }


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError(f"{path}: expected a JSON list of episode objects")
        rows.extend(data)
    return rows


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input does not exist: {path}")
    return sorted(path.rglob("episodes.json"))


def flatten(row: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = json.dumps(value, allow_nan=False) if isinstance(value, list) else value
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    flat = [flatten(clean(row)) for row in rows]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def markdown_table(title: str, rows: list[dict]) -> str:
    flat = [flatten(clean(row)) for row in rows]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    if not fields:
        return f"## {title}\n\nNo rows.\n"

    def cell(value):
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [f"## {title}", "", "| " + " | ".join(fields) + " |"]
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    lines.extend("| " + " | ".join(cell(row.get(key)) for key in fields) + " |" for row in flat)
    return "\n".join(lines) + "\n"


def write_report(
    output: Path, name: str, payload: dict, tables: dict[str, list[dict]], *, command: str
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{name}.json", payload)
    write_json(output / f"{command}_provenance.json", payload["provenance"])
    sections = [f"# {name}", "", f"Input and source SHA-256: `{command}_provenance.json`.", ""]
    for key, value in payload.items():
        if key not in tables and key != "provenance":
            sections.extend([f"{key}: `{json.dumps(value, ensure_ascii=False)}`", ""])
    for title, rows in tables.items():
        write_csv(output / f"{title}.csv", rows)
        sections.append(markdown_table(title, rows))
    (output / f"{command}_summary.md").write_text("\n".join(sections), encoding="utf-8")
