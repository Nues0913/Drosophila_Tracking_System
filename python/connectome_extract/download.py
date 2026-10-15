"""Public official data downloader; standard-library only, with resumable partial files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
OFFICIAL_PAGE = "https://male-cns.janelia.org/download/"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _validate_feather(path: Path) -> None:
    # Feather v2 is Arrow IPC and has ARROW1 framing. Do not mistake an HTML error page for data.
    with path.open("rb") as stream:
        start = stream.read(6)
        stream.seek(-6, os.SEEK_END)
        end = stream.read(6)
    if start != b"ARROW1" or end != b"ARROW1":
        raise ValueError(f"{path.name} lacks Arrow IPC framing")


def download_one(url: str, destination: Path, timeout: float, retries: int) -> dict:
    partial = destination.with_suffix(destination.suffix + ".part")
    metadata_path = partial.with_suffix(partial.suffix + ".json")
    if destination.exists():
        _validate_feather(destination)
        return {
            "url": url,
            "status": "present",
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "flytrack/0.1 (research data acquisition)"}
        old_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if offset and old_metadata.get("etag"):
            headers["Range"] = f"bytes={offset}-"
            headers["If-Range"] = old_metadata["etag"]
        else:
            offset = 0
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an unexpected resume range")
                    total = int(content_range.rsplit("/", 1)[1])
                    mode = "ab"
                elif response.status == 200:
                    offset = 0
                    mode = "wb"
                    total_header = response.headers.get("Content-Length")
                    total = int(total_header) if total_header else None
                else:
                    raise ValueError(f"Unexpected HTTP status {response.status}")
                if (
                    total is not None
                    and shutil.disk_usage(destination.parent).free < total - offset
                ):
                    raise OSError("Insufficient free disk space for dataset")
                metadata = {"etag": response.headers.get("ETag"), "total_bytes": total}
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with partial.open(mode) as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                actual = partial.stat().st_size
                if total is not None and actual != total:
                    raise OSError(f"Incomplete data: {actual} bytes, expected {total}")
                _validate_feather(partial)
                digest = sha256(partial)
                partial.replace(destination)
                metadata_path.unlink(missing_ok=True)
                return {
                    "url": url,
                    "status": "downloaded",
                    "bytes": actual,
                    "sha256": digest,
                    "etag": metadata["etag"],
                }
        except urllib.error.HTTPError as exc:
            # Only retry transient requests. Authentication and policy denial are not bypassed.
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == retries:
                raise
        time.sleep(min(2**attempt, 8))
    raise RuntimeError("Download retry loop exhausted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--only", choices=tuple(FILES), nargs="+")
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0:
        parser.error("timeout must be positive and retries nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    status = {
        "dataset": "male-cns:v1.0",
        "official_page": OFFICIAL_PAGE,
        "license": "CC-BY-4.0",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    failures = []
    for key in args.only or FILES:
        name = FILES[key]
        url = f"{BASE_URL}/{name}"
        print(f"Downloading {name}", flush=True)
        try:
            status["files"][key] = download_one(url, args.output / name, args.timeout, args.retries)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            status["files"][key] = {"url": url, "status": "failed", "reason": str(exc)}
            failures.append(key)
        (args.output / "download_manifest.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if failures:
        raise SystemExit(f"Download incomplete: {', '.join(failures)}; see download_manifest.json")
    print("All selected official tables are present; local SHA-256 values recorded.")


if __name__ == "__main__":
    main()
