"""Create a source-only archive for a Colab/other-host handoff; does not train."""

import argparse
import zipfile
from pathlib import Path

EXCLUDE = {
    ".git",
    ".aws",
    ".codex",
    ".agents",
    ".venv",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    "runs",
    "artifacts",
    "build",
    "install",
    "log",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/flytrack-source.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and not path.is_symlink()
                and path.resolve() != destination
                and not any(
                    part in EXCLUDE or part.endswith(".egg-info") for part in relative.parts
                )
                and path.suffix not in {".pdf", ".pyc", ".zip", ".ipynb_checkpoints"}
            ):
                archive.write(path, relative)
    print(destination)


if __name__ == "__main__":
    main()
