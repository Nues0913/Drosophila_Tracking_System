from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline MaleCNS inventory and path extraction")
    parser.add_argument("command", choices=("inventory", "extract"))
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import yaml

    from .graph import extract, inventory

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Extraction config must be a mapping")
    if args.command == "inventory":
        inventory(
            Path(config.get("raw_dir", "data/raw")),
            Path(config.get("output", "data/subgraphs/main")),
            config.get("columns", {}),
        )
    else:
        print(extract(config))


if __name__ == "__main__":
    main()
