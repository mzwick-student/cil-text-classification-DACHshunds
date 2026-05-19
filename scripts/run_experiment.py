#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import configs_for_seeds, load_config, run_experiment


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = target
    for key in parts[:-1]:
        current = current.setdefault(key, {})
    current[parts[-1]] = value


def parse_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        _set_nested(overrides, key, _parse_value(value))
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible sentiment experiments.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help='Override config fields, e.g. --set seeds=[1,2,3] --set trainer="pcgrad"',
    )
    args = parser.parse_args()
    config = load_config(Path(args.config), overrides=parse_overrides(args.set))
    reports = []
    for seed, seed_config in configs_for_seeds(config):
        reports.append(run_experiment(seed_config, seed))
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
