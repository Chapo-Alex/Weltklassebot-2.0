"""CI coverage gate enforcing overall and per-package thresholds."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET


def main() -> int:
    if os.getenv("GITHUB_ACTIONS") not in {"true", "1"}:
        print("Coverage gate OFF locally.")
        return 0

    tree = ET.parse("coverage.xml")
    root = tree.getroot()

    threshold = float(os.getenv("COV_THR", "90"))
    packages = {"core": 0.0, "portfolio": 0.0, "risk": 0.0, "strategy": 0.0}

    for package in root.findall(".//package"):
        name = package.get("name", "")
        rate = float(package.get("line-rate", "0")) * 100
        for key in packages:
            if name.startswith(f"src.{key}"):
                packages[key] = max(packages[key], rate)

    overall = float(root.get("line-rate", "0")) * 100
    failing = [key for key, value in packages.items() if value < threshold]

    print("overall:", overall, "packages:", json.dumps(packages))

    if overall < threshold or failing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
