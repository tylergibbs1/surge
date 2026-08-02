"""Verify that the deployed forecasting stack is installed and import-compatible."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version

EXPECTED_VERSIONS = {
    "chronos-forecasting": "2.3.1",
    "holidays": "0.101",
    "numpy": "2.1.2",
    "peft": "0.20.0",
    "polars": "1.43.2",
    "pyarrow": "25.0.0",
    "torch": "2.8.0",
    "transformers": "4.57.6",
}

IMPORT_NAMES = {
    "chronos-forecasting": "chronos",
    "holidays": "holidays",
    "numpy": "numpy",
    "peft": "peft",
    "polars": "polars",
    "pyarrow": "pyarrow",
    "torch": "torch",
    "transformers": "transformers",
}


def main() -> None:
    for distribution, expected in EXPECTED_VERSIONS.items():
        actual = version(distribution)
        # PyTorch's CPU index adds a local build suffix (for example, +cpu)
        # while still satisfying the torch==2.8.0 package constraint.
        comparable = actual.partition("+")[0] if distribution == "torch" else actual
        if comparable != expected:
            raise RuntimeError(f"{distribution}: expected {expected}, found {actual}")
        import_module(IMPORT_NAMES[distribution])

    # Exercise the exact integration surfaces used by adapter-aware inference.
    from chronos import Chronos2Pipeline
    from transformers.utils.peft_utils import find_adapter_config_file

    if not callable(find_adapter_config_file):
        raise RuntimeError("transformers PEFT adapter helper is unavailable")
    if Chronos2Pipeline is None:
        raise RuntimeError("Chronos2Pipeline is unavailable")

    installed = ", ".join(
        f"{distribution}={version(distribution)}" for distribution in EXPECTED_VERSIONS
    )
    print(f"Runtime stack OK: {installed}")


if __name__ == "__main__":
    main()
