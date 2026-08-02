"""Print reproducible metrics for one committed forecast issuance."""

from __future__ import annotations

import argparse

from surge import verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issuance_id")
    args = parser.parse_args()
    print(verification.score_forecast(args.issuance_id).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
