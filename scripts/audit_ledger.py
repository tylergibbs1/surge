"""Audit committed forecast ledger hashes and structural invariants."""

from __future__ import annotations

import argparse
import sys

from surge import ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    records = ledger.list_forecasts(
        mode=None,
        limit=args.limit,
        include_unpublished=True,
    )
    failures: list[tuple[str, list[str]]] = []
    for record in records:
        errors = ledger.audit_forecast(record.issuance_id)
        if errors:
            failures.append((record.issuance_id, errors))

    print(f"audited={len(records)} failures={len(failures)}")
    for issuance_id, errors in failures:
        print(f"{issuance_id}: {', '.join(errors)}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
