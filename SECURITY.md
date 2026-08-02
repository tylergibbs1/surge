# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.2.x | Yes |
| 0.0.x | No |

This is alpha research software. Security support means maintainers will triage
reports and publish fixes where feasible; it is not an availability SLA.

## Reporting a vulnerability

Use the repository's private GitHub security-advisory flow at
<https://github.com/tylergibbs1/surge/security/advisories/new>. Include affected
versions, impact, reproduction steps, and any suggested mitigation.

If private reporting is unavailable, open a minimal public issue asking for a
private contact channel. Do **not** include exploit details, credentials,
private endpoint URLs, tokens, or user data in that issue.

Please allow reasonable time for triage before public disclosure. Maintainers
will acknowledge receipt, assess severity and affected versions, and coordinate
a fix and disclosure timeline when the report is confirmed.

## Scope notes

- Never commit `.env` files, provider tokens, Modal/Vercel secrets, or Hugging
  Face credentials.
- Model loading should use an immutable revision and safe weight formats.
- Forecast outputs are not trusted merely because they parse: freshness,
  provenance, finite values, quantile ordering, and completeness are security
  and integrity boundaries.
- Dependency or upstream-data compromise reports are welcome even when the
  issue originates outside this repository.
