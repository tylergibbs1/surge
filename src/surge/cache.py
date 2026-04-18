"""Legacy cache shim. Prefer `surge.store` for new code."""

from surge.store import _root as CACHE_ROOT_FN  # noqa: F401


def path_for(*args, **kwargs):  # pragma: no cover
    raise RuntimeError("surge.cache is deprecated — use surge.store instead")
