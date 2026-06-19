"""Source metadata helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SourceScope = Literal["user", "project", "temporary", "bundled"]
SourceOrigin = Literal["package", "top-level"]


@dataclass(frozen=True)
class SourceInfo:
    path: str
    source: str
    scope: SourceScope
    origin: SourceOrigin
    base_dir: str | None = None


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def create_source_info(path: str, metadata: Any) -> SourceInfo:
    return SourceInfo(
        path=path,
        source=_get_attr_or_key(metadata, "source"),
        scope=_get_attr_or_key(metadata, "scope"),
        origin=_get_attr_or_key(metadata, "origin"),
        base_dir=_get_attr_or_key(metadata, "base_dir", _get_attr_or_key(metadata, "baseDir")),
    )


def create_synthetic_source_info(
    path: str,
    *,
    source: str,
    scope: SourceScope = "temporary",
    origin: SourceOrigin = "top-level",
    base_dir: str | None = None,
) -> SourceInfo:
    return SourceInfo(path=path, source=source, scope=scope, origin=origin, base_dir=base_dir)


def source_info_to_dict(source_info: SourceInfo) -> dict[str, Any]:
    data: dict[str, Any] = {
        "path": source_info.path,
        "source": source_info.source,
        "scope": source_info.scope,
        "origin": source_info.origin,
    }
    if source_info.base_dir is not None:
        data["baseDir"] = source_info.base_dir
    return data


__all__ = [
    "SourceInfo",
    "SourceOrigin",
    "SourceScope",
    "create_source_info",
    "create_synthetic_source_info",
    "source_info_to_dict",
]
