"""ViHERMES project package."""

from importlib import metadata

__all__ = []

try:
    __version__ = metadata.version("ViHERMES")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

