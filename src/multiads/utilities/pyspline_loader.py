"""Local pyspline loader for native MADS geometry workflows."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path


def _candidate_dependency_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [repo_root / "project"]
    env_root = os.environ.get("MULTIADS_DEPENDENCY_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    cwd_project = Path.cwd() / "project"
    candidates.append(cwd_project)

    deduped: list[Path] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def ensure_local_dependency_paths() -> None:
    for project_dir in _candidate_dependency_roots():
        dependency_paths = (
            project_dir,
            project_dir / "pyspline",
            project_dir / "pygeo",
        )
        for dependency_dir in dependency_paths:
            dependency_path = str(dependency_dir)
            if dependency_dir.is_dir() and dependency_path not in sys.path:
                sys.path.insert(0, dependency_path)


def patch_pyspline_for_pygeo() -> None:
    try:
        surface_module = importlib.import_module("pyspline.pySurface")
        curve_module = importlib.import_module("pyspline.pyCurve")
        volume_module = importlib.import_module("pyspline.pyVolume")
        utils_module = importlib.import_module("pyspline.utils")
        shim = importlib.import_module("pyspline")
    except Exception:
        surface_module = importlib.import_module("pyspline.pyspline.pySurface")
        curve_module = importlib.import_module("pyspline.pyspline.pyCurve")
        volume_module = importlib.import_module("pyspline.pyspline.pyVolume")
        utils_module = importlib.import_module("pyspline.pyspline.utils")
        shim = importlib.import_module("pyspline.pyspline")
        sys.modules["pyspline"] = shim
        sys.modules["pyspline.pySurface"] = surface_module
        sys.modules["pyspline.pyCurve"] = curve_module
        sys.modules["pyspline.pyVolume"] = volume_module
        sys.modules["pyspline.utils"] = utils_module

    surface = surface_module.Surface
    curve = curve_module.Curve
    volume = volume_module.Volume

    if not isinstance(shim, types.ModuleType):
        shim = types.ModuleType("pyspline")
        sys.modules["pyspline"] = shim

    if not hasattr(shim, "Surface"):
        shim.Surface = surface
    if not hasattr(shim, "Curve"):
        shim.Curve = curve
    if not hasattr(shim, "Volume"):
        shim.Volume = volume


def load_pyspline_curve():
    ensure_local_dependency_paths()
    patch_pyspline_for_pygeo()
    try:
        from pyspline import Curve
    except Exception:
        pyspline_pkg = importlib.import_module("pyspline.pyspline")
        sys.modules["pyspline"] = pyspline_pkg
        Curve = pyspline_pkg.Curve
    return Curve


def load_pygeo_class():
    ensure_local_dependency_paths()
    patch_pyspline_for_pygeo()
    try:
        from pygeo import pyGeo
    except Exception:
        pygeo_pkg = importlib.import_module("pygeo.pygeo")
        sys.modules["pygeo"] = pygeo_pkg
        pyGeo = pygeo_pkg.pyGeo
    return pyGeo
