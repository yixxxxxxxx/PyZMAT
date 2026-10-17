"""Keep campaigns opt-in, including CUDA parametrizations in legacy tests."""
import os
import pytest


def pytest_configure(config):
    for level in ("L0", "L1", "L2", "L3"):
        config.addinivalue_line("markers", level + ": scientific test tier")


def pytest_collection_modifyitems(config, items):
    requested = os.environ.get("TIER_LEVEL")
    deselected=[]
    for item in list(items):
        callspec = getattr(item, "callspec", None)
        gpu = callspec is not None and callspec.params.get("device") == "cuda"
        gpu = gpu or "cuda_matches_cpu" in item.name or "real_mace_warmed" in item.name
        level = next((n for n in ("L3", "L2", "L1", "L0") if item.get_closest_marker(n)), "L0")
        if gpu:
            level = "L3"
        item.add_marker(getattr(pytest.mark, level))
        if requested and level != requested:
            items.remove(item); deselected.append(item)
        elif not requested and level == "L3":
            item.add_marker(pytest.mark.skip(reason="Explicit tier selection required: " + level))
    if deselected:
        config.hook.pytest_deselected(items=deselected)
