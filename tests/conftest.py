import pytest


def pytest_addoption(parser):
    parser.addoption("--run-model-checks", action="store_true", help="Run checks requiring pretrained model assets")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-model-checks"):
        return
    skip = pytest.mark.skip(reason="Requires pretrained assets; enable with --run-model-checks")
    for item in items:
        if "model_assets" in item.keywords:
            item.add_marker(skip)
