import pytest

from cxr_enhance.config import validate_config


@pytest.mark.parametrize("change", [{"epohs": 3}, {"epochs": 0}, {"microbatch_size": 3},
                                    {"model": "tiny"}, {"threshold": 2}, {"seeds": [42, 42]},
                                    {"methods": ["unknown"]}, {"learning_rate": float("nan")}])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        validate_config(change)


def test_balanced_protocol():
    config = validate_config({})
    assert len(config["methods"]) * len(config["seeds"]) == 15
    assert config["threshold"] == .5
    assert config["split_seed"] == 2026
