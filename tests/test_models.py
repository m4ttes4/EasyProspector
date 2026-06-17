import pytest

from config import FitConfig


models = pytest.importorskip("models")


def test_registered_model_names_include_existing_models():
    assert "AmirModel" in models.available_models()
    assert "ContinuitySFH" in models.available_models()


def test_build_model_uses_aliases():
    config = FitConfig(model_type="continuity", redshift=1.0)

    model = models.build_model(config)

    assert isinstance(model, models.ContinuitySFH)
    assert "agebins" in model.model_params


def test_amir_model_requires_redshift():
    config = FitConfig(model_type="AmirModel", use_spectroscopy=False)

    with pytest.raises(ValueError, match="requires a redshift"):
        models.build_model(config)


def test_unknown_model_fails_with_choices():
    config = FitConfig(model_type="NoSuchModel")

    with pytest.raises(ValueError, match="Available models"):
        models.build_model(config)


def test_fixed_z_fixes_redshift_for_continuity_model():
    config = FitConfig(model_type="ContinuitySFH", redshift=1.0, fixed_z=True)

    model = models.build_model(config)

    assert model.model_params["zred"]["isfree"] is False
