from types import SimpleNamespace

import numpy as np
import pytest

sps_module = pytest.importorskip("sps")


class FakeSSP:
    def __init__(self):
        self.params = {}
        self.lsf = None

    def set_lsf(self, wave_lsf, delta_v):
        self.lsf = (wave_lsf, delta_v)


class FakeSPS:
    def __init__(self, zcontinuous=1):
        self.zcontinuous = zcontinuous
        self.ssp = FakeSSP()


def test_get_lsf_rejects_no_positive_dispersion():
    builder = sps_module.ProspectorSPSBuilder(
        SimpleNamespace(redshift=0.0),
        data_handler=SimpleNamespace(spectroscopy=None),
        model=SimpleNamespace(model_params={"agebins": {}}),
    )

    with pytest.raises(ValueError, match="finite positive"):
        builder._get_lsf(np.array([5000.0, 5100.0]), np.array([np.nan, -1.0]))


def test_get_lsf_rejects_no_miles_overlap():
    builder = sps_module.ProspectorSPSBuilder(
        SimpleNamespace(redshift=0.0),
        data_handler=SimpleNamespace(spectroscopy=None),
        model=SimpleNamespace(model_params={"agebins": {}}),
    )

    wave_lsf, delta_v = builder._get_lsf(
        np.array([1000.0, 1100.0]), np.array([100.0, 100.0])
    )

    assert len(wave_lsf) == 0
    assert len(delta_v) == 0


def test_photometry_only_build_sps_does_not_access_spectroscopy(monkeypatch):
    monkeypatch.setattr(sps_module, "FastStepBasis", FakeSPS)

    config = SimpleNamespace(
        z_continuous=1,
        add_sigmav=True,
        use_spectroscopy=False,
        dispersion_file="unused.fits",
        redshift=0.0,
    )
    data_handler = SimpleNamespace(spectroscopy=None)
    model = SimpleNamespace(model_params={"agebins": {}})

    built = sps_module.ProspectorSPSBuilder(config, data_handler, model).build_sps()

    assert isinstance(built, FakeSPS)
    assert built.ssp.lsf is None
