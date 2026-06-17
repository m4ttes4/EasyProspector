import numpy as np
import pytest
import h5py

import data_reader
from config import FitConfig
from data_reader import GalaxyDataManager


class FakeFilter:
    def __init__(self, name):
        self.name = str(name)
        self.wave_effective = float(len(self.name) * 100)


class FakeObservate:
    Filter = FakeFilter


def write_hdf5(path, photometry=True, spectroscopy=True, metadata=True, **overrides):
    with h5py.File(path, "w") as h5:
        group = h5.create_group("V1")

        if photometry:
            phot = group.create_group("Photometry")
            phot.create_dataset(
                "flux", data=overrides.get("phot_flux", np.array([1.0, 2.0]))
            )
            phot.create_dataset(
                "flux_err", data=overrides.get("phot_flux_err", np.array([0.1, 0.2]))
            )
            phot.create_dataset(
                "filters", data=overrides.get("filters", np.array([b"f1", b"f22"]))
            )
            if "phot_mask" in overrides:
                phot.create_dataset("mask", data=overrides["phot_mask"])

        if spectroscopy:
            spec = group.create_group("Spectroscopy")
            spec.create_dataset(
                "wavelength", data=overrides.get("wave", np.array([5000.0, 5100.0]))
            )
            spec.create_dataset(
                "flux", data=overrides.get("spec_flux", np.array([1.0, 1.5]))
            )
            spec.create_dataset(
                "flux_err", data=overrides.get("spec_flux_err", np.array([0.1, 0.2]))
            )
            if "spec_mask" in overrides:
                spec.create_dataset("mask", data=overrides["spec_mask"])

        if metadata:
            meta = group.create_group("Metadata")
            meta.create_dataset("redshift", data=overrides.get("redshift", 1.0))


def test_missing_requested_photometry_fails(tmp_path):
    path = tmp_path / "galaxy.h5"
    write_hdf5(path, photometry=False)

    manager = GalaxyDataManager(FitConfig(file=str(path)))

    with pytest.raises(ValueError, match="Photometry is enabled"):
        manager.load_data()


def test_empty_requested_photometry_group_fails(tmp_path):
    path = tmp_path / "galaxy.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("V1")
        group.create_group("Photometry")
        spec = group.create_group("Spectroscopy")
        spec.create_dataset("wavelength", data=np.array([5000.0, 5100.0]))
        spec.create_dataset("flux", data=np.array([1.0, 1.5]))
        spec.create_dataset("flux_err", data=np.array([0.1, 0.2]))

    manager = GalaxyDataManager(FitConfig(file=str(path), use_spectroscopy=False))

    with pytest.raises(ValueError, match="missing required"):
        manager.load_data()


def test_missing_requested_spectroscopy_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data_reader, "observate", FakeObservate)
    path = tmp_path / "galaxy.h5"
    write_hdf5(path, spectroscopy=False)

    manager = GalaxyDataManager(FitConfig(file=str(path)))

    with pytest.raises(ValueError, match="Spectroscopy is enabled"):
        manager.load_data()


def test_empty_requested_spectroscopy_group_fails(tmp_path):
    path = tmp_path / "galaxy.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("V1")
        group.create_group("Spectroscopy")

    manager = GalaxyDataManager(FitConfig(file=str(path), use_photometry=False))

    with pytest.raises(ValueError, match="missing required"):
        manager.load_data()


def test_photometry_length_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data_reader, "observate", FakeObservate)
    path = tmp_path / "galaxy.h5"
    write_hdf5(path, phot_flux_err=np.array([0.1]))

    manager = GalaxyDataManager(FitConfig(file=str(path), use_spectroscopy=False))

    with pytest.raises(ValueError, match="matching lengths"):
        manager.load_data()


def test_spectroscopy_length_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data_reader, "observate", FakeObservate)
    path = tmp_path / "galaxy.h5"
    write_hdf5(path, wave=np.array([5000.0]))

    manager = GalaxyDataManager(FitConfig(file=str(path), use_photometry=False))

    with pytest.raises(ValueError, match="matching lengths"):
        manager.load_data()


def test_invalid_mask_values_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(data_reader, "observate", FakeObservate)
    path = tmp_path / "galaxy.h5"
    write_hdf5(path, phot_mask=np.array([0, 2]))

    manager = GalaxyDataManager(FitConfig(file=str(path), use_spectroscopy=False))

    with pytest.raises(ValueError, match="mask must contain"):
        manager.load_data()


def test_all_invalid_photometry_after_filtering_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data_reader, "observate", FakeObservate)
    path = tmp_path / "galaxy.h5"
    write_hdf5(path, phot_flux=np.array([-1.0, -2.0]))

    manager = GalaxyDataManager(FitConfig(file=str(path), use_spectroscopy=False))

    with pytest.raises(ValueError, match="no valid points"):
        manager.load_data()


def test_photometry_only_obs_dict_does_not_index_spectroscopy(tmp_path, monkeypatch):
    monkeypatch.setattr(data_reader, "observate", FakeObservate)
    path = tmp_path / "galaxy.h5"
    write_hdf5(path, spectroscopy=False)

    manager = GalaxyDataManager(FitConfig(file=str(path), use_spectroscopy=False))
    manager.load_data()
    obs = manager.to_dict()

    assert obs["spectrum"] is None
    assert obs["maggies"].tolist() == [1.0, 2.0]


def test_spectroscopy_only_obs_dict_does_not_index_photometry(tmp_path):
    path = tmp_path / "galaxy.h5"
    write_hdf5(path, photometry=False)

    manager = GalaxyDataManager(FitConfig(file=str(path), use_photometry=False))
    manager.load_data()
    obs = manager.to_dict()

    assert obs["maggies"] is None
    assert obs["spectrum"].tolist() == [1.0, 1.5]


def test_invalid_metadata_redshift_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data_reader, "observate", FakeObservate)
    path = tmp_path / "galaxy.h5"
    write_hdf5(path, redshift=np.nan)

    manager = GalaxyDataManager(FitConfig(file=str(path), use_spectroscopy=False))

    with pytest.raises(ValueError, match="Metadata/redshift"):
        manager.load_data()
