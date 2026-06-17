import pytest

from config import FitConfig


def test_file_list_missing_fails():
    with pytest.raises(FileNotFoundError, match="File list does not exist"):
        FitConfig(file_list="does-not-exist.txt")


def test_file_and_file_list_are_mutually_exclusive(tmp_path):
    file_list = tmp_path / "targets.txt"
    file_list.write_text("target.h5\n")

    with pytest.raises(ValueError, match="either --file or --file-list"):
        FitConfig(file="one.h5", file_list=str(file_list))


def test_at_least_one_data_component_required():
    with pytest.raises(ValueError, match="At least one data component"):
        FitConfig(use_photometry=False, use_spectroscopy=False)


def test_interactive_requires_spectroscopy():
    with pytest.raises(ValueError, match="interactive requires spectroscopy"):
        FitConfig(interactive=True, use_spectroscopy=False)


def test_invalid_redshift_fails():
    with pytest.raises(ValueError, match="redshift"):
        FitConfig(redshift=-1.0)


def test_unknown_cli_argument_fails():
    config = FitConfig(file="target.h5")

    with pytest.raises(SystemExit):
        config.update_from_cli(["--redshfit", "1.0"])


def test_hyphenated_cli_aliases_update_config():
    config = FitConfig()

    config.update_from_cli(
        [
            "--file",
            "target.h5",
            "--out-folder",
            "results/custom",
            "--model",
            "ContinuitySFH",
        ]
    )

    assert config.out_folder == "results/custom"
    assert config.model_type == "ContinuitySFH"


def test_emcee_is_rejected_because_run_py_does_not_wire_it():
    with pytest.raises(ValueError, match="emcee"):
        FitConfig(emcee=True)


def test_no_dynesty_is_rejected_until_optimize_only_writing_exists():
    with pytest.raises(ValueError, match="no-dynesty"):
        FitConfig(dynesty=False, optimize=True)
