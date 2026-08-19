"""R11 tests for PipelineConfig's canonical backing store."""

from pathlib import Path
import pickle

from qmri_neuropipe.core.config import PipelineConfig


def test_config_data_seeds_typed_attribute_views(tmp_path):
    config = PipelineConfig(
        config_data={
            "bids_dir": str(tmp_path / "bids"),
            "output_dir": str(tmp_path / "out"),
            "models_dir": str(tmp_path / "models"),
            "n_cpus": 6,
            "skip_existing": False,
        }
    )

    assert config.bids_dir == tmp_path / "bids"
    assert config.output_dir == tmp_path / "out"
    assert config.models_dir == tmp_path / "models"
    assert config.work_dir == tmp_path / "out" / "work"
    assert config.n_cpus == config.get("n_cpus") == 6
    assert config.skip_existing is config.get("skip_existing") is False


def test_explicit_constructor_values_override_store_values(tmp_path):
    config = PipelineConfig(
        output_dir=tmp_path / "explicit-out",
        n_cpus=8,
        config_data={
            "output_dir": str(tmp_path / "stored-out"),
            "n_cpus": 2,
        },
    )

    assert config.output_dir == tmp_path / "explicit-out"
    assert config.config_data["output_dir"] == tmp_path / "explicit-out"
    assert config.n_cpus == config.config_data["n_cpus"] == 8


def test_attribute_set_and_config_data_view_share_one_value(tmp_path):
    config = PipelineConfig(output_dir=tmp_path / "out")

    config.n_cpus = 4
    assert config.config_data["n_cpus"] == 4
    assert config.to_dict()["n_cpus"] == 4

    config.config_data["n_cpus"] = 10
    assert config.n_cpus == 10
    assert config.get("n_cpus") == 10


def test_dotted_set_and_standard_set_use_same_store(tmp_path):
    config = PipelineConfig(output_dir=tmp_path / "out")

    config.set("n_cpus", 12)
    config.set("dmri.preprocessing.denoising.enabled", True)
    config.set("bids_dir", str(tmp_path / "bids"))

    assert config.n_cpus == config.config_data["n_cpus"] == 12
    assert config.get("dmri.preprocessing.denoising.enabled") is True
    assert config.config_data["dmri"]["preprocessing"]["denoising"] == {
        "enabled": True
    }
    assert config.bids_dir == Path(tmp_path / "bids")


def test_to_dict_round_trip_preserves_canonical_values(tmp_path):
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        n_cpus=3,
        tracker=object(),
        config_data={"dmri": {"modeling": {"dti": {"enabled": True}}}},
    )
    serialized = config.to_dict()

    assert serialized["bids_dir"] == str(tmp_path / "bids")
    assert serialized["output_dir"] == str(tmp_path / "out")
    assert "tracker" not in serialized

    restored = PipelineConfig.from_dict(serialized)
    assert restored.bids_dir == config.bids_dir
    assert restored.output_dir == config.output_dir
    assert restored.work_dir == config.work_dir
    assert restored.n_cpus == 3
    assert restored.get("dmri.modeling.dti.enabled") is True


def test_validation_mutation_is_visible_in_store(tmp_path):
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        debug=True,
    )

    config.validate()

    assert config.log_level == "DEBUG"
    assert config.config_data["log_level"] == "DEBUG"
    assert config.to_dict()["log_level"] == "DEBUG"


def test_config_remains_pickle_safe_for_parallel_workers(tmp_path):
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        config_data={"jobs": 3, "dmri": {"modeling": {}}},
    )

    restored = pickle.loads(pickle.dumps(config))

    assert restored.to_dict() == config.to_dict()
    assert restored.get("jobs") == 3
