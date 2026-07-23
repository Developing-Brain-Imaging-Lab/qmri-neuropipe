from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_apptainer_prefetches_synb0_weights_to_persistent_image_path():
    definition = (REPO_ROOT / "Apptainer.def").read_text(encoding="utf-8")

    assert "export DIPY_HOME=/opt/dipy" in definition
    assert "from dipy.data import fetch_synb0_weights; fetch_synb0_weights()" in definition
    assert definition.count("export DIPY_HOME=/opt/dipy") >= 2


def test_docker_prefetches_synb0_weights_to_persistent_image_path():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ENV DIPY_HOME=/opt/dipy" in dockerfile
    assert "from dipy.data import fetch_synb0_weights; fetch_synb0_weights()" in dockerfile
