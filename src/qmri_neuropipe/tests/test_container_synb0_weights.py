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


def test_apptainer_installs_and_checks_segmentation_tools():
    definition = (REPO_ROOT / "Apptainer.def").read_text(encoding="utf-8")

    assert "PYTHON_EXTRAS=dmipy-cuda12," in definition
    assert "hdbet,antspynet,tractseg" in definition
    assert "import ants; import antspynet" in definition
    assert "import dmipy_fit; import jax; import jaxopt" in definition
    assert "import torch; import tractseg" in definition
    assert "antsRegistration TractSeg Tracking hd-bet dwi2response" in definition
    assert "from tractseg.libs.pytorch_utils import load_checkpoint" in definition
    assert "from zipfile import is_zipfile" not in definition
    assert "%test" in definition
    assert "/opt/conda/bin/python -m pip check" in definition


def test_tractseg_extra_supplies_undeclared_pytorch_dependency():
    project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    tractseg_extra = project.split("tractseg = [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    all_extra = project.split("all = [", maxsplit=1)[1].split("]", maxsplit=1)[0]

    assert '"TractSeg>=2.10,<2.11"' in tractseg_extra
    assert '"torch>=2.0"' in tractseg_extra
    assert '"torch>=2.0"' in all_extra
