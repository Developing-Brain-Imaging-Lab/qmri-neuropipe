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


def test_container_definitions_upgrade_freesurfer_torch_with_cpu_and_cuda_support():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    definition = (REPO_ROOT / "Apptainer.def").read_text(encoding="utf-8")

    for source in (dockerfile, definition):
        assert "FREESURFER_TORCH_VERSION=2.4.1" in source
        assert "https://download.pytorch.org/whl/cu121" in source
        assert 'torch.version.cuda is not None' in source
        # A CUDA-enabled wheel must also pass a host-independent CPU tensor
        # operation during the image build.
        assert "torch.ones(1).item() == 1" in source
        assert 'python/bin/python3.8' in source


def test_apptainer_runtime_test_is_lightweight_and_host_independent():
    definition = (REPO_ROOT / "Apptainer.def").read_text(encoding="utf-8")
    runtime_test = definition.split("%test", maxsplit=1)[1]

    assert "/opt/conda/bin/python -m pip check" in runtime_test
    assert "test -x /opt/conda/bin/qmri-neuropipe" in runtime_test
    assert "import ants" not in runtime_test
    assert "import jax" not in runtime_test
    assert "load_checkpoint" not in runtime_test


def test_tractseg_extra_supplies_undeclared_pytorch_dependency():
    project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    tractseg_extra = project.split("tractseg = [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    all_extra = project.split("all = [", maxsplit=1)[1].split("]", maxsplit=1)[0]

    assert '"TractSeg>=2.10,<2.11"' in tractseg_extra
    assert '"torch>=2.0"' in tractseg_extra
    assert '"torch>=2.0"' in all_extra


def test_container_definitions_install_tortoise_fftw_runtime_and_check_loader():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    definition = (REPO_ROOT / "Apptainer.def").read_text(encoding="utf-8")

    for source in (dockerfile, definition):
        assert "libfftw3-double3" in source
        assert "libfftw3-single3" in source
        assert "TORTOISEProcess_cuda" in source
        assert "ldd" in source
        assert "Missing TORTOISEProcess_cuda libraries" in source

    assert "libfftw3\\.so\\.3" in definition
    assert "libfftw3f\\.so\\.3" in definition
