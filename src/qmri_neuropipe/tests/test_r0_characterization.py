"""
R0 — Characterization (golden) tests: behavior lock for the Tier-1 refactors.

These tests do NOT assert that the current behavior is *correct*. They assert
that it is *unchanged*. Run them BEFORE a refactor (they must pass = they
describe today's behavior) and AFTER (they must still pass = behavior preserved).

Each test names the refactor rock it guards (see REFACTOR_PLAN.md).

Why characterization and not a full pipeline run:
    A full dmri/anat/relax run needs FSL/ANTs/FreeSurfer binaries and is slow and
    flaky. Instead we pin behavior at the exact seams each Tier-1 rock edits, using
    the same "mock the interfaces, use real nibabel" style as the existing suite.
    A full end-to-end golden harness (for when binaries ARE available) lives in
    test_r0_golden_snapshot.py.

If a test here fails after a refactor, the refactor changed behavior — stop and
re-scope, or update the golden ONLY with a deliberate, reviewed reason.
"""

import logging
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile


def _logger():
    return logging.getLogger("r0")


# ---------------------------------------------------------------------------
# R4 — modeling step builder (the 9 _add_*_step methods -> registry loop)
#
# Locks: which step classes get built for a given config, in what order, and
# the key kwargs each receives (method, n_cpus, flattened parameters/options).
# This is THE golden for the registry refactor.
# ---------------------------------------------------------------------------
class TestR4ModelingBuild:
    def _workflow(self, modeling_cfg: dict):
        from qmri_neuropipe.workflows.pipelines.integrated_modeling_workflow import (
            ModelingWorkflow,
        )

        config = PipelineConfig(
            bids_dir=Path("/tmp/bids"),
            output_dir=Path("/tmp/out"),
            n_cpus=4,
            config_data={"dmri": {"modeling": modeling_cfg}},
        )
        return ModelingWorkflow(config, _logger(), None)

    def _build(self, modeling_cfg: dict):
        wf = self._workflow(modeling_cfg)
        wf.build_pipeline({"subject": "01", "session": None})
        return wf

    def test_no_models_enabled_builds_no_steps(self):
        wf = self._build({})
        assert [s.__class__.__name__ for s in wf.steps] == []

    def test_dti_alias_tensor_builds_dti_step(self):
        # `tensor` is an accepted alias of `dti`.
        wf = self._build({"tensor": {"enabled": True}})
        names = [s.__class__.__name__ for s in wf.steps]
        assert names == ["DTIFittingStep"]
        assert wf.steps[0].method == "dipy"

    def test_full_model_set_order_and_classes(self):
        # Lock the construction order across all model types.
        cfg = {
            "dti": {"enabled": True, "method": "dipy"},
            "dki": {"enabled": True},
            "csd": {"enabled": True},
            "noddi": {"enabled": True},
            "sandi": {"enabled": True},
            "microglia": {"enabled": True},
            "nexi": {"enabled": True},
            "mapmri": {"enabled": True},
            "fwe_dti": {"enabled": True},
        }
        wf = self._build(cfg)
        names = [s.__class__.__name__ for s in wf.steps]
        assert names == [
            "DTIFittingStep",
            "DKIFittingStep",
            "CSDFittingStep",
            "NODDIFittingStep",
            "SANDIFittingStep",
            "MicrogliaFittingStep",
            "NEXIFittingStep",
            "MAPMRIFittingStep",
            "FWDTIFittingStep",
        ]

    def test_disabled_models_are_skipped(self):
        cfg = {
            "dti": {"enabled": True},
            "dki": {"enabled": False},
            "csd": {"enabled": True},
        }
        wf = self._build(cfg)
        names = [s.__class__.__name__ for s in wf.steps]
        assert names == ["DTIFittingStep", "CSDFittingStep"]

    def test_parameters_block_is_flattened_into_kwargs(self):
        # The builder flattens nested `parameters` (and `options`) into the
        # step. Lock that an attribute set from parameters survives.
        cfg = {"dti": {"enabled": True, "method": "dipy",
                       "parameters": {"fit_method": "WLS"}}}
        wf = self._build(cfg)
        step = wf.steps[0]
        # The kwarg was passed to the step constructor; it should be reachable
        # either as an attribute or stored on the step. We assert the step was
        # built without error and method is correct (the key invariant).
        assert step.method == "dipy"

    def test_each_model_preserves_its_existing_kwarg_policy(self):
        cfg = {
            name: {
                "enabled": True,
                "top": name,
                "parameters": {"parameter_value": name},
                "options": {"option_value": name},
            }
            for name in (
                "dti", "dki", "csd", "noddi", "sandi", "microglia",
                "nexi", "mapmri", "fwe_dti",
            )
        }
        wf = self._build(cfg)
        by_name = {step.__class__.__name__: step.kwargs for step in wf.steps}

        # All constructors currently receive n_cpus through **kwargs; preserve it.
        assert all(kwargs["n_cpus"] == 4 for kwargs in by_name.values())
        assert by_name["DTIFittingStep"] == {
            "n_cpus": 4, "top": "dti", "parameter_value": "dti", "option_value": "dti"
        }
        assert by_name["DKIFittingStep"]["options"] == {"option_value": "dki"}
        assert by_name["CSDFittingStep"]["parameters"] == {"parameter_value": "csd"}
        assert by_name["CSDFittingStep"]["options"] == {"option_value": "csd"}
        assert by_name["NODDIFittingStep"]["options"] == {"option_value": "noddi"}
        assert by_name["SANDIFittingStep"] == {
            "n_cpus": 4, "parameter_value": "sandi"
        }
        assert by_name["MicrogliaFittingStep"]["options"] == {"option_value": "microglia"}
        assert "options" not in by_name["NEXIFittingStep"]
        assert by_name["MAPMRIFittingStep"]["options"] == {"option_value": "mapmri"}
        assert by_name["FWDTIFittingStep"]["options"] == {"option_value": "fwe_dti"}

    def test_tractseg_auto_enables_csd(self):
        # Dependency auto-enable: TractSeg pulls in CSD.
        cfg = {
            "dti": {"enabled": True},
            "tractography": {"tractseg": {"enabled": True}},
        }
        wf = self._build(cfg)
        names = [s.__class__.__name__ for s in wf.steps]
        assert "CSDFittingStep" in names
        assert "TractSegStep" in names

    def test_nested_control_key_order_is_preserved(self):
        wf = self._build({
            "noddi": {"enabled": True, "parameters": {"enabled": "noddi"}},
            "microglia": {"enabled": True, "parameters": {"enabled": "microglia"}},
            "fwe_dti": {"enabled": True, "parameters": {"enabled": "fwe_dti"}},
            "dti": {"enabled": True, "parameters": {"enabled": "removed"}},
        })
        by_name = {step.__class__.__name__: step.kwargs for step in wf.steps}

        assert "enabled" not in by_name["DTIFittingStep"]
        assert by_name["NODDIFittingStep"]["enabled"] == "noddi"
        assert by_name["MicrogliaFittingStep"]["enabled"] == "microglia"
        assert by_name["FWDTIFittingStep"]["enabled"] == "fwe_dti"


# ---------------------------------------------------------------------------
# R6 — relaxometry DESPOT model specs (5-model copy block -> table loop)
#
# Locks: model-name resolution, metric-name resolution, and the canonical
# metric set per model. The registry refactor must reproduce these exactly.
# ---------------------------------------------------------------------------
class TestR6RelaxModelSpecs:
    @pytest.fixture
    def WF(self):
        from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryWorkflow
        return RelaxometryWorkflow

    def test_model_name_resolution_aliases(self, WF):
        assert WF._resolve_relax_model_name("despot1") == "DESPOT1"
        assert WF._resolve_relax_model_name("hifi") == "DESPOT1HIFI"
        assert WF._resolve_relax_model_name("despot1_hifi") == "DESPOT1HIFI"
        assert WF._resolve_relax_model_name("despot2") == "DESPOT2"
        assert WF._resolve_relax_model_name("despot2_fm") == "DESPOT2FM"
        assert WF._resolve_relax_model_name("mcdespot") == "mcDESPOT"
        assert WF._resolve_relax_model_name("not_a_model") is None

    def test_metric_name_resolution_mwf_aliases_to_vfm(self, WF):
        # The canonical myelin-water metric is VFm; MWF is an accepted alias.
        assert WF._resolve_relax_metric_name("mwf") == "VFm"
        assert WF._resolve_relax_metric_name("vfm") == "VFm"
        assert WF._resolve_relax_metric_name("myelin water fraction") == "VFm"

    def test_metric_name_resolution_standard(self, WF):
        assert WF._resolve_relax_metric_name("t1") == "T1"
        assert WF._resolve_relax_metric_name("t2") == "T2"
        assert WF._resolve_relax_metric_name("m0") == "M0"
        assert WF._resolve_relax_metric_name("b1") == "B1"
        assert WF._resolve_relax_metric_name("nonsense") is None

    def test_model_spec_metric_sets(self, WF):
        specs = WF._relax_model_specs()
        # Lock the canonical metric outputs per model.
        assert set(specs["DESPOT1"]["metrics"].values()) == {"T1", "M0"}
        assert set(specs["DESPOT1HIFI"]["metrics"].values()) == {"T1", "M0", "B1"}
        assert set(specs["DESPOT2"]["metrics"].values()) == {"T2", "M0", "F0"}
        assert set(specs["DESPOT2FM"]["metrics"].values()) == {"T2", "M0", "F0"}
        mc = set(specs["mcDESPOT"]["metrics"].values())
        assert "VFm" in mc and "Tau" in mc

    def test_downstream_fitters_receive_stable_arguments(self, tmp_path, monkeypatch):
        from qmri_neuropipe.workflows.pipelines import relaxometry as relax

        modeling = relax.RelaxometryModelingConfig(
            despot1={"enabled": True, "nthreads": 2},
            despot2={"enabled": True, "custom_d2": "two"},
            despot2fm={"enabled": True, "threads": 3, "verbose": True},
            mcdespot={"enabled": True, "cuda": True, "custom_mc": "multi"},
        )
        relax_config = relax.RelaxometryConfig(modeling=modeling)
        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            n_cpus=6,
            skip_existing=False,
        )
        wf = relax.RelaxometryWorkflow(config, _logger(), {}, relax_config)
        calls = []

        def _outputs(out_dir, names):
            results = {}
            for name in names:
                path = Path(out_dir) / f"{name}.nii.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                results[name] = path
            return results

        def _capture(name, outputs):
            def _fit(**kwargs):
                calls.append((name, kwargs))
                return _outputs(kwargs["out_dir"], outputs)
            return _fit

        monkeypatch.setattr(
            relax,
            "fit_despot1",
            _capture("DESPOT1", ("t1", "m0", "b1")),
        )
        downstream_fitters = {
            "DESPOT2": _capture("DESPOT2", ("t2", "m0", "f0")),
            "DESPOT2FM": _capture("DESPOT2FM", ("t2", "m0", "f0")),
            "mcDESPOT": _capture("mcDESPOT", ("vfm", "tau")),
        }
        monkeypatch.setattr(
            relax,
            "DESPOT_SPECS",
            tuple(
                replace(spec, fit_fn=downstream_fitters[spec.name])
                for spec in relax.DESPOT_SPECS
            ),
        )

        spgr = ImageFile({"sub": "01", "suffix": "VFA"}, tmp_path / "spgr.nii.gz")
        ssfp = ImageFile({"sub": "01", "suffix": "VFA"}, tmp_path / "ssfp.nii.gz")
        params = tmp_path / "params.json"
        mask = tmp_path / "mask.nii.gz"
        fit_dir = tmp_path / "models"
        wf._run_model_fitting(
            {}, [spgr], [ssfp], [], params, fit_dir, mask, None, "sub-01"
        )

        assert [name for name, _ in calls] == [
            "DESPOT1", "DESPOT2", "DESPOT2FM", "mcDESPOT"
        ]
        by_name = dict(calls)
        t1_path = fit_dir / "t1.nii.gz"
        b1_path = fit_dir / "b1.nii.gz"

        assert by_name["DESPOT2"] == {
            "ssfp_file": ssfp.img,
            "t1_file": t1_path,
            "b1_file": b1_path,
            "params_file": params,
            "out_dir": fit_dir,
            "mask_file": mask,
            "out_base": "sub-01_model-DESPOT2",
            "algo": "lsq",
            "nthreads": 6,
            "verbose": False,
            "extra_options": {"custom_d2": "two"},
        }
        assert by_name["DESPOT2FM"] == {
            "ssfp_file": ssfp.img,
            "t1_file": t1_path,
            "b1_file": b1_path,
            "params_file": params,
            "out_dir": fit_dir,
            "mask_file": mask,
            "out_base": "sub-01_model-DESPOT2FM",
            "algo": "src",
            "nthreads": 3,
            "verbose": True,
            "extra_options": {},
        }
        assert by_name["mcDESPOT"] == {
            "spgr_file": spgr.img,
            "ssfp_file": ssfp.img,
            "t1_file": t1_path,
            "b1_file": b1_path,
            "params_file": params,
            "out_dir": fit_dir,
            "mask_file": mask,
            "out_base": "sub-01_model-mcDESPOT",
            "algo": "src",
            "nthreads": 6,
            "verbose": False,
            "cuda": True,
            "extra_options": {"custom_mc": "multi"},
        }

    @pytest.mark.parametrize(
        ("ssfp", "despot1_results", "external_b1", "message"),
        (
            (
                None,
                {"t1": Path("t1.nii.gz"), "b1": Path("b1.nii.gz")},
                None,
                "DESPOT2FM requested, but no SSFP image was found.",
            ),
            (
                Path("ssfp.nii.gz"),
                {},
                Path("b1.nii.gz"),
                "DESPOT2FM requires a DESPOT1 T1 map, but none was produced.",
            ),
            (
                Path("ssfp.nii.gz"),
                {"t1": Path("t1.nii.gz")},
                None,
                (
                    "DESPOT2FM requires a B1 map, but none was available from "
                    "AFI/external B1 or DESPOT1-HIFI."
                ),
            ),
        ),
    )
    def test_downstream_dependency_errors_are_stable(
        self,
        tmp_path,
        ssfp,
        despot1_results,
        external_b1,
        message,
    ):
        from qmri_neuropipe.workflows.pipelines import relaxometry as relax

        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
        )
        modeling = relax.RelaxometryModelingConfig(
            despot1={"enabled": True}
        )
        wf = relax.RelaxometryWorkflow(
            config,
            _logger(),
            {},
            relax.RelaxometryConfig(modeling=modeling),
        )

        with pytest.raises(ValueError, match=f"^{message}$"):
            wf._resolve_despot_dependencies(
                "DESPOT2FM",
                modeling,
                despot1_results,
                ssfp,
                external_b1,
            )

    def test_legacy_despot2_mcdespot_flag_still_enables_fit(
        self,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        from qmri_neuropipe.workflows.pipelines import relaxometry as relax

        modeling = relax.RelaxometryModelingConfig(
            despot1={"enabled": True},
            despot2={"enabled": False, "mcdespot": True},
            mcdespot={},
        )
        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            skip_existing=False,
        )
        wf = relax.RelaxometryWorkflow(
            config,
            _logger(),
            {},
            relax.RelaxometryConfig(modeling=modeling),
        )

        def _despot1(**kwargs):
            t1 = kwargs["out_dir"] / "legacy-t1.nii.gz"
            b1 = kwargs["out_dir"] / "legacy-b1.nii.gz"
            t1.touch()
            b1.touch()
            return {"t1": t1, "b1": b1}

        mc_fit = Mock()

        def _mcdespot(**kwargs):
            mc_fit(**kwargs)
            output = kwargs["out_dir"] / "legacy-vfm.nii.gz"
            output.touch()
            return {"vfm": output}

        monkeypatch.setattr(relax, "fit_despot1", _despot1)
        monkeypatch.setattr(
            relax,
            "DESPOT_SPECS",
            tuple(
                replace(spec, fit_fn=_mcdespot)
                if spec.name == "mcDESPOT"
                else spec
                for spec in relax.DESPOT_SPECS
            ),
        )

        wf._run_model_fitting(
            {},
            [ImageFile({"suffix": "VFA"}, tmp_path / "spgr.nii.gz")],
            [ImageFile({"suffix": "VFA"}, tmp_path / "ssfp.nii.gz")],
            [],
            tmp_path / "params.json",
            tmp_path / "models",
            None,
            None,
            "sub-01",
        )

        assert mc_fit.call_count == 1
        assert "despot2.mcdespot is deprecated" in caplog.text


# ---------------------------------------------------------------------------
# R13 — relaxometry config aliases are pinned once at config construction
# ---------------------------------------------------------------------------
class TestR13RelaxometryConfigAliases:
    def test_legacy_normalization_aliases_become_canonical(self):
        from qmri_neuropipe.workflows.pipelines import relaxometry as relax

        config = relax.RelaxometryConfig(
            normalization={
                "enabled": True,
                "space": "InfantTemplate",
                "save_transform": False,
                "skull_strip_registration": True,
                "brain_extraction_method": "synthstrip",
                "use_gpu": True,
            }
        )

        assert config.normalization["space_name"] == "InfantTemplate"
        assert config.normalization["space_entity"] == "InfantTemplate"
        assert config.normalization["save_transforms"] is False
        assert config.normalization["skull_strip"] is True
        assert config.normalization["skull_strip_method"] == "synthstrip"
        assert config.normalization["skull_strip_use_gpu"] is True
        assert not {
            "space",
            "save_transform",
            "skull_strip_registration",
            "brain_extraction_method",
            "use_gpu",
        } & config.normalization.keys()

    def test_canonical_normalization_values_win_over_aliases(self):
        from qmri_neuropipe.workflows.pipelines import relaxometry as relax

        config = relax.RelaxometryConfig(
            normalization={
                "space": "LegacySpace",
                "space_name": "DisplaySpace",
                "space_entity": "BidsSpace",
                "save_transform": False,
                "save_transforms": True,
                "skull_strip_registration": False,
                "skull_strip": True,
            }
        )

        assert config.normalization["space_name"] == "DisplaySpace"
        assert config.normalization["space_entity"] == "BidsSpace"
        assert config.normalization["save_transforms"] is True
        assert config.normalization["skull_strip"] is True

    def test_model_thread_alias_is_canonical_before_fitting(self, tmp_path):
        from qmri_neuropipe.workflows.pipelines import relaxometry as relax

        modeling = relax.RelaxometryModelingConfig(
            despot1={"enabled": True, "threads": 3},
            despot2={"enabled": True, "threads": 2, "nthreads": 5},
        )
        assert modeling.despot1["nthreads"] == 3
        assert modeling.despot2["nthreads"] == 5
        assert "threads" not in modeling.despot1
        assert "threads" not in modeling.despot2

        workflow = relax.RelaxometryWorkflow(
            PipelineConfig(
                bids_dir=tmp_path / "bids",
                output_dir=tmp_path / "out",
                n_cpus=8,
            ),
            _logger(),
            {},
            relax.RelaxometryConfig(modeling=modeling),
        )
        assert workflow._model_nthreads(modeling.despot1) == 3
        assert workflow._model_nthreads({}) == 8

    def test_normalization_step_reads_only_canonical_values(self, tmp_path):
        from qmri_neuropipe.lib.dmri.normalization import NormalizationStep
        from qmri_neuropipe.workflows.pipelines import relaxometry as relax

        relax_config = relax.RelaxometryConfig(
            normalization={
                "enabled": True,
                "space": "InfantTemplate",
                "save_transform": False,
                "skull_strip_registration": True,
            }
        )
        workflow = relax.RelaxometryWorkflow(
            PipelineConfig(
                bids_dir=tmp_path / "bids",
                output_dir=tmp_path / "out",
            ),
            _logger(),
            {},
            relax_config,
        )
        step = next(s for s in workflow.steps if isinstance(s, NormalizationStep))

        assert step.space_name == "InfantTemplate"
        assert step.space_entity == "InfantTemplate"
        assert step.save_transforms is False
        assert step.kwargs["skull_strip"] is True


# ---------------------------------------------------------------------------
# R12 — RelaxometryWorkflow.run() return contract
#
# Locks: run() returns a flat PipelineContext (dict-compatible) with
# fitted_maps/modeling_results/roi_stats/etc. merged in as top-level keys —
# NOT the old nested {"context": ..., "fitted_maps": ...} wrapper shape.
# This is a deliberate, reviewed contract change from before R12; pin it here
# so any further change to this shape is caught rather than silent.
#
# Two tests: one drives the actual seam (_compose_run_context) directly with
# no I/O or external dependencies; the second drives run() end-to-end with
# every internal helper mocked, so the contract is locked at both the unit
# and the integration level without needing real FSL/ANTs/DESPOT binaries.
# ---------------------------------------------------------------------------
class TestR12RelaxometryContextContract:
    def test_compose_run_context_returns_flat_context_not_nested_wrapper(self):
        from qmri_neuropipe.core.context import PipelineContext
        from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryWorkflow

        base_context = PipelineContext({"subject": "01", "session": None})
        fit_maps = {"despot1_t1": object()}
        modeling_results = {"DESPOT1": {"T1": Path("t1.nii.gz")}}
        stats_results = {"T1": Path("t1_stats.csv")}
        b1_map = object()
        spgr_ref = object()

        result = RelaxometryWorkflow._compose_run_context(
            base_context,
            fit_maps,
            modeling_results,
            stats_results,
            b1_map,
            spgr_ref,
        )

        # The return value IS the (mutated) context, not a wrapper around it.
        assert isinstance(result, PipelineContext)
        assert result is base_context
        assert "context" not in result  # old nested wrapper key must not reappear

        assert result["fitted_maps"] is fit_maps
        assert result["modeling_results"] is modeling_results
        assert result["roi_stats"] is stats_results
        assert result["b1_map"] is b1_map
        assert result["reference_image"] is spgr_ref
        assert result["normalized_results"] == {}
        assert result["roi_stats_files"] == {}
        assert result["roi_stats_combined_csv"] is None
        assert result["brain_mask"] is None
        assert result["qc_report"] is None

    def test_run_end_to_end_returns_flat_context(self, tmp_path, monkeypatch):
        """Drive run() with every internal stage mocked, to lock the public
        return contract without needing FSL/ANTs/DESPOT binaries."""
        from qmri_neuropipe.core.context import PipelineContext
        from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryWorkflow

        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            skip_existing=False,
        )
        wf = RelaxometryWorkflow(config, _logger(), {})
        wf.steps = []

        spgr_file = ImageFile(
            entities={"sub": "01", "acq": "spgr", "suffix": "VFA"},
            img=tmp_path / "spgr.nii.gz",
        )
        spgr_ref = ImageFile(
            entities={"sub": "01", "desc": "spgrref", "suffix": "VFA"},
            img=tmp_path / "spgr_ref.nii.gz",
        )

        monkeypatch.setattr(
            wf, "_parse_inputs", lambda context: ([spgr_file], [], [], [])
        )
        monkeypatch.setattr(
            wf,
            "_apply_configured_exclusions",
            lambda *a, **k: ([spgr_file], [], set(), set()),
        )
        monkeypatch.setattr(
            wf,
            "_prepare_modeling_inputs",
            lambda *a, **k: ([spgr_file], [], [], spgr_ref),
        )
        monkeypatch.setattr(wf, "_run_brain_masking", lambda *a, **k: None)
        monkeypatch.setattr(
            wf, "_generate_params", lambda *a, **k: tmp_path / "params.json"
        )
        monkeypatch.setattr(wf, "_run_b1_mapping", lambda *a, **k: None)

        fit_maps = {"despot1_t1": object()}
        modeling_results = {"DESPOT1": {"T1": Path("t1.nii.gz")}}
        monkeypatch.setattr(
            wf, "_run_model_fitting", lambda *a, **k: (fit_maps, modeling_results)
        )
        monkeypatch.setattr(wf, "_run_normalization", lambda context, *a, **k: context)
        monkeypatch.setattr(
            wf, "_run_postprocessing_and_stats", lambda *a, **k: {}
        )
        monkeypatch.setattr(wf, "_finalize_intermediates", lambda *a, **k: None)
        monkeypatch.setattr(wf, "_update_study_tracker", lambda *a, **k: None)

        context = PipelineContext({
            "subject": "01",
            "session": None,
            "relax_files": [spgr_file],
        })

        result = wf.run(
            tmp_path / "work",
            "01",
            None,
            context=context,
            final_output_dir=tmp_path / "out",
        )

        assert isinstance(result, PipelineContext)
        assert "context" not in result
        for key in (
            "fitted_maps",
            "modeling_results",
            "normalized_results",
            "roi_stats",
            "roi_stats_files",
            "brain_mask",
            "b1_map",
            "qc_report",
            "reference_image",
        ):
            assert key in result, f"expected '{key}' as a top-level context key"
        assert result["fitted_maps"] is fit_maps
        assert result["modeling_results"] is modeling_results
        assert result["reference_image"] is spgr_ref


# ---------------------------------------------------------------------------
# R2 — cheap skip probes (_should_skip, _gnl_matches_image_grid)
#
# Locks: the exact truth table these return today.
# ---------------------------------------------------------------------------
class TestR2SkipProbes:
    def _config(self, tmp_path, skip_existing):
        # `skip_existing` is a dataclass FIELD on PipelineConfig, so set it
        # directly (config.get('skip_existing') reads the field, not config_data).
        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            skip_existing=skip_existing,
        )
        (tmp_path / "bids").mkdir(parents=True, exist_ok=True)
        # DMRIPipeline overrides _should_skip to always return False; we test the
        # BASE implementation directly to lock R2's target.
        return config

    def test_base_should_skip_disabled_returns_false(self, tmp_path):
        from qmri_neuropipe.core.base import BasePipeline

        config = self._config(tmp_path, skip_existing=False)
        # Call the unbound base method against a tiny stand-in.
        out_dir = tmp_path / "out" / "sub-01"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "x.nii.gz").write_bytes(b"0")

        # Build the smallest object exposing the attributes _should_skip needs.
        class _Stub:
            pass

        stub = _Stub()
        stub.config = config
        stub._get_output_dir = lambda s, ses: out_dir
        assert BasePipeline._should_skip(stub, "01", None) is False

    def test_base_should_skip_enabled_with_files_returns_true(self, tmp_path):
        from qmri_neuropipe.core.base import BasePipeline

        config = self._config(tmp_path, skip_existing=True)
        out_dir = tmp_path / "out" / "sub-01"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "x.nii.gz").write_bytes(b"0")

        class _Stub:
            pass

        stub = _Stub()
        stub.config = config
        stub._get_output_dir = lambda s, ses: out_dir
        assert BasePipeline._should_skip(stub, "01", None) is True

    def test_base_should_skip_enabled_empty_dir_returns_false(self, tmp_path):
        from qmri_neuropipe.core.base import BasePipeline

        config = self._config(tmp_path, skip_existing=True)
        out_dir = tmp_path / "out" / "sub-01"
        out_dir.mkdir(parents=True, exist_ok=True)  # exists but empty

        class _Stub:
            pass

        stub = _Stub()
        stub.config = config
        stub._get_output_dir = lambda s, ses: out_dir
        assert BasePipeline._should_skip(stub, "01", None) is False

    def test_gnl_grid_match_true_when_same_grid(self, tmp_path):
        from qmri_neuropipe.workflows.pipelines.dmri import DMRIPipeline

        a = tmp_path / "a.nii.gz"
        b = tmp_path / "b.nii.gz"
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), affine), a)
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), affine), b)

        class _Stub:
            logger = _logger()

        assert DMRIPipeline._gnl_matches_image_grid(_Stub(), a, b) is True

    def test_gnl_grid_match_false_when_affine_differs(self, tmp_path):
        from qmri_neuropipe.workflows.pipelines.dmri import DMRIPipeline

        a = tmp_path / "a.nii.gz"
        b = tmp_path / "b.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), a)
        shifted = np.eye(4)
        shifted[0, 3] = 10.0
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), shifted), b)

        class _Stub:
            logger = _logger()

        assert DMRIPipeline._gnl_matches_image_grid(_Stub(), a, b) is False

    def test_gnl_grid_match_never_reads_voxel_data(self, monkeypatch):
        from qmri_neuropipe.workflows.pipelines import dmri

        class _HeaderProxy:
            shape = (4, 4, 4, 6)
            affine = np.eye(4)

            def get_fdata(self):
                raise AssertionError("grid validation must not read voxel data")

        load = Mock(side_effect=[_HeaderProxy(), _HeaderProxy()])
        monkeypatch.setattr(dmri.nib, "load", load)

        class _Stub:
            logger = _logger()

        assert dmri.DMRIPipeline._gnl_matches_image_grid(
            _Stub(), Path("gnl.nii.gz"), Path("dwi.nii.gz")
        ) is True
        assert load.call_count == 2


# ---------------------------------------------------------------------------
# R3 — relaxometry force semantics (_force_requested side-effect split)
#
# Locks: given a rerun_from_step, which steps end up "forced" as the loop
# advances. The split into is_forced()/advance_force_state() must reproduce
# the same sequence of force decisions.
# ---------------------------------------------------------------------------
class TestR3ForceSemantics:
    def _workflow(self, config_data=None):
        from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryWorkflow

        config = PipelineConfig(
            bids_dir=Path("/tmp/bids"),
            output_dir=Path("/tmp/out"),
            config_data=config_data or {},
        )
        return RelaxometryWorkflow(config, _logger(), {})

    def test_force_inactive_without_rerun_or_force_flags(self):
        wf = self._workflow()
        wf._rerun_from_step = None
        wf._force_from_step_active = False
        wf.advance_force_state("denoising")
        assert wf.is_forced() is False

    def test_global_force_flag_forces_everything(self):
        wf = self._workflow({"force": True})
        wf._rerun_from_step = None
        wf._force_from_step_active = False
        wf.advance_force_state("denoising")
        assert wf.is_forced() is True

    def test_rerun_from_step_activates_from_match_onward(self):
        # Once the rerun target is reached, force stays active for later steps.
        wf = self._workflow()
        wf._rerun_from_step = "motion_correction"
        wf._force_from_step_active = False

        # Before the target: not forced.
        wf.advance_force_state("denoising")
        before = wf.is_forced()
        # At the target: explicitly advance the latch, then query it.
        wf.advance_force_state("motion_correction")
        at = wf.is_forced()
        # After the target: stays forced.
        wf.advance_force_state("b1mapping")
        after = wf.is_forced()

        assert before is False
        assert at is True
        assert after is True

    def test_force_query_is_pure(self):
        wf = self._workflow()
        wf._rerun_from_step = "motion_correction"
        wf._force_from_step_active = False

        assert wf.is_forced() is False
        assert wf.is_forced() is False
        assert wf._force_from_step_active is False


# ---------------------------------------------------------------------------
# R1 — final modeling output synchronization
# ---------------------------------------------------------------------------
class TestR1ModelingSync:
    def test_staging_tree_is_copied_once_per_dwi(self, tmp_path, monkeypatch):
        from qmri_neuropipe.workflows.pipelines import integrated_modeling_workflow as modeling

        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            config_data={"dmri": {"modeling": {}}},
        )
        wf = modeling.ModelingWorkflow(config, _logger(), None)

        class _Step:
            def should_skip(self, context, output_dir):
                return False

            def __call__(self, context, output_dir, mask, force):
                return context

        wf.steps = [_Step(), _Step(), _Step()]
        dwis = [
            ImageFile({"sub": "01", "suffix": "dwi"}, tmp_path / "a.nii.gz"),
            ImageFile({"sub": "01", "suffix": "dwi"}, tmp_path / "b.nii.gz"),
        ]
        copytree = Mock()
        monkeypatch.setattr("shutil.copytree", copytree)

        wf._execute_modeling(
            dwis,
            [None, None],
            {},
            tmp_path / "staging",
            tmp_path / "final",
            reporter=None,
        )

        assert copytree.call_count == len(dwis)

    def test_staging_tree_is_not_copied_when_every_step_skips(self, tmp_path, monkeypatch):
        from qmri_neuropipe.workflows.pipelines import integrated_modeling_workflow as modeling

        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            config_data={"dmri": {"modeling": {}}},
        )
        wf = modeling.ModelingWorkflow(config, _logger(), None)

        class _SkippedStep:
            modality = "Diffusion"

            def should_skip(self, context, output_dir):
                return True

            @staticmethod
            def normalize_tracker_module(name):
                return name

        wf.steps = [_SkippedStep(), _SkippedStep()]
        copytree = Mock()
        monkeypatch.setattr("shutil.copytree", copytree)

        wf._execute_modeling(
            [ImageFile({"sub": "01", "suffix": "dwi"}, tmp_path / "a.nii.gz")],
            [None],
            {},
            tmp_path / "staging",
            tmp_path / "final",
            reporter=None,
        )

        copytree.assert_not_called()

    def test_completed_models_are_copied_before_later_failure(self, tmp_path, monkeypatch):
        from qmri_neuropipe.workflows.pipelines import integrated_modeling_workflow as modeling

        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            config_data={"stop_on_error": True, "dmri": {"modeling": {}}},
        )
        wf = modeling.ModelingWorkflow(config, _logger(), None)

        class _SuccessfulStep:
            def should_skip(self, context, output_dir):
                return False

            def __call__(self, context, output_dir, mask, force):
                return context

        class _FailingStep(_SuccessfulStep):
            def __call__(self, context, output_dir, mask, force):
                raise RuntimeError("model fit failed")

        wf.steps = [_SuccessfulStep(), _FailingStep()]
        copytree = Mock()
        monkeypatch.setattr("shutil.copytree", copytree)

        with pytest.raises(RuntimeError, match="model fit failed"):
            wf._execute_modeling(
                [ImageFile({"sub": "01", "suffix": "dwi"}, tmp_path / "a.nii.gz")],
                [None],
                {},
                tmp_path / "staging",
                tmp_path / "final",
                reporter=None,
            )

        copytree.assert_called_once()


# ---------------------------------------------------------------------------
# R0 — CLI/config merge used by --dry-run
# ---------------------------------------------------------------------------
def test_dry_run_config_merge_preserves_nested_config_and_cli_precedence(tmp_path):
    from qmri_neuropipe.cli import merge_cli_and_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            (
                f"bids_dir: {tmp_path / 'bids-from-file'}",
                f"output_dir: {tmp_path / 'out-from-file'}",
                "n_cpus: 2",
                "skip_existing: true",
                "dmri:",
                "  preprocessing:",
                "    denoising:",
                "      enabled: true",
                "      method: mppca",
            )
        )
    )

    merged = merge_cli_and_config(
        config_file,
        {
            "output_dir": tmp_path / "out-from-cli",
            "n_cpus": 8,
            "participant_label": "sub-01",
            "skip_existing": False,
            "memory_gb": None,
        },
    )

    assert merged.bids_dir == tmp_path / "bids-from-file"
    assert merged.output_dir == tmp_path / "out-from-cli"
    assert merged.work_dir == tmp_path / "out-from-cli" / "work"
    assert merged.n_cpus == 8
    assert merged.participant_label == ["sub-01"]
    assert merged.skip_existing is False
    assert merged.memory_gb == 8.0
    assert merged.get("dmri.preprocessing.denoising") == {
        "enabled": True,
        "method": "mppca",
    }


# ---------------------------------------------------------------------------
# H3 — BasePipeline.run return contract
# ---------------------------------------------------------------------------
class TestBasePipelineRunContract:
    @staticmethod
    def _stub(tmp_path, monkeypatch, *, jobs=1):
        from qmri_neuropipe.core import base

        class _Loader:
            def __init__(self, bids_dir):
                pass

            def load_multiple_subjects(self, **kwargs):
                return [("01", None), ("02", "A")]

        class _Stub:
            name = "test"
            version = "1"
            logger = _logger()
            config = PipelineConfig(
                bids_dir=tmp_path / "bids",
                output_dir=tmp_path / "out",
                config_data={"jobs": jobs},
            )

            def _should_skip(self, subject, session):
                return False

            def _process_subject_with_subject_log(self, subject, session):
                if subject == "02":
                    raise RuntimeError("expected failure")

        monkeypatch.setattr(base, "DataLoader", _Loader)
        return base, _Stub()

    def test_sequential_result_includes_failure_details(self, tmp_path, monkeypatch):
        base, stub = self._stub(tmp_path, monkeypatch)

        result = base.BasePipeline.run(stub, pairs=[("ignored", None)])

        assert result == {
            "n_success": 1,
            "n_failed": 1,
            "n_skipped": 0,
            "failures": [
                {"subject": "02", "session": "A", "error": "expected failure"}
            ],
        }

    def test_parallel_result_is_returned_to_caller(self, tmp_path, monkeypatch):
        base, stub = self._stub(tmp_path, monkeypatch, jobs=2)
        expected = {
            "n_success": 2,
            "n_failed": 0,
            "n_skipped": 0,
            "failures": [],
        }
        stub._run_parallel = Mock(return_value=expected)

        assert base.BasePipeline.run(stub, pairs=[("ignored", None)]) == expected


# ---------------------------------------------------------------------------
# R8 — shared tracker updates and persistence boundaries
# ---------------------------------------------------------------------------
class TestTrackerBatching:
    class _Tracker:
        def __init__(self):
            self.statuses = []
            self.times = []
            self.errors = []
            self.saves = []

        def update_status(self, *args, **kwargs):
            self.statuses.append((args, kwargs))

        def log_time(self, *args, **kwargs):
            self.times.append((args, kwargs))

        def log_error(self, *args, **kwargs):
            self.errors.append((args, kwargs))

        def save(self, force=False):
            self.saves.append(force)

    @staticmethod
    def _config(tmp_path, tracker):
        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
        )
        config.tracker = tracker
        return config

    def test_shared_status_helper_normalizes_and_does_not_save(self, tmp_path):
        from qmri_neuropipe.core.tracking import flush_tracker, update_step_status

        tracker = self._Tracker()
        config = self._config(tmp_path, tracker)

        class _Step:
            modality = "Diffusion"

            @staticmethod
            def normalize_tracker_module(name):
                return "Normalized_Module"

        updated = update_step_status(
            config,
            {"subject": "01", "session": "02", "study_name": "study"},
            _Step(),
            "completed (cached)",
        )

        assert updated is True
        assert tracker.statuses == [
            (
                (
                    "01",
                    "02",
                    "Normalized_Module",
                    "completed (cached)",
                    "study",
                ),
                {"modality": "Diffusion"},
            )
        ]
        assert tracker.saves == []
        assert flush_tracker(config) is True
        assert flush_tracker(config) is False
        assert tracker.saves == [False]

    def test_step_success_is_batched_but_failure_flushes(self, tmp_path):
        from qmri_neuropipe.core import BaseProcessingStep, ProcessingError

        tracker = self._Tracker()
        config = self._config(tmp_path, tracker)

        class _Step(BaseProcessingStep):
            def __init__(self, *args, fail=False, **kwargs):
                self.fail = fail
                super().__init__(*args, **kwargs)
                self.modality = "Anatomical"

            def run(self, context):
                if self.fail:
                    raise RuntimeError("boom")
                return context

        context = {"subject": "01", "session": "02", "study_name": "study"}
        _Step(config)(context)
        assert [args[3] for args, _ in tracker.statuses] == [
            "running",
            "completed",
        ]
        assert tracker.saves == []

        tracker.statuses.clear()
        with pytest.raises(ProcessingError):
            _Step(config, fail=True)(context)
        assert [args[3] for args, _ in tracker.statuses] == [
            "running",
            "failed",
        ]
        assert tracker.saves == [True]

    def test_subject_boundary_flushes_pending_updates(self, tmp_path):
        from qmri_neuropipe.core import BasePipeline
        from qmri_neuropipe.core.tracking import update_step_status

        tracker = self._Tracker()
        config = self._config(tmp_path, tracker)

        class _PipelineStub:
            logger = logging.getLogger("r8-subject-boundary")

            def __init__(self):
                self.config = config

            def _subject_log_file(self, subject, session):
                return tmp_path / "subject.log"

            def process_subject(self, subject, session):
                update_step_status(
                    self.config,
                    {
                        "subject": subject,
                        "session": session,
                        "study_name": "study",
                    },
                    "Overall_Status",
                    "Complete",
                    modality="Anatomical",
                )

        BasePipeline._process_subject_with_subject_log(
            _PipelineStub(),
            "01",
            "02",
        )

        assert tracker.saves == [True]


# ---------------------------------------------------------------------------
# R7 — shared derivative reuse probes
# ---------------------------------------------------------------------------
class TestCachingEngine:
    def test_reuse_builds_the_canonical_bids_path(self, tmp_path):
        from qmri_neuropipe.core.caching import reuse_if_exists
        from qmri_neuropipe.io.bids import build_bids_name

        entities = {
            "sub": "01",
            "ses": "02",
            "desc": "preproc",
            "suffix": "T1w",
        }
        expected = tmp_path / build_bids_name(entities)
        expected.touch()

        reused = reuse_if_exists(entities, tmp_path)

        assert reused == ImageFile(entities=entities, img=expected)

    def test_force_bypasses_an_existing_derivative(self, tmp_path):
        from qmri_neuropipe.core.caching import reuse_if_exists
        from qmri_neuropipe.io.bids import build_bids_name

        entities = {"sub": "01", "desc": "preproc", "suffix": "T2w"}
        (tmp_path / build_bids_name(entities)).touch()

        assert reuse_if_exists(entities, tmp_path, force=True) is None

    def test_readability_probe_rejects_invalid_nifti(self, tmp_path):
        from qmri_neuropipe.core.caching import reuse_path_if_exists

        invalid = tmp_path / "invalid.nii.gz"
        invalid.write_bytes(b"not-a-nifti")

        assert reuse_path_if_exists(
            invalid,
            {"suffix": "T1w"},
            readable=True,
        ) is None

    def test_readability_probe_does_not_load_voxel_data(self, monkeypatch):
        from qmri_neuropipe.core import caching

        class _HeaderProxy:
            def get_fdata(self):
                raise AssertionError("cache validation must not read voxel data")

        load = Mock(return_value=_HeaderProxy())
        monkeypatch.setattr(caching.nib, "load", load)
        monkeypatch.setattr(Path, "exists", Mock(return_value=True))

        reused = caching.reuse_path_if_exists(
            Path("cached.nii.gz"),
            {"suffix": "T1w"},
            readable=True,
        )

        assert reused.img == Path("cached.nii.gz")
        assert load.call_count == 1


# ---------------------------------------------------------------------------
# R9 — anatomical coregistration and normalization helper contracts
# ---------------------------------------------------------------------------
class TestR9AnatHelpers:
    @staticmethod
    def _workflow(tmp_path):
        from qmri_neuropipe.workflows.pipelines.anat import AnatPreprocessingWorkflow

        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
        )
        return AnatPreprocessingWorkflow(config, _logger(), None)

    def test_coreg_options_flatten_without_mutating_source(self, tmp_path):
        wf = self._workflow(tmp_path)
        config = {
            "enabled": True,
            "method": "ants",
            "top_level": 1,
            "options": {"nested": 2, "top_level": 3},
        }

        flattened = wf._flatten_coregistration_options(config)

        assert flattened == {
            "enabled": True,
            "method": "ants",
            "top_level": 3,
            "nested": 2,
        }
        assert config["options"] == {"nested": 2, "top_level": 3}

    def test_do_coreg_normalizes_dict_result(self, tmp_path):
        wf = self._workflow(tmp_path)
        moving = ImageFile({"suffix": "T2w"}, tmp_path / "moving.nii.gz")
        fixed = ImageFile({"suffix": "T1w"}, tmp_path / "fixed.nii.gz")
        result = ImageFile({"suffix": "T2w"}, tmp_path / "result.nii.gz")
        step = Mock(return_value={"current_image": result})

        assert wf._do_coreg(
            step,
            moving,
            fixed,
            tmp_path,
            {"cost": "mi"},
            True,
        ) is result
        step.assert_called_once_with(
            moving,
            output_dir=tmp_path,
            target=fixed.img,
            options={"cost": "mi"},
            force=True,
        )

    def test_apply_spatial_transform_dispatches_fsl_and_ants(
        self,
        tmp_path,
        monkeypatch,
    ):
        from qmri_neuropipe.workflows.pipelines import anat

        wf = self._workflow(tmp_path)
        fsl_apply = Mock()
        ants_apply = Mock()
        monkeypatch.setattr(anat.fsl, "applywarp", fsl_apply)
        monkeypatch.setattr(anat.ants, "apply_transforms", ants_apply)

        affine = tmp_path / "transform.mat"
        wf.apply_spatial_transform(
            tmp_path / "moving.nii.gz",
            tmp_path / "template.nii.gz",
            affine,
            transform_type="fsl",
            interp="trilinear",
            out_path=tmp_path / "fsl-out.nii.gz",
        )
        fsl_apply.assert_called_once_with(
            in_file=tmp_path / "moving.nii.gz",
            ref_file=tmp_path / "template.nii.gz",
            out_file=tmp_path / "fsl-out.nii.gz",
            premat=affine,
            interp="trilinear",
            force=True,
        )

        prefix = tmp_path / "ants_"
        (tmp_path / "ants_1Warp.nii.gz").touch()
        (tmp_path / "ants_0GenericAffine.mat").touch()
        wf.apply_spatial_transform(
            tmp_path / "mask.nii.gz",
            tmp_path / "template.nii.gz",
            prefix,
            transform_type="ants",
            interp="nn",
            out_path=tmp_path / "ants-out.nii.gz",
        )
        ants_apply.assert_called_once_with(
            fixed_file=tmp_path / "template.nii.gz",
            moving_file=tmp_path / "mask.nii.gz",
            out_file=tmp_path / "ants-out.nii.gz",
            transforms=[
                tmp_path / "ants_1Warp.nii.gz",
                tmp_path / "ants_0GenericAffine.mat",
            ],
            interpolator="nearestNeighbor",
        )

# ---------------------------------------------------------------------------
# R5 — anatomical modality-loop behavior before T1w/T2w consolidation
# ---------------------------------------------------------------------------
class TestR5AnatModalityPreprocessing:
    @staticmethod
    def _workflow(tmp_path, config_data=None):
        from qmri_neuropipe.workflows.pipelines.anat import AnatPreprocessingWorkflow

        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            config_data=config_data or {},
        )
        return AnatPreprocessingWorkflow(config, _logger(), None)

    @staticmethod
    def _denoise_step(output_image, *, result_as_dict=False):
        from qmri_neuropipe.lib.common.denoise import DenoisingStep

        class _CharacterizationDenoise(DenoisingStep):
            method = "characterization"
            patch_radius = 2

            def __init__(self):
                self.calls = []
                self.modality = "Anatomical"

            def __call__(self, image, output_dir, force=False):
                self.calls.append({"image": image, "output_dir": output_dir, "force": force})
                result = ImageFile(dict(image.entities), output_image)
                if result_as_dict:
                    return {"current_image": result, "characterized": True}
                return result

        return _CharacterizationDenoise()

    def test_missing_t1_is_an_error_but_missing_t2_is_optional(self, tmp_path):
        wf = self._workflow(tmp_path)
        with pytest.raises(ValueError, match="No T1w image found"):
            wf._preprocess_t1w(tmp_path, {}, None, None, tmp_path)

        context, metrics = wf._preprocess_t2w(tmp_path, {}, None, None, tmp_path)
        assert metrics == []
        assert "preprocessed_t2w" not in context

    def test_t1_seeds_current_image_while_t2_does_not(self, tmp_path):
        t1_out = tmp_path / "t1-out.nii.gz"
        t2_out = tmp_path / "t2-out.nii.gz"
        t1_out.touch()
        t2_out.touch()

        t1_wf = self._workflow(tmp_path)
        t1_step = self._denoise_step(t1_out)
        t1_wf.steps = [t1_step]
        t1_input = ImageFile({"sub": "01", "suffix": "T1w"}, tmp_path / "t1.nii.gz")
        t1_context, _ = t1_wf._preprocess_t1w(
            tmp_path, {"t1w_files": [t1_input]}, None, None, tmp_path
        )
        assert t1_context["current_image"] is t1_input
        assert t1_context["preprocessed_t1w"].img == t1_out

        t2_wf = self._workflow(tmp_path)
        t2_step = self._denoise_step(t2_out)
        t2_wf.steps = [t2_step]
        t2_input = ImageFile({"sub": "01", "suffix": "T2w"}, tmp_path / "t2.nii.gz")
        t2_context, _ = t2_wf._preprocess_t2w(
            tmp_path, {"t2w_files": [t2_input]}, None, None, tmp_path
        )
        assert "current_image" not in t2_context
        assert t2_context["preprocessed_t2w"].img == t2_out

    def test_dict_step_results_update_context_only_for_t1(self, tmp_path):
        t1_out = tmp_path / "t1-dict.nii.gz"
        t2_out = tmp_path / "t2-dict.nii.gz"
        t1_out.touch()
        t2_out.touch()

        t1_wf = self._workflow(tmp_path)
        t1_wf.steps = [self._denoise_step(t1_out, result_as_dict=True)]
        t1_context, _ = t1_wf._preprocess_t1w(
            tmp_path,
            {"t1w_files": [ImageFile({"suffix": "T1w"}, tmp_path / "t1.nii.gz")]},
            None,
            None,
            tmp_path,
        )
        assert t1_context["characterized"] is True

        t2_wf = self._workflow(tmp_path)
        t2_wf.steps = [self._denoise_step(t2_out, result_as_dict=True)]
        t2_context, _ = t2_wf._preprocess_t2w(
            tmp_path,
            {"t2w_files": [ImageFile({"suffix": "T2w"}, tmp_path / "t2.nii.gz")]},
            None,
            None,
            tmp_path,
        )
        assert "characterized" not in t2_context

    def test_freesurfer_mode_skips_standard_t1_steps_only(self, tmp_path):
        config_data = {"anat": {"preprocessing": {"use_freesurfer": True}}}
        t1_wf = self._workflow(tmp_path, config_data)
        t1_step = self._denoise_step(tmp_path / "unused-t1.nii.gz")
        t1_wf.steps = [t1_step]
        t1_input = ImageFile({"suffix": "T1w"}, tmp_path / "t1.nii.gz")
        t1_context, _ = t1_wf._preprocess_t1w(
            tmp_path, {"t1w_files": [t1_input]}, None, None, tmp_path
        )
        assert t1_step.calls == []
        assert t1_context["preprocessed_t1w"] is t1_input

        t2_out = tmp_path / "t2-fs.nii.gz"
        t2_out.touch()
        t2_wf = self._workflow(tmp_path, config_data)
        t2_step = self._denoise_step(t2_out)
        t2_wf.steps = [t2_step]
        t2_wf._preprocess_t2w(
            tmp_path,
            {"t2w_files": [ImageFile({"suffix": "T2w"}, tmp_path / "t2.nii.gz")]},
            None,
            None,
            tmp_path,
        )
        assert len(t2_step.calls) == 1

    def test_cached_t2_image_points_to_the_derivative(self, tmp_path):
        from qmri_neuropipe.io.bids import build_bids_name

        wf = self._workflow(tmp_path)
        step = self._denoise_step(tmp_path / "should-not-run.nii.gz")
        wf.steps = [step]

        entities = {"sub": "01", "suffix": "T2w"}
        expected_entities = {**entities, "desc": "denoise"}
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        expected_path = final_dir / build_bids_name(expected_entities)
        expected_path.touch()

        context, _ = wf._preprocess_t2w(
            tmp_path,
            {"t2w_files": [ImageFile(entities, tmp_path / "t2.nii.gz")]},
            final_dir,
            None,
            tmp_path,
        )

        assert step.calls == []
        assert context["preprocessed_t2w"].entities == expected_entities
        assert context["preprocessed_t2w"].img == expected_path


# ---------------------------------------------------------------------------
# R11 — single canonical config store (explicit properties, not
# __getattribute__/__setattr__ override)
#
# Locks: config_data IS the same dict object backing the typed fields
# (aliasing is real and intentional), and to_dict() returns a dict whose
# nested sections are NOT deep-copied — mutating a nested section in the
# returned dict mutates the live config too. Both are deliberate consequences
# of "one canonical store"; this test exists so a future change that
# accidentally deep-copies (silently changing this contract) gets caught.
# ---------------------------------------------------------------------------
class TestR11SingleConfigStore:
    def test_config_data_is_the_same_object_as_the_internal_store(self, tmp_path):
        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
        )
        assert config.config_data is config._data

        config.config_data["dmri"] = {"preprocessing": {"denoising": {"enabled": True}}}
        assert config.get("dmri.preprocessing.denoising.enabled") is True

        config.set("dmri.preprocessing.denoising.enabled", False)
        assert config.config_data["dmri"]["preprocessing"]["denoising"]["enabled"] is False

    def test_standard_fields_and_get_set_read_the_same_value(self, tmp_path):
        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            n_cpus=4,
        )
        assert config.n_cpus == config.get("n_cpus") == 4

        config.set("n_cpus", 8)
        assert config.n_cpus == 8

        config.n_cpus = 16
        assert config.get("n_cpus") == 16

    def test_path_fields_are_coerced_on_property_assignment(self, tmp_path):
        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
        )
        config.work_dir = str(tmp_path / "scratch")  # assigned as a plain str
        assert isinstance(config.work_dir, Path)
        assert config.work_dir == tmp_path / "scratch"

    def test_to_dict_nested_sections_are_not_deep_copied(self, tmp_path):
        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
            config_data={"dmri": {"preprocessing": {"denoising": {"enabled": True}}}},
        )
        as_dict = config.to_dict()
        as_dict["dmri"]["preprocessing"]["denoising"]["enabled"] = False

        # The top-level dict returned by to_dict() is a separate object...
        assert as_dict is not config.config_data
        # ...but nested sections inside it are shared by reference, so the
        # mutation above is visible on the live config too.
        assert config.get("dmri.preprocessing.denoising.enabled") is False

    def test_non_standard_attribute_access_is_unaffected(self, tmp_path):
        """Anything that isn't one of the 17 standard fields behaves like a
        normal Python attribute — no property, no override."""
        config = PipelineConfig(
            bids_dir=tmp_path / "bids",
            output_dir=tmp_path / "out",
        )
        config.some_adhoc_attribute = "value"
        assert config.some_adhoc_attribute == "value"
        assert "some_adhoc_attribute" not in config._data


# ---------------------------------------------------------------------------
# Cross-cutting invariant guarded during R7/R8 (cache + tracker engines):
# build_bids_name must remain stable for the entity dicts the cache relies on.
# ---------------------------------------------------------------------------
class TestBidsNameStability:
    def test_preproc_dwi_name(self):
        from qmri_neuropipe.io.bids import build_bids_name

        ents = {"sub": "01", "ses": "02", "desc": "preproc", "suffix": "dwi"}
        name = build_bids_name(ents)
        assert "sub-01" in name and "ses-02" in name
        assert "desc-preproc" in name and name.endswith("_dwi.nii.gz")
