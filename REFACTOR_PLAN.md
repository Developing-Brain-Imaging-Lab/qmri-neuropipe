# qmri-neuropipe — Refactor Plan

> Goal: upgrade code quality, scalability, and maintainability **without changing
> functionality**. Same inputs → same outputs, same files on disk, same tracker rows.
> Every item below is behavior-preserving.

This plan was produced from a read-through of: `core/base.py`, `core/config.py`,
`core/run.py`, `core/step_control.py`, `cli.py`, `io/data_loader.py`,
`workflows/pipelines/dmri.py`, `integrated_modeling_workflow.py`,
`workflows/pipelines/anat.py`, `workflows/pipelines/relaxometry.py`.

---

## How to use this plan

- Rocks are ranked by **payoff ÷ effort**. Do high-payoff/low-effort first.
- **Rule:** one rock per commit. Run the safety net (R0) after each.
- "No behavior change" is the hard constraint. If a change alters any output
  file name, path, or tracker value, it does NOT belong in this plan.

---

## R0 — Safety net (DO THIS FIRST, blocks everything else)

**Effort:** S · **Payoff:** critical (enables every other rock safely)

Add characterization tests that snapshot current behavior so refactors can be
proven safe:

1. Pick one small subject per pipeline: `dmri`, `anat`, `relaxometry`.
2. Run end-to-end into a temp output dir.
3. Snapshot and commit as golden:
   - sorted list of every output file path (relative)
   - tracker rows (status + module names)
   - a checksum (e.g. sha1 of header+affine, not full data) of 2–3 key maps
4. Add a `--dry-run` config-merge test: feed a known config + CLI flags, assert
   the merged `PipelineConfig` matches expected.

After each later rock: re-run, diff against golden. Zero diff = safe.

---

## TIER 1 — High payoff, low/medium effort

### R1 — Kill `copytree`-in-loop (modeling)
**File:** `integrated_modeling_workflow.py` → `_execute_modeling`
**Effort:** S · **Payoff:** high (wall-clock)

Today `shutil.copytree(staging → final)` runs inside the per-DWI × per-step loop
= `N_dwi × N_models` full-tree copies. Move it to **once per DWI, after that
DWI's step loop**. Same files land in the same place; copy count drops to `N_dwi`.

```python
for dwi, mask in zip(dwis, masks):
    context['current_image'] = dwi
    self._maybe_prepare_gnl(context, dwi)
    force_active = False
    for step in self.steps:
        force_active = step_force_active(force_active, step, rerun_from_step)
        self._run_or_skip(step, context, staging_dir, final_dir, mask,
                          force_active, reporter, dwi, progress, task_id)
    if final_dir:                                  # ONE sync, after the steps
        shutil.copytree(staging_dir, final_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("figures"))
```

### R2 — Cheap skip probes (no full walks / no data loads)
**Files:** `core/base.py` (`_should_skip`), `dmri.py` (`_gnl_matches_image_grid`)
**Effort:** S · **Payoff:** medium-high

- `_should_skip`: replace `any(out.rglob('*.*'))` with `next(out.rglob('*.*'), None) is not None` (stops at first hit instead of walking whole tree).
- `_gnl_matches_image_grid`: keep `nib.load(...)` proxies (shape/affine only); never call `get_fdata()`. (Verify no accidental data read sneaks in.)

### R3 — Fix `_force_requested` side-effect (relaxometry)
**File:** `relaxometry.py`
**Effort:** S · **Payoff:** medium-high (correctness trap)

A predicate currently mutates `self._force_from_step_active`. Split into a pure
query and an explicit mutator. Same forcing behavior, no hidden state change.

```python
def is_forced(self) -> bool:                  # PURE
    return bool(self.config.get("force", False)
                or self.config.get("force_run", False)
                or self.config.get("force_rerun", False)
                or self._force_from_step_active)

def advance_force_state(self, step) -> None:  # MUTATES, call once per step
    was = self._force_from_step_active
    self._force_from_step_active = step_force_active(
        self._force_from_step_active, step, self._rerun_from_step)
    if self._force_from_step_active and not was:
        name = step if isinstance(step, str) else step.__class__.__name__
        self.logger.info(f"Forcing relaxometry from {name} (rerun_from_step reached).")
```

Replace each `self._force_requested(step)` call site with the right one
(`advance_force_state(step)` in loops, `is_forced()` for the question).

### R4 — Table-drive the 9 model-step builders
**File:** `integrated_modeling_workflow.py` → the nine `_add_*_step` methods
**Effort:** M · **Payoff:** high (kills ~200 lines, new model = 1 row)

Replace the nine near-identical methods with a registry + one loop. Same steps
added, same kwargs flattening (`parameters`/`options`), same log lines.

```python
@dataclass(frozen=True)
class ModelSpec:
    step_cls: type
    cfg_keys: tuple[str, ...]          # alias list, first present wins
    default_method: str
    flatten: tuple[str, ...] = ("parameters", "options")

MODEL_REGISTRY = (
    ModelSpec(DTIFittingStep,       ("dti", "tensor"),    "dipy"),
    ModelSpec(DKIFittingStep,       ("dki",),             "dipy"),
    ModelSpec(CSDFittingStep,       ("csd",),             "msmt_csd"),
    ModelSpec(NODDIFittingStep,     ("noddi",),           "dmipy"),
    ModelSpec(SANDIFittingStep,     ("sandi",),           "amico"),
    ModelSpec(MicrogliaFittingStep, ("microglia",),       "dmipy"),
    ModelSpec(NEXIFittingStep,      ("nexi",),            "nexi"),
    ModelSpec(MAPMRIFittingStep,    ("mapmri",),          "dipy"),
    ModelSpec(FWDTIFittingStep,     ("fwe_dti", "fwdti"), "dipy"),
)

def _add_model_steps(self, modeling_cfg: dict) -> None:
    for spec in MODEL_REGISTRY:
        cfg = next((modeling_cfg[k] for k in spec.cfg_keys if modeling_cfg.get(k)), None)
        if not cfg or not cfg.get("enabled", False):
            continue
        method = cfg.get("method", spec.default_method)
        kwargs = dict(cfg)
        for nest in spec.flatten:
            nested = kwargs.pop(nest, None)
            if isinstance(nested, dict):
                kwargs.update(nested)
        kwargs.pop("enabled", None); kwargs.pop("method", None)
        self.logger.info(f"Adding {spec.step_cls.__name__} (method={method})")
        self.add_step(spec.step_cls(config=self.config, logger=self.logger,
                                    provenance=self.provenance, method=method,
                                    n_cpus=self.config.n_cpus, **kwargs))
```

> Watch: SANDI/tractography used slightly different kwarg handling — verify each
> step still receives identical kwargs (diff the constructed step objects in a test).

### R5 — Merge the anat T1w/T2w twin methods
**File:** `anat.py` → `_preprocess_t1w` + `_preprocess_t2w`
**Effort:** M · **Payoff:** high (~250 duplicated lines, drift risk)

One parametric `_preprocess_modality(suffix, skip_types, ...)`. T1w and T2w
become one-line callers with different `skip_types`. Move the copied
`step_desc` if/elif chain to one module-level dict.

```python
STEP_DESC = {ResampleStep: "resample", ReorientStep: "reorient",
             DenoisingStep: "denoise", GibbsUnringingStep: "gibbs",
             BiasCorrectionStep: "bias", SharpeningStep: "sharpen",
             NonlinearRegistrationStep: "normalize"}
T1W_SKIP = (CoregistrationStep, BrainMaskingStep, NonlinearRegistrationStep,
            SegmentationStep, FreeSurferStatsStep)
T2W_SKIP = (ReconAllStep, NonlinearRegistrationStep, BrainMaskingStep,
            CoregistrationStep, SegmentationStep, FreeSurferStatsStep)
```

The shared per-step body (skip-check → run → save-intermediate → report) moves
into `_run_one_anat_step(...)`. Preserve the T1w-only FreeSurfer skip branch and
the T1w `current_image` seeding exactly.

### R6 — Table-drive the downstream DESPOT models
**File:** `relaxometry.py` → `_run_model_fitting`
**Effort:** M · **Payoff:** high (~180 lines + 3 copied dependency checks)

DESPOT1/HIFI stay special (they PRODUCE the shared T1/B1). DESPOT2, DESPOT2FM,
mcDESPOT become one loop over a spec table; the triple-copied
"needs T1 / needs B1 / needs SSFP" check collapses into `_resolve_deps`.

```python
@dataclass(frozen=True)
class DespotSpec:
    name: str; cfg_attr: str; fit_fn: Callable
    needs_ssfp: bool = False; needs_t1: bool = False
    needs_b1: bool = False;   default_algo: str = "lsq"

DESPOT_SPECS = (
    DespotSpec("DESPOT2",   "despot2",   fit_despot2,    needs_ssfp=True, needs_t1=True, needs_b1=True),
    DespotSpec("DESPOT2FM", "despot2fm", fit_despot2_fm, needs_ssfp=True, needs_t1=True, needs_b1=True, default_algo="src"),
    DespotSpec("mcDESPOT",  "mcdespot",  fit_mcdespot,   needs_ssfp=True, needs_t1=True, needs_b1=True, default_algo="src"),
)
```

> Watch: mcDESPOT has extra `cuda` kwarg and the legacy `despot2.mcdespot`
> deprecation path — keep both. Verify out_base strings are byte-identical.

---

## TIER 2 — High payoff, higher effort (shared engines)

### R7 — One caching/skip engine
**New:** `core/caching.py` · **Touches:** dmri (4 sites), anat (5), relax (9)
**Effort:** L · **Payoff:** high (biggest source of scattered logic)

Extract the repeated "build expected name → exists? → readable? → not forced? →
wrap as ImageFile → (optionally) poke tracker cached" into one helper:

```python
def reuse_if_exists(entities, out_dir, *, suffix=None, force=False,
                    readable=False) -> Optional[ImageFile]:
    name = build_bids_name(entities, suffix=suffix) if suffix else build_bids_name(entities)
    path = out_dir / (name if name.endswith(".nii.gz") else name + ".nii.gz")
    if force or not path.exists():
        return None
    if readable:
        try: nib.load(str(path))
        except Exception: return None
    return ImageFile(entities=entities, img=path)
```

Migrate call sites one at a time (each its own commit + golden diff). Decide the
fate of the dead `OutputValidationMixin` from `OPTIMIZATION_GUIDE.md`: either
adopt it as the engine everywhere, or delete the guide. **Not both.**

### R8 — One tracker helper + debounced save
**New/extend:** `core/tracking.py` · **Touches:** base.py, modeling, anat, relax
**Effort:** M · **Payoff:** medium-high (kills write storm + copied pokes)

Single `update_step_status(config, context, step, status)` replaces the copied
`_update_tracker_*` blocks. Replace `tracker.save()` after every step with a
debounced/batched save (flush at subject end + on failure). Same final rows.

### R9 — Split the god-methods
**Files:** `anat.py` (`_run_normalization`, `_run_coregistration`), `relaxometry.py` (`_run_model_fitting`, `run`)
**Effort:** M · **Payoff:** medium-high (readability, testability)

- `_run_normalization` → `_normalize_primary`, `_apply_warp_to_secondary`,
  `_normalize_mask`, `_publish_norm`. Extract the doubled FSL/ANTs apply branch
  into ONE `apply_spatial_transform(...)` (see snippet in chat / below).
- `_run_coregistration` → target-selector table + one `_do_coreg(moving, fixed, ...)`
  helper; the 4 big branches (supersynth-multivariate / supersynth / t2w / t1w)
  shrink to config + one call each.

```python
def apply_spatial_transform(self, in_img, template, transform, *,
                            transform_type, interp, out_path):
    transform = Path(transform)
    if transform_type == "fsl" or transform.suffix == ".mat":
        fsl.applywarp(in_file=in_img, ref_file=Path(template), out_file=out_path,
                      premat=transform, interp=interp, force=True)
    else:
        warp   = transform.parent / f"{transform.name}1Warp.nii.gz"
        affine = transform.parent / f"{transform.name}0GenericAffine.mat"
        if not (warp.exists() and affine.exists()):
            raise FileNotFoundError(f"Missing {warp.name} or {affine.name}")
        ants.apply_transforms(fixed_file=Path(template), moving_file=in_img,
                              out_file=out_path, transforms=[warp, affine],
                              interpolator="linear" if interp != "nn" else "nearestNeighbor")
    return out_path
```

---

## TIER 3 — Structural, highest blast radius (do last)

### R10 — One parallel runner
**Files:** `cli.py` (`_run_parallel_worker` + `UIState`), `base.py` (`_run_parallel`)
**Effort:** L · **Payoff:** medium (scale + correctness)

Two parallel implementations exist; `base.py`'s is effectively dead. Move the
UI + worker into `core/parallel.py`, have both `cli.py` and `BasePipeline` call
the one runner. Keep the FD-redirect logic but isolate + unit-test it (current
`os.dup2` juggling on FDs 1/2 is fragile across pool reuse). Add skip-before-
dispatch so parallel `n_skipped` matches sequential.

### R11 — One config store
**File:** `core/config.py`
**Effort:** L · **Payoff:** medium (clarity, fewer "where does this live" bugs)

Today values live in BOTH typed dataclass fields AND `config_data` dict;
`get()` checks both. Pick one source of truth (recommend: keep the dataclass as
the public surface, derive it from a single validated dict). Keep `get`/`set`
dot-notation API identical so callers don't change.

### R12 — Typed context (migration-safe)
**New:** `core/context.py` · **Touches:** all workflows/steps (gradually)
**Effort:** L · **Payoff:** medium (kills silent typo bugs)

Introduce `PipelineContext` dataclass with the known keys + `extra` escape hatch,
implementing `get/__getitem__/__setitem__/setdefault` so existing
`context['current_image']` code keeps working. Swap in gradually. Also unify the
return contract: `RelaxometryWorkflow.run` returns a results-dict with `context`
nested, while anat/dmri return `context` directly — pick one shape.

### R13 — Pin config aliases once (relaxometry especially)
**File:** `relaxometry.py` config dataclasses
**Effort:** S-M · **Payoff:** medium

Normalize alias spellings ONCE at config-build time:
`space/space_name/space_entity`, `save_transform/save_transforms`,
`threads/nthreads`, `skull_strip/skull_strip_registration`. Rest of code reads
canonical names only. Removes the repeated `cfg.get("a", cfg.get("b", cfg.get("c", default)))` chains.

---

## TIER 4 — Hygiene (anytime, low risk)

- **H1** Replace broad `except Exception: pass` around rich/progress imports with
  narrow `except ImportError` (or a tiny `_safe_progress_update` helper). Don't
  swallow real errors.
- **H2** Hoist function-local `import` statements to module top where they are NOT
  guarding optional deps. Keep lazy imports only for genuinely optional packages
  (pyAFQ, tensorflow, amico, etc.).
- **H3** Fix `BasePipeline.run` docstring: it returns a stats dict, not `None`.
  Make sequential + parallel paths return the SAME shape (include `failures`).
- **H4** Remove the `config.debug if 'config' in locals() else debug` smell in
  `cli.py`'s outer `except` — resolve config earlier so it always exists.
- **H5** Delete commented-out dead code (`_get_subject_sessions` block in base.py).

---

## Suggested order (one commit each)

```
R0  (safety net)            ← gate
R1  copytree-in-loop        ← fast win
R2  skip probes
R3  force_requested fix
H3  run() return shape
R4  modeling registry
R5  anat twin merge
R6  despot registry
R8  tracker helper + debounce
R7  caching engine          ← migrate sites incrementally
R9  split god-methods
H1 H2 H4 H5                 ← hygiene, fold in opportunistically
R13 config alias pin
R10 parallel runner         ← structural
R11 config store
R12 typed context
```

## Definition of done (per rock)
- Golden snapshots (R0) unchanged.
- No output file path/name/count change.
- Tracker rows identical.
- Public CLI flags and config keys unchanged.

---
*Generated as a behavior-preserving refactor guide. No functionality changes are
intended by any item; if implementing one forces an output change, stop and
re-scope it.*
