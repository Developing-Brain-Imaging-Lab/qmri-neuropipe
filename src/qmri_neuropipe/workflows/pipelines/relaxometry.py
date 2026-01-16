import logging
from pathlib import Path
from typing import Optional, List, Dict
import shutil
import json

from ...core import BaseWorkflow, PipelineConfig
from ...core.types import ImageFile
from ...io.bids import build_bids_name, get_entities_from_path



# Import Steps
from ...lib.relax.motion import SPGRMotionCorrectionStep
from ...lib.relax.b1 import B1MappingStep
from ...lib.common.reorient import ReorientStep
from ...lib.common.denoise import DenoisingStep
from ...lib.common.mask import BrainMaskingStep
from ...lib.common.gibbs import GibbsUnringingStep
from ...lib.common.stats import ROIStatsStep
from ...interfaces.relaxometry import fit_despot1, fit_despot1_hifi, fit_despot2, fit_despot2_fm
from ...utils.relax_params import generate_acq_params

class RelaxometryWorkflow(BaseWorkflow):
    """
    Pipeline for processing Relaxometry (DESPOT1/DESPOT2) data.
    """
    
    def _initialize_steps(self):
        # Configuration
        relax_cfg = self.config.get("relaxometry", {})
        preproc_cfg = relax_cfg.get("preprocessing", {})
        

        # 0. Reorientation
        reorient_cfg = preproc_cfg.get("reorient", {})
        if reorient_cfg.get("enabled", False):
             self.add_step(ReorientStep(self.config, self.logger, self.provenance))

        # 1. Denoising
        den_cfg = preproc_cfg.get("denoising", {})
        if den_cfg.get("enabled", True):
             self.add_step(DenoisingStep(self.config, self.logger, self.provenance, 
                                         method=den_cfg.get("method", "mrtrix")))

        # 2. Gibbs Ringing
        gibbs_cfg = preproc_cfg.get("degibbs", {})
        if gibbs_cfg.get("enabled", True):
             self.add_step(GibbsUnringingStep(self.config, self.logger, self.provenance,
                                              method=gibbs_cfg.get("method", "mrtrix")))


        # 3. Motion Correction
        moco_cfg = preproc_cfg.get("motion_correction", {})
        if moco_cfg.get("enabled", True):
             # Extract extra options (exclude 'enabled' and 'method')
             moco_opts = {k:v for k,v in moco_cfg.items() if k not in ['enabled', 'method']}
             
             self.add_step(SPGRMotionCorrectionStep(self.config, self.logger, self.provenance,
                                                    method=moco_cfg.get("method", "ants"),
                                                    options=moco_opts))
        
        # 4. B1 Mapping
        b1_cfg = preproc_cfg.get("b1", {})
        self.add_step(B1MappingStep(self.config, self.logger, self.provenance,
                                    method=b1_cfg.get("method", "afi"),
                                    smoothing_fwhm=b1_cfg.get("smoothing_fwhm", 2.0)))
        
    def run(self, output_dir: Path, context: dict, final_output_dir: Optional[Path] = None, reporter=None) -> dict:
        self.logger.info("Starting RelaxometryWorkflow")
        
        # 1. Parse Inputs from Context
        # Context should be populated by BIDS search before calling run?
        # Or we extract from 'relax_files' list?
        relax_files: List[ImageFile] = context.get('relax_files', [])
        
        # Initialize config
        relax_cfg = self.config.get("relaxometry", {})
        preproc_cfg = relax_cfg.get("preprocessing", {})
        
        spgr_files = []
        ssfp_files = []
        irspgr_files = []
        b1_files = [] # Map or AFI source
        b1_ref_file = None # AFI Ref
        
        for f in relax_files:
            # Check entities
            acq = f.entities.get('acq', '').lower()
            desc = f.entities.get('desc', '').lower()
            suffix = f.entities.get('suffix', '').lower()
            
            # SPGR detection
            if 'spgr' in acq or 'spgr' in desc:
                if 'ir' in acq or 'ir' in desc:
                    irspgr_files.append(f)
                else:
                    spgr_files.append(f)
            # SSFP detection
            elif 'ssfp' in acq or 'ssfp' in desc:
                ssfp_files.append(f)
            # B1 Map/AFI detection
            elif 'afi' in acq or 'afi' in desc or 'b1' in suffix:
                b1_files.append(f)
                
        # Sort files?
        spgr_files.sort(key=lambda x: str(x.img))
        ssfp_files.sort(key=lambda x: str(x.img))
        
        self.logger.info(f"Found inputs: {len(spgr_files)} SPGR, {len(ssfp_files)} SSFP, {len(irspgr_files)} IR-SPGR, {len(b1_files)} B1.")
        
        if not spgr_files:
             raise ValueError("No SPGR images found. Cannot proceed with Relaxometry.")
             
        # Reference selection deferred until after preprocessing

        
        # BIDS Standard Output Paths
        anat_out_dir = output_dir / "anat"
        fmap_out_dir = output_dir / "fmap"
        anat_out_dir.mkdir(parents=True, exist_ok=True)
        fmap_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Directory Structure
        work_dir = self.config.work_dir
        if work_dir:
            # Construct sub/ses path for working directory
            subj = context.get('subject')
            sess = context.get('session')
            wd_subj = work_dir / f"sub-{subj}"
            if sess:
                wd_subj = wd_subj / f"ses-{sess}"
            intermediate_dir = wd_subj / "anat" / "intermediate"
        else:
             intermediate_dir = anat_out_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        


        # 2. Preprocessing Loop
        
        # A. Denoise/Gibbs/Reorient (Per Image)
        def _pre_proc_list(img_list, modality_name):
             out_list = []
             for img in img_list:
                 curr = img
                 for step in self.steps:
                     # Check step types strictly
                     if isinstance(step, (DenoisingStep, GibbsUnringingStep, ReorientStep)):
                         # ReorientStep usually takes 1 arg.
                         # Denoising/Gibbs take 1 arg + kwargs.
                         curr = step.run(curr, output_dir=intermediate_dir)
                 out_list.append(curr)
             return out_list

        spgr_pre = _pre_proc_list(spgr_files, "SPGR")
        ssfp_pre = _pre_proc_list(ssfp_files, "SSFP")
        ir_pre = _pre_proc_list(irspgr_files, "IRSPGR")

        # Select Reference from Preprocessed SPGR (Max FA)
        # Ensures reference is reoriented/denoised like the rest
        ref_img = spgr_pre[0] # Default
        max_fa = -1
        from ...utils.relax_params import _extract_bids_param
        
        for img in spgr_pre:
             # Extract FA from original entities or JSON sidecar (still valid)
             fa = _extract_bids_param(img, "FlipAngle", 0.0)
             if isinstance(fa, list): fa = max(fa) if fa else 0.0
             if float(fa) > max_fa:
                 max_fa = float(fa)
                 ref_img = img
                 
        context['relax_reference'] = ref_img
        self.logger.info(f"Selected Relaxometry Reference (Preprocessed): {ref_img.img.name}")
        
        # B. Motion Correction (Group)
        moco_step = next((s for s in self.steps if isinstance(s, SPGRMotionCorrectionStep)), None)
        
        spgr_moco = spgr_pre
        ssfp_moco = ssfp_pre
        # IR?
        
        if moco_step:
             # Run on ALL SPGR (and SSFP?)
             # Usually SSFP registered to SPGR reference.
             # Pass ALL concatenated? Or run separately with same ref?
             # Run SPGR first to get Ref stable.
             # NOTE: Output dir for moco is anat_out_dir (Final Preproc)
             spgr_moco = moco_step.run(spgr_pre, output_dir=anat_out_dir, reference_image=ref_img, modality="SPGR")
             
             # Register SSFP to same SPGR Ref
             if ssfp_pre:
                 ssfp_moco = moco_step.run(ssfp_pre, output_dir=anat_out_dir, reference_image=ref_img, modality="SSFP")
                 
             if ir_pre:
                 # Ensure IR is also processed
                 ir_moco = moco_step.run(ir_pre, output_dir=intermediate_dir, reference_image=ref_img, modality="IR-SPGR")
                 
        context['processed_spgr'] = spgr_moco
        context['processed_ssfp'] = ssfp_moco
        

        
        # C. Brain Masking (Early)
        # Run on SPGR Reference Image (MoCo Target)
        # Check preprocessing.brain_masking first, then relaxometry.masking
        mask_cfg = preproc_cfg.get("brain_masking", {})
        if not mask_cfg:
             mask_cfg = relax_cfg.get("masking", {})
             
        mask_method = mask_cfg.get("method", "fsl")
        
        mask_step = next((s for s in self.steps if isinstance(s, BrainMaskingStep)), None)
        if not mask_step:
             mask_step = BrainMaskingStep(self.config, self.logger, self.provenance, method=mask_method)
             
        # Prepare Target Mask Name
        subj = context.get('subject')
        sess = context.get('session')
        base_prefix = f"sub-{subj}"
        if sess: base_prefix += f"_ses-{sess}"
        
        target_mask_name = anat_out_dir / f"{base_prefix}_desc-brain-mask.nii.gz"
        
        if target_mask_name.exists():
             self.logger.info(f"Skipping Brain Masking (Exists): {target_mask_name}")
             mask_file = target_mask_name
             context['brain_mask'] = mask_file
        else:
             self.logger.info(f"Running Brain Masking on Reference: {ref_img.img.name}")
             # Save intermediate mask files to intermediate_dir (e.g. anat/intermediate/brainmasking or just anat/intermediate)
             # return_mask=True gives us the mask object.
             # We let it output to intermediate_dir. 
             # BrainMaskingStep usually creates a subfolder 'brainmasking' inside output_dir?
             # Let's check BrainMaskingStep impl... defaulting to intermediate_dir
             masked_ref, mask_obj = mask_step.run(ref_img, output_dir=intermediate_dir, return_mask=True)
             
             if mask_obj.img != target_mask_name:
                  # Move ONLY the final mask to anat_out_dir
                  import shutil
                  shutil.move(mask_obj.img, target_mask_name)
                  mask_file = target_mask_name
             else:
                  mask_file = mask_obj.img
             
             context['brain_mask'] = mask_file
             self.logger.info(f"Generated Brain Mask: {mask_file}")
             
        # 3. Parameter Generation (to anat dir)
        # Let's save ACQ params to anat dir as it is a result
        ir_final = ir_moco if 'ir_moco' in locals() else ir_pre
        
        # BIDS Name: sub-XX[_ses-YY]_desc-AcqParams.json
        # Manual construction to ensure exact format requested
        params_name = f"{base_prefix}_desc-AcqParams.json"

        params_json = anat_out_dir / params_name
        generate_acq_params(spgr_moco, ssfp_moco, ir_final, output_path=params_json)
        
        # 4. B1 Mapping (to fmap dir)
        b1_step = next((s for s in self.steps if isinstance(s, B1MappingStep)), None)
        b1_map = None
        
        if b1_files and b1_step:
             # Identiy if AFI pair exists
             curr_b1 = b1_files[0] 
             b1_ref = None # Extract from b1_files if distinct?
             
             # Output to fmap/intermediate to keep fmap clean
             fmap_inter_dir = fmap_out_dir / "intermediate"
             fmap_inter_dir.mkdir(parents=True, exist_ok=True)
             
             b1_map_inter = b1_step.run(curr_b1, reference_image=ref_img, output_dir=fmap_inter_dir, b1_ref_image=b1_ref)
             
             # Move final Map to fmap_out_dir
             # Warning: b1_map_inter.img might be 'TB1map.nii.gz' or similar. 
             # We want standard BIDS name? It generates sub-XX_ses-YY_TB1map.nii.gz
             final_b1_path = fmap_out_dir / b1_map_inter.img.name
             
             import shutil
             if b1_map_inter.img.exists():
                 if b1_map_inter.img != final_b1_path:
                      shutil.move(b1_map_inter.img, final_b1_path)
                 b1_map = ImageFile(img=final_b1_path, entities=b1_map_inter.entities)
             else:
                 # It might already be there if we resumed?
                 # But we ran step with fmap_inter_dir. If resume logic inside step checks existing in OUTPUT dir...
                 # Resume check inside B1MappingStep.run uses output_dir. 
                 # So if we changed output_dir to fmap_inter, it won't see the one in fmap_out_dir.
                 # Fix: Check fmap_out_dir/TB1Map first?
                 pass
                 
             # Wait, if we use fmap_inter for run(), resume check looks in fmap_inter. 
             # If we moved it last time, it won't find it and re-run.
             # We should check existence in FINAL location first.
             
             # BIDS Name assumption
             # B1MappingStep generates using BIDS Utils.
             # Let's reconstruct expected name to check existence
             expected_b1_name = fmap_out_dir / f"{base_prefix}_TB1map.nii.gz" # Approx?
             # Actually B1MappingStep uses input entities to generate name.
             
             # If we want to support Resume properly with Move:
             # We should rely on `b1_map` result. 
             # If we pass fmap_inter_dir, it will generate there. 
             # If we want to avoid re-run, check final destination first.
             
             # Optimization:
             # 1. Check if final B1 map exists in fmap_out_dir
             # 2. If yes, use it.
             # 3. If no, run step (into intermediate), then move.
             
             # We need to guess the final name though? Or search?
             # B1MappingStep output name is predictable if we know entities.
             # Let's search fmap_out_dir for *TB1map*?
             existing_b1 = list(fmap_out_dir.glob("*TB1map.nii.gz"))
             if existing_b1 and not self.config.get("force_rerun", False):
                  self.logger.info(f"Found existing B1 Map: {existing_b1[0].name}")
                  b1_map = ImageFile(img=existing_b1[0], entities={}) # Entities?
             else:
                  b1_map_inter = b1_step.run(curr_b1, reference_image=ref_img, output_dir=fmap_inter_dir, b1_ref_image=b1_ref)
                  final_path = fmap_out_dir / b1_map_inter.img.name
                  shutil.move(b1_map_inter.img, final_path)
                  b1_map = ImageFile(img=final_path, entities=b1_map_inter.entities)
             


        # 5. Fitting Strategy
        # relax_cfg already loaded
        model_cfg = relax_cfg.get("modeling", {})
        # Output to anat dir
        fit_out_dir = anat_out_dir 
        
        # Helper to check if model enabled (default True if not config present but files exist?)
        # User requested specific config pattern.
        # If modeling section exists, we respect it strictly?
        # Let's support: modeling: despot1: enabled: true
        
        run_despot1 = model_cfg.get("despot1", {}).get("enabled", False) # Default False
        use_hifi = model_cfg.get("despot1", {}).get("use_hifi", True)
        
        run_despot2 = model_cfg.get("despot2", {}).get("enabled", False) # Default False
        use_mcdespot = model_cfg.get("despot2", {}).get("mcdespot", True)
        
        # Check inputs availability
        has_spgr = bool(spgr_moco)
        has_ir = bool(ir_pre) # Start with pre as moco might be missing
        if 'ir_moco' in locals() and ir_moco: has_ir = True # Upgrade if moco exists
        has_ssfp = bool(ssfp_moco)
        
        # --- DESPOT1 / HIFI ---
        if run_despot1 and has_spgr:
            # Determine HIFI or Standard
            # If IR exists, prefer HIFI? Or Configurable?
            # modeling.despot1.hifi : bool ?
            d1_cfg = model_cfg.get("despot1", {})
            use_hifi = d1_cfg.get("use_hifi", True) # Default prefer High-Fi if available
            
            # Merge SPGR (Required for both)
            from ...interfaces.fsl import merge
            spgr_4d = intermediate_dir / "spgr_4d.nii.gz"
            merge(spgr_moco, spgr_4d, dimension='t')
            
            # Construct Base Name
            # base_prefix already defined
            
            despot1_base = f"{base_prefix}_despot1_hifi" if (has_ir and use_hifi) else f"{base_prefix}_despot1"
            expected_t1 = fit_out_dir / f"{despot1_base}_T1map.nii.gz"
            expected_m0 = fit_out_dir / f"{despot1_base}_M0map.nii.gz"
            
            if expected_t1.exists() and expected_m0.exists():
                 self.logger.info(f"Skipping DESPOT1 Fitting (Exists): {expected_t1}")
                 res = {
                     't1': expected_t1,
                     'm0': expected_m0
                 }
                 # Add potential other outputs if they exist
                 if (fit_out_dir / f"{despot1_base}_B1map.nii.gz").exists():
                     res['b1'] = fit_out_dir / f"{despot1_base}_B1map.nii.gz"
            else:
                if has_ir and use_hifi:
                     self.logger.info("Running DESPOT1-HIFI Fitting...")
                     # Handle IR list (usually 1 file)
                     ir_in = ir_final[0] if isinstance(ir_final, list) else ir_final
                     
                     res = fit_despot1_hifi(
                         spgr_file=spgr_4d, 
                         irspgr_file=ir_in,
                         params_file=params_json,
                         out_dir=fit_out_dir,
                         out_base=despot1_base,
                         mask_file=mask_file
                     )
                else:
                     self.logger.info("Running DESPOT1 Standard Fitting...")
                     res = fit_despot1(
                         spgr_file=spgr_4d,
                         params_file=params_json,
                         out_dir=fit_out_dir,
                         b1_file=b1_map,
                         out_base=despot1_base,
                         mask_file=mask_file
                     )
            context.update(res)
            # Update B1 if HIFI generated it
            if "b1" in res and not b1_map: 
                b1_map = ImageFile(img=Path(res["b1"]), entities={})

        # --- DESPOT2 / mcDESPOT ---
        if run_despot2 and has_ssfp and has_spgr:
             # Requires T1 map (from DESPOT1)
             t1_map = context.get("t1")
             if not t1_map:
                  self.logger.warning("DESPOT2 requested but no T1 map found (DESPOT1 failed or disabled?). Skipping DESPOT2.")
             else:
                  self.logger.info("Running DESPOT2 Fitting...")
                  
                  # Merge SSFP
                  from ...interfaces.fsl import merge
                  ssfp_4d = intermediate_dir / "ssfp_4d.nii.gz"
                  merge(ssfp_moco, ssfp_4d, dimension='t')
                  
                  # Check mcDESPOT vs DESPOT2
                  # modeling.despot2.mcdespot : bool
                  d2_cfg = model_cfg.get("despot2", {})
                  use_mcdespot = d2_cfg.get("mcdespot", False)
                  
                  # Construct Base Name
                  # base_prefix already defined

                  despot2_base = f"{base_prefix}_despot2_fm" if use_mcdespot else f"{base_prefix}_despot2"
                  expected_t2 = fit_out_dir / f"{despot2_base}_T2map.nii.gz"
                  expected_mwf = fit_out_dir / f"{despot2_base}_MWFmap.nii.gz" # Only for mcDESPOT really?
                  
                  # Check key output (T2)
                  if expected_t2.exists():
                       self.logger.info(f"Skipping DESPOT2 Fitting (Exists): {expected_t2}")
                       res2 = {
                           't2': expected_t2
                       }
                       if use_mcdespot and expected_mwf.exists():
                           res2['mwf'] = expected_mwf
                       # Check for others (tau, off_res/fm)
                       if (fit_out_dir / f"{despot2_base}_Taumap.nii.gz").exists():
                           res2['tau'] = fit_out_dir / f"{despot2_base}_Taumap.nii.gz"
                  else:
                      if use_mcdespot:
                           self.logger.info("Running mcDESPOT (2-Component) Fitting...")
                           res2 = fit_despot2_fm(
                               ssfp_file=ssfp_4d,
                               t1_file=t1_map,
                               b1_file=b1_map or t1_map,
                               params_file=params_json,
                               out_dir=fit_out_dir,
                               out_base=despot2_base,
                               mask_file=mask_file
                           )
                      else:
                           self.logger.info("Running DESPOT2 (1-Component) Fitting...")
                           res2 = fit_despot2(
                               ssfp_file=ssfp_4d,
                               t1_file=t1_map,
                               b1_file=b1_map or t1_map, # Fallback B1=T1 is weird but if B1 missing?
                               params_file=params_json,
                               out_dir=fit_out_dir,
                               out_base=despot2_base,
                               mask_file=mask_file
                           )
                  context.update(res2)





        # 6. Post-Processing (Coreg/Norm/Stats)
        
        # A. Get Anatomical Reference (T1w)
        # Assuming context has 'anat_t1w' or we passed it in inputs?
        # If running standalone, we might rely on 't1w_file' in context.
        t1w_anat = context.get('t1w_file') or context.get('preprocessed_t1w')
        
        # B. Get Maps from context
        # Gather all quantitative maps generated
        maps = []
        for k, v in context.items():
             if k in ["t1", "t2", "m0", "mwf", "tau", "b1", "f0", "t1_fast", "t1_slow", "t2_fast", "t2_slow"] and isinstance(v, (Path, str)):
                  # Convert Path to ImageFile if needed
                  # Context usually stores Path from wrapper return
                  # Check if it's in fit_out_dir to be sure it's ours
                  if str(fit_out_dir) in str(v):
                      maps.append((k, ImageFile(img=Path(v), entities={}))) # Entities?
             elif k in ["t1", "t2", "m0", "mwf", "tau", "b1", "f0", "t1_fast", "t1_slow", "t2_fast", "t2_slow"] and isinstance(v, ImageFile):
                  maps.append((k, v))

        if t1w_anat and maps:
             self.logger.info("Running Post-Processing (Coregistration -> Stats)...")
             
             # 1. Calculate Transform: SPGR Ref -> T1w Anat
             # Reuse CoregistrationStep logic or call interface directly.
             # We want to Save the transform.
             # Use SPGR Ref (motion corrected)
             spgr_ref = context.get('relax_reference') # Usually valid
             
             from ...interfaces.fsl import flirt, applywarp, fast
             
             # Coreg Output Dir
             post_out = output_dir / "derivatives"
             post_out.mkdir(exist_ok=True)
             
             # Calculate Transform
             xfm_mat = post_out / "spgr_to_t1w.mat"
             spgr_in_t1w = post_out / "spgr_in_t1w.nii.gz"
             
             if xfm_mat.exists():
                  self.logger.info("Skipping Coregistration Calculation (Exists)")
             else:
                  self.logger.info("Calculating Coregistration (SPGR -> T1w)")
                  # flirt -in <spgr> -ref <t1w> -omat <mat>
                  flirt(in_file=spgr_ref.img, ref_file=t1w_anat.img, out_file=spgr_in_t1w, omat=xfm_mat, dof=6)
             
             # 2. Apply to All Maps (and setup stats target)
             registered_maps = []
             
             # Get Template Warp (T1w -> MNI) if available
             template_warp = context.get('template_warp')
             template_ref = context.get('template_ref')
             
             # Segmentation Source?
             # 1. Context 'segmentation' (e.g. from previous Anat step)
             # 2. Or Generate NEW one on T1w
             seg_file = context.get('segmentation')
             if not seg_file or not seg_file.img.exists():
                  # Attempt to generate simple segmentation using FAST on T1w
                  self.logger.info("No segmentation found in context. Running FAST on T1w...")
                  fast_out_base = post_out / "t1w_fast"
                  # fast wrapper: checks existence internally
                  fast(in_files=t1w_anat.img, out_base=fast_out_base, img_type=1) 
                  # FAST produces <base>_seg.nii.gz
                  seg_path = post_out / "t1w_fast_seg.nii.gz"
                  if seg_path.exists():
                      seg_file = ImageFile(img=seg_path, entities=t1w_anat.entities)
             
             stats_step = ROIStatsStep(self.config, self.logger, self.provenance)
             
             for map_name, map_img in maps:
                 # A. T1w Space
                 out_name = f"{map_name}_in_t1w.nii.gz"
                 out_path = post_out / out_name
                 
                 final_map_for_stats = None
                 
                 if out_path.exists():
                      self.logger.info(f"Skipping Resampling to T1w (Exists): {out_name}")
                      final_map_for_stats = ImageFile(img=out_path, entities=map_img.entities)
                 else:
                      # Apply XFM
                      applywarp(in_file=map_img.img, ref_file=t1w_anat.img, out_file=out_path, premat=xfm_mat, interp="trilinear")
                      final_map_for_stats = ImageFile(img=out_path, entities=map_img.entities)
                 
                 # B. MNI Space (if warp available)
                 if template_warp and template_ref:
                      norm_out = post_out / f"{map_name}_in_mni.nii.gz"
                      
                      if norm_out.exists():
                           self.logger.info(f"Skipping Normalization to MNI (Exists): {norm_out.name}")
                           # If seg is in MNI, use this? usually seg is in T1w or specific method. Both valid.
                           # Prioritize T1w stats if segmentation is in T1w.
                      else:
                           # applywarp using warp
                           # FSL: --premat is applied to input before warp. 
                           applywarp(in_file=map_img.img, ref_file=template_ref, out_file=norm_out, warp=template_warp, premat=xfm_mat)
                           
                 # C. ROI Stats
                 # Run on T1w space map using T1w space segmentation
                 if seg_file and final_map_for_stats:
                      stats_step.run(final_map_for_stats, seg_file, output_dir=post_out)


        # 7. Save Intermediates (if requested and using separate work_dir)
        save_inter = self.config.get("save_intermediates", False)
        if save_inter:
             final_inter_dir = anat_out_dir / "intermediate"
             # If using work_dir, intermediate_dir != final_inter_dir (paths differ)
             # But if they point to same location (user set work_dir=output_dir/anat), check existence.
             if intermediate_dir != final_inter_dir and intermediate_dir.exists():

                 self.logger.info(f"Saving intermediate files to {final_inter_dir}")
                 try:
                     # Remove destination if exists to ensure clean copy? Or merge?
                     # copytree(dirs_exist_ok=True) merges/overwrites.
                     if not final_inter_dir.exists():
                         final_inter_dir.mkdir(parents=True)
                     
                     # Using copytree with dirs_exist_ok=True
                     shutil.copytree(intermediate_dir, final_inter_dir, dirs_exist_ok=True)
                 except Exception as e:
                     self.logger.warning(f"Failed to copy intermediates: {e}")

        return context

from ...core import BasePipeline
from ...io.relax.bids import bids_find_relax
from ...lib.reporting.report import ReportGenerator

class RelaxometryPipeline(BasePipeline):
    """
    Relaxometry Processing Pipeline (DESPOT1/2).
    """
    
    @property
    def name(self):
        return 'relaxometry-pipeline'

    @property
    def version(self):
        return '1.0.0'
        
    def _initialize_pipeline(self):
        # Initialize the Workflow
        self.workflow = RelaxometryWorkflow(self.config, self.logger, self.provenance)
        
    def _should_skip(self, subject: str, session: Optional[str]) -> bool:
        # Custom skip logic?
        # For now return False and let workflow steps decide or if output exists?
        # Similar to DMRIPipeline, safer to return False.
        return False
        
    def process_subject(self, subject: str, session: Optional[str]):
        ses_str = f"ses-{session}" if session else ""
        subj_dir = Path(self.config.bids_dir) / f"sub-{subject}"
        if session: subj_dir = subj_dir / ses_str
        
        if not subj_dir.exists():
            self.logger.warning(f"Subject directory not found: {subj_dir}")
            return

        # Prepare Output
        # Prepare Output
        output_dir = self._get_output_dir(subject, session)

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare Context
        # Find files
        relax_files = bids_find_relax(subj_dir)
        
        if not relax_files:
             self.logger.warning(f"No relaxometry files found for sub-{subject} {ses_str}. Skipping.")
             return
             
        self.logger.info(f"Found {len(relax_files)} relaxometry files for sub-{subject}")
        
        # T1w for Coregistration
        # Find T1w in anat
        # Crude approach: search list for T1w if not tagged as SPGR/Relax?
        # Usually relaxometry needs a reference structural if specifically requested.
        # Or maybe the pipeline uses one of the SPGRs as reference? 
        # Standard: Coreg to 'anat' workflow output or raw T1w?
        # Let's try to find a standard T1w in anat folder
        t1w_files = list((subj_dir / 'anat').glob("*_T1w.nii.gz"))
        t1w_file = ImageFile(img=t1w_files[0], entities={}) if t1w_files else None
        
        context = {
            'relax_files': relax_files,
            't1w_file': t1w_file,
            'subject': subject,
            'session': session
        }
        
        # Reporter
        report_title = f"Relaxometry Pipeline Report: sub-{subject} {ses_str}"
        reporter = ReportGenerator(output_dir.parent, title=report_title)
        
        # Run Workflow
        try:
            self.workflow.run(output_dir, context, final_output_dir=output_dir, reporter=reporter)
        except Exception as e:
            self.logger.error(f"Error processing sub-{subject}: {e}")
            if self.config.stop_on_error:
                raise e
