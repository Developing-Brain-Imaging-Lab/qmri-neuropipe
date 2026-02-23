import numpy as np

def _safe_exp(numerator, denominator):
    """Safely compute exp(-numerator/denominator) handling zeros."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.exp(-numerator / denominator)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def synthetic_se(t1_map, t2_map, m0_map, tr, te):
    """
    Generate synthetic Spin Echo (SE) signal.
    """
    e1 = _safe_exp(tr, t1_map)
    e2 = _safe_exp(te, t2_map)
    return m0_map * (1 - e1) * e2

def synthetic_spgr(t1_map, m0_map, tr, fa_deg):
    """
    Generate synthetic Spoiled Gradient Recalled Echo (SPGR/FLASH) signal.
    """
    fa_rad = np.radians(fa_deg)
    e1 = _safe_exp(tr, t1_map)
    
    num = (1 - e1) * np.sin(fa_rad)
    den = 1 - e1 * np.cos(fa_rad)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        signal = m0_map * (num / den)
    return np.nan_to_num(signal, nan=0.0)

def synthetic_ir(t1_map, t2_map, m0_map, tr, te, ti):
    """
    Generate synthetic Inversion Recovery (IR) signal.
    """
    e_ti = _safe_exp(ti, t1_map)
    e_tr = _safe_exp(tr, t1_map)
    e_te = _safe_exp(te, t2_map)
    
    return m0_map * np.abs(1 - 2*e_ti + e_tr) * e_te

def synthetic_flair(t1_map, t2_map, m0_map, tr, te, ti):
    """
    Generate synthetic Fluid-Attenuated IR (FLAIR).
    Identical equation to IR, but parameterized differently in UI usually.
    """
    return synthetic_ir(t1_map, t2_map, m0_map, tr, te, ti)

def synthetic_ssfp(t1_map, t2_map, m0_map, tr, fa_deg):
    """
    Generate synthetic Steady-State Free Precession (bSSFP / TrueFISP) signal.
    """
    fa_rad = np.radians(fa_deg)
    e1 = _safe_exp(tr, t1_map)
    e2 = _safe_exp(tr, t2_map)
    
    num = (1 - e1) * np.sin(fa_rad)
    den = 1 - (e1 - e2)*np.cos(fa_rad) - e1*e2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        signal = m0_map * (num / den)
        
    return np.nan_to_num(signal, nan=0.0)

def synthetic_mprage(t1_map, m0_map, tr, ti, fa_deg):
    """
    Generate synthetic MPRAGE (T1-weighted) signal.
    Approximate steady-state model for the readout train.
    Simplified as an IR sequence for generic prototyping.
    """
    e_ti = _safe_exp(ti, t1_map)
    e_tr = _safe_exp(tr, t1_map)
    return m0_map * np.abs(1 - 2*e_ti + e_tr) * np.sin(np.radians(fa_deg))

def synthetic_dwi(d_map, s0_map, bval):
    """
    Generate simple ADC-based synthetic diffusion signal.
    """
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        e_b = np.exp(-bval * d_map)
    signal = s0_map * e_b
    return np.nan_to_num(signal, nan=0.0)
