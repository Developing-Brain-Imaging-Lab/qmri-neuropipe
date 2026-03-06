import re, gzip
import numpy as np
import nibabel as nib
import pydicom
from pydicom.tag import Tag

DICOM_1    = "/scratch/mrphys/dw500out/00009.dw500_pe0_out/00009.dw500_pe0_out.000001.dcm"
# DICOM_1    = "/scratch/mrphys/dw500out/00010.dw500_pe1_out/00010.dw500_pe1_out.000001.dcm"
NIFTI_B0   = "/study/mrphys/DWI_3DGeo/processed-data/sub-dw500out/dwi-test/sub-dw500out_dir-LR_dwi_V000.nii.gz"
COEFFS_DAT = "/study3/isla/processing-code/qmri-neuropipe/dwi/gw_coils_magnus.dat"
OUT_NII    = "/study/mrphys/DWI_3DGeo/processed-data/sub-dw500out/dwi-test/sub-dw500out_dir-LR_dwi_V000_graddev_c_py_matchTORTOISE.nii.gz"

# DICOM_1    = "/scratch/mrphys/dw500in/00003.dw500_pe0_in/00003.dw500_pe0_in.000001.dcm"
# NIFTI_B0   = "/study/mrphys/DWI_3DGeo/processed-data/sub-dw500in/dwi-test/sub-dw500in_dir-LR_dwi_V000.nii.gz"
# COEFFS_DAT = "/study3/isla/processing-code/qmri-neuropipe/dwi/gw_coils_magnus.dat"
# OUT_NII    = "/study/mrphys/DWI_3DGeo/processed-data/sub-dw500in/dwi-test/sub-dw500in_dir-LR_dwi_V000_graddev_c_py_matchTORTOISE.nii.gz"

# DICOM_1    = "/scratch/mrphys/dw500center/00016.dw500_pe0_center/00016.dw500_pe0_center.000001.dcm"
# NIFTI_B0   = "/study/mrphys/DWI_3DGeo/processed-data/sub-dw500center/dwi-test/sub-dw500center_dir-LR_dwi_V000.nii.gz"
# COEFFS_DAT = "/study3/isla/processing-code/qmri-neuropipe/dwi/gw_coils_magnus.dat"
# OUT_NII    = "/study/mrphys/DWI_3DGeo/processed-data/sub-dw500center/dwi-test/sub-dw500center_dir-LR_dwi_V000_graddev_c_py_matchTORTOISE.nii.gz"

R0_MM  = 250.0
RGE_MM = 10.0

def extract_pdb_text(ds) -> str:
    tag = Tag(0x0025, 0x101B)
    pdb = ds[tag].value
    idx = pdb.find(b"\x1f\x8b")
    return gzip.decompress(pdb[idx:]).decode("latin1", errors="replace")

def kv_from_pdb(txt: str) -> dict:
    kv = {}
    for line in txt.splitlines():
        line = line.strip()
        m = re.match(r'^([A-Z0-9_]+)\s+"(.*)"\s*$', line)
        if m:
            kv[m.group(1)] = m.group(2)
    return kv

def parse_letter_value(token: str) -> float:
    token = token.strip().strip('"')
    m = re.match(r'^([LRAPSI])\s*([-+]?\d+(?:\.\d+)?)$', token)
    letter, val = m.group(1), float(m.group(2))
    return {"R":+val,"L":-val,"A":+val,"P":-val,"S":+val,"I":-val}[letter]

def pdb_center_ras_rel_iso(kv: dict) -> np.ndarray:
    x0 = parse_letter_value(kv["SLOC1"]); x1 = parse_letter_value(kv["ELOC1"])
    xc = 0.5*(x0+x1)
    y  = parse_letter_value(kv["FOVCNT1"])
    z  = parse_letter_value(kv["FOVCNT2"])
    return np.array([xc,y,z], float)

def nifti_center_ras_world(img: nib.Nifti1Image) -> np.ndarray:
    aff = img.affine
    sh = img.shape[:3]
    cijk = (np.array(sh, float) - 1.0)/2.0
    return (aff @ np.array([cijk[0],cijk[1],cijk[2],1.0]))[:3]

def direction_from_affine(aff: np.ndarray) -> np.ndarray:
    A = aff[:3,:3]
    spacing = np.sqrt((A*A).sum(axis=0))
    return A @ np.diag(1.0/spacing)

def read_ge_format_like_tortoise(coeffs_path: str, R0=R0_MM, rge=RGE_MM):
    renorm = np.ones(10, float)
    xynorm = np.ones(10, float)
    znorm  = np.ones(10, float)
    xynorm[1] = 1.0/np.sqrt(3.0)
    xynorm[2] = np.sqrt(8.0/3.0)
    xynorm[3] = np.sqrt(8.0/5.0)
    xynorm[4] = np.sqrt(64.0/15.0)
    znorm[1] = 2.0
    znorm[2] = 2.0
    znorm[3] = 8.0
    znorm[4] = 8.0
    for i in range(10):
        temp = (R0/rge)**i
        xynorm[i] *= temp
        znorm[i]  *= temp
        renorm[i]  = temp

    Xkeys=[]; Ykeys=[]; Zkeys=[]
    Xcoef=[]; Ycoef=[]; Zcoef=[]
    rx = re.compile(r'^(SCALE[XYZ])\s*([0-9]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$')
    with open(coeffs_path,"r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            m=rx.match(line)
            if not m: continue
            axis=m.group(1); tempind=int(m.group(2)); temp=float(m.group(3))
            if temp==0.0:
                if axis=="SCALEX" and len(Xcoef)==0: temp+=1.0
                elif axis=="SCALEY" and len(Ycoef)==0: temp+=1.0
                elif axis=="SCALEZ" and len(Zcoef)==0: temp+=1.0
                else: continue
            idx=tempind-1
            if idx<0 or idx>=10: continue
            if axis=="SCALEX":
                Xcoef.append(temp*xynorm[idx]); Xkeys += [tempind, 1]
            elif axis=="SCALEY":
                Ycoef.append(temp*xynorm[idx]); Ykeys += [tempind,-1]
            else:
                Zcoef.append(temp*znorm[idx]);  Zkeys += [tempind, 0]
    return (np.array(Xkeys,np.int32),np.array(Xcoef,float),
            np.array(Ykeys,np.int32),np.array(Ycoef,float),
            np.array(Zkeys,np.int32),np.array(Zcoef,float))

def make_factorials(nmax: int) -> np.ndarray:
    f = np.ones(nmax+1, float)
    for i in range(2,nmax+1): f[i]=f[i-1]*i
    return f

# --- load geometry / iso ---
img = nib.load(NIFTI_B0)
aff = img.affine.astype(float)
Rdir = direction_from_affine(aff).astype(float)

ds = pydicom.dcmread(DICOM_1, stop_before_pixels=True, force=True)
kv = kv_from_pdb(extract_pdb_text(ds))
ctr_ras_rel_iso = pdb_center_ras_rel_iso(kv)
# iso_ras_world = nifti_center_ras_world(img) - ctr_ras_rel_iso
# iso_ras_world = np.array([0.0, 0.0, 100.0], dtype=np.float64)


Xkeys,Xcoef,Ykeys,Ycoef,Zkeys,Zcoef = read_ge_format_like_tortoise(COEFFS_DAT)
max_ll = int(max(Xkeys[0::2].max(initial=1), Ykeys[0::2].max(initial=1), Zkeys[0::2].max(initial=1)))
FACT = make_factorials(max(64, max_ll+16))

try:
    import numba as nb
except Exception as e:
    raise SystemExit("Please install numba (conda/pip) for this block; pure-Python version omitted for brevity.")

@nb.njit(cache=True)
def plgndr(ll, mm, x):
    pmm = 1.0
    if mm > 0:
        somx2 = np.sqrt(1.0 - x*x)
        fact = 1.0
        for _ in range(1, mm+1):
            pmm = pmm * (-fact * somx2)
            fact += 2.0
    if ll == mm:
        return pmm
    pmmp1 = x*(2.0*mm + 1.0)*pmm
    if ll == mm+1:
        return pmmp1
    pll = 0.0
    for lm in range(mm+2, ll+1):
        pll = (x*(2.0*lm - 1.0)*pmmp1 - (lm + mm - 1.0)*pmm) / (lm - mm)
        pmm = pmmp1
        pmmp1 = pll
    return pll

@nb.njit(cache=True)
def sphericalz(ll, mm, phi, zz, fact):
    mma = mm if mm>=0 else -mm
    if mm == 0:
        return plgndr(ll,0,zz)
    f1 = fact[ll-mma]
    f2 = fact[ll+mma]
    pl = plgndr(ll,mma,zz)
    f = np.sqrt(2.0*f1/f2)
    result = f*pl
    sign = -1.0 if (mm & 1) else 1.0
    if mm < 0:
        return sign * result * np.sin(mma*phi)
    return sign * result * np.cos(mma*phi)

@nb.njit(cache=True)
def dshdx_X(ll, mml, phi, zz, fact):
    if mml >= 0:
        mm=mml
        if mm==0:
            if ll==1: return 0.0
            r1=np.sqrt(ll*(ll-1)/2.0)
            return -sphericalz(ll-1,1,phi,zz,fact)*r1
        r1=np.sqrt((ll+mm-1)*(ll+mm))
        result=r1*sphericalz(ll-1,mm-1,phi,zz,fact)/2.0
        if mm==1: result*=np.sqrt(2.0)
        if mm<=ll-2:
            r1=np.sqrt((ll-mm-1)*(ll-mm))/2.0
            result -= sphericalz(ll-1,mm+1,phi,zz,fact)*r1
        return result
    mm=-mml
    if mm==1:
        result=0.0
    else:
        r1=np.sqrt((ll+mm-1)*(ll+mm))
        result=r1*sphericalz(ll-1,-(mm-1),phi,zz,fact)/2.0
    if mm<=ll-2:
        r1=np.sqrt((ll-mm-1)*(ll-mm))/2.0
        result -= sphericalz(ll-1,-(mm+1),phi,zz,fact)*r1
    return result

@nb.njit(cache=True)
def dshdx_Y(ll, mml, phi, zz, fact):
    if mml >= 0:
        mm=mml
        if mm==0:
            if ll==1: return 0.0
            r1=np.sqrt(ll*(ll-1)/2.0)
            return -sphericalz(ll-1,-1,phi,zz,fact)*r1
        if mm==1:
            result=0.0
        else:
            r1=-np.sqrt((ll+mm-1)*(ll+mm))
            result=r1*sphericalz(ll-1,-(mm-1),phi,zz,fact)/2.0
        if mm<=ll-2:
            r1=np.sqrt((ll-mm-1)*(ll-mm))/2.0
            result -= sphericalz(ll-1,-(mm+1),phi,zz,fact)*r1
        return result
    mm=-mml
    r1=np.sqrt((ll+mm-1)*(ll+mm))
    result=r1*sphericalz(ll-1,(mm-1),phi,zz,fact)/2.0
    if mm==1: result*=np.sqrt(2.0)
    if mm<=ll-2:
        r1=np.sqrt((ll-mm-1)*(ll-mm))/2.0
        result += r1*sphericalz(ll-1,(mm+1),phi,zz,fact)
    return result

@nb.njit(cache=True)
def dshdx_Z(ll, mml, phi, zz, fact):
    if mml >= 0:
        mm=mml
        if ll==mm: return 0.0
        r1=np.sqrt((ll-mm)*(ll+mm))
        return sphericalz(ll-1,mm,phi,zz,fact)*r1
    mm=-mml
    if ll==mm: return 0.0
    r1=np.sqrt((ll-mm)*(ll+mm))
    return sphericalz(ll-1,mml,phi,zz,fact)*r1

@nb.njit(cache=True)
def dshdx(grad, ll, mml, rr, phi, zz, fact):
    if grad==0: base=dshdx_X(ll,mml,phi,zz,fact)
    elif grad==1: base=dshdx_Y(ll,mml,phi,zz,fact)
    else: base=dshdx_Z(ll,mml,phi,zz,fact)
    return base * (rr ** (ll-1))

@nb.njit(cache=True)
def pixel_bmatrix(point_mm, R0, Xkeys,Xcoef,Ykeys,Ycoef,Zkeys,Zcoef,fact):
    x1=point_mm[0]/R0; y1=point_mm[1]/R0; z1=point_mm[2]/R0
    rr=np.sqrt(x1*x1+y1*y1+z1*z1)
    if rr==0.0:
        return np.eye(3)
    phi=np.arctan2(y1,x1)
    zz=z1/rr
    axx=ayy=azz=0.0; axy=azy=0.0; ayx=azx=0.0; axz=ayz=0.0
    for kk in range(Xcoef.size):
        temp=Xcoef[kk]
        if temp==0.0: continue
        ll=int(Xkeys[2*kk]); mm=int(Xkeys[2*kk+1])
        axx += temp*dshdx(0,ll,mm,rr,phi,zz,fact)
        ayx += temp*dshdx(1,ll,mm,rr,phi,zz,fact)
        azx += temp*dshdx(2,ll,mm,rr,phi,zz,fact)
    for kk in range(Ycoef.size):
        temp=Ycoef[kk]
        if temp==0.0: continue
        ll=int(Ykeys[2*kk]); mm=int(Ykeys[2*kk+1])
        axy += temp*dshdx(0,ll,mm,rr,phi,zz,fact)
        ayy += temp*dshdx(1,ll,mm,rr,phi,zz,fact)
        azy += temp*dshdx(2,ll,mm,rr,phi,zz,fact)
    for kk in range(Zcoef.size):
        temp=Zcoef[kk]
        if temp==0.0: continue
        ll=int(Zkeys[2*kk]); mm=int(Zkeys[2*kk+1])
        axz += temp*dshdx(0,ll,mm,rr,phi,zz,fact)
        ayz += temp*dshdx(1,ll,mm,rr,phi,zz,fact)
        azz += temp*dshdx(2,ll,mm,rr,phi,zz,fact)
    M=np.empty((3,3))
    M[0,0]=axx; M[0,1]=axy; M[0,2]=axz
    M[1,0]=ayx; M[1,1]=ayy; M[1,2]=ayz
    M[2,0]=azx; M[2,1]=azy; M[2,2]=azz
    return M.T  # matches C++ pixel_bmatrix return


# 3x3 constant matrices
LPS2RAS_3 = np.array([[-1.0, 0.0, 0.0],
                      [ 0.0,-1.0, 0.0],
                      [ 0.0, 0.0, 1.0]], dtype=np.float64)
RAS2LPS_3 = LPS2RAS_3.copy()  # same

def prep_itk_lps_geometry(img: nib.Nifti1Image):
    """
    Build:
      aff_lps         : NIfTI affine expressed in ITK/LPS world
      D_itk           : ITK direction cosines (LPS)
      spacing         : voxel spacing (mm)
      aff_grad_lps    : aff_lps with GE Z-origin hack (only z-translation replaced)
    """
    aff_ras = img.affine.astype(np.float64)
    ras2lps_4 = np.diag([-1.0,-1.0, 1.0, 1.0])
    aff_lps = ras2lps_4 @ aff_ras

    A = aff_lps[:3,:3]
    spacing = np.sqrt((A*A).sum(axis=0))
    D_itk = A @ np.diag(1.0/spacing)

    shape = img.shape[:3]
    indv = (np.array(shape, dtype=np.float64) - 1.0)/2.0
    new_origv = -(D_itk @ (spacing * indv))   # matches TORTOISE GE origin hack formula
    aff_grad_lps = aff_lps.copy()
    aff_grad_lps[2,3] = new_origv[2]          # ONLY z origin replaced, like TORTOISE

    return aff_lps, D_itk, spacing, aff_grad_lps




@nb.njit(cache=False)
def inv3x3(M):
    a,b,c = M[0,0], M[0,1], M[0,2]
    d,e,f = M[1,0], M[1,1], M[1,2]
    g,h,i = M[2,0], M[2,1], M[2,2]

    A =  (e*i - f*h)
    B = -(d*i - f*g)
    C =  (d*h - e*g)
    D = -(b*i - c*h)
    E =  (a*i - c*g)
    F = -(a*h - b*g)
    G =  (b*f - c*e)
    H = -(a*f - c*d)
    I =  (a*e - b*d)

    det = a*A + b*B + c*C

    # Guard against singular / near-singular matrices
    if np.abs(det) < 1e-10:
        return np.eye(3, dtype=np.float64), det

    invdet = 1.0 / det
    Minv = np.empty((3,3), dtype=np.float64)
    Minv[0,0] = A*invdet; Minv[0,1] = D*invdet; Minv[0,2] = G*invdet
    Minv[1,0] = B*invdet; Minv[1,1] = E*invdet; Minv[1,2] = H*invdet
    Minv[2,0] = C*invdet; Minv[2,1] = F*invdet; Minv[2,2] = I*invdet

    return Minv, det


@nb.njit(parallel=True, cache=False)
def compute_graddev(out,
                    aff_eval_lps,            # 4x4 float64
                    iso_scanner_ras,         # (3,) float64
                    D_itk,                   # 3x3 float64 (ITK direction in LPS)
                    R0,
                    Xkeys,Xcoef,Ykeys,Ycoef,Zkeys,Zcoef,
                    fact,
                    lps2ras_3):              # 3x3 float64

    nx,ny,nz,_ = out.shape

    for k in nb.prange(nz):
        for j in range(ny):
            for i in range(nx):
                # # 1) ITK physical point in LPS, using the GE-hacked affine
                # pt_lps0 = aff_grad_lps[0,0]*i + aff_grad_lps[0,1]*j + aff_grad_lps[0,2]*k + aff_grad_lps[0,3]
                # pt_lps1 = aff_grad_lps[1,0]*i + aff_grad_lps[1,1]*j + aff_grad_lps[1,2]*k + aff_grad_lps[1,3]
                # pt_lps2 = aff_grad_lps[2,0]*i + aff_grad_lps[2,1]*j + aff_grad_lps[2,2]*k + aff_grad_lps[2,3]
                # pt_lps0 = aff_lps[0,0]*i + aff_lps[0,1]*j + aff_lps[0,2]*k + aff_lps[0,3]
                # pt_lps1 = aff_lps[1,0]*i + aff_lps[1,1]*j + aff_lps[1,2]*k + aff_lps[1,3]
                # pt_lps2 = aff_lps[2,0]*i + aff_lps[2,1]*j + aff_lps[2,2]*k + aff_lps[2,3]
                pt_lps0 = aff_eval_lps[0,0]*i + aff_eval_lps[0,1]*j + aff_eval_lps[0,2]*k + aff_eval_lps[0,3]
                pt_lps1 = aff_eval_lps[1,0]*i + aff_eval_lps[1,1]*j + aff_eval_lps[1,2]*k + aff_eval_lps[1,3]
                pt_lps2 = aff_eval_lps[2,0]*i + aff_eval_lps[2,1]*j + aff_eval_lps[2,2]*k + aff_eval_lps[2,3]

                # 2) Convert LPS -> scanner RAS
                pt_ras0 = lps2ras_3[0,0]*pt_lps0 + lps2ras_3[0,1]*pt_lps1 + lps2ras_3[0,2]*pt_lps2
                pt_ras1 = lps2ras_3[1,0]*pt_lps0 + lps2ras_3[1,1]*pt_lps1 + lps2ras_3[1,2]*pt_lps2
                pt_ras2 = lps2ras_3[2,0]*pt_lps0 + lps2ras_3[2,1]*pt_lps1 + lps2ras_3[2,2]*pt_lps2

                # 3) Apply isocenter offset in SCANNER RAS (this is the only “non-TORTOISE” part)
                pt_ras0 -= iso_scanner_ras[0]
                pt_ras1 -= iso_scanner_ras[1]
                pt_ras2 -= iso_scanner_ras[2]

                point = np.array([pt_ras0, pt_ras1, pt_ras2], dtype=np.float64)

                # 4) pixel_bmatrix gives BACKWARD Jacobian in scanner coords
                B = pixel_bmatrix(point, R0, Xkeys,Xcoef,Ykeys,Ycoef,Zkeys,Zcoef,fact)

                # 5) Convert Jacobian scanner(RAS) -> ITK(LPS):  L = ras2lps * B * lps2ras
                # Here ras2lps == lps2ras (diag[-1,-1,+1])
                # So: B_lps = lps2ras * B * lps2ras
                # (same as TORTOISE: Lmat = lps2ras * Lmat * lps2ras)
                B = lps2ras_3 @ B @ lps2ras_3

                # 6) TORTOISE convention:
                B = B.T                      # "make forward"
                B = D_itk.T @ B @ D_itk      # to ijk (native space)
                B = B.T                      # "HCP ordering" before storing

                out[i,j,k,0]=B[0,0]; out[i,j,k,1]=B[0,1]; out[i,j,k,2]=B[0,2]
                out[i,j,k,3]=B[1,0]; out[i,j,k,4]=B[1,1]; out[i,j,k,5]=B[1,2]
                out[i,j,k,6]=B[2,0]; out[i,j,k,7]=B[2,1]; out[i,j,k,8]=B[2,2]

                # At this point, B is your final L_ijk (after TORTOISE transpose conventions)


img = nib.load(NIFTI_B0)
aff_lps, D_itk, spacing, aff_grad_lps = prep_itk_lps_geometry(img)

out = np.zeros((*img.shape[:3], 9), np.float32)


A = img.affine[:3,:3].astype(np.float64)
sp = np.sqrt((A*A).sum(axis=0))

# # --- TEST 1: Match TORTOISE (no offset) ---
# iso_scanner_ras = np.array([0.0,0.0,0.0], dtype=np.float64)

# # --- TEST 2: Use your PDB-based isocenter offset (scanner RAS mm) ---
# # iso_scanner_ras = iso_ras_world_from_PDB.astype(np.float64)
# # Compute image center in the SAME RAS frame used for coefficient evaluation
# shape = img.shape[:3]
# cijk = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
# pt_lps_center = (aff_grad_lps @ np.array([cijk[0], cijk[1], cijk[2], 1.0]))[:3]
# pt_ras_center = LPS2RAS_3 @ pt_lps_center

# # Now isocenter in that same RAS frame:
# iso_scanner_ras = pt_ras_center - ctr_ras_rel_iso.astype(np.float64)
# # iso_scanner_ras = iso_ras_world
# print("ctr_ras_rel_iso (PDB):", ctr_ras_rel_iso)
# print("pt_ras_center (eval frame):", pt_ras_center)
# print("iso_scanner_ras used:", iso_scanner_ras)
# Choose evaluation affine (must match isocenter computation!)
USE_GE_HACK = True  # True -> aff_grad_lps, False -> aff_lps
aff_eval_lps = aff_grad_lps if USE_GE_HACK else aff_lps
shape = img.shape[:3]
cijk = (np.array(shape, dtype=np.float64) - 1.0) / 2.0

pt_lps_center = (aff_eval_lps @ np.array([cijk[0], cijk[1], cijk[2], 1.0]))[:3]
pt_ras_center = LPS2RAS_3 @ pt_lps_center

iso_scanner_ras = pt_ras_center - ctr_ras_rel_iso.astype(np.float64)

print("USE_GE_HACK:", USE_GE_HACK)
print("ctr_ras_rel_iso (PDB):", ctr_ras_rel_iso)
print("pt_ras_center (eval frame):", pt_ras_center)
print("iso_scanner_ras used:", iso_scanner_ras)

# compute_graddev(out,
#                 aff_grad_lps.astype(np.float64),
#                 iso_scanner_ras,
#                 D_itk.astype(np.float64),
#                 R0_MM,
#                 Xkeys, Xcoef, Ykeys, Ycoef, Zkeys, Zcoef,
#                 FACT,
#                 LPS2RAS_3)

compute_graddev(out,
                aff_eval_lps.astype(np.float64),     # <-- swap here
                iso_scanner_ras,
                D_itk.astype(np.float64),
                R0_MM,
                Xkeys, Xcoef, Ykeys, Ycoef, Zkeys, Zcoef,
                FACT,
                LPS2RAS_3)

nib.save(nib.Nifti1Image(out, img.affine, img.header),
         OUT_NII)


# @nb.njit(parallel=True, cache=True)
# def compute_graddev(out, aff, iso_ras_world, Rdir, R0, Xkeys,Xcoef,Ykeys,Ycoef,Zkeys,Zcoef,fact):
#     nx,ny,nz,_=out.shape
#     for k in nb.prange(nz):
#         for j in range(ny):
#             for i in range(nx):
#                 p0 = aff[0,0]*i + aff[0,1]*j + aff[0,2]*k + aff[0,3]
#                 p1 = aff[1,0]*i + aff[1,1]*j + aff[1,2]*k + aff[1,3]
#                 p2 = aff[2,0]*i + aff[2,1]*j + aff[2,2]*k + aff[2,3]
#                 r0 = p0 - iso_ras_world[0]
#                 r1 = p1 - iso_ras_world[1]
#                 r2 = p2 - iso_ras_world[2]
#                 point=np.array([r0,r1,r2])

#                 B = pixel_bmatrix(point, R0, Xkeys,Xcoef,Ykeys,Ycoef,Zkeys,Zcoef,fact)  # backward
#                 B = B.T                                  # forward (TORTOISE convention)
#                 B = Rdir.T @ B @ Rdir                    # to ijk
#                 B = B.T                                  # HCP ordering transpose

#                 out[i,j,k,0]=B[0,0]; out[i,j,k,1]=B[0,1]; out[i,j,k,2]=B[0,2]
#                 out[i,j,k,3]=B[1,0]; out[i,j,k,4]=B[1,1]; out[i,j,k,5]=B[1,2]
#                 out[i,j,k,6]=B[2,0]; out[i,j,k,7]=B[2,1]; out[i,j,k,8]=B[2,2]

# img = nib.load(NIFTI_B0)
# aff = img.affine.astype(np.float64)
# Rdir = direction_from_affine(aff).astype(np.float64)
# out = np.zeros((*img.shape[:3],9), np.float32)

# compute_graddev(out, aff, iso_ras_world.astype(np.float64), Rdir, R0_MM,
#                 Xkeys, Xcoef, Ykeys, Ycoef, Zkeys, Zcoef, FACT)

# nib.save(nib.Nifti1Image(out, aff, img.header), OUT_NII)
print("Wrote:", OUT_NII)


# --- Sweep offsets around PDB-based isocenter estimate ---
# iso0 = iso_scanner_ras.copy()   # baseline from PDB + geometry

# # Choose sweep axis: 'x', 'y', or 'z'
# SWEEP_AXIS = "z"

# # Offsets to try (mm). Example: +/- 60 mm in 10 mm steps.
# # Adjust as you like.
# deltas_mm = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float64)

# # Helper to build output filenames
# def outname_for_delta(base_out, axis, d):
#     # d in mm, keep sign and compact formatting
#     tag = f"iso{axis.upper()}{'p' if d>=0 else 'm'}{abs(d):g}mm"
#     if base_out.endswith(".nii.gz"):
#         return base_out.replace(".nii.gz", f"_{tag}.nii.gz")
#     if base_out.endswith(".nii"):
#         return base_out.replace(".nii", f"_{tag}.nii.gz")
#     return base_out + f"_{tag}.nii.gz"

# for d in deltas_mm:
#     iso = iso0.copy()
#     if SWEEP_AXIS == "x":
#         iso[0] += d
#     elif SWEEP_AXIS == "y":
#         iso[1] += d
#     else:
#         iso[2] += d

#     out = np.zeros((*img.shape[:3], 9), np.float32)

#     print(f"\n--- Sweep {SWEEP_AXIS} delta {d:+g} mm ---")
#     print("iso_scanner_ras:", iso)

#     compute_graddev(out,
#                     aff_grad_lps.astype(np.float64),
#                     iso,
#                     D_itk.astype(np.float64),
#                     R0_MM,
#                     Xkeys, Xcoef, Ykeys, Ycoef, Zkeys, Zcoef,
#                     FACT,
#                     LPS2RAS_3)

#     out_nii = outname_for_delta(OUT_NII, SWEEP_AXIS, d)
#     nib.save(nib.Nifti1Image(out, img.affine, img.header), out_nii)
#     print("Wrote:", out_nii)

# print("\nDone sweep.")