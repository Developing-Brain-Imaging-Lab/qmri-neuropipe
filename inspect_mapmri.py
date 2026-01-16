
import numpy as np
import dipy.reconst.mapmri as mapmri
from dipy.core.gradients import gradient_table

# Create dummy data
bvals = np.concatenate([[0], np.tile(1000, 29)])
# Fix bvecs shape: [[0,0,0]] is (1,3), random is (29,3)
bvecs = np.concatenate([[[0,0,0]], np.random.random((29, 3))])
# Normalize bvecs
norms = np.linalg.norm(bvecs, axis=1)
norms[norms==0] = 1
bvecs = bvecs / norms[:, None]

gtab = gradient_table(bvals, bvecs=bvecs)

print("Fitting MapmriModel...")
model = mapmri.MapmriModel(gtab, radial_order=4, laplacian_regularization=True, laplacian_weighting=0.2)
# Create a single voxel signal (must be matching number of bvals)
signal = np.random.random(30)
fit = model.fit(signal)

print("\n--- MapmriFit Attributes ---")
print(dir(fit))

print("\n--- Checking specific attributes ---")
for attr in ['model_params', 'mapmri_params', 'mapmri_coeffs', 'coeffs', '_mapmri_coeffs']:
    if hasattr(fit, attr):
        print(f"Found: {attr}")
    else:
        print(f"Missing: {attr}")
