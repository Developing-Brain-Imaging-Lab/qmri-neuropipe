import numpy as np

def correct_bvals_bvecs(bvals, bvecs, grad_nonlin):
    """
    Corrects b-values and b-vectors for gradient non-linearity effects in Python.

    Args:
        bvals (np.array): 1D numpy array containing the original b-values.
        bvecs (np.array): 2D numpy array containing the original b-vectors (shape: [3, N]).
        grad_nonlin (np.array): 1D numpy array containing the parameters describing the gradient
                                non-linearity correction (length: 9).

    Returns:
        Tuple[np.array, np.array]: Corrected b-values and b-vectors.
    """
    # Initialize the corrected b-values and b-vectors arrays with the original values
    bvals_c = np.copy(bvals)
    bvecs_c = np.copy(bvecs)

    # Construct the gradient coil tensor L from the gradient non-linearity parameters
    L = np.array([[grad_nonlin[0], grad_nonlin[3], grad_nonlin[6]],
                  [grad_nonlin[1], grad_nonlin[4], grad_nonlin[7]],
                  [grad_nonlin[2], grad_nonlin[5], grad_nonlin[8]]])

    # Identity matrix
    Id = np.eye(3)

    # Iterate over each diffusion measurement to correct the b-values and b-vectors
    for i in range(bvals.shape[0]):
        if bvals[i] > 0:  # Correct only for non-zero b-values
            # Correct the b-vector for the current gradient
            bvecs_c[i] = np.matmul(L, bvecs[i])

            # Normalize the corrected b-vector
            mag = np.linalg.norm(bvecs_c[i])
            if mag != 0:
                bvecs_c[i] = bvecs_c[i] / mag

            # Adjust the b-value
            bvals_c[i] = mag**2 * bvals[i]

    return bvals_c, bvecs_c
