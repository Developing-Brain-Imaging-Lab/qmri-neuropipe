"""Optional FireANTs affine-initialized deformable registration."""

from pathlib import Path


def registration(
    fixed_file: Path,
    moving_file: Path,
    warp_file: Path,
    *,
    device: str = "cuda",
    scales=(4, 2, 1),
    affine_iterations=(200, 100, 50),
    iterations=(200, 100, 25),
) -> Path:
    """Estimate and export a forward image warp in ANTs physical coordinates."""
    if not scales or len(scales) != len(iterations) or len(scales) != len(affine_iterations):
        raise ValueError("FireANTs scales and both iteration schedules must have the same nonzero length.")

    try:
        from fireants.io import BatchedImages, Image
        from fireants.registration.affine import AffineRegistration
        from fireants.registration.greedy import GreedyRegistration
    except ImportError as exc:
        raise RuntimeError("Install FireANTs and its PyTorch dependencies to use this backend.") from exc

    fixed = BatchedImages([Image.load_file(str(fixed_file), device=device)])
    moving = BatchedImages([Image.load_file(str(moving_file), device=device)])
    affine = AffineRegistration(
        list(scales), list(affine_iterations), fixed, moving,
        optimizer="Adam", optimizer_lr=0.003, cc_kernel_size=5,
    )
    affine.optimize()
    deformable = GreedyRegistration(
        scales=list(scales), iterations=list(iterations),
        fixed_images=fixed, moving_images=moving,
        cc_kernel_size=5, deformation_type="compositive", smooth_grad_sigma=1,
        optimizer="adam", optimizer_lr=0.5, init_affine=affine.get_affine_matrix().detach(),
    )
    deformable.optimize()
    warp_file.parent.mkdir(parents=True, exist_ok=True)
    deformable.save_as_ants_transforms(str(warp_file))
    return warp_file
