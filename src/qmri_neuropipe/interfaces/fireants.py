"""Optional FireANTs rigid, affine, and deformable registration."""

from pathlib import Path


def registration(
    fixed_file: Path,
    moving_file: Path,
    warp_file: Path,
    *,
    transform_type: str = "nonlinear",
    device: str = "cuda",
    scales=(4, 2, 1),
    affine_iterations=(200, 100, 50),
    iterations=None,
) -> Path:
    """Export an ANTs-compatible matrix or affine-initialized deformable warp."""
    if transform_type not in {"rigid", "affine", "nonlinear"}:
        raise ValueError(f"Unsupported FireANTs transform: {transform_type}")
    nonlinear = transform_type == "nonlinear"
    if iterations is None:
        iterations = (200, 100, 25) if nonlinear else affine_iterations
    if not scales or len(scales) != len(iterations) or (nonlinear and len(scales) != len(affine_iterations)):
        raise ValueError("Each FireANTs iteration schedule used must have one entry per scale.")

    try:
        from fireants.io import BatchedImages, Image
        if transform_type == "rigid":
            from fireants.registration.rigid import RigidRegistration as LinearRegistration
        else:
            from fireants.registration.affine import AffineRegistration as LinearRegistration
        if nonlinear:
            from fireants.registration.greedy import GreedyRegistration
    except ImportError as exc:
        raise RuntimeError("Install FireANTs and its PyTorch dependencies to use this backend.") from exc

    fixed = BatchedImages([Image.load_file(str(fixed_file), device=device)])
    moving = BatchedImages([Image.load_file(str(moving_file), device=device)])
    linear = LinearRegistration(
        list(scales), list(affine_iterations if nonlinear else iterations), fixed, moving,
        optimizer="Adam", optimizer_lr=0.003, cc_kernel_size=5,
    )
    linear.optimize()
    warp_file.parent.mkdir(parents=True, exist_ok=True)
    if not nonlinear:
        linear.save_as_ants_transforms(str(warp_file))
        return warp_file

    deformable = GreedyRegistration(
        scales=list(scales), iterations=list(iterations),
        fixed_images=fixed, moving_images=moving,
        cc_kernel_size=5, deformation_type="compositive", smooth_grad_sigma=1,
        optimizer="adam", optimizer_lr=0.5, init_affine=linear.get_affine_matrix().detach(),
    )
    deformable.optimize()
    deformable.save_as_ants_transforms(str(warp_file))
    return warp_file
