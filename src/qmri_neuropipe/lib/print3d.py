from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


def _require_marching_cubes():
    try:
        from skimage import measure
    except ImportError as exc:
        raise RuntimeError(
            "3D print mesh export requires scikit-image. Install with: "
            "pip install scikit-image"
        ) from exc
    return measure.marching_cubes


def _parse_labels(labels: str | Iterable[int] | None) -> set[int] | None:
    if labels is None or labels == "":
        return None
    if isinstance(labels, str):
        return {int(item.strip()) for item in labels.split(",") if item.strip()}
    return {int(item) for item in labels}


def _prepare_volume(
    image_path: Path,
    *,
    threshold: float,
    labels: str | Iterable[int] | None,
    smooth_sigma: float,
    fill_holes: bool,
    closing_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(image_path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Input must be a 3D mask/segmentation image: {image_path}")

    label_set = _parse_labels(labels)
    if label_set is not None:
        mask = np.isin(data.astype(np.int64), list(label_set))
    else:
        mask = data > threshold

    if not np.any(mask):
        raise ValueError(f"No voxels selected from {image_path}")

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("3D print mesh export requires scipy.") from exc

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if closing_iterations > 0:
        mask = ndimage.binary_closing(mask, iterations=closing_iterations)

    volume = mask.astype(np.float32)
    if smooth_sigma > 0:
        volume = ndimage.gaussian_filter(volume, sigma=smooth_sigma)
    return volume, img.affine


def mesh_from_nifti_mask(
    image_path: Path,
    *,
    threshold: float = 0.5,
    labels: str | Iterable[int] | None = None,
    smooth_sigma: float = 0.5,
    fill_holes: bool = True,
    closing_iterations: int = 0,
) -> Mesh:
    marching_cubes = _require_marching_cubes()
    volume, affine = _prepare_volume(
        image_path,
        threshold=threshold,
        labels=labels,
        smooth_sigma=smooth_sigma,
        fill_holes=fill_holes,
        closing_iterations=closing_iterations,
    )
    level = 0.5 if labels is not None else threshold
    if smooth_sigma > 0:
        level = 0.5
    verts, faces, _normals, _values = marching_cubes(volume, level=level)
    verts_h = np.c_[verts, np.ones(len(verts), dtype=verts.dtype)]
    world = (affine @ verts_h.T).T[:, :3]
    return Mesh(vertices=world.astype(np.float32), faces=faces.astype(np.int64))


def combine_meshes(meshes: list[Mesh]) -> Mesh:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    if not vertices:
        raise ValueError("No meshes to combine")
    return Mesh(vertices=np.vstack(vertices), faces=np.vstack(faces))


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _streamline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _downsample_streamline(points: np.ndarray, point_step: int) -> np.ndarray:
    point_step = max(1, int(point_step))
    if point_step == 1 or len(points) <= 2:
        return points
    sampled = points[::point_step]
    if not np.array_equal(sampled[-1], points[-1]):
        sampled = np.vstack([sampled, points[-1]])
    return sampled


def _initial_normal(tangent: np.ndarray) -> np.ndarray:
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(tangent, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    normal = np.cross(tangent, ref)
    return _unit_vector(normal)


def tube_mesh_from_streamline(
    points: np.ndarray,
    *,
    radius: float = 0.4,
    sides: int = 8,
    cap_ends: bool = True,
) -> Mesh:
    points = np.asarray(points, dtype=np.float32)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        raise ValueError("Streamline must contain at least two finite points")
    if sides < 3:
        raise ValueError("Tube mesh requires at least 3 sides")

    tangents = np.zeros_like(points, dtype=np.float32)
    tangents[0] = _unit_vector(points[1] - points[0])
    tangents[-1] = _unit_vector(points[-1] - points[-2])
    for idx in range(1, len(points) - 1):
        tangents[idx] = _unit_vector(points[idx + 1] - points[idx - 1])

    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring_vertices: list[np.ndarray] = []
    faces: list[list[int]] = []

    normal = _initial_normal(tangents[0])
    for idx, (point, tangent) in enumerate(zip(points, tangents)):
        projected = normal - np.dot(normal, tangent) * tangent
        if np.linalg.norm(projected) < 1e-6:
            projected = _initial_normal(tangent)
        normal = _unit_vector(projected)
        binormal = _unit_vector(np.cross(tangent, normal))
        ring = [
            point + radius * (np.cos(angle) * normal + np.sin(angle) * binormal)
            for angle in angles
        ]
        ring_vertices.extend(ring)

        if idx == 0:
            continue
        prev = (idx - 1) * sides
        curr = idx * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            faces.append([prev + side, prev + next_side, curr + side])
            faces.append([curr + side, prev + next_side, curr + next_side])

    if cap_ends:
        start_center = len(ring_vertices)
        ring_vertices.append(points[0])
        end_center = len(ring_vertices)
        ring_vertices.append(points[-1])
        end_base = (len(points) - 1) * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            faces.append([start_center, next_side, side])
            faces.append([end_center, end_base + side, end_base + next_side])

    return Mesh(
        vertices=np.asarray(ring_vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
    )


def _select_streamline_indices(
    n_streamlines: int,
    *,
    max_streamlines: int | None,
    every_n: int,
    random_sample: bool,
    seed: int,
) -> np.ndarray:
    indices = np.arange(n_streamlines)
    every_n = max(1, int(every_n))
    if every_n > 1:
        indices = indices[::every_n]
    if max_streamlines and len(indices) > max_streamlines:
        if random_sample:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(indices, size=max_streamlines, replace=False))
        else:
            indices = indices[:max_streamlines]
    return indices


def mesh_from_streamlines(
    tractogram_path: Path,
    *,
    radius: float = 0.4,
    sides: int = 8,
    max_streamlines: int | None = 2000,
    every_n: int = 1,
    random_sample: bool = False,
    seed: int = 1,
    min_length: float = 5.0,
    point_step: int = 2,
    cap_ends: bool = True,
) -> Mesh:
    tractogram = nib.streamlines.load(str(tractogram_path))
    streamlines = tractogram.streamlines
    indices = _select_streamline_indices(
        len(streamlines),
        max_streamlines=max_streamlines,
        every_n=every_n,
        random_sample=random_sample,
        seed=seed,
    )

    meshes: list[Mesh] = []
    skipped = 0
    for idx in indices:
        points = np.asarray(streamlines[int(idx)], dtype=np.float32)
        if _streamline_length(points) < min_length:
            skipped += 1
            continue
        points = _downsample_streamline(points, point_step)
        if len(points) < 2:
            skipped += 1
            continue
        try:
            meshes.append(tube_mesh_from_streamline(points, radius=radius, sides=sides, cap_ends=cap_ends))
        except ValueError:
            skipped += 1

    if not meshes:
        raise ValueError(f"No printable streamlines selected from {tractogram_path}; skipped={skipped}")
    return combine_meshes(meshes)


def _face_normal(tri: np.ndarray) -> np.ndarray:
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.zeros(3, dtype=np.float32)
    return (normal / norm).astype(np.float32)


def write_ascii_stl(mesh: Mesh, output: Path, solid_name: str = "qmri_neuropipe_mesh") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write(f"solid {solid_name}\n")
        for face in mesh.faces:
            tri = mesh.vertices[face]
            normal = _face_normal(tri)
            f.write(f"  facet normal {normal[0]:.7g} {normal[1]:.7g} {normal[2]:.7g}\n")
            f.write("    outer loop\n")
            for vertex in tri:
                f.write(f"      vertex {vertex[0]:.7g} {vertex[1]:.7g} {vertex[2]:.7g}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def write_obj(mesh: Mesh, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write("# Generated by qmri-neuropipe\n")
        for vertex in mesh.vertices:
            f.write(f"v {vertex[0]:.7g} {vertex[1]:.7g} {vertex[2]:.7g}\n")
        for face in mesh.faces:
            a, b, c = face + 1
            f.write(f"f {a} {b} {c}\n")


def write_mesh(mesh: Mesh, output: Path, file_format: str = "auto") -> None:
    fmt = file_format.lower()
    if fmt == "auto":
        fmt = output.suffix.lower().lstrip(".")
    if fmt == "stl":
        write_ascii_stl(mesh, output)
    elif fmt == "obj":
        write_obj(mesh, output)
    else:
        raise ValueError("Output format must be stl, obj, or auto from .stl/.obj suffix")


def export_print_mesh(
    inputs: list[Path],
    output: Path,
    *,
    threshold: float = 0.5,
    labels: str | None = None,
    smooth_sigma: float = 0.5,
    fill_holes: bool = True,
    closing_iterations: int = 0,
    file_format: str = "auto",
) -> Mesh:
    meshes = [
        mesh_from_nifti_mask(
            path,
            threshold=threshold,
            labels=labels,
            smooth_sigma=smooth_sigma,
            fill_holes=fill_holes,
            closing_iterations=closing_iterations,
        )
        for path in inputs
    ]
    mesh = combine_meshes(meshes)
    write_mesh(mesh, output, file_format=file_format)
    return mesh


def export_streamline_tube_mesh(
    tractogram_path: Path,
    output: Path,
    *,
    radius: float = 0.4,
    sides: int = 8,
    max_streamlines: int | None = 2000,
    every_n: int = 1,
    random_sample: bool = False,
    seed: int = 1,
    min_length: float = 5.0,
    point_step: int = 2,
    cap_ends: bool = True,
    file_format: str = "auto",
) -> Mesh:
    mesh = mesh_from_streamlines(
        tractogram_path,
        radius=radius,
        sides=sides,
        max_streamlines=max_streamlines,
        every_n=every_n,
        random_sample=random_sample,
        seed=seed,
        min_length=min_length,
        point_step=point_step,
        cap_ends=cap_ends,
    )
    write_mesh(mesh, output, file_format=file_format)
    return mesh
