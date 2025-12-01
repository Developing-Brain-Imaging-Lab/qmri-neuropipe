from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

@dataclass
class ImageFile:
    """Generic BIDS-like image + sidecar."""
    entities: dict[str, str]     # e.g. {'sub': '01', 'ses': '01', 'modality': 'dwi'}
    img: Path                    # NIfTI file
    json: Optional[Path] = None  # sidecar JSON (if any)

    def __repr__(self) -> str:
        ents = ",".join(f"{k}-{v}" for k, v in self.entities.items())
        return f"ImageFile({ents}, img={self.img.name})"


@dataclass
class DWIFile(ImageFile):
    bval: Optional[Path] = None
    bvec: Optional[Path] = None

    def __repr__(self) -> str:
        ents = ",".join(f"{k}-{v}" for k, v in self.entities.items())
        return (
            f"DWIFile({ents}, img={self.img.name}, "
            f"bval={self.bval.name if self.bval else None}, "
            f"bvec={self.bvec.name if self.bvec else None})"
        )
    

ImageLike = Union[ImageFile, DWIFile]