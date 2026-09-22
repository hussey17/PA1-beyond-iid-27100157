"""PACS discovery, validation, and PyTorch dataset wrappers.

The training API deliberately exposes an unlabeled target view.  Target class
labels are loaded only by the final-evaluation entry point.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

PACS_DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
PACS_CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_PACS_GDRIVE_ID = "1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"


@dataclass(frozen=True)
class PACSSample:
    path: Path
    sample_id: str
    domain: str
    class_name: str
    label: int


def discover_pacs_root(search_root: str | Path) -> Path:
    """Find the directory whose children are the four PACS domains."""
    search_root = Path(search_root).expanduser().resolve()
    candidates = [search_root]
    if search_root.exists():
        candidates.extend(path for path in search_root.rglob("*") if path.is_dir())
    for candidate in candidates:
        if all((candidate / domain).is_dir() for domain in PACS_DOMAINS):
            return candidate
    raise FileNotFoundError(
        f"Could not find PACS below {search_root}. Expected a directory containing "
        f"{', '.join(PACS_DOMAINS)}."
    )


def scan_domain(pacs_root: str | Path, domain: str) -> list[PACSSample]:
    """Return a stable, validated inventory for one PACS domain."""
    root = discover_pacs_root(pacs_root)
    if domain not in PACS_DOMAINS:
        raise ValueError(f"Unknown PACS domain: {domain}")
    samples: list[PACSSample] = []
    for label, class_name in enumerate(PACS_CLASSES):
        class_dir = root / domain / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing PACS class directory: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(
                    PACSSample(
                        path=path,
                        sample_id=path.relative_to(root).as_posix(),
                        domain=domain,
                        class_name=class_name,
                        label=label,
                    )
                )
    if not samples:
        raise RuntimeError(f"No images found for PACS domain {domain!r} in {root}")
    return samples


def scan_unlabeled_domain(pacs_root: str | Path, domain: str) -> list[tuple[Path, str]]:
    """Inventory a domain without deriving or storing class labels."""
    root = discover_pacs_root(pacs_root)
    if domain not in PACS_DOMAINS:
        raise ValueError(f"Unknown PACS domain: {domain}")
    domain_root = root / domain
    paths = [
        path
        for path in sorted(domain_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    items = [
        (path, f"{domain}/unlabeled_{index:06d}") for index, path in enumerate(paths)
    ]
    if not items:
        raise RuntimeError(f"No images found for PACS domain {domain!r} in {root}")
    return items


def validate_inventory(samples: Iterable[PACSSample]) -> None:
    """Fail early on duplicate identifiers or unreadable images."""
    samples = list(samples)
    identifiers = [sample.sample_id for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("PACS inventory contains duplicate sample identifiers")
    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                image.verify()
        except Exception as error:
            raise ValueError(f"Unreadable PACS image: {sample.path}") from error


def download_and_prepare_pacs(
    download_root: str | Path,
    file_id: str = DEFAULT_PACS_GDRIVE_ID,
) -> Path:
    """Download PACS atomically from the public benchmark mirror and verify it."""
    import gdown

    download_root = Path(download_root).expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    try:
        return discover_pacs_root(download_root)
    except FileNotFoundError:
        pass

    archive = download_root / "PACS.zip"
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        partial = download_root / "PACS.zip.part"
        if partial.exists():
            partial.unlink()
        result = gdown.download(id=file_id, output=str(partial), quiet=False)
        if result is None or not zipfile.is_zipfile(partial):
            raise RuntimeError("PACS download did not produce a valid ZIP archive")
        partial.replace(archive)

    temporary_root = Path(tempfile.mkdtemp(prefix="pacs_extract_", dir=download_root))
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination = (temporary_root / member.filename).resolve()
                if (
                    temporary_root not in destination.parents
                    and destination != temporary_root
                ):
                    raise RuntimeError(
                        f"Unsafe path in PACS archive: {member.filename}"
                    )
            bundle.extractall(temporary_root)
        extracted_root = discover_pacs_root(temporary_root)
        final_root = download_root / "PACS"
        if final_root.exists():
            raise RuntimeError(
                f"{final_root} exists but is not a valid PACS root; move it aside and rerun"
            )
        shutil.move(str(extracted_root), str(final_root))
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    root = discover_pacs_root(final_root)
    inventories = [scan_unlabeled_domain(root, domain) for domain in PACS_DOMAINS]
    if sum(map(len, inventories)) != 9991:
        raise RuntimeError(
            "Prepared PACS inventory does not contain the expected 9,991 images"
        )
    return root


class PACSLabeledDataset:
    """Labeled source or final-evaluation dataset."""

    def __init__(
        self, samples: Iterable[PACSSample], transform: Callable | None = None
    ):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label, sample.sample_id


class PACSUnlabeledDataset:
    """Target view that never returns or stores a class label."""

    def __init__(
        self,
        paths_and_ids: Iterable[tuple[Path, str]],
        transform: Callable | None = None,
    ):
        self.items = [(Path(path), str(sample_id)) for path, sample_id in paths_and_ids]
        self.transform = transform

    @classmethod
    def from_samples(
        cls, samples: Iterable[PACSSample], transform: Callable | None = None
    ):
        return cls(
            ((sample.path, sample.sample_id) for sample in samples), transform=transform
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, sample_id = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample_id
