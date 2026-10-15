from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from torch.utils.data import Dataset
from torchvision import transforms as T

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
WEIGHTS = np.array([.1, .1, .1, .2, .5], dtype=np.float64)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_table(csv_path, image_root):
    frame = pd.read_csv(csv_path)
    if "image_path" not in frame:
        raise ValueError("CSV must contain image_path.")
    if {"target_name", "target"}.issubset(frame.columns):
        if frame.duplicated(["image_path", "target_name"]).any():
            raise ValueError("Duplicate image/target rows.")
        meta_cols = [c for c in frame if c not in {"sample_id", "target_name", "target"}]
        for col in meta_cols:
            if frame.groupby("image_path")[col].nunique(dropna=False).max() > 1:
                raise ValueError(f"Inconsistent metadata for an image: {col}")
        meta = frame[meta_cols].drop_duplicates("image_path").set_index("image_path")
        frame = meta.join(frame.pivot(index="image_path", columns="target_name", values="target")).reset_index()
    missing = set(TARGETS) - set(frame)
    if missing:
        raise ValueError(f"Missing targets: {sorted(missing)}")
    if frame.image_path.duplicated().any():
        raise ValueError("Each biological image must have one row.")
    values = frame[TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Targets must be finite, non-negative grams.")
    if not np.allclose(values[:, 3], values[:, 0] + values[:, 2], atol=.15):
        raise ValueError("GDM != Green + Clover; audit target semantics.")
    total_residual = values[:, 4] - values[:, 3] - values[:, 1]
    discrepancy_count = int((np.abs(total_residual) > .15).sum())
    identity_audit = {"total_identity_discrepancies": discrepancy_count,
                      "max_abs_total_residual_g": float(np.abs(total_residual).max())}
    if discrepancy_count:
        warnings.warn(f"{discrepancy_count} annotation(s) have Total != GDM + Dead "
                      f"by >0.15 g (max {identity_audit['max_abs_total_residual_g']:.4f} g). "
                      "Keeping original ground truth; discrepancy is recorded in the data audit.",
                      UserWarning, stacklevel=2)
    root = Path(image_root).resolve()
    paths = [(root / str(p)).resolve() for p in frame.image_path]
    if any(not p.is_relative_to(root) for p in paths):
        raise ValueError("image_path escapes image_root.")
    absent = [str(p) for p in paths if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing {len(absent)} images; first: {absent[0]}")
    frame["absolute_path"] = [str(p) for p in paths]
    result = frame.sort_values("image_path").reset_index(drop=True)
    result.attrs["target_identity_audit"] = identity_audit
    return result


def assign_folds(frame, group_column="Sampling_Date", folds=5, seed=42):
    if group_column not in frame or frame[group_column].isna().any():
        raise ValueError(f"Need a non-missing acquisition group column: {group_column}")
    groups = frame[group_column].astype(str)
    if groups.nunique() < folds:
        raise ValueError(f"Only {groups.nunique()} groups for {folds} folds.")
    # Public Sampling_Date is a documented proxy, not an invented true site ID.
    if "State" in frame and not frame.State.isna().any():
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        splits = splitter.split(frame, frame.State.astype(str), groups)
    else:
        splits = GroupKFold(folds).split(frame, groups=groups)
    result = frame.copy()
    result["group_id"] = groups
    result["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(splits):
        if set(groups.iloc[train_idx]) & set(groups.iloc[val_idx]):
            raise AssertionError("Acquisition group crossed folds.")
        result.loc[val_idx, "fold"] = fold
    if (result.fold < 0).any():
        raise AssertionError("Incomplete fold assignment.")
    return result


def external_manifest(roots, supervised, excluded_parts=None):
    excluded = set(excluded_parts or ["synthetic", "synthetic_images", "biomass", "test", "masks", "labels"])
    seen = {}
    supervised_hashes = {file_sha256(p) for p in supervised.absolute_path}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                continue
            if excluded.intersection(part.lower() for part in path.relative_to(root).parts[:-1]):
                continue
            digest = file_sha256(path)
            if digest in supervised_hashes:
                raise ValueError(f"CSIRO image appears in external SSL corpus: {path}")
            seen.setdefault(digest, {"path": str(path), "sha256": digest})
    if not seen:
        raise ValueError("No external real images found; specify the real/unlabelled directory explicitly.")
    return sorted(seen.values(), key=lambda item: item["path"])


def normalize_transform(size, train=False):
    # Full field of view is retained; no random crop changes the labelled harvest area.
    ops = [T.Resize((size, size), antialias=True)]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(.15, .15, .1, .02)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class ExternalImages(Dataset):
    def __init__(self, manifest, size, augmentation="legacy"):
        self.records = json.loads(Path(manifest).read_text())
        self.transform = normalize_transform(size, train=True)
        if augmentation == "crop":
            # Unlabelled crops have no harvested-area mass label to invalidate.
            self.transform = T.Compose([T.RandomResizedCrop(size, scale=(.5, 1.), ratio=(.8, 1.2)),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(.2, .2, .15, .03),
                T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]["path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            return self.transform(image)


def patch_class_fractions(mask, grid, classes=6):
    """Soft semantic occupancy of each transformer patch; no biomass label conversion."""
    values = torch.as_tensor(np.asarray(mask).copy(), dtype=torch.long)
    if values.ndim != 2 or values.min() < 0 or values.max() >= classes:
        raise ValueError('Segmentation mask must be single-channel with documented class IDs.')
    probabilities = torch.nn.functional.one_hot(values, classes).permute(2, 0, 1).float()
    return torch.nn.functional.adaptive_avg_pool2d(probabilities, (grid, grid)).flatten(1).T.contiguous()


class BiomassImages(Dataset):
    def __init__(self, frame, size, train=False, views=4, augmentation="legacy", segmentation_root=None,
                 quarter_turns=True, block_shuffle_grid=0, block_shuffle_variant=0):
        if block_shuffle_grid not in (0, 4, 10):
            raise ValueError('Block shuffle grid must be 0, 4 or 10.')
        if block_shuffle_grid and (train or segmentation_root or views != 2):
            raise ValueError('Block shuffle requires an explicit frozen two-view augmentation bank.')
        if block_shuffle_variant not in range(4):
            raise ValueError('Block shuffle variant must be 0..3.')
        self.block_shuffle_grid = block_shuffle_grid
        self.block_shuffle_variant = block_shuffle_variant
        self.frame = frame.reset_index(drop=True)
        self.segmentation_root = Path(segmentation_root) if segmentation_root else None
        self.grid = size//16
        self.quarter_turns = quarter_turns
        if self.segmentation_root and (not train or augmentation == 'legacy'):
            raise ValueError('Semantic masks require training mode with shared/strong aligned augmentation.')
        self.views = views
        self.augmentation = augmentation
        self.transform = normalize_transform(size, train=train and augmentation == "legacy")
        self.shared_color = T.ColorJitter(.2, .2, .15, .03)
        self.train = train

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.absolute_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            mask = None
            if self.segmentation_root:
                with Image.open(self.segmentation_root/(Path(row.absolute_path).stem+'.png')) as raw_mask:
                    mask = raw_mask.copy()
                if mask.size != image.size:
                    raise ValueError('Image and semantic mask dimensions do not match.')
            if self.train:
                if torch.rand(()) < .5:
                    image = ImageOps.mirror(image)
                    if mask is not None: mask = ImageOps.mirror(mask)
                if torch.rand(()) < .5:
                    image = ImageOps.flip(image)
                    if mask is not None: mask = ImageOps.flip(mask)
                if self.augmentation in {"shared", "strong"}:
                    # Apply one photometric draw to the complete labelled field of view.
                    image = self.shared_color(image)
                if self.augmentation == "strong" and torch.rand(()) < .3:
                    image = T.functional.adjust_gamma(image, float(torch.empty(()).uniform_(.85, 1.15)))
            width, height = image.size
            side = min(width, height)
            # Original four overlapping windows, all kept within the same image/fold.
            if width >= height:
                boxes = [(round(i * (width-side)/(self.views-1)), 0,
                          round(i * (width-side)/(self.views-1))+side, height) for i in range(self.views)]
            else:
                boxes = [(0, round(i * (height-side)/(self.views-1)), width,
                          round(i * (height-side)/(self.views-1))+side) for i in range(self.views)]
            quarter_turn = int(torch.randint(4, ())) if self.train and self.augmentation == "strong" and self.quarter_turns else 0
            processed = []
            for view_index, box in enumerate(boxes):
                tile = image.crop(box)
                if self.block_shuffle_grid:
                    from .block_shuffle import permutation_seed, shuffle_square_tile
                    if width != 2*height:
                        raise ValueError('Shuffle bank requires two nonoverlapping landscape halves.')
                    seed = permutation_seed(str(row.image_sha256), self.block_shuffle_grid,
                                            self.block_shuffle_variant, view_index)
                    tile = shuffle_square_tile(tile, self.block_shuffle_grid, seed)
                processed.append(self.transform(T.functional.rotate(tile, 90*quarter_turn)))
            tiles = torch.stack(processed)
            segmentation = (torch.stack([patch_class_fractions(
                T.functional.rotate(mask.crop(box), 90*quarter_turn), self.grid) for box in boxes])
                if mask is not None else None)
            if self.train and self.augmentation == "strong" and torch.rand(()) < .15:
                sigma = float(torch.empty(()).uniform_(.1, .6))
                tiles = T.functional.gaussian_blur(tiles, [3, 3], [sigma, sigma])
        target = torch.tensor(row[TARGETS].to_numpy(dtype=np.float32))
        if segmentation is not None:
            return tiles, target, index, segmentation
        return tiles, target, index
