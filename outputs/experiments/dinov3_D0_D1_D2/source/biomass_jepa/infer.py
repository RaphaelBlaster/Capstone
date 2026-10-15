"""Offline inference from retained predictor-transfer weights; no training."""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import TARGETS, BiomassImages, file_sha256
from .models import Encoder, DinoMLPBiomassModel, PredictorBiomassModel
from .reconcile import reconcile
from .train import load_checkpoint, load_changing_state


def prediction_postprocessor(manifest):
    kind = manifest.get('postprocessing')
    if kind is None:
        return None
    if (kind != 'weighted_sums_nonnegative_v1'
            or manifest.get('package_format') != 'predictor_transfer_v2'
            or manifest.get('signature', {}).get('independent_targets') is not True):
        raise ValueError('Unknown or incompatible inference postprocessing.')
    return reconcile


def submission_table(test, image_paths, predictions):
    required = {"sample_id", "image_path", "target_name"}
    if not required.issubset(test) or test.sample_id.duplicated().any() or test.sample_id.isna().any():
        raise ValueError("Invalid competition test schema or duplicate sample IDs.")
    predictions = np.asarray(predictions)
    if predictions.shape != (len(image_paths), len(TARGETS)):
        raise ValueError("Prediction dimensions do not match images and targets.")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Predictions must be finite nonnegative grams.")
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("Duplicate inferred image paths.")
    wide = pd.DataFrame(predictions, columns=TARGETS)
    wide.insert(0, "image_path", image_paths)
    long = wide.melt(id_vars="image_path", var_name="target_name", value_name="target")
    result = test[["sample_id", "image_path", "target_name"]].merge(long, on=["image_path", "target_name"], how="left", validate="many_to_one", sort=False)
    if len(result) != len(test) or result.target.isna().any():
        raise ValueError("Missing target predictions; never fill unmatched test rows with zero.")
    return result[["sample_id", "target"]]


def reconstruct_model(initial, selected):
    """Identical retained-weight reconstruction for inference and precision diagnostics."""
    sig = selected['signature']
    encoder = Encoder(sig['model'], sig['size'], pretrained=False,
                      last_blocks=sig['last_blocks'], tiny=sig['tiny'])
    encoder.load_state_dict(initial['encoder'], strict=True)
    if sig.get('head') == 'mlp':
        if (initial.get('initialization_kind') != 'published_dino_encoder_v1'
                or initial.get('step') != 0 or 'predictor' in initial
                or sig.get('encoder_initialization') != 'published_dino_no_ssl'):
            raise ValueError('The plain DINO control must use its original zero-SSL-step export.')
        model = DinoMLPBiomassModel(encoder,sig['output_space'],sig['predictor_width'],
            sig.get('auxiliary_bins',0),sig.get('head_dropout',.2),
            sig.get('regression_hidden'),sig.get('independent_targets',False))
        load_changing_state(model,selected['model'])
        return model
    model = PredictorBiomassModel(encoder, sig['predictor_width'], sig['predictor_depth'],
        sig['output_space'], pretrained_predictor=(None if sig.get('predictor_init') == 'fresh' else initial['predictor']),
        auxiliary_bins=sig.get('auxiliary_bins', 0), predictor_dropout=sig.get('predictor_dropout', 0.),
        head_dropout=sig.get('head_dropout', .1), regression_hidden=sig.get('regression_hidden'),
        segmentation_classes=sig.get('segmentation_classes', 0),
        independent_targets=sig.get('independent_targets', False),
        predictor_readout=sig.get('predictor_readout', 'context'),
        observed_latent_weight=sig.get('observed_latent_weight', 0.))
    load_changing_state(model, selected['model'])
    return model


@torch.inference_mode()
def infer(weights, test_csv, image_root, output, batch_size=2, workers=2, require_cuda=False, precision="fp32", fold=None):
    started = time.monotonic()
    weights = Path(weights)
    manifest = json.loads((weights / "inference_manifest.json").read_text())
    source_version = manifest.get("source_version")
    package_format = manifest.get("package_format")
    legacy = package_format is None and source_version in {6, 8, 9}
    current = (package_format == "predictor_transfer_v2" and isinstance(source_version, int)
               and not isinstance(source_version, bool) and source_version >= 10)
    if (manifest.get("source_kernel") != "meghmayur1186/capstone"
            or not (legacy or current)
            or not manifest.get("folds")
            or sorted(set(manifest["folds"])) != manifest["folds"]
            or any(f not in range(6) for f in manifest["folds"])):
        raise ValueError("Unsupported or invalid retained CAPSTONE weight provenance.")
    postprocessor = prediction_postprocessor(manifest)
    required_weights = {"jepa_encoder.pt", *(f"fold_{f}_best.pt" for f in manifest["folds"])}
    if not required_weights.issubset(manifest.get("sha256", {})):
        raise ValueError("Manifest must hash every retained initialization and selected model.")
    signature = manifest.get('signature', {})
    if (signature.get("predictor_readout") == "masked_latent" or signature.get('predictor_init') == 'fresh') and not current:
        raise ValueError("New predictor-interface packages require current provenance; never relabel them as historical weights.")
    if manifest["source_version"] == 6 and manifest["folds"] != list(range(6)):
        raise ValueError("The archived v6 package must retain all six models.")
    if manifest["source_version"] == 9 and manifest["folds"] != [0, 1, 2]:
        raise ValueError("The completed v9 selected head package must retain its three comparison folds.")
    if fold is not None and fold not in manifest["folds"]:
        raise ValueError("Requested fold is absent from the saved models.")
    selected_folds = manifest["folds"] if fold is None else [fold]
    for relative, digest in manifest["sha256"].items():
        if file_sha256(weights / relative) != digest:
            raise ValueError(f"Weight artifact hash mismatch: {relative}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Inference GPU missing; stop instead of running the hidden set on CPU.")
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("FP16 inference requires CUDA; CPU software checks must request FP32.")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    print("INFERENCE DEVICE:", torch.cuda.get_device_name() if device.type == "cuda" else "CPU functional check", flush=True)
    print("Inference uses FP32 saved parameters with", precision, "forward passes; no optimizer, gradients or internet.", flush=True)
    test = pd.read_csv(test_csv)
    paths = test.image_path.drop_duplicates().tolist()
    frame = pd.DataFrame({"image_path": paths, "absolute_path": [str(Path(image_root) / p) for p in paths]})
    for target in TARGETS:
        frame[target] = 0.  # ignored placeholders for the identical validation image loader
    sig = manifest["signature"]
    batches = DataLoader(BiomassImages(frame, sig["size"], views=sig["views"], train=False),
                         batch_size=batch_size, num_workers=workers, shuffle=False,
                         pin_memory=device.type == "cuda")
    initial = load_checkpoint(weights / "jepa_encoder.pt")
    initial_hash = manifest["sha256"]["jepa_encoder.pt"]
    total = np.zeros((len(paths), len(TARGETS)), dtype=np.float64)
    fold_times = []
    for fold in selected_folds:
        fold_started = time.monotonic()
        selected = load_checkpoint(weights / f"fold_{fold}_best.pt")
        if selected["signature"] != sig or sig["init_sha256"] != initial_hash:
            raise ValueError("Selected model signature or JEPA initialization hash mismatch.")
        model = reconstruct_model(initial, selected)
        model.to(device).eval()
        for tiles, _, indices in tqdm(batches, desc=f"Inference saved fold {fold}", mininterval=2):
            with torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else contextlib.nullcontext():
                prediction = model(tiles.to(device, non_blocking=True)).float().cpu().numpy()
            total[indices.numpy()] += prediction
        del model, selected
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fold_times.append(time.monotonic() - fold_started)
        print(json.dumps({"fold": fold, "inference_seconds": fold_times[-1], "images": len(paths)}), flush=True)
    predictions = total / len(selected_folds)
    if postprocessor is not None:
        predictions = postprocessor(predictions)
    result = submission_table(test, paths, predictions)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    record = {"source_version": manifest["source_version"], "folds": selected_folds, "images": len(paths), "rows": len(result),
              "seconds": time.monotonic() - started, "fold_seconds": fold_times,
              "ensemble": "single selected fold" if len(selected_folds) == 1 else "equal arithmetic mean in grams", "tta": False, "precision": precision,
              "test_labels_accessed": False, "submission_sha256": file_sha256(output)}
    if postprocessor is not None:
        record['postprocessing'] = manifest['postprocessing']
    output.with_suffix(".inference.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--fold", type=int, choices=range(6), help="Use one selected saved fold for a fast single-model submission.")
    args = p.parse_args()
    infer(**vars(args))


if __name__ == "__main__":
    main()
