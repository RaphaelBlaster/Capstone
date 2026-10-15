"""Package the completed F2 checkpoints and build offline one-image T4 inference."""
import ast
import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path


RUN = Path("outputs/dino_resume_rca/full_launch/results/dino_finetune_full")
BASE = Path("outputs/final_dual_study/offline")
WEIGHT_DATASET = "meghmayur1186/img2biomass-dino-f2-weights-2026-10-10"
KERNEL = "meghmayur1186/img2biomass-dino-f2-t4-diagnostic-v3"
FULL_INFERENCE_KERNEL = "meghmayur1186/img2biomass-dino-f2-full-inference-v4"
MODEL_SHA = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _notebook_source(payload, manifest_hash, *, one_image_smoke=True):
    payload64 = base64.b64encode(payload).decode("ascii")
    image_limit_guard = (
        'if len(image_paths) != 1:\n'
        '    raise ValueError(f"Diagnostic requires exactly one visible test image, got {len(image_paths)}")\n'
        if one_image_smoke else ""
    )
    source = f'''import time
from datetime import datetime, timezone
CELL_STARTED = time.perf_counter()
print("CELL_START_UTC:", datetime.now(timezone.utc).isoformat(), flush=True)

def stage_start(name):
    print(f"STAGE_START {{name}} utc={{datetime.now(timezone.utc).isoformat()}}", flush=True)
    return time.perf_counter()

def stage_done(name, started):
    print(f"STAGE_DONE {{name}} seconds={{time.perf_counter()-started:.3f}}", flush=True)

import base64, contextlib, hashlib, io, json, os, sys, zipfile
from pathlib import Path
_t = stage_start('imports')
import numpy as np
import pandas as pd
import torch
import subprocess
from tqdm.auto import tqdm
stage_done('imports', _t)

assert torch.cuda.is_available(), "T4 CUDA is required; refusing CPU fallback."
assert "T4" in torch.cuda.get_device_name().upper(), torch.cuda.get_device_name()
print("INFERENCE GPU:", torch.cuda.get_device_name(), flush=True)
WORK = Path('/kaggle/working')
CODE = WORK/'dino_inference_code'
CODE.mkdir(exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode('{payload64}'))).extractall(CODE)
sys.path.insert(0, str(CODE))

_t = stage_start('single_input_index_scan')
INPUT_INDEX = {{}}
for directory, dirs, names in os.walk('/kaggle/input', followlinks=False):
    dirs[:] = sorted(d for d in dirs if not os.path.islink(os.path.join(directory, d)))
    for name in names:
        INPUT_INDEX.setdefault(name, []).append(Path(directory)/name)
stage_done('single_input_index_scan', _t)

def unique_file(name):
    found = INPUT_INDEX.get(name, [])
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {{name}}, found {{len(found)}}")
    return found[0]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

timm_wheels = [path for directory in INPUT_INDEX.values() for path in directory
               if path.name.startswith('timm-1.0.30-') and path.name.endswith('.whl')]
if len(timm_wheels) != 1:
    raise ValueError("Expected one pinned offline timm wheel")
_t = stage_start('offline_dependency_install')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
    '--find-links', str(timm_wheels[0].parent), 'timm==1.0.30',
    'huggingface-hub==0.36.2', 'safetensors', 'tqdm', 'packaging'], check=True)
stage_done('offline_dependency_install', _t)
_t = stage_start('source_imports')
from biomass_jepa.dinov3_dense import DINODensityHead, dense_grid, load_dinov3, patch_tokens
from biomass_jepa.within_image_jigsaw import native_views, preprocess
stage_done('source_imports', _t)

_t = stage_start('find_and_verify_weight_manifest')
manifest_path = unique_file('inference_manifest.json')
WEIGHTS = manifest_path.parent
manifest = json.loads(manifest_path.read_text())
assert digest(manifest_path) == '{manifest_hash}', "Weight manifest changed"
assert manifest['arm'] == 'F2' and manifest['folds'] == [0, 1]
assert manifest['training'] is False and manifest['one_image_smoke'] is True
for name, expected in manifest['sha256'].items():
    assert digest(WEIGHTS/name) == expected, f"Checkpoint hash mismatch: {{name}}"
print("SAVED MODEL:", manifest['arm'], manifest['folds'], flush=True)
stage_done('find_and_verify_weight_manifest', _t)

repo = CODE/'dinov3_reference'
metadata = json.loads((CODE/'dinov3_metadata.json').read_text())
_t = stage_start('find_and_load_verified_dinov3')
checkpoints = INPUT_INDEX.get('dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', [])
if len(checkpoints) != 1:
    raise ValueError(f"Expected one official DINOv3 base checkpoint; got {{len(checkpoints)}}")
base_checkpoint = checkpoints[0]
model, model_provenance = load_dinov3(repo, base_checkpoint, metadata, device='cuda')
assert model_provenance['checkpoint_sha256'] == '{MODEL_SHA}', "Official DINOv3 base hash mismatch"
model.eval()
model.requires_grad_(False)
for block in model.blocks[-2:]:
    block.requires_grad_(True)
model.norm.requires_grad_(True)
for module in model.modules():
    if hasattr(module, 'bias_mask'):
        if (module.bias is None or torch.count_nonzero(module.bias_mask)
                or torch.count_nonzero(module.bias.detach())):
            raise ValueError("Unexpected DINO masked attention bias state")
        module.bias.requires_grad_(False)
expected_trainable = {{n for n,p in model.named_parameters() if p.requires_grad}}
stage_done('find_and_load_verified_dinov3', _t)

_t = stage_start('read_competition_test_csv')
test_files = []
for csv_path in INPUT_INDEX.get('test.csv', []):
    frame = pd.read_csv(csv_path, nrows=0)
    if {{'sample_id', 'image_path', 'target_name'}}.issubset(frame.columns):
        test_files.append(csv_path)
if len(test_files) != 1:
    raise ValueError(f"Expected the CSIRO competition test.csv, found {{len(test_files)}}")
TEST_CSV = test_files[0]
test = pd.read_csv(TEST_CSV)
if test.sample_id.isna().any() or test.sample_id.duplicated().any():
    raise ValueError("Competition sample IDs are invalid")
image_paths = test.image_path.drop_duplicates().tolist()
{image_limit_guard}if not image_paths:
    raise ValueError("The competition test CSV contains no images")
print("VISIBLE TEST PHOTOS:", len(image_paths), flush=True)
stage_done('read_competition_test_csv', _t)

def predict_fold(fold, views):
    _fold_started = time.perf_counter()
    checkpoint_path = WEIGHTS/f'fold_{{fold}}_F2_selected.pt'
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (ck['signature']['fold'] != fold or ck['signature']['arm'] != 'F2'
            or ck['completed_epochs'] != 20 or ck['selection'] != 'fixed_final_epoch'):
        raise ValueError("Selected F2 checkpoint metadata differs")
    delta = ck['backbone_delta']
    if set(delta) != expected_trainable:
        raise ValueError("F2 trainable transformer scope differs")
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        for name, value in delta.items():
            if parameters[name].shape != value.shape:
                raise ValueError("F2 checkpoint tensor shape differs")
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))
    head = DINODensityHead(ck['head']['scale'])
    head.load_state_dict(ck['head'], strict=True)
    head = head.cuda().eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        raw = model.forward_features(views.flatten(0,1).cuda(non_blocking=True))
        patches = patch_tokens(raw, 2)
        dense = dense_grid(patches, 1)
        prediction = head(dense).float().cpu().numpy()[0]
    torch.cuda.synchronize()
    fold_seconds = time.perf_counter()-_fold_started
    if {one_image_smoke!r}:
        print(f"FOLD_{{fold}}_SECONDS: {{fold_seconds:.3f}}", flush=True)
    del head, ck
    torch.cuda.empty_cache()
    if prediction.shape != (5,) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Model emitted invalid biomass predictions")
    return prediction, fold_seconds

_inference_started = time.perf_counter()
target_names = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
fold_seconds_total = {{fold: 0.0 for fold in manifest['folds']}}
prediction_rows = []
for image_path in tqdm(image_paths, desc='DINOv3 F2 inference', unit='image'):
    image_file = TEST_CSV.parent/image_path
    views = torch.stack([preprocess(tile, 512) for tile in native_views(image_file)]).unsqueeze(0)
    fold_predictions = []
    for fold in manifest['folds']:
        fold_prediction, elapsed = predict_fold(fold, views)
        fold_predictions.append(fold_prediction)
        fold_seconds_total[fold] += elapsed
    predictions = np.stack(fold_predictions).mean(0)
    prediction_rows.append([image_path, *predictions.tolist()])
wide = pd.DataFrame(prediction_rows, columns=['image_path', *target_names])
print("INFERENCE_SECONDS_BY_FOLD:", json.dumps(fold_seconds_total), flush=True)
print("INFERENCE_SECONDS_TOTAL:", round(time.perf_counter()-_inference_started, 3), flush=True)
long = wide.melt(id_vars='image_path', var_name='target_name', value_name='target')
submission = test[['sample_id', 'image_path', 'target_name']].merge(
    long, on=['image_path', 'target_name'], how='left', validate='many_to_one', sort=False)
if len(submission) != len(test) or submission.target.isna().any():
    raise ValueError("Predictions did not map to every requested target row")
result = submission[['sample_id', 'target']]
if not np.isfinite(result.target).all() or (result.target < 0).any():
    raise ValueError("Submission contains invalid values")
OUT = WORK/'submission.csv'
result.to_csv(OUT, index=False)
if len(image_paths) == 1:
    print("ONE-IMAGE INFERENCE OUTPUT (grams):", flush=True)
    print(result.to_string(index=False), flush=True)
else:
    print("FULL-IMAGE INFERENCE ROWS:", len(result), "photos:", len(image_paths), flush=True)
evidence = {{
    'source_kernel': manifest['source_kernel'], 'source_version': manifest['source_version'],
    'arm': manifest['arm'], 'folds': manifest['folds'], 'images': len(image_paths),
    'rows': len(result), 'precision': 'fp16', 'training': False, 'test_labels_accessed': False,
    'actual_gpu': torch.cuda.get_device_name(), 'base_dinov3_sha256': model_provenance['checkpoint_sha256'],
    'weight_manifest_sha256': digest(manifest_path), 'submission_sha256': digest(OUT),
    'scope': 'one_image_diagnostic' if {one_image_smoke!r} else 'all_test_images',
    'timing_seconds_total_cell': time.perf_counter()-CELL_STARTED
}}
(WORK/'submission.inference.json').write_text(json.dumps(evidence, indent=2))
print("INFERENCE EVIDENCE:", json.dumps(evidence), flush=True)
print("CELL_DONE_SECONDS:", round(time.perf_counter()-CELL_STARTED, 3), flush=True)
'''
    ast.parse(source)
    return source


def build_package(root, destination, *, one_image_smoke=True):
    root, destination = Path(root), Path(destination)
    source_run = root / RUN
    if json.loads((source_run / "STATUS.json").read_text())["status"] not in {
            "completed_pending_independent_P1", "COMPLETE"}:
        raise ValueError("Completed F1/F2 run is required")
    results = json.loads((source_run / "results.json").read_text())
    if results.get("images") != 105 or results.get("reused_development_only") is not True:
        raise ValueError("Unexpected or unverified development result provenance")
    result_f2 = next((row for row in results["models"] if row.get("arm") == "F2"), None)
    if result_f2 is None:
        raise ValueError("F2 development metric is missing")
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = []
    for fold in (0, 1):
        name = f"fold_{fold}_F2_selected.pt"
        expected_names.append(name)
        source = source_run / f"fold_{fold}/F2/selected.pt"
        expected_hash = json.loads((source_run / "FILES_SHA256.json").read_text())[
            f"fold_{fold}/F2/selected.pt"]
        if sha256(source) != expected_hash:
            raise ValueError(f"checkpoint hash mismatch for source {name}")
        target = destination / name
        if target.exists():
            if sha256(target) != expected_hash:
                raise ValueError(f"checkpoint hash mismatch in existing package: {name}")
        else:
            shutil.copy2(source, target)

    metadata_source = root / BASE / "dinov3_metadata.json"
    (destination / "dinov3_metadata.json").write_bytes(metadata_source.read_bytes())
    hashes = {name: sha256(destination / name) for name in expected_names}
    manifest = {
        "format": "img2biomass_dinov3_f2_inference_v1",
        "arm": "F2", "folds": [0, 1], "training": False,
        "one_image_smoke": True,
        "source_kernel": "meghmayur1186/img2biomass-dino-finetune-f1-f2",
        "source_version": 1,
        "training_notebook_sha256": "30885af0d676666adc2a93eebe0fd9af2b1ba79e66c432a3fcdbfc3061957005",
        "development_results": {
            "global_weighted_r2": result_f2["global_weighted_r2_raw"],
            "images": 105, "folds": [0, 1], "seed": 42,
            "scope": "reused development folds; not full OOF or competition-scored"},
        "dino_source": "https://github.com/facebookresearch/dinov3",
        "dino_revision": json.loads(metadata_source.read_text())["revision"],
        "dino_base_checkpoint_sha256": MODEL_SHA,
        "sha256": hashes,
    }
    manifest_path = destination / "inference_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing inference manifest differs; preserve it and inspect")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    dataset_meta = {"id": WEIGHT_DATASET,
                    "title": "Img2BioMass DINOv3 F2 selected inference weights",
                    "licenses": [{"name": "CC-BY-NC-4.0"}]}
    (destination / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2))

    files = io.BytesIO()
    repo = root / BASE / "dinov3_reference"
    with zipfile.ZipFile(files, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("dinov3_dense.py", "within_image_jigsaw.py"):
            path = root / "biomass_jepa" / name
            archive.write(path, str(Path("biomass_jepa") / name))
        for path in sorted(repo.rglob("*")):
            if path.is_file() and (path.suffix == ".py" or path.name.upper().startswith("LICENSE")):
                archive.write(path, str(Path("dinov3_reference") / path.relative_to(repo)))
        archive.write(metadata_source, "dinov3_metadata.json")
    manifest_hash = sha256(manifest_path)
    source = _notebook_source(files.getvalue(), manifest_hash, one_image_smoke=one_image_smoke)
    notebook_name = ("dino_f2_one_image_t4_inference.ipynb" if one_image_smoke
                     else "dino_f2_all_image_t4_inference.ipynb")
    kernel = KERNEL if one_image_smoke else FULL_INFERENCE_KERNEL
    title = ("Img2BioMass DINO F2 T4 Diagnostic V3" if one_image_smoke
             else "Img2BioMass DINO F2 Full Inference V4")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                ("# Img2BioMass DINOv3 F2 inference-only T4 diagnostic smoke test\n" if one_image_smoke
                 else "# Img2BioMass DINOv3 F2 all-image competition inference\n"),
                ("Loads the two saved F2 fold checkpoints and runs one visible CSIRO test image.\n" if one_image_smoke
                 else "Loads the two saved F2 fold checkpoints and predicts every distinct test image.\n"),
                "No training, optimizer, internet, test labels, or test-time adaptation.\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": source.splitlines(keepends=True)}
        ]}
    notebook_path = destination.parent / notebook_name
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=1))
    kernel_metadata = {
        "id": kernel, "title": title,
        "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [WEIGHT_DATASET, "meghmayur1186/img2biomass-offline-deps",
                             "robiulislamistaken/dinov3-vitb16"],
        "competition_sources": ["csiro-biomass"], "kernel_sources": [], "model_sources": []}
    metadata_path = notebook_path.parent / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(kernel_metadata, indent=2))
    return destination, notebook_path, kernel_metadata


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/dino_f2_competition_inference/weights"
    package, notebook, metadata = build_package(root, out)
    print(json.dumps({"weights": str(package), "notebook": str(notebook),
                      "kernel": metadata["id"], "notebook_sha256": sha256(notebook)}, indent=2))
