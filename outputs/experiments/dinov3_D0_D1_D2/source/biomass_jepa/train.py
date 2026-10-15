"""python -m biomass_jepa.train {prepare,pretrain,finetune,report}"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import (TARGETS, BiomassImages, ExternalImages, assign_folds,
                   external_manifest, file_sha256, image_table)
from .metrics import group_bootstrap, scores
from .models import (BiomassModel, DinoMLPBiomassModel, PredictorBiomassModel, Encoder, JEPA, Predictor,
                     block_mask, configure_fp8, initialize_jepa_from_export)

ACTIVE_CHECKPOINT = None
STOP_REQUESTED = False


class CheckpointWriteError(RuntimeError):
    """A failed checkpoint write; the prior committed file remains intact."""


def interrupt_training(signum, frame):
    # Let the active batch/optimizer call finish; checkpoint only at a safe boundary.
    global STOP_REQUESTED
    STOP_REQUESTED = True


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.tiny:
        raise RuntimeError("No CUDA GPU attached. Real training requires the H100/Blackwell host.")
    if device.type == "cpu":
        torch.set_num_threads(2)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception as error:
        # Do not leave a partial multi-GB file or overwrite the last good checkpoint.
        temporary.unlink(missing_ok=True)
        free = shutil.disk_usage(path.parent).free
        raise CheckpointWriteError(f"Checkpoint write failed: {path}; free_bytes={free}; {error}") from error


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False))


def log(path, payload):
    line = json.dumps(payload, allow_nan=False)
    with open(path, "a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def gradient_norms(model):
    """Small, opt-in diagnostic; does not alter gradients or training decisions."""
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = name.split('.')[0]
        squared = parameter.grad.detach().float().square().sum()
        groups[group] = groups.get(group, 0.) + squared
    result = {name: float(value.sqrt()) for name, value in groups.items()}
    for name in ['predictor.mask_token', 'predictor.output_proj.weight']:
        parameter = dict(model.named_parameters()).get(name)
        if parameter is not None and parameter.grad is not None:
            result[name] = float(parameter.grad.detach().float().norm())
    return result


def runtime_metadata(args, device, precision):
    import timm
    import importlib.metadata
    return {"command": vars(args), "torch": str(torch.__version__), "timm": timm.__version__,
            "torchao": importlib.metadata.version("torchao") if args.precision == "fp8" else None,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU functional validation only",
            "precision": precision, "tiny_functional_check": args.tiny,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def rng_state():
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            # Store NumPy's MT19937 words as primitive integers so weights_only=True
            # can restore the checkpoint without importing NumPy pickle globals.
            "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])},
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(payload):
    torch.set_rng_state(payload["torch"].cpu())
    random.setstate(payload["python"])
    if "numpy" in payload:  # Older checkpoints did not capture this RNG.
        state = payload["numpy"]
        np.random.set_state((state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
                             state["position"], state["has_gauss"], state["cached_gaussian"]))
    if torch.cuda.is_available() and payload["cuda"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda"]])


def make_epoch_batches(order, batch_size):
    """Split a saved sample order into the exact batches used by one epoch."""
    return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]


def accumulation_window_samples(batch_sizes, batch_index, accumulate):
    start = (batch_index // accumulate) * accumulate
    return sum(batch_sizes[start:start + accumulate])


def remaining_batches(epoch_batches, next_batch_cursor):
    """Yield only batches whose optimizer updates have not been committed."""
    return enumerate(epoch_batches[next_batch_cursor:], start=next_batch_cursor)


class EpochBatchSampler:
    """A stable, explicitly checkpointable sequence of per-epoch minibatches."""
    def __init__(self):
        self.batches = []

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def load_checkpoint(path):
    # Accept tensor/primitive checkpoints from this runner, not arbitrary pickled model objects.
    return torch.load(path, map_location="cpu", weights_only=True)


def changing_state(module):
    names = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    names.update(name for name, _ in module.named_buffers())
    return {name: value for name, value in module.state_dict().items() if name in names}


def load_changing_state(module, state):
    expected = set(changing_state(module))
    if set(state) != expected:
        raise ValueError("Changing-parameter checkpoint keys do not match this model.")
    module.load_state_dict(state, strict=False)


def compact_completed_checkpoint(fold_dir, expected_images):
    """Remove redundant optimizer state only after the selected model and predictions exist."""
    fold_dir = Path(fold_dir)
    needed = [fold_dir / name for name in ["best.pt", "metrics.json", "predictions.csv"]]
    if any(not path.is_file() or not path.stat().st_size for path in needed):
        return False
    metrics = json.loads((fold_dir / "metrics.json").read_text())
    if metrics.get("status") != "completed_epochs":
        return False
    predictions = pd.read_csv(fold_dir / "predictions.csv")
    if len(predictions) != expected_images or predictions.image_path.duplicated().any():
        return False
    latest = fold_dir / "latest.pt"
    if latest.exists():
        latest.unlink()
        return True
    return False


def fold_is_complete(fold_dir):
    path = Path(fold_dir) / "metrics.json"
    return path.is_file() and json.loads(path.read_text()).get("status") == "completed_epochs"


def make_encoder(args, size, pretrained):
    encoder = Encoder(args.model, size, pretrained=pretrained and not args.backbone_checkpoint,
                      last_blocks=args.last_blocks, tiny=args.tiny)
    if args.backbone_checkpoint and pretrained:
        from timm.models.eva import checkpoint_filter_fn
        from timm.models._helpers import load_state_dict
        state = load_state_dict(args.backbone_checkpoint)
        encoder.backbone.load_state_dict(checkpoint_filter_fn(state, encoder.backbone), strict=True)
    encoder.checkpoint_blocks = args.grad_checkpoint
    return encoder


def loader(dataset, args, shuffle, batch_size=None, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size or args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=args.workers > 0, drop_last=drop_last)


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = image_table(args.csv, args.image_root)
    frame = assign_folds(frame, args.group_column, args.folds, args.seed)
    # Exact duplicates of a labelled image must not create independent validation samples.
    frame["image_sha256"] = [file_sha256(p) for p in frame.absolute_path]
    if frame.image_sha256.duplicated().any():
        raise ValueError("Duplicate labelled image bytes; deduplicate/group before preparing folds.")
    frame.to_csv(output / "folds.csv", index=False)
    manifest = external_manifest(args.external_root, frame) if args.external_root else None
    if manifest is not None:
        write_json(output / "external.json", manifest)
    write_json(output / "data_audit.json", {
        "images": len(frame), "groups": int(frame.group_id.nunique()),
        "group_column": args.group_column,
        "group_limit": "Sampling_Date groups are acquisition-date proxies; not verified true site/plot identifiers.",
        "fold_images": frame.groupby("fold").size().to_dict(),
        "targets": TARGETS, "units": "grams per labelled CSIRO quadrat",
        "target_identity_audit": frame.attrs.get("target_identity_audit", {}),
        "external_unique_images": len(manifest) if manifest is not None else 0,
        "dedupe_scope": "Exact file SHA256, not perceptual/re-encoding deduplication.",
        "ssl_protocol": "External directories only; no CSIRO train/validation/test pixels.",
        "csv_sha256": file_sha256(args.csv)})
    print((output / "data_audit.json").read_text(), flush=True)


def pretrain(args):
    global ACTIVE_CHECKPOINT
    started = time.monotonic()
    seed_all(args.seed)
    device = device_for(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = ExternalImages(args.manifest, args.size, args.ssl_augmentation)
    batches = loader(data, args, shuffle=True, drop_last=True)
    if not len(batches):
        raise ValueError("External corpus smaller than batch size.")
    corpus_hash = hashlib.sha256(json.dumps(sorted(r["sha256"] for r in data.records)).encode()).hexdigest()
    signature = {"model": args.model, "size": args.size, "last_blocks": args.last_blocks,
                 "training_precision": args.precision,
                 "predictor_width": args.predictor_width, "predictor_depth": args.predictor_depth,
                 "tiny": args.tiny, "corpus_sha256": corpus_hash,
                 "batch_size": args.batch_size, "accumulate": args.accumulate, "seed": args.seed,
                 "epochs": args.epochs, "steps": args.steps, "lr": args.lr,
                 "encoder_lr": args.encoder_lr, "ssl_augmentation": args.ssl_augmentation}
    if args.init:
        signature['init_sha256'] = file_sha256(args.init)
    saved = load_checkpoint(args.resume) if args.resume else None
    if saved and saved["signature"] != signature:
        raise ValueError("Pretraining checkpoint architecture/corpus/batch/seed mismatch.")
    encoder = make_encoder(args, args.size, pretrained=saved is None and args.init is None and not args.tiny)
    predictor = Predictor(encoder.dim, encoder.grid, args.predictor_width, args.predictor_depth)
    if args.init:
        initialize_jepa_from_export(encoder, predictor, load_checkpoint(args.init),
                                   args.model, args.last_blocks, args.tiny)
    model = JEPA(encoder, predictor)
    if saved:
        model.load_state_dict(saved["model"])
    model.to(device).train()
    precision = configure_fp8(model, args.precision, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad],
         "lr": args.encoder_lr if args.encoder_lr is not None else args.lr},
        {"params": list(model.predictor.parameters()), "lr": args.lr}], weight_decay=.04)
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
    step = saved["step"] if saved else 0
    previous_seconds = saved.get("total_seconds", 0) if saved else 0
    total_steps = args.steps or math.ceil(args.epochs * len(batches) / args.accumulate)
    metadata = runtime_metadata(args, device, precision)
    metadata.update({"signature": signature, "total_planned_steps": total_steps,
                     "trainable_parameters": sum(p.numel() for p in parameters),
                     "resume_limit": "Optimizer/model/RNG restored; shuffled loader restarts, so sample order is not bit-identical.",
                     "export_encoder": "EMA target encoder"})
    write_json(output / "pretrain_config.json", metadata)
    iterator = iter(batches)
    last_save = time.monotonic()
    stopped = "completed_steps"
    low_std_steps = 0
    deadline = started + max(0, args.max_hours * 3600 - args.reserve_seconds)
    def save(reason):
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "step": step, "signature": signature, "rng": rng_state(),
                   "total_seconds": previous_seconds + time.monotonic()-started,
                   "reason": reason, "metadata": metadata}
        atomic_save(payload, output / "jepa_latest.pt")
        atomic_save({"encoder": model.teacher.state_dict(), "predictor": model.predictor.state_dict(), "signature": signature,
                     "step": step, "metadata": metadata, "reason": reason}, output / "jepa_encoder.pt")
    ACTIVE_CHECKPOINT = save
    progress_bar = tqdm(total=total_steps, initial=step, desc="JEPA optimizer steps", dynamic_ncols=True, mininterval=2.)
    try:
        while step < total_steps:
            if time.monotonic() >= deadline:
                stopped = "session_time_budget"
                break
            optimizer.zero_grad(set_to_none=True)
            progress = step / max(1, total_steps-1)
            warmup = min(1, (step+1)/max(1, int(total_steps*.05)))
            lr = args.lr * warmup * (.1 + .9*.5*(1+math.cos(math.pi*progress)))
            optimizer.param_groups[0]["lr"] = (args.encoder_lr if args.encoder_lr is not None else args.lr) * warmup * (.1 + .9*.5*(1+math.cos(math.pi*progress)))
            optimizer.param_groups[1]["lr"] = lr
            losses, stds = [], []
            for _ in range(args.accumulate):
                try:
                    image = next(iterator)
                except StopIteration:
                    iterator = iter(batches)
                    image = next(iterator)
                image = image.to(device, non_blocking=True)
                visible, targets = block_mask(encoder.grid, device)
                with amp(device):
                    loss, std = model(image, visible, targets)
                if not torch.isfinite(loss) or not torch.isfinite(std):
                    raise FloatingPointError("Non-finite JEPA loss or representation variance.")
                (loss/args.accumulate).backward()
                losses.append(loss.item()); stds.append(std.item())
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            momentum = 1 - (1-.996) * .5*(1+math.cos(math.pi*progress))
            model.update_teacher(momentum)
            step += 1
            progress_bar.update(1)
            std_mean = float(np.mean(stds))
            progress_bar.set_postfix(loss=f"{np.mean(losses):.4f}", std=f"{std_mean:.3f}",
                                     hours_left=f"{max(0,deadline-time.monotonic())/3600:.2f}", refresh=False)
            low_std_steps = low_std_steps+1 if std_mean < 1e-3 else 0
            if low_std_steps >= 50:
                raise RuntimeError("Target representation variance persistently collapsed; checkpoint retained.")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                log(output / "pretrain_log.jsonl", {"step": step, "loss": float(np.mean(losses)),
                    "target_std": std_mean, "ema": momentum, "lr": lr,
                    "encoder_lr": optimizer.param_groups[0]["lr"],
                    "session_seconds": time.monotonic()-started})
            if time.monotonic()-last_save >= args.save_minutes*60:
                save("periodic")
                last_save = time.monotonic()
            if STOP_REQUESTED:
                stopped = "interrupted"
                break
    except KeyboardInterrupt:
        stopped = "interrupted"
    except Exception:
        save("failed")
        raise
    save(stopped)
    progress_bar.close()
    ACTIVE_CHECKPOINT = None
    if step == 0:
        raise RuntimeError("No JEPA training steps completed; increase session budget.")
    print(f"JEPA export: {output/'jepa_encoder.pt'} ({step} steps; {stopped})", flush=True)


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    predictions, targets, indices = [], [], []
    for tiles, target, index in batches:
        with amp(device):
            output = model(tiles.to(device, non_blocking=True))
        predictions.append(output.float().cpu().numpy())
        targets.append(target.numpy())
        indices.append(index.numpy())
    order = np.argsort(np.concatenate(indices))
    return np.concatenate(targets)[order], np.concatenate(predictions)[order]


def fold_signature(frame, args, init_hash):
    columns = ["image_path", "image_sha256", "group_id", "fold"] + TARGETS
    return {"folds_sha256": hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest(),
            "model": args.model, "size": args.size, "last_blocks": args.last_blocks,
            "training_precision": args.precision,
            "seed": args.seed, "epochs": args.epochs, "tiny": args.tiny,
            "init_sha256": init_hash, "batch_size": args.batch_size, "accumulate": args.accumulate,
            "lr": args.lr, "head_lr": args.head_lr, "mixup_alpha": args.mixup_alpha,
            **({"predictor_lr": args.predictor_lr}
               if getattr(args, "predictor_lr", None) is not None else {}),
            "mixup_probability": args.mixup_probability, "mixup_sigma": args.mixup_sigma,
            "huber_delta": args.huber_delta, "weight_decay": args.weight_decay,
            "swa_start": args.swa_start, "views": args.views,
            "augmentation": args.augmentation, "output_space": args.output_space,
            "initialize_median": args.initialize_median, "head": args.head,
            "predictor_width": args.predictor_width, "predictor_depth": args.predictor_depth,
            **({"predictor_readout": args.predictor_readout} if args.predictor_readout != "context" else {}),
            **({"observed_latent_weight": args.observed_latent_weight} if args.observed_latent_weight else {}),
            **({"predictor_init": "fresh"} if args.fresh_predictor else {}),
            **({"regression_hidden": args.regression_hidden} if args.regression_hidden is not None else {}),
            **({"encoder_warmup_epochs": args.encoder_warmup_epochs} if args.encoder_warmup_epochs else {}),
            **({"independent_targets": True} if args.independent_targets else {}),
            **({"independent_log_huber": "weighted_all_five_v1"}
               if args.independent_targets and args.regression_loss == 'log_huber' else {}),
            **({"regression_loss": args.regression_loss,
                "raw_huber_delta": args.raw_huber_delta} if args.regression_loss != 'log_huber' else {}),
            **({"quarter_turns": False} if args.disable_quarter_turns else {}),
            **({"segmentation_classes": 6, "segmentation_weight": args.segmentation_weight,
                "segmentation_sha256": hashlib.sha256(json.dumps([
                    (Path(p).stem, file_sha256(Path(args.segmentation_root)/(Path(p).stem+'.png')))
                    for p in frame.image_path]).encode()).hexdigest()}
               if args.segmentation_root else {}),
            **({"auxiliary_bins": args.auxiliary_bins, "auxiliary_weight": args.auxiliary_weight,
                "predictor_dropout": args.predictor_dropout, "head_dropout": args.head_dropout,
                "raw_loss_weight": args.raw_loss_weight}
               if args.auxiliary_bins or args.predictor_dropout or args.head_dropout != .1 or args.raw_loss_weight else {}),
            **({"fold_epoch_limits": args.fold_epoch_limits} if args.fold_epoch_limits else {})}


def task_objective(prediction, target, delta=5., auxiliary_logits=None, bin_edges=None,
                   auxiliary_weight=0., raw_weight=0., raw_scale=1., regression_loss='log_huber',
                   raw_huber_delta=1., independent_targets=False):
    weights = prediction.new_tensor([.1, .1, .1, .2, .5])
    huber = F.huber_loss(torch.log1p(prediction[:, :3]), torch.log1p(target[:, :3]), delta=delta)
    raw = (((prediction-target).square()*weights).sum(dim=1).mean()/raw_scale**2
           if raw_weight or regression_loss == 'weighted_mse' else prediction.new_zeros(()))
    if regression_loss == 'log_huber':
        if independent_targets:
            all_targets = F.huber_loss(torch.log1p(prediction), torch.log1p(target),
                                      delta=delta, reduction='none')
            regression = (all_targets * weights).sum(1).mean()
        else:
            regression = huber
    elif regression_loss == 'weighted_mse':
        regression = raw
    elif regression_loss == 'weighted_huber':
        robust = F.huber_loss(prediction/raw_scale, target/raw_scale,
                             delta=raw_huber_delta, reduction='none')
        regression = (robust*weights).sum(1).mean()
    else:
        raise ValueError('Unknown regression loss.')
    auxiliary = prediction.new_zeros(())
    if auxiliary_logits is not None:
        labels = torch.stack([torch.bucketize(target[:, i].contiguous(), bin_edges[i]) for i in range(5)], dim=1)
        cross_entropy = F.cross_entropy(auxiliary_logits.float().flatten(0, 1), labels.flatten(), reduction="none")
        auxiliary = (cross_entropy.reshape(-1, 5)*weights).sum(dim=1).mean()
    loss = regression + raw_weight*raw + auxiliary_weight*auxiliary
    return loss, {"huber_log_primitive": float(huber.detach()),
                  "regression_loss": regression_loss, "base_regression": float(regression.detach()),
                  "raw_weighted_mse_normalized": float(raw.detach()), "auxiliary_cross_entropy": float(auxiliary.detach())}


def validate_predictor_learning_rate(args):
    value = getattr(args, "predictor_lr", None)
    if value is not None and (not math.isfinite(value) or value <= 0
            or args.head != "predictor" or not args.init or args.fresh_predictor):
        raise ValueError("A separate predictor LR requires positive finite LR and a pretrained predictor initialization.")


def regression_parameter_groups(model, args):
    """Preserve the legacy grouping unless a pretrained predictor LR is explicit."""
    validate_predictor_learning_rate(args)
    encoder = {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.lr}
    if getattr(args, "predictor_lr", None) is None:
        return [encoder, {"params": [p for name, p in model.named_parameters()
                                     if not name.startswith("encoder.")], "lr": args.head_lr}]
    if not isinstance(getattr(model, "predictor", None), torch.nn.Module):
        raise ValueError("Separate predictor LR requires a predictor module.")
    return [encoder,
            {"params": list(model.predictor.parameters()), "lr": args.predictor_lr},
            {"params": [p for name, p in model.named_parameters()
                        if not name.startswith(("encoder.", "predictor."))], "lr": args.head_lr}]


def set_regression_learning_rates(optimizer, base_lrs, epoch, epochs):
    if len(optimizer.param_groups) != len(base_lrs):
        raise ValueError("Optimizer groups differ from the declared learning-rate schedule.")
    progress = epoch/max(1, epochs-1)
    scale = min(1., (epoch+1)/3) * (.1 + .9*.5*(1+math.cos(math.pi*progress)))
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base*scale


def finetune(args, *, encoder_factory=None, predictor_model_factory=None):
    global ACTIVE_CHECKPOINT
    validate_predictor_learning_rate(args)
    run_started = time.monotonic()
    device = device_for(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.folds_csv, dtype={"group_id": str})
    if args.fold_epoch_limits and len(args.fold_epoch_limits) != int(frame.fold.max()) + 1:
        raise ValueError("Supply one epoch cap for every prepared fold.")
    if args.image_root:
        frame["absolute_path"] = [str((Path(args.image_root)/p).resolve()) for p in frame.image_path]
    if frame.groupby("group_id").fold.nunique().max() != 1:
        raise ValueError("Groups cross fold boundaries.")
    if frame.image_path.duplicated().any() or frame.image_sha256.duplicated().any():
        raise ValueError("Duplicate labelled image in folds table.")
    for path, digest in zip(frame.absolute_path, frame.image_sha256):
        if file_sha256(path) != digest:
            raise ValueError(f"Prepared image changed: {path}")
    init = load_checkpoint(args.init) if args.init else None
    if args.allow_backbone_scope_change and init is None:
        raise ValueError('An explicit initialization is required for a backbone scope change.')
    if init:
        expected = init["signature"]
        if expected["model"] != args.model or expected["tiny"] != args.tiny:
            raise ValueError("JEPA encoder configuration differs from downstream encoder.")
        changed_scope = expected['last_blocks'] != args.last_blocks
        if changed_scope and not args.allow_backbone_scope_change:
            raise ValueError('Encoder training scope differs; an explicit scope-change option is required.')
        if args.allow_backbone_scope_change and not changed_scope:
            raise ValueError('Scope-change option requires a different downstream block count.')
        published_dino = init.get('initialization_kind') == 'published_dino_encoder_v1'
        official_ijepa = init.get('initialization_kind') == 'official_ijepa_domain_v1'
        if official_ijepa:
            from .official_ijepa import OFFICIAL_MODEL, OFFICIAL_REVISION
            if (encoder_factory is None or predictor_model_factory is None
                    or args.model != OFFICIAL_MODEL or args.head != 'predictor'
                    or init.get('official_repository_revision') != OFFICIAL_REVISION
                    or init.get('official_source_epoch', 0) < 1):
                raise ValueError('Official I-JEPA requires its matching encoder and predictor factories and source provenance.')
        if published_dino and (args.head != 'mlp' or init['step'] != 0 or 'predictor' in init):
            raise ValueError('Published DINO initialization is a zero-SSL-step plain MLP control.')
        if not published_dino and not official_ijepa and init["step"] < 1:
            raise ValueError("JEPA checkpoint has no completed training steps.")
    initial_hash = file_sha256(args.init) if args.init else (file_sha256(args.backbone_checkpoint) if args.backbone_checkpoint else "published_dino_weights")
    signature = fold_signature(frame, args, initial_hash)
    if init and init['signature']['last_blocks'] != args.last_blocks:
        signature['initial_encoder_last_blocks'] = init['signature']['last_blocks']
    if init and init.get('initialization_kind') == 'published_dino_encoder_v1':
        signature['encoder_initialization'] = 'published_dino_no_ssl'
    if init and init.get('initialization_kind') == 'official_ijepa_domain_v1':
        signature.update(encoder_initialization='official_ijepa',
                         official_source_epoch=init['official_source_epoch'],
                         official_repository_revision=init['official_repository_revision'],
                         grass_domain_steps=init['step'])
    selected = sorted(frame.fold.unique()) if not args.fold else args.fold
    if any(f not in set(frame.fold) for f in selected):
        raise ValueError("Requested fold is absent from prepared data.")
    for fold_position, fold in enumerate(selected):
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        config_path = fold_dir / "config.json"
        existing = json.loads(config_path.read_text()) if config_path.exists() else None
        if existing and existing["signature"] != signature:
            raise ValueError(f"Existing output belongs to a different run: {fold_dir}")
        if fold_is_complete(fold_dir):
            if args.compact_checkpoints:
                compact_completed_checkpoint(fold_dir, int((frame.fold == fold).sum()))
            print(f"Fold {fold} already finished; preserving output.", flush=True)
            continue
        incomplete_marker = fold_dir / "metrics.json"
        if incomplete_marker.exists():
            # Preserve historical partial diagnostics but never treat them as completion.
            os.replace(incomplete_marker, fold_dir / "incomplete_metrics.json")
        if time.monotonic()-run_started >= max(0, args.max_hours*3600-args.reserve_seconds):
            break
        fold_started = time.monotonic()
        seed_all(args.seed+int(fold))
        train_frame = frame[frame.fold != fold].copy()
        val_frame = frame[frame.fold == fold].copy()
        train_data = BiomassImages(train_frame, args.size, train=True, views=args.views, augmentation=args.augmentation,
                                  segmentation_root=args.segmentation_root, quarter_turns=not args.disable_quarter_turns)
        epoch_sampler = EpochBatchSampler()
        train_batches = DataLoader(train_data, batch_sampler=epoch_sampler,
                                   num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                                   persistent_workers=args.workers > 0)
        val_batches = loader(BiomassImages(val_frame, args.size, views=args.views), args, shuffle=False)
        latest = fold_dir / "latest.pt"
        saved = load_checkpoint(latest) if latest.exists() else None
        if saved and saved["signature"] != signature:
            raise ValueError("Downstream checkpoint signature mismatch.")
        encoder = (encoder_factory or make_encoder)(
            args, args.size, pretrained=init is None and not args.tiny)
        if args.allow_backbone_scope_change and not 0 <= args.last_blocks <= len(encoder.backbone.blocks):
            raise ValueError('Requested backbone scope exceeds the actual transformer depth.')
        if init:
            encoder.load_state_dict(init["encoder"], strict=True)
        if args.head == "predictor":
            if init and "predictor" not in init:
                raise ValueError("Predictor transfer requires a new export containing trained predictor weights.")
            model = (predictor_model_factory or PredictorBiomassModel)(encoder, args.predictor_width, args.predictor_depth,
                        args.output_space, pretrained_predictor=init["predictor"] if init and not args.fresh_predictor else None,
                        auxiliary_bins=args.auxiliary_bins, predictor_dropout=args.predictor_dropout,
                        head_dropout=args.head_dropout, regression_hidden=args.regression_hidden,
                        segmentation_classes=6 if args.segmentation_root else 0,
                        independent_targets=args.independent_targets,
                        predictor_readout=args.predictor_readout,
                        observed_latent_weight=args.observed_latent_weight)
        elif args.head == 'mlp':
            model = DinoMLPBiomassModel(encoder,args.output_space,args.predictor_width,
                        args.auxiliary_bins,args.head_dropout,args.regression_hidden,args.independent_targets)
        else:
            model = BiomassModel(encoder, output_space=args.output_space)
        if args.initialize_median and not saved:
            model.initialize_from_training_targets(train_frame[TARGETS if args.independent_targets else TARGETS[:3]].to_numpy())
        if saved:
            if saved.get("checkpoint_format") == "changing_parameters_v1":
                load_changing_state(model, saved["model"])
            else:
                model.load_state_dict(saved["model"])
        model.to(device)
        precision = configure_fp8(model, args.precision, device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        groups = regression_parameter_groups(model, args)
        base_lrs = [group["lr"] for group in groups]
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        swa = AveragedModel(model)
        epoch_start, best_score = 0, -float("inf")
        next_batch_cursor, epoch_order, saved_loss_values = 0, None, []
        swa_last_epoch = -1
        if saved:
            optimizer.load_state_dict(saved["optimizer"])
            if saved.get("checkpoint_format") == "changing_parameters_v1":
                load_changing_state(swa, saved["swa"])
            else:
                swa.load_state_dict(saved["swa"])
            epoch_start, best_score = saved["next_epoch"], saved["best_score"]
            next_batch_cursor = int(saved.get("next_batch_cursor", 0))
            epoch_order = saved.get("epoch_order")
            saved_loss_values = saved.get("loss_values", [])
            swa_last_epoch = int(saved.get("swa_last_epoch", epoch_start - 1))
            restore_rng(saved["rng"])
        metadata = runtime_metadata(args, device, precision)
        if getattr(args, "predictor_lr", None) is not None:
            metadata["optimizer_base_learning_rates"] = dict(
                encoder=args.lr, predictor=args.predictor_lr, readout=args.head_lr)
        training_targets = torch.tensor(train_frame[TARGETS].to_numpy(), dtype=torch.float32, device=device)
        target_weights = training_targets.new_tensor([.1, .1, .1, .2, .5])
        weighted_mean = (training_targets*target_weights).sum()/len(training_targets)
        raw_scale = (((training_targets-weighted_mean).square()*target_weights).sum()/len(training_targets)).sqrt().clamp_min(1.)
        bin_edges = (torch.quantile(training_targets, torch.arange(1, args.auxiliary_bins, device=device).float()/args.auxiliary_bins,
                                  dim=0).T.contiguous() if args.auxiliary_bins else None)
        metadata.update({"signature": signature, "fold": int(fold),
                         "auxiliary_bin_edges_training_only": bin_edges.tolist() if bin_edges is not None else None,
                         "raw_loss_training_scale_g": float(raw_scale),
                         "train_images": len(train_frame), "validation_images": len(val_frame),
        "trainable_parameters": sum(p.numel() for p in trainable),
                         "resume_behavior": "Mid-epoch checkpoints preserve the epoch sample order and next committed batch. Worker-side random augmentation draws can differ after restart.",
                         "selection_metric": "global_weighted_r2_raw", "validation_use": "Checkpoint selection on this fold; not a nested unbiased model-selection estimate.",
                         "baseline_origin": ("Published DINO initialization" if init is None
                             or init.get("initialization_kind") == "published_dino_encoder_v1"
                             else "JEPA initialization")})
        if init and init.get('initialization_kind') == 'official_ijepa_domain_v1':
            metadata['baseline_origin'] = 'Official I-JEPA encoder and pretrained predictor'
        if args.allow_backbone_scope_change or (init and init.get('initialization_kind') == 'official_ijepa_domain_v1'):
            metadata['encoder_trainable_block_indices'] = [
                i for i, block in enumerate(model.encoder.backbone.blocks)
                if any(parameter.requires_grad for parameter in block.parameters())]
        write_json(config_path, metadata)
        label_std = torch.tensor(np.log1p(train_frame[TARGETS[:3]].to_numpy()).std(axis=0),
                                 dtype=torch.float32, device=device).clamp_min(.01)
        status = "completed_epochs"
        next_epoch = epoch_start
        epochs_this_session = 0
        diagnostic_updates = 0
        def save_latest(reason):
            atomic_save({"model": changing_state(model), "optimizer": optimizer.state_dict(),
                         "swa": changing_state(swa), "next_epoch": next_epoch,
                         "next_batch_cursor": next_batch_cursor, "epoch_order": epoch_order,
                         "loss_values": saved_loss_values,
                         "swa_last_epoch": swa_last_epoch,
                         "best_score": best_score, "signature": signature,
                         "rng": rng_state(), "reason": reason,
                         "checkpoint_format": "changing_parameters_v1"}, latest)
        ACTIVE_CHECKPOINT = save_latest
        last_save = time.monotonic()
        # Keep the original 60-epoch LR schedule while matching completed exposure.
        epoch_limit = args.epochs
        if args.fold_epoch_limits:
            epoch_limit = args.fold_epoch_limits[int(fold)]
        for epoch in range(epoch_start, epoch_limit):
            if epoch != epoch_start or epoch_order is None:
                epoch_order = torch.randperm(len(train_data)).tolist()
                next_batch_cursor = 0
                saved_loss_values = []
            epoch_batches = make_epoch_batches(epoch_order, args.batch_size)
            if not 0 <= next_batch_cursor <= len(epoch_batches):
                raise ValueError("Checkpoint batch cursor is outside its saved epoch order.")
            epoch_sampler.batches = [batch for _, batch in remaining_batches(epoch_batches, next_batch_cursor)]
            batch_sizes = [len(indices) for indices in epoch_batches]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            set_regression_learning_rates(optimizer, base_lrs, epoch, args.epochs)
            loss_values = list(saved_loss_values) if epoch == epoch_start else []
            pending_losses = []
            progress_bar = tqdm(train_batches, total=len(epoch_batches), initial=next_batch_cursor,
                                desc=f"Fold {fold} ({fold_position+1}/{len(selected)}) epoch {epoch+1}/{epoch_limit}",
                                dynamic_ncols=True, mininterval=2.)
            loss_components = {}
            for batch_index, batch in enumerate(progress_bar, start=next_batch_cursor):
                tiles, target, _ = batch[:3]
                segmentation = batch[3].to(device, non_blocking=True) if args.segmentation_root else None
                tiles, target = tiles.to(device, non_blocking=True), target.to(device, non_blocking=True)
                if args.mixup_alpha > 0 and len(tiles) > 1 and np.random.rand() < args.mixup_probability:
                    lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                    labels = torch.log1p(target[:, :3])/label_std
                    affinity = (-torch.cdist(labels, labels).square()/(2*args.mixup_sigma**2))
                    affinity.fill_diagonal_(-float("inf"))
                    order = torch.multinomial(affinity.softmax(dim=1), 1).flatten()
                    # Mix primitive targets in grams; derive sums, then apply log1p in loss.
                    tiles = lam*tiles+(1-lam)*tiles[order]
                    target = lam*target+(1-lam)*target[order]
                    if segmentation is not None:
                        segmentation = lam*segmentation+(1-lam)*segmentation[order]
                with amp(device):
                    if segmentation is not None:
                        prediction, auxiliary_logits, semantic_logits = model(tiles, return_segmentation=True,
                            detach_encoder=epoch < args.encoder_warmup_epochs)
                    elif args.auxiliary_bins:
                        prediction, auxiliary_logits = model(tiles, return_auxiliary=True,
                            detach_encoder=epoch < args.encoder_warmup_epochs)
                    else:
                        prediction = (model(tiles, detach_encoder=epoch < args.encoder_warmup_epochs)
                                      if args.head in {'predictor','mlp'} else model(tiles))
                        auxiliary_logits = None
                    loss, loss_components = task_objective(prediction, target, args.huber_delta,
                        auxiliary_logits, bin_edges, args.auxiliary_weight, args.raw_loss_weight, raw_scale,
                        regression_loss=args.regression_loss, raw_huber_delta=args.raw_huber_delta,
                        independent_targets=args.independent_targets)
                    if segmentation is not None:
                        semantic_loss = -(segmentation*F.log_softmax(semantic_logits.float(), dim=-1)).sum(-1).mean()
                        loss = loss + args.segmentation_weight*semantic_loss
                        loss_components['semantic_soft_cross_entropy'] = float(semantic_loss.detach())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite supervised loss.")
                # Weight each batch mean by its actual sample count, including a short tail.
                window_samples = accumulation_window_samples(batch_sizes, batch_index, args.accumulate)
                (loss * (len(tiles) / window_samples)).backward()
                pending_losses.append(loss.item())
                progress_bar.set_postfix(loss=f"{loss.item():.4f}",
                    hours_left=f"{max(0,args.max_hours-(time.monotonic()-run_started)/3600):.2f}", refresh=False)
                if (batch_index+1) % args.accumulate == 0 or batch_index+1 == len(epoch_batches):
                    if diagnostic_updates < args.log_gradient_steps:
                        log(fold_dir / 'gradient_diagnostic.jsonl', {
                            'event': 'optimizer_gradient_diagnostic', 'fold': int(fold),
                            'epoch': epoch+1, 'update_in_session': diagnostic_updates+1,
                            'gradient_norms_before_clipping': gradient_norms(model)})
                    diagnostic_updates += 1
                    torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Cursor advances only after this optimizer update has completed.
                    next_batch_cursor = batch_index + 1
                    loss_values.extend(pending_losses)
                    pending_losses.clear()
                    saved_loss_values = list(loss_values)
                    if time.monotonic()-last_save >= args.save_minutes*60:
                        next_epoch = epoch
                        save_latest("periodic_mid_epoch")
                        last_save = time.monotonic()
                    if STOP_REQUESTED:
                        status = "interrupted"
                        break
                if STOP_REQUESTED:
                    status = "interrupted"
                    break
                if time.monotonic()-run_started >= max(0, args.max_hours*3600-args.reserve_seconds):
                    status = "session_time_budget"
                    break
                if time.monotonic()-fold_started >= args.fold_hours*3600:
                    status = "fold_time_budget"
                    break
            progress_bar.close()
            if status != "completed_epochs":
                # Any uncommitted accumulation window is discarded and replayed.
                next_epoch = epoch
                save_latest(status)
                break
            if epoch >= args.swa_start and swa_last_epoch < epoch:
                swa.update_parameters(model)
            swa_last_epoch = epoch
            # Persist the full-epoch cursor and SWA marker before validation so a
            # signal during evaluation cannot replay training or average twice.
            next_epoch = epoch
            save_latest("epoch_training_complete")
            evaluator = swa if int(swa.n_averaged) else model
            truth, prediction = predict(evaluator, val_batches, device)
            score = scores(truth, prediction)["global_weighted_r2_raw"]
            if score is None:
                raise ValueError("Validation fold has zero weighted target variance.")
            if STOP_REQUESTED:
                status = "interrupted"
                save_latest(status)
                break
            next_epoch = epoch+1
            next_batch_cursor = 0
            epoch_order = None
            saved_loss_values = []
            epochs_this_session += 1
            if score > best_score:
                best_score = score
                selected_model = evaluator.module if int(swa.n_averaged) else model
                atomic_save({"model": changing_state(selected_model), "signature": signature,
                             "epoch": epoch, "score": score,
                             "checkpoint_format": "changing_parameters_v1"}, fold_dir / "best.pt")
                result = val_frame[["image_path", "group_id", "fold"]+TARGETS].copy()
                for j, name in enumerate(TARGETS):
                    result[f"pred_{name}"] = prediction[:, j]
                result.to_csv(fold_dir / "predictions.csv", index=False)
            log(fold_dir / "train_log.jsonl", {"epoch": epoch+1, "loss": float(np.mean(loss_values)),
                "encoder_frozen": epoch < args.encoder_warmup_epochs,
                "last_batch_loss_components": loss_components,
                "global_weighted_r2_raw": score, "best": best_score,
                "swa_epochs": int(swa.n_averaged), "session_seconds": time.monotonic()-fold_started})
            save_latest("epoch_complete")
            if STOP_REQUESTED:
                status = "interrupted"
                break
        # Time limit interrupted an epoch; session must resume before advancing to next fold.
        if status in {"session_time_budget", "fold_time_budget"}:
            ACTIVE_CHECKPOINT = None
            print(f"{status} in fold {fold}; retained unfinished optimizer state for resume.", flush=True)
            break
        if status == "interrupted":
            print(f"Fold {fold} safely checkpointed after interruption; rerun same command to resume.", flush=True)
            break
        if not (fold_dir / "predictions.csv").exists():
            if status == "fold_time_budget":
                print(f"Fold {fold} has no complete epoch yet; resume with larger --fold-hours.", flush=True)
                break
            raise RuntimeError("No completed downstream epoch/predictions.")
        result = pd.read_csv(fold_dir / "predictions.csv")
        fold_metrics = scores(result[TARGETS].to_numpy(), result[[f"pred_{t}" for t in TARGETS]].to_numpy())
        fold_metrics.update({"status": status, "completed_epochs": next_epoch,
                             "epochs_completed_this_session": epochs_this_session,
                             "seed": args.seed, "tiny_functional_check": args.tiny})
        write_json(fold_dir / "metrics.json", fold_metrics)
        ACTIVE_CHECKPOINT = None
        if args.compact_checkpoints and compact_completed_checkpoint(fold_dir, len(val_frame)):
            print(f"Fold {fold}: retained best model and full predictions; removed completed-fold optimizer checkpoint.", flush=True)
        print(f"Fold {fold} finished: {status}", flush=True)
    report(args)


def report(args):
    output = Path(args.output)
    parts = []
    for path in sorted(output.glob("fold_*/predictions.csv")):
        # Partial session predictions stay available locally but do not count as a finished fold.
        if fold_is_complete(path.parent):
            parts.append(pd.read_csv(path, dtype={"group_id": str}))
    if not parts:
        print("No completed folds to report yet.", flush=True)
        return
    result = pd.concat(parts, ignore_index=True).sort_values("image_path")
    if result.image_path.duplicated().any():
        raise ValueError("OOF predictions duplicate images.")
    result.to_csv(output / "oof_predictions.csv", index=False)
    truth = result[TARGETS].to_numpy()
    prediction = result[[f"pred_{t}" for t in TARGETS]].to_numpy()
    metrics = scores(truth, prediction)
    folds_path = getattr(args, "folds_csv", None)
    expected = len(pd.read_csv(folds_path)) if folds_path else None
    metrics.update({"predicted_images": len(result), "expected_images": expected,
                    "complete_oof": expected == len(result) if expected is not None else None,
                    "folds": sorted(int(f) for f in result.fold.unique()),
                    "group_bootstrap": group_bootstrap(truth, prediction, result.group_id.to_numpy()),
                    "comparison_limit": "Historical capstone scores are reported separately; comparable only after matching scoring, folds and selection protocol."})
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    q = commands.add_parser("prepare")
    q.add_argument("--csv", required=True)
    q.add_argument("--image-root", required=True)
    q.add_argument("--external-root", action="append", default=[])
    q.add_argument("--group-column", default="Sampling_Date")
    q.add_argument("--folds", type=int, default=6)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--output", required=True)
    for name in ["pretrain", "finetune"]:
        q = commands.add_parser(name)
        q.add_argument("--output", required=True)
        q.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
        q.add_argument("--backbone-checkpoint", help="Local timm or original Meta DINOv3 weight file, for offline GPU hosts.")
        q.add_argument("--size", type=int, default=224 if name == "pretrain" else 256)
        q.add_argument("--last-blocks", type=int, default=8)
        q.add_argument("--batch-size", type=int, default=32 if name == "pretrain" else 4)
        q.add_argument("--accumulate", type=int, default=4)
        q.add_argument("--epochs", type=int, default=50 if name == "pretrain" else 40)
        q.add_argument("--max-hours", type=float, default=4.5 if name == "pretrain" else 18)
        q.add_argument("--reserve-seconds", type=float, default=120)
        q.add_argument("--precision", choices=["fp8", "bf16"], default="fp8")
        q.add_argument("--workers", type=int, default=4)
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--grad-checkpoint", action="store_true")
        q.add_argument("--tiny", action="store_true", help="Random tiny model for CPU functional verification; never publication results.")
        q.add_argument("--lr", type=float, default=1e-4 if name == "pretrain" else 5e-6)
        if name == "pretrain":
            q.add_argument("--manifest", required=True)
            q.add_argument("--init", help="Start a new SSL schedule from a completed EMA encoder/predictor export; distinct from exact --resume.")
            q.add_argument("--encoder-lr", type=float, help="Separate pretrained encoder LR; predictor uses --lr.")
            q.add_argument("--ssl-augmentation", choices=["legacy", "crop"], default="legacy")
            q.add_argument("--resume")
            q.add_argument("--steps", type=int)
            q.add_argument("--predictor-width", type=int, default=384)
            q.add_argument("--predictor-depth", type=int, default=6)
            q.add_argument("--log-every", type=int, default=20)
            q.add_argument("--save-minutes", type=float, default=10)
        else:
            q.add_argument("--folds-csv", required=True)
            q.add_argument('--allow-backbone-scope-change', action='store_true',
                           help='Explicitly change the number of downstream trainable blocks from the initialization scope; preserves all initial weights and records both scopes.')
            q.add_argument("--views", type=int, choices=[2, 4], default=4)
            q.add_argument("--augmentation", choices=["legacy", "shared", "strong"], default="legacy")
            q.add_argument("--auxiliary-bins", type=int, default=0)
            q.add_argument("--auxiliary-weight", type=float, default=0.)
            q.add_argument("--raw-loss-weight", type=float, default=0.)
            q.add_argument("--predictor-dropout", type=float, default=0.)
            q.add_argument("--head-dropout", type=float, default=.1)
            q.add_argument("--regression-hidden", type=int, nargs="+",
                           help="Hidden widths of the supervised mass MLP; default is one predictor-width layer. Keeps the pretrained predictor architecture unchanged.")
            q.add_argument("--encoder-warmup-epochs", type=int, default=0,
                           help="Initially train predictor/readouts with encoder detached, then jointly fine-tune. Default preserves previous runs.")
            q.add_argument('--independent-targets', action='store_true',
                           help='Predict all five nonnegative masses directly; default derives GDM/Total from three primitives.')
            q.add_argument("--segmentation-root", help="Released training-only labels/ directory; masks follow the image stem.")
            q.add_argument("--segmentation-weight", type=float, default=.1)
            q.add_argument("--regression-loss", choices=['log_huber','weighted_mse','weighted_huber'], default='log_huber')
            q.add_argument("--raw-huber-delta", type=float, default=1., help="Huber transition measured in training-only global weighted target standard deviations.")
            q.add_argument("--disable-quarter-turns", action='store_true', help="Isolate 90-degree augmentation under the strong policy; other augmentation stays fixed.")
            q.add_argument("--output-space", choices=["grams", "log1p"], default="grams")
            q.add_argument("--initialize-median", action="store_true")
            q.add_argument("--compact-checkpoints", action="store_true", help="Keep selected models/predictions and unfinished optimizer checkpoints; remove redundant completed-fold latest.pt files.")
            q.add_argument("--fold-epoch-limits", type=int, nargs="+", help="Completed epoch cap per fold; --epochs still sets the LR schedule.")
            q.add_argument("--head", choices=["fusion", "predictor", "mlp"], default="fusion")
            q.add_argument("--predictor-readout", choices=["context", "masked_latent"], default="context",
                           help="Historical context-block transfer, or full predictor queries/output projection with two complementary spatial masks. Distinct ablation, not a proven score fix.")
            q.add_argument("--fresh-predictor", action="store_true",
                           help="Matched control: retain the JEPA encoder initialization, initialize predictor weights afresh.")
            q.add_argument('--observed-latent-weight', type=float, default=0.,
                           help='Mix normalized observed patch latents into full masked predictions after prediction; [0,1). Default keeps the prediction-only readout.')
            q.add_argument("--predictor-width", type=int, default=384)
            q.add_argument("--predictor-depth", type=int, default=6)
            q.add_argument("--image-root", help="Relocate CSIRO images when moving checkpoint/data to another host.")
            q.add_argument("--init", help="jepa_encoder.pt; omit for the published-DINO initialization.")
            q.add_argument("--head-lr", type=float, default=3e-4)
            q.add_argument("--predictor-lr", type=float,
                           help="Separate supervised LR for the initialized pretrained predictor; default shares --head-lr. Recorded in the resume signature.")
            q.add_argument("--fold-hours", type=float, default=2.75)
            q.add_argument("--fold", type=int, action="append")
            q.add_argument("--mixup-alpha", type=float, default=.3)
            q.add_argument("--mixup-probability", type=float, default=.4)
            q.add_argument("--mixup-sigma", type=float, default=1.)
            q.add_argument("--huber-delta", type=float, default=5.)
            q.add_argument("--weight-decay", type=float, default=.02)
            q.add_argument("--save-minutes", type=float, default=10)
            q.add_argument("--log-gradient-steps", type=int, default=0,
                           help="Log component gradient norms for the first N supervised updates; no effect on optimization.")
            q.add_argument("--swa-start", type=int, default=20)
    q = commands.add_parser("report")
    q.add_argument("--output", required=True)
    q.add_argument("--folds-csv")
    return p


def main():
    args = parser().parse_args()
    global STOP_REQUESTED
    STOP_REQUESTED = False
    for key in ["batch_size", "accumulate", "epochs", "folds", "size", "predictor_depth", "predictor_width"]:
        if hasattr(args, key) and getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if hasattr(args, "size") and args.size % 16:
        raise ValueError("Image size must be divisible by the 16-pixel patch size.")
    if hasattr(args, "mixup_probability") and not 0 <= args.mixup_probability <= 1:
        raise ValueError("mixup_probability must be in [0,1].")
    if hasattr(args, "mixup_sigma") and args.mixup_sigma <= 0:
        raise ValueError("mixup_sigma must be positive.")
    if args.command == "finetune":
        if args.log_gradient_steps < 0:
            raise ValueError('log-gradient-steps must be nonnegative.')
        if args.fresh_predictor and (args.head != "predictor" or not args.init):
            raise ValueError("Fresh-predictor control requires --head predictor and --init for the common JEPA encoder.")
        if args.predictor_readout != "context" and args.head != "predictor":
            raise ValueError("Masked-latent readout requires --head predictor.")
        if (not 0 <= args.observed_latent_weight < 1
                or args.observed_latent_weight and (args.head != 'predictor' or args.predictor_readout != 'masked_latent')):
            raise ValueError('Observed-latent weight requires --head predictor --predictor-readout masked_latent and must be in [0,1).')
        if args.independent_targets and args.head not in {'predictor','mlp'}:
            raise ValueError('Independent targets require the predictor or plain MLP head.')
        if args.raw_huber_delta <= 0 or (args.regression_loss != 'log_huber' and args.raw_loss_weight):
            raise ValueError('Raw Huber delta must be positive; raw-only objectives must not add the legacy raw MSE term.')
        if args.segmentation_weight < 0 or (args.segmentation_root and args.segmentation_weight == 0):
            raise ValueError('Semantic supervision requires a positive weight.')
        if args.segmentation_root and (args.head != 'predictor' or args.augmentation == 'legacy'):
            raise ValueError('Semantic supervision requires predictor head and aligned shared/strong augmentation.')
        minimum_epochs = min(args.fold_epoch_limits) if args.fold_epoch_limits else args.epochs
        if args.encoder_warmup_epochs < 0 or args.encoder_warmup_epochs >= minimum_epochs:
            raise ValueError("Encoder warm-up must leave at least one joint-training epoch in every fold.")
        if args.encoder_warmup_epochs and args.head not in {'predictor','mlp'}:
            raise ValueError("Encoder warm-up requires the predictor or plain MLP head.")
        if args.regression_hidden is not None and (args.head not in {'predictor','mlp'} or any(n < 1 for n in args.regression_hidden)):
            raise ValueError("Regression hidden widths must be positive and require a predictor or MLP head.")
        if args.auxiliary_bins not in {0} and args.auxiliary_bins < 2:
            raise ValueError("auxiliary_bins must be zero or at least two.")
        if args.auxiliary_bins and args.head not in {'predictor','mlp'}:
            raise ValueError("Auxiliary heads require the predictor or plain MLP regressor.")
        if args.auxiliary_weight < 0 or args.raw_loss_weight < 0:
            raise ValueError("Loss weights must be nonnegative.")
        if args.auxiliary_weight and not args.auxiliary_bins:
            raise ValueError("An auxiliary loss requires configured auxiliary heads.")
        if not 0 <= args.predictor_dropout < 1 or not 0 <= args.head_dropout < 1:
            raise ValueError("Dropout must be in [0,1).")
    if getattr(args, "fold_epoch_limits", None):
        if any(not 0 < n <= args.epochs for n in args.fold_epoch_limits):
            raise ValueError("Supply one positive epoch cap per fold, no greater than --epochs.")
    signal.signal(signal.SIGTERM, interrupt_training)
    signal.signal(signal.SIGINT, interrupt_training)
    try:
        globals()[args.command](args)
    except BaseException as error:
        if ACTIVE_CHECKPOINT is not None and not isinstance(error, CheckpointWriteError):
            try:
                ACTIVE_CHECKPOINT("interrupted_or_failed")
            except Exception as checkpoint_error:
                print(f"Emergency checkpoint failed; preserving original error: {checkpoint_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
