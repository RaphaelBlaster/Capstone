"""Offline official I-JEPA grass adaptation and matched biomass regression."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from . import train as common
from .data import ExternalImages, file_sha256
from .ijepa_reference.multiblock import MaskCollator
from .models import configure_fp8
from .official_ijepa import (OFFICIAL_MODEL, OFFICIAL_REVISION, encoder_factory,
    regression_factory, load_official_models, ssl_latents, validate_ssl_masks)

GRASS_CORPUS_SHA256 = '080e5b83117c7a4121e15c135d022f41c9ae62d937a38bbf7a0ec657a1d96ce2'


class StrictOfficialCollator:
    """Use official per-image multi-block masks; reject any overlap on retry."""
    def __init__(self, size=224, patch_size=14):
        if size // patch_size < 8:
            raise ValueError("Official random multi-block masks require at least an 8x8 patch grid.")
        self.size = size
        self.patch_size = patch_size
        self.reference = MaskCollator(input_size=size, patch_size=patch_size,
            enc_mask_scale=(.85, 1.), pred_mask_scale=(.15, .2),
            aspect_ratio=(.75, 1.5), nenc=1, npred=4,
            min_keep=1 if size < 224 else 10, allow_overlap=False)

    def __call__(self, batch):
        for _ in range(64):
            images, contexts, targets = self.reference(batch)
            try:
                validate_ssl_masks(contexts, targets, len(images), (self.size//self.patch_size)**2)
                return images, contexts, targets
            except ValueError:
                continue
        raise ValueError("Could not sample valid official non-overlapping per-image masks.")


def export(encoder, predictor, teacher, path, args, source, step, status):
    common.atomic_save({
        "initialization_kind": "official_ijepa_domain_v1",
        "encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
        **({"target_encoder": teacher.state_dict()} if step else {}),
        "step": step, "official_source_epoch": source["epoch"],
        "official_repository_revision": OFFICIAL_REVISION,
        "official_compact_sha256": args.source_sha256,
        "signature": {"model": OFFICIAL_MODEL, "last_blocks": args.last_blocks,
                      "tiny": args.tiny, "size": args.size,
                      "predictor_width": 32 if args.tiny else 384,
                      "predictor_depth": 2 if args.tiny else 12},
        "domain_adaptation_status": status,
    }, path)


def adapt(args):
    session_started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.tiny:
        if (not torch.cuda.is_available()
                or "rtx pro 6000" not in torch.cuda.get_device_name().lower()
                or torch.cuda.get_device_capability()[0] < 10 or args.precision != "fp8"):
            raise RuntimeError("Actual RTX PRO 6000 Blackwell and FP8 are required before training.")
    common.STOP_REQUESTED = False
    common.seed_all(args.seed)
    if args.tiny:
        torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if file_sha256(args.init) != args.source_sha256:
        raise ValueError("Official offline checkpoint bytes differ from the declared source hash.")
    source = torch.load(args.init, map_location="cpu", weights_only=True, mmap=True)
    if not args.tiny and source.get("official_source", {}).get("repository_revision") != OFFICIAL_REVISION:
        raise ValueError("Official source revision was not preserved in the offline input.")
    dataset = ExternalImages(args.manifest, args.size)
    corpus_hash = hashlib.sha256(json.dumps(sorted(row["sha256"] for row in dataset.records)).encode()).hexdigest()
    if not args.tiny and (len(dataset) != 30663 or corpus_hash != GRASS_CORPUS_SHA256):
        raise ValueError("Expected the audited 30663 unique real GrassClover images.")
    # The official single-view transform is RandomResizedCrop + ImageNet normalization.
    dataset.transform = T.Compose([T.RandomResizedCrop(args.size, scale=(.3, 1.)),
        T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    signature = {"model": OFFICIAL_MODEL, "revision": OFFICIAL_REVISION,
        "initial_sha256": args.source_sha256, "corpus_sha256": corpus_hash,
        "images": len(dataset), "size": args.size, "last_blocks": args.last_blocks,
        "steps": args.steps, "batch_size": args.batch_size, "accumulate": args.accumulate,
        "seed": args.seed, "encoder_lr": args.lr, "predictor_lr": args.predictor_lr,
        "weight_decay": args.weight_decay, "precision": args.precision, "tiny": args.tiny}
    progress_path = output/"progress.json"
    if progress_path.exists():
        completed = json.loads(progress_path.read_text())
        if completed.get("signature") != signature:
            raise ValueError("Official adaptation output belongs to a different configuration.")
        if completed.get("status") == "completed_steps":
            for name, digest in completed["export_sha256"].items():
                if file_sha256(output/name) != digest:
                    raise ValueError("Completed official export no longer matches its recorded bytes.")
            print("Official adaptation already completed; verified exports retained.", flush=True)
            return
    latest = output/"latest.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
    if saved and saved["signature"] != signature:
        raise ValueError("Official adaptation resume signature differs.")
    encoder, predictor, teacher = load_official_models(
        source, last_blocks=args.last_blocks, size=args.size, tiny=args.tiny)
    foundation_path = output/"foundation.pt"
    if foundation_path.exists():
        foundation = torch.load(foundation_path, map_location="cpu", weights_only=True, mmap=True)
        expected = {"model": OFFICIAL_MODEL, "last_blocks": args.last_blocks,
                    "tiny": args.tiny, "size": args.size,
                    "predictor_width": 32 if args.tiny else 384,
                    "predictor_depth": 2 if args.tiny else 12}
        if (foundation.get("step") != 0 or foundation.get("signature") != expected
                or foundation.get("official_compact_sha256") != args.source_sha256
                or foundation.get("official_repository_revision") != OFFICIAL_REVISION):
            raise ValueError("Existing original official export does not match the selected source.")
        del foundation
    else:
        export(encoder, predictor, teacher, output/"foundation.pt", args, source, 0, "original_official_weights")
    if saved:
        common.load_changing_state(encoder, saved["encoder"])
        predictor.load_state_dict(saved["predictor"], strict=True)
        teacher.load_state_dict(saved["teacher"], strict=True)
    system = nn.Module()
    system.encoder, system.predictor, system.teacher = encoder, predictor, teacher
    system.to(device)
    precision = configure_fp8(system, args.precision, device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": args.lr},
        {"params": [p for p in predictor.parameters() if p.requires_grad], "lr": args.predictor_lr}],
        weight_decay=args.weight_decay)
    if [name for name, _ in encoder.named_parameters()] != [name for name, _ in teacher.named_parameters()]:
        raise ValueError("Official student and EMA teacher parameter namespaces differ.")
    step, epoch, cursor, order = 0, 0, 0, None
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, cursor, order = saved["step"], saved["epoch"], saved["next_batch_cursor"], saved["epoch_order"]
        common.restore_rng(saved["rng"])
    sampler = common.EpochBatchSampler()
    collator = StrictOfficialCollator(args.size)
    if saved:
        with collator.reference._itr_counter.get_lock():
            collator.reference._itr_counter.value = saved.get("mask_counter", -1)
    batches = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator,
                         num_workers=args.workers, pin_memory=device.type == "cuda",
                         persistent_workers=args.workers > 0)
    common.write_json(output/"config.json", {
        "signature": signature, "precision": precision, "official_source_epoch": source["epoch"],
        "actual_gpu": torch.cuda.get_device_name() if device.type == "cuda" else "CPU functional check only",
        "encoder_trainable_block_indices": [i for i, block in enumerate(encoder.backbone.blocks)
            if any(p.requires_grad for p in block.parameters())],
        "masking": "Official per-image four-target multi-block collator; strict context exclusion.",
        "resume_limit": "Committed batch order and optimizer/RNG preserved; worker augmentation draws may differ."})
    started = session_started
    last_save = time.monotonic()
    last_loss = saved.get("last_loss") if saved else None
    seen_indices = set(saved.get('seen_image_indices', [])) if saved else set()
    sample_exposures = saved.get('sample_exposures', 0) if saved else 0
    reason = "completed_steps"
    def save(reason):
        common.atomic_save({"signature": signature, "encoder": common.changing_state(encoder),
            "predictor": predictor.state_dict(), "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
            "next_batch_cursor": cursor, "epoch_order": order, "rng": common.rng_state(),
            "reason": reason, "last_loss": last_loss,
            "seen_image_indices": sorted(seen_indices), "sample_exposures": sample_exposures,
            "mask_counter": collator.reference._itr_counter.value}, latest)
    bar = tqdm(total=args.steps, initial=step, desc="Official I-JEPA grass adaptation",
               dynamic_ncols=True, mininterval=2.)
    optimizer.zero_grad(set_to_none=True)
    while step < args.steps:
        if order is None:
            order = torch.randperm(len(dataset)).tolist()
            cursor = 0
        epoch_batches = common.make_epoch_batches(order, args.batch_size)
        sampler.batches = epoch_batches[cursor:]
        sizes = [len(row) for row in epoch_batches]
        pending = []
        pending_indices = []
        encoder.train()
        predictor.train()
        for batch_index, (images, contexts, targets) in enumerate(batches, start=cursor):
            if batch_index % args.accumulate == 0:
                progress = step/max(1, args.steps-1)
                factor = min(1., (step+1)/50) * (.1+.9*.5*(1+math.cos(math.pi*progress)))
                for group, lr in zip(optimizer.param_groups, [args.lr, args.predictor_lr]):
                    group["lr"] = lr*factor
            images = images.to(device, non_blocking=True)
            contexts = [mask.to(device, non_blocking=True) for mask in contexts]
            targets = [mask.to(device, non_blocking=True) for mask in targets]
            with common.amp(device):
                prediction, truth = ssl_latents(encoder, predictor, teacher, images,
                                                contexts, targets, masks_validated=True)
                loss = F.smooth_l1_loss(prediction.float(), truth.float())
            if not torch.isfinite(loss):
                raise ValueError("Official adaptation latent loss is non-finite.")
            window = common.accumulation_window_samples(sizes, batch_index, args.accumulate)
            (loss*(len(images)/window)).backward()
            pending.append(float(loss.detach()))
            pending_indices.extend(epoch_batches[batch_index])
            boundary = (batch_index+1) % args.accumulate == 0 or batch_index+1 == len(epoch_batches)
            if not boundary:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad], 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            momentum = .996+.004*step/max(1, args.steps)
            with torch.no_grad():
                for student, target in zip(encoder.parameters(), teacher.parameters()):
                    target.lerp_(student.detach(), 1.-momentum)
            step += 1
            seen_indices.update(pending_indices)
            sample_exposures += len(pending_indices)
            pending_indices = []
            cursor = batch_index+1
            last_loss = sum(pending)/len(pending)
            pending = []
            bar.update(1)
            bar.set_postfix(loss=f"{last_loss:.5f}", ema=f"{momentum:.6f}")
            if step % 50 == 0 or step == 1:
                common.log(output/"metrics.jsonl", {"step": step, "loss": last_loss,
                    "ema": momentum, "images": len(dataset),
                    "unique_images_trained": len(seen_indices), "sample_exposures": sample_exposures})
            stop = (common.STOP_REQUESTED or time.monotonic()-started >= args.max_hours*3600
                    or (args.stop_after_step and step >= args.stop_after_step))
            if stop or step >= args.steps:
                reason = "completed_steps" if step >= args.steps else "checkpointed_budget_or_interrupt"
                save(reason)
                break
            if time.monotonic()-last_save >= 600:
                save("periodic_committed_step")
                last_save = time.monotonic()
        if step >= args.steps or reason != "completed_steps":
            break
        order = None
        epoch += 1
    bar.close()
    if step < 1:
        raise ValueError("No completed grass-domain optimizer step; no adapted export.")
    export(encoder, predictor, teacher, output/"adapted.pt", args, source, step, reason)
    common.write_json(progress_path, {"status": reason, "step": step,
        "requested_steps": args.steps, "unique_grass_images": len(dataset),
        "unique_images_trained": len(seen_indices), "sample_exposures": sample_exposures,
        "equivalent_corpus_passes": sample_exposures/len(dataset),
        "signature": signature,
        "export_sha256": {name: file_sha256(output/name) for name in ("foundation.pt", "adapted.pt")},
        "seconds_this_session": time.monotonic()-started, "last_loss": last_loss,
        "foundation_export": "foundation.pt", "adapted_export": "adapted.pt"})
    # A completed final export preserves the full adapted student/predictor/teacher.
    # Remove only its now-redundant large optimizer checkpoint before regression.
    if step >= args.steps:
        latest.unlink()


def main():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, common.interrupt_training)
    if len(sys.argv) > 1 and sys.argv[1] == "finetune":
        args = common.parser().parse_args()
        if args.model != OFFICIAL_MODEL or len(set(args.fold or [])) not in (2, 3):
            raise ValueError("Official regression requires its model tag and exactly two or three selected folds.")
        return common.finetune(args, encoder_factory=encoder_factory, predictor_model_factory=regression_factory)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["adapt"])
    parser.add_argument("--init", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--max-hours", type=float, default=2.)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--last-blocks", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--predictor-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.04)
    parser.add_argument("--precision", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--tiny", action="store_true", help="CPU functional tests only.")
    parser.add_argument("--stop-after-step", type=int, help="Controlled interruption for functional validation.")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.accumulate) < 1 or args.max_hours <= 0:
        parser.error("Positive step, batch and time budgets are required.")
    return adapt(args)


if __name__ == "__main__":
    main()
