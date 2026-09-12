"""Compute CLIP score, LPIPS, and Object Accuracy for each method's saved
generations, and compile a side-by-side comparison table (Section 4.2-4.3).

Object Accuracy is computed by re-detecting objects in each generated image
with Grounding DINO, using the same ground-truth object names recorded
during dataset construction (``gcd.data.entity_linking``), and matching
detected labels to GT names with a sentence-embedding similarity threshold
(objects are expressed by the VLM/LLM pipeline with free-form nouns, so
detected-label <-> GT-name matching is not always an exact string match at
evaluation time, unlike the exact-key join used during dataset construction
in ``gcd.data.entity_linking``).

Usage
-----
    python -m gcd.evaluation.metrics \\
        --input-dir outputs/ \\
        --descriptions data/descriptions.jsonl \\
        --parsed data/parsed_images.jsonl \\
        --out results/metrics_summary.json \\
        --csv-out results/metrics_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import warnings
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

METHODS = ["attention_modulation", "attend_and_excite", "context_switching", "simple"]


def load_jsonl_map(path: str) -> Dict[str, dict]:
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = os.path.basename(rec.get("file") or rec.get("image") or "")
            if key:
                records[key] = rec
    return records


def clean_prompt(text: str) -> str:
    """Lowercase only — DINO matches camelCase as a single token better
    than a spaced-out phrase (matches what manual diagnostics use)."""
    return text.lower().strip()


class MetricsDataset(Dataset):
    """Loads generated images plus their matched ground-truth caption/objects."""

    def __init__(self, valid_entries: List[tuple], desc_map: dict, parsed_map: dict, gt_root: Optional[str]) -> None:
        self.valid_entries = valid_entries
        self.desc_map = desc_map
        self.parsed_map = parsed_map
        self.gt_root = gt_root

    def __len__(self) -> int:
        return len(self.valid_entries)

    def __getitem__(self, idx: int):
        img_path, base = self.valid_entries[idx]
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

        desc = self.desc_map.get(base, self.desc_map.get(base.replace(".jpg", ""), ""))
        if isinstance(desc, dict):
            desc = desc.get("description", "")

        gt_img = None
        gt_rec = self.parsed_map.get(base)
        if gt_rec:
            gt_file = gt_rec.get("file")
            if self.gt_root and gt_file and not os.path.exists(gt_file):
                gt_file = os.path.join(self.gt_root, os.path.basename(gt_file))
            if gt_file and os.path.exists(gt_file):
                try:
                    gt_img = Image.open(gt_file).convert("RGB")
                except Exception:  # noqa: BLE001
                    pass

        gt_obj_names: List[str] = []
        if gt_rec:
            objs = gt_rec.get("objects") or {}
            gt_obj_names = list(objs.keys()) if isinstance(objs, dict) else list(objs)

        clean_prompts = [clean_prompt(o) for o in gt_obj_names]
        dino_prompt = " . ".join(clean_prompts) + " ." if clean_prompts else ""

        print(f"[PROMPT-DEBUG] {base}: gt_names={gt_obj_names[:3]} "
            f"clean={clean_prompts[:3]} "
            f"final={dino_prompt[:180]!r}")
        return {
            "base": base, "pil": pil_img, "clip_desc": desc, "gt_img": gt_img,
            "gt_obj_names": gt_obj_names, "dino_prompt": dino_prompt,
        }


def metrics_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    return {
        "bases": [b["base"] for b in batch],
        "pils": [b["pil"] for b in batch],
        "clip_descs": [b["clip_desc"] for b in batch],
        "gt_imgs": [b["gt_img"] for b in batch],
        "gt_obj_names_list": [b["gt_obj_names"] for b in batch],
        "dino_prompts": [b["dino_prompt"] for b in batch],
    }


def compute_clip_score_batch(model, processor, pil_images, texts, device) -> List[float]:
    inputs = processor(text=texts, images=pil_images, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    img_emb = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
    txt_emb = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
    return torch.sum(img_emb * txt_emb, dim=1).tolist()


def compute_lpips_batch(lpips_net, pil_as, pil_bs, device) -> List[float]:
    from torchvision import transforms

    tr = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((256, 256)), transforms.ToTensor()])
    ta = torch.stack([tr(img) for img in pil_as]).to(device) * 2 - 1
    tb = torch.stack([tr(img) for img in pil_bs]).to(device) * 2 - 1
    with torch.no_grad():
        d = lpips_net(ta, tb)
    result = d.squeeze().cpu().tolist()
    return result if isinstance(result, list) else [result]


def _find_valid_entries(folder: Path) -> List[tuple]:
    valid_entries = []
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in (".jpg", ".jpeg", ".png"):
            valid_entries.append((entry, entry.name))
        elif entry.is_dir():
            candidate = entry / "generated_image.png"
            if candidate.exists():
                valid_entries.append((candidate, entry.name + ".jpg"))
            else:
                images = [p for p in entry.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
                if images:
                    valid_entries.append((max(images, key=lambda p: p.stat().st_size), entry.name + ".jpg"))
    return valid_entries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute CLIP / LPIPS / Object Accuracy across method folders")
    parser.add_argument("--input-dir", required=True, help="Folder containing one subfolder per method")
    parser.add_argument("--descriptions", default="descriptions.jsonl")
    parser.add_argument("--parsed", default="parsed_images.jsonl")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--out", default="metrics_summary.json")
    parser.add_argument("--csv-out", default="metrics_comparison.csv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--box-thresh", type=float, default=0.25)
    parser.add_argument("--text-thresh", type=float, default=0.25)
    parser.add_argument("--sim-thresh", type=float, default=0.70)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    return parser


def main() -> None:  # noqa: C901 - the evaluation loop is intentionally linear/inspectable
    args = build_arg_parser().parse_args()
    base_dir = Path(args.input_dir)

    desc_map = load_jsonl_map(args.descriptions) if os.path.exists(args.descriptions) else {}
    parsed_map = load_jsonl_map(args.parsed) if os.path.exists(args.parsed) else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_device = "cuda" if "cuda" in device else "cpu"
    cache_kwargs = {}
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        cache_kwargs["cache_dir"] = args.hf_cache_dir

    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", **cache_kwargs).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", **cache_kwargs)

    try:
        import lpips

        lpips_net = lpips.LPIPS(net="alex").to(device).eval()
    except Exception:  # noqa: BLE001
        lpips_net = None
        print("lpips not available; LPIPS will be skipped.")

    dino_model = dino_processor = sem_model = util = None
    dino_uses_unified_api = False
    try:
        import inspect

        from sentence_transformers import SentenceTransformer, util as st_util
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        util = st_util
        dino_id = "IDEA-Research/grounding-dino-base"
        dino_processor = AutoProcessor.from_pretrained(dino_id, **cache_kwargs)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id, **cache_kwargs).to(device).eval()
        dino_sig = inspect.signature(dino_processor.post_process_grounded_object_detection)
        dino_uses_unified_api = "box_threshold" not in dino_sig.parameters

        sem_cache_kwargs = {"cache_folder": args.hf_cache_dir} if args.hf_cache_dir else {}
        sem_model = SentenceTransformer("all-MiniLM-L6-v2", **sem_cache_kwargs).to(device).eval()
    except Exception:  # noqa: BLE001
        print("Object Accuracy will not be calculated (missing detection/embedding models).")

    csv_reports: Dict[str, dict] = {}
    results: Dict[str, dict] = {}

    for method in args.methods:
        folder = base_dir / method
        if not folder.exists():
            continue
        print(f"\nProcessing method: {method}")

        method_details: Dict[str, dict] = {}
        valid_entries = _find_valid_entries(folder)
        dataset = MetricsDataset(valid_entries, desc_map, parsed_map, args.gt_root)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.workers,
            collate_fn=metrics_collate_fn, pin_memory=(device == "cuda"),
        )

        for batch in tqdm(dataloader, desc=f"Evaluating {method}"):
            if not batch:
                continue
            pils, bases = batch["pils"], batch["bases"]

            for idx, base in enumerate(bases):
                gt_len = len(batch["gt_obj_names_list"][idx])
                method_details.setdefault(base, {
                    "clip": None, "lpips": None,
                    "object_accuracy": None if gt_len == 0 else 0.0, "intersect": 0, "desired": gt_len,
                })
                csv_reports.setdefault(base, {})
                csv_reports[base].setdefault(method, {
                    "clip": None, "lpips": None, "object_accuracy": None if gt_len == 0 else 0.0,
                })

            _score_clip(clip_model, clip_processor, batch, bases, pils, device, method_details, csv_reports, method)
            _score_lpips(lpips_net, batch, bases, pils, device, method_details, csv_reports, method)
            _score_object_accuracy(
                dino_model, dino_processor, dino_uses_unified_api, sem_model, util, batch, bases, pils,
                device, autocast_device, args, method_details, csv_reports, method,
            )

        results[method] = _aggregate(method_details, len(valid_entries))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote JSON summary to {args.out}")

    _write_csv(args.csv_out, csv_reports, [m for m in args.methods if m in results])
    print(f"Wrote CSV report to {args.csv_out}")


def _score_clip(clip_model, clip_processor, batch, bases, pils, device, method_details, csv_reports, method) -> None:
    valid_idx = [i for i, d in enumerate(batch["clip_descs"]) if d]
    if not valid_idx:
        return
    try:
        sims = compute_clip_score_batch(
            clip_model, clip_processor, [pils[i] for i in valid_idx], [batch["clip_descs"][i] for i in valid_idx], device
        )
        for idx, score in zip(valid_idx, sims):
            method_details[bases[idx]]["clip"] = float(score)
            csv_reports[bases[idx]][method]["clip"] = float(score)
    except Exception:  # noqa: BLE001
        pass


def _score_lpips(lpips_net, batch, bases, pils, device, method_details, csv_reports, method) -> None:
    if lpips_net is None:
        return
    valid_idx = [i for i, gt in enumerate(batch["gt_imgs"]) if gt is not None]
    if not valid_idx:
        return
    try:
        scores = compute_lpips_batch(lpips_net, [pils[i] for i in valid_idx], [batch["gt_imgs"][i] for i in valid_idx], device)
        for idx, score in zip(valid_idx, scores):
            method_details[bases[idx]]["lpips"] = float(score)
            csv_reports[bases[idx]][method]["lpips"] = float(score)
    except Exception:  # noqa: BLE001
        pass


def _score_object_accuracy(
    dino_model, dino_processor, dino_uses_unified_api, sem_model, util, batch, bases, pils,
    device, autocast_device, args, method_details, csv_reports, method,
) -> None:
    if dino_model is None or sem_model is None:
        return
    valid_idx = [i for i, p in enumerate(batch["dino_prompts"]) if p]
    if not valid_idx:
        return
    try:
        iou_pils = [pils[i] for i in valid_idx]
        prompts = [batch["dino_prompts"][i] for i in valid_idx]
        gt_names_batch = [batch["gt_obj_names_list"][i] for i in valid_idx]

        inputs = dino_processor(images=iou_pils, text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)

        target_sizes = [img.size[::-1] for img in iou_pils]
        if dino_uses_unified_api:
            dino_res = dino_processor.post_process_grounded_object_detection(
                outputs, threshold=args.box_thresh, target_sizes=target_sizes
            )
        else:
            dino_res = dino_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=args.box_thresh, text_threshold=args.text_thresh,
                target_sizes=target_sizes,
            )  
        for j, res in enumerate(dino_res):
            labels_j = res.get("text_labels", res.get("labels", []))
            print(f"[EVAL-DEBUG] base={bases[valid_idx[j]]} "
                f"n_det={len(labels_j)} "
                f"labels={[str(x) for x in labels_j[:10]]}")
            gt_names = gt_names_batch[j]
            detected = res.get("text_labels", res.get("labels", []))
            desired_count = len(gt_names)
            intersect = 0
            if detected:
                gt_embs = sem_model.encode(gt_names, convert_to_tensor=True)
                det_embs = sem_model.encode(detected, convert_to_tensor=True)
                max_sims, _ = torch.max(util.cos_sim(gt_embs, det_embs), dim=1)
                intersect = torch.sum(max_sims >= args.sim_thresh).item()

            capped_intersect = min(intersect, desired_count)
            base_img = bases[valid_idx[j]]
            val = float(capped_intersect) / desired_count if desired_count > 0 else 0.0

            method_details[base_img]["object_accuracy"] = val
            method_details[base_img]["intersect"] = capped_intersect
            csv_reports[base_img][method]["object_accuracy"] = val
    except Exception:  # noqa: BLE001
        # except Exception as e:  # noqa: BLE001
        print(f"[OA-DEBUG] {method}: EXCEPTION: {e!r}")
        import traceback
        traceback.print_exc()


def _aggregate(method_details: Dict[str, dict], n_images: int) -> dict:
    clip_vals = [d["clip"] for d in method_details.values() if d["clip"] is not None]
    lpips_vals = [d["lpips"] for d in method_details.values() if d["lpips"] is not None]
    total_intersect = sum(d["intersect"] for d in method_details.values())
    total_desired = sum(d["desired"] for d in method_details.values())
    return {
        "summary": {
            "n_images_processed": n_images,
            "avg_clip": mean(clip_vals) if clip_vals else None,
            "avg_lpips": mean(lpips_vals) if lpips_vals else None,
            "overall_object_accuracy": float(total_intersect) / total_desired if total_desired > 0 else None,
        },
        "images": method_details,
    }


def _write_csv(csv_out: str, csv_reports: Dict[str, dict], methods: List[str]) -> None:
    headers = ["Image_File"]
    for m in methods:
        headers.extend([f"{m}_CLIP", f"{m}_LPIPS", f"{m}_ObjectAccuracy"])

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for base_img, per_method in sorted(csv_reports.items()):
            row = [base_img]
            for m in methods:
                data = per_method.get(m, {})
                row.extend([data.get("clip") or "", data.get("lpips") or "", data.get("object_accuracy") or ""])
            writer.writerow(row)


if __name__ == "__main__":
    main()
