# Guided Cumulative Diffusion (GCD)

Reference implementation for *Guided Cumulative Diffusion: Graph-Guided
Entity Prioritization and Log-Bias Attention Modulation for Compositional
Text-to-Image Synthesis*.

This repository is organized into three stages:

1. **Dataset construction** (`gcd.data`) — build the scene-graph dataset used
   to train the entity-priority GCN (Section 3.2).
2. **GNN training** (`gcd.gnn`) — train the priority predictor (Section 3.3).
3. **Generation + evaluation** (`gcd.diffusion`, `gcd.evaluation`) —
   inference-only experiment runners for four methods, and the CLIP / LPIPS /
   Object-Accuracy evaluation suite (Sections 3.4-3.5, 4).

No diffusion-model **training** code is included: every generation method
runs entirely at inference time on a frozen, pretrained backbone.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## 1. Dataset construction

```bash
python main.py data caption          --image-dir data/raw_images --output data/descriptions.jsonl
python main.py data parse-graph      --input data/descriptions.jsonl --output data/parsed_images.jsonl
python main.py data background       --descriptions data/descriptions.jsonl --parsed data/parsed_images.jsonl --output data/backgroundContext.jsonl
python main.py data merge            --parsed data/parsed_images.jsonl --context data/backgroundContext.jsonl --output data/merged.jsonl
python main.py data extract-prompts  --input data/parsed_images.jsonl --output data/detection_prompts.txt
python main.py data detect           --image-dir data/raw_images --output data/boundingBoxCoordinates.jsonl --parsed-images data/parsed_images.jsonl
python main.py data link-entities    --parsed data/parsed_images.jsonl --detections data/boundingBoxCoordinates.jsonl --output data/semanticMatches.jsonl
python main.py data build-graphs     --merged data/merged.jsonl --bboxes data/boundingBoxCoordinates.jsonl --output-dir data/gnn_graphs
python main.py data explore          --semantic-path data/semanticMatches.jsonl --merged-path data/merged.jsonl
```

## 2. Train the entity-priority GCN

```bash
python main.py gnn train --graph-dir data/gnn_graphs --model-type gcn --epochs 100 \
    --checkpoint-out checkpoints/gnn_model.pt
```

## 3. Generate with each method

Two scene-loading modes are available for every generation command:

- **Precomputed** (use this — you already have these files): reuses your
  existing `parsed_images.jsonl` / `backgroundContext.jsonl` (or a merged
  `merged.jsonl`) and the independent per-image `gnn_graphs/{file_id}.json`
  files, so the GNN scores priority using the *real* object-relation edges
  it was trained on, and no LLM is loaded at generation time at all.
- **Live** (fallback if you don't have precomputed graphs for an image):
  re-parses the raw description with an LLM on the fly, using hashed,
  edge-less features — matching the original `gnn_inference.py` behavior.

```bash
# Precomputed mode (recommended — reuses your existing dataset files):
python main.py generate \
    --method attention_modulation \
    --descriptions data/descriptions.jsonl \
    --graphs-dir data/gnn_graphs \
    --parsed data/parsed_images.jsonl \
    --background data/backgroundContext.jsonl \
    --test-ids data/splits/test_ids.txt \
    --gnn-checkpoint checkpoints/gnn.pt \
    --output-root outputs/attention_modulation

# Same, but using merged.jsonl (already has background_context — omit --background):
python main.py generate \
    --method attention_modulation \
    --descriptions data/descriptions.jsonl \
    --graphs-dir data/gnn_graphs \
    --parsed data/merged.jsonl \
    --gnn-checkpoint checkpoints/gnn.pt \
    --output-root outputs/attention_modulation

# Attention-Modulation ramp-size / seed sweep:
python main.py generate \
    --method attention_modulation \
    --descriptions data/descriptions.jsonl \
    --graphs-dir data/gnn_graphs \
    --parsed data/parsed_images.jsonl \
    --background data/backgroundContext.jsonl \
    --gnn-checkpoint checkpoints/gnn.pt \
    --output-root outputs/ramp_sweep \
    --ramp-sizes 0 5 10 15 20 \
    --seeds 0 1 2

# Multi-backbone (SD v1.5 / Dreamshaper-8 / SDXL) matrix over the other three methods
# — add --graphs-dir/--parsed/--background the same way for precomputed mode:
python main.py generate-multi-backbone \
    --descriptions data/descriptions.jsonl \
    --graphs-dir data/gnn_graphs \
    --parsed data/parsed_images.jsonl \
    --background data/backgroundContext.jsonl \
    --test-ids data/splits/test_ids.txt \
    --gnn-checkpoint checkpoints/gnn.pt \
    --output-root outputs/multi_backbone \
    --backbone sd15=runwayml/stable-diffusion-v1-5 \
    --backbone dreamshaper8=Lykon/dreamshaper-8 \
    --backbone sdxl=stabilityai/stable-diffusion-xl-base-1.0 \
    --steps sdxl=500
```

### Exactly which files are required for testing, and where to put them

Only these — everything else (`boundingBoxCoordinates*.jsonl`,
`semanticMatches.jsonl`, `parsed_images_failures.jsonl`) was only needed
earlier, to *build* the GNN training graphs, and is not read at
generation/evaluation time:

| File | Put it at (example) | Used for |
|---|---|---|
| `descriptions.jsonl` | `data/descriptions.jsonl` | enumerating test images (`file`, `description`) + CLIP ground-truth text at eval time |
| `parsed_images.jsonl` (or `merged.jsonl`) | `data/parsed_images.jsonl` | objects + relations for prompt-stacking/sanitization |
| `backgroundContext.jsonl` | `data/backgroundContext.jsonl` | background text (omit if using `merged.jsonl`, which already has it) |
| `gnn_graphs/{file_id}.json` (independent per-image files, **flat** — not nested under `train/`/`test/`) | `data/gnn_graphs/` | node features + real edges the GNN was trained on |
| your trained checkpoint | `checkpoints/gnn.pt` | pass via `--gnn-checkpoint checkpoints/gnn.pt` |
| `test_ids.txt` (one COCO image-id per line, e.g. `000000002261`) | `data/splits/test_ids.txt` | optional — restricts `--descriptions` to your test split |

`{file_id}` is the COCO image id (the description/graph `file` path's stem,
e.g. `.../000000002261.jpg` → `000000002261`) — that's how `descriptions.jsonl`
/ `parsed_images.jsonl` entries get matched to their `gnn_graphs/*.json` file.

**Without `--test-ids`**, every image in `--descriptions` is a candidate, but
`--max-images`/`--n-images` (default **50**) still randomly downsamples from
that pool. Pass `--max-images 0` (or `--n-images 0` for the multi-backbone
runner) to run on every image with no sampling at all.

`outputs/<root>/<method>/<image_id>/generated_image.png` +
`inference_result.json` is written per run; `intermediate/` holds phase
snapshots for Context Switching and Attention Modulation.

## 4. Evaluate

```bash
python main.py evaluate \
    --input-dir outputs/attention_modulation_root \
    --descriptions data/descriptions.jsonl \
    --parsed data/parsed_images.jsonl \
    --out results/metrics_summary.json \
    --csv-out results/metrics_comparison.csv

python main.py report --summary results/metrics_summary.json
```

`--input-dir` should contain one subfolder per method
(`simple/`, `attend_and_excite/`, `context_switching/`, `attention_modulation/`),
matching the folder-per-method layout produced by `generate` /
`generate-multi-backbone`.

## Directory structure

See `REPO_STRUCTURE.md` for the full annotated file tree.

## What's intentionally excluded

- Any diffusion-model **training** code (LoRA fine-tuning, staged-training
  curricula). All four methods are inference-only.
- Image blurring / cumulative-blur-view rendering. That dataset only feeds
  the paper's *unsuccessful* fine-tuning ablation (Section 3.6), which this
  repository does not implement.
