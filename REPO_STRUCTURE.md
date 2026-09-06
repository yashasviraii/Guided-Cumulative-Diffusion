# Repository Structure

```
gcd_repo/
├── README.md                          Quick-start commands for every stage
├── pyproject.toml                     Package metadata (installable as `gcd`)
├── requirements.txt                   Pinned high-level dependencies
├── main.py                            Single CLI entrypoint; dispatches to every
│                                       gcd.* module's own argparse parser
│
├── configs/                           Declarative defaults (edit these instead of
│   │                                   passing every flag by hand)
│   ├── data_pipeline.yaml             Paths + model IDs for dataset construction
│   ├── gnn.yaml                       GCN architecture / training hyperparameters
│   └── experiments.yaml               Method list, backbone list, ramp/seed sweep
│
├── data/                              (gitignored) raw images, JSONL intermediates,
│   └── splits/                        constructed graphs — populated by running
│                                       the `data` subcommands
├── checkpoints/                       (gitignored) trained GNN checkpoints
├── outputs/                           (gitignored) generated images per method
├── results/                           (gitignored) evaluation JSON/CSV reports
│
└── src/gcd/                           Installable package (`pip install -e .`)
    │
    ├── __init__.py
    │
    ├── data/                          ── STAGE 1: Dataset construction (Sec. 3.2) ──
    │   ├── vlm_captioning.py          Step 1: image -> caption (VLM)
    │   ├── graph_parsing.py           Step 2: caption -> {objects, relations} (LLM,
    │   │                               batched, 3-tier JSON-recovery fallback)
    │   ├── background_extraction.py   Step 2b: caption -> background text (LLM,
    │   │                               batched — ported from extractBackground.py)
    │   ├── merge_context.py           Step 2c: join parsed graph + background
    │   ├── prompt_extraction.py       Step 3: unique object vocabulary -> detection
    │   │                               prompts
    │   ├── object_detection.py        Step 4: Grounding DINO -> GT bounding boxes
    │   ├── entity_linking.py          Step 4b: exact-key join of parsed entities to
    │   │                               detected boxes, area-fraction computation
    │   ├── graph_builder.py           Step 5: SceneGraphBuilder — final GCN
    │   │                               training graphs (vocabulary-fitted features
    │   │                               + area-fraction targets)
    │   └── dataset_explorer.py        QA: stats / priority-distribution / integrity
    │                                   checks (blur-dataset-specific code removed —
    │                                   out of scope, see exclusions below)
    │
    ├── gnn/                           ── STAGE 2: Priority-predictor training (Sec 3.3) ──
    │   ├── models.py                  NodeLevelGNN (3-layer GCN, Eq. 1), SimpleMLP
    │   │                               ablation, SimpleGCNInference (deployment loader)
    │   ├── dataset.py                 PyTorch Geometric GraphDataset
    │   ├── trainer.py                 GNNTrainer: train/eval loop, early stopping
    │   └── train.py                   CLI entrypoint (`python main.py gnn train`)
    │
    ├── parsing/                       ── Shared inference-time LLM parsing ──
    │   └── description_parser.py      DescriptionParser: extract_objects /
    │                                   extract_relations / extract_background /
    │                                   rewrite_prompt_stack — used by every
    │                                   generation method
    │
    ├── graph/                         ── Shared inference-time graph utilities ──
    │   ├── graph_builder.py           InferenceGraphBuilder (7-dim hashed features,
    │   │                               single-image, no dataset-wide vocabulary —
    │   │                               used only by the *live* LLM-parse fallback)
    │   ├── graph_io.py                Loader for independent gnn_graphs/{file_id}.json
    │   │                               files (flat directory, as actually exported —
    │   │                               not nested under train/test)
    │   ├── priority_scorer.py         PriorityScorer: clip + renormalize to sum 1,
    │   │                               forwards real edge_index to a GCN checkpoint
    │   └── sanitize.py                sanitize_background: strip foreground mentions
    │
    ├── diffusion/                     ── STAGE 3: Inference-only generation (Sec 3.4-3.5) ──
    │   ├── attention_processor.py     GCDAttentionProcessor (Eq. 2-3 log-bias),
    │   │                               find_token_spans, build_weight_schedule
    │   ├── backbone.py                Shared pipeline loading / scheduler-swap /
    │   │                               latent-decode helpers (single-backbone path)
    │   ├── prompt_stacking.py         build_cumulative_prompt, compute_step_allocations
    │   │                               — shared by 3 of the 4 methods for a fair,
    │   │                               identical-prompt comparison
    │   ├── precomputed_scene.py       load_precomputed_scene: reuses your existing
    │   │                               parsed_images.jsonl / backgroundContext.jsonl
    │   │                               (or merged.jsonl) + gnn_graphs/*.json instead
    │   │                               of re-parsing with a live LLM — the recommended
    │   │                               path once the data pipeline has already run
    │   ├── methods/
    │   │   ├── simple_diffusion.py            Method 1: unmodified one-shot baseline
    │   │   ├── attend_and_excite.py           Method 2: Chefer et al. 2023 (via
    │   │   │                                   diffusers' official pipeline)
    │   │   ├── context_switching.py           Method 3: hard prompt-switching
    │   │   │                                   ablation (Crystallization Problem)
    │   │   └── attention_modulation.py        Method 4: GCD (ours)
    │   ├── runner.py                  Unified single-backbone batch-inference CLI,
    │   │                               with ramp-size / seed sweep for Attention
    │   │                               Modulation (replaces the three near-duplicate
    │   │                               run_batch_inference*.py scripts)
    │   └── multi_backbone.py          Multi-backbone (SD 1.x + SDXL) driver for
    │                                   Simple / Context-Switching / Attention-
    │                                   Modulation, reproducing Table 1's three-
    │                                   backbone comparison (ported from run_all_batch.py)
    │
    └── evaluation/                    ── Evaluation suite (Sec 4.2-4.3) ──
        ├── metrics.py                 CLIP score, LPIPS, Object Accuracy (Grounding
        │                               DINO re-detection + sentence-embedding match)
        └── report.py                  Compile metrics_summary.json into a
                                        Table-1-style Markdown comparison table
```

## What was deliberately left out of the codebase's `codebase.txt`

Per the filtering rules in the task brief, the following original scripts were
**not** ported, because they are diffusion-model *training* code or blur/image
-processing utilities unrelated to the attention-modulation pipeline:

- `diffusion_staged_training.py`, `gssd_masked_training.py`,
  `gssd_combined_training.py` — LoRA fine-tuning training loops.
- `diffusion_staged_inference.py` — inference for the *trained* LoRA checkpoint
  from the (excluded) fine-tuning ablation.
- `quickstart.py` — an interactive wizard around the excluded LoRA training script.
- The blur-stage generation and cumulative-blur-view logic inside the original
  `gnn_graph_builder.py` ("Optional: blurring" step) and `data_explorer.py`'s
  `visualize_sample` / blur-folder checks — this data only feeds the paper's
  *unsuccessful* fine-tuning ablation (Section 3.6).

Two files were **easy to mistake as out-of-scope by name but are actually
required, load-bearing pipeline steps** — flagged here since it's the kind of
thing a name-based filter would wrongly drop:

- `extractBackground.py` sounds like image/background *pixel* processing, but
  it is actually the batched **LLM text** call that produces background
  *descriptions* — it feeds `merge_context.py` and is required. Ported as
  `gcd/data/background_extraction.py`.
- `run_all_batch.py` sounds like a duplicate of the other
  `run_batch_inference*.py` scripts, but it is the *only* place implementing
  the SDXL-compatible denoising loop needed to reproduce Table 1's SDXL row.
  Ported as `gcd/diffusion/multi_backbone.py`.

`data_explorer.py` was partially ported: its blur-stage inspection methods were
dropped, but its dataset-statistics, priority-distribution, and data-integrity
checks (unrelated to blurring) were kept in `gcd/data/dataset_explorer.py`.
