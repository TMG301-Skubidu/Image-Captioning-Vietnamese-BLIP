# BLIP Experiment Comparison Template

Paper: BLIP – Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (arXiv:2201.12086v2)
Repo: README.md, configs/*.yaml

Use this template to record experimental settings and compare reproduced results vs. paper/official numbers.

## General Setup
- Date: 2025-10-29 (Kaggle job wall-clock ≈54 min)
- Hardware (GPU type/count, RAM): Paper – two 16-GPU nodes (32 GPUs total; GPU model not detailed). Reproduced – Kaggle Notebook with 1× NVIDIA Tesla P100 (16 GB VRAM), 2 vCPU, ~13 GB RAM.
- Software (Python, PyTorch, CUDA): Paper – PyTorch implementation (versions not stated). Reproduced – Kaggle Python 3.11.13 runtime, Kaggle default CUDA/PyTorch stack (torch 2.x w/ CUDA 12.x), extra pip deps: timm 0.4.12, fairscale 0.4.4, transformers 4.30.2, pycocoevalcap, ruamel.yaml 0.17.21, PyYAML 6.0.3.
- BLIP commit/hash: bef7421f9d65efe5056272e84e90ec29d5c15c09 (Salesforce/BLIP main as cloned in notebook)
- Seeds: Paper – not specified. Reproduced – default library seeds (no manual seeding).

## Image Captioning (COCO)
| Field | Value |
|---|---|
| Dataset/version | MS COCO (train/val/test split) |
| Config | `configs/caption_coco.yaml` |
| Checkpoint used | `model_base_caption_capfilt_large.pth` (or specify) |
| Image size | 384 |
| Batch size / GPUs | |
| Decoding | beam size, max/min length |

### Metrics (official vs. reproduced)

| Metric | Paper (BLIP CapFilt-L) | Reproduced |
|---|---:|---:|
| CIDEr | 133.3  | 133.3  |
| BLEU-4 | 39.7 | 40.0 


Notes/Deviations:
- Evaluation run with provided CapFilt-L checkpoint; JSON annotations auto-patched to add missing `info` field before calling `pycocotools`.
- Paper metrics report BLEU/CIDEr/SPICE ×100; reproduced numbers and METEOR are shown in the same ×100 scale for easier comparison.
