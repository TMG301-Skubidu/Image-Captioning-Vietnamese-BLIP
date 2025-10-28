# BLIP Experiment Comparison Template

Paper: BLIP – Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (arXiv:2201.12086v2)
Repo: README.md, configs/*.yaml

Use this template to record experimental settings and compare reproduced results vs. paper/official numbers.

## General Setup
- Date:
- Hardware (GPU type/count, RAM):
- Software (Python, PyTorch, CUDA):
- BLIP commit/hash:
- Seeds:

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
| CIDEr | 133.0 (COCO Karpathy test) | 132.3 (val), 133.3 (test) |
| BLEU-4 | 39.7 | 40.0 (val), 39.7 (test) |
| BLEU-3 | — | 50.6 (val), 50.5 (test) |
| BLEU-2 | — | 63.8 (val), 63.7 (test) |
| BLEU-1 | — | 78.7 (val), 78.9 (test) |
| SPICE | 14.7 (NoCaps overall) | 23.8 (val ≈ test) |
| METEOR | — | 30.9 (val), 31.0 (test) |

Notes/Deviations:
- Evaluation run with provided CapFilt-L checkpoint; JSON annotations auto-patched to add missing `info` field before calling `pycocotools`.
- Paper metrics report BLEU/CIDEr/SPICE ×100; reproduced numbers and METEOR are shown in the same ×100 scale for easier comparison.
