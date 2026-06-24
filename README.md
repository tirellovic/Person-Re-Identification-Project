# Person Search on the PRW Dataset

Machine Learning for Computer Vision — Assignment A.Y. 2025/2026
Alessandro Tirelli (ID 0001189769) — alessandro.tirelli2@studio.unibo.it

## Overview

**Person Search** is the joint task of **pedestrian detection** and **person re-identification**: given a query image with a single bounding box around a person, the goal is to detect and match that same person across a gallery of raw, uncropped scene frames, based on body shape and clothing.

I tackle the problem with a **two-stage pipeline**, training each stage separately:

1. **First stage — Detector.** A `RetinaNet` with a ResNet-34 + FPN backbone localizes every pedestrian in each frame. The anchors are tuned to the tall aspect ratios of standing people rather than the generic RetinaNet defaults, exploiting the dataset statistics.
2. **Second stage — Embedder.** A ResNet-50 feature extractor followed by a 2048-d projection and a **BNNeck** (the *Bag of Tricks* design) maps each detected crop, resized to 128×256, to an appearance descriptor. It is trained as a **metric-learning** problem with a batch-hard **triplet loss** plus an auxiliary **identity-classification loss**: embeddings of the same identity are pulled together and different identities pushed apart.

At test time the stages are chained: every detection in the gallery is embedded, each query box is embedded with the same network, and queries are matched against the gallery by cosine similarity. The identity classifier is used only during training to provide more discriminative gradients; it is discarded at inference, keeping the system open-world (not constrained to a fixed set of known identities).

## Results

Evaluated on the PRW test set with the provided `eval_search_prw` function (literature-standard setting `ignore_cam_id=True`; a detection is a true positive only if IoU > 0.5 with the ground truth **and** the identity matches):

| Metric | Value |
|---|---|
| mAP | **36.42%** |
| top-1 | **78.12%** |

These results are in line with two-stage Person Search methods reported on PRW. The high top-1 with a lower mAP is typical of the dataset: a correct match is usually retrieved at the top of the ranking, but ranking *all* occurrences of an identity ahead of the many visually similar distractors is much harder.

**Ablation.** The main variation studied was the auxiliary identity loss on top of the triplet loss. Supervising the embeddings with identity labels yields better-separated, more discriminative descriptors than the triplet loss alone, while leaving the model open-world at inference. The full comparison and training curves are in `scripts/second-stage-embedder.ipynb`.

## Project structure

```
person-search/
├── README.md
├── scripts/
│   ├── main.ipynb                   # Submission notebook: loads weights, runs the full
│   │                                #   pipeline on the test set, evaluation + qualitative results
│   ├── first-stage-detector.ipynb   # Detector training procedure (RetinaNet)
│   └── second-stage-embedder.ipynb  # Embedder training procedure + ablation
└── utility/
    ├── utils.py                     # Helper functions (detection, cropping, drawing, parsing, ...)
    └── eval_function.py             # Provided eval_search_prw metric (mAP / top-1)
```

`main.ipynb` is the notebook to grade: it is **inference-only** (training disabled, trained weights loaded) and intertwines code with textual explanations, plus interactive qualitative visualizations.

## How to run

The project was developed on **Kaggle** (GPU, attached datasets) and is intended to be run there.

1. Open `scripts/main.ipynb` on Kaggle.
2. In *Notebook Settings*, set the accelerator to **GPU T4** and enable **Internet** (ImageNet-pretrained backbones are downloaded on first run).
3. Attach the following **public** Kaggle resources (already referenced by the paths in the *Parameters* cell):
   - **Dataset** — `edoardomerli/prw-person-re-identification-in-the-wild` (the PRW dataset)
   - **Dataset** — `alessandrotirelli/person-search-utility` (`utils.py` and `eval_function.py`)
   - **Model** — `alessandrotirelli/person-search-retina-resnet34-detector` (detector weights, `.pt`)
   - **Model** — `alessandrotirelli/person-search-resnet50-bnneck-embedder` (embedder weights, `.pt`)
4. *Run All*. The notebook detects pedestrians, builds and embeds the gallery and queries, prints the mAP / top-1 metrics, and renders the interactive qualitative tools.

No `requirements.txt` is provided because the notebook targets the Kaggle environment, where all dependencies are preinstalled. Running locally would require re-pointing the `data_root` and weight paths and installing the corresponding PyTorch / torchvision / scipy / ipywidgets stack.
