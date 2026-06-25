# Person Search on the PRW Dataset

Machine Learning for Computer Vision — Assignment A.Y. 2025/2026

Alessandro Tirelli (ID 0001189769) — alessandro.tirelli2@studio.unibo.it

## Overview

**Person Search** is the joint task of **pedestrian detection** and **person re-identification**: given a query image with a single bounding box around a person, the goal is to detect and match that same person across a gallery of raw, uncropped scene frames, based on body shape and clothing. The task and the PRW dataset were introduced by Zheng et al. [[1]](https://arxiv.org/abs/1604.02531).

I tackled the problem with a **two-stage pipeline**, training each stage separately:

1. **First stage — Detector.** A `RetinaNet` [[6]](https://arxiv.org/abs/1708.02002) with a ResNet-34 + FPN backbone localizes every pedestrian in each frame; its focal loss handles the heavy foreground/background imbalance of dense scenes. The anchors are tuned to the tall aspect ratios of standing people rather than the generic RetinaNet defaults, exploiting the dataset statistics.
2. **Second stage — Embedder.** A ResNet-50 feature extractor followed by a 2048-d projection and a **BNNeck** maps each detected crop, resized to 128×256, to an appearance descriptor. The BNNeck and the overall training recipe follow the *Bag of Tricks* strong baseline for Re-ID [[7]](https://arxiv.org/abs/1903.07071). It is trained as a **metric-learning** problem with a batch-hard **triplet loss** [[8]](https://arxiv.org/abs/1703.07737) plus an auxiliary **identity-classification loss**: embeddings of the same identity are pulled together and different identities pushed apart.

At test time the stages are chained: every detection in the gallery is embedded, each query box is embedded with the same network, and queries are matched against the gallery by cosine similarity. The identity classifier is used only during training to provide more discriminative gradients; it is discarded at inference, keeping the system open-world (not constrained to a fixed set of known identities).

## Results

Evaluated on the PRW test set with the provided `eval_search_prw` function (literature-standard setting `ignore_cam_id=True`; a detection is a true positive only if IoU > 0.5 with the ground truth and the identity matches):
<p align="center">

| Metric | Value |
|---|---|
| mAP | **36.42%** |
| top-1 | **78.12%** |
</p>

These results are in line with two-stage Person Search methods reported on PRW. The high top-1 with a lower mAP is typical of the dataset: a correct match is usually retrieved at the top of the ranking, but ranking *all* occurrences of an identity ahead of the many visually similar distractors is much harder.

## Ablation studies

The detector was kept fixed; all the experimentation focused on the **second-stage embedder**, which I built up incrementally. The table reports the full-pipeline metrics at each step.

<p align="center">

| Embedder configuration | mAP | top-1 |
|---|---|---|
| ResNet-18, 512-d, vanilla triplet loss | 19.36% | 59.26% |
| ResNet-18, 512-d, vanilla triplet loss + random Gaussian blur | 24.68% | 70.59% |
| ResNet-50, 2048-d, triplet + Gaussian blur (no ID loss) | ≈30%* | ≈75%* |
| **ResNet-50, 2048-d, triplet + ID loss + Gaussian blur (final)** | **36.42%** | **78.12%** |
</p>

<sub>* Approximate values recalled from the experiment; the exact metrics for this run were not logged.</sub>

**Starting point — vanilla triplet, ResNet-18, 512-d.** The first embedder used a plain batch-hard triplet loss (no identity head), a ResNet-18 backbone and a smaller 512-d descriptor. It did learn to associate identities with their clothes and accessories, but a clear failure mode emerged: detections of the *same* person at *different scales* landed far apart in embedding space, i.e. the network was implicitly encoding **scale** rather than pure appearance. The pipeline reached only mAP = 19.36%, top-1 = 59.26% — not satisfactory.

**Adding random Gaussian blur.** To discourage the model from relying on scale-dependent, high-frequency cues and to improve generalization, I randomly blurred a fraction of the training detections with Gaussian kernels. This pushed the embedder toward scale-invariant, more semantic features and lifted results substantially, to mAP = 24.68%, top-1 = 70.59%. A residual issue remained: the embedder still absorbed too much **context** — background textures, clutter such as fences in front of the subject, and pose variations that hide accessories like bags or backpacks — which pollutes the descriptor with non-identity information.

**Scaling up the backbone — ResNet-50, 2048-d (no ID loss).** Replacing ResNet-18/512-d with the stronger ResNet-50 backbone and a larger 2048-d descriptor (keeping the triplet loss and Gaussian blur) further improved both metrics over the ResNet-18 baselines, to roughly **mAP ≈ 30%, top-1 ≈ 75%** (approximate values recalled from the experiment, not logged at the time). It served as the starting point for the final design decision below.

**Adding the identity-classification loss (final).** On top of the ResNet-50 configuration, I added an auxiliary **identity-classification loss** alongside the triplet loss. Supervising the embeddings with identity labels injects more discriminative gradients and yields better-separated descriptors, while the classifier head is discarded at inference so the system stays open-world. This produced a further, clearly noticeable improvement — **most markedly on mAP** — giving the final mAP = 36.42%, top-1 = 78.12%. The context-leakage problem is mitigated but not fully solved even by the more capable backbone, and remains the most promising direction for further improvement.

## Project structure

```
person-search/
├── README.md
├── scripts/
│   ├── main.ipynb                   # Main notebook
│   ├── first-stage-detector.ipynb   # Detector training procedure (RetinaNet)
│   └── second-stage-embedder.ipynb  # Embedder training procedure + ablation
└── utility/
    ├── utils.py                     # Helper functions (detection, cropping, drawing, parsing, ...)
    └── eval_function.py             # Provided eval_search_prw metric (mAP / top-1)
```

## How to run

The project was developed on **Kaggle** (GPU, attached datasets) and is intended to be run there.

### Recommended: run the public Kaggle notebook (no setup)

The easiest and recommended way to run the project is to open my public Kaggle notebook (available at [this link](https://www.kaggle.com/code/alessandrotirelli/person-search-main)), which is identical to `scripts/main.ipynb`. The PRW dataset, the utility code and both model checkpoints are already attached to it, so no import or manual configuration is needed. Just *Copy & Edit* (or *Run All*), making sure the accelerator is set to **GPU T4** and **Internet** is enabled (the ImageNet-pretrained backbones are downloaded on first run).

### Alternative: import the notebook manually

If you prefer to run the submitted `scripts/main.ipynb` file directly:

1. Open `scripts/main.ipynb` on Kaggle.
2. In *Notebook Settings*, set the accelerator to **GPU T4** and enable **Internet**.
3. Attach the following **public** Kaggle resources (already referenced by the paths in the *Parameters* cell):
   - **Dataset** — `edoardomerli/prw-person-re-identification-in-the-wild` (the PRW dataset)
   - **Dataset** — `alessandrotirelli/person-search-utility` (`utils.py` and `eval_function.py`)
   - **Model** — `alessandrotirelli/person-search-retina-resnet34-detector` (detector weights, `.pt`)
   - **Model** — `alessandrotirelli/person-search-resnet50-bnneck-embedder` (embedder weights, `.pt`)
4. *Run All*. The notebook detects pedestrians, builds and embeds the gallery and queries, prints the mAP / top-1 metrics, and renders the interactive qualitative tools.

> [!Note]
> No `requirements.txt` is provided because the notebook targets the Kaggle environment, where all dependencies are preinstalled. Running locally would require re-pointing the `data_root` and weight paths and installing the corresponding PyTorch / torchvision / scipy / ipywidgets stack.

## References

The dataset and task references suggested in the assignment guidelines [1–5], plus the works whose solutions I directly adopted in my pipeline [6–8].

[[1]](https://arxiv.org/abs/1604.02531) L. Zheng et al. *Person Re-identification in the Wild.* CVPR 2017. — PRW dataset and the Person Search task.  
[[2]](https://arxiv.org/abs/2210.12903) L. Jaffe et al. *Gallery Filter Network for Person Search.* WACV 2023.  
[[3]](https://arxiv.org/abs/2203.09642) R. Yu et al. *Cascade Transformers for End-to-End Person Search.* CVPR 2022.  
[[4]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Instance_Guided_Proposal_Network_for_Person_Search_CVPR_2020_paper.pdf) W. Dong et al. *Instance Guided Proposal Network for Person Search.* CVPR 2020.  
[[5]](https://arxiv.org/abs/2103.11617) Y. Yan et al. *Anchor-Free Person Search.* CVPR 2021.  
[[6]](https://arxiv.org/abs/1708.02002) T.-Y. Lin et al. *Focal Loss for Dense Object Detection (RetinaNet).* ICCV 2017. — first-stage detector.  
[[7]](https://arxiv.org/abs/1903.07071) H. Luo et al. *Bag of Tricks and a Strong Baseline for Deep Person Re-Identification.* CVPRW 2019. — BNNeck and embedder training recipe.  
[[8]](https://arxiv.org/abs/1703.07737) A. Hermans et al. *In Defense of the Triplet Loss for Person Re-Identification.* arXiv 2017. — batch-hard triplet loss.
