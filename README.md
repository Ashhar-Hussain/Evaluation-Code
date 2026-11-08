# Benchmark Accuracy Overstates the Performance of Explainable Wildfire Detection Models Under Distribution Shift

**Evaluation code for the paper.** This repository contains everything needed to
reproduce the experiments, tables and figures: six image-based wildfire detection
models trained under one controlled protocol, evaluated on a public benchmark and,
without retraining, on an independent dataset of deliberately confusing scenes.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B%20%28Keras%203%29-FF6F00)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

> **Paper.** A. Hussain, A. Kabir, N. Ali, M. Yasir Shabir, M. Rashid and Z. Mehmood.
> *Benchmark Accuracy Overstates the Performance of Explainable Wildfire Detection
> Models Under Distribution Shift.* Submitted to *Discover Artificial Intelligence*
> (Springer Nature), 2026. 

---

## What the study found

We compared six convolutional models under a single protocol: **MWAMNet**, an
explainable ensemble of DenseNet201 and MobileNetV3Large with Grad-CAM and SHAP, and
five established single-backbone models.

- **The public benchmark is saturated.** On the held-out test set of the Kaggle
  Forest Fire Dataset all six models score 97.9–99.0% and no pairwise difference is
  statistically significant. With 380 test images, a significant difference cannot
  even be reached.
- **Every model loses about 24 accuracy points under distribution shift.** On The
  Wildfire Dataset, used without retraining, accuracy falls to 68–76% and the
  models find fewer than two thirds of real fires.
- **Under shift the models do separate.** MWAMNet significantly outperforms
  DenseNet201, MobileNetV3Large, ResNet50 and VGG16. In the pre-specified study it
  is in turn significantly outperformed by a single-backbone InceptionV3.
- **Preprocessing matters.** A uniform `1/255` rescaling, applied to backbones that
  expect other input ranges, can understate some models by several accuracy points.
  Every model here uses its own reference preprocessing.

| Model | In-domain accuracy (n = 380) | External accuracy (n = 2,699) | Drop | External recall |
|---|---:|---:|---:|---:|
| InceptionV3 | 0.9842 | 0.7644 | −0.220 | 0.578 |
| **MWAMNet** | **0.9895** | **0.7518** | −0.238 | 0.548 |
| DenseNet201 | 0.9842 | 0.7399 | −0.244 | 0.558 |
| ResNet50 | 0.9895 | 0.7277 | −0.262 | 0.495 |
| MobileNetV3Large | 0.9842 | 0.7192 | −0.265 | 0.528 |
| VGG16 | 0.9789 | 0.6795 | −0.299 | 0.415 |

*Values are from each model's representative run, the median of five seeds. Means
and standard deviations over all five seeds are in the paper and in `results/`.*


## Quick start (Google Colab, recommended)

The notebooks were developed and run on **Google Colab with a T4 GPU**.

1. Open a notebook.
2. Choose **Runtime → Change runtime type → T4 GPU**. Do not choose a TPU runtime:
   Colab's TPU runtimes do not provide TensorFlow in the form these notebooks need.
3. Choose **Runtime → Run all**. Allow Colab to mount Google Drive when asked: the
   notebooks keep their cached features in `MyDrive/MWAMNet_revision`, so a
   restarted runtime resumes instead of starting again.


Run Stage 1 first. Stage 3 reuses the external features cached by Stage 2. Stages 4
and 5 can be run on their own.

**Running locally** (a CUDA GPU is strongly recommended):


The notebooks are written for Colab. To run them elsewhere, remove or skip the
cells that mount Google Drive.

---

## Data

Both datasets are public and are downloaded automatically with
[`kagglehub`](https://github.com/Kaggle/kagglehub). No images are stored in this
repository.

| Dataset | Role | Size | Source |
|---|---|---|---|
| Forest Fire Dataset | training, validation, in-domain test | 1,900 images (950 fire, 950 no-fire) | [kaggle.com/datasets/alik05/forest-fire-dataset](https://www.kaggle.com/datasets/alik05/forest-fire-dataset) |
| The Wildfire Dataset | external validation only, never used for training | 2,699 images | [kaggle.com/datasets/elmadafri/the-wildfire-dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset) |

If `kagglehub` asks for credentials, sign in to Kaggle, create an API token, and add
your username and key as Colab secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`).

**Partition of the Forest Fire Dataset.** The 380-image `Testing` folder is held out
and used once. The 1,520-image `Training` folder is split 80/20 by stratified
sampling with seed 42.

| Partition | Images | No-fire / Fire | Share |
|---|---:|---:|---:|
| Training | 1,216 | 608 / 608 | 64% |
| Validation | 304 | 152 / 152 | 16% |
| Test (held out) | 380 | 190 / 190 | 20% |

---

## Reproducing the paper

| In the paper | Produced by | Output |
|---|---|---|
| Table 1, dataset partition | Stage 1 | `split_manifest.json` |
| Section 4, duplicate check | `scripts/duplicate_check.py` | console output |
| Table 4, in-domain results | Stage 1 | `table2.csv` |
| Section 7.1, in-domain McNemar tests | Stage 1 | `mcnemar.csv` |
| Figure 9, in-domain confusion matrices | Stage 1 | `confusion_matrices.pdf` |
| Table 5, ablation | Stage 1 | `ablation.csv` |
| Table 6, effect of input normalisation | Stage 1 (corrected) and `legacy/` (earlier) | `table2.csv` and the saved outputs of the legacy notebooks |
| Table 7, external validation | Stage 2 | `external.csv` |
| Table 8, external McNemar tests | Stage 2 | `mcnemar_external.csv` |
| Figure 10, external confusion matrices | Stage 2 | `confusion_external.pdf` |
| Table 9, pre-specified study | Stage 3 | `config_grid_dev.csv`, `stage3_hold.csv`, `stage3_mcnemar_hold.csv` |
| Table 10 and the ROC-AUC bootstrap | Stage 3 | printed by the threshold and bootstrap cell |
| Table 11, computational cost | Stage 4 | `efficiency.csv` |
| Figures 11–13, Grad-CAM and SHAP | Stage 5 | `gradcam_layer_comparison.pdf`, `gradcam_panels.pdf`, `shap_panels.pdf` |

---

## The protocol in brief

The same protocol is applied to every model, so that differences can be attributed
to the backbone rather than to how it was treated.

- **Backbones** are frozen ImageNet feature extractors with global average pooling.
  Only the classifier head is trained; there is no fine-tuning.
- **Preprocessing** uses each backbone's own Keras `preprocess_input`, never a
  uniform `1/255`.
- **Classifier head:** batch normalisation and dropout (0.5), dense layers of 1024
  and 512 units with L2 regularisation (0.01), and a sigmoid output.
- **Training:** AdamW (learning rate 1e-4, weight decay 1e-4), cosine decay applied
  per step, batch size 32, up to 100 epochs, early stopping on validation accuracy
  (patience 15, best weights restored).
- **Augmentation** (training images only): four seeded variants of each image using
  horizontal flip, brightness, contrast, saturation, hue and JPEG quality.
- **Seeds:** every model is trained five times, with seeds 42–46. Single-run values
  come from the run with the median accuracy; mean ± SD covers all five runs.
- **Statistics:** Wilson 95% intervals for individual accuracies. McNemar's test for
  paired comparisons: exact when the models disagree on fewer than 25 images, χ²
  with continuity correction otherwise. Holm–Bonferroni correction across the five
  baseline comparisons. A 2,000-resample bootstrap for differences in ROC-AUC.
- **Explanations:** Grad-CAM at the final convolutional block of each backbone, and
  SHAP with the partition explainer, an `inpaint_telea` masker and 1,200 evaluations
  per image. The partition explainer is deterministic, so no random seed is needed.

## Practical notes

- **GPU memory.** DenseNet201 is memory-hungry. Feature extraction starts with a
  batch of 16 and halves it automatically if the GPU runs out of memory.
- **Run time.** Feature extraction takes the longest, typically tens of minutes on a
  free T4, and is cached. Training the heads on cached features takes minutes. In
  Stage 3, setting `DO_TTA = False` skips test-time augmentation and saves time.
- **Versions.** The notebooks were run on Colab in September 2026 with TensorFlow 2
  and Keras 3. Stage 5 reads layer shapes in the Keras 3 way (`layer.output.shape`).

---

## Citation

If this code or these findings are useful in your work, please cite:

```bibtex
@article{hussain2026benchmark,
  title   = {Benchmark Accuracy Overstates the Performance of Explainable Wildfire
             Detection Models Under Distribution Shift},
  author  = {Hussain, Ashhar and Kabir, Asif and Ali, Nabeel and
             Yasir Shabir, Muhammad and Rashid, Muhammad and Mehmood, Zahid},
  journal = {Discover Artificial Intelligence},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

---

## Acknowledgements

We are grateful to the creators of the
[Forest Fire Dataset](https://www.kaggle.com/datasets/alik05/forest-fire-dataset) and
of [The Wildfire Dataset](https://doi.org/10.3390/f14091697) (El-Madafri, Peña and
Olmedo-Torre, 2023) for making their data openly available, and to the reviewers,
whose questions shaped this evaluation.

## License

The code is released under the [MIT License](LICENSE). The datasets are subject to
their own licences on Kaggle.

## Contact

**Ashhar Hussain** (corresponding author), Department of Computer Science and
Information Technology, University of Kotli Azad Jammu and Kashmir, Pakistan.
Email: asharhussain.csit@uokajk.edu.pk · ORCID
[0009-0009-1568-9852](https://orcid.org/0009-0009-1568-9852)

Questions, corrections and reproductions are welcome: please open an issue.
