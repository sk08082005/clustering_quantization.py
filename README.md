# Clustering-Driven Image Feature Quantization

Mini project (Topic 17, CO1 / CO2, BTL-5) by **Pushp Prakash Tandon** and **Suraj Kapoor**.

**Goal:** use clustering to quantize image features, and investigate how the *representative cluster centers* (their number and their quality) affect a downstream pattern-recognition task. The downstream task here is classifying a photo as an **original image** or a **photo of a screen** (screen recapture).

---

## Method

1. **Patch features.** Each image is downscaled (longest side 384 px) and split into non-overlapping 32x32 patches. Each patch is described by 10 features: mean brightness, contrast, Laplacian variance (sharpness), Sobel edge strength, FFT high-frequency ratio, FFT peak ratio (moire cue), sensor noise, saturation, color spread and banding.
2. **Codebook.** K-Means is fitted on the *training* patches only (features are standardized first). The `k` cluster centers form the codebook.
3. **Quantization.** Every patch is replaced by the ID of its nearest center. An image becomes a normalized `k`-bin histogram (a bag of visual words).
4. **Classification.** A RandomForest classifier is trained on the histograms.
5. **Evaluation.** Everything uses stratified 5-fold cross-validation. The codebook is rebuilt inside each fold, so no test data leaks into training.

## Experiments

| Experiment | Question |
|---|---|
| A. Baseline | What accuracy do we get without quantization (mean and std of the patch features per image)? |
| B. Effect of k | How do accuracy and quantization error change for k = 2, 4, 8, 16, 32, 64, 128? |
| C. Codebook quality | At k = 32, do properly clustered centers beat weaker codebooks (random-init K-Means, 1-iteration K-Means, random patches, random Gaussian centers)? |
| D. Center analysis | What does each cluster center represent (representative patch, class purity, feature profile, PCA view)? |

## Results

**Baseline (no quantization):** about 0.94 accuracy.

**Effect of k** (5-fold CV, mean +/- std across folds):

| k | Bits per patch | Accuracy | Macro-F1 | Test distortion |
|---|---|---|---|---|
| 2 | 1 | 0.732 +/- 0.028 | 0.730 | 7.60 |
| 4 | 2 | 0.850 +/- 0.068 | 0.849 | 5.81 |
| 8 | 3 | 0.900 +/- 0.037 | 0.899 | 4.19 |
| 16 | 4 | 0.921 +/- 0.049 | 0.920 | 3.21 |
| 32 | 5 | 0.945 +/- 0.036 | 0.945 | 2.50 |
| 64 | 6 | 0.947 +/- 0.027 | 0.947 | 1.98 |
| 128 | 7 | 0.943 +/- 0.031 | 0.942 | 1.58 |

**Codebook quality at k = 32** (3 seeds):

| Codebook | Accuracy | Macro-F1 | Test distortion |
|---|---|---|---|
| K-Means (k-means++) | 0.938 | 0.937 | 2.50 |
| K-Means (random init) | 0.940 | 0.940 | 2.51 |
| K-Means (1 iteration) | 0.931 | 0.930 | 2.92 |
| Random patches | 0.944 | 0.943 | 3.82 |
| Random Gaussian centers | 0.936 | 0.936 | 7.33 |

![Experiment results](results/experiment_results.png)

### Key findings

- **About 32 centers are enough.** Accuracy rises steeply up to k = 32 and then plateaus. Quantizing each patch to a 5-bit cluster ID matches the unquantized baseline.
- **Distortion and accuracy are decoupled.** Quantization error keeps falling as k grows, but accuracy stops improving after k = 32.
- **Center quality mattered little in this setting.** A random Gaussian codebook has about three times the distortion of K-Means (7.33 vs 2.50) yet nearly the same accuracy. The number of regions the feature space is split into mattered far more than how well the centers fit the data (random Gaussian at k = 32 scores 0.936, while K-Means at k = 2 scores 0.732). The gaps between codebooks (up to about 0.013) are smaller than the fold-to-fold spread (about 0.03), so they should not be read as one codebook being better than another.
- **Single patches are only weakly informative; histograms are not.** Most centers have a screen-patch share between 8% and 89%, with many near the overall 49%. The class signal comes from how an image's patches are distributed across the centers.

### What the centers represent

![Representative patches](results/representative_patches.png)
![Cluster profiles](results/cluster_profiles.png)
![PCA view](results/pca_clusters.png)

- **Screen-leaning centers (C1, C5, C6):** flat, dark, smooth patches with little detail. C1 also has a high FFT peak ratio, consistent with periodic (moire-like) structure.
- **Original-leaning centers (C3, C8, C15):** rough natural texture (gravel, foliage) with high sharpness, edge strength and noise, or bright smooth areas.
- **Mixed centers (C7, C9, C11, C14):** UI elements, hair and blurry objects that occur in both classes, so they carry little class information on their own.

## Limitations

- **Weak patch labels.** Every patch inherits its image's label, so a cluster with about 50% screen patches means that kind of patch appears in both classes.
- **Possible source or content bias.** Some centers capture image content (keyboard keys, a taskbar, foliage) rather than screen artifacts. Part of the accuracy may come from differences in subject matter or image source between the two classes, so the model should not be assumed to detect screens purely from moire.
- **Differences within the noise.** The codebook-quality differences are inside the cross-validation spread. A paired comparison with repeated cross-validation, especially at small k, would be needed to claim an effect.
- **Small dataset.** Results come from a limited set of images and may not generalize to other devices, lighting or screens.

## Project structure

```
clustering_quantization.py      # full pipeline and experiments
clustering_quantization.ipynb   # same pipeline as a notebook
requirements.txt
results/                        # graphs, CSVs, saved model (created when the script runs)
original/                       # NOT included - your original photos
screen/                         # NOT included - your photos of screens
```

The image folders are not part of this repository.

**Dataset:** `original/` contains _[describe your original images and their source]_. `screen/` contains _[describe your screen photos: devices, number of images]_.

## How to run

1. Create the two image folders next to the script: `original/` and `screen/`.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the script:
   ```
   python clustering_quantization.py
   ```
   The figures open one at a time (close each window to continue). Everything is also saved in `results/`. A full run takes a few minutes.

Settings such as patch size, the list of k values and the number of folds are at the top of the script.

## Outputs

- `results/experiment_results.png`: accuracy vs k, distortion vs k, codebook quality
- `results/representative_patches.png`, `cluster_profiles.png`, `pca_clusters.png`: analysis of the cluster centers
- `results/sweep_k.csv`, `results/codebook_strategies.csv`: result tables
- `results/quantized_model.pkl`: the fitted scaler, codebook and classifier (trained on all images; use the cross-validation numbers above for accuracy)

## Tech stack

Python, OpenCV, NumPy, scikit-learn, Matplotlib, joblib.
