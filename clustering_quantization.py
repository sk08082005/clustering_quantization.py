# %% [markdown]
# # Clustering-Driven Image Feature Quantization
# **Goal:** Use K-Means clustering to quantize image features (build a codebook) and investigate how the quality and the number (k) of the cluster centers affect a downstream pattern-recognition task (Screen vs Original image classification).
# 
# **Pipeline**
# 1. Split every image into 32x32 patches and extract 10 small features from each patch
# 2. Run **K-Means** on the training patches -> `k` cluster centers = the **codebook**
# 3. Assign every patch to its nearest center (**quantization**); an image becomes a k-bin histogram (bag of visual words)
# 4. Train a classifier (RandomForest) on the histograms
# 5. Experiments: vary k, vary the quality of the codebook, and compare against a baseline
# 
# **Folder structure (next to this notebook):** the `original/` and `screen/` folders. Restart the kernel and use **Run All**, from top to bottom.
# 
# If a module is missing: `pip install opencv-python numpy scikit-learn matplotlib joblib`

# %%
import os, csv, time, warnings
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances_argmin, pairwise_distances_argmin_min
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# %% [markdown]
# ## 1. Settings (everything is controlled from here)

# %%
ORIGINAL_DIR = "original"
SCREEN_DIR   = "screen"

PATCH      = 32          # patch size (pixels)
MAX_SIDE   = 384         # longest image side is downscaled to this size
K_LIST     = [2, 4, 8, 16, 32, 64, 128]   # codebook sizes to test
STRATEGY_K = 32          # k used for the codebook-quality experiment
FINAL_K    = 16          # k used for visualization and the saved model
N_FOLDS    = 5
SEEDS      = [0, 1, 2]   # random seeds for the codebook-quality experiment
FIT_SAMPLE = 15000       # max patches used to fit K-Means
RF_TREES   = 100

# %% [markdown]
# ## 2. Patch-level feature extraction
# 10 features per 32x32 patch: brightness, contrast, sharpness (Laplacian), edge strength (Sobel), FFT high-frequency ratio, FFT peak ratio (moire), sensor noise, saturation, color spread and banding.

# %%
FEATURE_NAMES = ["mean_gray", "std_gray", "log_lap_var", "sobel_mean", "fft_hf_ratio",
                 "fft_peak_ratio", "noise_std", "mean_sat", "color_spread", "banding"]

def _to_patches(arr, P):
    H, W = arr.shape[:2]
    gh, gw = H // P, W // P
    if arr.ndim == 2:
        return arr.reshape(gh, P, gw, P).transpose(0, 2, 1, 3).reshape(-1, P, P)
    C = arr.shape[2]
    return arr.reshape(gh, P, gw, P, C).transpose(0, 2, 1, 3, 4).reshape(-1, P, P, C)

def load_and_resize(path, max_side=MAX_SIDE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not read image")
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    h2, w2 = (h // PATCH) * PATCH, (w // PATCH) * PATCH
    if h2 < PATCH or w2 < PATCH:
        raise ValueError("image is too small")
    return img[:h2, :w2]

_yy, _xx = np.ogrid[:PATCH, :PATCH]
_rad = np.sqrt((_yy - PATCH // 2) ** 2 + (_xx - PATCH // 2) ** 2)
_RING = _rad > 2
_HF = _rad > PATCH * 0.25
_hann = np.hanning(PATCH).astype(np.float32)
_WIN = np.outer(_hann, _hann)

def patch_features(img):
    # returns: (n_patches, 10) float features, (n_patches, P, P) uint8 gray patches
    P = PATCH
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = _to_patches(gray, P)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    sob = np.sqrt(sx ** 2 + sy ** 2)
    resid = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    sat = _to_patches(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32), P)
    bgr = _to_patches(img.astype(np.float32), P)

    mean_gray = g.mean((1, 2)) / 255
    std_gray = g.std((1, 2)) / 255
    log_lap_var = np.log1p(_to_patches(lap, P).var((1, 2)))
    sobel_mean = _to_patches(sob, P).mean((1, 2)) / 255
    noise_std = _to_patches(resid, P).std((1, 2)) / 255
    mean_sat = sat.mean((1, 2)) / 255
    color_spread = bgr.mean((1, 2)).std(1) / 255

    F = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean((1, 2), keepdims=True)) * _WIN, axes=(1, 2)), axes=(1, 2)))
    power = F ** 2
    fft_hf_ratio = power[:, _HF].sum(1) / (power[:, _RING].sum(1) + 1e-6)
    Fr = F[:, _RING]
    fft_peak_ratio = Fr.max(1) / (Fr.mean(1) + 1e-6)

    d = np.abs(np.diff(g, axis=2))
    banding = ((d > 1) & (d < 4)).mean((1, 2))

    feats = np.stack([mean_gray, std_gray, log_lap_var, sobel_mean, fft_hf_ratio,
                      fft_peak_ratio, noise_std, mean_sat, color_spread, banding], axis=1).astype(np.float32)
    return feats, g.astype(np.uint8)

# %% [markdown]
# ## 3. Load the dataset
# Labels: `original` = 0, `screen` = 1. The patch features of every image are stored.

# %%
def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts))

feats_list, patches_list, labels, names = [], [], [], []
t0 = time.time()
for label, folder in enumerate([ORIGINAL_DIR, SCREEN_DIR]):
    for path in list_images(folder):
        try:
            f, p = patch_features(load_and_resize(path))
            feats_list.append(f); patches_list.append(p); labels.append(label); names.append(path)
        except Exception as e:
            print("skip:", os.path.basename(path), "->", e)

y = np.array(labels)
print(f"Images loaded: {len(y)}  (original={np.sum(y == 0)}, screen={np.sum(y == 1)})")
print(f"Total patches: {sum(len(f) for f in feats_list)}   |   time: {time.time() - t0:.1f}s")

# For the baseline: mean + std of each image's patch features (a continuous representation, no quantization)
pooled = np.array([np.concatenate([f.mean(0), f.std(0)]) for f in feats_list])

assert min(np.sum(y == 0), np.sum(y == 1)) >= 2, "Each class needs at least 2 images"
n_splits = int(min(N_FOLDS, np.sum(y == 0), np.sum(y == 1)))
folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
print("CV folds:", n_splits)

# %% [markdown]
# ## 4. Codebook and quantization functions
# Codebook strategies (used to compare the quality of the centers):
# - `kmeans++`: fully converged K-Means (best case)
# - `kmeans_random_init`: random initialization, converged
# - `kmeans_1_iter`: K-Means run for only 1 iteration (under-trained centers)
# - `random_patches`: no clustering, random patches are used as the centers
# - `random_gaussian`: purely random (meaningless) centers

# %%
def build_codebook(strategy, Xtr, k, seed):
    rng = np.random.RandomState(seed)
    if strategy == "kmeans++":
        return KMeans(k, init="k-means++", n_init=3, random_state=seed).fit(Xtr).cluster_centers_
    if strategy == "kmeans_random_init":
        return KMeans(k, init="random", n_init=1, random_state=seed).fit(Xtr).cluster_centers_
    if strategy == "kmeans_1_iter":
        return KMeans(k, init="random", n_init=1, max_iter=1, random_state=seed).fit(Xtr).cluster_centers_
    if strategy == "random_patches":
        return Xtr[rng.choice(len(Xtr), k, replace=False)]
    if strategy == "random_gaussian":
        return rng.randn(k, Xtr.shape[1])
    raise ValueError(strategy)

def stack_patches(idx):
    return np.vstack([feats_list[i] for i in idx])

def encode_images(idx_list, scaler, centers):
    # assign each patch to its nearest center -> image = normalized histogram; distortion = mean squared quantization error
    X = scaler.transform(stack_patches(idx_list))
    assign, dist = pairwise_distances_argmin_min(X, centers)
    H, pos = [], 0
    for i in idx_list:
        n = len(feats_list[i])
        h = np.bincount(assign[pos:pos + n], minlength=len(centers)).astype(np.float32)
        H.append(h / n); pos += n
    return np.array(H), float((dist ** 2).mean())

def run_experiment(strategy, k, seed=0):
    accs, f1s, d_tr, d_te = [], [], [], []
    rng = np.random.RandomState(seed)
    for tr, te in folds:
        Xtr = stack_patches(tr)
        scaler = StandardScaler().fit(Xtr)
        Xs = scaler.transform(Xtr)
        if len(Xs) > FIT_SAMPLE:
            Xs = Xs[rng.choice(len(Xs), FIT_SAMPLE, replace=False)]
        centers = build_codebook(strategy, Xs, k, seed)
        Htr, dtr = encode_images(tr, scaler, centers)
        Hte, dte = encode_images(te, scaler, centers)
        rf = RandomForestClassifier(n_estimators=RF_TREES, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(Htr, y[tr])
        pred = rf.predict(Hte)
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
        d_tr.append(dtr); d_te.append(dte)
    return dict(acc=float(np.mean(accs)), acc_std=float(np.std(accs)),
                f1=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
                dist_tr=float(np.mean(d_tr)), dist_te=float(np.mean(d_te)))

# %% [markdown]
# ## 5. Experiment A - Baseline (no quantization)
# Same patch features, but each image is represented by the mean and std of its patch features (continuous). The quantized models are compared against this.

# %%
accs, f1s = [], []
for tr, te in folds:
    rf = RandomForestClassifier(n_estimators=RF_TREES, class_weight="balanced", random_state=0, n_jobs=-1)
    rf.fit(pooled[tr], y[tr]); pred = rf.predict(pooled[te])
    accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
base_acc, base_acc_std, base_f1 = float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))
print(f"BASELINE (no quantization): accuracy = {base_acc:.4f} +/- {base_acc_std:.4f} | macro-F1 = {base_f1:.4f}")

# %% [markdown]
# ## 6. Experiment B - effect of k (codebook size)
# Small k = more compression, large k = more detail. Observe how accuracy changes with k.

# %%
sweep = {}
print(f"{'k':>5} {'bits':>5} {'accuracy':>16} {'macro-F1':>9} {'test distortion':>16}")
for k in K_LIST:
    t0 = time.time()
    sweep[k] = run_experiment("kmeans++", k, seed=0)
    r = sweep[k]
    print(f"{k:>5} {np.log2(k):>5.1f} {r['acc']:>9.4f} +/-{r['acc_std']:.3f} {r['f1']:>9.4f} {r['dist_te']:>16.4f}   ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 7. Experiment C - effect of cluster-center quality
# Five different kinds of codebooks at the same k (`STRATEGY_K`). Question: *do representative (properly clustered) centers actually matter?*

# %%
strategies = ["kmeans++", "kmeans_random_init", "kmeans_1_iter", "random_patches", "random_gaussian"]
strat_res = {}
print(f"k = {STRATEGY_K}, seeds = {SEEDS}")
print(f"{'strategy':<20} {'accuracy':>9} {'seed-std':>9} {'macro-F1':>9} {'test distortion':>16}")
for s in strategies:
    runs = [run_experiment(s, STRATEGY_K, seed=sd) for sd in SEEDS]
    strat_res[s] = dict(acc=np.mean([r["acc"] for r in runs]), acc_seed_std=np.std([r["acc"] for r in runs]),
                        f1=np.mean([r["f1"] for r in runs]), dist_te=np.mean([r["dist_te"] for r in runs]))
    r = strat_res[s]
    print(f"{s:<20} {r['acc']:>9.4f} {r['acc_seed_std']:>9.4f} {r['f1']:>9.4f} {r['dist_te']:>16.4f}")

# %% [markdown]
# ## 8. Result graphs

# %%
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))

ks = K_LIST
ax[0].errorbar(ks, [sweep[k]["acc"] for k in ks], yerr=[sweep[k]["acc_std"] for k in ks], marker="o", capsize=3, label="Quantized: accuracy")
ax[0].plot(ks, [sweep[k]["f1"] for k in ks], marker="s", label="Quantized: macro-F1")
ax[0].axhline(base_acc, ls="--", color="gray", label="Baseline accuracy (no quantization)")
ax[0].set_xscale("log", base=2); ax[0].set_xticks(ks); ax[0].set_xticklabels(ks)
ax[0].set_xlabel("Codebook size k"); ax[0].set_ylabel("Score"); ax[0].set_title("Accuracy vs k"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

ax[1].plot(ks, [sweep[k]["dist_te"] for k in ks], marker="o", color="tab:red")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(ks); ax[1].set_xticklabels(ks)
ax[1].set_xlabel("Codebook size k"); ax[1].set_ylabel("Mean squared quantization error (test)")
ax[1].set_title("Distortion vs k (larger k = less information loss)"); ax[1].grid(alpha=.3)

names_s = list(strat_res.keys())
ax[2].bar(range(len(names_s)), [strat_res[s]["acc"] for s in names_s], yerr=[strat_res[s]["acc_seed_std"] for s in names_s], capsize=4,
          color=["tab:green", "tab:olive", "tab:orange", "tab:purple", "tab:gray"])
ax[2].set_xticks(range(len(names_s))); ax[2].set_xticklabels(names_s, rotation=25, ha="right", fontsize=8)
lo = min(strat_res[s]["acc"] for s in names_s)
ax[2].set_ylim(max(0, lo - 0.1), 1.0)
ax[2].axhline(base_acc, ls="--", color="gray")
ax[2].set_title(f"Codebook quality (k={STRATEGY_K})"); ax[2].set_ylabel("CV accuracy"); ax[2].grid(alpha=.3, axis="y")

plt.tight_layout(); plt.savefig("results/experiment_results.png", dpi=150); plt.show()

with open("results/sweep_k.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["k", "bits", "acc", "acc_std", "f1", "dist_test"])
    for k in ks: w.writerow([k, np.log2(k), sweep[k]["acc"], sweep[k]["acc_std"], sweep[k]["f1"], sweep[k]["dist_te"]])
with open("results/codebook_strategies.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["strategy", "acc", "acc_seed_std", "f1", "dist_test"])
    for s in names_s: w.writerow([s, strat_res[s]["acc"], strat_res[s]["acc_seed_std"], strat_res[s]["f1"], strat_res[s]["dist_te"]])
print("Saved: results/experiment_results.png, sweep_k.csv, codebook_strategies.csv")

# %% [markdown]
# ## 9. Representative cluster centers - what does each center represent?
# Here we build `FINAL_K` centers on the full dataset (for analysis and visualization only). Note: a patch gets the label of its image (a weak label).

# %%
all_idx = np.arange(len(y))
Xall = stack_patches(all_idx)
scaler = StandardScaler().fit(Xall)
Xs_all = scaler.transform(Xall)
rng = np.random.RandomState(0)
fit_X = Xs_all[rng.choice(len(Xs_all), min(FIT_SAMPLE, len(Xs_all)), replace=False)]
centers = build_codebook("kmeans++", fit_X, FINAL_K, 0)

assign = pairwise_distances_argmin(Xs_all, centers)
patch_label = np.concatenate([[y[i]] * len(feats_list[i]) for i in all_idx])
all_gray_patches = np.vstack(patches_list)

counts = np.bincount(assign, minlength=FINAL_K)
screen_frac = np.array([patch_label[assign == c].mean() if counts[c] else np.nan for c in range(FINAL_K)])

# (a) Representative patch (the real patch nearest to each center) + screen %
nearest = pairwise_distances_argmin(centers, Xs_all)
cols = int(np.ceil(np.sqrt(FINAL_K))); rows = int(np.ceil(FINAL_K / cols))
fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows))
for c, a in enumerate(np.array(axes).ravel()):
    a.axis("off")
    if c < FINAL_K:
        a.imshow(all_gray_patches[nearest[c]], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a.set_title(f"C{c}  n={counts[c]}\nscreen={100 * screen_frac[c]:.0f}%", fontsize=8)
plt.suptitle(f"Representative patch of each cluster center (k={FINAL_K})", y=1.02)
plt.tight_layout(); plt.savefig("results/representative_patches.png", dpi=150, bbox_inches="tight"); plt.show()

# (b) Per-cluster screen purity + (c) center feature profile
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
ax[0].bar(range(FINAL_K), 100 * screen_frac, color=plt.cm.coolwarm(screen_frac))
ax[0].axhline(100 * patch_label.mean(), ls="--", color="gray", label="overall screen-patch %")
ax[0].set_xlabel("Cluster center"); ax[0].set_ylabel("% screen patches"); ax[0].set_title("Cluster purity (0% = pure original, 100% = pure screen)"); ax[0].legend()
im = ax[1].imshow(centers.T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
ax[1].set_yticks(range(len(FEATURE_NAMES))); ax[1].set_yticklabels(FEATURE_NAMES, fontsize=8)
ax[1].set_xticks(range(FINAL_K)); ax[1].set_xlabel("Cluster center"); ax[1].set_title("Center profile (z-score of each feature)")
plt.colorbar(im, ax=ax[1]); plt.tight_layout(); plt.savefig("results/cluster_profiles.png", dpi=150); plt.show()

# (d) PCA view
sel = rng.choice(len(Xs_all), min(6000, len(Xs_all)), replace=False)
pca = PCA(2).fit(Xs_all)
Z, Zc = pca.transform(Xs_all[sel]), pca.transform(centers)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].scatter(Z[:, 0], Z[:, 1], c=assign[sel], cmap="tab20" if FINAL_K <= 20 else "viridis", s=5, alpha=.6)
ax[0].scatter(Zc[:, 0], Zc[:, 1], c="black", marker="X", s=120, edgecolor="white"); ax[0].set_title("Patches colored by cluster (X = centers)")
ax[1].scatter(Z[:, 0], Z[:, 1], c=patch_label[sel], cmap="coolwarm", s=5, alpha=.6)
ax[1].scatter(Zc[:, 0], Zc[:, 1], c="black", marker="X", s=120, edgecolor="white"); ax[1].set_title("Patches colored by class (blue=original, red=screen)")
for a in ax: a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.savefig("results/pca_clusters.png", dpi=150); plt.show()

# %% [markdown]
# ## 10. Save the final model + single-image prediction
# (For accuracy, refer to the cross-validation numbers above; this model is only for demos and app integration.)

# %%
H_all, _ = encode_images(list(all_idx), scaler, centers)
final_rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0, n_jobs=-1).fit(H_all, y)
joblib.dump(dict(scaler=scaler, centers=centers, rf=final_rf, patch=PATCH, max_side=MAX_SIDE), "results/quantized_model.pkl")
print("Saved: results/quantized_model.pkl")

def predict_image(path, threshold=0.5):
    f, _ = patch_features(load_and_resize(path))
    a = pairwise_distances_argmin(scaler.transform(f), centers)
    h = np.bincount(a, minlength=len(centers)).astype(np.float32); h /= h.sum()
    p = float(final_rf.predict_proba([h])[0][1])
    return ("Screen Image" if p >= threshold else "Original Image"), p

for path in [names[0], names[-1]]:
    label, p = predict_image(path)
    print(f"{path}  ->  {label}  (screen probability = {p:.3f})")

# %% [markdown]
# ## 11. Auto-generated conclusions (for the viva)

# %%
best_k = max(sweep, key=lambda k: sweep[k]["acc"])
best_acc = sweep[best_k]["acc"]
enough_k = min(k for k in sweep if sweep[k]["acc"] >= best_acc - 0.01)

print(f"1. Baseline (no quantization) accuracy      : {base_acc:.3f}")
print(f"2. Best quantized accuracy                  : {best_acc:.3f}  at k = {best_k}")
print(f"3. Smallest k within 1% of best accuracy    : k = {enough_k}  ({np.log2(enough_k):.0f} bits per patch)")
diff = best_acc - base_acc
print(f"4. Quantized vs baseline                    : {'+' if diff >= 0 else ''}{100 * diff:.1f} percentage points")
print(f"5. Distortion falls from {sweep[K_LIST[0]]['dist_te']:.3f} (k={K_LIST[0]}) to {sweep[K_LIST[-1]]['dist_te']:.3f} (k={K_LIST[-1]})")
print()
print(f"Codebook quality at k={STRATEGY_K}:")
for s in strat_res:
    print(f"   {s:<20} accuracy = {strat_res[s]['acc']:.3f}   distortion = {strat_res[s]['dist_te']:.3f}")
gap = strat_res["kmeans++"]["acc"] - strat_res["random_gaussian"]["acc"]
print(f"-> Properly clustered centers vs meaningless random centers: {100 * gap:+.1f} percentage points")
