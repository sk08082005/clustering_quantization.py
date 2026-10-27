from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)

# ---- Load the trained model (must be saved from your notebook as model.pkl) ----
MODEL_PATH = "model.pkl"
clf = None
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)
else:
    print(f"WARNING: {MODEL_PATH} not found. Run the save-model cell in your notebook first.")


def _resize_keep_aspect(img, target=256):
    """Aspect ratio preserve karke resize + pad, taaki moire frequency distort na ho."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _fft_peak_features(gray):
    """
    Fixed band dekhne ke bajaye, FFT magnitude spectrum mein actual
    periodic peaks dhundte hain (moire pattern ka signature).
    """
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # DC component (center) ko mask karo — hume periodic peaks chahiye, average brightness nahi
    mask = np.ones_like(magnitude)
    r = 8
    mask[cy - r:cy + r, cx - r:cx + r] = 0
    masked = magnitude * mask

    # Top-K peaks nikaalo — screen recapture mein sharp, high-value peaks honge
    flat = masked.flatten()
    k = 20
    top_k = np.partition(flat, -k)[-k:]
    peak_mean = float(np.mean(top_k))
    peak_std = float(np.std(top_k))

    # Peak-to-background ratio — ye scale-invariant hai (dono images resize se pehle
    # alag resolution ki ho sakti hain, ye ratio phir bhi comparable rehta hai)
    background_mean = float(np.mean(masked[masked > 0])) if np.any(masked > 0) else 1e-6
    peak_to_background = peak_mean / (background_mean + 1e-6)

    return peak_mean, peak_std, peak_to_background


def _color_banding_score(img):
    """
    Screen recapture (especially compressed/re-encoded) mein color transitions
    kam smooth hote hain — banding/quantization zyada dikhta hai.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    grad_x = np.abs(np.diff(l_channel, axis=1))
    grad_y = np.abs(np.diff(l_channel, axis=0))

    # "Flat jumps" — small but non-zero jumps jo banding indicate karte hain
    banding_x = np.mean((grad_x > 1) & (grad_x < 4))
    banding_y = np.mean((grad_y > 1) & (grad_y < 4))
    return float((banding_x + banding_y) / 2)


def _texture_regularity(gray):
    """
    Screen ka pixel-grid ek regular micro-texture create karta hai.
    Simple local variance ka standard deviation (patch-wise) use karke
    texture ki 'regularity' measure karte hain — camera sensor noise
    zyada random hota hai, screen-grid zyada regular.
    """
    h, w = gray.shape
    patch = 16
    variances = []
    for y in range(0, h - patch, patch):
        for x in range(0, w - patch, patch):
            block = gray[y:y + patch, x:x + patch].astype(np.float32)
            variances.append(np.var(block))
    variances = np.array(variances)
    if len(variances) == 0:
        return 0.0
    # Low std of local variances = zyada regular/repetitive texture (screen-like)
    return float(np.std(variances) / (np.mean(variances) + 1e-6))


def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = _resize_keep_aspect(img, target=256)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- FFT-based moire/periodicity features (naya, sabse important) ---
    peak_mean, peak_std, peak_to_background = _fft_peak_features(gray)

    # --- Purane useful features (kept, but normalized where needed) ---
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    b, g, r = cv2.split(img)
    color_std = np.std([np.mean(b), np.mean(g), np.mean(r)])

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # pixel_std ko normalize kiya (mean brightness se divide) taaki ye
    # sirf "kitni bright/dark image hai" na batae, balki relative contrast batae
    mean_val = np.mean(gray) + 1e-6
    normalized_contrast = np.std(gray) / mean_val

    # --- Naye features ---
    color_banding = _color_banding_score(img)
    texture_regularity = _texture_regularity(gray)

    feats = [
        peak_mean, peak_std, peak_to_background,
        laplacian_var, color_std, edge_density,
        normalized_contrast, color_banding, texture_regularity,
    ]
    names = [
        "fft_peak_mean", "fft_peak_std", "fft_peak_to_background",
        "laplacian_var", "color_std", "edge_density",
        "normalized_contrast", "color_banding", "texture_regularity",
    ]
    return feats, names


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if clf is None:
        return jsonify({"error": "Model not loaded. Save model.pkl from your notebook first."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    temp_path = os.path.join("temp_upload", file.filename)
    os.makedirs("temp_upload", exist_ok=True)
    file.save(temp_path)

    try:
        feats, names = extract_features(temp_path)
        proba = clf.predict_proba([feats])[0]

        THRESHOLD = 0.5  # default threshold - gave the best overall accuracy (92.86%) after retraining
        prob_screen = proba[1]
        label = "Screen Image" if prob_screen >= THRESHOLD else "Original Image"
        confidence = float(max(proba)) * 100

        response = {
            "label": label,
            "confidence": round(confidence, 2),
            "prob_original": round(float(proba[0]) * 100, 2),
            "prob_screen": round(float(proba[1]) * 100, 2),
            "features": [{"name": n, "value": round(float(v), 2)} for n, v in zip(names, feats)]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
