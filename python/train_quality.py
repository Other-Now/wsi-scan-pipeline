"""Train the tile-quality classifier and export it to ONNX.

Four classes: sharp / blurred / folded / bubble.

The labels are free because the degradations are synthetic: take a patch of
tissue that is known to be clean, and either leave it alone, defocus it, fold
it, or drop a bubble on it. That is also the honest limitation -- the model is
trained on this generator's idea of a fold, and the number that matters is not
the held-out F1 but how many of the *scanner's* bad tiles it catches, which is
what wsi_scan reports against the slide's ground truth.

Patches are 128x128 at native resolution, which is what the scanner feeds the
model: a 512 px FOV is cut into a 4x4 grid. Downsampling first would throw away
exactly the high-frequency content the model is being asked to judge.

    python python/train_quality.py --slide data/slide.ppm --out models/
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

PATCH = 128
# "background" earns its place the hard way. Without it the first in-loop run
# called 182 of 688 scanned patches blurred and 147 of them bubbles: the model
# had only ever seen patches that were >=85% tissue, and a patch of empty glass
# is flat and bright, which is exactly what a defocused patch and the inside of
# a bubble look like. The scanner's real tiles are full of glass.
CLASSES = ["sharp", "blurred", "folded", "bubble", "background"]


def load_slide(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def tissue_mask(rgb, scale=32):
    """Otsu on the downsampled luminance -- the same rule the C++ uses."""
    small = np.asarray(Image.fromarray(rgb).resize(
        (rgb.shape[1] // scale, rgb.shape[0] // scale), Image.BOX).convert("L"))
    hist = np.bincount(small.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    idx = np.arange(256)
    w_b = np.cumsum(hist)
    w_f = total - w_b
    sum_all = (idx * hist).sum()
    sum_b = np.cumsum(idx * hist)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
    var[~np.isfinite(var)] = -1
    thresh = int(np.argmax(var))
    return small <= thresh, thresh, scale


def load_artifacts(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("artifacts", [])


def overlaps_artifact(x, y, size, artifacts, pad=64):
    for a in artifacts:
        if (x < a["x"] + a["w"] + pad and x + size > a["x"] - pad and
                y < a["y"] + a["h"] + pad and y + size > a["y"] - pad):
            return True
    return False


def sample_locations(rgb, mask, scale, artifacts, n, rng, min_tissue=0.85, max_tissue=1.01):
    """Patch origins whose tissue coverage is in [min_tissue, max_tissue),
    away from any real artefact so the synthetic label is the only one."""
    h, w = rgb.shape[:2]
    out = []
    tries = 0
    while len(out) < n and tries < n * 80:
        tries += 1
        x = int(rng.integers(0, w - PATCH))
        y = int(rng.integers(0, h - PATCH))
        mx0, my0 = x // scale, y // scale
        mx1, my1 = (x + PATCH) // scale, (y + PATCH) // scale
        block = mask[my0:my1 + 1, mx0:mx1 + 1]
        if block.size == 0 or not (min_tissue <= block.mean() < max_tissue):
            continue
        if overlaps_artifact(x, y, PATCH, artifacts):
            continue
        out.append((x, y))
    return out


def add_fold(patch, rng):
    """A dark, displaced band -- the same recipe the slide generator uses."""
    out = patch.astype(np.float32)
    angle = rng.uniform(-1.2, 1.2)
    half_w = rng.integers(14, 34)
    cx = rng.integers(24, PATCH - 24)
    cy = rng.integers(24, PATCH - 24)
    ys, xs = np.mgrid[0:PATCH, 0:PATCH]
    u = (xs - cx) * np.cos(angle) + (ys - cy) * np.sin(angle)
    band = np.abs(u) <= half_w
    shifted = np.roll(np.roll(out, 9, axis=0), 9, axis=1)
    out[band] = (0.5 * out[band] + 0.5 * shifted[band]) * rng.uniform(0.60, 0.75)
    return out


def add_bubble(patch, rng):
    out = patch.astype(np.float32)
    r = rng.integers(55, 150)
    cx = rng.integers(-20, PATCH + 20)
    cy = rng.integers(-20, PATCH + 20)
    ys, xs = np.mgrid[0:PATCH, 0:PATCH]
    d = np.hypot(xs - cx, ys - cy)
    inside = d <= r
    if inside.sum() < 0.25 * PATCH * PATCH:  # keep the artefact visible
        inside = d <= r + 40
    rim = inside & (d > r - 6)
    out[inside] = 0.25 * out[inside] + 0.75 * 250.0
    out[rim] *= 0.55
    return out


def blur(patch, sigma):
    if sigma <= 0.05:
        return patch.astype(np.float32)
    img = Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))).astype(np.float32)


def finish(patch, rng):
    """Gain jitter and read noise, so the model does not key on absolute level."""
    out = patch * rng.uniform(0.93, 1.07) + rng.normal(0, 2.0, patch.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_dataset(rgb, tissue_locs, background_locs, rng):
    """Returns patches, labels, and the x of each patch (for the spatial split)."""
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    xs, ys, pos = [], [], []

    def emit(patch, cls, x):
        xs.append(finish(patch, rng))
        ys.append(cls)
        pos.append(x)

    for (x, y) in tissue_locs:
        base = gray[y:y + PATCH, x:x + PATCH].astype(np.float32)
        emit(blur(base, rng.uniform(0.0, 0.4)), 0, x)
        emit(blur(base, rng.uniform(1.2, 4.0)), 1, x)
        emit(blur(add_fold(base, rng), rng.uniform(0.0, 0.6)), 2, x)
        emit(blur(add_bubble(base, rng), rng.uniform(0.0, 0.6)), 3, x)

    # Background gets the same defocus range as everything else: the scanner
    # does not stop acquiring just because a FOV is half glass, and "blurred
    # background" must land in the background class, not the blurred one.
    for (x, y) in background_locs:
        base = gray[y:y + PATCH, x:x + PATCH].astype(np.float32)
        emit(blur(base, rng.uniform(0.0, 4.0)), 4, x)

    return np.stack(xs), np.array(ys), np.array(pos)


class TileNet(nn.Module):
    """Four stride-2 conv blocks, global average pool, linear head.

    GroupNorm, not BatchNorm, and that is not a style choice. With BatchNorm the
    first version of this model scored 0.94 accuracy while it was training and
    0.39 in eval mode: normalising over the batch let it learn "this patch is
    brighter than the others in this batch", which is a real signal during
    training and does not exist at all when the scanner hands it one tile.
    GroupNorm normalises within each sample, so training and inference see the
    same function -- which is the only version that survives an ONNX export.
    """

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, cout), cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.mean(dim=(2, 3))
        return self.head(h)


def f1_report(y_true, y_pred, n_classes=len(CLASSES)):
    out = {}
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    for c in range(n_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[CLASSES[c]] = {"precision": round(prec, 4), "recall": round(rec, 4),
                           "f1": round(f1, 4), "support": int(conf[c].sum())}
    out["macro_f1"] = round(float(np.mean([out[c]["f1"] for c in CLASSES])), 4)
    out["accuracy"] = round(float(np.trace(conf) / conf.sum()), 4)
    out["confusion"] = conf.tolist()
    return out


def throughput(model, device, batch=256, iters=20):
    model = model.to(device).eval()
    x = torch.randn(batch, 1, PATCH, PATCH, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return batch * iters / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="data/slide.ppm")
    ap.add_argument("--truth", default="data/slide_truth.json")
    ap.add_argument("--out", default="models")
    ap.add_argument("--locations", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = load_slide(args.slide)
    mask, thresh, scale = tissue_mask(rgb)
    artifacts = load_artifacts(args.truth)
    print(f"slide {rgb.shape[1]}x{rgb.shape[0]}, otsu={thresh}, "
          f"tissue={mask.mean() * 100:.1f}%, {len(artifacts)} known artefacts")

    locs = sample_locations(rgb, mask, scale, artifacts, args.locations, rng)
    # Background: anything from bare glass to a patch straddling a section edge,
    # which is what the scanner sees at the rim of every tissue region.
    bg_locs = sample_locations(rgb, mask, scale, artifacts, args.locations, rng,
                               min_tissue=0.0, max_tissue=0.35)
    print(f"sampled {len(locs)} tissue and {len(bg_locs)} background locations")
    t0 = time.perf_counter()
    X, y, pos = build_dataset(rgb, locs, bg_locs, rng)
    print(f"built {len(X)} patches in {time.perf_counter() - t0:.1f}s")

    # Split by position, not at random: patches from the same location differ
    # only by degradation, so a random split would leak the same tissue into
    # both sides and report an F1 that means nothing.
    split_x = np.quantile(pos, 0.75)
    is_test = pos > split_x
    Xtr, ytr = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]
    print(f"train {len(Xtr)}, test {len(Xte)} (split at x={split_x:.0f})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TileNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr).float().div_(255).unsqueeze(1)
    ytr_t = torch.from_numpy(ytr).long()
    Xte_t = torch.from_numpy(Xte).float().div_(255).unsqueeze(1)

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total / len(perm):.4f}")
    train_s = time.perf_counter() - t0

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte_t), 256):
            preds.append(model(Xte_t[i:i + 256].to(device)).argmax(1).cpu().numpy())
    report = f1_report(yte, np.concatenate(preds))
    print(json.dumps(report, indent=2))

    tp_gpu = throughput(model, device) if device.type == "cuda" else None
    tp_cpu = throughput(model, torch.device("cpu"))
    print(f"torch throughput: cpu {tp_cpu:.0f} tiles/s" +
          (f", cuda {tp_gpu:.0f} tiles/s" if tp_gpu else ""))

    onnx_path = out_dir / "tile_quality.onnx"
    model_cpu = model.to("cpu").eval()
    torch.onnx.export(
        model_cpu,
        torch.randn(1, 1, PATCH, PATCH),
        str(onnx_path),
        input_names=["tile"],
        output_names=["logits"],
        dynamic_axes={"tile": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")

    metrics = {
        "patch": PATCH,
        "classes": CLASSES,
        "params": int(n_params),
        "train_patches": int(len(Xtr)),
        "test_patches": int(len(Xte)),
        "epochs": args.epochs,
        "train_seconds": round(train_s, 1),
        "device": str(device),
        "held_out": report,
        "throughput_tiles_per_s": {
            "torch_cpu": round(tp_cpu, 1),
            "torch_cuda": round(tp_gpu, 1) if tp_gpu else None,
        },
    }
    (out_dir / "tile_quality_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir / 'tile_quality_metrics.json'}")


if __name__ == "__main__":
    main()
