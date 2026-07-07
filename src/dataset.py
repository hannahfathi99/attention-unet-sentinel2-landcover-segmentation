"""
dataset script
"""

# Extract maps
ndvi_map  = img_with_indices[..., NAME2IDX["NDVI"]]
ndwi_map  = img_with_indices[..., NAME2IDX["NDWI"]]
mndwi_map = img_with_indices[..., NAME2IDX["MNDWI"]] if "MNDWI" in NAME2IDX else None
awei_map  = img_with_indices[..., NAME2IDX["AWEI_sh"]] if "AWEI_sh" in NAME2IDX else None

# Build RGB preview with CLAHE
rgb = img_with_indices[..., [NAME2IDX["B4"], NAME2IDX["B3"], NAME2IDX["B2"]]].copy()
rgb = np.clip(rgb, 0, 10000) / 10000.0
rgb_disp = exposure.equalize_adapthist(rgb, clip_limit=0.02)

# Vegetation mask from NDVI
finite_ndvi = np.isfinite(ndvi_map)
ndvi_clean = np.zeros_like(ndvi_map, dtype=np.float32)
ndvi_clean[finite_ndvi] = ndvi_map[finite_ndvi]
t_ndvi = threshold_otsu(ndvi_clean[finite_ndvi])
veg_mask = ndvi_clean > t_ndvi
veg_mask = binary_opening(veg_mask, footprint=disk(1))
veg_mask = binary_closing(veg_mask, footprint=disk(1))

# Water mask from MNDWI (preferred) or NDWI
finite_ndwi = np.isfinite(ndwi_map)
ndwi_clean = np.zeros_like(ndwi_map, dtype=np.float32)
ndwi_clean[finite_ndwi] = ndwi_map[finite_ndwi]

if mndwi_map is not None:
    finite_mndwi = np.isfinite(mndwi_map)
    mndwi_clean = np.zeros_like(mndwi_map, dtype=np.float32)
    mndwi_clean[finite_mndwi] = mndwi_map[finite_mndwi]
    t_mndwi = threshold_otsu(mndwi_clean[finite_mndwi])
    water_mask_base = mndwi_clean > max(0.1, t_mndwi)
else:
    t_ndwi = threshold_otsu(ndwi_clean[finite_ndwi])
    water_mask_base = ndwi_clean > max(0.1, t_ndwi)

# Add water in shadow using AWEI if available
if awei_map is not None:
    finite_awei = np.isfinite(awei_map)
    awei_clean = np.zeros_like(awei_map, dtype=np.float32)
    awei_clean[finite_awei] = awei_map[finite_awei]
    shadow_water = awei_clean > 0
    water_mask_base = water_mask_base | shadow_water

# Final water mask (remove vegetation)
water_mask = water_mask_base & (~veg_mask)
water_mask = binary_opening(water_mask, footprint=disk(1))
water_mask = binary_closing(water_mask, footprint=disk(1))

# Soil mask
soil_mask = finite_ndvi & (~veg_mask) & (~water_mask)

# Build label map
label_map = np.full(ndvi_map.shape, fill_value=-1, dtype=np.int16)
label_map[soil_mask]  = SOIL
label_map[veg_mask]   = VEG
label_map[water_mask] = WATER

# Show preview figure
plt.figure(figsize=(16,6))
plt.subplot(1,3,1); plt.imshow(rgb_disp); plt.title("Sentinel-2 RGB"); plt.axis("off")
plt.subplot(1,3,2); im=plt.imshow(ndvi_clean, cmap="YlGn", vmin=-1, vmax=1)
plt.title(f"NDVI (Otsu>{t_ndvi:.2f})"); plt.axis("off"); plt.colorbar(im, fraction=0.046)
plt.subplot(1,3,3)
preview_rgb = np.zeros((H, W, 3), dtype=np.uint8)
for cls_id, color in PALETTE_RGB.items():
    preview_rgb[label_map == cls_id] = color
plt.imshow(preview_rgb); plt.title("Auto-label preview"); plt.axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_rgb_ndvi_labels.png", dpi=200)
plt.show()

print(f"[INFO] NDVI Otsu threshold: {t_ndvi:.3f}")
if mndwi_map is not None:
    print(f"[INFO] MNDWI threshold: {t_mndwi:.3f}")
print(f"[INFO] Class pixel counts: soil={np.sum(label_map==SOIL)}, veg={np.sum(label_map==VEG)}, water={np.sum(label_map==WATER)}")

_ = (label_map,)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.morphology import opening, closing, square, remove_small_objects, remove_small_holes
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
from skimage.filters import threshold_otsu

assert 'img_with_indices' in globals(), "[ERROR] Run previous cells first."

H, W, _ = img_with_indices.shape
ndvi_full = np.nan_to_num(img_with_indices[..., NAME2IDX["NDVI"]], nan=0.0).astype(np.float32)
ndwi_full = np.nan_to_num(img_with_indices[..., NAME2IDX["NDWI"]], nan=0.0).astype(np.float32)
mndwi_full = np.nan_to_num(img_with_indices[..., NAME2IDX.get("MNDWI", 0)], nan=0.0).astype(np.float32) if "MNDWI" in NAME2IDX else None
awei_full  = np.nan_to_num(img_with_indices[..., NAME2IDX.get("AWEI_sh", 0)], nan=0.0).astype(np.float32) if "AWEI_sh" in NAME2IDX else None
has_evi   = "EVI" in NAME2IDX
evi_full  = np.nan_to_num(img_with_indices[..., NAME2IDX.get("EVI", 0)], nan=0.0).astype(np.float32) if has_evi else None

# Parameters
BLOCK = 1024
STEP  = 512
W_RULE = 2.5  # weight for rule-based
W_KM   = 1.0
MIN_OBJ = 50

# Clean binary mask
def clean_mask(m, min_size=MIN_OBJ):
    m = opening(m, square(3))
    m = closing(m, square(3))
    m = binary_fill_holes(m)
    m = remove_small_objects(m, min_size)
    return m

# Initial cluster centers for KMeans
def seed_centers(has_evi=True):
    if has_evi:
        return np.array([
            [0.60, -0.10, 0.50],  # vegetation
            [-0.10, 0.50, -0.05], # water
            [0.10, -0.05, 0.05],  # soil
        ], dtype=np.float32)
    else:
        return np.array([
            [0.60, -0.10],  # vegetation
            [-0.10, 0.50],  # water
            [0.10, -0.05],  # soil
        ], dtype=np.float32)

# 1) Rule-based labels for full image
finite_ndvi = np.isfinite(ndvi_full)
t_v_global = threshold_otsu(ndvi_full[finite_ndvi]) if finite_ndvi.any() else 0.3
veg_rb_full = clean_mask(ndvi_full > t_v_global, MIN_OBJ)

if mndwi_full is not None:
    water_base = (mndwi_full > 0.1)
else:
    water_base = (ndwi_full > 0.1)

if awei_full is not None:
    water_base = water_base | (awei_full > 0)

blue_full = img_with_indices[..., NAME2IDX["B2"]]
nir_full  = img_with_indices[..., NAME2IDX["B8"]]
water_rb_full = clean_mask(water_base & (blue_full > nir_full), MIN_OBJ)

veg_rb_full[water_rb_full] = False
soil_rb_full = ~(veg_rb_full | water_rb_full)

label_rule = np.full((H,W), SOIL, np.uint8)
label_rule[veg_rb_full]   = VEG
label_rule[water_rb_full] = WATER

# 2) Fusion with KMeans (sliding windows)
scores = np.zeros((3, H, W), dtype=np.float32)
for ys in range(0, H, STEP):
    for xs in range(0, W, STEP):
        ye, xe = min(ys+BLOCK, H), min(xs+BLOCK, W)
        ndvi = ndvi_full[ys:ye, xs:xe]
        ndwi = ndwi_full[ys:ye, xs:xe]
        evi  = evi_full[ys:ye, xs:xe] if has_evi else None

        # Local rule-based masks
        t_v = threshold_otsu(ndvi) if np.unique(ndvi).size > 1 else t_v_global
        veg_rb = clean_mask(ndvi > t_v, MIN_OBJ)

        if mndwi_full is not None:
            water_loc = (mndwi_full[ys:ye, xs:xe] > 0.1)
        else:
            water_loc = (ndwi > 0.1)

        if awei_full is not None:
            water_loc = water_loc | (awei_full[ys:ye, xs:xe] > 0)

        water_rb = clean_mask(water_loc & (img_with_indices[ys:ye, xs:xe, NAME2IDX["B2"]] >
                                           img_with_indices[ys:ye, xs:xe, NAME2IDX["B8"]]), MIN_OBJ)
        veg_rb[water_rb] = False
        soil_rb = ~(veg_rb | water_rb)

        # KMeans clustering
        feats = [ndvi.ravel(), ndwi.ravel()]
        if has_evi: feats.append(evi.ravel())
        X = np.stack(feats, axis=1)
        init = seed_centers(has_evi)
        km = KMeans(n_clusters=3, init=init, n_init=1, random_state=0, max_iter=200)
        if X.shape[0] > 150_000:
            idx = np.random.RandomState(0).choice(X.shape[0], size=150_000, replace=False)
            km.fit(X[idx])
        else:
            km.fit(X)
        klabels = km.predict(X).reshape(ndvi.shape)

        # Map clusters to classes
        means = []
        for k in range(3):
            m = (klabels == k)
            mu_ndvi = float(np.mean(ndvi[m])) if m.any() else -1.0
            mu_ndwi = float(np.mean(ndwi[m])) if m.any() else -1.0
            means.append((k, mu_ndvi, mu_ndwi))
        k_veg   = sorted(means, key=lambda t: t[1], reverse=True)[0][0]
        k_water = sorted(means, key=lambda t: t[2], reverse=True)[0][0]
        rest = ({0,1,2} - {k_veg, k_water})
        k_soil = rest.pop() if len(rest) == 1 else sorted(means, key=lambda t: t[1])[0][0]

        lbl_km = np.full(ndvi.shape, SOIL, dtype=np.uint8)
        lbl_km[klabels == k_veg]   = VEG
        lbl_km[klabels == k_water] = WATER

        onehot_rb = np.stack([soil_rb, veg_rb, water_rb], 0).astype(np.float32)
        onehot_km = np.stack([lbl_km == SOIL, lbl_km == VEG, lbl_km == WATER], 0).astype(np.float32)
        scores[:, ys:ye, xs:xe] += (W_RULE * onehot_rb + W_KM * onehot_km)

label_fused = scores.argmax(0).astype(np.uint8)

# 3) Clean final masks
for cls in (VEG, WATER):
    m = (label_fused == cls)
    m = opening(m, square(3))
    m = closing(m, square(3))
    m = remove_small_holes(m, 64)
    m = remove_small_objects(m, MIN_OBJ)
    label_fused[(label_fused == cls) & (~m)] = SOIL
    label_fused[m] = cls

# 4) Metrics vs rule-based
y_true = label_rule.ravel()
y_pred = label_fused.ravel()

metrics = {}
for cls_id, cls_name in enumerate(CLASS_NAMES):
    cls_prec = precision_score(y_true == cls_id, y_pred == cls_id, zero_division=0)
    cls_rec  = recall_score(y_true == cls_id, y_pred == cls_id, zero_division=0)
    cls_f1   = f1_score(y_true == cls_id, y_pred == cls_id, zero_division=0)
    cls_iou  = jaccard_score(y_true == cls_id, y_pred == cls_id, zero_division=0)
    metrics[cls_name] = {"Precision": cls_prec, "Recall": cls_rec, "F1": cls_f1, "IoU": cls_iou}

cm = confusion_matrix(y_true, y_pred, labels=[SOIL, VEG, WATER])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Pred (Fusion)"); plt.ylabel("GT (Rule)"); plt.title("Fusion vs Rule")
plt.tight_layout()
plt.savefig(FIG_DIR / "cm_fusion_vs_rule.png", dpi=220)
plt.show()

print("[INFO] Fused labeling complete.")
print("[INFO] Baseline comparison (Fusion vs Rule-only):")
for cls, vals in metrics.items():
    print(f"  {cls:>11s} | " + " | ".join([f"{k}={v:.3f}" for k, v in vals.items()]))

label_map = label_fused.copy()


import random, pathlib, numpy as np
from skimage.morphology import dilation, square
import json


# Params

PATCH_SIZE = 256              # patch size (px)
STRIDE_BG  = 256              # stride for non-water
STRIDE_WAT = PATCH_SIZE // 2  # stride for water (oversample)
MAX_NODATA_RATIO = 0.5        # skip if >50% NoData
FG_MIN_RATIO     = 0.02       # skip low-foreground unless water
WATER_MIN_RATIO  = 0.002      # keep if water >0.2%
WATER_MARGIN_PX  = 32         # dilate water mask (px)
RANDOM_SEED      = 42         # reproducibility
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)

FG_SKIP_PROB = 0.5            # random skip prob for low-FG

H, W = label_map.shape


# Build NoData mask

nodata_val = meta.get("nodata")
if nodata_val is not None:
    nodata_mask = np.any(
        (img_hwc[..., :ORIG_BANDS] == nodata_val) | np.isnan(img_hwc[..., :ORIG_BANDS]),
        axis=-1
    )
else:
    nodata_mask = np.isnan(img_hwc[..., :ORIG_BANDS]).any(axis=-1)


# Crop to multiples of PATCH_SIZE

new_h = (H // PATCH_SIZE) * PATCH_SIZE
new_w = (W // PATCH_SIZE) * PATCH_SIZE
X_full = img_with_indices[:new_h, :new_w, :]
Y_full = label_map[:new_h, :new_w]
N_full = nodata_mask[:new_h, :new_w]


# Water mask (+ margin)

water_mask = (Y_full == WATER)
if WATER_MARGIN_PX > 0:
    # use odd-sized square structuring element
    water_mask = dilation(water_mask, square(2 * (WATER_MARGIN_PX // 2) + 1))


# Adaptive tile generator
# pass1: dense on water; pass2: sparser elsewhere

def gen_tiles_adaptive(h, w, size, stride_bg, stride_wat, water_mask):
    # pass 1: water areas
    for y in range(0, h - size + 1, stride_wat):
        for x in range(0, w - size + 1, stride_wat):
            if water_mask[y:y+size, x:x+size].any():
                yield x, y
    # pass 2: other areas
    for y in range(0, h - size + 1, stride_bg):
        for x in range(0, w - size + 1, stride_bg):
            if water_mask[y:y+size, x:x+size].any():
                continue
            yield x, y


# Save patches

(PATCH_ROOT/"X").mkdir(parents=True, exist_ok=True)
(PATCH_ROOT/"Y").mkdir(parents=True, exist_ok=True)
uids, class_pix = [], {SOIL: 0, VEG: 0, WATER: 0}

def save_patch(x, y):
    # slice windows
    x2, y2 = x + PATCH_SIZE, y + PATCH_SIZE
    tx = X_full[y:y2, x:x2, :]
    ty = Y_full[y:y2, x:x2]
    tn = N_full[y:y2, x:x2]

    # skip high NoData
    if tn.mean() > MAX_NODATA_RATIO:
        return False

    # skip if all unlabeled
    valid = (ty >= 0)
    if not valid.any():
        return False

    # treat unlabeled as background (SOIL) for stats/ratios
    ty_eff = np.where(valid, ty, SOIL)

    # ratios for filtering
    fg_ratio    = (ty_eff > 0).mean()
    water_ratio = (ty_eff == WATER).mean()

    # optionally skip low-FG low-water tiles
    if water_ratio < WATER_MIN_RATIO and fg_ratio < FG_MIN_RATIO and random.random() < FG_SKIP_PROB:
        return False

    # persist arrays
    uid = f"y{y:05d}_x{x:05d}_ps{PATCH_SIZE}"
    np.save(PATCH_ROOT/"X"/f"{uid}.npy", tx.astype(np.float32))
    np.save(PATCH_ROOT/"Y"/f"{uid}.npy", ty.astype(np.uint8))
    uids.append(uid)

    # update per-class pixel counts (only valid pixels)
    u, cnts = np.unique(ty[valid], return_counts=True)
    for k, v in zip(u.tolist(), cnts.tolist()):
        class_pix[int(k)] = class_pix.get(int(k), 0) + int(v)
    return True

saved = 0
for x, y in gen_tiles_adaptive(new_h, new_w, PATCH_SIZE, STRIDE_BG, STRIDE_WAT, water_mask):
    if save_patch(x, y):
        saved += 1
print(f"[INFO] Patches saved: {saved}")


# Split into Train/Val/Test by fixed ratio (60/20/20)

def parse_uid(uid):
    # uid format: y{yyyyy}_x{xxxxx}_ps{patch}
    parts = uid.split("_")
    y = int(parts[0][1:])
    x = int(parts[1][1:])
    return y, x

random.shuffle(uids)
n_total = len(uids)
n_train = int(0.6 * n_total)
n_val   = int(0.2 * n_total)
n_test  = n_total - n_train - n_val  # guard against rounding

TRAIN_UIDS = uids[:n_train]
VAL_UIDS   = uids[n_train:n_train+n_val]
TEST_UIDS  = uids[n_train+n_val:]


# Ensure each split has at least one water patch
def contains_water(uid_list):
    for uid in uid_list:
        y, x = parse_uid(uid)
        if (Y_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE] == WATER).any():
            return True
    return False

def steal_one_water(from_list, to_list):
    # move first water-containing uid from 'from_list' to 'to_list'
    for i, uid in enumerate(from_list):
        y, x = parse_uid(uid)
        if (Y_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE] == WATER).any():
            to_list.append(uid)
            from_list.pop(i)
            return True
    return False

splits = [("TRAIN", TRAIN_UIDS), ("VAL", VAL_UIDS), ("TEST", TEST_UIDS)]
for name, lst in splits:
    if not contains_water(lst) and contains_water(TRAIN_UIDS) and name != "TRAIN":
        steal_one_water(TRAIN_UIDS, lst)

print(f"[INFO] Train: {len(TRAIN_UIDS)} | Val: {len(VAL_UIDS)} | Test: {len(TEST_UIDS)}")

print(f"[INFO] Train: {len(TRAIN_UIDS)} | Val: {len(VAL_UIDS)} | Test: {len(TEST_UIDS)}")

# print ratios 60/20/20 style
n_total = len(TRAIN_UIDS) + len(VAL_UIDS) + len(TEST_UIDS)

def print_ratio(name, n):
    frac = n / n_total
    pct = round(frac * 100)
    print(f"{name}: {n} → {n} ÷ {n_total} = {frac:.1f} = {pct}%")

print_ratio("Train", len(TRAIN_UIDS))
print_ratio("Val  ", len(VAL_UIDS))
print_ratio("Test ", len(TEST_UIDS))



# Save manifest JSON

manifest = {
    "patch_root": str(PATCH_ROOT),
    "patch_size": PATCH_SIZE,
    "stride_bg": STRIDE_BG,
    "stride_wat": STRIDE_WAT,
    "channels": X_full.shape[-1],
    "classes": {"soil": SOIL, "vegetation": VEG, "water": WATER},
    "train": TRAIN_UIDS,
    "val": VAL_UIDS,
    "test": TEST_UIDS,
    "class_pixel_ratio": {
        int(k): {"count": int(v), "pct": 100.0 * float(v) / float(new_h * new_w)}
        for k, v in class_pix.items()
    },
    "band_stats": BAND_STATS,
    "channel_names": names,
}
with open(PATCH_ROOT/"splits.json", "w") as f:
    json.dump(manifest, f, indent=2)

# Water presence stats per split
def split_stats(uids_):
    has_w = sum(
        (Y_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE] == WATER).any()
        for uid in uids_
        for y, x in [parse_uid(uid)]
    )
    return {"total": len(uids_), "with_water": has_w}

print("[INFO] Split water presence:",
      f"\n  TRAIN: {split_stats(TRAIN_UIDS)}",
      f"\n  VAL  : {split_stats(VAL_UIDS)}",
      f"\n  TEST : {split_stats(TEST_UIDS)}")


import importlib.metadata, sys, json, pathlib, random, math, collections
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler

# Ensure specific package versions if needed
def ensure_pkg(pkg, ver):
    try:
        cur = importlib.metadata.version(pkg)
        if cur != ver:
            print(f"[INFO] Updating {pkg} {cur} -> {ver}")
            get_ipython().system(f"pip install -q {pkg}=={ver}")
    except importlib.metadata.PackageNotFoundError:
        print(f"[INFO] Installing {pkg}=={ver}")
        get_ipython().system(f"pip install -q {pkg}=={ver}")

ensure_pkg("albumentations", "1.3.1")
ensure_pkg("opencv-python-headless", "4.10.0.84")

import albumentations as A
import cv2

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load manifest and metadata
with open(PATCH_ROOT/"splits.json", "r") as f:
    MF = json.load(f)

PATCH_ROOT_STR = MF["patch_root"]

# --- Define raw vs derived channels (edit to your real set & order) ---
RAW_BANDS = MF.get("raw_band_names", ["B2","B3","B4","B8","B11","B12"])  # only sensor bands stored in X[..., :ORIG_BANDS]
DERIVED_NAMES = MF.get("derived_names", ["NDVI","NDWI","MNDWI","EVI","AWEI_sh"])  # indices/features

# SELECTED_BANDS should be raw only
SELECTED_BANDS = list(RAW_BANDS)
ORIG_BANDS = len(SELECTED_BANDS)

# band_stats must match raw bands (used for per-band scaling)
BAND_STATS = MF["band_stats"]
if len(BAND_STATS) != ORIG_BANDS:
    # keep code robust if MF contains more stats than raw bands
    BAND_STATS = BAND_STATS[:ORIG_BANDS]
assert len(BAND_STATS) == ORIG_BANDS, "band_stats must match number of raw bands"

# Initial channel names = raw + derived (final fix happens after first batch via sanitize block)
CHANNEL_NAMES = RAW_BANDS + DERIVED_NAMES

# Augmentations (mild and RS-friendly)
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10, rotate_limit=15,
                       p=0.30, border_mode=cv2.BORDER_REFLECT_101),
    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.25),
    A.GaussNoise(var_limit=(0.0, 0.002), p=0.20),
], p=1.0)

val_tf = None  # no augmentation for val/test

# Dataset for (H,W,C) numpy patches → (C,H,W) tensors
class PatchDataset(Dataset):
    """
    Loads patches and returns (x, y) where:
    - First ORIG_BANDS are scaled to [0,1] by p2–p98.
    - Indices (NDVI/NDWI/EVI) are kept as-is.
    """
    def __init__(self, uids, root_dir, orig_bands, band_stats, transform=None, mmap: bool=True):
        self.uids       = list(uids) if uids is not None else []
        self.root       = Path(root_dir)
        self.orig_bands = int(orig_bands)
        self.band_stats = list(band_stats)
        self.transform  = transform
        self.mmap       = mmap

    def __len__(self):
        return len(self.uids)

    def _load_pair(self, uid):
        x = np.load(self.root/"X"/f"{uid}.npy", mmap_mode='r' if self.mmap else None)
        y = np.load(self.root/"Y"/f"{uid}.npy", mmap_mode='r' if self.mmap else None).astype(np.uint8)
        return np.array(x), np.array(y)

    def _scale_orig_bands(self, x):
        x = x.copy()
        for c, (lo, hi) in enumerate(self.band_stats[:self.orig_bands]):
            denom = (hi - lo) if (hi - lo) > 1e-6 else 1.0
            x[..., c] = np.clip((x[..., c] - lo) / denom, 0.0, 1.0)
        return x

    def __getitem__(self, idx):
        uid = self.uids[idx]
        x, y = self._load_pair(uid)
        x = np.nan_to_num(x, nan=0.0)
        x_scaled = self._scale_orig_bands(x)
        if self.transform is not None:
            aug = self.transform(image=x_scaled, mask=y)
            x_scaled, y = aug["image"], aug["mask"]
        x_t = torch.from_numpy(np.transpose(x_scaled, (2, 0, 1))).float()
        y_t = torch.from_numpy(y.astype(np.int64))
        return x_t, y_t

# Build splits (with fallback if train is empty)
train_uids = MF.get("train", []) or []
val_uids   = MF.get("val", []) or []
test_uids  = MF.get("test", []) or []

if len(train_uids) == 0:
    pool = (val_uids or []) + (test_uids or [])
    if len(pool) > 0:
        print("[FALLBACK] No training patches; sampling from val/test.")
        random.shuffle(pool)
        take = max(1, int(0.5 * len(pool)))
        train_uids = pool[:take]
        MF["train"] = train_uids
    else:
        print("[ERROR] No patches available. Revisit patching thresholds.")

train_ds = PatchDataset(train_uids, PATCH_ROOT_STR, ORIG_BANDS, BAND_STATS, transform=train_tf) if len(train_uids)>0 else None
val_ds   = PatchDataset(val_uids,   PATCH_ROOT_STR, ORIG_BANDS, BAND_STATS, transform=val_tf)   if len(val_uids)>0   else None
test_ds  = PatchDataset(test_uids,  PATCH_ROOT_STR, ORIG_BANDS, BAND_STATS, transform=val_tf)  if len(test_uids)>0  else None

# Index patches containing a target class (e.g., water)
def list_indices_with_class(uids, cls_id=WATER):
    idx_pos, idx_neg = [], []
    yroot = Path(MF["patch_root"]) / "Y"
    for i, uid in enumerate(uids):
        y = np.load(yroot / f"{uid}.npy")
        (idx_pos if (y == cls_id).any() else idx_neg).append(i)
    return idx_pos, idx_neg

# Batch sampler that enforces a fraction of water patches
class BalancedWaterBatchSampler(BatchSampler):
    def __init__(self, idx_pos, idx_neg, batch_size, water_frac=0.5, epoch_batches=None, seed=42):
        assert batch_size >= 1
        self.idx_pos = list(idx_pos)
        self.idx_neg = list(idx_neg)
        self.bs      = int(batch_size)
        self.water_k = int(max(1, round(self.bs * float(water_frac))))
        self.other_k = self.bs - self.water_k
        self.epoch_batches = int(epoch_batches) if epoch_batches is not None else None
        self.rng = random.Random(seed)

    def __iter__(self):
        self.rng.shuffle(self.idx_pos)
        self.rng.shuffle(self.idx_neg)
        if self.epoch_batches is None:
            n = len(self.idx_pos) + len(self.idx_neg)
            self.epoch_batches = max(1, math.ceil(n / self.bs))
        pos_ptr, neg_ptr = 0, 0
        for _ in range(self.epoch_batches):
            batch = []
            for _ in range(self.water_k):
                if pos_ptr >= len(self.idx_pos):
                    if len(self.idx_pos) == 0: break
                    pos_ptr = 0; self.rng.shuffle(self.idx_pos)
                batch.append(self.idx_pos[pos_ptr] if len(self.idx_pos)>0 else None)
                pos_ptr += 1
            while len(batch) < self.bs:
                if neg_ptr >= len(self.idx_neg):
                    if len(self.idx_neg) == 0: break
                    neg_ptr = 0; self.rng.shuffle(self.idx_neg)
                batch.append(self.idx_neg[neg_ptr] if len(self.idx_neg)>0 else None)
                neg_ptr += 1
            batch = [b for b in batch if b is not None]
            if not batch: continue
            yield batch

    def __len__(self):
        if self.epoch_batches is not None:
            return self.epoch_batches
        n = len(self.idx_pos) + len(self.idx_neg)
        return max(1, math.ceil(n / self.bs))

# Build loaders with adaptive sampling
def build_loader_with_samplers(train_ds, val_ds, test_ds,
                               activate_balanced_if_frac_lt=0.20,
                               water_target_frac=0.50,
                               default_bs=8):
    num_workers  = 2 if torch.cuda.is_available() else 0
    pin_mem      = torch.cuda.is_available()
    persist_flag = num_workers > 0

    def safe_bs(n_items, default_bs=8):
        return max(1, min(default_bs, int(n_items))) if n_items and n_items > 0 else 1

    train_loader = []
    if (train_ds is not None) and (len(train_ds) > 0):
        idx_pos, idx_neg = list_indices_with_class(train_uids, WATER)
        water_frac = (len(idx_pos) / max(1, (len(idx_pos)+len(idx_neg))))
        bs = safe_bs(len(train_ds), default_bs=default_bs)

        if (len(idx_pos) > 0) and (water_frac < activate_balanced_if_frac_lt) and (bs >= 2):
            print(f"[SAMPLER] BalancedWaterBatchSampler | water_frac={water_frac:.3f} < {activate_balanced_if_frac_lt}")
            epoch_batches = math.ceil(len(train_uids) / bs)
            batch_sampler = BalancedWaterBatchSampler(idx_pos, idx_neg, batch_size=bs,
                                                      water_frac=water_target_frac,
                                                      epoch_batches=epoch_batches, seed=RANDOM_SEED)
            train_loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                                      num_workers=num_workers, pin_memory=pin_mem,
                                      persistent_workers=persist_flag)
        else:
            if (len(idx_pos) > 0) and (water_frac < 0.35):
                print(f"[SAMPLER] WeightedRandomSampler | water_frac={water_frac:.3f}")
                weights = np.ones(len(train_uids), dtype=np.float32)
                for i in idx_pos: weights[i] = 3.0
                from torch.utils.data import WeightedRandomSampler
                sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_uids), replacement=True)
                train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, drop_last=False,
                                          num_workers=num_workers, pin_memory=pin_mem, persistent_workers=persist_flag)
            else:
                print(f"[SAMPLER] Plain shuffle | water_frac={water_frac:.3f}")
                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False,
                                          num_workers=num_workers, pin_memory=pin_mem, persistent_workers=persist_flag)
    else:
        print("[WARN] Train dataset is empty.")

    val_bs  = safe_bs(len(val_ds)  if val_ds  is not None else 0, default_bs=default_bs)
    test_bs = safe_bs(len(test_ds) if test_ds is not None else 0, default_bs=default_bs)

    val_loader = (DataLoader(val_ds, batch_size=val_bs, shuffle=False, drop_last=False,
                             num_workers=num_workers, pin_memory=pin_mem, persistent_workers=persist_flag)
                  if (val_ds is not None and len(val_ds)>0) else [])

    test_loader = (DataLoader(test_ds, batch_size=test_bs, shuffle=False, drop_last=False,
                              num_workers=num_workers, pin_memory=pin_mem, persistent_workers=persist_flag)
                   if (test_ds is not None and len(test_ds)>0) else [])

    return train_loader, val_loader, test_loader

# Create data loaders
train_loader, val_loader, test_loader = build_loader_with_samplers(
    train_ds, val_ds, test_ds,
    activate_balanced_if_frac_lt=0.20,
    water_target_frac=0.50,
    default_bs=8
)

# Attach dataset metadata (bands / ROI / AOI) after loaders are built
# Short, defensive, and explicit so logs prove Iran ROI training.

# Bands list (fallback covers 16 channels: 11 bands + 5 indices)
try:
    bands_list = None
    if 'CHANNEL_NAMES' in globals():
        bands_list = CHANNEL_NAMES
    elif isinstance(MF.get('bands', None), list):
        bands_list = MF['bands']

    if bands_list is None:
        # Fallback example: tweak as needed to match your pipeline order
        bands_list = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12","SCL","NDVI","NDWI","NDBI","NDMI","DEM"]

    if train_ds is not None: train_ds.bands = bands_list
    if val_ds   is not None: val_ds.bands   = bands_list
    if test_ds  is not None: test_ds.bands  = bands_list
except Exception as e:
    print("[WARN] Could not set bands:", e)

# ROI names
try:
    roi_train = MF.get("roi_name_train", MF.get("roi_name", "Iran_ROI_Train"))
    roi_val   = MF.get("roi_name_val",   MF.get("roi_name", "Iran_ROI_Val"))
    roi_test  = MF.get("roi_name_test",  MF.get("roi_name", "Iran_ROI_Test"))
    if train_ds is not None: train_ds.roi_name = roi_train
    if val_ds   is not None: val_ds.roi_name   = roi_val
    if test_ds  is not None: test_ds.roi_name  = roi_test
except Exception as e:
    print("[WARN] Could not set ROI names:", e)

# AOI bounds (if present in MF)
try:
    if "aoi_bounds_train" in MF and train_ds is not None:
        train_ds.aoi_bounds = MF["aoi_bounds_train"]
    if "aoi_bounds_val" in MF and val_ds is not None:
        val_ds.aoi_bounds = MF["aoi_bounds_val"]
    if "aoi_bounds_test" in MF and test_ds is not None:
        test_ds.aoi_bounds = MF["aoi_bounds_test"]
except Exception as e:
    print("[WARN] Could not set AOI bounds:", e)

# DATASET/ROI AUDIT
try:
    print("Train samples:", len(train_ds) if train_ds is not None else 0,
          "Val samples:", len(val_ds) if val_ds is not None else 0)
    print("Bands:", getattr(train_ds, "bands", getattr(train_ds, "CHANNEL_NAMES", "unknown")) if train_ds is not None else "unknown")
    print("ROI train:", getattr(train_ds, "roi_name", getattr(train_ds, "aoi_name", "unknown")) if train_ds is not None else "unknown")
    print("ROI val:",   getattr(val_ds,   "roi_name", getattr(val_ds,   "aoi_name", "unknown")) if val_ds is not None else "unknown")
    if (train_ds is not None) and hasattr(train_ds, "aoi_bounds"):
        print("Train AOI bounds:", train_ds.aoi_bounds)
    if (val_ds is not None) and hasattr(val_ds, "aoi_bounds"):
        print("Val AOI bounds:", val_ds.aoi_bounds)
except Exception as e:
    print("[WARN] ROI audit failed:", e)

# Quick sanity checks
print(f"[INFO] DataLoaders → Train {len(train_loader)} | Val {len(val_loader)} | Test {len(test_loader)}")
if len(train_loader) > 0:
    xb, yb = next(iter(train_loader))

    print(f"[INFO] Batch X: {tuple(xb.shape)} dtype={xb.dtype} min={float(xb.min()):.3f} max={float(xb.max()):.3f}")
    print(f"[INFO] Batch y: {tuple(yb.shape)} classes={sorted(list(set(yb.cpu().numpy().ravel().tolist())))}")
    print(f"[INFO] Input channels: {xb.shape[1]} | original_bands={ORIG_BANDS} | indices={xb.shape[1]-ORIG_BANDS}")

    # Build exact band names (raw + derived) to match input channels
    derived_count = xb.shape[1] - ORIG_BANDS
    raw = list(SELECTED_BANDS)  # raw sensor bands only

    base_derived = ["NDVI", "NDWI", "MNDWI", "EVI", "AWEI_sh"]
    if derived_count > len(base_derived):
        base_derived += [f"IDX{i}" for i in range(1, derived_count - len(base_derived) + 1)]
    derived = base_derived[:derived_count]

    bands_list = raw + derived
    assert len(bands_list) == xb.shape[1], f"Expected {xb.shape[1]} names, got {len(bands_list)}"

    # Attach to datasets and print
    if train_ds is not None: train_ds.bands = bands_list
    if val_ds   is not None: val_ds.bands   = bands_list
    if test_ds  is not None: test_ds.bands  = bands_list
    print("Bands (final):", bands_list)

else:
    print("[WARN] Train loader is empty; adjust patching thresholds.")

# Class presence summary for each split
def summarize(uids):
    cnt = collections.Counter()
    yroot = Path(MF["patch_root"]) / "Y"
    for uid in uids:
        y = np.load(yroot/f"{uid}.npy")
        s = set(np.unique(y).tolist())
        if 2 in s: cnt["water_patches"] += 1
        if 1 in s: cnt["veg_patches"]   += 1
        if 0 in s: cnt["soil_patches"]  += 1
    cnt["total"] = len(uids)
    return cnt

print("TRAIN:", summarize(train_uids))
print("VAL  :", summarize(val_uids))
print("TEST :", summarize(test_uids))

# Optional quick plot (RGB + mask)
try:
    if len(train_loader) > 0:
        import matplotlib.pyplot as plt
        b4 = CHANNEL_NAMES.index("B4"); b3 = CHANNEL_NAMES.index("B3"); b2 = CHANNEL_NAMES.index("B2")
        rgb = np.transpose(xb[0, [b4, b3, b2]].cpu().numpy(), (1, 2, 0))
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1); plt.imshow(np.clip(rgb, 0, 1)); plt.title("RGB"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(yb[0].cpu().numpy(), cmap="tab20"); plt.title("Mask"); plt.axis("off")
        plt.tight_layout(); plt.show()
    else:
        print("[INFO] Skipping plot (empty train).")
except Exception as e:
    print("[WARN] Plot skipped:", e)
