t0 = time.time()

# Load selected bands from GeoTIFF
def _read_selected_bands(tif_path, band_map, selected):
    with rasterio.open(tif_path, sharing=False) as src:
        idxs = [band_map[b] for b in selected]  # band indices (1-based)
        arr = src.read(indexes=idxs)            # (C,H,W)
        meta = src.meta.copy()
        bounds = src.bounds
        nodata = src.nodata
    return np.transpose(arr, (1, 2, 0)).astype(np.float32), meta, bounds, nodata

# Read image if not already in memory
if 'img_hwc' not in globals() or 'meta' not in globals():
    print("[WARN] Previous cell not found in memory → reading image now.")
    img_hwc, meta, bounds, nodata = _read_selected_bands(TIF_PATH, BAND_MAP, SELECTED_BANDS)
    nodata_mask = None

# Image shape and pixel size
H, W, C = img_hwc.shape
tr = meta.get('transform', None)
px_size_x = tr[0] if tr is not None else None
px_size_y = -tr[4] if tr is not None else None

print(f"[INFO] Loaded: {TIF_PATH}")
print(f"[INFO] Shape: {H}×{W}×{C} | CRS: {meta.get('crs')} | Pixel: {px_size_x}×{px_size_y} m | NoData: {meta.get('nodata')}")

# Compute robust min/max (p2–p98) for reflectance bands
def compute_band_stats(arr, orig_band_names, p_lo=2, p_hi=98):
    stats = []
    for bname in orig_band_names:
        idx = BAND_MAP[bname] - 1
        vals = arr[..., idx]
        finite = np.isfinite(vals)
        if not np.any(finite):
            stats.append((0.0, 1.0))
            continue
        lo, hi = np.nanpercentile(vals[finite], [p_lo, p_hi])
        if hi - lo < 1e-6:
            lo = float(np.nanmin(vals[finite]))
            hi = float(np.nanmax(vals[finite]) + 1e-6)
        stats.append((float(lo), float(hi)))
    return stats

# Select original reflectance bands (10m + 20m resampled)
orig_reflectance = [b for b in ['B2','B3','B4','B8','B11','B12'] if b in SELECTED_BANDS]
BAND_STATS = compute_band_stats(img_hwc, orig_reflectance, 2, 98)

print("[INFO] BAND_STATS (p2–p98) for reflectance bands:")
for bname, (lo, hi) in zip(orig_reflectance, BAND_STATS):
    print(f"  {bname}: lo={lo:.3f}, hi={hi:.3f}")

# Compute statistics for water indices
water_indices = [b for b in ['NDWI','MNDWI','AWEI_sh'] if b in SELECTED_BANDS]
if water_indices:
    print("\n[WATER-QA] Water index statistics:")
    for wb in water_indices:
        idx = BAND_MAP[wb] - 1
        vals = img_hwc[..., idx]
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            print(f"  {wb}: no valid data")
            continue
        mean_val = float(np.nanmean(vals))
        water_frac = 100.0 * np.sum(vals > 0.3) / finite.sum()
        print(f"  {wb}: mean={mean_val:.3f} | >0.3 fraction={water_frac:.2f}%")

print(f"\n[INFO] Done in {time.time()-t0:.2f}s")

# Save band statistics for later use
ORIG_BANDS = len(orig_reflectance)
_ = (ORIG_BANDS, BAND_STATS, orig_reflectance)


import time, numpy as np, rasterio

EPS = 1e-6
ADD_EVI = True  # set False if you do not want EVI

t0 = time.time()

# Copy image and build a finite-pixel mask
img = img_hwc.astype(np.float32, copy=True)
finite_mask = np.isfinite(img).all(axis=-1)

# Build band index map (1-based names → 0-based array indices)
idx = {b: SELECTED_BANDS.index(b) for b in SELECTED_BANDS}

# Aliases for common bands (set to None if missing)
blue  = img[..., idx["B2"]]
green = img[..., idx["B3"]]
red   = img[..., idx["B4"]]
nir   = img[..., idx["B8"]]
swir1 = img[..., idx["B11"]] if "B11" in idx else None
swir2 = img[..., idx["B12"]] if "B12" in idx else None

# Compute indices
with np.errstate(divide='ignore', invalid='ignore'):
    ndvi  = (nir - red)   / (nir + red + EPS)
    ndwi  = (green - nir) / (green + nir + EPS)
    mndwi = (green - swir1) / (green + swir1 + EPS) if swir1 is not None else None

    evi = None
    if ADD_EVI:
        evi = 2.5 * (nir - red) / (nir + 6.0*red - 7.5*blue + 1.0 + EPS)

    awei_sh = None
    if swir1 is not None and swir2 is not None:
        awei_sh = (4 * (green - swir1) - (0.25 * nir + 2.75 * swir2))

# Clip to a safe range where appropriate
def safe_clip(arr, lo=-1.0, hi=1.0):
    out = np.full_like(arr, np.nan, dtype=np.float32)
    out[finite_mask] = np.clip(arr[finite_mask], lo, hi)
    return out[..., None]

ndvi  = safe_clip(ndvi)
ndwi  = safe_clip(ndwi)
mndwi = safe_clip(mndwi) if mndwi is not None else None
if evi is not None:
    evi = safe_clip(evi)
if awei_sh is not None:
    # keep raw range for water detection
    awei_sh = awei_sh[..., None].astype(np.float32)

# Stack channels and names
channels = [img, ndvi, ndwi]
names    = SELECTED_BANDS + ["NDVI", "NDWI"]

if mndwi is not None:
    channels.append(mndwi)
    names.append("MNDWI")
if ADD_EVI and evi is not None:
    channels.append(evi)
    names.append("EVI")
if awei_sh is not None:
    channels.append(awei_sh)
    names.append("AWEI_sh")

img_with_indices = np.concatenate(channels, axis=-1).astype(np.float32)
NAME2IDX = {n: i for i, n in enumerate(names)}

print("[INFO] Channels prepared:", names)
print("[INFO] Final tensor shape:", img_with_indices.shape)


