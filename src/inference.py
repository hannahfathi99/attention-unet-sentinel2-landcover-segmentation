"""
inference script
"""

# params
PATCH_SIZE = 256   # tile size
STRIDE     = 128   # overlap step
BATCH_SIZE = 8     # batch size
USE_CRF    = True  # DenseCRF on/off
ALPHA_PRIOR= 0.6   # fusion weight for index priors

Ck = img_with_indices.shape[-1]

# helpers
def scale_orig_bands(x_hw_c, band_stats, orig_bands=ORIG_BANDS):
    """Scale original bands to [0,1]."""
    x = np.nan_to_num(x_hw_c, nan=0.0).astype(np.float32, copy=True)
    for c, (lo, hi) in enumerate(band_stats[:orig_bands]):
        x[..., c] = np.clip((x[..., c] - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return x

def preprocess_tile(tile, band_stats, orig_bands=ORIG_BANDS):
    return scale_orig_bands(tile, band_stats, orig_bands)

def tta_forward(xb_t: torch.Tensor) -> np.ndarray:
    """Simple TTA: flips + rotations."""
    outs = []
    outs.append(torch.softmax(model(xb_t), dim=1))  # id
    outs.append(torch.flip(torch.softmax(model(torch.flip(xb_t, [-1])), dim=1), [-1]))
    outs.append(torch.flip(torch.softmax(model(torch.flip(xb_t, [-2])), dim=1), [-2]))
    outs.append(torch.flip(torch.softmax(model(torch.flip(xb_t, [-1, -2])), dim=1), [-1, -2]))
    r = torch.rot90(xb_t, 1, dims=[-2, -1])
    outs.append(torch.rot90(torch.softmax(model(r), dim=1), -1, dims=[-2, -1]))
    r = torch.rot90(xb_t, 3, dims=[-2, -1])
    outs.append(torch.rot90(torch.softmax(model(r), dim=1), 1, dims=[-2, -1]))
    return torch.stack(outs, 0).mean(0).cpu().numpy()

# cosine blend to reduce seams
yy, xx = np.meshgrid(np.linspace(0, np.pi, PATCH_SIZE),
                     np.linspace(0, np.pi, PATCH_SIZE), indexing='ij')
BLEND_WIN = ((0.5 - 0.5*np.cos(yy)) * (0.5 - 0.5*np.cos(xx))).astype(np.float32)

# sliding-window inference
t0 = time.time()
prob  = np.zeros((NUM_CLASSES, H, W), np.float32)
wacc  = np.zeros((H, W), np.float32)
tiles_batch, coords_batch = [], []

model.eval()
autocast_on = (DEVICE.type == 'cuda')
with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=autocast_on):
    for y in tqdm(range(0, H, STRIDE), desc="Full-scene inference"):
        for x in range(0, W, STRIDE):
            tile = img_with_indices[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
            ph, pw = tile.shape[:2]
            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                pad = np.zeros((PATCH_SIZE, PATCH_SIZE, Ck), np.float32)
                pad[:ph, :pw] = tile
                tile = pad
            tile = preprocess_tile(tile, BAND_STATS, ORIG_BANDS)
            xb = torch.from_numpy(np.transpose(tile, (2, 0, 1)))
            tiles_batch.append(xb)
            coords_batch.append((y, x, ph, pw))

            if len(tiles_batch) == BATCH_SIZE:
                xb_t = torch.stack(tiles_batch).to(DEVICE, non_blocking=True)
                p_batch = tta_forward(xb_t)
                for p, (yy0, xx0, hh, ww) in zip(p_batch, coords_batch):
                    w = BLEND_WIN[:hh, :ww]
                    prob[:, yy0:yy0+hh, xx0:xx0+ww] += p[:, :hh, :ww] * w
                    wacc[yy0:yy0+hh, xx0:xx0+ww]    += w
                tiles_batch.clear(); coords_batch.clear()

    if tiles_batch:
        xb_t = torch.stack(tiles_batch).to(DEVICE, non_blocking=True)
        p_batch = tta_forward(xb_t)
        for p, (yy0, xx0, hh, ww) in zip(p_batch, coords_batch):
            w = BLEND_WIN[:hh, :ww]
            prob[:, yy0:yy0+hh, xx0:xx0+ww] += p[:, :hh, :ww] * w
            wacc[yy0:yy0+hh, xx0:xx0+ww]    += w

prob /= np.maximum(wacc[None, ...], 1e-6)
prob  = np.nan_to_num(prob, nan=1.0/NUM_CLASSES)

# priors (NDVI/EVI, NDWI/MNDWI)
ndwi = np.nan_to_num(img_with_indices[..., NAME2IDX["NDWI"]], nan=0.0).astype(np.float32)
w_pr = np.clip((ndwi + 1.0)/2.0, 0.0, 1.0)
if "MNDWI" in NAME2IDX:
    mndwi = np.nan_to_num(img_with_indices[..., NAME2IDX["MNDWI"]], nan=0.0).astype(np.float32)
    w_pr = 0.6*w_pr + 0.4*np.clip((mndwi + 1.0)/2.0, 0.0, 1.0)

ndvi = np.nan_to_num(img_with_indices[..., NAME2IDX["NDVI"]], nan=0.0).astype(np.float32)
v_pr = np.clip((ndvi + 1.0)/2.0, 0.0, 1.0)
if "EVI" in NAME2IDX:
    evi = np.nan_to_num(img_with_indices[..., NAME2IDX["EVI"]], nan=0.0).astype(np.float32)
    v_pr = 0.7*v_pr + 0.3*np.clip((evi + 1.0)/2.0, 0.0, 1.0)

s_pr = np.clip(1.0 - np.maximum(v_pr, w_pr), 0.0, 1.0)
priors = np.stack([s_pr, v_pr, w_pr], axis=0)
priors = np.clip(priors, 1e-4, 1.0)

# fusion
logp   = np.log(np.clip(prob, 1e-6, 1.0))
logpri = np.log(priors)
fused  = np.exp(logp + ALPHA_PRIOR * logpri)
fused  = fused / np.maximum(fused.sum(axis=0, keepdims=True), 1e-6)

# optional DenseCRF
if USE_CRF:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
        rgb01 = scale_orig_bands(np.nan_to_num(img_with_indices, nan=0.0), BAND_STATS, ORIG_BANDS)[..., [NAME2IDX["B4"], NAME2IDX["B3"], NAME2IDX["B2"]]]
        rgb_u8 = np.clip(rgb01*255.0, 0, 255).astype(np.uint8)
        d = dcrf.DenseCRF2D(W, H, NUM_CLASSES)
        U = unary_from_softmax(fused.astype(np.float32))
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=4)
        feats = create_pairwise_bilateral(sdims=(60,60), schan=(6,6,6), img=rgb_u8, chdim=2)
        d.addPairwiseEnergy(feats, compat=6)
        Q = np.array(d.inference(6)).reshape((NUM_CLASSES, H, W))
        pred_full = Q.argmax(axis=0).astype(np.uint8)
    except Exception as e:
        print("[WARN] DenseCRF failed, fallback used:", e)
        pred_full = fused.argmax(0).astype(np.uint8)
else:
    pred_full = fused.argmax(0).astype(np.uint8)

# cleanup
for cls, min_size, k in [(VEG, 36, 3), (WATER, 9, 3)]:
    m = (pred_full == cls)
    m = opening(m, square(k))
    m = closing(m, square(k))
    m = remove_small_objects(m, min_size)
    pred_full[(pred_full == cls) & (~m)] = SOIL
    pred_full[m] = cls

# export
t1 = time.time()
print(f"[TIME] Inference{' + CRF' if USE_CRF else ''}: {t1 - t0:.1f}s | Scene: {H}x{W}")

EXP_DIR.mkdir(parents=True, exist_ok=True)
OUT_TIF = EXP_DIR / "prediction_fullscene.tif"
OUT_PNG = EXP_DIR / "prediction_fullscene_rgb.png"

profile = meta.copy(); profile.update(count=1, dtype='uint8', compress='lzw')
with rasterio.open(OUT_TIF, "w", **profile) as dst:
    dst.write(pred_full, 1)

def mask_to_rgb_full(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, c in PALETTE_RGB.items():
        rgb[mask == k] = c
    return rgb

rgb_pred = mask_to_rgb_full(pred_full)
Image.fromarray(rgb_pred).save(OUT_PNG)

print(f"[INFO] GeoTIFF → {OUT_TIF}")
print(f"[INFO] Preview PNG → {OUT_PNG}")

# quick preview
rgb_disp = (np.clip(
    scale_orig_bands(np.nan_to_num(img_with_indices, nan=0.0), BAND_STATS, ORIG_BANDS)[..., [NAME2IDX["B4"], NAME2IDX["B3"], NAME2IDX["B2"]]],
    0, 1
) * 255).astype(np.uint8)

plt.figure(figsize=(16,6))
plt.subplot(1,2,1); plt.imshow(rgb_disp); plt.title("Scaled RGB (B4,B3,B2)"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(rgb_pred); plt.title(f"Prediction{' + DenseCRF' if USE_CRF else ''}"); plt.axis("off")
plt.tight_layout(); plt.show()
