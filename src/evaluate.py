"""
Evaluation script
"""


# save dir
EVAL_DIR = (RUN_ROOT / "eval")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# optional pandas
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

@torch.no_grad()
def evaluate_model(model, dataloader, device, class_names, tag="test", water_class_name="water"):
    # setup
    model.eval()
    if len(dataloader) == 0:
        raise RuntimeError("[ERROR] Empty dataloader.")

    C = len(class_names)
    labels = list(range(C))
    try:
        WATER_ID = class_names.index(water_class_name)
    except ValueError:
        WATER_ID = C - 1

    # collect preds
    y_true_all, y_pred_all, y_prob_water_all = [], [], []
    for xb, yb in dataloader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)
        y_true_all.append(yb.detach().cpu().numpy().ravel())
        y_pred_all.append(preds.detach().cpu().numpy().ravel())
        y_prob_water_all.append(probs[:, WATER_ID].detach().cpu().numpy().ravel())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    pw     = np.concatenate(y_prob_water_all)

    # drop -1 labels
    vm = y_true >= 0
    if not np.all(vm):
        y_true, y_pred, pw = y_true[vm], y_pred[vm], pw[vm]

    # metrics
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    inter = np.diag(cm).astype(float)
    union = cm.sum(1) + cm.sum(0) - inter
    iou  = np.divide(inter, np.maximum(union, 1e-6), out=np.zeros_like(inter, float), where=(union > 0))
    miou = float(np.nanmean(iou))

    freq  = cm.sum(1) / max(cm.sum(), 1)
    fwiou = float(np.nansum(freq * iou))

    macro_f1   = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_rec  = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    total = cm.sum()
    pe = (cm.sum(0) * cm.sum(1)).sum() / (total**2 + 1e-6) if total > 0 else 0.0
    kappa = float((acc - pe) / (1 - pe + 1e-6)) if (1 - pe) > 0 else 0.0

    prec_per = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_per  = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # water diagnostics (no plots)
    y_water_true = (y_true == WATER_ID).astype(np.uint8)
    try: water_ap  = float(average_precision_score(y_water_true, pw))
    except Exception: water_ap = float('nan')
    try: water_auc = float(roc_auc_score(y_water_true, pw))
    except Exception: water_auc = float('nan')

    # save JSON
    metrics_json = {
        "tag": tag,
        "class_names": class_names,
        "pixel_accuracy": float(acc),
        "mean_iou": float(miou),
        "fw_iou": float(fwiou),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "kappa": float(kappa),
        "per_class": {
            class_names[i]: {
                "IoU": float(iou[i]),
                "precision": float(prec_per[i]),
                "recall": float(rec_per[i]),
                "support": int(cm[i].sum()),
            } for i in labels
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_norm": (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).tolist(),
        "water_diagnostics": {"class_id": int(WATER_ID), "AP": water_ap, "ROC_AUC": water_auc}
    }
    with open(EVAL_DIR / f"metrics_{tag}.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    # optional CSV
    if _HAS_PANDAS:
        rows = [{"class": class_names[i], "IoU": float(iou[i]),
                 "precision": float(prec_per[i]), "recall": float(rec_per[i]),
                 "support": int(cm[i].sum())} for i in labels]
        pd.DataFrame(rows).to_csv(EVAL_DIR / f"per_class_{tag}.csv", index=False)

    # Confusion Matrix (count)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({tag})")
    plt.tight_layout()
    path_cm = EVAL_DIR / f"confusion_matrix_{tag}.png"
    plt.savefig(path_cm, dpi=300)
    plt.show(); plt.close()
    print(f"[SAVED] {path_cm}")

    # Confusion Matrix (normalized)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(5,4))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    plt.title(f"Confusion Matrix (normalized) — {tag}")
    plt.tight_layout()
    path_cmn = EVAL_DIR / f"confusion_matrix_norm_{tag}.png"
    plt.savefig(path_cmn, dpi=300)
    plt.show(); plt.close()
    print(f"[SAVED] {path_cmn}")

    # Per-class IoU
    plt.figure(figsize=(5,4))
    plt.bar(class_names, iou, width=0.6)
    plt.ylim(0, 1)
    plt.ylabel("IoU")
    plt.title(f"Per-class IoU — {tag}")
    plt.tight_layout()
    path_iou = EVAL_DIR / f"per_class_iou_{tag}.png"
    plt.savefig(path_iou, dpi=300)
    plt.show(); plt.close()
    print(f"[SAVED] {path_iou}")

    # print summary
    print("=== Metrics ===")
    print(f"Pixel Acc : {acc:.4f}")
    print(f"Mean IoU  : {miou:.4f} | FWIoU: {fwiou:.4f}")
    print(f"Macro F1  : {macro_f1:.4f}")
    print(f"Macro P/R : {macro_prec:.4f} / {macro_rec:.4f}")
    print(f"Kappa     : {kappa:.4f}")
    for i, cn in enumerate(class_names):
        print(f"{cn:>11s} → IoU={iou[i]:.4f}")

    return metrics_json, cm, iou

# run
_ = evaluate_model(model, test_loader, DEVICE, CLASS_NAMES, tag="test")


# Show samples; skip if not valid
import os, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

SAMPLES_DIR = RUN_ROOT / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

def mask_to_rgb(mask_np: np.ndarray) -> np.ndarray:
    # map class ids to RGB
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for k, color in PALETTE_RGB.items():
        rgb[mask_np == k] = color
    return rgb

def save_indexed_png(mask_np: np.ndarray, out_path):
    # save mask as indexed PNG with palette
    pal = [0] * 768
    for cls, (r, g, b) in PALETTE_RGB.items():
        pal[cls*3:cls*3+3] = [r, g, b]
    img_p = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    img_p.putpalette(pal)
    img_p.save(out_path, optimize=True)

def get_water_id(class_names, fallback=WATER):
    # resolve water class id safely
    try:
        return class_names.index("water")
    except ValueError:
        return fallback

@torch.no_grad()
def plot_one_sample(ds, idx, save=True):
    # forward + visualize one sample (RGB / GT / Prediction)
    model.eval()
    x, y = ds[idx]
    logits = model(x.unsqueeze(0).to(DEVICE))
    probs  = torch.softmax(logits, dim=1)
    pred   = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

    # RGB from bands B4,B3,B2
    rgb = x[[2,1,0], ...].permute(1,2,0).cpu().numpy()
    rgb_8u = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    gt_np   = y.numpy() if torch.is_tensor(y) else y
    gt_rgb  = mask_to_rgb(gt_np)
    pr_rgb  = mask_to_rgb(pred)

    base = f"sample_{idx:04d}"
    if save:
        Image.fromarray(rgb_8u).save(SAMPLES_DIR / f"{base}_rgb.png")
        Image.fromarray(gt_rgb).save(SAMPLES_DIR / f"{base}_gt.png")
        Image.fromarray(pr_rgb).save(SAMPLES_DIR / f"{base}_pred.png")
        save_indexed_png(gt_np, SAMPLES_DIR / f"{base}_gt_idx.png")
        save_indexed_png(pred,  SAMPLES_DIR / f"{base}_pred_idx.png")

    # show 3 panels
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(rgb_8u); axs[0].set_title("RGB"); axs[0].axis("off")
    axs[1].imshow(gt_rgb); axs[1].set_title("Ground Truth"); axs[1].axis("off")
    axs[2].imshow(pr_rgb); axs[2].set_title("Prediction"); axs[2].axis("off")
    plt.tight_layout(); plt.show()

# pick same evenly spaced indices as before, then apply a light sanity filter
N = min(5, len(test_ds))
if N == 0:
    print("[WARN] test_ds is empty; skipping samples.")
else:
    idxs = np.linspace(0, len(test_ds) - 1, N, dtype=int).tolist()
    water_id = get_water_id(CLASS_NAMES, fallback=WATER)

    for i in idxs:
        # basic sanity check: skip masks with no positive pixels for the target id
        _, y = test_ds[i]
        y_np = y.numpy() if torch.is_tensor(y) else y
        if not (y_np == water_id).any():
            print(f"[SKIP] Sample {i} skipped.")
            continue
        plot_one_sample(test_ds, idx=i, save=True)
