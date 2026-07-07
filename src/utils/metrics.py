# metrics
@torch.no_grad()
def evaluate_metrics(logits, targets, num_classes=NUM_CLASSES):
    preds = torch.argmax(logits, 1).cpu().numpy()
    t     = targets.cpu().numpy()
    cm = confusion_matrix(t.ravel(), preds.ravel(), labels=list(range(num_classes)))
    acc = (np.diag(cm).sum() / cm.sum()) if cm.sum()>0 else 0.0
    iou = np.diag(cm) / (cm.sum(1)+cm.sum(0)-np.diag(cm)+1e-6)
    miou = float(np.nanmean(iou))
    prec = np.diag(cm)/(cm.sum(0)+1e-6)
    rec  = np.diag(cm)/(cm.sum(1)+1e-6)
    f1   = 2*prec*rec/(prec+rec+1e-6)
    macro_f1 = float(np.nanmean(f1))
    pe = (cm.sum(0)*cm.sum(1)).sum()/(cm.sum()**2 + 1e-6)
    kappa = float((acc-pe)/(1-pe+1e-6))
    return acc, miou, macro_f1, kappa

# train
best_miou, epochs_no_improve = -1.0, 0
val_loader_eff = val_loader if val_loader else test_loader

for epoch in range(1, EPOCHS+1):
    t_epoch = time.time()
    model.train(); running, n_batches, t_batch_total = 0.0, 0, 0.0
    pbar = tqdm(enumerate(train_loader,1), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}")

    for bi,(xb,yb) in pbar:
        t_b0 = time.time()
        xb,yb = xb.to(DEVICE,non_blocking=True), yb.to(DEVICE,non_blocking=True)

        # check channels once
        if bi == 1 and getattr(train_ds, "bands", None):
            assert len(train_ds.bands) == xb.shape[1], \
                f"Mismatch: len(train_ds.bands)={len(train_ds.bands)} vs input channels={xb.shape[1]}"

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(xb); loss = loss_fn(logits, yb)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        else:
            logits = model(xb); loss = loss_fn(logits, yb); loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)

        if bi == 1: save_gradients(epoch, bi)

        if use_amp: scaler.step(optimizer); scaler.update()
        else: optimizer.step()
        scheduler.step()

        running += float(loss.item())*xb.size(0)
        n_batches += 1; t_batch_total += (time.time()-t_b0)
        pbar.set_postfix(loss=f"{running/(bi*xb.size(0)):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    train_loss = running / max(len(train_loader.dataset), 1)

    # validate
    model.eval(); val_running=0.0; accs=[]; mious=[]; f1s=[]; kappas=[]
    for xb,yb in val_loader_eff:
        xb,yb = xb.to(DEVICE,non_blocking=True), yb.to(DEVICE,non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(xb); loss = loss_fn(logits, yb)
        else:
            logits = model(xb); loss = loss_fn(logits, yb)
        val_running += float(loss.item())*xb.size(0)
        a,m,f,k = evaluate_metrics(logits, yb)
        accs.append(a); mious.append(m); f1s.append(f); kappas.append(k)

    val_loss = val_running / max(len(val_loader_eff.dataset), 1)
    acc, miou = float(np.mean(accs)), float(np.mean(mious))
    mf1, kap  = float(np.mean(f1s)), float(np.mean(kappas))
    epoch_sec = time.time()-t_epoch
    batch_ms  = (t_batch_total/max(n_batches,1))*1000.0

    rec = {"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"acc":acc,"mIoU":miou,"macro_f1":mf1,"kappa":kap,
           "lr":float(optimizer.param_groups[0]["lr"]), "epoch_sec":epoch_sec,"batch_ms":batch_ms}
    with open(log_path,"a") as f: f.write(json.dumps(rec)+"\n")
    with open(log_csv,"a") as f:
        f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{acc:.6f},{miou:.6f},{mf1:.6f},{kap:.6f},"
                f"{optimizer.param_groups[0]['lr']:.6e},{epoch_sec:.2f},{batch_ms:.2f}\n")

    print(f"[E{epoch:02d}] train={train_loss:.4f} val={val_loss:.4f} acc={acc:.3f} mIoU={miou:.3f} "
          f"F1={mf1:.3f} Kappa={kap:.3f} epoch={epoch_sec:.1f}s batch≈{batch_ms:.1f}ms")

    save_checkpoint("last", epoch, best_miou)
    if miou > best_miou + 1e-6:
        best_miou = miou
        epochs_no_improve = 0
        save_checkpoint("best", epoch, best_miou)
        print("[INFO] New BEST saved.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"[EARLY STOP] best mIoU = {best_miou:.4f}")
            break

print(f"[INFO] Training done. Best mIoU = {best_miou:.4f}")

# plots and extra eval
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv(log_csv)

# plot losses
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["train_loss"], label="Train")
plt.plot(df["epoch"], df["val_loss"],   label="Val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (Train vs Val)")
plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig(LOG_DIR/"plot_loss.png", dpi=180)
plt.show()

# plot val metrics
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["miou"],     label="mIoU")
plt.plot(df["epoch"], df["acc"],      label="Acc")
plt.plot(df["epoch"], df["macro_f1"], label="Macro-F1")
plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Validation Metrics")
plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig(LOG_DIR/"plot_val_metrics.png", dpi=180)
plt.show()

# plot LR
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["lr"])
plt.xlabel("Epoch"); plt.ylabel("LR"); plt.title("Learning Rate")
plt.grid(True)
plt.tight_layout(); plt.savefig(LOG_DIR/"plot_lr.png", dpi=180)
plt.show()

# per-class IoU on validation
@torch.no_grad()
def compute_cm_on_loader(model, loader, num_classes):
    from sklearn.metrics import confusion_matrix
    model.eval()
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
        preds = torch.argmax(logits, 1)
        y_np = yb.cpu().numpy().ravel()
        p_np = preds.cpu().numpy().ravel()
        cm = confusion_matrix(y_np, p_np, labels=list(range(num_classes)))
        cm_total += cm
    return cm_total

if len(val_loader_eff) > 0:
    cm_val = compute_cm_on_loader(model, val_loader_eff, NUM_CLASSES)
    diag = np.diag(cm_val).astype(np.float32)
    denom = (cm_val.sum(1) + cm_val.sum(0) - np.diag(cm_val)).astype(np.float32) + 1e-6
    iou_per_class = diag / denom
    cls_names = [str(i) for i in range(NUM_CLASSES)]
    try:
        if isinstance(CLASS_NAMES, (list, tuple)) and len(CLASS_NAMES) == NUM_CLASSES:
            cls_names = list(CLASS_NAMES)
    except:
        pass
    plt.figure(figsize=(7.5,4))
    plt.bar(range(NUM_CLASSES), iou_per_class)
    plt.xticks(range(NUM_CLASSES), cls_names)
    plt.ylabel("IoU"); plt.title("Per-class IoU (Validation)")
    for i,v in enumerate(iou_per_class):
        plt.text(i, v+0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(LOG_DIR/"plot_per_class_iou.png", dpi=180)
    plt.show()

# qualitative samples from validation
try:
    if len(val_loader_eff) > 0:
        xb, yb = next(iter(val_loader_eff))
        xb = xb.to(DEVICE, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
        preds = logits.argmax(1).cpu().numpy()
        gt    = yb.cpu().numpy()
        n_show = min(3, xb.shape[0])
        fig, axs = plt.subplots(n_show, 2, figsize=(6, 2*n_show))
        if n_show == 1: axs = np.array([axs])
        for i in range(n_show):
            axs[i,0].imshow(gt[i], cmap="tab20");    axs[i,0].set_title("GT");   axs[i,0].axis("off")
            axs[i,1].imshow(preds[i], cmap="tab20"); axs[i,1].set_title("Pred"); axs[i,1].axis("off")
        plt.tight_layout(); plt.savefig(LOG_DIR/"plot_qualitative_val.png", dpi=180)
        plt.show()
except Exception as e:
    print("[WARN] Qualitative plot skipped:", e)
