# losses
def dice_loss(logits, targets, eps=1e-6):
    C = logits.shape[1]
    probs  = torch.softmax(logits, 1)
    onehot = torch.nn.functional.one_hot(targets, C).permute(0,3,1,2).float()
    num = 2*(probs*onehot).sum((0,2,3))
    den = (probs+onehot).sum((0,2,3)) + eps
    return 1.0 - (num/den).mean()

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, dice_w=1.0, ce_w=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dw, self.cw = dice_w, ce_w
    def forward(self, logits, targets):
        return self.cw*self.ce(logits, targets) + self.dw*dice_loss(logits, targets)

loss_fn = CombinedLoss(class_weights=class_weights)

# optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS,
                                                 steps_per_epoch=BPE, pct_start=0.1)
             if BPE >= 3 else
             torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS))

# amp
use_amp = (DEVICE.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# optim log
for i, g in enumerate(optimizer.param_groups):
    print(f"[OPTIM] group {i}: lr={g['lr']:.2e}, weight_decay={g.get('weight_decay',0)} params={len(g['params'])}")

# run config snapshot
run_cfg = {
    "in_channels": IN_CHANNELS,
    "num_classes": NUM_CLASSES,
    "base_ch": BASE_CH,
    "optimizer": "AdamW",
    "lr": LR, "epochs": EPOCHS, "weight_decay": 1e-4,
    "scheduler": type(scheduler).__name__,
    "use_amp": use_amp,
    "roi_train": getattr(train_ds, "roi_name", "unknown"),
    "roi_val": getattr(val_ds, "roi_name", "unknown"),
    "bands": getattr(train_ds, "bands", None),
}
print("[RUNCFG]", run_cfg)

# io
CKPT_DIR = RUN_ROOT/"checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
GRAD_DIR = RUN_ROOT/"gradients";  GRAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR  = RUN_ROOT/"logs";       LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR/"train_log.jsonl"; open(log_path,"w").close()
log_csv  = LOG_DIR/"train_log.csv"; open(log_csv,"w").write(
    "epoch,train_loss,val_loss,acc,miou,macro_f1,kappa,lr,epoch_sec,batch_ms\n"
)

