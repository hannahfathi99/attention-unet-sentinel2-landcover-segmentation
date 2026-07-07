"""
Attention U-Net Architecture
"""


# WS Conv
class WSConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        mean = w.mean(dim=(1,2,3), keepdim=True)
        std  = w.std (dim=(1,2,3), keepdim=True) + 1e-5
        w = (w - mean) / std
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

#  Norm helper
def make_norm(num_ch, use_gn=True, groups=8):
    return nn.GroupNorm(min(groups, num_ch), num_ch) if use_gn else nn.BatchNorm2d(num_ch)

# SE block
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(1, ch // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)

# Attention gate
class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.theta_x = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.phi_g   = nn.Conv2d(g_ch, inter_ch, 1, bias=False)
        self.psi     = nn.Conv2d(inter_ch, 1, 1)
        self.act, self.sig = nn.SiLU(), nn.Sigmoid()
    def forward(self, g, x):
        g_, x_ = self.phi_g(g), self.theta_x(x)
        if g_.shape[-2:] != x_.shape[-2:]:
            g_ = F.interpolate(g_, size=x_.shape[-2:], mode="bilinear", align_corners=False)
        a = self.sig(self.psi(self.act(g_ + x_)))
        return x * a

# Conv-Norm-Act
class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, use_gn=True, use_ws=True):
        super().__init__()
        Conv = WSConv2d if use_ws else nn.Conv2d
        self.conv = Conv(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = make_norm(out_ch, use_gn)
        self.act  = nn.SiLU()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

# Double conv
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_gn=True, use_ws=True, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_ch, out_ch, use_gn, use_ws),
            ConvNormAct(out_ch, out_ch, use_gn, use_ws),
        )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
    def forward(self, x): return self.se(self.block(x))

# Down block
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_gn=True, use_ws=True, use_se=True):
        super().__init__()
        self.pool, self.conv = nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, use_gn, use_ws, use_se)
    def forward(self, x): return self.conv(self.pool(x))

# Up block
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_gn=True, use_ws=True, use_se=True, p_drop=0.0):
        super().__init__()
        Conv = WSConv2d if use_ws else nn.Conv2d
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv(in_ch, in_ch//2, 1, bias=False),
        )
        inter_ch = max(1, out_ch//2)
        self.att  = AttentionGate(in_ch//2, skip_ch, inter_ch)
        self.conv = DoubleConv(in_ch//2 + skip_ch, out_ch, use_gn, use_ws, use_se)
        self.drop = nn.Dropout2d(p_drop) if p_drop>0 else nn.Identity()
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)
        dh, dw = skip.shape[-2]-x.shape[-2], skip.shape[-1]-x.shape[-1]
        if dh!=0 or dw!=0:
            x = F.pad(x, (0,max(dw,0),0,max(dh,0)))
            x = x[..., :skip.shape[-2], :skip.shape[-1]]
        x = torch.cat([skip, x], 1)
        return self.drop(self.conv(x))

# U-Net
class WaterAwareUNet(nn.Module):
    def __init__(self, in_channels, num_classes, base=32,
                 use_gn=True, use_ws=True, use_se=True, dec_dropout=0.1):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_channels, b, use_gn, use_ws, use_se)
        self.enc2 = Down(b, b*2, use_gn, use_ws, use_se)
        self.enc3 = Down(b*2, b*4, use_gn, use_ws, use_se)
        self.enc4 = Down(b*4, b*8, use_gn, use_ws, use_se)
        self.bott = DoubleConv(b*8, b*16, use_gn, use_ws, use_se)
        self.up4  = Up(b*16, b*8, b*8, use_gn, use_ws, use_se, dec_dropout)
        self.up3  = Up(b*8, b*4, b*4, use_gn, use_ws, use_se, dec_dropout)
        self.up2  = Up(b*4, b*2, b*2, use_gn, use_ws, use_se, dec_dropout/2)
        self.up1  = Up(b*2, b,   b,   use_gn, use_ws, use_se, dec_dropout/4)
        self.head = nn.Conv2d(b, num_classes, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, WSConv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None: nn.init.ones_(m.weight)
            if m.bias   is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bott(e4)
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)

# Build model
IN_CHANNELS = next(iter(train_loader))[0].shape[1]
NUM_CLASSES = len(CLASS_NAMES)
BASE_CH = 32

model = WaterAwareUNet(IN_CHANNELS, NUM_CLASSES, BASE_CH, True, True, True, 0.1).to(DEVICE)

# Model summary log
print(f"[INFO] Model → in={IN_CHANNELS}, classes={NUM_CLASSES}, base={BASE_CH}, "
      f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Trainable check (prove training vs inference)
for n, p in model.named_parameters():
    print(f"{n:60s} trainable={p.requires_grad}")
print("Trainable params:",
      sum(p.numel() for p in model.parameters() if p.requires_grad),
      "/ Total:",
      sum(p.numel() for p in model.parameters()))

# Optional: channel consistency (only if bands metadata exists here)
if getattr(train_ds, "bands", None):
    assert len(train_ds.bands) == IN_CHANNELS, \
        f"Mismatch: len(train_ds.bands)={len(train_ds.bands)} vs IN_CHANNELS={IN_CHANNELS}"


