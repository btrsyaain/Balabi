import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock          # 雙 conv 殘差塊
import torchvision.models

def set_seed(seed=500):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
set_seed(42)  # Call the seed function with your desired seed

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class simpleCNN(nn.Module):
    def __init__(self,args=None):
        super(simpleCNN,self).__init__()
        self.multies = args.P
        self.device = args.device

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.batchnorm = nn.BatchNorm1d(int(4096/self.multies))
        self.fully = nn.Linear(int(4096/self.multies),args.bottomModel_output_dim)
        
        self.apply(weights_init)
    
    def forward(self,x):
        x = x.to(device = self.device)

        x = F.relu(self.conv1(x)) 
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x)) 
        x = self.max_pool(F.relu(self.conv4(x))) 

        x = x.view(x.shape[0],-1)
        x = self.batchnorm(x)
        # x = self.fully(F.relu(x))
        x = self.fully(F.relu(x, inplace=False))

        return x

class VGG19Bottom(nn.Module):
    """Canonical VGG‑19 bottom model tailored for vertical FL (2 – 8 parties).

    Design goals
    -------------
    1. **Participant‑agnostic architecture** – depth no longer depends on `P`.
    2. **Strict VGG‑19 backbone** – 16 convolution layers grouped in five blocks.
    3. **Fixed 512‑dimensional embedding** – after GAP the tensor flattens to
       512 features, fed directly into the classifier.
    4. **Robust to narrow slices** – pooling uses `ceil_mode=True` so width never
       collapses to zero even when the slice is 1‑pixel wide.
    """
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        # VGG‑19 convolutional blocks (for 3‑channel inputs)
        self.features = nn.Sequential(
            # Block 1: 2×Conv64 + MP
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),
            # Block 2: 2×Conv128 + MP
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),
            # Block 3: 4×Conv256 + MP
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),
            # Block 4: 4×Conv512 + MP
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),
            # Block 5: 4×Conv512 + GAP
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
        )

        # Global average pooling → (B, 512, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier – first layer receives *512* features
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, 256), nn.ReLU(True), nn.Dropout(),
            nn.BatchNorm1d(256),
            nn.Linear(256, args.bottomModel_output_dim)
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.features(x)
        x = self.gap(x)          # (B, 512, 1, 1)
        x = x.flatten(1)         # (B, 512)
        return self.classifier(x)

class VGG16Bottom(nn.Module):
    """Canonical VGG‑16 bottom model tailored for vertical FL (2–8 parties).

    Design goals
    -------------
    1. Participant‑agnostic architecture – depth不受 P 影響
    2. Strict VGG‑16 backbone – **13** convolution layers grouped in five blocks
    3. Fixed 512‑D embedding – GAP 後 flatten 為 512
    4. Robust to narrow slices – pooling 使用 `ceil_mode=True`
    """
    def __init__(self, args):
        super().__init__()
        self.device = args.device

        # VGG‑16 convolutional backbone
        self.features = nn.Sequential(
            # Block 1: 2×Conv64 + MP
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),

            # Block 2: 2×Conv128 + MP
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),

            # Block 3: **3×Conv256** + MP   ← 少 1 層
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),

            # Block 4: **3×Conv512** + MP   ← 少 1 層
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),

            # Block 5: **3×Conv512** + GAP   ← 少 1 層
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (B,512,1,1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(),
            nn.BatchNorm1d(128),
            nn.Linear(128, args.bottomModel_output_dim)
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)  # → (B, 512)
        return self.classifier(x)

class ResNet18Bottom(nn.Module):
    """ResNet‑18 bottom model tailored for vertical FL (2–8 parties).

    Design goals
    -------------
    1. Participant‑agnostic architecture
    2. CIFAR‑friendly stem: 3×3 conv, stride 1, no max‑pool
    3. Fixed 512‑D embedding (GAP → 512)
    4. Robust to narrow slices – down‑sample 只在 height 方向 (stride=(2,1))
    """
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.in_channels = 64            # tracker for _make_layer

        # ---- Stem ----
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # ---- Residual Stages ----
        # 每層 tuple: (out_channels, blocks, first_stride)
        cfg = [( 64, 2, 1),
               (128, 2, (2,1)),         # ↓ height ×½，寬度保持
               (256, 2, (2,1)),
               (512, 2, (2,1))]

        layers = []
        for out_c, num_blk, stride in cfg:
            layers.append(self._make_layer(out_c, num_blk, stride))
        self.features = nn.Sequential(*layers)

        # ---- Head ----
        self.gap = nn.AdaptiveAvgPool2d((1,1))        # (B,512,1,1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(),
            nn.BatchNorm1d(128),
            nn.Linear(128, args.bottomModel_output_dim)
        )
        self.to(self.device)

    # ---------------- private helpers ----------------
    def _make_layer(self, out_c, blocks, first_stride):
        """build residual stage just like torchvision _make_layer"""
        strides = [first_stride] + [1]*(blocks-1)
        layers  = []
        for stride in strides:
            downsample = None
            if stride != 1 or self.in_channels != out_c:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_c, 1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(out_c)
                )
            layers.append(BasicBlock(self.in_channels,
                                     out_c,
                                     stride=stride,
                                     downsample=downsample))
            self.in_channels = out_c
        return nn.Sequential(*layers)

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.features(x)
        x = self.gap(x).flatten(1)     # (B,512)
        return self.classifier(x)

'''
以下是 mini-GoogLeNet
'''
# --------------------------------------------------
# Mini‑GoogLeNet (13 learnable conv layers)
# Based on: Zhang et al., “Understanding Deep Learning Requires Rethinking
# Generalization,” ICLR 2017, Appendix C (Mini‑GoogLeNet for CIFAR‑10)
# --------------------------------------------------

class _BasicConv(nn.Module):
    """Conv‑BN‑ReLU with default eps=0.001."""
    def __init__(self, in_c, out_c, **kw):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, bias=False, **kw)
        self.bn   = nn.BatchNorm2d(out_c, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class _MiniInception(nn.Module):
    """3‑branch mini‑inception (1×1, 3×3 reduce→3×3, 3×3 pool‑proj)."""
    def __init__(self, in_c, ch1, ch3red, ch3, pool_proj):
        super().__init__()
        self.b1 = _BasicConv(in_c, ch1, kernel_size=1)
        self.b2 = nn.Sequential(
            _BasicConv(in_c, ch3red, kernel_size=1),
            _BasicConv(ch3red, ch3,   kernel_size=3, padding=1)
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            _BasicConv(in_c, pool_proj, kernel_size=1)
        )
    def forward(self,x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], 1)

class MiniGoogLeNetBottom(nn.Module):
    """Mini‑GoogLeNet bottom model for vertical FL (2–8 participants).

    * 13 conv layers (3 stem + 3×2 inception modules + tail conv).
    * CIFAR/CINIC‑friendly: 所有 stride=1，僅在 height 方向 pool。
    * Global Average Pool → 192‑D embedding → 小型 classifier → out_dim.
    """
    def __init__(self, args):
        super().__init__()
        self.device = args.device

        # ---- Stem ----
        self.stem = nn.Sequential(
            _BasicConv(3,   96, kernel_size=3, padding=1),   # conv1
            _BasicConv(96,  96, kernel_size=3, padding=1),   # conv2
            _BasicConv(96,  96, kernel_size=3, padding=1)    # conv3
        )

        # ---- Inception blocks (channels follow Zhang et al.) ----
        self.in3a = _MiniInception(96,  32,  48,  64, 16)   # out 32+64+16 =112
        self.in3b = _MiniInception(112, 32,  48,  64, 16)   # out 112
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))  # only height ↓

        self.in4a = _MiniInception(112, 64,  64,  96, 32)   # out 64+96+32 =192
        self.in4b = _MiniInception(192, 64,  64,  96, 32)   # out 192
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        # ---- Tail conv ----
        self.conv_tail = _BasicConv(192, 192, kernel_size=3, padding=1)

        # ---- Head ----
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(True), nn.Dropout(),
            nn.Linear(128, 64),  nn.ReLU(True), nn.Dropout(),
            nn.BatchNorm1d(64),
            nn.Linear(64, args.bottomModel_output_dim)
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        x = self.stem(x)
        x = self.pool1(self.in3b(self.in3a(x)))
        x = self.pool2(self.in4b(self.in4a(x)))
        x = self.conv_tail(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)

# class BottomModelBM(nn.Module):
#     def __init__(self, in_dim: int, embed_dim: int = 32, args):
#         super().__init__()
#         self.input_dim = 
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 64, bias=False),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, embed_dim)   # 傳給頂層模型或防禦模型
#         )
#     def forward(self, x):
#         return self.net(x)

class BottomModelBM(nn.Module):
    """Lightweight MLP bottom model for tabular data (Bank-Marketing)."""
    def __init__(self, in_dim: int, embed_dim: int, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512, bias=True),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            
            nn.Linear(512, 256,  bias=True),
            nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.1),
            
            nn.Linear(256, 128,  bias=True),
            nn.BatchNorm1d(128),  nn.ReLU(), nn.Dropout(0.1),

            nn.Linear(128, 64,  bias=True),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),

            nn.Linear(64, embed_dim)          # 送往 top model／防禦模型
        )
        self.device = args.device
        self.to(self.device)

    def forward(self, x):
        return self.net(x.to(self.device))
    
    
class BottomFCN_NUS(nn.Module):
    """
    Bottom network for NUS-WIDE vertical-FL setting.
    • in_dim  : length of local feature slice assigned to this party
    • out_dim : fixed 128-D embedding for concatenation
    """
    def __init__(self, in_dim: int, out_dim: int = 128, p_dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024, bias=True), nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(1024, 512, bias=True),    nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(512, 256, bias=True),     nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(256, 128, bias=True),     nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(128, out_dim)                         # no activation
        )

    def forward(self, x):
        return self.net(x)

# class TopModel(nn.Module):
#     def __init__(self, args=None):
#         super(TopModel,self).__init__()
        
#         input_size = args.P * args.bottomModel_output_dim
#         self.fc1 = nn.Linear(input_size, input_size)
#         self.fc2 = nn.Linear(input_size, int(input_size/2))
#         self.output = nn.Linear(int(input_size/2), args.num_class)
        
#         self.apply(weights_init)
#         self.device = args.device
#         self.to(device=self.device)

#     def forward(self,x):
#         x.to(device=self.device)
        
#         B = x.size()[0]
#         x = F.relu(self.fc1(x.view(B,-1)))
#         x = F.dropout(F.relu(self.fc2(x)), 0.5, training=self.training)
#         x = self.output(x)
#         return x
    
class TopModel(nn.Module):
    """Fully‑connected top model for vertical FL.

    * 連接所有 participant 的嵌入後 → 兩層 FC → softmax logits
    * 使用 `flatten(1)` 取代 `view(B,‑1)`，可安全處理 batch_size = 0
    """
    def __init__(self, args):
        super().__init__()
        input_dim = args.P * args.bottomModel_output_dim   # ← 拼接後維度

        self.fc1   = nn.Linear(input_dim, input_dim)
        self.fc2   = nn.Linear(input_dim, input_dim // 2)
        self.out   = nn.Linear(input_dim // 2, args.num_class)

        self.apply(weights_init)          # 你的自訂初始化
        self.device = args.device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)              # 保證張量在正確 GPU
        x = F.relu(self.fc1(x.flatten(1))) # <- 取代 view，批量為 0 也 OK
        x = F.dropout(F.relu(self.fc2(x)), p=0.1, training=self.training)
        return self.out(x)

class Model(nn.Module):
    def __init__(self, args, feature_slices=None):
        super(Model, self).__init__()
        self.multies = args.P
        self.D = args.D
        if args.D in ["CIFAR10","CINIC10"]:
            self.bottommodels = nn.ModuleList([MiniGoogLeNetBottom(args) for i in range(self.multies)])
        elif args.D == "BM":
            assert feature_slices is not None, "BM 需傳入 feature_slices"
            self.feature_slices = feature_slices
            # 累積切點  e.g. [0, 20, 40, 60, 70]  (P=4)
            # self.cuts = [0]
            # for s in feature_slices:
            #     self.cuts.append(self.cuts[-1] + s)
            self.cuts = [0]
            for s in feature_slices:
                self.cuts.append(self.cuts[-1] + s)


            self.bottommodels = nn.ModuleList([
                BottomModelBM(in_dim=feature_slices[i],
                              embed_dim=args.bottomModel_output_dim,
                              args=args)
                for i in range(self.multies)
            ])
        elif args.D == "NUSWIDE":                       # ← 新增
            assert feature_slices is not None, "NUSWIDE 需傳入 feature_slices"
            self.feature_slices = feature_slices
            self.cuts = [0]
            for s in feature_slices:
                self.cuts.append(self.cuts[-1] + s)

            self.bottommodels = nn.ModuleList([
                BottomFCN_NUS(in_dim=feature_slices[i],
                            out_dim=args.bottomModel_output_dim,
                            p_dropout=0.1)
                for i in range(self.multies)
            ])

        else:
            raise ValueError("Unsupported dataset")
        # self.bottommodels = nn.ModuleList([ResNet18Bottom(args) for i in range(self.multies)])
        # self.bottommodels = nn.ModuleList([VGG16Bottom(args) for i in range(self.multies)])
        # self.bottommodels = nn.ModuleList([VGG19Bottom(args) for i in range(self.multies)])
        # self.bottommodels = nn.ModuleList([simpleCNN(args) for i in range(self.multies)])
        self.top = TopModel(args)
        self.device = args.device
        self.to(self.device)
        self.DM = args.DM
        if args.DM == "VFLMonitor":
            self.center_loss_weight = args.center_loss_weight
        # -------- Center Loss -------- ★ 新增區塊
        if args.DM == "VFLMonitor":
            # 假設每個 bottom 輸出同尺寸 embed_dim
            self.bottom_out_dim = args.bottomModel_output_dim
            self.center_loss = CenterLoss(num_classes=args.num_class,
                                          feat_dim=self.bottom_out_dim,
                                          device=args.device)
            # 建議於外層 train_new.py 另起 optimizer 給 self.center_loss
        self._cached_embed = None

    def forward(self, x):
        x = x.to(self.device)

        # -------- 影像資料 (CIFAR10 / CINIC10) ----------
        if self.D in ["CIFAR10", "CINIC10"]:
            H = x.size(2)
            piece = H // self.multies                         # 基本片高
            x_list = list(torch.split(x, piece, dim=2))       # 可能 P 或 P+1 片
            if len(x_list) > self.multies:                    # 有餘數片
                x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)  # 併到最後一人
                x_list = x_list[:-1]                          # 移除空殼，len = P

        # -------- 表格資料 (Bank-Marketing) ----------
        # elif self.D == "BM":
        #     F = x.size(1)
        #     piece = F // self.multies
        #     # 前 P-1 人各拿 piece 個特徵
        #     x_list = [ x[:, i*piece:(i+1)*piece] for i in range(self.multies-1) ]
        #     # 最後一人拿剩下所有特徵 (piece + r)
        #     x_list.append(x[:, (self.multies-1)*piece : ])     # len = P
        
        elif self.D == "BM":
            # 捨棄舊的平均切法，用 self.cuts
            x_list = [
                x[:, self.cuts[i]: self.cuts[i+1]]     # (B, slice_i)
                for i in range(self.multies)
            ]
        elif self.D in ["BM", "NUSWIDE"]:
            x_list = [ x[:, self.cuts[i]: self.cuts[i+1]] for i in range(self.multies) ]

        else:
            raise RuntimeError(f"Unsupported dataset {self.D}")

        # -------- 底層模型 + Top model ----------
        embedding_list   = [ self.bottommodels[i](x_list[i]) for i in range(self.multies) ]
        Embedding_Vector = torch.cat(embedding_list, dim=1)    # (B, P*D)
        
        # ★ 把當前 batch 的 concatenated embedding 暫存，方便 loss() 讀取
        self._cached_embed = Embedding_Vector

        return self.top(Embedding_Vector)


    def loss(self, pred, label):
        # label = label.to(self.device)
        # classification_loss = F.cross_entropy(pred, label)
        # return classification_loss
        
        """
        保持呼叫介面不變：
            loss = model.loss(pred, label)
        啟用 Center Loss 時自動加權；否則僅 Cross-Entropy。
        """
        label = label.to(self.device)
        classification_loss = F.cross_entropy(pred, label)

        # 若未設定 center_loss_weight，直接回傳 CE
        if self.DM != "VFLMonitor":
            return classification_loss

        # 取剛才 forward 暫存的 embedding，平均 P 個 slice → (B, embed_dim)
        B, PD = self._cached_embed.shape
        embed_dim = PD // self.multies
        feats = self._cached_embed.view(B, self.multies, embed_dim).mean(dim=1)

        cl = self.center_loss(feats, label)
        return classification_loss + self.center_loss_weight * cl

class CenterLoss(nn.Module):
    """
    Center Loss ( Wen et al., 2016 )：
        L_c = 1/2 * Σ‖f_i − c_{y_i}‖²
    每類一個 center 參數，透過獨立 optimizer 更新。
    """
    def __init__(self, num_classes: int, feat_dim: int, device=None):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_centers = self.centers[labels]          # (B, d)
        return 0.5 * (feats - batch_centers).pow(2).sum(dim=1).mean()