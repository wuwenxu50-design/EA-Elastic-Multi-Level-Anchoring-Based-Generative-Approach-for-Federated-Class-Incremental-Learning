import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class NLGenerator(nn.Module):
    def __init__(
        self,
        ngf=64, img_size=32, nc=3, nl=100,
        label_emb=None, le_emb_size=256, le_size=512, sbz=200,
        film_enabled: bool = False,
        film_hidden: Optional[int] = None,
    ):
        super(NLGenerator, self).__init__()

        self.params = (ngf, img_size, nc, nl, le_emb_size, le_size, sbz, film_enabled, film_hidden)
        self.le_emb_size = le_emb_size
        if not torch.is_tensor(label_emb):
            label_emb = torch.tensor(label_emb, dtype=torch.float32)
        self.label_emb = nn.Parameter(label_emb, requires_grad=False)
        self.init_size = img_size // 4
        self.le_size = le_size
        self.nl = nl
        self.sbz = sbz
        self.nle = int(np.ceil(self.sbz / self.nl))

        self.n1 = nn.BatchNorm1d(le_size)
        self.cond_bn = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for _ in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * 2 * self.init_size ** 2))

        # -------- FiLM modulation (optional) --------
        self.film_enabled = bool(film_enabled)
        if self.film_enabled:
            h = int(film_hidden) if film_hidden is not None else le_size
            self.film_gamma = nn.Sequential(
                nn.Linear(le_size, h),
                nn.ReLU(inplace=True),
                nn.Linear(h, le_size),
            )
            self.film_beta = nn.Sequential(
                nn.Linear(le_size, h),
                nn.ReLU(inplace=True),
                nn.Linear(h, le_size),
            )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, targets=None, cond=None, le_override=None):
        # 1) get first-level anchor e^(1)
        if le_override is None:
            if targets is None:
                raise ValueError("NLGenerator.forward: targets is None while le_override is None.")
            le = self.label_emb[targets]
        else:
            le = le_override

        # 2) optional BN (keep your original behavior)
        le = self.n1(le)
        # 3) FiLM modulation: e~(1)=gamma*e(1)+beta
        if self.film_enabled and (cond is not None):
            cond = cond.to(device=le.device, dtype=le.dtype)
            cond = self.cond_bn(cond)
            if cond.dim() != 2 or cond.size(0) != le.size(0) or cond.size(1) != le.size(1):
                raise ValueError(
                    f"NLGenerator.forward: cond shape must be [B, le_size], "
                    f"got {tuple(cond.shape)}, expected [{le.size(0)}, {le.size(1)}]"
                )
            gamma = torch.sigmoid(self.film_gamma(cond)) * 2.0
            beta = self.film_beta(cond)
            le = gamma * le + beta

        # 4) keep your original packing logic
        v = None
        for i in range(self.nle):
            sle = le[i * self.nl:] if (i + 1) * self.nl > le.shape[0] else le[i * self.nl:(i + 1) * self.nl]
            sv = self.le1[i](sle)
            v = sv if v is None else torch.cat((v, sv), dim=0)

        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def reinit(self):

        return NLGenerator(
            self.params[0], self.params[1], self.params[2], self.params[3],
            label_emb=self.label_emb.data,
            le_emb_size=self.params[4],
            le_size=self.params[5],
            sbz=self.params[6],
            film_enabled=self.params[7],
            film_hidden=self.params[8],
        ).cuda()


class NLGenerator_IN(nn.Module):
    def __init__(
        self,
        ngf=64, img_size=224, nc=3, nl=100,
        label_emb=None, le_emb_size=256, le_size=512, sbz=200,
        film_enabled: bool = False,
        film_hidden: Optional[int] = None,
    ):
        super(NLGenerator_IN, self).__init__()

        self.params = (ngf, img_size, nc, nl, le_emb_size, le_size, sbz, film_enabled, film_hidden)
        self.le_emb_size = le_emb_size
        if not torch.is_tensor(label_emb):
            label_emb = torch.tensor(label_emb, dtype=torch.float32)
        self.label_emb = nn.Parameter(label_emb, requires_grad=False)
        self.init_size = img_size // 16
        self.le_size = le_size
        self.nl = nl
        self.sbz = sbz
        self.nle = int(np.ceil(sbz / nl))

        self.n1 = nn.BatchNorm1d(le_size)
        self.cond_bn = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for _ in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * 2 * self.init_size ** 2))

        # -------- FiLM modulation (optional) --------
        self.film_enabled = bool(film_enabled)
        if self.film_enabled:
            h = int(film_hidden) if film_hidden is not None else le_size
            self.film_gamma = nn.Sequential(
                nn.Linear(le_size, h),
                nn.ReLU(inplace=True),
                nn.Linear(h, le_size),
            )
            self.film_beta = nn.Sequential(
                nn.Linear(le_size, h),
                nn.ReLU(inplace=True),
                nn.Linear(h, le_size),
            )

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2 * ngf, 2 * ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2 * ngf, 2 * ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2 * ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def forward(self, targets=None, cond=None, le_override=None):
        # 1) get first-level anchor e^(1)
        if le_override is None:
            if targets is None:
                raise ValueError("NLGenerator_IN.forward: targets is None while le_override is None.")
            le = self.label_emb[targets]
        else:
            le = le_override

        # 2) BN
        le = self.n1(le)

        # 3) FiLM modulation
        if self.film_enabled and (cond is not None):
            cond = cond.to(device=le.device, dtype=le.dtype)
            cond = self.cond_bn(cond)
            if cond.dim() != 2 or cond.size(0) != le.size(0) or cond.size(1) != le.size(1):
                raise ValueError(
                    f"NLGenerator_IN.forward: cond shape must be [B, le_size], "
                    f"got {tuple(cond.shape)}, expected [{le.size(0)}, {le.size(1)}]"
                )
            gamma = torch.sigmoid(self.film_gamma(cond)) * 2.0
            beta = self.film_beta(cond)
            le = gamma * le + beta

        # 4) keep your original packing logic
        v = None
        for i in range(self.nle):
            sle = le[i * self.nl:] if (i + 1) * self.nl > le.shape[0] else le[i * self.nl:(i + 1) * self.nl]
            sv = self.le1[i](sle)
            v = sv if v is None else torch.cat((v, sv), dim=0)

        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def reinit(self):
        return NLGenerator_IN(
            self.params[0], self.params[1], self.params[2], self.params[3],
            label_emb=self.label_emb.data,
            le_emb_size=self.params[4],
            le_size=self.params[5],
            sbz=self.params[6],
            film_enabled=self.params[7],
            film_hidden=self.params[8],
        ).cuda()


