from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention1D(nn.Module):
    """U-Net jump gated attention (Attention Gate, 1D)."""

    def __init__(self, x_ch: int, g_ch: int, inter_ch: int):
        super().__init__()
        self.theta_x = nn.Conv1d(x_ch, inter_ch, kernel_size=1, bias=False)
        self.phi_g = nn.Conv1d(g_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv1d(inter_ch, 1, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_skip: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x_skip: [B, Cx, T]
        # g: [B, Cg, T]
        att = self.act(self.theta_x(x_skip) + self.phi_g(g))
        alpha = self.sigmoid(self.psi(att))  # [B,1,T]
        return x_skip * alpha


class GatedAttentionBypass(nn.Module):
    """Use when use_gated_attention is turned off: the same interface as GatedAttention1D, directly returns skip."""

    def forward(self, x_skip: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return x_skip


class CBAM1D(nn.Module):
    """CBAM (1D version).

    Standard CBAM consists of channel attention and spatial attention, and only returns weighted features.
    """

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)

        # 1x1 Conv
        self.channel_fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )

        # avg/max 2 Conv1d [B,1,T]
        self.spatial_conv = nn.Conv1d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        # Comment translated from Chinese.
        avg_pool = torch.mean(x, dim=-1, keepdim=True)  # [B,C,1]
        max_pool, _ = torch.max(x, dim=-1, keepdim=True)  # [B,C,1]
        ca = torch.sigmoid(self.channel_fc(avg_pool) + self.channel_fc(max_pool))  # [B,C,1]
        x_channel = x * ca

        # spatial
        avg_t = torch.mean(x_channel, dim=1, keepdim=True)  # [B,1,T]
        max_t, _ = torch.max(x_channel, dim=1, keepdim=True)  # [B,1,T]
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_t, max_t], dim=1)))  # [B,1,T]
        return x_channel * sa


class DepthwiseSeparableConv1d(nn.Module):
    """1D depthwise separable convolution: DepthwiseConv + PointwiseConv."""

    def __init__(self, in_ch: int, out_ch: int, k: int, dropout: float = 0.0):
        super().__init__()
        pad = k // 2
        # Comment translated from Chinese.
        self.depthwise = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size=k,
            padding=pad,
            groups=in_ch,
            bias=False,
        )
        # pointwise 1x1
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MultiScaleConv1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernels=(7,),
        dropout: float = 0.0,
        use_separable: bool = False,
        use_cbam: bool = False,
    ):
        super().__init__()

        branch_cls = DepthwiseSeparableConv1d if use_separable else None
        branches = []
        for k in kernels:
            if branch_cls is None:
                layers: list[nn.Module] = [
                    nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                ]
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
                branches.append(nn.Sequential(*layers))
            else:
                branches.append(DepthwiseSeparableConv1d(in_ch, out_ch, k, dropout=dropout))
        self.branches = nn.ModuleList(branches)
        n_scales = len(kernels)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch * n_scales, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM1D(out_ch, reduction=8, spatial_kernel=7) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        out = self.fuse(torch.cat(feats, dim=1))
        return self.cbam(out)


class PhaseNetUNet(nn.Module):
    """PhaseNet style 1D U‑Net (ported according to the official source code UNet structure)."""

    def __init__(
        self,
        in_ch: int = 3,
        n_class: int = 3,
        depths: int = 5,
        filters_root: int = 8,
        kernels=(7,),
        pool_size: int = 4,
        drop_rate: float = 0.0,
        use_cbam: bool = False,
        use_separable: bool = False,
        use_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        assert depths >= 1
        self.depths = depths
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.kernel_size = kernels[0] if len(kernels) == 1 else 7
        self.kernels = kernels
        self.use_cbam = use_cbam
        self.use_separable = use_separable
        self.use_gated_attention = use_gated_attention

        def conv_block(in_c: int, out_c: int) -> nn.Module:
            if (
                len(self.kernels) == 1
                and self.kernels[0] == self.kernel_size
                and not self.use_cbam
            ):
                layers: list[nn.Module] = [
                    nn.Conv1d(in_c, out_c, self.kernel_size, padding=self.kernel_size // 2, bias=False),
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(inplace=True),
                ]
                if drop_rate and drop_rate > 0.0:
                    layers.append(nn.Dropout(p=drop_rate))
                return nn.Sequential(*layers)

            return MultiScaleConv1d(
                in_ch=in_c,
                out_ch=out_c,
                kernels=self.kernels,
                dropout=drop_rate,
                use_separable=self.use_separable,
                use_cbam=self.use_cbam,
            )

        in_layers: list[nn.Module] = [
            nn.Conv1d(in_ch, filters_root, self.kernel_size, padding=self.kernel_size // 2, bias=False),
            nn.BatchNorm1d(filters_root),
            nn.ReLU(inplace=True),
        ]
        if drop_rate and drop_rate > 0.0:
            in_layers.append(nn.Dropout(p=drop_rate))
        self.input_conv = nn.Sequential(*in_layers)

        downs: list[nn.Module] = []
        pools: list[nn.Module] = []
        ch_in = filters_root
        for d in range(depths):
            filters = int((2 ** d) * filters_root)
            downs.append(conv_block(ch_in, filters))
            ch_in = filters
            if d < depths - 1:
                pool_layers: list[nn.Module] = [
                    nn.Conv1d(
                        ch_in,
                        ch_in,
                        kernel_size=self.kernel_size,
                        stride=pool_size,
                        padding=self.kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(ch_in),
                    nn.ReLU(inplace=True),
                ]
                if drop_rate and drop_rate > 0.0:
                    pool_layers.append(nn.Dropout(p=drop_rate))
                pools.append(nn.Sequential(*pool_layers))
        self.downs = nn.ModuleList(downs)
        self.pools = nn.ModuleList(pools)

        ups: list[nn.Module] = []
        up_convs: list[nn.Module] = []
        att_gates: list[nn.Module] = []
        ch = ch_in
        for d in reversed(range(depths - 1)):
            filters = int((2 ** d) * filters_root)
            up_convs.append(
                nn.ConvTranspose1d(ch, filters, kernel_size=pool_size, stride=pool_size, padding=0, bias=False)
            )
            if self.use_gated_attention:
                inter_ch = max(1, filters // 2)
                att_gates.append(GatedAttention1D(x_ch=filters, g_ch=filters, inter_ch=inter_ch))
            else:
                att_gates.append(GatedAttentionBypass())
            ch = filters * 2
            ups.append(conv_block(ch, filters))
            ch = filters
        self.up_convs = nn.ModuleList(up_convs)
        self.att_gates = nn.ModuleList(att_gates)
        self.ups = nn.ModuleList(ups)

        self.head = nn.Conv1d(ch, n_class, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        downs: list[torch.Tensor] = []
        for i, block in enumerate(self.downs):
            x = block(x)
            downs.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)

        for i, (up_conv, att_gate, up_block) in enumerate(zip(self.up_convs, self.att_gates, self.ups)):
            x = up_conv(x)
            skip = downs[-(i + 2)]
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                pad_left = diff // 2
                pad_right = diff - pad_left
                x = F.pad(x, (pad_left, pad_right))
            skip = att_gate(skip, x)
            x = torch.cat([x, skip], dim=1)
            x = up_block(x)

        logits = self.head(x)
        return logits

