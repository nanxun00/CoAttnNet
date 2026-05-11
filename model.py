"""
Model definition (PyTorch version)

This paper implements a one-dimensional U‑Net model for phase (background/P/S) probabilistic prediction of seismic three-component sequences.
Key points:
- Depthwise Separable Conv: Decompose standard convolution into depth convolution (grouping = number of channels) + point convolution (1x1), reducing parameters and increasing speed;
- Multi-scale one-dimensional convolution (temporal kernel size 3/7/15 parallel) to enhance feature extraction capabilities at different time scales;
- CBAM attention (Channel + Spatial Attention for 1D): introduced at the decoding end after "skip connection splicing" and bottleneck to improve feature expression;
- Typical U‑Net encoding and decoding structure: down-sampling on the encoding side extracts semantic information, and up-sampling on the decoding side is spliced ​​with skip connections of the corresponding layers;
- Skip connection is implemented in DecoderBlock.forward through cat([skip, upsampled], dim=1);
- 输出为 [B, K, T] 的未归一化 logits，后续在损失函数内做 softmax/log_softmax。

Innovation points:
1. Dynamic threshold adjustment mechanism: the environment awareness network evaluates data quality and outputs recommended thresholds
2. Feature transfer optimization: feature selection and transformation on skip connections, and dense connection topology

Notice:
- Input tensor shape is [B, C, T] (batch, channel, time);
- Since multiple downsampling/upsampling may introduce rounding differences, the skip is center-cropped before splicing to ensure consistent time dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_padding_1d(kernel_size: int, dilation: int = 1) -> int:
    """Computes "nearly same" padding for 1D convolutions (only for odd-numbered kernels)."""
    return (kernel_size // 2) * dilation


class ConvBNReLU1d(nn.Module):
    """(Reserved in case needed) Standard convolution block: Conv1d + BN + ReLU (+ Dropout).

    Note: This model has been switched to "depth separable convolution" in the multi-scale module. If you need to change back to standard convolution, you can use this module.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=1,
            padding=_same_padding_1d(k, dilation),
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        """Forward: keep the time length unchanged (because padding is set to same)."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution block (1D) = depthwise convolution (groups=C) + point convolution (1x1).

    structure:
    - Depthwise Conv1d(in_ch->in_ch, groups=in_ch) + BN + ReLU
    - Pointwise Conv1d(in_ch->out_ch, kernel=1) + BN + ReLU + Dropout
    """

    def __init__(self, in_ch: int, out_ch: int, k: int, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        pad = _same_padding_1d(k, dilation)
        self.dw = nn.Conv1d(
            in_ch, in_ch, kernel_size=k, padding=pad, dilation=dilation, groups=in_ch, bias=False
        )
        self.dw_bn = nn.BatchNorm1d(in_ch)
        self.dw_act = nn.ReLU(inplace=True)

        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_ch)
        self.pw_act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_act(self.pw_bn(self.pw(x)))
        x = self.drop(x)
        return x


class MultiScaleConv1d(nn.Module):
    """Multi-scale 1D convolution module (using depthwise separable convolution)

    Concatenate convolutions of different time scales (kernel sizes) in parallel (Depthwise Separable), and then fuse them through 1x1 convolution,
    to simultaneously capture short/medium/long time range features.
    """

    def __init__(self, in_ch: int, out_ch: int, kernels=(3, 7, 15), dropout: float = 0.0, use_separable: bool = True):
        super().__init__()
        # Use standard convolution or depthwise separable convolution depending on the switch
        branch_cls = DepthwiseSeparableConv1d if use_separable else ConvBNReLU1d
        self.branches = nn.ModuleList([branch_cls(in_ch, out_ch, k, dropout=dropout) for k in kernels])
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch * len(kernels), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward: multi-branch convolution -> channel dimension splicing -> 1x1 fusion."""
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        return x


class CBAM1D(nn.Module):
    """CBAM attention module (1D version): channel attention + spatial attention.

    - Channel Attention: global pooling (avg/max) → shared MLP → addition → sigmoid;
    - Spatial Attention: Find avg/max → splicing (2xT) → Conv1d(k) → sigmoid along the channel.
    """

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        # Shared MLP: C -> C/reduction -> C, use Conv1d instead of Linear to avoid CuBLAS
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv1d(2, 1, kernel_size=spatial_kernel, padding=_same_padding_1d(spatial_kernel), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        # channel attention
        avg_pool = torch.mean(x, dim=-1, keepdim=True)              # [B, C, 1]
        max_pool, _ = torch.max(x, dim=-1, keepdim=True)            # [B, C, 1]
        avg_pool = torch.nan_to_num(avg_pool.contiguous())
        max_pool = torch.nan_to_num(max_pool.contiguous())
        ca_input = (self.mlp(avg_pool) + self.mlp(max_pool)).contiguous()
        ca = torch.sigmoid(ca_input)
        x = x * ca
        # spatial attention
        avg_c = torch.mean(x, dim=1, keepdim=True)    # [B, 1, T]
        max_c, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, T]
        cat_spatial = torch.cat([avg_c, max_c], dim=1).contiguous()
        sa = torch.sigmoid(self.spatial(cat_spatial))  # [B,1,T]
        x = x * sa
        return x


class ResidualBlock1D(nn.Module):
    """Lightweight 1D residual block: Conv-BN-ReLU-Conv-BN + residual connection"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + identity, inplace=True)
        return out


def _extract_stft_band_ratios(x: torch.Tensor, sr: float = 100.0, n_fft: int = 256) -> torch.Tensor:
    """Extract low-frequency/medium-frequency/high-frequency energy ratio from waveform (Typical P wave 5-20Hz, noise <1Hz or >30Hz)
    Args:
        x: [B, C, T] input waveform (take the previous segment or the entire segment)
    Returns:
        [B, 3] Three-band energy ratio (low<5Hz, mid 5-20Hz, high>20Hz), the sum of each row is 1
    """
    B, C, T = x.shape
    x_mean = x.mean(dim=1)  # [B, T]
    if T < n_fft:
        return torch.zeros(B, 3, device=x.device, dtype=x.dtype)
    hop = max(1, n_fft // 2)
    s = torch.stft(x_mean, n_fft=n_fft, hop_length=hop, return_complex=True, center=False)
    mag = torch.abs(s)  # [B, n_fft//2+1, n_frames]
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(x.device)
    n_bins = mag.shape[1]
    low_mask = freqs < 5.0
    mid_mask = (freqs >= 5.0) & (freqs < 20.0)
    high_mask = freqs >= 20.0
    low_mask = low_mask[:n_bins]
    mid_mask = mid_mask[:n_bins]
    high_mask = high_mask[:n_bins]
    e_low = (mag * low_mask.unsqueeze(0).unsqueeze(-1)).sum(dim=(1, 2), keepdim=True)   # [B, 1, 1]
    e_mid = (mag * mid_mask.unsqueeze(0).unsqueeze(-1)).sum(dim=(1, 2), keepdim=True)
    e_high = (mag * high_mask.unsqueeze(0).unsqueeze(-1)).sum(dim=(1, 2), keepdim=True)
    total = e_low + e_mid + e_high + 1e-8
    out = torch.cat([e_low / total, e_mid / total, e_high / total], dim=1)  # [B, 3, 1]
    return out.squeeze(-1)  # [B, 3]


class FeatureFusionNet(nn.Module):
    """Feature branch: fuse global/multi-segment/STFT features, estimate P/S uncertainty (u_feat ∈ [0,1]^2)."""

    def __init__(
        self,
        in_ch: int = 3,
        context_len: int = 500,
        hidden_dim: int = 64,
        use_multi_segment: bool = False,
        use_stft: bool = False,
        n_segments: int = 5,
        segment_len: int = 600,
        sr: float = 100.0,
    ):
        super().__init__()
        self.context_len = context_len
        self.use_multi_segment = use_multi_segment
        self.use_stft = use_stft
        self.n_segments = n_segments
        self.segment_len = segment_len
        self.sr = sr

        self.feature_extractor = nn.Sequential(
            ResidualBlock1D(in_ch, 32),
            nn.AdaptiveAvgPool1d(1),
        )

        if use_multi_segment:
            self.segment_extractor = nn.Sequential(
                nn.Conv1d(in_ch, 32, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )

        feat_dim = 32
        if use_multi_segment:
            feat_dim += 32 * n_segments
        if use_stft:
            feat_dim += 3

        # 1x1 Conv CUBLAS sgemm INVALID_VALUE CBAM
        self.noise_estimator = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 2, kernel_size=1, bias=True),
        )
        self.signal_estimator = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        head_in_dim = feat_dim + 2 + 1 + 2
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(head_in_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, probs: torch.Tensor | None = None) -> dict:
        B, C, T = x.shape
        x_global = x[..., :min(self.context_len, T)]
        feat = self.feature_extractor(x_global).squeeze(-1)

        if self.use_multi_segment:
            # Always divide according to the fixed segment length segment_len, and fill in zeros if necessary to avoid inconsistency between dimensions and feat_dim
            seg_len = self.segment_len
            seg_feats = []
            for i in range(self.n_segments):
                start = i * seg_len
                end = start + seg_len
                if start >= T:
                    seg = x.new_zeros((B, C, seg_len))
                else:
                    seg = x[..., start:min(end, T)]
                    if seg.shape[-1] < seg_len:
                        seg = F.pad(seg, (0, seg_len - seg.shape[-1]))
                f = self.segment_extractor(seg).squeeze(-1)  # [B, 32]
                seg_feats.append(f)
            feat = torch.cat([feat] + seg_feats, dim=1)

        if self.use_stft:
            x_stft = x[..., :min(T, self.context_len)] if T > self.context_len else x
            stft_feat = _extract_stft_band_ratios(x_stft, sr=self.sr)
            feat = torch.cat([feat, stft_feat], dim=1)

        feat_1d = feat.unsqueeze(-1)  # [B, feat_dim, 1]
        noise_stats = self.noise_estimator(feat_1d).squeeze(-1)     # [B, 2]
        signal_strength = self.signal_estimator(feat_1d).squeeze(-1)  # [B, 1]

        if probs is not None and probs.shape[1] >= 3:
            max_prob_p = probs[:, 1, :].detach().max(dim=-1, keepdim=True)[0]
            max_prob_s = probs[:, 2, :].detach().max(dim=-1, keepdim=True)[0]
        else:
            device = feat.device
            max_prob_p = torch.full((B, 1), 0.5, device=device, dtype=feat.dtype)
            max_prob_s = torch.full((B, 1), 0.5, device=device, dtype=feat.dtype)

        combined = torch.cat([feat, noise_stats, signal_strength, max_prob_p, max_prob_s], dim=1)
        combined = torch.nan_to_num(combined).float().contiguous()
        combined = combined.unsqueeze(-1)  # [B, D, 1]
        with torch.cuda.amp.autocast(enabled=False):
            u_feat = self.uncertainty_head(combined).squeeze(-1)

        return {
            "noise_stats": noise_stats,
            "signal_strength": signal_strength,
            "u_feat": u_feat,
        }


class UncertaintyHead(nn.Module):
    """Independent uncertainty head: learning sample-level u_score based on probs + signal_strength."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        # Use 1×1 Conv instead of Linear, bypassing cuBLAS INVALID_VALUE
        self.fc1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True)

    def forward(self, probs: torch.Tensor, signal_strength: torch.Tensor | None = None) -> torch.Tensor:
        B = probs.shape[0]
        max_prob_p = probs[:, 1, :].max(dim=-1, keepdim=True)[0]
        max_prob_s = probs[:, 2, :].max(dim=-1, keepdim=True)[0]
        if signal_strength is None:
            signal_strength = probs.new_zeros((B, 1))
        combined = torch.cat([max_prob_p, max_prob_s, signal_strength], dim=1)
        combined = torch.nan_to_num(combined).float().contiguous()
        combined = combined.unsqueeze(-1)  # [B, 3, 1]
        with torch.cuda.amp.autocast(enabled=False):
            x = F.relu(self.fc1(combined))
            x = torch.nan_to_num(x).contiguous()
            u_score = torch.sigmoid(self.fc2(x))
        return u_score.squeeze(-1).squeeze(-1)


class EvidentialHead(nn.Module):
    """Evidential Learning output header: Predict evidence to build Dirichlet distribution."""

    def __init__(self, in_ch: int, hidden_dim: int = 64, n_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        evidence = F.softplus(self.fc2(x))
        return evidence

class ContextAwareNetwork(nn.Module):
    """Environment-aware network: evaluate data quality and output recommended thresholds
    
    Supports multi-segment local feature (use_multi_segment) and frequency domain feature (use_stft) enhancement.
    Receive a waveform and output three key indicators:
    1. Statistical characteristics of background noise (noise_stats)
    2. Potential strength of the signal (signal_strength)
    3. Suggested judgment threshold (suggested_threshold)
    """
    
    def __init__(
        self,
        in_ch: int = 3,
        context_len: int = 500,
        hidden_dim: int = 64,
        use_multi_segment: bool = False,
        use_stft: bool = False,
        n_segments: int = 5,
        segment_len: int = 600,
        sr: float = 100.0,
        predict_uncertainty: bool = False,
    ):
        """
        Args:
            predict_uncertainty: If True, output uncertainty [0,1] and then map it to a threshold; if False, output the threshold directly (original implementation)
        """
        super().__init__()
        self.context_len = context_len
        self.use_multi_segment = use_multi_segment
        self.predict_uncertainty = predict_uncertainty
        self._thr_min, self._thr_max = 0.42, 0.70
        self.use_stft = use_stft
        self.n_segments = n_segments
        self.segment_len = segment_len
        self.sr = sr
        
        # Global feature extraction (before context_len point, retain original logic)
        self.feature_extractor = nn.Sequential(
            ResidualBlock1D(in_ch, 32),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Multi-segment local feature extraction (each segment is independent Conv → pool)
        if use_multi_segment:
            self.segment_extractor = nn.Sequential(
                nn.Conv1d(in_ch, 32, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
        
        # Calculate total feature dimensions
        feat_dim = 32
        if use_multi_segment:
            feat_dim += 32 * n_segments
        if use_stft:
            feat_dim += 3
        
        # Use 1x1 Conv to avoid CUBLAS sgemm related INVALID_VALUE
        self.noise_estimator = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 2, kernel_size=1, bias=True),
        )
        
        self.signal_estimator = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        
        # Input: feat + noise_stats(2) + signal_strength(1) + max_prob_p(1) + max_prob_s(1)
        self._threshold_input_dim = feat_dim + 2 + 1 + 2
        # P/S separates predictors to avoid shared network + similarity loss leading to merged solutions
        self.threshold_predictor_p = nn.Sequential(
            nn.Conv1d(self._threshold_input_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.threshold_predictor_s = nn.Sequential(
            nn.Conv1d(self._threshold_input_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        for pred in [self.threshold_predictor_p, self.threshold_predictor_s]:
            last_linear = pred[-2]
            if isinstance(last_linear, nn.Linear):
                nn.init.xavier_uniform_(last_linear.weight, gain=0.5)
                nn.init.uniform_(last_linear.bias, 0.2, 0.6)

    def forward(self, x: torch.Tensor, probs: torch.Tensor = None) -> dict:
        """
        Args:
            x: [B, C, T] input waveform
            probs: [B, K, T] Probabilistic output of the main UNet (optional). If provided, extract max_prob_p/max_prob_s
                   As sample-level features (see ASNet ITH), enabling threshold predictions to be adaptive based on model confidence.
        
        Returns:
            dict: noise_stats, signal_strength, suggested_threshold [B, 2]
        """
        B, C, T = x.shape
        
        # Global features: the first context_len point (or the entire segment if T <= context_len)
        x_global = x[..., :min(self.context_len, T)]
        feat = self.feature_extractor(x_global)  # [B, 32, 1]
        feat = feat.squeeze(-1)  # [B, 32]
        
        # Multi-segment local features: The entire segment is divided into n_segments, and each segment extracts features independently.
        # Always collect n_segments segments (pad if insufficient) to avoid linear errors caused by inconsistent dimensions and feat_dim
        if self.use_multi_segment:
            seg_len = self.segment_len
            seg_feats = []
            for i in range(self.n_segments):
                start = i * seg_len
                end = start + seg_len
                if start >= T:
                    seg = x.new_zeros((B, C, seg_len))
                else:
                    seg = x[..., start:min(end, T)]
                    if seg.shape[-1] < seg_len:
                        seg = F.pad(seg, (0, seg_len - seg.shape[-1]))
                f = self.segment_extractor(seg).squeeze(-1)  # [B, 32]
                seg_feats.append(f)
            feat = torch.cat([feat] + seg_feats, dim=1)  # [B, 32 + 32*n_segments]
        
        # Frequency domain features: do STFT on the first context_len or the entire segment
        if self.use_stft:
            x_stft = x[..., :min(T, self.context_len)] if T > self.context_len else x
            stft_feat = _extract_stft_band_ratios(x_stft, sr=self.sr)  # [B, 3]
            feat = torch.cat([feat, stft_feat], dim=1)
        
        feat_1d = feat.unsqueeze(-1)  # [B, feat_dim, 1]
        noise_stats = self.noise_estimator(feat_1d).squeeze(-1)     # [B, 2]
        signal_strength = self.signal_estimator(feat_1d).squeeze(-1)  # [B, 1]
        
        # Sample-level features: main UNet’s max_prob (refer to ASNet’s ITH), making the threshold adaptive with model confidence
        if probs is not None and probs.shape[1] >= 3:
            max_prob_p = probs[:, 1, :].max(dim=-1, keepdim=True)[0]  # [B, 1]
            max_prob_s = probs[:, 2, :].max(dim=-1, keepdim=True)[0]  # [B, 1]
        else:
            # Use 0.5 placeholder when there are no probs (compatible with old calls)
            device = feat.device
            max_prob_p = torch.full((B, 1), 0.5, device=device, dtype=feat.dtype)
            max_prob_s = torch.full((B, 1), 0.5, device=device, dtype=feat.dtype)
        
        combined = torch.cat([feat, noise_stats, signal_strength, max_prob_p, max_prob_s], dim=1)
        combined = torch.nan_to_num(combined).contiguous()
        combined = combined.unsqueeze(-1)  # [B, D, 1]
        raw_p = self.threshold_predictor_p(combined).squeeze(-1)  # [B, 1]
        raw_s = self.threshold_predictor_s(combined).squeeze(-1)  # [B, 1]
        raw = torch.cat([raw_p, raw_s], dim=1)  # [B, 2]
        if self.predict_uncertainty:
            # Output uncertainty [0,1], and then rule-based mapping to threshold. u↑→thr↓, no gradient collapse
            suggested_threshold = (self._thr_max - (self._thr_max - self._thr_min) * raw).clamp(self._thr_min, self._thr_max)
        else:
            # Original implementation: direct output threshold [0.42, 0.70]
            suggested_threshold = 0.42 + 0.28 * raw

        out = {
            'noise_stats': noise_stats,
            'signal_strength': signal_strength,
            'suggested_threshold': suggested_threshold,
        }
        if self.predict_uncertainty:
            out['uncertainty'] = raw  # [B, 2] Raw uncertainty, for optional calibration loss
        return out


class SkipConnectionRefinement(nn.Module):
    """Jump connection feature purification module
    
    "Purification" of features before delivery involves two steps:
    1. Feature selection: Identifying truly useful parts through a small attention mechanism
    2. Feature transformation: The selected features undergo nonlinear transformation to make them more suitable for fusion with deep features.
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Args:
            channels: input feature channel number
            reduction: dimensionality reduction ratio of the attention mechanism
        """
        super().__init__()
        hidden = max(1, channels // reduction)
        
        # Feature Selection: Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # Feature transformation: nonlinear transformation
        self.transform = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),  # Depth convolution
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),  # point convolution
            nn.BatchNorm1d(channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input features
        
        Returns:
            [B, C, T] Purified features
        """
        # Feature selection: through attention mechanism
        ca = self.channel_attention(x)  # [B, C, 1]
        x_selected = x * ca  # [B, C, T]
        
        # Feature transformation
        x_transformed = self.transform(x_selected)
        
        # residual connection
        x_refined = x_selected + x_transformed
        
        return x_refined


class EncoderBlock(nn.Module):
    """Encoder block: multi-scale convolution + MaxPool downsampling.

    Return value: (x_down, x_skip)
    - x_down: The downsampled features are passed to deeper layers;
    - x_skip: Features before downsampling, used for decoding end skip connection.
    """

    def __init__(self, in_ch: int, out_ch: int, kernels=(3, 7, 15), dropout: float = 0.0, pool_stride: int = 4, use_separable: bool = True):
        super().__init__()
        self.ms = MultiScaleConv1d(in_ch, out_ch, kernels=kernels, dropout=dropout, use_separable=use_separable)
        self.pool = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)

    def forward(self, x):
        """Forward: extract features first, then save jump connection features, and finally do downsampling."""
        x = self.ms(x)
        skip = x
        x = self.pool(x)
        return x, skip


def center_crop_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Center-aligned crop/padding to target length (time dimension)."""
    cur = x.shape[-1]
    if cur == target_len:
        return x
    if cur < target_len:
        # Pad both ends with zeros to the target length
        pad_total = target_len - cur
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(x, (pad_left, pad_right))
    # Crop to the middle
    start = (cur - target_len) // 2
    return x[..., start:start + target_len]


class DecoderBlock(nn.Module):
    """Decoder block: transposed convolution upsampling + skip connection stitching (+ CBAM) + multi-scale convolution fusion.
    
    Supports dense connections: can receive features from multiple encoding layers (dense_skips)
    Support skip connection optimization: purify skip features before splicing
    """

    def __init__(
        self, 
        skip_ch: int, 
        in_ch: int, 
        out_ch: int, 
        kernels=(3, 7, 15), 
        dropout: float = 0.0, 
        up_stride: int = 4, 
        use_cbam: bool = True, 
        cbam_reduction: int = 8, 
        cbam_spatial_kernel: int = 7, 
        use_separable: bool = True,
        use_skip_refinement: bool = False,
        dense_skip_chs: list = None,  # List of extra skip channels for dense connections
    ):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=up_stride, stride=up_stride)
        self.use_cbam = use_cbam
        self.use_skip_refinement = use_skip_refinement
        
        # Calculate the total number of input channels: upsampled features + main skip + densely connected skips
        dense_skip_chs = dense_skip_chs or []
        total_skip_ch = skip_ch + sum(dense_skip_chs)
        total_ch = out_ch + total_skip_ch
        
        # Skip connection optimization module
        if use_skip_refinement:
            self.skip_refinement = SkipConnectionRefinement(skip_ch, reduction=4)
            if dense_skip_chs:
                self.dense_refinements = nn.ModuleList([
                    SkipConnectionRefinement(ch, reduction=4) for ch in dense_skip_chs
                ])
            else:
                self.dense_refinements = nn.ModuleList()
        else:
            self.skip_refinement = nn.Identity()
            self.dense_refinements = nn.ModuleList()
        
        self.cbam = CBAM1D(total_ch, reduction=cbam_reduction, spatial_kernel=cbam_spatial_kernel) if use_cbam else nn.Identity()
        self.ms = MultiScaleConv1d(total_ch, out_ch, kernels=kernels, dropout=dropout, use_separable=use_separable)

    def forward(self, x, skip, dense_skips=None):
        """Forward:
        1) Upsampling to higher resolution;
        2) Center-crop the feature skip of the corresponding layer at the encoding end to make the time dimension consistent with that after upsampling;
        3) If dense connections are enabled, handle additional skip features;
        4) Purify skip features (if enabled);
        5) After channel dimension splicing, fuse through multi-scale convolution.
        
        Args:
            x: Features before upsampling [B, in_ch, T_low]
            skip: main skip connection feature [B, skip_ch, T_high]
            dense_skips: A list of additional skip features for dense connections, each element is [B, ch, T_high]
        """
        x = self.up(x)  # [B, out_ch, T_high]
        T_target = x.shape[-1]
        
        # Handle main skip
        skip = center_crop_1d(skip, T_target)
        if self.use_skip_refinement:
            skip = self.skip_refinement(skip)
        
        # Handling densely connected skips
        skip_list = [skip]
        if dense_skips is not None:
            for i, dense_skip in enumerate(dense_skips):
                dense_skip = center_crop_1d(dense_skip, T_target)
                if self.use_skip_refinement and i < len(self.dense_refinements):
                    dense_skip = self.dense_refinements[i](dense_skip)
                skip_list.append(dense_skip)
        
        # Splice all features
        x = torch.cat([x] + skip_list, dim=1)  # [B, total_ch, T_high]
        
        if self.use_cbam:
            x = self.cbam(x)  # attention enhancement fusion
        
        x = self.ms(x)
        return x


class UNet1D(nn.Module):
    """One-dimensional U‑Net backbone (with skip connections, depth-separable convolution, CBAM optional)
    
    Innovative features:
    1. Environment-aware network: evaluate data quality and output recommended thresholds
    2. Dense connection: Each decoding layer can receive features from multiple encoding layers
    3. Skip connection optimization: feature selection and transformation
    """

    def __init__(
        self,
        in_ch: int = 3,      # Number of input channels (default three components)
        n_class: int = 3,    # Number of categories (Background/P/S)
        base_ch: int = 16,   # Number of shallowest channels
        depth: int = 4,      # Number of encoding/decoding layers
        kernels=(3, 7, 15),  # multi-scale time kernel
        dropout: float = 0.0,
        factor: int = 4,     # Downsampling/upsampling step size
        use_cbam: bool = True,
        cbam_reduction: int = 8,
        cbam_spatial_kernel: int = 7,
        use_separable: bool = True,
        use_context_aware: bool = False,  # Whether to use context-aware networks
        use_dense_connection: bool = False,  # Whether to use dense connections
        use_skip_refinement: bool = False,  # Whether to use skip connection optimization
        context_len: int = 500,  # Context length for environment-aware networks
        context_hidden_dim: int = 64,  # Hidden layer dimensions of environment-aware networks
        use_context_multi_segment: bool = False,  # Environment perception: multi-segment local features
        use_context_stft: bool = False,  # Environmental Perception: Frequency Domain Characteristics
        context_n_segments: int = 5,
        context_segment_len: int = 600,
        context_sr: float = 100.0,
        use_context_predict_uncertainty: bool = False,  # Prediction uncertainty remapping threshold (for ablation)
        use_feature_fusion_uncertainty: bool = False,
        feature_fusion_lambda: float = 0.5,
        use_feature_conditioning: bool = False,
        use_uncertainty_head: bool = False,
        uncertainty_head_hidden_dim: int = 64,
        use_evidential_head: bool = False,
        evidential_hidden_dim: int = 64,
        use_tcn_refine: bool = False,
        tcn_refine_dropout: float = 0.1,
        use_transformer_refine: bool = False,
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_layers: int = 1,
        transformer_dropout: float = 0.1,
        use_split_heads: bool = False,
    ):
        super().__init__()
        assert depth >= 1
        self.depth = depth
        self.factor = factor
        self.use_cbam = use_cbam
        self.cbam_reduction = cbam_reduction
        self.cbam_spatial_kernel = cbam_spatial_kernel
        self.use_separable = use_separable
        self.use_context_aware = use_context_aware
        self.use_dense_connection = use_dense_connection
        self.use_skip_refinement = use_skip_refinement
        self.use_feature_fusion_uncertainty = use_feature_fusion_uncertainty
        self.feature_fusion_lambda = feature_fusion_lambda
        self.use_feature_conditioning = use_feature_conditioning
        self.use_uncertainty_head = use_uncertainty_head
        self._uncertainty_head_hidden_dim = uncertainty_head_hidden_dim
        self.use_evidential_head = use_evidential_head
        self.use_tcn_refine = use_tcn_refine
        self.use_transformer_refine = use_transformer_refine
        self.use_split_heads = use_split_heads

        # environment aware network
        if use_context_aware:
            self.context_net = ContextAwareNetwork(
                in_ch=in_ch,
                context_len=context_len,
                hidden_dim=context_hidden_dim,
                use_multi_segment=use_context_multi_segment,
                use_stft=use_context_stft,
                n_segments=context_n_segments,
                segment_len=context_segment_len,
                sr=context_sr,
                predict_uncertainty=use_context_predict_uncertainty,
            )
        else:
            self.context_net = None

        if self.use_feature_fusion_uncertainty or self.use_feature_conditioning:
            self.feature_fusion_net = FeatureFusionNet(
                in_ch=in_ch,
                context_len=context_len,
                hidden_dim=context_hidden_dim,
                use_multi_segment=use_context_multi_segment,
                use_stft=use_context_stft,
                n_segments=context_n_segments,
                segment_len=context_segment_len,
                sr=context_sr,
            )
        else:
            self.feature_fusion_net = None

        # Encoder
        enc = []  # Encoder module list
        ch_in = in_ch
        chs = []
        for d in range(depth):
            ch_out = base_ch * (2 ** d)
            enc.append(EncoderBlock(ch_in, ch_out, kernels=kernels, dropout=dropout, pool_stride=factor, use_separable=use_separable))
            chs.append(ch_out)
            ch_in = ch_out
        self.enc = nn.ModuleList(enc)

        film_ch = ch_in * 2
        if self.use_feature_conditioning:
            self.film_gamma = nn.Conv1d(1, film_ch, kernel_size=1, bias=True)
            self.film_beta = nn.Conv1d(1, film_ch, kernel_size=1, bias=True)
        else:
            self.film_gamma = None
            self.film_beta = None

        # Bottleneck (bottom-level semantic aggregation) + optional CBAM
        self.bottleneck = MultiScaleConv1d(ch_in, ch_in * 2, kernels=kernels, dropout=dropout, use_separable=use_separable)
        self.bottleneck_cbam = CBAM1D(ch_in * 2, reduction=cbam_reduction, spatial_kernel=cbam_spatial_kernel) if use_cbam else nn.Identity()

        # Decoder (symmetrical to the encoder, upsampling layer by layer in reverse order and fused with the corresponding skip)
        dec = []
        ch_cur = ch_in * 2
        for d in reversed(range(depth)):
            skip_ch = chs[d]
            ch_out = chs[d]
            
            # Calculate the number of extra skip channels for dense connections
            dense_skip_chs = None
            if use_dense_connection:
                # Each decoding layer can receive features from all shallow coding layers
                # For example: decoder[0] receives encoder[0,1,2,3], decoder[1] receives encoder[1,2,3], etc.
                dense_skip_chs = [chs[i] for i in range(d, depth) if i != d]
            
            dec.append(DecoderBlock(
                skip_ch,
                ch_cur,
                ch_out,
                kernels=kernels,
                dropout=dropout,
                up_stride=factor,
                use_cbam=use_cbam,
                cbam_reduction=cbam_reduction,
                cbam_spatial_kernel=cbam_spatial_kernel,
                use_separable=use_separable,
                use_skip_refinement=use_skip_refinement,
                dense_skip_chs=dense_skip_chs,
            ))
            ch_cur = ch_out
        self.dec = nn.ModuleList(dec)

        # Classification header: 1x1 convolution mapped to number of categories
        self.use_split_heads = use_split_heads
        if self.use_split_heads:
            self.head_bg = nn.Conv1d(ch_cur, 1, kernel_size=1)
            self.head_p = nn.Conv1d(ch_cur, 1, kernel_size=1)
            self.head_s = nn.Conv1d(ch_cur, 1, kernel_size=1)
        else:
            self.head = nn.Conv1d(ch_cur, n_class, kernel_size=1)
        self.uncertainty_head = None
        if self.use_evidential_head:
            self.evidential_head = EvidentialHead(in_ch=ch_cur, hidden_dim=evidential_hidden_dim, n_classes=n_class)
        else:
            self.evidential_head = None
        if self.use_tcn_refine:
            self.tcn_refiner = TCNRefiner(channels=n_class, dropout=tcn_refine_dropout)
        else:
            self.tcn_refiner = None
        if self.use_transformer_refine:
            self.transformer_refiner = TransformerRefiner(
                channels=n_class,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_layers,
                dropout=transformer_dropout,
            )
        else:
            self.transformer_refiner = None

    def forward(self, x, return_context=False):
        """Forward propagation

        parameter:
        - x: [B, C, T]
        - return_context: whether to return the output of the environment awareness network

        Process: encoding -> save each layer skip -> bottleneck -> decode (connect with skip layer by layer, support dense connection) -> classification header.
        return:
        - logits: [B, K, T]
        - context_info: (optional) output dictionary of the context-aware network
        """
        # Save the original input (required by the environment-aware network, and probs need to be passed in after getting logits in the main path)
        x_in = x
        context_info = None
        feature_info = None
        if self.use_feature_conditioning and self.feature_fusion_net is not None:
            feature_info = self.feature_fusion_net(x_in, probs=None)
        
        # coding
        skips = []
        for blk in self.enc:
            x, skip = blk(x)   # Encode and save skip connection features
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        if self.use_cbam:
            x = self.bottleneck_cbam(x)
        
        if self.use_feature_conditioning and feature_info is not None:
            signal_strength = feature_info.get("signal_strength")
            if signal_strength is not None and self.film_gamma is not None and self.film_beta is not None:
                ss = torch.nan_to_num(signal_strength.float()).unsqueeze(-1).contiguous()  # [B,1,1]
                gamma = self.film_gamma(ss)
                beta = self.film_beta(ss)
                x = x * (1.0 + gamma) + beta

        # Decoding (supports dense connections)
        for d_idx, blk in enumerate(self.dec):
            # Main skip (features of the corresponding layer)
            skip = skips[-(d_idx + 1)]  # Take from back to front
            
            # Extra skips for dense connections
            dense_skips = None
            if self.use_dense_connection:
                # Coding layer index corresponding to the current decoding layer index
                enc_idx = self.depth - 1 - d_idx
                # Get features of all shallow layers (including the current layer)
                dense_skips = [skips[i] for i in range(enc_idx + 1, len(skips))]
            
            x = blk(x, skip, dense_skips)
        
        # Classification header
        if self.use_split_heads:
            logits_bg = self.head_bg(x)
            logits_p = self.head_p(x)
            logits_s = self.head_s(x)
            logits = torch.cat([logits_bg, logits_p, logits_s], dim=1)
        else:
            logits = self.head(x)  # [B, K, T]
        if self.use_tcn_refine and self.tcn_refiner is not None:
            logits = self.tcn_refiner(logits)
        if self.use_transformer_refine and self.transformer_refiner is not None:
            logits = self.transformer_refiner(logits)
        probs = None
        if self.use_context_aware or self.use_uncertainty_head:
            probs = F.softmax(logits, dim=1)

        # Context-aware network: requires probs of the main path as sample-level features (refer to ASNet ITH)
        if self.use_context_aware and self.context_net is not None:
            context_info = self.context_net(x_in, probs=probs)

        if self.use_uncertainty_head:
            if self.uncertainty_head is None:
                self.uncertainty_head = UncertaintyHead(
                    input_dim=3,
                    hidden_dim=self._uncertainty_head_hidden_dim,
                ).to(logits.device)  # Dynamically created modules need to be on the same device as the input to avoid mat1 on cuda / weight on cpu
            signal_strength = (context_info or {}).get("signal_strength")
            if signal_strength is None:
                signal_strength = logits.new_zeros((logits.shape[0], 1))
            else:
                signal_strength = signal_strength.to(logits.device)
            u_score = self.compute_uncertainty_score(probs, signal_strength=signal_strength)
            ctx = context_info or {}
            ctx["uncertainty_score"] = u_score
            context_info = ctx

        if self.use_evidential_head and return_context:
            evidence = self.compute_evidence(x)
            if evidence is not None:
                ctx = context_info or {}
                ctx["evidence"] = evidence
                ctx["evidential_uncertainty"] = self.evidential_uncertainty(evidence)
                context_info = ctx

        if return_context and context_info is None:
            context_info = {}
        if return_context:
            return logits, context_info
        return logits

    def compute_feature_uncertainty(self, x: torch.Tensor, probs: torch.Tensor | None = None):
        """Auxiliary interface: Uncertainty estimation by FeatureFusionNet (for eval / training calibration)."""
        if self.feature_fusion_net is None:
            return None
        return self.feature_fusion_net(x, probs=probs)

    def compute_uncertainty_score(self, probs: torch.Tensor, signal_strength: torch.Tensor | None = None):
        """Scenario C: independent uncertainty head output u_score ∈ [0,1]."""
        if self.uncertainty_head is None:
            return None
        probs = torch.nan_to_num(probs.detach()).contiguous()
        if signal_strength is not None:
            signal_strength = torch.nan_to_num(signal_strength.detach()).contiguous()
        return self.uncertainty_head(probs, signal_strength)

    def compute_evidence(self, features: torch.Tensor):
        if self.evidential_head is None:
            return None
        agg = torch.mean(features, dim=-1)
        return self.evidential_head(agg)

    @staticmethod
    def evidential_uncertainty(evidence: torch.Tensor | None):
        if evidence is None:
            return None
        alpha = evidence + 1.0
        total = alpha.sum(dim=1, keepdim=True)
        u = evidence.shape[-1] / total
        return u


class DeepEnsemble(nn.Module):
    """Option B: Deep integration, uncertainty estimation through multi-model prediction and variance."""

    def __init__(self, n_models: int, model_class: type[nn.Module], *args, **kwargs):
        super().__init__()
        self.models = nn.ModuleList([model_class(*args, **kwargs) for _ in range(n_models)])

    def forward(self, x: torch.Tensor, return_context: bool = False):
        logits_list = [model(x) for model in self.models]
        stacked = torch.stack(logits_list, dim=0)
        logits = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False).clamp(min=1e-8)
        if return_context:
            return logits, {"variance": variance}
        return logits

    def ensemble_uncertainty(self, variance: torch.Tensor) -> torch.Tensor:
        var_p = variance[:, 1, :].max(dim=-1)[0]
        var_s = variance[:, 2, :].max(dim=-1)[0]
        return torch.stack([var_p, var_s], dim=1)


class TCNRefiner(nn.Module):
    """TCN smoothing module: multi-layer dilated conv + residual + dropout.

    Only the logits are changed, the spatial dimensions are not changed.
    """

    def __init__(self, channels: int, kernels: tuple[int, ...] = (3, 3, 3), dilations: tuple[int, ...] = (1, 2, 4), dropout: float = 0.1):
        super().__init__()
        assert len(kernels) == len(dilations)
        layers = []
        for k, d in zip(kernels, dilations):
            pad = _same_padding_1d(k, d)
            layers.append(nn.Conv1d(channels, channels, kernel_size=k, dilation=d, padding=pad, bias=False))
            layers.append(nn.BatchNorm1d(channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TransformerRefiner(nn.Module):
    """Transformer smooth module: per-channel projection -> self-attention -> projection."""

    def __init__(self, channels: int, d_model: int = 128, nhead: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.proj_in = nn.Conv1d(channels, d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Conv1d(d_model, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> (B, T, C)
        x_perm = torch.nan_to_num(x.contiguous())
        # Avoid triggering cublasSgemm INVALID_VALUE at half precision (especially on some CUDA/cuBLAS versions), force FP32 to be used as self-attn, and then return to the original dtype as needed
        with torch.cuda.amp.autocast(enabled=False):
            feat = self.proj_in(x_perm.float())  # [B, d_model, T]
            feat = feat.permute(0, 2, 1).contiguous()  # [B, T, d_model]
            refined = self.transformer(feat)
            refined = refined.permute(0, 2, 1).contiguous()  # back to [B, d_model, T]
            out = self.proj_out(refined).contiguous()
        return out.to(x.dtype).contiguous()

