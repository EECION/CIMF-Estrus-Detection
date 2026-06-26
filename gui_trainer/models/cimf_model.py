import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, Any, Tuple


class FourierAugmentation(nn.Module):
    def __init__(self, tau: float = 0.4):
        super().__init__()
        self.tau = tau

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fft_x1 = torch.fft.fft2(x1, dim=(-2, -1))
        fft_x2 = torch.fft.fft2(x2, dim=(-2, -1))
        amp_x1, phase_x1 = torch.abs(fft_x1), torch.angle(fft_x1)
        amp_x2, phase_x2 = torch.abs(fft_x2), torch.angle(fft_x2)
        enhanced_amp_x1 = (1 - self.tau) * amp_x1 + self.tau * amp_x2
        enhanced_amp_x2 = (1 - self.tau) * amp_x2 + self.tau * amp_x1
        enhanced_fft_x1 = enhanced_amp_x1 * torch.exp(1j * phase_x1)
        enhanced_fft_x2 = enhanced_amp_x2 * torch.exp(1j * phase_x2)
        enhanced_x1 = torch.fft.ifft2(enhanced_fft_x1, dim=(-2, -1)).real
        enhanced_x2 = torch.fft.ifft2(enhanced_fft_x2, dim=(-2, -1)).real
        return enhanced_x1, enhanced_x2


class BackboneNetwork(nn.Module):
    def __init__(self, backbone: str = "ResNet50-512", num_classes: int = 3):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        if backbone == "ResNet50-512":
            self.feature_dim = 512
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Linear(2048, 512)
        elif backbone == "ResNet50-64":
            self.feature_dim = 64
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Linear(2048, 64)
        elif backbone == "ViT-B-768":
            self.feature_dim = 768
            self.backbone = nn.Sequential(
                nn.AdaptiveAvgPool2d((14, 14)),
                nn.Flatten(),
                nn.Linear(14 * 14 * 3, 1024),
                nn.ReLU(),
                nn.Linear(1024, 768),
            )
            self.fc = nn.Identity()
        elif backbone == "Swin-T-768":
            self.feature_dim = 768
            self.backbone = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(7 * 7 * 3, 1024),
                nn.ReLU(),
                nn.Linear(1024, 768),
            )
            self.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        features = self.fc(features)
        return F.normalize(features, p=2, dim=1)


class MCAFModule(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, gamma: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.gamma = gamma
        self.eeg_proxies = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.audio_proxies = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.hybrid_proxies = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.eeg_proxies, gain=0.5)
        nn.init.xavier_uniform_(self.audio_proxies, gain=0.5)
        nn.init.xavier_uniform_(self.hybrid_proxies, gain=0.5)

    def compute_unified_metric_loss(
        self,
        features: torch.Tensor,
        proxies: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)
        features = F.normalize(features, p=2, dim=1)
        proxies = F.normalize(proxies, p=2, dim=1)
        similarities = torch.matmul(features, proxies.t())
        total_loss = 0.0
        valid_samples = 0
        for i in range(batch_size):
            anchor_label = labels[i]
            anchor_similarities = similarities[i]
            positive_sim = anchor_similarities[anchor_label]
            negative_mask = torch.arange(self.num_classes, device=device) != anchor_label
            negative_sims = anchor_similarities[negative_mask]
            if len(negative_sims) == 0:
                continue
            k = min(5, len(negative_sims))
            hard_negative_sims, _ = torch.topk(negative_sims, k, largest=True)
            gamma_scaled_positive = self.gamma * positive_sim
            gamma_scaled_negatives = self.gamma * hard_negative_sims
            max_scaled = torch.max(gamma_scaled_positive, gamma_scaled_negatives.max())
            exp_positive = torch.exp(gamma_scaled_positive - max_scaled)
            exp_negatives = torch.exp(gamma_scaled_negatives - max_scaled)
            numerator = exp_positive
            denominator = exp_positive + exp_negatives.sum()
            sample_loss = -(torch.log(exp_positive + 1e-8) - torch.log(denominator + 1e-8))
            total_loss += sample_loss
            valid_samples += 1
        if valid_samples == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / valid_samples

    def compute_in_modal_loss(
        self,
        eeg_features: torch.Tensor,
        audio_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                eeg_avg = eeg_features[mask].mean(dim=0)
                audio_avg = audio_features[mask].mean(dim=0)
                self.eeg_proxies.data[label] = F.normalize(eeg_avg, p=2, dim=0)
                self.audio_proxies.data[label] = F.normalize(audio_avg, p=2, dim=0)
                hybrid_proxy = (self.eeg_proxies.data[label] + self.audio_proxies.data[label]) / 2
                self.hybrid_proxies.data[label] = F.normalize(hybrid_proxy, p=2, dim=0)
        eeg_loss = self.compute_unified_metric_loss(eeg_features, self.hybrid_proxies, labels)
        audio_loss = self.compute_unified_metric_loss(audio_features, self.hybrid_proxies, labels)
        return (eeg_loss + audio_loss) / 2

    def compute_cross_modal_loss(
        self,
        eeg_features: torch.Tensor,
        audio_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                eeg_avg = eeg_features[mask].mean(dim=0)
                audio_avg = audio_features[mask].mean(dim=0)
                self.eeg_proxies.data[label] = F.normalize(eeg_avg, p=2, dim=0)
                self.audio_proxies.data[label] = F.normalize(audio_avg, p=2, dim=0)
        cross_m1_loss = self.compute_unified_metric_loss(audio_features, self.eeg_proxies, labels)
        cross_m2_loss = self.compute_unified_metric_loss(eeg_features, self.audio_proxies, labels)
        return cross_m1_loss, cross_m2_loss

    def compute_single_modal_loss(
        self,
        eeg_features: torch.Tensor,
        audio_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        single_m1_loss = self.compute_unified_metric_loss(eeg_features, self.eeg_proxies, labels)
        single_m2_loss = self.compute_unified_metric_loss(audio_features, self.audio_proxies, labels)
        return single_m1_loss, single_m2_loss

    def compute_hybrid_loss(
        self,
        eeg_features: torch.Tensor,
        audio_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        combined_features = (eeg_features + audio_features) / 2
        return self.compute_unified_metric_loss(combined_features, self.hybrid_proxies, labels)


class CIREModule(nn.Module):
    def __init__(self, feature_dim: int, gamma: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.gamma = gamma

    def l2_normalize(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features, p=2, dim=1)

    def compute_causal_loss(
        self,
        original_features: torch.Tensor,
        enhanced_features: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        device = original_features.device
        n, d = original_features.shape
        if n <= 1 or d <= 1:
            return torch.tensor(0.0, device=device)
        orig_mean = original_features.mean(dim=0, keepdim=True)
        orig_std = original_features.std(dim=0, keepdim=True) + 1e-8
        orig_zscore = (original_features - orig_mean) / orig_std
        enh_mean = enhanced_features.mean(dim=0, keepdim=True)
        enh_std = enhanced_features.std(dim=0, keepdim=True) + 1e-8
        enh_zscore = (enhanced_features - enh_mean) / enh_std
        orig_norm = self.l2_normalize(orig_zscore)
        enh_norm = self.l2_normalize(enh_zscore)
        total_loss = 0.0
        valid_dims = 0
        for j in range(d):
            v_j = orig_norm[:, j]
            tilde_v_j = enh_norm[:, j]
            c_jj = F.cosine_similarity(v_j.unsqueeze(0), tilde_v_j.unsqueeze(0), dim=1).squeeze()
            negative_sims = []
            for l in range(d):
                if l != j:
                    tilde_v_l = enh_norm[:, l]
                    c_jl = F.cosine_similarity(v_j.unsqueeze(0), tilde_v_l.unsqueeze(0), dim=1).squeeze()
                    negative_sims.append(c_jl)
            if len(negative_sims) == 0:
                continue
            gamma_scaled_positive = self.gamma * c_jj
            gamma_scaled_negatives = torch.stack([self.gamma * sim for sim in negative_sims])
            max_scaled = torch.maximum(gamma_scaled_positive, gamma_scaled_negatives.max())
            exp_positive = torch.exp(gamma_scaled_positive - max_scaled)
            exp_negatives = torch.exp(gamma_scaled_negatives - max_scaled)
            numerator = exp_positive
            denominator = exp_positive + exp_negatives.sum()
            dim_loss = -(torch.log(numerator + 1e-8) - torch.log(denominator + 1e-8))
            total_loss += dim_loss
            valid_dims += 1
        if valid_dims == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / valid_dims


class CIMFModel(nn.Module):
    def __init__(
        self,
        backbone: str = "ResNet50-512",
        num_classes: int = 3,
        lambda_in: float = 1.0,
        lambda_cross: float = 1.0,
        lambda_hybrid: float = 1.0,
        lambda_single: float = 1.0,
        lambda_causal: float = 1.0,
        lambda_cls: float = 1.0,
        gamma: float = 1.0,
        tau: float = 0.4,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.feature_dim = 512 if "ResNet50-512" in backbone else 64 if "ResNet50-64" in backbone else 768
        self.gamma = gamma
        self.tau = tau
        self.lambda_weights = {
            "in": lambda_in,
            "cross_m1": lambda_cross,
            "cross_m2": lambda_cross,
            "hybrid": lambda_hybrid,
            "single_m1": lambda_single,
            "single_m2": lambda_single,
            "causal_m1": lambda_causal,
            "causal_m2": lambda_causal,
            "cls": lambda_cls,
        }
        self.fourier_aug = FourierAugmentation(tau=self.tau)
        self.backbone_shared = BackboneNetwork(self.backbone_name, self.num_classes)
        self.backbone_eeg = self.backbone_shared
        self.backbone_audio = self.backbone_shared
        self.backbone_enh_eeg = self.backbone_shared
        self.backbone_enh_audio = self.backbone_shared
        self.feature_dim = self.backbone_eeg.feature_dim
        self.mcaf = MCAFModule(self.feature_dim, self.num_classes, self.gamma)
        self.cire = CIREModule(self.feature_dim, self.gamma)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.5)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        eeg_inputs = data_dict["eeg"]
        audio_inputs = data_dict["audio"]
        enhanced_eeg, enhanced_audio = self.fourier_aug(eeg_inputs, audio_inputs)
        orig_eeg_features = self.backbone_eeg(eeg_inputs)
        orig_audio_features = self.backbone_audio(audio_inputs)
        enh_eeg_features = self.backbone_enh_eeg(enhanced_eeg)
        enh_audio_features = self.backbone_enh_audio(enhanced_audio)
        fused_features = (orig_eeg_features + orig_audio_features) / 2
        logits = self.classifier(fused_features)
        results = {
            "features": fused_features,
            "logits": logits,
            "orig_eeg_features": orig_eeg_features,
            "orig_audio_features": orig_audio_features,
            "enh_eeg_features": enh_eeg_features,
            "enh_audio_features": enh_audio_features,
        }
        if labels is not None:
            results.update(
                self.compute_losses(
                    orig_eeg_features,
                    orig_audio_features,
                    enh_eeg_features,
                    enh_audio_features,
                    logits,
                    labels,
                )
            )
        return results

    def compute_losses(
        self,
        orig_eeg_features: torch.Tensor,
        orig_audio_features: torch.Tensor,
        enh_eeg_features: torch.Tensor,
        enh_audio_features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_in = self.mcaf.compute_in_modal_loss(orig_eeg_features, orig_audio_features, labels)
        loss_cross_m1, loss_cross_m2 = self.mcaf.compute_cross_modal_loss(
            orig_eeg_features, orig_audio_features, labels
        )
        loss_single_m1, loss_single_m2 = self.mcaf.compute_single_modal_loss(
            orig_eeg_features, orig_audio_features, labels
        )
        loss_hybrid = self.mcaf.compute_hybrid_loss(orig_eeg_features, orig_audio_features, labels)
        loss_causal_m1 = self.cire.compute_causal_loss(orig_eeg_features, enh_eeg_features, "eeg")
        loss_causal_m2 = self.cire.compute_causal_loss(orig_audio_features, enh_audio_features, "audio")
        loss_cls = F.cross_entropy(logits, labels)
        total_loss = (
            self.lambda_weights["in"] * loss_in
            + self.lambda_weights["cross_m1"] * loss_cross_m1
            + self.lambda_weights["cross_m2"] * loss_cross_m2
            + self.lambda_weights["hybrid"] * loss_hybrid
            + self.lambda_weights["single_m1"] * loss_single_m1
            + self.lambda_weights["single_m2"] * loss_single_m2
            + self.lambda_weights["causal_m1"] * loss_causal_m1
            + self.lambda_weights["causal_m2"] * loss_causal_m2
            + self.lambda_weights["cls"] * loss_cls
        )
        return {
            "total_loss": total_loss,
            "loss_in": loss_in,
            "loss_cross_m1": loss_cross_m1,
            "loss_cross_m2": loss_cross_m2,
            "loss_hybrid": loss_hybrid,
            "loss_single_m1": loss_single_m1,
            "loss_single_m2": loss_single_m2,
            "loss_causal_m1": loss_causal_m1,
            "loss_causal_m2": loss_causal_m2,
            "loss_cls": loss_cls,
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "lambda_weights": self.lambda_weights,
            "tau": self.tau,
            "gamma": self.gamma,
        }
