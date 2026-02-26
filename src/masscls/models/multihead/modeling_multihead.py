from typing import List, Dict


import timm
import torch
import torch.nn as nn

from .configuration_multihead import MultiHeadCNNConfig

# for `transformers` library
# from masscls.loss import TaskSpecificFocalLoss, MultiTaskLoss


class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) from CBAM"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Average pool
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pool
        max_out = self.fc(self.max_pool(x).view(b, c))

        # Combine
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) from CBAM"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))

        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Squeeze
        y = self.squeeze(x).view(b, c)

        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)

        return x * y


class ClassificationHead(nn.Module):
    """
    Improved classification head with:
    - Deeper MLP
    - Batch normalization
    - Dropout
    - Residual connections
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual
        layers = []

        prev_dim = in_features
        for i, dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))

            # Activation
            layers.append(nn.ReLU(inplace=True))

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiHeadCNNForClassification(nn.Module):
    def __init__(self, config: MultiHeadCNNConfig):
        super().__init__()

        self.backbone = timm.create_model(
            config.backbone,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            if isinstance(dummy_features, dict):
                feature_dim = list(dummy_features.values())[0].shape[1]
            else:
                feature_dim = dummy_features.shape[1]

        in_features: int = feature_dim

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if config.attention == "cbam":
            self.attention = CBAM(in_features)
        elif config.attention == "se":
            self.attention = SEBlock(in_features)
        else:
            self.attention = nn.Identity()

        self.heads = nn.ModuleDict()
        for head_name, head_dim in zip(config.head_names, config.head_dims):
            self.heads[head_name] = ClassificationHead(
                in_features=in_features,
                num_classes=config.head_names[head_name],
                hidden_dims=head_dim,
                dropout_rate=0.3,
                use_batch_norm=True,
                use_residual=True,
            )
        # for `transformers` library
        # loss_fns = {}
        # for head_name, _ in zip(config.head_names, config.num_classes):
        #     loss_fns[head_name] = TaskSpecificFocalLoss(
        #         alpha=(
        #             torch.tensor(config.class_weights[head_name])
        #             if config.class_weights is not None
        #             else None
        #         ),
        #         gamma=config.gamma[head_name],
        #         label_smoothing=config.smoothing[head_name],
        #     )
        # self.multi_task_loss = MultiTaskLoss(
        #     task_losses=loss_fns,
        #     task_weights=config.task_weights,
        #     learnable_weights=False,
        # )

    # for `transformers` library
    # def calculate_loss(
    #     self,
    #     logits: Dict[str, torch.FloatTensor],
    #     targets: Dict[str, torch.FloatTensor],
    # ) -> torch.Tensor:
    #     return self.multi_task_loss(logits=logits, targets=targets)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.attention(features)
        features = self.global_pool(features)
        features = features.flatten(1)

        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(features)

        return logits

    @torch.no_grad
    def predict_tta(self, x: torch.Tensor):
        assert hasattr(self, "tta_transform")

        outputs = []
        for transform in getattr(self, "tta_transform"):
            x = transform(x)
            outputs.append(self.forward(x))

        return outputs

    # for `transformers` library
    # def forward(
    #     self,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     labels: Optional[Dict[str, torch.FloatTensor]] = None,
    # ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], None]]:
    #     features = self.backbone(pixel_values)
    #     features = self.attention(features)
    #     features = self.global_pool(features)
    #     features = features.flatten(1)

    #     logits = {}
    #     for name, head in self.heads.items():
    #         logits[name] = head(features)

    #     loss = None
    #     if labels is not None:
    #         loss = self.calculate_loss(logits=logits, targets=labels)

    #     return {"loss": loss, "logits": logits}
