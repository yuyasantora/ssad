import torch
import torch.nn as nn
import torchvision.models as models

# モデルの定義
class SwinBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinBackbone, self).__init__()
        if pretrained:
            self.backbone = models.swin_v2_t(weights='IMAGENET1K_V1' if pretrained else None)
            # FPN用の出力を取得するためのレイヤー 768channels
            self.out_channels = self.backbone.head.in_features
            # 各ステージの出力チャネル数を取得
            dims = [96, 192, 384, 768]
            self_feature_info = [
                dict(num_channels=dims[0], reduction=4),
                dict(num_channels=dims[1], reduction=8),
                dict(num_channels=dims[2], reduction=16),
                dict(num_channels=dims[3], reduction=32),
            ]
        else:
            self.backbone = models.swin_v2_t()

    def forward(self, x):
        features = []

        # 各ステージの出力を取得
        x = self.backbone.features(x)
        # 具体的な実装はモデルのバージョンによって異なる可能性があります
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if isinstance(layer, nn.Sequential):  # ステージの終わりを検出
                features.append(x)

        return features

