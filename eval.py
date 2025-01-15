import torch
import cv

from demo.detection import FcosDetector

"""モデルの読み込み"""
# 学習済みの重みのパス
encoder_weight = "encoder_1000.pth"
detector_weight = "detector_1000.pth"

# モデルの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FcosDetector()
# 重みを読み込む
model.load_state_dict(torch.load(encoder_weight, map_location=device))

# 評価モード
model.eval()

"""テスト画像の読み込み"""
image = cv.imread("test.png")
image = torch.from_numpy(image)
# 画像を正規化
image = image / 255.0

"""モデルの推論"""
with torch.no_grad():
    # モデルに画像を入力
    output = model(image)

# 出力を表示
print(output)


