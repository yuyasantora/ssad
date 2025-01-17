import torch
import cv2

from demo.detection import FCOSDetector

"""モデルの読み込み"""
# 学習済みの重みのパス
encoder_weight = "encoder_1000.pth"
detector_weight = "detector_1000.pth"

# モデルの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCOSDetector()
# 重みを読み込む
model.load_state_dict(torch.load(encoder_weight, map_location=device))

# モデルをcudaに移動
model.to(device)

# 評価モード
model.eval()

"""テスト画像の読み込み"""
image = cv2.imread("dataset/simmim/xrays/val_0.ipynb")
image = cv2.resize(image, (512, 512))
image = torch.from_numpy(image)
image = image.unsqueeze(0)
image = image.permute(0, 3, 1, 2)
# 画像を正規化
image = image.float() / 255.0
image = image.to(device)

"""モデルの推論"""
with torch.no_grad():
    # モデルに画像を入力
    output = model(image)

# 出力を表示
print("------------------------------------------------------------------------------------------")
print(f"出力: {output}")



