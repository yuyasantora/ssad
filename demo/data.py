import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

"""データセットの読み込み"""



import torch
import torchvision
from torchvision.datasets import CocoDetection

class MyCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(MyCocoDetection, self).__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        # CocoDetection.__getitem__ -> (PIL.Image, list of annotation dict)
        img, anno_list = super(MyCocoDetection, self).__getitem__(idx)

        # anno_list は以下のようなリスト:
        # [
        #   {
        #       "bbox": [x, y, w, h],
        #       "category_id": 1,
        #       "area": ...,
        #       "iscrowd": 0 or 1,
        #       ...
        #   },
        #   ...
        # ]

        # バウンディングボックス情報を作成
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in anno_list:
            xmin = obj["bbox"][0]
            ymin = obj["bbox"][1]
            w    = obj["bbox"][2]
            h    = obj["bbox"][3]
            # coco annotation の bbox は (x, y, width, height)
            # → 物体検出タスクでは (xmin, ymin, xmax, ymax) に変換
            xmax = xmin + w
            ymax = ymin + h

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj["category_id"])
            areas.append(obj["area"])
            iscrowd.append(obj["iscrowd"])

        # list → Tensor へ変換
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # image_id は CocoDetection の場合 self.ids[idx] で取れる
        image_id = torch.tensor([self.ids[idx]])

        # 最終的に return する target を辞書形式にまとめる
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        # もし拡張や前処理など transforms がある場合は、img と target に適用
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

