"""
This script wraps the raw images and RLE annotations in a 
PyTorch Dataset class for training deep learning models. 
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import functional as F
import torchvision.ops as ops

def masks_to_valid_boxes(masks_t: torch.Tensor):
    """
    masks_t: [N,H,W] tensor (uint8/bool/float)
    returns: (boxes [M,4], keep [N] bool) where M==keep.sum()
    Filters fully empty masks *before* masks_to_boxes to avoid crashes.
    """
    if not isinstance(masks_t, torch.Tensor):
        masks_t = torch.as_tensor(masks_t)

    N = masks_t.shape[0]
    if N == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.bool)

    # remove fully empty masks first
    nonempty = (masks_t.reshape(N, -1).sum(dim=1) > 0)  
    if nonempty.sum() == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((N,), dtype=torch.bool)

    m_kept = masks_t[nonempty].float()
    boxes = ops.masks_to_boxes(m_kept)  

    # filter that degenerates boxes
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if valid.sum() == 0:
        keep = torch.zeros((N,), dtype=torch.bool)
        return torch.zeros((0, 4), dtype=torch.float32), keep

    # maps back to original indices
    orig_idx = nonempty.nonzero(as_tuple=False).squeeze(1)[valid]
    keep = torch.zeros((N,), dtype=torch.bool)
    keep[orig_idx] = True
    return boxes[valid], keep

class SartoriusInstanceDataset(Dataset):
    """
    This class pairs raw FOVs to their corresponding RLE annotations
    """
    def __init__(self, images_dir, csv_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_ids = [x.split(".")[0] for x in os.listdir(images_dir)]

        self.rle_dict = defaultdict(list)
        rle_df = pd.read_csv(csv_path)
        for _, row in rle_df.iterrows():
            self.rle_dict[row['id']].append(row['annotation'])


    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):

      image_id = self.image_ids[index]

      # load image
      img_path = os.path.join(self.images_dir, f"{image_id}.png")
      img = Image.open(img_path).convert("RGB")
      img_arr = np.array(img)

      # decoding RLEs 
      raw_masks = []
      for rle in self.rle_dict[image_id]:
          raw_masks.append(self.decode_rle(rle))
      if len(raw_masks) == 0:
          H, W = img_arr.shape[:2]
          raw_masks = np.zeros((0, H, W), dtype=np.uint8)
      else:
          raw_masks = np.stack(raw_masks, axis=0)  

      if self.transform:
          transformed = self.transform(
              image=img_arr,
              masks=[m.astype(np.uint8) for m in raw_masks]
          )
          img_tensor = transformed["image"]  

          # Albumentations returns a list of masks after transforms
          if len(transformed["masks"]) > 0:
              masks_t = torch.stack([m.to(torch.uint8) for m in transformed["masks"]])  # [N,H,W]
          else:
              Ht, Wt = img_tensor.shape[1:]
              masks_t = torch.zeros((0, Ht, Wt), dtype=torch.uint8)

          # recompute boxes/area from transformed masks
          boxes_t, keep = masks_to_valid_boxes(masks_t)
          if keep.numel() > 0 and keep.dtype == torch.bool:
              masks_t = masks_t[keep]

          labels_t = torch.ones((boxes_t.shape[0],), dtype=torch.int64)
          if boxes_t.shape[0] > 0:
              area_t = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
          else:
              area_t = torch.zeros((0,), dtype=torch.float32)

          target = {
              "boxes": boxes_t,
              "labels": labels_t,
              "masks": masks_t,
              "image_id": torch.tensor([index]),   # numeric id for torchvision
              "img_id_str": image_id,              # keep your string id for logging
              "area": area_t,
              "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
          }
          return img_tensor, target

      # ----- no transforms: build tensors + boxes on original masks for consistency
      img_tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1) / 255.0
      masks_t = torch.as_tensor(raw_masks, dtype=torch.uint8)  # [N,H,W]
      boxes_t, keep = masks_to_valid_boxes(masks_t)
      if keep.numel() > 0 and keep.dtype == torch.bool:
          masks_t = masks_t[keep]
      labels_t = torch.ones((boxes_t.shape[0],), dtype=torch.int64)
      if boxes_t.shape[0] > 0:
          area_t = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
      else:
          area_t = torch.zeros((0,), dtype=torch.float32)
      
      # target dictionary

      target = {
          "boxes": boxes_t,
          "labels": labels_t,
          "masks": masks_t,
          "image_id": torch.tensor([index]),
          "img_id_str": image_id,
          "area": area_t,
          "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64)
      }
      return img_tensor, target
      
    
    def decode_rle(self, rle, shape=(520, 704)):
        """
        Helper function that decodes rles
        """
        s = list(map(int, rle.split()))
        starts, lengths = s[::2], s[1::2]
        starts = np.array(starts) - 1
        ends = starts + lengths

        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)