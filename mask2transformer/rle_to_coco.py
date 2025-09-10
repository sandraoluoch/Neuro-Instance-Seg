"""
This script converts RLE annotations into COCO-style JSON files for training a Mask2Former model.
"""

import os, json, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as cocomask

# Directories
TRAIN_IMG_DIR = "/content/drive/MyDrive/colab-projects/train"
VAL_IMG_DIR   = "/content/drive/MyDrive/colab-projects/val"
TRAIN_CSV     = "/content/drive/MyDrive/colab-projects/train_data.csv"
VAL_CSV       = "/content/drive/MyDrive/colab-projects/val_data.csv"
OUT_DIR       = "/content/drive/MyDrive/colab-projects/coco"

def rle_decode(rle, shape, order="F"):
    """
    Decode an RLE string into a binary mask
    """
    if not isinstance(rle, str) or rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)
    s = list(map(int, rle.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.asarray(starts) - 1
    ends = starts + np.asarray(lengths)
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        flat[lo:hi] = 1
    return flat.reshape(shape, order=order)

def _bbox(mask):
    """
    This function creates the bounding boxes
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

def _looks_stripe(mask, tol_h=0.85, tol_w=0.25, min_fill=0.35):
    """
    This function filers out the stripe-like artifacts from the RLE annotations.
    """
    H, W = mask.shape
    x, y, w, h = _bbox(mask)
    if w == 0 or h == 0:
        return True  
    if (h / H) >= tol_h and (w / W) <= tol_w:
        # require some solidity; super-sparse tall strips are likely bogus
        fill = mask.sum() / float(w * h)
        return fill < min_fill
    return False

# cell labels
CELL_MAP = {"shsy5y": 1, "astro": 2, "cort": 3}

def csv_to_coco(images_dir, csv_path, out_json, multiclass=True):
    """
    This function converts the csv file into coco JSON files
    """
    df = pd.read_csv(csv_path)
    has_type = ("cell_type" in df.columns) and multiclass
    cats = ([{"id":1,"name":"cell","supercategory":"cell"}] if not has_type else
            [{"id":1,"name":"shsy5y","supercategory":"cell"},
             {"id":2,"name":"astro","supercategory":"cell"},
             {"id":3,"name":"cort","supercategory":"cell"}])

    images, annotations = [], []
    g = df.groupby("id")
    ann_id = 1

    # for stripe artifacts
    stripes_F = stripes_C = total = dropped_stripes = 0  

    for img_idx, (img_id, sub) in enumerate(tqdm(g, desc=f"COCO {os.path.basename(csv_path)}")):
        fn_png, fn_jpg = f"{img_id}.png", f"{img_id}.jpg"
        file_name = fn_png if os.path.exists(os.path.join(images_dir, fn_png)) else fn_jpg
        path = os.path.join(images_dir, file_name)
        if not os.path.exists(path):
            print("missing:", path)
            continue

        with Image.open(path) as im:
            w, h = im.size

        images.append({"id": img_idx, "file_name": file_name, "width": w, "height": h})

        for _, row in sub.iterrows():
            rle_str = row.get("annotation", "")
            
            mF = rle_decode(rle_str, (h, w), order="F")
            mC = rle_decode(rle_str, (h, w), order="C")

            
            xF, yF, wF, hF = _bbox(mF)
            xC, yC, wC, hC = _bbox(mC)
            stripeF = _looks_stripe(mF)
            stripeC = _looks_stripe(mC)

            total += 1
            stripes_F += int(stripeF)
            stripes_C += int(stripeC)

            
            if stripeF and not stripeC:
                m = mC
            elif stripeC and not stripeF:
                m = mF
            elif stripeF and stripeC:
                dropped_stripes += 1
                continue  # skip this annotation entirely if it is a stripe annotation
            else:
                m = mF if (wF * hF) >= (wC * hC) else mC

            if m.sum() < 30:
                continue  

            # Encode to COCO RLE (always column-major)
            rle = cocomask.encode(np.asfortranarray(m.astype(np.uint8)))
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("ascii")

            area = int(cocomask.area(rle))
            bbox = cocomask.toBbox(rle).tolist()
            cat = (CELL_MAP.get(row.get("cell_type", ""), 1) if has_type else 1)

            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": cat,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            ann_id += 1 

    coco = {"images": images, "annotations": annotations, "categories": cats}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"Saved {out_json} | images={len(images)} anns={len(annotations)}")
    if total:
        print(f"stripe heuristic: F={stripes_F}/{total}, C={stripes_C}/{total}")


# generate train and val COCO JSON files.
os.makedirs(OUT_DIR, exist_ok=True)
csv_to_coco(TRAIN_IMG_DIR, TRAIN_CSV, f"{OUT_DIR}/train_coco.json", multiclass=True)
csv_to_coco(VAL_IMG_DIR,   VAL_CSV,   f"{OUT_DIR}/val_coco.json",   multiclass=True)
