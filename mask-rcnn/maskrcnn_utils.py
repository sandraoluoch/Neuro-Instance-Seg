import torch
import numpy as np
import os
from PIL import Image
from torchvision.utils import draw_segmentation_masks
from tqdm.auto import tqdm
from torch.amp import autocast
import torch.nn.functional as F

# TRAINING FUNCTION
def train(model, loader, optimizer, device, scaler=None, use_amp=False):
    """
    This function trains the model.
    """
    
    model.train()
    totals = {}
    count_batches = 0

    loop = tqdm(loader, total=len(loader), desc="Train", leave=False)
    for images, targets in loop:
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        # forward (shared)
        with autocast("cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            standard = sum(loss_dict.values())
            loss = standard

        # backward / step
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # logging
        count_batches += 1
        for key, val in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(val.item())
        
        # log the standard (no-dice) total
        totals["standard_total"] = totals.get("standard_total", 0.0) + float(standard.item())

        # existing: total includes Dice if present
        totals["total"] = totals.get("total", 0.0) + float(loss.item())
        loop.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]["lr"])

    return {key: totals[key] / count_batches for key in totals}

# VALIDATION FUNCTION

@torch.no_grad()
def validate(model, loader, device,
             score_thresh: float = 0.05,
             mask_bin_thresh: float = 0.25,
             min_pixels: int | None = 10,
             iou_match_thresh: float = 0.5,
             topk: int = 200,           
             downsample: int = 4):   

    """
    This function runs the validation step for the model.
    """   
    model.eval()

    totals = {
        "val_loss_sum": 0.0,
        "matched": 0, "pred": 0, "gt": 0,
        "mean_iou_sum": 0.0, "mean_dice_sum": 0.0,
        "num_metric_batches": 0,
    }
    count_batches = 0

    loop = tqdm(loader, total=len(loader), desc="Val", leave=False)
    for images, targets in loop:
        
        images_cuda  = [img.to(device) for img in images]
        targets_cuda = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()}
                        for t in targets]

        model.train()
        loss_dict = model(images_cuda, targets_cuda)
        model.eval()

        batch_loss = float(sum(loss_dict.values()).item())
        totals["val_loss_sum"] += batch_loss
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())

        outputs = model(images_cuda)

        batch_iou, batch_dice = [], []
        for out, tgt in zip(outputs, targets_cuda):
            scores = out["scores"].detach()
            if scores.numel() == 0:
                totals["gt"] += int(tgt["masks"].shape[0])
                continue

            keep = scores >= score_thresh
            idx = torch.nonzero(keep).squeeze(1)
            if idx.numel() > topk:  
                sub = torch.topk(scores[idx], k=topk).indices
                idx = idx[sub]

            totals["pred"] += int(idx.numel())

            gm = tgt["masks"].to(device).bool()     
            totals["gt"] += int(gm.shape[0])
            if idx.numel() == 0 or gm.numel() == 0:
                continue

            pm = out["masks"][idx, 0]               

            # downsample in case of OOM
            if downsample > 1:
                scale = 1.0 / float(downsample)
                pm = F.interpolate(pm.unsqueeze(1), scale_factor=scale,
                                   mode="bilinear", align_corners=False).squeeze(1)
                gm = F.interpolate(gm.float().unsqueeze(1), scale_factor=scale,
                                   mode="nearest").squeeze(1).bool()

            pm_bool = pm > mask_bin_thresh         
            if min_pixels is not None and pm_bool.numel():
                min_pix_eff = max(1, int(min_pixels / (downsample * downsample)))
                sizes = pm_bool.flatten(1).sum(1)
                pm_bool = pm_bool[sizes >= min_pix_eff]
            if pm_bool.numel() == 0:
                continue

            iou_mat = calculate_iou(pm_bool, gm)    
            matches = _greedy_match_from_iou(iou_mat, iou_match_thresh)
            totals["matched"] += len(matches)
            if not matches:
                continue

            p_idx = torch.tensor([p for p, g in matches], device=iou_mat.device, dtype=torch.long)
            g_idx = torch.tensor([g for p, g in matches], device=iou_mat.device, dtype=torch.long)
            ious  = iou_mat[p_idx, g_idx]
            dices = (2 * ious) / (1 + ious + 1e-6)

            batch_iou.append(ious.mean())
            batch_dice.append(dices.mean())

        if batch_iou:
            totals["mean_iou_sum"]  += float(torch.stack(batch_iou).mean().item())
            totals["mean_dice_sum"] += float(torch.stack(batch_dice).mean().item())
            totals["num_metric_batches"] += 1

        count_batches += 1
        loop.set_postfix(val_loss=f"{batch_loss:.4f}")

    # averaged losses 
    avg_losses = {k: totals[k] / max(1, count_batches) for k in totals if k.startswith("loss_")}
    val_loss_total = totals["val_loss_sum"] / max(1, count_batches)

    # metrics
    mean_iou  = (totals["mean_iou_sum"]  / totals["num_metric_batches"]) if totals["num_metric_batches"] > 0 else 0.0
    mean_dice = (totals["mean_dice_sum"] / totals["num_metric_batches"]) if totals["num_metric_batches"] > 0 else 0.0
    precision = totals["matched"] / max(1, totals["pred"])
    recall    = totals["matched"] / max(1, totals["gt"])
    f1_inst   = (2 * precision * recall) / max(1e-6, (precision + recall))

    return {
        **avg_losses,
        "val_loss_total": float(val_loss_total),
        "jaccard_iou":    float(mean_iou),
        "dice_f1":        float(mean_dice),
        "precision":      float(precision),
        "recall":         float(recall),
        "f1_instance":    float(f1_inst),
        "matched": int(totals["matched"]),
        "pred":    int(totals["pred"]),
        "gt":      int(totals["gt"]),
    }

def calculate_iou(pred_masks_bool: torch.Tensor, gt_masks_bool: torch.Tensor) -> torch.Tensor:
   
    P = pred_masks_bool.shape[0]
    G = gt_masks_bool.shape[0]
    if P == 0 or G == 0:
        return pred_masks_bool.new_zeros((P, G), dtype=torch.float32)

    pm = pred_masks_bool.reshape(P, -1).float()   
    gm = gt_masks_bool.reshape(G, -1).float()     

    inter = pm @ gm.t()                           
    union = (pm.sum(1, keepdim=True) + gm.sum(1, keepdim=True).t() - inter).clamp_min(1.0)
    return inter / union

def _greedy_match_from_iou(iou_mat: torch.Tensor, iou_thresh: float = 0.5):
    """
    Helper function for calculating IOU
    """
    matches = []
    if iou_mat.numel() == 0:
        return matches
    P, G = iou_mat.shape
    used_p = set(); used_g = set()
    vals, idx = torch.sort(iou_mat.flatten(), descending=True)
    for v, f in zip(vals.tolist(), idx.tolist()):
        if v < iou_thresh:
            break
        p = f // G
        g = f % G
        if p in used_p or g in used_g:
            continue
        used_p.add(p); used_g.add(g)
        matches.append((p, g))
    return matches

# SAVE PREDICTIONS FUNCTION

@torch.no_grad()
def save_color_overlays(
    
    model, loader, device,
    out_dir="preds_overlay",
    score_thresh=0.30,          # overlay filter
    alpha=0.5,
    min_pixels=80,            # e.g., 30-80 to drop tiny specks
    mask_bin_thresh=0.45,        # binarize prob masks
    background="image"
):
    """
    This function overlays the predictions on the raw image
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
        
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        outputs = model(images)

        for img_t, out, tgt in zip(images, outputs, targets):
            # unnormalize to uint8
            mean = torch.tensor([0.485, 0.456, 0.406], device=img_t.device).view(-1,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=img_t.device).view(-1,1,1)
            img_uint8 = ((img_t * std + mean).clamp(0,1).cpu() * 255).to(torch.uint8)
            base = img_uint8 if background == "image" else torch.zeros_like(img_uint8)
            
            scores = out["scores"].detach().cpu()
            keep = scores >= score_thresh
            if keep.sum().item() == 0:
                print(f"[{tgt['img_id_str']}] NO MASKS passed score threshold.")
                vis = base
            else:
                # probs [K,H,W], sort by score (high→low)
                masks = out["masks"][keep, 0].cpu()
                order = torch.argsort(scores[keep], descending=True)
                masks = masks[order]

                # binarize once
                bin_masks = masks > mask_bin_thresh  # bool [K,H,W]

                # greedy pixel claiming → zero overlap
                uniq_masks = []
                if bin_masks.numel():
                    taken = torch.zeros_like(bin_masks[0], dtype=torch.bool)
                    for m in bin_masks:
                        m_unique = m & ~taken
                        if m_unique.any():
                            uniq_masks.append(m_unique)
                            taken |= m_unique

                masks_bool = (torch.stack(uniq_masks, dim=0)
                              if len(uniq_masks) > 0
                              else torch.empty((0, *masks.shape[1:]), dtype=torch.bool))

                # optional tiny-object filter AFTER carving
                if min_pixels is not None and masks_bool.numel():
                    sizes = masks_bool.reshape(masks_bool.shape[0], -1).sum(1)
                    masks_bool = masks_bool[sizes >= min_pixels]

                if masks_bool.shape[0] > 0:
                    print(f"[{tgt['img_id_str']}] SUCCESS: {masks_bool.shape[0]} masks (zero-overlap).")
                    colors = [tuple(np.random.randint(0, 256, size=3)) for _ in range(masks_bool.shape[0])]
                    vis = draw_segmentation_masks(base, masks_bool, alpha=alpha, colors=colors)
                else:
                    print(f"[{tgt['img_id_str']}] WARNING: All masks filtered out after carving.")
                    vis = base

            out_path = os.path.join(out_dir, f"preds_{tgt['img_id_str']}.png")
            Image.fromarray(vis.permute(1, 2, 0).numpy()).save(out_path)

