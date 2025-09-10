"""
This script trains a Mask R-CNN model. The goal is to create instance 
segmentations of microglia, astrocytes, cortical neurons, and shsy5y cells.
Data from the Sartorious Image Dataset
"""
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from maskrcnn_dataset import SartoriusInstanceDataset
from maskrcnn_utils import train, validate, save_color_overlays
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import GradScaler

torch.backends.cudnn.benchmark = True

# PARAMETERS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = (DEVICE.type == "cuda")
SCALER = GradScaler(device="cuda", enabled=USE_AMP)
PIN_MEMORY = False
BATCH_SIZE = 1  # Increased batch size for better gradient estimates
NUM_WORKERS = 0
NUM_CLASSES = 2  # Background + cells
EPOCHS = 15  # More epochs for better convergence
HIDDEN_VAL = 256
LEARNING_RATE = 3e-4  # previously 5e-4
WEIGHT_DECAY = 1e-5  # Reduced weight decay
STEP_SIZE = 7  # Adjusted step size
GAMMA = 0.5  # Less aggressive lr decay


BASE_DIR = "/content/drive/MyDrive/colab-projects"
TRAIN_IMG_DIR  = f"{BASE_DIR}/train/"
TRAIN_CSV_PATH = f"{BASE_DIR}/train_data.csv"
VAL_IMG_DIR    = f"{BASE_DIR}/val/"
VAL_CSV_PATH   = f"{BASE_DIR}/val_data.csv"
SAVE_CKPT_PATH = f"{BASE_DIR}/best_ckpt_sartorius_dataset.pth"

# COLLATE FUNCTION (NEEDED FOR MASK R-CNN)
def collate_fn(batch):
    return tuple(zip(*batch))

# TRANSFORMS FOR TRAINING AND VALIDATION
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),

    
    A.Affine(
        scale=(0.85, 1.15),
        translate_percent=(0.05, 0.05),
        rotate=(-20, 20),
        shear={'x': (-20, 20), 'y': (-10, 10)},
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.3
    ),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], p=1.0)

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)), 
    ToTensorV2()
])

# DATASETS AND DATALOADERS
train_ds = SartoriusInstanceDataset(TRAIN_IMG_DIR, TRAIN_CSV_PATH, transform=train_transform)
val_ds = SartoriusInstanceDataset(VAL_IMG_DIR, VAL_CSV_PATH, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn = collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn = collate_fn)


# INSTANTIATE MODEL
model = maskrcnn_resnet50_fpn(weights="DEFAULT")

model.transform.min_size = [896, 1024]
model.transform.max_size = 1333

# features: replace box head predictor 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

# features: replace mask head predictor
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, HIDDEN_VAL, NUM_CLASSES)

# ADD RPN ANCHORS (SIZE DETECTION)
model.rpn.anchor_generator.sizes = ((8,), (16,), (32,), (64,), (128,))
model.rpn.anchor_generator.aspect_ratios = ((0.2, 0.33, 0.5, 1.0, 2.0),) * len(model.rpn.anchor_generator.sizes)

# MORE RPN PARAMETERS (NUMBER OF MASKS DETECTED)
model.rpn.pre_nms_top_n_train  = 6000    
model.rpn.post_nms_top_n_train = 1000    
model.rpn.pre_nms_top_n_test   = 6000
model.rpn.post_nms_top_n_test  = 2000    
model.rpn.nms_thresh           = 0.7     

model.rpn.box_fg_iou_thresh = 0.5
model.rpn.box_bg_iou_thresh = 0.3
model.rpn.batch_size_per_image = 512
model.rpn.positive_fraction = 0.5

model.roi_heads.box_fg_iou_thresh = 0.5
model.roi_heads.box_bg_iou_thresh = 0.4
model.roi_heads.batch_size_per_image = 512
model.roi_heads.positive_fraction = 0.33

# ROI POOLING SIZES

model.roi_heads.mask_roi_pool.output_size = (32, 32)
model.roi_heads.score_thresh = 0.30    
model.roi_heads.nms_thresh   = 0.50   


# ROI HEADS THRESHOLDS FOR TRAINING

if hasattr(model.roi_heads, "detections_per_img"):
    model.roi_heads.detections_per_img = 200 # previous: 400


model.to(DEVICE)

# MODEL PARAMETER SET UP

params = []
for p in model.parameters():
    if p.requires_grad:
        params.append(p)

# Use different learning rates for different parts of the model
backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

# OPTIMIZER AND SCHEDULER SET UP

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # Lower LR for backbone
    {'params': head_params, 'lr': LEARNING_RATE}             # Higher LR for heads
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
patience = 5
patience_counter = 0

# TRAINING AND FITTING THE MODEL

best_val = float("inf")

for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    # Training
    train_metrics = train(model, train_loader, optimizer, device=DEVICE, scaler=SCALER, use_amp=USE_AMP)
    
    # Validation
    val_metrics = validate(model, val_loader, device=DEVICE,
                            score_thresh=0.25, mask_bin_thresh=0.40,
                            min_pixels=80, iou_match_thresh=0.5,
                            topk = 200,          
                            downsample = 4)   
                          
    
    # Update learning rate
    scheduler.step()

    # calculate loss
    train_std  = train_metrics.get("standard_total",
                               sum(v for k, v in train_metrics.items() if k.startswith("loss_")))
    train_dice = train_metrics.get("aux_dice_loss", 0.0)
    train_loss = train_metrics.get("total", train_std + train_dice)

    # val has no dice; sum only the loss_* keys
    val_loss = sum(v for k, v in val_metrics.items() if k.startswith("loss_"))

    print(f"Train (std): {train_std:.4f} | Train (dice): {train_dice:.4f} | "
          f"Train (total): {train_loss:.4f} | Val (total): {val_loss:.4f}")

    print(f"Val IoU: {val_metrics['jaccard_iou']:.3f} | "
      f"Dice: {val_metrics['dice_f1']:.3f} | "
      f"P: {val_metrics['precision']:.3f} | "
      f"R: {val_metrics['recall']:.3f} | "
      f"F1(inst): {val_metrics['f1_instance']:.3f}")

    # SAVE THE BEST CHECKPOINT
    
    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }, SAVE_CKPT_PATH)
        
        print(f"saved checkpoint to {SAVE_CKPT_PATH}")

        # Save predictions for best model
        print("Generating prediction overlays...")
        save_color_overlays(model, val_loader, device=DEVICE, 
                          out_dir="maskrcnn_preds", background="image")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")

    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping after {epoch} epochs")
        break

print("Training completed!")

# Load best model and generate final predictions
print("Loading best model for final predictions...")
checkpoint = torch.load(SAVE_CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])

print("Generating final prediction overlays...")
save_color_overlays(model, val_loader, device=DEVICE, 
                  out_dir="final_preds", alpha=1, background="black")
