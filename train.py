"""
Chest X-Ray Pneumonia Classifier
Fine-tunes ViT-B/16 on the pneumonia_x_ray dataset.
All hyperparams configurable via environment variables.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report
import wandb
import numpy as np
from PIL import Image
import time

# ── Hyperparameters (all via env vars) ──────────────────────────────
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
DROPOUT = float(os.environ.get("DROPOUT", "0.0"))
USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "false").lower() == "true"
AUGMENTATION = os.environ.get("AUGMENTATION", "none")  # none, basic, medical
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "chamber-xray-demo")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "jasonong-chamberai")
RUN_NAME = os.environ.get("RUN_NAME", f"xray-v{int(time.time())}")
SKIP_RGB_CONVERT = os.environ.get("SKIP_RGB_CONVERT", "false").lower() == "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def get_transforms():
    """Get train/test transforms based on AUGMENTATION setting."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if AUGMENTATION == "medical":
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif AUGMENTATION == "basic":
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


class XRayDataset(torch.utils.data.Dataset):
    """Wraps HuggingFace dataset for PyTorch DataLoader."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not SKIP_RGB_CONVERT and image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_model():
    """Build ViT-B/16 with custom classification head."""
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    # Freeze early layers, fine-tune later ones
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last 4 encoder blocks + head
    for i in range(8, 12):
        for param in model.encoder.layers[i].parameters():
            param.requires_grad = True

    # Replace classification head
    in_features = model.heads.head.in_features
    if DROPOUT > 0:
        model.heads.head = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, 2)
        )
    else:
        model.heads.head = nn.Linear(in_features, 2)

    return model.to(device)


def get_sampler(labels):
    """Create weighted random sampler for class imbalance."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {running_loss/(batch_idx+1):.4f} "
                  f"Acc: {100.*correct/total:.1f}%")

    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100. * np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    normal_acc = 100. * cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    pneumonia_acc = 100. * cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    # Sensitivity (recall for pneumonia) and Specificity (recall for normal)
    sensitivity = pneumonia_acc
    specificity = normal_acc

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "normal_accuracy": normal_acc,
        "pneumonia_accuracy": pneumonia_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
    }


def main():
    print("=" * 60)
    print("Chamber X-Ray Pneumonia Classifier")
    print("=" * 60)
    print(f"Config: batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, epochs={EPOCHS}")
    print(f"  dropout={DROPOUT}, class_weights={USE_CLASS_WEIGHTS}")
    print(f"  augmentation={AUGMENTATION}, warmup={WARMUP_EPOCHS}")
    print(f"  weight_decay={WEIGHT_DECAY}, num_workers={NUM_WORKERS}")
    print()

    # ── W&B Init ──
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=RUN_NAME,
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "dropout": DROPOUT,
            "use_class_weights": USE_CLASS_WEIGHTS,
            "augmentation": AUGMENTATION,
            "warmup_epochs": WARMUP_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "model": "vit_b_16",
            "dataset": "mmenendezg/pneumonia_x_ray",
        }
    )

    # ── Dataset ──
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("mmenendezg/pneumonia_x_ray")
    train_transform, test_transform = get_transforms()

    train_dataset = XRayDataset(ds["train"], transform=train_transform)
    val_dataset = XRayDataset(ds["validation"], transform=test_transform)
    test_dataset = XRayDataset(ds["test"], transform=test_transform)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ── DataLoaders ──
    train_labels = np.array([item["label"] for item in ds["train"]])
    if USE_CLASS_WEIGHTS:
        sampler = get_sampler(train_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=NUM_WORKERS,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    # ── Model ──
    print("Building ViT-B/16 model...")
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ── Loss ──
    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(train_labels)
        weights = torch.FloatTensor([1.0 / class_counts[0], 1.0 / class_counts[1]])
        weights = weights / weights.sum() * 2  # normalize
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        print(f"Using class weights: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── LR Scheduler ──
    if WARMUP_EPOCHS > 0:
        def lr_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            # Cosine decay after warmup
            progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training Loop ──
    best_val_acc = 0
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\n{'─'*40}")
        print(f"Epoch {epoch+1}/{EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.6f})")
        print(f"{'─'*40}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.1f}%")
        print(f"  Normal Acc: {val_metrics['normal_accuracy']:.1f}% | "
              f"Pneumonia Acc: {val_metrics['pneumonia_accuracy']:.1f}%")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_normal_accuracy": val_metrics["normal_accuracy"],
            "val_pneumonia_accuracy": val_metrics["pneumonia_accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ New best model saved (val_acc={best_val_acc:.1f}%)")

    # ── Final Test Evaluation ──
    print(f"\n{'='*60}")
    print("Final Test Evaluation (using best model)")
    print(f"{'='*60}")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion)

    print(f"\nTest Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Normal Accuracy: {test_metrics['normal_accuracy']:.2f}%")
    print(f"Pneumonia Accuracy: {test_metrics['pneumonia_accuracy']:.2f}%")
    print(f"Sensitivity: {test_metrics['sensitivity']:.2f}%")
    print(f"Specificity: {test_metrics['specificity']:.2f}%")
    print(f"\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])
    print(f"\nClassification Report:")
    print(classification_report(test_metrics["labels"], test_metrics["predictions"],
                                target_names=["Normal", "Pneumonia"]))

    # Log final test metrics to W&B
    wandb.log({
        "test_accuracy": test_metrics["accuracy"],
        "normal_accuracy": test_metrics["normal_accuracy"],
        "pneumonia_accuracy": test_metrics["pneumonia_accuracy"],
        "sensitivity": test_metrics["sensitivity"],
        "specificity": test_metrics["specificity"],
    })
    wandb.summary["test_accuracy"] = test_metrics["accuracy"]
    wandb.summary["normal_accuracy"] = test_metrics["normal_accuracy"]
    wandb.summary["pneumonia_accuracy"] = test_metrics["pneumonia_accuracy"]
    wandb.summary["sensitivity"] = test_metrics["sensitivity"]
    wandb.summary["specificity"] = test_metrics["specificity"]

    # Save model artifact
    artifact = wandb.Artifact("pneumonia-classifier", type="model",
                              description=f"ViT-B/16 test_acc={test_metrics['accuracy']:.1f}%")
    artifact.add_file("best_model.pth")
    wandb.log_artifact(artifact)

    wandb.finish()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
