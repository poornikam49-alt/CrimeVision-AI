
import os
import cv2
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

# ---------------- SETTINGS ----------------

DATA_DIR = "dataset_split"
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

CLASSES = ["fight", "non_fight"]
LABELS = {name: index for index, name in enumerate(CLASSES)}

FRAMES_PER_VIDEO = 16
EPOCHS = 5
MODEL_OUTPUT = "models/fighting_videomae"

device = torch.device("cpu")

print("CrimeVision AI - VideoMAE Training")
print("Device:", device)
print("Loading pretrained model...")

# ---------------- LOAD MODEL ----------------

processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(CLASSES),
    id2label={0: "fight", 1: "non_fight"},
    label2id={"fight": 0, "non_fight": 1},
    ignore_mismatched_sizes=True,
)

model.to(device)

# Freeze the pretrained VideoMAE backbone
for parameter in model.videomae.parameters():
    parameter.requires_grad = False

# Train only the fight/non-fight classifier
for parameter in model.classifier.parameters():
    parameter.requires_grad = True

print("Pretrained backbone frozen.")
print("Only the classification layer will be trained.")

# ---------------- VIDEO DATASET ----------------

class FightingDataset(Dataset):
    def __init__(self, split):
        self.items = []

        for class_name in CLASSES:
            folder = os.path.join(DATA_DIR, split, class_name)

            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"Dataset folder not found: {folder}"
                )

            for filename in os.listdir(folder):
                if filename.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                ):
                    self.items.append(
                        (
                            os.path.join(folder, filename),
                            LABELS[class_name],
                        )
                    )

        if not self.items:
            raise ValueError(f"No videos found for split: {split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]

        video = cv2.VideoCapture(path)

        if not video.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            video.release()
            raise RuntimeError(f"Video has no readable frames: {path}")

        positions = np.linspace(
            0, total_frames - 1, FRAMES_PER_VIDEO
        ).astype(int)

        frames = []

        for position in positions:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            success, frame = video.read()

            if not success:
                video.release()
                raise RuntimeError(
                    f"Could not read frame from: {path}"
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        video.release()

        inputs = processor(
            images=frames,
            return_tensors="pt",
        )

        return inputs["pixel_values"].squeeze(0), label


# ---------------- PREPARE DATA ----------------

train_data = FightingDataset("train")
val_data = FightingDataset("val")

train_loader = DataLoader(
    train_data, batch_size=1, shuffle=True
)
val_loader = DataLoader(
    val_data, batch_size=1, shuffle=False
)

print("Training videos:", len(train_data))
print("Validation videos:", len(val_data))

# ---------------- TRAIN MODEL ----------------

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for step, (pixel_values, labels) in enumerate(
        train_loader, start=1
    ):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        outputs.loss.backward()
        optimizer.step()

        total_loss += outputs.loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Video {step}/{len(train_loader)} | "
            f"Loss: {outputs.loss.item():.4f}"
        )

    print(
        "Average training loss:",
        round(total_loss / len(train_loader), 4),
    )


# ---------------- VALIDATE MODEL ----------------

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for pixel_values, labels in val_loader:
        outputs = model(
            pixel_values=pixel_values.to(device)
        )

        predictions = outputs.logits.argmax(dim=-1).cpu()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total if total else 0.0

print(f"Validation accuracy: {accuracy:.2%}")

# ---------------- SAVE MODEL ----------------

os.makedirs(MODEL_OUTPUT, exist_ok=True)

model.save_pretrained(MODEL_OUTPUT)
processor.save_pretrained(MODEL_OUTPUT)

print("Model saved to:", MODEL_OUTPUT)
print("Training complete.")
print(
    "This prototype is not proof of a crime. "
    "Human review is required."
)
