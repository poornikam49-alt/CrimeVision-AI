
import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

DATA_DIR = "dataset_split"
MODEL_DIR = "models/fighting_videomae"
FRAMES_PER_VIDEO = 16

device = torch.device("cpu")

processor = VideoMAEImageProcessor.from_pretrained(MODEL_DIR)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

print("\nClass labels:", model.config.id2label)

for split in ["train", "val"]:
    print(f"\n--- {split.upper()} RESULTS ---")

    for class_name in ["fight", "non_fight"]:
        folder = os.path.join(DATA_DIR, split, class_name)

        if not os.path.isdir(folder):
            print("Missing folder:", folder)
            continue

        for filename in os.listdir(folder):
            if not filename.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):
                continue

            path = os.path.join(folder, filename)
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if not cap.isOpened() or total_frames <= 0:
                print("Cannot read:", filename)
                cap.release()
                continue

            positions = np.linspace(
                0, total_frames - 1, FRAMES_PER_VIDEO
            ).astype(int)

            frames = []
            for position in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                success, frame = cap.read()

                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))

            cap.release()

            if len(frames) != FRAMES_PER_VIDEO:
                print("Skipped unreadable video:", filename)
                continue

            inputs = processor(images=frames, return_tensors="pt")

            with torch.no_grad():
                outputs = model(
                    pixel_values=inputs["pixel_values"].to(device)
                )
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            scores = {
                model.config.id2label[i]: round(
                    probs[i].item() * 100, 2
                )
                for i in range(len(probs))
            }
            predicted = max(scores, key=scores.get)

            print(
                f"Actual: {class_name:10} | "
                f"Predicted: {predicted:10} | "
                f"Scores: {scores} | "
                f"File: {filename}"
            )

print("\nTest complete. No files were modified.")
