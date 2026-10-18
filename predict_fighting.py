
import os
import cv2
import torch
import numpy as np

from PIL import Image
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

MODEL_DIR = "models/fighting_videomae"
FRAMES_PER_VIDEO = 16

video_path = input("Enter the full video path: ").strip().strip('"')

if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

print("Loading trained model...")

processor = VideoMAEImageProcessor.from_pretrained(MODEL_DIR)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR)
model.eval()

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise RuntimeError("Could not open this video.")

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames <= 0:
    video.release()
    raise RuntimeError("No readable frames found.")

positions = np.linspace(
    0, total_frames - 1, FRAMES_PER_VIDEO
).astype(int)

frames = []

for position in positions:
    video.set(cv2.CAP_PROP_POS_FRAMES, int(position))
    success, frame = video.read()

    if not success:
        video.release()
        raise RuntimeError("Could not read a video frame.")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(Image.fromarray(frame))

video.release()

inputs = processor(images=frames, return_tensors="pt")

with torch.no_grad():
    outputs = model(pixel_values=inputs["pixel_values"])
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

print("\nPrediction results:")

for index, probability in enumerate(probabilities):
    label = model.config.id2label[index]
    print(f"{label}: {probability.item() * 100:.2f}%")

predicted_index = probabilities.argmax().item()
predicted_label = model.config.id2label[predicted_index]

print("\nPredicted class:", predicted_label)
print("Experimental result only; human review is required.")
