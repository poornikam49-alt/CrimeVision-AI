
import os
import cv2

DATASET_DIR = "dataset"
CLASSES = ["fight", "non_fight"]

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

print("CrimeVision AI - Dataset Check")
print("-" * 35)

for class_name in CLASSES:
    folder = os.path.join(DATASET_DIR, class_name)
    valid_videos = 0

    if not os.path.isdir(folder):
        print(f"\nMissing folder: {folder}")
        continue

    print(f"\nClass: {class_name}")

    for filename in os.listdir(folder):
        if not filename.lower().endswith(VIDEO_EXTENSIONS):
            continue

        path = os.path.join(folder, filename)
        video = cv2.VideoCapture(path)

        if video.isOpened():
            frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            duration = frames / fps if fps > 0 else 0

            print(
                f"{filename}: "
                f"{frames} frames, "
                f"{duration:.1f} seconds"
            )
            valid_videos += 1
        else:
            print(f"Cannot read: {filename}")

        video.release()

    print(f"Readable videos: {valid_videos}")

print("\nDataset check complete.")
print("This script checks videos; it does not train a model.")
