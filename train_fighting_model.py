
import os

DATASET_DIR = "dataset"
CLASSES = ["fight", "non_fight"]
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

print("CrimeVision AI - Training Readiness Check")
print("=" * 42)

counts = {}

for class_name in CLASSES:
    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(folder):
        counts[class_name] = 0
        print(f"{class_name}: folder missing")
        continue

    videos = [
        name for name in os.listdir(folder)
        if name.lower().endswith(VIDEO_EXTENSIONS)
    ]

    counts[class_name] = len(videos)
    print(f"{class_name}: {len(videos)} video(s)")

if all(counts[name] >= 10 for name in CLASSES):
    print("\nBasic data-count check passed.")
    print("Next: inspect labels and prepare a proper train/validation split.")
else:
    print("\nTraining is not ready yet.")
    print("Collect more varied, correctly labeled clips in both classes.")
    print("Ten per class is only an initial target, not a guarantee of accuracy.")

print("\nNo model was trained and app.py was not changed.")
