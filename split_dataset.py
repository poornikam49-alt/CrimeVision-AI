
import os
import random
import shutil

SOURCE_DIR = "dataset"
OUTPUT_DIR = "dataset_split"

CLASSES = ["fight", "non_fight"]
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

random.seed(42)

for class_name in CLASSES:
    source_folder = os.path.join(SOURCE_DIR, class_name)
    videos = [
        name for name in os.listdir(source_folder)
        if name.lower().endswith(VIDEO_EXTENSIONS)
    ]

    random.shuffle(videos)

    split_index = int(len(videos) * 0.7)
    training_videos = videos[:split_index]
    validation_videos = videos[split_index:]

    for split_name, selected_videos in [
        ("train", training_videos),
        ("val", validation_videos),
    ]:
        destination = os.path.join(
            OUTPUT_DIR, split_name, class_name
        )
        os.makedirs(destination, exist_ok=True)

        for filename in selected_videos:
            shutil.copy2(
                os.path.join(source_folder, filename),
                os.path.join(destination, filename),
            )

        print(
            f"{split_name}/{class_name}: "
            f"{len(selected_videos)} videos"
        )

print("\nDataset split complete.")
print("Original dataset has been preserved.")
