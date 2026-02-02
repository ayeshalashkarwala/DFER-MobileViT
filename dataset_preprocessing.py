import os
import shutil

source_dir = "datasets/KMUMTCN_Unsorted"
dest_dir = "datasets/KMUMTCN"

emotion_map = {
    'AN': 'Anger',
    'SU': 'Surprise',
    'SA': 'Sadness',
    'HA': 'Happy',
    'DI': 'Disgust',
    'FE': 'Fear'
}

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for emotion in emotion_map.values():
    os.makedirs(os.path.join(dest_dir, emotion), exist_ok=True)

print(f"Processing files from: {source_dir}...")

count = 0
for filename in os.listdir(source_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):

        parts = filename.split('_')

        if len(parts) >= 2:
            emotion_code = parts[1]  # "HA", "SA", etc.

            if emotion_code in emotion_map:
                target_folder = emotion_map[emotion_code]

                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(dest_dir, target_folder, filename)

                shutil.move(src_path, dst_path)
                count += 1
            else:
                print(f"Skipping {filename}: Unknown code '{emotion_code}'")
        else:
            print(f"Skipping {filename}: Format mismatch")

print(f"Success! Moved {count} images into '{dest_dir}'.")