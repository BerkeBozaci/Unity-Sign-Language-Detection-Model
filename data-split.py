import os
import random
import csv

DATA_DIR = "processed_frames" 
OUTPUT_SPLIT_CSV = "dataset_split.csv" 
random_seed = 42  

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

assert (train_ratio + val_ratio + test_ratio) == 1.0, "Ratios must sum up to 1."

random.seed(random_seed)

dataset_split = []

for label_name in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label_name)
    if not os.path.isdir(label_dir):
        continue  

    all_files = []
    for fname in os.listdir(label_dir):
        if fname.endswith(".npy"):
            full_path = os.path.join(label_dir, fname)
            all_files.append(full_path)

    random.shuffle(all_files)

    total_files = len(all_files)
    train_count = int(train_ratio * total_files)
    val_count = int(val_ratio * total_files)
    test_count = total_files - (train_count + val_count)

    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    for f in train_files:
        dataset_split.append((f, label_name, "train"))
    for f in val_files:
        dataset_split.append((f, label_name, "val"))
    for f in test_files:
        dataset_split.append((f, label_name, "test"))

with open(OUTPUT_SPLIT_CSV, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filepath", "label", "split"])
    for row in dataset_split:
        writer.writerow(row)

print(f"Split completed. {len(dataset_split)} total entries written to {OUTPUT_SPLIT_CSV}.")
