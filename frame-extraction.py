import os
import cv2
import numpy as np

TARGET_FRAME_COUNT = 120  
FRAME_SIZE = (256, 256)  
DATASET_DIR = "videos_10"  
OUTPUT_DIR = "processed_frames_10"  

def extract_frames(video_path, target_frame_count, frame_size):
    """
    Extract frames from a video, pad with blank frames if needed, and return the processed frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    cap.release()

    total_frames = len(frames)
    blank_frame = np.zeros_like(frames[0]) if frames else np.zeros((*frame_size, 3), dtype=np.uint8)

    if total_frames < target_frame_count:
        padding = target_frame_count - total_frames
        front_pad = padding // 2
        back_pad = padding - front_pad
        frames = [blank_frame] * front_pad + frames + [blank_frame] * back_pad
    elif total_frames > target_frame_count:
        frames = frames[:target_frame_count]

    return np.array(frames)

def process_videos(dataset_dir, output_dir, target_frame_count, frame_size):
    """
    Process all videos in the dataset directory, extract and pad frames, and save them.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for word in os.listdir(dataset_dir):
        word_dir = os.path.join(dataset_dir, word)
        if not os.path.isdir(word_dir):
            continue  

        print(f"Processing word: {word}")
        word_output_dir = os.path.join(output_dir, word)
        os.makedirs(word_output_dir, exist_ok=True)

        for video_file in os.listdir(word_dir):
            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue 

            video_path = os.path.join(word_dir, video_file)
            frames = extract_frames(video_path, target_frame_count, frame_size)

            output_file = os.path.join(word_output_dir, f"{os.path.splitext(video_file)[0]}.npy")
            np.save(output_file, frames)
            print(f"Saved processed frames for {video_file} to {output_file}")

process_videos(DATASET_DIR, OUTPUT_DIR, TARGET_FRAME_COUNT, FRAME_SIZE)


## MARK: VERIFICATION OF FRAME COUNTS

# OUTPUT_DIR = "processed_frames"

# def validate_processed_frames(output_dir, target_frame_count, frame_size):
#     for word in os.listdir(output_dir):
#         word_dir = os.path.join(output_dir, word)
#         if not os.path.isdir(word_dir):
#             continue  
        
#         for file in os.listdir(word_dir):
#             if file.endswith(".npy"):
#                 file_path = os.path.join(word_dir, file)
#                 data = np.load(file_path)
                
#                 expected_shape = (target_frame_count, *frame_size, 3)  # (120, 256, 256, 3)
#                 if data.shape != expected_shape:
#                     print(f"File {file_path} has incorrect shape: {data.shape}. Expected: {expected_shape}")
#                 else:
#                     print(f"File {file_path} is valid with shape: {data.shape}")

# validate_processed_frames(OUTPUT_DIR, TARGET_FRAME_COUNT, FRAME_SIZE)