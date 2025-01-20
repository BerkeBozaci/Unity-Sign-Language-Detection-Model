import os
import cv2
import numpy as np

# # Parameters
TARGET_FRAME_COUNT = 120  # The target number of frames for each video
FRAME_SIZE = (256, 256)  # Resize frames to this size
DATASET_DIR = "videos"  # Path to the directory containing videos organized by word
OUTPUT_DIR = "processed_frames"  # Directory to save processed frames

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
        # Resize frame to the specified size
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    cap.release()

    # Pad or truncate frames to match the target frame count
    total_frames = len(frames)
    blank_frame = np.zeros_like(frames[0]) if frames else np.zeros((*frame_size, 3), dtype=np.uint8)

    if total_frames < target_frame_count:
        # Calculate padding
        padding = target_frame_count - total_frames
        front_pad = padding // 2
        back_pad = padding - front_pad
        frames = [blank_frame] * front_pad + frames + [blank_frame] * back_pad
    elif total_frames > target_frame_count:
        # Truncate frames if there are too many
        frames = frames[:target_frame_count]

    return np.array(frames)

def process_videos(dataset_dir, output_dir, target_frame_count, frame_size):
    """
    Process all videos in the dataset directory, extract and pad frames, and save them.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each word folder
    for word in os.listdir(dataset_dir):
        word_dir = os.path.join(dataset_dir, word)
        if not os.path.isdir(word_dir):
            continue  # Skip non-directory files

        print(f"Processing word: {word}")
        word_output_dir = os.path.join(output_dir, word)
        os.makedirs(word_output_dir, exist_ok=True)

        # Loop through each video in the word folder
        for video_file in os.listdir(word_dir):
            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue  # Skip non-video files

            video_path = os.path.join(word_dir, video_file)
            frames = extract_frames(video_path, target_frame_count, frame_size)

            # Save frames as .npy file
            output_file = os.path.join(word_output_dir, f"{os.path.splitext(video_file)[0]}.npy")
            np.save(output_file, frames)
            print(f"Saved processed frames for {video_file} to {output_file}")

# Run the processing function
process_videos(DATASET_DIR, OUTPUT_DIR, TARGET_FRAME_COUNT, FRAME_SIZE)


## MARK: VERIFICATION OF FRAME COUNTS

# OUTPUT_DIR = "processed_frames"  # Ensure this matches your output directory

# def validate_processed_frames(output_dir, target_frame_count, frame_size):
#     for word in os.listdir(output_dir):
#         word_dir = os.path.join(output_dir, word)
#         if not os.path.isdir(word_dir):
#             continue  # Skip non-directory files
        
#         for file in os.listdir(word_dir):
#             if file.endswith(".npy"):
#                 file_path = os.path.join(word_dir, file)
#                 data = np.load(file_path)
                
#                 # Check the shape
#                 expected_shape = (target_frame_count, *frame_size, 3)  # (120, 256, 256, 3)
#                 if data.shape != expected_shape:
#                     print(f"File {file_path} has incorrect shape: {data.shape}. Expected: {expected_shape}")
#                 else:
#                     print(f"File {file_path} is valid with shape: {data.shape}")

# # Validate the output
# validate_processed_frames(OUTPUT_DIR, TARGET_FRAME_COUNT, FRAME_SIZE)