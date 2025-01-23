# Unity-Sign-Language-Detection-Model

## Data Preparation

- Will extract frames of each video and make each video of each word has same frame count in order not to have problems while trainning the model.
- If frame counts wont hold between videos blank frames will be added to beggining and end of videos.
- Each video of each word has numpy shape of (120, 256, 256, 3)
- Splits the processed data to 14 train, 3 validate, 3 test data
- Searched for model architecture
  - Option A: 2D CNN + LSTM (common approach). For each frame, pass it through a CNN (like ResNet) to get a 512-dim or 2048-dim vector, then pass the sequence of vectors into an LSTM.
  - Option B: 3D CNN (like I3D, C3D, or R(2+1)D). This handles the spatiotemporal nature directly, no separate LSTM needed.
  - Option C: Directly feed raw frames to an LSTM. (Not recommended in practice—very large input size.)

### RNN (LSTM)

- Before feeding raw pixel data directly into the LSTM, I plan to use a CNN (e.g., a pretrained ResNet) to extract a compact feature vector from each frame.
- This will greatly reduce the input size per frame and make it more manageable for the LSTM to learn the temporal patterns.

- Tried a sample LSTM training model with 10 epoch with current dataset but result is not that good.

  - Epoch [1/10] | Train Loss: 1.1036 | Val Loss: 1.0839 | Val Acc: 33.33%
  - Epoch [2/10] | Train Loss: 1.0214 | Val Loss: 1.2446 | Val Acc: 44.44%
  - Epoch [3/10] | Train Loss: 0.9519 | Val Loss: 1.1946 | Val Acc: 55.56%
  - Epoch [4/10] | Train Loss: 0.9544 | Val Loss: 1.2375 | Val Acc: 55.56%
  - Epoch [5/10] | Train Loss: 0.9364 | Val Loss: 1.1617 | Val Acc: 55.56%
  - Epoch [6/10] | Train Loss: 0.9017 | Val Loss: 1.2131 | Val Acc: 55.56%
  - Epoch [7/10] | Train Loss: 0.8520 | Val Loss: 1.3265 | Val Acc: 55.56%
  - Epoch [8/10] | Train Loss: 0.8528 | Val Loss: 1.2766 | Val Acc: 66.67%
  - Epoch [9/10] | Train Loss: 0.8131 | Val Loss: 1.3327 | Val Acc: 55.56%
  - Epoch [10/10] | Train Loss: 0.8092 | Val Loss: 1.3636 | Val Acc: 55.56%
  - Test Loss: 1.3916, Test Accuracy: 55.56%

- There is overfitting, caused from small dataset but before going complex we wanted to move with a small dataset.

#### Direct LSTM Approach

- Tested twice with different unseen dataset
- Room -> Room, Room
- Teach -> Teach, Teach
- Class -> Teach, Class

- Class and teach are not very similar to each other

### 3D CNN

- Used same dataset as LSTM trainning dataset

- Epoch [1/10] | Train Loss: 0.7902 | Val Loss: 0.0574 | Val Acc: 100.00%
- Epoch [2/10] | Train Loss: 0.3963 | Val Loss: 0.0095 | Val Acc: 100.00%
- Epoch [3/10] | Train Loss: 0.3648 | Val Loss: 0.0191 | Val Acc: 100.00%
- Epoch [4/10] | Train Loss: 0.3410 | Val Loss: 0.0058 | Val Acc: 100.00%
- Epoch [5/10] | Train Loss: 0.1836 | Val Loss: 0.0063 | Val Acc: 100.00%
- Epoch [6/10] | Train Loss: 0.2123 | Val Loss: 0.0107 | Val Acc: 100.00%
- Epoch [7/10] | Train Loss: 0.2024 | Val Loss: 0.0031 | Val Acc: 100.00%
- Epoch [8/10] | Train Loss: 0.1141 | Val Loss: 0.0057 | Val Acc: 100.00%
- Epoch [9/10] | Train Loss: 0.1413 | Val Loss: 0.0061 | Val Acc: 100.00%
- Epoch [10/10] | Train Loss: 0.0746 | Val Loss: 0.0039 | Val Acc: 100.00%
- Test Loss: 0.0077, Test Accuracy: 100.00%
- 46 minute of trainning

- Tested twice with different unseen dataset
- Room -> Room, Room
- Teach -> Teach, Teach
- Class -> Class, Class

## Retrained Direct LSTM Approach with Larger Dataset (20 videos per 20 word = 400 videos)
