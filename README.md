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
