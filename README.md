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

- Epoch [1/20] | Train Loss: 2.9426 | Val Loss: 2.8036 | Val Acc: 15.00%
- Epoch [3/20] | Train Loss: 2.6649 | Val Loss: 2.7415 | Val Acc: 13.33%
- Epoch [2/20] | Train Loss: 2.7332 | Val Loss: 2.7787 | Val Acc: 13.33%
- Epoch [4/20] | Train Loss: 2.6351 | Val Loss: 2.7327 | Val Acc: 16.67%
- Epoch [5/20] | Train Loss: 2.6161 | Val Loss: 2.7207 | Val Acc: 18.33%
- Epoch [6/20] | Train Loss: 2.5942 | Val Loss: 2.7522 | Val Acc: 18.33%
- Epoch [7/20] | Train Loss: 2.5669 | Val Loss: 2.7204 | Val Acc: 20.00%
- Epoch [8/20] | Train Loss: 2.5593 | Val Loss: 2.7328 | Val Acc: 16.67%
- Epoch [9/20] | Train Loss: 2.5374 | Val Loss: 2.7250 | Val Acc: 18.33%
- Epoch [10/20] | Train Loss: 2.5352 | Val Loss: 2.7103 | Val Acc: 18.33%
- Epoch [11/20] | Train Loss: 2.5023 | Val Loss: 2.7313 | Val Acc: 20.00%
- Epoch [12/20] | Train Loss: 2.5070 | Val Loss: 2.7889 | Val Acc: 16.67%
- Epoch [13/20] | Train Loss: 2.4987 | Val Loss: 2.7675 | Val Acc: 20.00%
- Epoch [14/20] | Train Loss: 2.4679 | Val Loss: 2.7913 | Val Acc: 20.00%
- Epoch [15/20] | Train Loss: 2.4636 | Val Loss: 2.7707 | Val Acc: 18.33%
- Epoch [16/20] | Train Loss: 2.4491 | Val Loss: 2.8213 | Val Acc: 18.33%
- Epoch [17/20] | Train Loss: 2.4609 | Val Loss: 2.8498 | Val Acc: 18.33%
- Epoch [18/20] | Train Loss: 2.4599 | Val Loss: 2.8224 | Val Acc: 18.33%
- Epoch [19/20] | Train Loss: 2.4339 | Val Loss: 2.8308 | Val Acc: 16.67%
- Epoch [20/20] | Train Loss: 2.4154 | Val Loss: 2.8442 | Val Acc: 15.00%

  - Test Loss: 2.7634, Test Accuracy: 8.33%
  - Model saved.

- Tested twice with different unseen dataset
- Room -> Right, Break
- Teach -> Train, Right
- Class -> Car, Book, Bicycle

---

=== Testing Model on Training Data ===

Predicted: right, Actual: answer
Predicted: teach, Actual: bicycle
Predicted: die, Actual: book
Predicted: die, Actual: break
Predicted: right, Actual: car
Predicted: room, Actual: class
Predicted: walk, Actual: correct
Predicted: walk, Actual: die
Predicted: lose, Actual: exam
Predicted: left, Actual: how
Predicted: right, Actual: left
Predicted: lose, Actual: lose --- epochta exam predictlemiş
Predicted: page, Actual: model --- epochta class predictlemiş
Predicted: right, Actual: now --- epochta teach predictlemiş
Predicted: right, Actual: page --- epochta teach predictlemiş
Predicted: right, Actual: right --- epochta left predictlemiş
Predicted: room, Actual: room --- epochta class predictlemiş
Predicted: page, Actual: teach --- aynı
Predicted: train, Actual: train ---- aynı
Predicted: walk, Actual: walk ---- aynı
Test Accuracy: 25.00%

--- Epoch 15 ---
Sample 1: Predicted: now, Actual: answer
Sample 2: Predicted: bicycle, Actual: bicycle
Sample 3: Predicted: correct, Actual: book
Sample 4: Predicted: book, Actual: break
Sample 5: Predicted: teach, Actual: car
Sample 6: Predicted: room, Actual: class
Sample 7: Predicted: break, Actual: correct
Sample 8: Predicted: train, Actual: die
Sample 9: Predicted: exam, Actual: exam
Sample 10: Predicted: bicycle, Actual: how
Sample 11: Predicted: right, Actual: left
Sample 12: Predicted: exam, Actual: lose
Sample 13: Predicted: bicycle, Actual: model
Sample 14: Predicted: teach, Actual: now
Sample 15: Predicted: teach, Actual: page
Sample 16: Predicted: left, Actual: right
Sample 17: Predicted: class, Actual: room
Sample 18: Predicted: page, Actual: teach
Sample 19: Predicted: train, Actual: train
Sample 20: Predicted: walk, Actual: walk
Correct: 4
Total: 20
Epoch [15/15] | Train Loss: 1.7524 | Train Acc: 20.00%
Model saved.

---

--- Epoch 15 ---
Sample 1: Predicted: now, Actual: answer
Sample 2: Predicted: bicycle, Actual: bicycle
Sample 3: Predicted: die, Actual: book
Sample 4: Predicted: walk, Actual: break
Sample 5: Predicted: car, Actual: car
Sample 6: Predicted: room, Actual: class
Sample 7: Predicted: walk, Actual: correct
Sample 8: Predicted: walk, Actual: die
Sample 9: Predicted: exam, Actual: exam
Sample 10: Predicted: car, Actual: how
Sample 11: Predicted: left, Actual: left
Sample 12: Predicted: lose, Actual: lose
Sample 13: Predicted: now, Actual: model
Sample 14: Predicted: car, Actual: now
Sample 15: Predicted: answer, Actual: page
Sample 16: Predicted: right, Actual: right
Sample 17: Predicted: room, Actual: room
Sample 18: Predicted: bicycle, Actual: teach
Sample 19: Predicted: train, Actual: train
Sample 20: Predicted: break, Actual: walk
Correct: 8
Total: 20
Epoch [15/15] | Train Loss: 1.5790 | Train Acc: 40.00%
Model saved.

=== Testing Model on Training Data ===

Predicted: model, Actual: answer
Predicted: walk, Actual: bicycle
Predicted: train, Actual: book
Predicted: train, Actual: break
Predicted: model, Actual: car
Predicted: class, Actual: class
Predicted: train, Actual: correct
Predicted: train, Actual: die
Predicted: exam, Actual: exam
Predicted: bicycle, Actual: how
Predicted: left, Actual: left
Predicted: exam, Actual: lose
Predicted: bicycle, Actual: model
Predicted: model, Actual: now
Predicted: model, Actual: page
Predicted: right, Actual: right
Predicted: room, Actual: room
Predicted: bicycle, Actual: teach
Predicted: train, Actual: train
Predicted: break, Actual: walk

Test Accuracy: 30.00%

---

Lasl Epoch With Probabilities Distribution:
--- Epoch 15 ---
Sample 1:
Predicted: break, Actual: book
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 2:
Predicted: car, Actual: car
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 3:
Predicted: model, Actual: model
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 4:
Predicted: right, Actual: left
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 5:
Predicted: right, Actual: right
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 6:
Predicted: room, Actual: room
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 7:
Predicted: lose, Actual: lose
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 8:
Predicted: train, Actual: train
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 9:
Predicted: model, Actual: how
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 10:
Predicted: book, Actual: walk
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 11:
Predicted: car, Actual: now
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 12:
Predicted: car, Actual: answer
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 13:
Predicted: bicycle, Actual: bicycle
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 14:
Predicted: walk, Actual: die
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 15:
Predicted: book, Actual: break
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 16:
Predicted: room, Actual: class
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 17:
Predicted: car, Actual: page
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 18:
Predicted: walk, Actual: correct
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 19:
Predicted: exam, Actual: exam
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Sample 20:
Predicted: model, Actual: teach
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%
Correct: 8
Total: 20
Epoch [15/15] | Train Loss: 1.4420 | Train Acc: 40.00%
Model saved.

Test File: model-1.npy
Predicted Label: car
Probabilities:
answer: 9.22%
bicycle: 6.43%
book: 1.65%
break: 1.26%
car: 11.32%
class: 1.14%
correct: 1.56%
die: 2.53%
exam: 1.56%
how: 8.43%
left: 9.00%
lose: 1.77%
model: 6.98%
now: 8.38%
page: 8.79%
right: 6.42%
room: 1.32%
teach: 7.58%
train: 1.52%
walk: 3.12%
Test File: break-1.npy
Predicted Label: walk
Probabilities:
answer: 1.78%
bicycle: 8.78%
book: 8.73%
break: 6.19%
car: 2.24%
class: 0.52%
correct: 13.42%
die: 13.77%
exam: 0.92%
how: 3.38%
left: 1.39%
lose: 1.24%
model: 6.73%
now: 2.84%
page: 2.80%
right: 1.43%
room: 0.60%
teach: 5.54%
train: 3.92%
walk: 13.78%
Test File: train-1.npy
Predicted Label: correct
Probabilities:
answer: 0.80%
bicycle: 2.04%
book: 17.13%
break: 19.52%
car: 0.79%
class: 0.35%
correct: 19.77%
die: 11.18%
exam: 1.54%
how: 0.86%
left: 0.46%
lose: 2.29%
model: 1.40%
now: 0.69%
page: 0.88%
right: 0.76%
room: 0.38%
teach: 1.03%
train: 12.50%
walk: 5.62%
Test File: room-1.npy
Predicted Label: class
Probabilities:
answer: 2.50%
bicycle: 2.67%
book: 3.73%
break: 2.56%
car: 2.62%
class: 23.02%
correct: 2.56%
die: 3.37%
exam: 7.75%
how: 3.49%
left: 3.03%
lose: 6.57%
model: 2.68%
now: 3.31%
page: 3.06%
right: 2.30%
room: 15.21%
teach: 3.56%
train: 3.77%
walk: 2.25%
Test File: how-1.npy
Predicted Label: car
Probabilities:
answer: 9.60%
bicycle: 6.08%
book: 1.67%
break: 1.31%
car: 11.44%
class: 1.21%
correct: 1.54%
die: 2.46%
exam: 1.63%
how: 8.08%
left: 9.63%
lose: 1.86%
model: 6.49%
now: 8.23%
page: 8.78%
right: 6.87%
room: 1.40%
teach: 7.10%
train: 1.59%
walk: 3.03%
Test File: car-1.npy
Predicted Label: car
Probabilities:
answer: 9.91%
bicycle: 5.69%
book: 1.73%
break: 1.43%
car: 11.29%
class: 1.36%
correct: 1.58%
die: 2.41%
exam: 1.77%
how: 7.59%
left: 10.45%
lose: 2.03%
model: 5.90%
now: 7.94%
page: 8.68%
right: 7.48%
room: 1.57%
teach: 6.51%
train: 1.74%
walk: 2.97%
Test File: die-1.npy
Predicted Label: bicycle
Probabilities:
answer: 2.57%
bicycle: 12.90%
book: 4.71%
break: 3.07%
car: 3.44%
class: 0.64%
correct: 6.90%
die: 8.94%
exam: 0.86%
how: 5.78%
left: 2.37%
lose: 1.13%
model: 11.49%
now: 4.81%
page: 4.40%
right: 1.96%
room: 0.74%
teach: 10.04%
train: 2.35%
walk: 10.90%
Test File: walk-1.npy
Predicted Label: model
Probabilities:
answer: 3.60%
bicycle: 12.65%
book: 3.03%
break: 1.91%
car: 5.09%
class: 0.76%
correct: 3.95%
die: 5.90%
exam: 0.95%
how: 7.76%
left: 3.44%
lose: 1.18%
model: 12.99%
now: 6.45%
page: 5.75%
right: 2.61%
room: 0.88%
teach: 11.79%
train: 1.77%
walk: 7.53%
Test File: book-1.npy
Predicted Label: walk
Probabilities:
answer: 1.79%
bicycle: 8.84%
book: 8.69%
break: 6.15%
car: 2.25%
class: 0.52%
correct: 13.35%
die: 13.75%
exam: 0.91%
how: 3.40%
left: 1.40%
lose: 1.23%
model: 6.77%
now: 2.86%
page: 2.81%
right: 1.43%
room: 0.60%
teach: 5.58%
train: 3.90%
walk: 13.78%
Test File: right-1.npy
Predicted Label: left
Probabilities:
answer: 9.58%
bicycle: 5.35%
book: 2.05%
break: 1.84%
car: 10.06%
class: 1.92%
correct: 1.86%
die: 2.56%
exam: 2.25%
how: 6.86%
left: 10.75%
lose: 2.53%
model: 5.29%
now: 7.40%
page: 8.27%
right: 8.06%
room: 2.16%
teach: 5.92%
train: 2.18%
walk: 3.11%
Test File: class-1.npy
Predicted Label: class
Probabilities:
answer: 2.54%
bicycle: 2.39%
book: 3.39%
break: 2.58%
car: 2.73%
class: 24.63%
correct: 2.55%
die: 3.08%
exam: 7.65%
how: 3.25%
left: 2.94%
lose: 6.31%
model: 2.54%
now: 3.24%
page: 3.30%
right: 2.32%
room: 15.54%
teach: 3.28%
train: 3.48%
walk: 2.27%
Test File: exam-1.npy
Predicted Label: exam
Probabilities:
answer: 0.78%
bicycle: 0.39%
book: 2.09%
break: 2.84%
car: 0.32%
class: 1.03%
correct: 1.14%
die: 0.80%
exam: 51.56%
how: 0.47%
left: 0.42%
lose: 28.63%
model: 0.22%
now: 0.31%
page: 0.41%
right: 0.49%
room: 0.64%
teach: 0.32%
train: 6.64%
walk: 0.48%
Test File: bicycle-1.npy
Predicted Label: car
Probabilities:
answer: 7.61%
bicycle: 7.84%
book: 1.71%
break: 1.22%
car: 10.19%
class: 1.02%
correct: 1.74%
die: 2.94%
exam: 1.39%
how: 9.27%
left: 7.17%
lose: 1.58%
model: 8.77%
now: 8.57%
page: 8.52%
right: 5.12%
room: 1.18%
teach: 9.12%
train: 1.41%
walk: 3.61%
Test File: correct-1.npy
Predicted Label: bicycle
Probabilities:
answer: 2.56%
bicycle: 12.88%
book: 4.74%
break: 3.10%
car: 3.42%
class: 0.64%
correct: 6.96%
die: 9.00%
exam: 0.86%
how: 5.75%
left: 2.35%
lose: 1.13%
model: 11.45%
now: 4.79%
page: 4.38%
right: 1.95%
room: 0.73%
teach: 9.99%
train: 2.37%
walk: 10.95%
Test File: left-1.npy
Predicted Label: left
Probabilities:
answer: 9.44%
bicycle: 5.33%
book: 2.14%
break: 1.95%
car: 9.75%
class: 2.06%
correct: 1.95%
die: 2.63%
exam: 2.38%
how: 6.76%
left: 10.57%
lose: 2.66%
model: 5.23%
now: 7.31%
page: 8.16%
right: 8.05%
room: 2.30%
teach: 5.87%
train: 2.29%
walk: 3.16%
Test File: page-1.npy
Predicted Label: car
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.28%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.81%
left: 10.11%
lose: 1.94%
model: 6.14%
now: 8.08%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.76%
train: 1.67%
walk: 2.99%
Test File: lose-1.npy
Predicted Label: exam
Probabilities:
answer: 0.90%
bicycle: 0.47%
book: 3.79%
break: 5.70%
car: 0.40%
class: 0.72%
correct: 1.79%
die: 1.18%
exam: 34.91%
how: 0.48%
left: 0.40%
lose: 33.13%
model: 0.26%
now: 0.31%
page: 0.45%
right: 0.56%
room: 0.57%
teach: 0.35%
train: 12.95%
walk: 0.69%
Test File: teach-1.npy
Predicted Label: car
Probabilities:
answer: 9.22%
bicycle: 6.43%
book: 1.65%
break: 1.26%
car: 11.31%
class: 1.14%
correct: 1.56%
die: 2.53%
exam: 1.56%
how: 8.42%
left: 9.00%
lose: 1.78%
model: 6.98%
now: 8.38%
page: 8.79%
right: 6.43%
room: 1.32%
teach: 7.57%
train: 1.52%
walk: 3.12%
Test File: answer-1.npy
Predicted Label: car
Probabilities:
answer: 9.91%
bicycle: 5.69%
book: 1.73%
break: 1.43%
car: 11.29%
class: 1.37%
correct: 1.58%
die: 2.41%
exam: 1.77%
how: 7.59%
left: 10.45%
lose: 2.03%
model: 5.90%
now: 7.94%
page: 8.68%
right: 7.48%
room: 1.57%
teach: 6.51%
train: 1.74%
walk: 2.97%
Test File: now-1.npy
Predicted Label: car
Probabilities:
answer: 9.81%
bicycle: 5.84%
book: 1.70%
break: 1.37%
car: 11.41%
class: 1.29%
correct: 1.55%
die: 2.42%
exam: 1.70%
how: 7.80%
left: 10.11%
lose: 1.95%
model: 6.14%
now: 8.07%
page: 8.73%
right: 7.22%
room: 1.48%
teach: 6.75%
train: 1.67%
walk: 2.99%

- Test File: model | Predicted Label: now
- Test File: break | Predicted Label: break
- Test File: train | Predicted Label: correct
- Test File: room | Predicted Label: walk
- Test File: how | Predicted Label: now
- Test File: car | Predicted Label: lose
- Test File: die | Predicted Label: break
- Test File: walk | Predicted Label: break
- Test File: book | Predicted Label: break
- Test File: right | Predicted Label: lose
- Test File: class | Predicted Label: walk
- Test File: exam | Predicted Label: correct
- Test File: bicycle | Predicted Label: now
- Test File: correct | Predicted Label: break
- Test File: left | Predicted Label: lose
- Test File: page | Predicted Label: now
- Test File: lose | Predicted Label: correct
- Test File: teach | Predicted Label: now
- Test File: answer | Predicted Label: lose
- Test File: now | Predicted Label: now
