# Unity-Sign-Language-Detection-Model

## Data Preparation

Will extract frames of each video and make each video of each word has same frame count in order not to have problems while trainning the model. \n
If frame counts wont hold between videos blank frames will be added to beggining and end of videos. \n
Each video of each word has numpy shape of (120, 256, 256, 3)
