# Ear-Canal EMG-Based Vowel Classification

## Project Overview
This project investigates the feasibility of decoding speech components from electromyography (EMG) signals recorded from the ear canal. Specifically, the model predicts whether a spoken word contains the vowel "A" using EMG signals captured during word production. The dataset includes time-domain EMG signals and spectrogram features, processed for sequential modeling with a BiLSTM network.

## Dataset
The dataset is organized as follows:

- **Training Data**: `train.mat` contains EMG signals, spectrogram features, word labels, and binary target labels indicating the presence of the vowel "A".  
  [Download Training Data](https://smu-my.sharepoint.com/:u:/g/personal/thamilezaia_smu_edu_sg/IQBbfHOUmbRmR6GwmUoo4zC3Ae0JHfrrZE8Ii21Qa43P2N4?e=1zknum)

- **Test Data**: `test.mat` contains EMG signals and labels for evaluation on unseen words.  
  [Download Test Data](https://smu-my.sharepoint.com/:u:/g/personal/thamilezaia_smu_edu_sg/IQBta0PFuKakQr8mJKbmyeh1AcQIo7hyHZkrO37TfqHIY?e=ahRJdh)

The training and test sets have no overlapping words to ensure robust evaluation.

## Methodology
1. **Data Preprocessing**:
   - EMG signals are segmented based on word boundaries extracted from aligned audio.
   - Time-frequency features are generated using Short-Time Fourier Transform (STFT) with 8-second windows and 50% overlap.
   - Spectrograms from both channels are vertically concatenated to form the input feature matrix.
   - Data is normalized using standard scaler fit on the training data.
   - Data augmentation with Gaussian noise is applied to balance classes.

2. **Model Architecture**:
   - Bidirectional Long Short-Term Memory (BiLSTM) network with 64 hidden units captures temporal dependencies in EMG sequences.
   - Batch normalization and dropout (0.5) improve generalization.
   - Two fully connected layers followed by a softmax layer perform binary classification.
   - Trained with Adam optimizer (learning rate 1e-4), binary cross-entropy loss, and early stopping.

3. **Evaluation**:
   - Performance metrics include Accuracy, Precision, Recall (Sensitivity), Specificity, and F1-Score.
   - 5-fold cross-validation on the training set is used for ablation studies and robustness assessment.
   - Independent test set evaluation demonstrates generalization to unseen words.

## Results
- The model demonstrates the feasibility of vowel classification from ear-canal EMG signals.
- Cross-validation and aggregated metrics provide insights into performance variability due to the limited dataset size.

