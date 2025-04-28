## How Our Music Similarity Model Works

**Overview:**  
We use a Siamese Convolutional Neural Network (CNN) to compare two pieces of music and output a score showing how similar they are. This helps detect copied or very similar songs automatically.

---

### 1. Data Collection

We start with a subset of the Free Music Archive (FMA) dataset, using about 8,000 short MP3 audio clips.

---

### 2. Audio Processing

- **Convert MP3 to WAV:** WAV files are easier for computers to process because they contain the raw, uncompressed audio data.
- **Create Spectrograms:** We turn each audio clip into a spectrogram—a visual “heatmap” that shows how the sound’s energy is spread across different frequencies over time. This is what our model actually looks at.

---

### 3. How Many Pairs We Created and How We Split Them

We started with around **8,000** original audio clips from the Free Music Archive (FMA-small) dataset.  
Instead of creating all possible clip pairs (which would be millions), we **manually generated** about **48,000 pairs**:

- **Similar Pairs:** A clip paired with a slightly modified version of itself (e.g., pitch-shifted, time-stretched).
    
- **Different Pairs:** A clip paired with a completely unrelated clip.
    

We balanced the dataset so that roughly half the pairs were similar and half were different.

To train and evaluate the model, we split the pairs as follows:

- **75% for training** (about 36,000 pairs)
    
- **15% for validation** (about 7,200 pairs)
    
- **10% for testing** (about 4,800 pairs)
    

This controlled approach kept the dataset size manageable while still allowing the model to learn generalizable patterns of similarity and difference between songs.

---

### 4. Splitting the Data

- **Training:** 75% of pairs (used to teach the model)
- **Validation:** 15% (used to tune the model)
- **Testing:** 10% (used to check how well the model works on new data)

---

### 5.  Model Training & Architecture

### **How Our Model Works**

![Basic Architectural Flow](/assets/fiddle.png)

- **Siamese CNN:**  
    Our model is called a Siamese Convolutional Neural Network. It’s designed to take in two audio spectrograms at the same time and figure out how similar they are.

### **Why Use a Siamese Network?**

- **Purpose-built for Comparison:**  
    Siamese networks are especially good at comparing two things. Instead of just classifying one input, they learn to measure the “distance” (difference) between two inputs. This makes them perfect for tasks like music similarity, face verification, or signature matching.
- **Learning Similarity:**  
    The network learns what makes two songs “close” (similar) or “far apart” (different) in terms of their musical features.

### **Model Architecture Details**

- **Input:**  
    Each input is a spectrogram (a visual representation of sound), shaped as (1, 128, 130). This means one channel, 128 frequency bands, and 130 time steps.
- **Twin CNN Branches:**  
    Both inputs are passed through the same convolutional neural network (CNN) branch. This branch extracts important features from each spectrogram.
- **Feature Embeddings:**  
    The CNN branch turns each spectrogram into a compact “embedding”—a summary of its most important features.
- **Distance Calculation:**  
    The model calculates the absolute difference between the two embeddings (L1 distance). This captures how similar or different the two inputs are.
- **Similarity Head:**  
    The difference is passed through a small neural network (the “similarity head”) that outputs a score between 0 and 1. A score close to 1 means the clips are very similar; close to 0 means they are very different.

### **Design Choices Explained**

- **Batch Normalization & Dropout:**  
    These techniques help the model learn better and avoid overfitting (memorizing the training data instead of generalizing).
- **Short, Fixed-Length Clips:**  
    Using 3-second clips and fixed-size spectrograms keeps the model’s input consistent and makes training more efficient.
- **Shared Weights:**  
    Both branches of the Siamese network share the same weights, ensuring that both inputs are processed in exactly the same way.

### **Why This Architecture?**

- **Consistency:**  
    By using the same CNN for both inputs, we ensure fair and consistent feature extraction.
- **Efficiency:**  
    Fixed input sizes and a compact architecture allow for fast training and inference.
- **Interpretability:**  
    The similarity score is easy to understand and can be used directly for flagging potentially infringing music.

---

### 6. Training Summary [Detailed]

#### Training Overview:

- Total Epochs: 15
- Training Duration: Approximately 4.5 hours (from 03:14 to 07:52)

#### Best Performance Metrics:

- Best Validation Accuracy: 0.9091 (Epoch 13)
- Best Training Accuracy: 0.9170 (Epoch 15)
- Lowest Training Loss: 0.5437 (Epoch 15)
- Lowest Validation Loss: 0.5570 (Epoch 14)

#### Model Checkpoints: Best model was saved 5 times at epochs:

- Epoch 3 (Val Acc: 0.8995)
- Epoch 4 (Val Acc: 0.9036)
- Epoch 6 (Val Acc: 0.9065)
- Epoch 13 (Val Acc: 0.9091)
- Epoch 14 (Val Acc: 0.9086)

#### Training Progression:

1. Initial Phase (Epochs 1-3):
    
    - Rapid improvement in training accuracy from 0.6172 to 0.8231
    - Validation accuracy improved from 0.8974 to 0.8995
2. Middle Phase (Epochs 4-9):
    
    - Steady improvement in training accuracy
    - Validation performance remained relatively stable
3. Final Phase (Epochs 10-15):
    
    - Training accuracy continued to improve gradually
    - Validation performance showed minor fluctuations

#### Observations:

1. The model shows good convergence with both training and validation metrics improving over time
2. No significant overfitting is observed as validation metrics remain stable
3. The improvements in later epochs are incremental, suggesting the model is approaching optimal performance
4. The final model achieves strong performance with >90% accuracy on both training and validation sets

### 7. How We Measure Success

- **Accuracy:** How often does the model correctly say two clips are similar or different?
- **Precision & Recall:** How well does it avoid false alarms and missed copies?
- **AUC-ROC:** Measures how well the model separates similar from different pairs, even as we change the similarity threshold.
- **Confusion Matrix:** Shows where the model gets confused.

---
### 8. Testing Summary [Detailed]

#### Key Testing Metrics:

1. Accuracy: 0.9011 (90.11%)
2. Precision: 0.8349 (83.49%)
3. Recall: 1.0000 (100%)
4. F1 Score: 0.9100 (91%)
5. AUC-ROC: 0.9925 (99.25%)

#### Error Analysis:

- False Positives: 158
- False Negatives: 0
- Error Rate: 0.0989 (9.89%)

#### Dataset Information:

- Total audio pairs tested: 1598
- Positive pairs: 799
- Negative pairs: 799

#### Testing Process:

- Started at: 2025-04-27 07:52:17
- Completed at: 2025-04-27 07:52:55
- Duration: ~38 seconds
- Testing was done in 50 iterations

#### Performance Analysis:

1. The model shows excellent overall performance with 90.11% accuracy
2. Perfect recall (1.0) indicates the model caught all positive cases
3. The high AUC-ROC score (0.9925) suggests excellent discrimination ability
4. The model tends to be somewhat over-predictive, with 158 false positives but no false negatives
5. Loss values increased gradually throughout testing, starting at 0.1392 and ending at 1.3552

#### Areas for Improvement:

1. The relatively higher number of false positives suggests the model could be fine-tuned to be more selective in its positive predictions
2. The precision (0.8349) could be improved to reduce false positives while maintaining the high recall

Overall, the model shows strong performance with some room for improvement in precision.

---

# Comprehensive Results Analysis

## Training Results (15 epochs)

Initial Performance:

- Train Loss: 0.6736, Train Acc: 61.72%

- Val Loss: 0.5651, Val Acc: 89.74%

Final Performance:

- Train Loss: 0.5437, Train Acc: 91.70%

- Val Loss: 0.5583, Val Acc: 90.53%

Best Performance (Epoch 13):

- Train Loss: 0.5467, Train Acc: 90.85%

- Val Loss: 0.5578, Val Acc: 90.91%

## Test Results

Final Metrics:

- Accuracy: 90.11%

- Precision: 83.49%

- Recall: 100%

- F1 Score: 91.00%

- AUC-ROC: 99.25%

Error Analysis:

- False Positives: 158

- False Negatives: 0

- Error Rate: 9.89%

![Evaluation Results](/results/evaluation_results.png)

## Confusion Matrix

![Confusion Matrix](/results/confusion_matrix.png)

## Key Observations:

1. **Model Convergence**

- Steady improvement in training accuracy (61.72% → 91.70%)
- Validation accuracy remained stable around 90%
- No significant overfitting (small gap between train/val metrics)

2. **Classification Performance**

- Perfect recall (100%): Caught all similar pairs
- High precision (83.49%): Some false positives
- Excellent AUC-ROC (99.25%): Strong discrimination ability

3. **Error Analysis**

- Model tends to be over-sensitive (158 false positives)
- No missed detections (0 false negatives)
- Conservative in classifying similar pairs

4. **Training Stability**

- Loss decreased consistently
- No significant fluctuations
- Converged well by epoch 13

The model demonstrates excellent performance in detecting similar audio pairs with perfect recall and high overall accuracy. It's particularly good at not missing any similar pairs, though it occasionally flags dissimilar pairs as similar. This behavior makes it well-suited for applications where missing similar pairs would be more costly than having false positives.

----

### 7. Demo Results

#### Similarity Scores
Running our demo with various audio modifications produced these results (0.0 = different, 1.0 = identical):

| Modification | Score | Spectrogram Comparison |
|-------------|--------|------------------------|
| Different song | 0.36 | ![Different Songs](/demo_results/different/comparison.png) |
| Pitch shifted up by 1 semitone | 0.94 | ![Slight Pitch](/demo_results/slight_pitch/comparison.png) |
| Pitch shifted up by 4 semitones | 0.87 | ![Heavy Pitch](/demo_results/heavy_pitch/comparison.png) |
| Tempo increased by 10% | 0.54 | ![Slight Tempo](/demo_results/slight_tempo/comparison.png) |
| Tempo increased by 30% | 0.20 | ![Heavy Tempo](/demo_results/heavy_tempo/comparison.png) |
| Light white noise added | 0.85 | ![Light Noise](/demo_results/light_noise/comparison.png) |
| Heavy white noise added | 0.58 | ![Heavy Noise](/demo_results/heavy_noise/comparison.png) |
| Combined (pitch + tempo + noise) | 0.58 | ![Combined](/demo_results/combined/comparison.png) |

#### Analysis of Results
- **Perfect Match**: Scores above 0.90 indicate nearly identical audio
- **Strong Similarity**: Scores 0.80-0.90 suggest minor modifications (slight pitch shift, light noise)
- **Moderate Changes**: Scores 0.50-0.79 indicate significant modifications
- **Major Differences**: Scores below 0.50 suggest different songs or heavy modifications

#### Audio Examples
<details>
<summary>Click to expand audio samples</summary>

- [Original Song](/data/demo_mp3/original.mp3)
- [Different Song](/data/demo_mp3/slight_pitch.mp3)

Modified Versions:
- [Slight Pitch Shift](/data/demo_mp3/slight_pitch.mp3)
- [Heavy Pitch Shift](/data/demo_mp3/heavy_pitch.mp3)
- [Slight Tempo Change](/data/demo_mp3/slight_tempo.mp3)
- [Heavy Tempo Change](/data/demo_mp3/heavy_tempo.mp3)
- [Light Noise](/data/demo_mp3/light_noise.mp3)
- [Heavy Noise](/data/demo_mp3/heavy_noise.mp3)
- [Combined Effects](/data/demo_mp3/combined.mp3)

</details>

#### Detailed Spectrograms
<details>
<summary>Click to see individual spectrograms</summary>

**Original vs Different Song:**
- Original: ![Original](/demo_results/different/song1_spec.png)
- Different: ![Different](/demo_results/different/song2_spec.png)

**Original vs Modified Versions:**
- Heavy Pitch Shift: ![Heavy Pitch](/demo_results/heavy_pitch/song2_spec.png)
- Heavy Tempo Change: ![Heavy Tempo](/demo_results/heavy_tempo/song2_spec.png)
- Heavy Noise: ![Heavy Noise](/demo_results/heavy_noise/song2_spec.png)

</details>
