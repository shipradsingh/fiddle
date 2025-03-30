# 🎻 Fiddle – Detecting Copyright-Infringing Music Using CNN-Based Similarity Models

## Introduction

### Motivation
With so much music being created, shared, and remixed every day, copyright infringement has become a growing concern. Platforms like YouTube, SoundCloud, and TikTok often struggle with flagging music that might be violating copyright—especially when the copying is subtle, like pitch-shifting, tempo adjustment, or sampling a short loop. Manual checks are slow and error-prone. The idea behind this project is to use deep learning to flag music pairs that are suspiciously similar. If a model can detect this kind of similarity reliably, it can be used as a tool to pre-screen content and assist in copyright review.

### Why existing solutions are inadequate
Most music copyright detection today is based on fingerprinting (like Shazam) or metadata. These work well for exact copies but struggle with transformations—like small changes in pitch, tempo, or short melodic sampling. Also, there are very few open tools that try to understand music similarity using deep learning. There’s a research gap in building flexible, learning-based systems that can detect “soft” or approximate similarity between musical pieces.

### Our proposal
We propose to build a CNN-based similarity model for music. Specifically, we’ll extract spectrograms from audio clips and use a Siamese neural network to learn how to measure similarity between two pieces of music. The network will be trained on positive pairs (original + transformed version) and negative pairs (unrelated songs). Our goal is not to legally determine infringement but to build a system that can flag potentially infringing audio pairs for human review. This is a novel framing of the problem, and it builds on existing deep learning methods for audio analysis.

---

## Related Work

### Topic: Music Similarity and Audio Embeddings
- Choi, K. et al. (2017). Convolutional recurrent neural networks for music classification. ICASSP.
- Lee, J. et al. (2018). Sample-level CNNs for Music Auto-tagging Using Raw Waveforms. arXiv:1703.01789.
- van den Oord, A. et al. (2013). Deep content-based music recommendation. NIPS.

_Our work is different because we focus on similarity learning between pairs, not tagging or classification. We use spectrograms and a contrastive or Siamese setup to directly train on music pairs._

### Topic: Siamese Networks and Similarity Learning
- Koch, G. et al. (2015). Siamese Neural Networks for One-shot Image Recognition. ICML Deep Learning Workshop.
- Hadsell, R. et al. (2006). Dimensionality reduction by learning an invariant mapping. CVPR.
- Bromley, J. et al. (1993). Signature verification using a Siamese time delay neural network. NIPS.

_Our model adapts this idea to the audio domain, using spectrograms of music clips. Instead of face verification or image comparison, we are applying this idea to music copyright flagging._

### Topic: Music Plagiarism and Copyright Detection
- Serra, J. et al. (2008). Measuring the Similarity of Harmonic Content in Music Recordings. IEEE Transactions on Audio, Speech, and Language Processing.
- Stamatatos, E. (2009). A Survey of Plagiarism Detection Research. ACM Computing Surveys.

_These works focus more on hand-crafted features or symbolic (MIDI) representations. Our work uses raw audio converted to spectrograms, and deep learning to learn representations instead of designing features._

---

## Methods

We are building a Siamese CNN model that takes two audio segments, processes their spectrograms, and outputs a similarity score. This setup allows the network to learn whether two clips are musically similar enough to warrant a potential copyright flag.

### Dataset and Preprocessing
We will simulate a dataset using a small collection of royalty-free music clips (e.g., from Free Music Archive or GTZAN).

- **Positive Pairs**: An original clip and a modified version (pitch-shifted, time-stretched, added noise, etc.).
- **Negative Pairs**: Two unrelated tracks from different genres or artists.

Each clip will be converted to a **Mel Spectrogram** using:
- 128 Mel bands
- hop length = 512
- window size = 2048

We'll use 3-second overlapping segments to increase training data. Spectrograms will be normalized and padded to size (128, 130).

### Model Architecture
Each branch of the Siamese CNN includes:
- Conv2D (32, 3x3) → ReLU → MaxPool
- Conv2D (64, 3x3) → ReLU → MaxPool
- BatchNorm, Dropout (0.3)
- Flatten → Dense(128)

The two outputs are compared using a distance function (e.g., cosine similarity or Euclidean distance). Optionally, we apply a final dense layer and sigmoid to predict a similarity score.

### Training Setup
- **Loss Function**: Contrastive loss or binary cross-entropy
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 32
- **Epochs**: 30–50 with early stopping
- **Augmentation**: pitch shift, time stretch, Gaussian noise, light reverb

---

## Experiments

We plan two core experiments to test whether our model can detect similarity in audio pairs that mimic common forms of copyright-related copying.

### Experiment 1: Baseline Similarity Detection

**Purpose**:  
Test whether the Siamese CNN can distinguish between original vs. modified clips and unrelated pairs.

**Setup**:
- 80% train, 10% validation, 10% test
- Inputs: Positive (original, modified), Negative (original, unrelated)
- Input size: (1, 128, 130)

**Evaluation Metrics**:
- Accuracy
- ROC-AUC and Precision-Recall curve
- Confusion Matrix

---

### Experiment 2: Threshold-based Flagging

**Purpose**:  
Evaluate whether a simple threshold on similarity score can be used to flag potentially infringing clips.

**Setup**:
- Compare similarity scores for true positive, true negative, and ambiguous pairs
- Select optimal threshold using validation data

**Evaluation Metrics**:
- False positive rate at various thresholds
- Precision and recall for flagging at a chosen threshold

## Contact
For questions, suggestions, or collaborations, feel free to reach out.
