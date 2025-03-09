# 🎻 Fiddle – Music Genre Classification with Deep Learning

Fiddle is a deep learning-powered music genre classification system that predicts the genre of a given audio track using spectrogram-based analysis. By leveraging Convolutional Neural Networks (CNNs), Fiddle aims to explore how different feature representations—MFCCs, Mel Spectrograms, and Chroma Features—impact classification accuracy.

This project is designed as a comparative study to evaluate which audio features contribute most to genre recognition and whether CNNs outperform simpler models like MLPs for this task.

---

## Project Overview
Music is incredibly diverse, and distinguishing genres can be challenging—even for humans. Fiddle tackles this problem using deep learning and audio signal processing. It transforms raw audio into spectrogram representations and trains a CNN to recognize patterns unique to each genre.

### What Fiddle Does
- Extracts audio features (MFCCs, Mel Spectrograms, Chroma Features)
- Trains a CNN & MLP to classify music into genres
- Compares feature representations to determine the most effective input for genre classification

---

## Project Structure
```
fiddle/
│── data/                 # Dataset (GTZAN or other)
│── notebooks/            # Jupyter notebooks for EDA & model training
│── src/
│   ├── data_processing.py # Audio feature extraction (Librosa)
│   ├── model.py          # CNN and MLP architectures
│   ├── train.py          # Training script
│   ├── evaluate.py       # Model evaluation
│── README.md             # You are here!
│── requirements.txt      # Dependencies
```

---

## Setup & Installation

### 1. Install Dependencies
Fiddle requires Python 3.8+ and the following libraries:
```bash
pip install -r requirements.txt
```
Key dependencies include TensorFlow/PyTorch, Librosa, NumPy, Matplotlib, and Scikit-learn.

### 2. Download the Dataset
The project uses the **GTZAN dataset** for training and evaluation. You can download it from [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and place it in the `data/` directory.

### 3. Run Feature Extraction
To preprocess audio and extract features, run:
```bash
python src/data_processing.py
```

### 4. Train the Model
To train the CNN or MLP model, run:
```bash
python src/train.py --model cnn
```
or
```bash
python src/train.py --model mlp
```

### 5. Evaluate Performance
After training, you can evaluate model performance:
```bash
python src/evaluate.py
```

---

## How It Works

### Step 1: Feature Extraction
Instead of feeding raw waveforms to the model, Fiddle extracts meaningful audio representations:
- **MFCCs**: Capture timbral texture
- **Mel Spectrograms**: Represent frequency over time
- **Chroma Features**: Capture harmonic structure

### Step 2: Training the Model
We train two models and compare their performance:
1. **CNN (Convolutional Neural Network)** – Processes 2D spectrogram images
2. **MLP (Multilayer Perceptron)** – Works on flattened feature vectors

### Step 3: Performance Analysis
The project evaluates which feature set leads to the highest classification accuracy.

---

## Results (To Be Updated)
- Which feature representation performs best?
- Which model generalizes better?
- Training time vs. accuracy trade-off

Results and visualizations will be updated as experiments are conducted.

---

## Contributing
If you'd like to contribute or suggest improvements, feel free to fork this repo and submit a pull request.

---

## Acknowledgments
- **GTZAN Dataset** – Used for training and evaluation
- **Librosa** – Feature extraction and audio processing
- **TensorFlow/PyTorch** – Model training

---

## Contact
For questions, suggestions, or collaborations, feel free to reach out.

