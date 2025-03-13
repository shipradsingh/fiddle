## 1. Datasets
### Recommended:
- **GTZAN Dataset**
  - 1000 audio tracks (30-second clips), split into 10 genres.
  - Most popular benchmark dataset for music genre classification tasks.
  - [[Download Link](http://marsyas.info/downloads/datasets.html)](http://marsyas.info/downloads/datasets.html)

### Alternative Datasets (optional):
- **FMA (Free Music Archive)**
  - Large-scale music dataset with over 100,000 tracks across various genres.
  - [[FMA dataset](https://github.com/mdeff/fma)](https://github.com/mdeff/fma)
- **MagnaTagATune**
  - Audio clips annotated with genre tags and mood information.
  - [[MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)

---

## 2. Libraries and Frameworks
### Deep Learning Frameworks (pick one):
- **TensorFlow / Keras**
  - Popular choice, especially suitable for beginners due to its ease-of-use and excellent documentation.
  - Easy integration with image-based CNNs.
  - [[TensorFlow](https://www.tensorflow.org/)](https://www.tensorflow.org/)

- **PyTorch**
  - Widely used in research; provides flexibility and intuitive debugging.
  - Often favored for custom CNN architectures and rapid experimentation.
  - [[PyTorch](https://pytorch.org/)](https://pytorch.org/)

### Audio Processing:
- **Librosa**
  - Industry-standard Python library for audio analysis.
  - Feature extraction (MFCCs, Mel Spectrograms, Chroma).
  - [[Librosa Documentation](https://librosa.org/doc/latest/index.html)](https://librosa.org/doc/latest/index.html)

- **PyDub (Optional)**
  - Useful for audio file manipulation (e.g., clipping, converting formats).
  - [[PyDub Documentation](http://pydub.com/)](http://pydub.com/)

### Data Analysis & Visualization:
- **NumPy**
  - Efficient numerical computations and data manipulation.

- **Matplotlib**
  - Essential for plotting spectrograms, confusion matrices, training curves.

- **Seaborn (Optional)**
  - Advanced visualization, especially useful for better-styled statistical plots.

### Model Evaluation:
- **Scikit-learn**
  - Provides easy-to-use evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix).

---

## 3. Resource Requirements (Hardware & Environment)
### Hardware:
- GPU (recommended for faster training): NVIDIA GPUs (RTX or Tesla series recommended).
- CPU can be used, but training CNNs on audio data will be significantly slower.

### Software Environment:
- Python 3.8+
- Virtual Environment (`venv`, `conda`) recommended.
- Dependencies listed in `requirements.txt`:
```bash
tensorflow # or torch
librosa
numpy
matplotlib
scikit-learn
jupyter # optional for notebooks
pandas # optional for dataset management
```

---

## 4. Helpful Resources and Tutorials
- [[Music Genre Classification with CNN (Kaggle Tutorial)](https://www.kaggle.com/code/andradaolteanu/music-genre-classification-with-cnn-keras)](https://www.kaggle.com/code/andradaolteanu/music-genre-classification-with-cnn-keras)
- [[Librosa Quickstart Guide](https://librosa.org/doc/latest/tutorial.html)](https://librosa.org/doc/latest/tutorial.html)
- [[PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [[Keras CNN Tutorial](https://keras.io/examples/vision/image_classification_from_scratch/)](https://keras.io/examples/vision/image_classification_from_scratch/)
