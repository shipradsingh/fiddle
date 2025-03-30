## 1. Datasets

### Recommended
- **GTZAN Dataset**  
  - 1000 audio tracks (30-second clips), across 10 genres.  
  - Can be used to generate simulated "infringing" pairs via pitch shift, time stretch, noise, etc.  
  - [Download Link](http://marsyas.info/downloads/datasets.html)

### Alternative / Supplemental Datasets
- **Free Music Archive (FMA)**  
  - Royalty-free audio dataset with 100,000+ tracks.  
  - Useful for creating both similar and dissimilar clip pairs.  
  - [FMA dataset](https://github.com/mdeff/fma)

- **SecondHandSongs**  
  - Community database for identifying original tracks and their covers.  
  - Can be used to simulate real-world "similar" song cases.  
  - [https://secondhandsongs.com/](https://secondhandsongs.com/)

- **MagnaTagATune** *(optional)*  
  - 25,000 audio clips tagged with genre, mood, and instruments.  
  - May help in selecting diverse negative pairs.  
  - [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)

---

## 2. Libraries and Frameworks

### Deep Learning Framework
- **PyTorch**  
  - Preferred for this project due to flexibility and easier custom model design (e.g., Siamese networks).  
  - [PyTorch](https://pytorch.org/)

### Audio Processing
- **Librosa**  
  - Spectrogram, MFCC extraction, audio augmentation (time-stretching, pitch-shifting).  
  - [Librosa Docs](https://librosa.org/doc/latest/index.html)

- **PyDub** *(optional)*  
  - Useful for trimming, converting, or slicing `.wav` files.  
  - [PyDub](http://pydub.com/)

### Data Manipulation and Visualization
- **NumPy**  
- **Pandas**  
- **Matplotlib**  
- **Seaborn** *(optional)*

### Model Evaluation
- **Scikit-learn**  
  - Accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC.  
  - [scikit-learn](https://scikit-learn.org/stable/)

---

## 3. Resource Requirements

### Hardware
- **GPU Recommended**  
  - Especially for training CNNs on spectrogram data.  
  - NVIDIA RTX / Tesla (Google Colab is fine for prototyping).

- **CPU Only**  
  - Possible, but much slower—especially for training with contrastive loss.

### Software
- **Python 3.8+**
- **Recommended Environment**: `venv` or `conda`
- **Dependencies (requirements.txt):**

  ```bash
  torch
  librosa
  numpy
  matplotlib
  scikit-learn
  pandas
  jupyter
