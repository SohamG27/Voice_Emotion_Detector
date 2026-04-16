#  Speech Emotion Recognition (SER) Model

A production-ready Speech Emotion Recognition (SER) system built using **OpenSMILE ComParE_2016 features** and a **LightGBM classifier**, trained on multiple benchmark emotional speech datasets.

This model detects human emotions directly from raw `.wav` audio files with high accuracy and efficiency.

---

##  Features

*  Multi-dataset training for strong generalization
*  High-dimensional acoustic feature extraction using OpenSMILE
*  Fast and efficient LightGBM classifier
*  Supports 6 core emotions:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad
* Automated training logs and model versioning
* Ready-to-deploy serialized model (`.pkl`)

---

## 🧠 Model Architecture

| Component          | Technology                     |
| ------------------ | ------------------------------ |
| Feature Extraction | OpenSMILE (ComParE_2016)       |
| Feature Type       | Functionals (~6,000+ features) |
| Classifier         | LightGBM                       |
| Label Encoding     | Scikit-learn                   |
| Evaluation         | Stratified Train/Test Split    |

---

##  Datasets Used

The model is trained on a combination of widely used emotional speech datasets:

* **CREMA-D**
* **RAVDESS**
* **TESS**
* **SAVEE**

These datasets provide diversity in:

* Speakers
* Accents
* Recording environments
* Emotional expressions

---

##  Installation

```bash
pip install numpy scikit-learn lightgbm opensmile joblib tqdm
```

---

##  Project Structure

```
.
├── main.py
├── models/
│   ├── audio_emotion_model_*.pkl
│   └── label_encoder_*.pkl
├── .venv/
│   ├── models/
│   └── logs/
└── datasets/
    ├── CREMA/
    ├── RAVDESS/
    ├── TESS/
    └── SAVEE/
```

---

## Training

Run the training pipeline:

```bash
python main.py
```

### Pipeline Overview

* Loads audio from all datasets
* Extracts ComParE_2016 features
* Encodes emotion labels
* Performs stratified train/test split
* Trains LightGBM classifier
* Evaluates performance
* Saves model, encoder, and logs

---

##  Evaluation

The model outputs:

*  Accuracy Score
*  Full Classification Report (precision, recall, F1-score)

**Expected Performance:**

```
Accuracy: ~85–95% (varies with dataset quality and balance)
```

---

##  Model Output

Saved artifacts:

* `models/audio_emotion_model_<timestamp>.pkl`
* `models/label_encoder_<timestamp>.pkl`

Latest versions:

* `.venv/models/audio_emotion_model.pkl`
* `.venv/models/label_encoder.pkl`

---

##  Inference Example

```python
import joblib
import opensmile

# Load model
model = joblib.load("models/audio_emotion_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Initialize feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# Extract features
features = smile.process_file("test.wav").values.astype("float32")

# Predict emotion
prediction = model.predict(features)
emotion = encoder.inverse_transform(prediction)

print("Predicted Emotion:", emotion[0])
```

---

##  Logging

Training logs are stored in:

```
.venv/logs/training_log.txt
```

Each run records:

* Timestamp
* Emotion classes
* Accuracy
* Model path

---

## 🔧 Customization

You can extend the system by:

*  Adding new datasets
*  Tuning LightGBM hyperparameters
*  Integrating auto-collected data (`auto_collected.csv`)
*  Adding cross-validation
*  Balancing dataset distribution

---

## Limitations

* Emotion recognition is inherently subjective
* Performance may vary with:

  * Background noise
  * Language/accent differences
  * Audio quality
* Limited to 6 emotion classes (no "surprise")

---

##  Future Improvements

*  Real-time emotion detection
*  Deep learning models (CNN / LSTM / Transformers)
*  Multilingual support
*  Model optimization for edge devices
*  Evaluation on real-world noisy data

---

##  License

This project is intended for research and production prototyping.
Ensure compliance with dataset licenses before redistribution.

---

##  Contribution

Contributions are welcome.

* Fork the repository
* Create a feature branch
* Submit a pull request

For major changes, please open an issue first.
