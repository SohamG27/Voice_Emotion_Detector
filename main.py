# ✅ Updated main.py with CREMA-D + RAVDESS + ComParE_2016 Features

import os
import opensmile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import joblib
from tqdm import tqdm
from datetime import datetime

# File paths
CREMA_PATH = r"D:/Crema"
RAVDESS_PATH = r"D:/vdir"
TESS_PATH=r"D:/TESS"
SAVEE_PATH=r"D:/SAVEE"
AUTO_DATA_CSV = "auto_collected.csv"
LOG_FILE = ".venv/logs/training_log.txt"

# Ensure necessary folders exist
os.makedirs(".venv/models", exist_ok=True)
os.makedirs(".venv/logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# OpenSMILE feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# Emotion mapping
crema_emotion_map = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

ravdess_emotion_map = {
    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust'
}

X, y = [], []

# Load TESS Dataset
print("🔁 Loading TESS samples...")
for root, dirs, files in os.walk(TESS_PATH):
    for file in files:
        if file.endswith(".wav"):
            try:
                emotion = None
                if "angry" in file.lower():
                    emotion = "angry"
                elif "disgust" in file.lower():
                    emotion = "disgust"
                elif "fear" in file.lower():
                    emotion = "fear"
                elif "happy" in file.lower():
                    emotion = "happy"
                elif "neutral" in file.lower():
                    emotion = "neutral"
                elif "sad" in file.lower():
                    emotion = "sad"
                # Skip 'ps' (pleasant surprise) for now
                if emotion:
                    path = os.path.join(root, file)
                    features = smile.process_file(path).values[0]
                    X.append(features)
                    y.append(emotion)
            except Exception as e:
                print(f"⚠️ Error processing TESS file {file}: {e}")
# Load CREMA-D
print("🔁 Loading CREMA-D samples...")
for file in tqdm(os.listdir(CREMA_PATH)):
    if file.endswith(".wav"):
        try:
            code = file.split("_")[2]
            emotion = crema_emotion_map.get(code)
            if emotion:
                path = os.path.join(CREMA_PATH, file)
                features = smile.process_file(path).values[0]
                X.append(features)
                y.append(emotion)
        except Exception as e:
            print(f"⚠️ Error processing CREMA-D file {file}: {e}")

# Load SAVEE Dataset
print("🔁 Loading SAVEE samples...")
savee_emotion_map = {
    'a': 'angry',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happy',
    'n': 'neutral',
    'sa': 'sad',
}
for file in os.listdir(SAVEE_PATH):
    if file.endswith(".wav"):
        try:
            prefix = file.split('_')[1][0:2].lower()  # e.g., 'sa', 'a', etc.
            emotion = savee_emotion_map.get(prefix)
            if emotion:
                path = os.path.join(SAVEE_PATH, file)
                features = smile.process_file(path).values[0]
                X.append(features)
                y.append(emotion)
        except Exception as e:
            print(f"⚠️ Error processing SAVEE file {file}: {e}")
# Load RAVDESS
print("🔁 Loading RAVDESS samples...")
for root, dirs, files in os.walk(RAVDESS_PATH):
    for file in files:
        if file.endswith(".wav"):
            try:
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = ravdess_emotion_map.get(emotion_code)
                if emotion:
                    path = os.path.join(root, file)
                    features = smile.process_file(path).values[0]
                    X.append(features)
                    y.append(emotion)
            except Exception as e:
                print(f"⚠️ Error processing RAVDESS file {file}: {e}")

# Load auto_collected.csv if exists
'''auto_count = 0
if os.path.exists(AUTO_DATA_CSV):
    auto_df = pd.read_csv(AUTO_DATA_CSV)
    auto_X = auto_df.iloc[:, :-1].values
    auto_y = auto_df["label"].values
    auto_count = len(auto_y)

    print(f"📥 Loaded {auto_count} auto-collected samples.")

    X = np.vstack([X, auto_X])
    y = np.concatenate([y, auto_y])

X = np.array(X)
y = np.array(y)'''

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# XGBoost expects float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Train lightgbm
clf = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print("\n📊 Classification Report:")
print(report)

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/audio_emotion_model_{timestamp}.pkl"
encoder_path = f"models/label_encoder_{timestamp}.pkl"

joblib.dump(clf, model_path)
joblib.dump(le, encoder_path)

# Save latest
joblib.dump(clf, ".venv/models/audio_emotion_model.pkl")
joblib.dump(le, ".venv/models/label_encoder.pkl")

print(f"\n✅ Model saved as: {model_path}")

# Log training
with open(LOG_FILE, "a") as log:
    log.write(f"--- {datetime.now()} ---\n")
    '''log.write(f"Total Samples: {len(y)} | Auto-Collected: {auto_count}\n")'''
    log.write(f"Unique Emotions: {list(le.classes_)}\n")
    log.write(f"Accuracy: {acc * 100:.2f}%\n")
    log.write(f"Saved Model: {model_path}\n\n")
