import sys
import numpy as np
import pandas as pd
import librosa
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# -------------------------
# 1. Load Kaggle dataset
# -------------------------
def load_dataset():
    dataset_path = kagglehub.dataset_download("primaryobjects/voicegender")
    csv_path = f"{dataset_path}/voice.csv"
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['mode', 'centroid'], errors='ignore')
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y

# -------------------------
# 2. Train ensemble model
# -------------------------
def train_models(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, max_features='sqrt', random_state=42
    )

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )

    # Voting ensemble (soft voting averages probabilities)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting="soft",
        n_jobs=-1
    )

    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[INFO] Ensemble Test Accuracy: {acc:.3f}")

    return ensemble, X.columns, scaler, le

# -------------------------
# 3. Feature extraction (with MFCC, Chroma, Spectral features)
# -------------------------
def extract_features_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)

    feats = {}

    # --- Existing spectral stats ---
    S, phase = librosa.magphase(librosa.stft(y))
    feats['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    feats['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
    feats['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
    feats['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    feats['rms'] = np.mean(librosa.feature.rms(y=y))

    # --- MFCCs (13 coefficients) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        feats[f"mfcc{i+1}_var"] = np.var(mfcc[i])

    # --- Chroma ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_{i+1}_mean"] = np.mean(chroma[i])

    return feats

# -------------------------
# 4. Predict gender
# -------------------------
def predict_gender(model, scaler, feature_names, audio_path, label_encoder):
    feats = extract_features_from_audio(audio_path)

    # Align features with training columns (missing features filled with 0)
    X_new = pd.DataFrame([feats])
    for col in feature_names:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[feature_names]

    X_new_scaled = scaler.transform(X_new)

    pred_enc = model.predict(X_new_scaled)[0]
    proba = model.predict_proba(X_new_scaled)[0]
    pred = label_encoder.inverse_transform([pred_enc])[0]
    conf = proba[pred_enc]
    return pred, conf

# -------------------------
# 5. Main script
# -------------------------
if __name__ == "__main__":
    X, y = load_dataset()
    model, feat_names, scaler, le = train_models(X, y)

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        print("Usage: python voice_gender_final.py <audiofile.wav|.mp3>")
        sys.exit(1)

    label, confidence = predict_gender(model, scaler, feat_names, test_file, le)
    print(f"Predicted Gender: {label} ({confidence*100:.1f}% confidence)")
