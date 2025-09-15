# voice_gender.py

# --- Imports ---
import numpy as np
import pandas as pd
import librosa
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys

# --- 1. Load Kaggle dataset ---
def load_dataset():
    dataset_path = kagglehub.dataset_download("primaryobjects/voicegender")
    csv_path = f"{dataset_path}/voice.csv"
    df = pd.read_csv(csv_path)
    # Drop unhelpful/redundant features
    df = df.drop(columns=['mode', 'centroid'], errors='ignore')
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y


# --- 2. Train model ---
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[INFO] Test Accuracy: {acc:.3f}")
    return clf, X.columns

# --- 3. Extract features from audio ---
def extract_features_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    # pad/truncate to 20s
    target_length = 20 * sr
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    N = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, 1/sr)
    mask = freqs <= 280
    freqs = freqs[mask]
    P = np.abs(Y[mask])**2
    if P.sum() == 0:
        return {}

    P /= P.sum()
    freqs_khz = freqs / 1000

    meanfreq = (P * freqs_khz).sum()
    cdf = np.cumsum(P)
    Q25 = freqs_khz[np.searchsorted(cdf, 0.25)]
    Q75 = freqs_khz[np.searchsorted(cdf, 0.75)]
    median = freqs_khz[np.searchsorted(cdf, 0.5)]
    IQR = Q75 - Q25
    eps = 1e-12
    entropy = -np.sum(P * np.log2(P + eps)) / np.log2(len(P))
    flatness = np.exp(np.mean(np.log(P + eps))) / np.mean(P + eps)
    mean_f = meanfreq
    sd_f = np.sqrt((P * (freqs_khz - mean_f)**2).sum())
    skewness = (P * (freqs_khz - mean_f)**3).sum() / (sd_f**3 + eps)
    kurtosis = (P * (freqs_khz - mean_f)**4).sum() / (sd_f**4 + eps)

    f0, _, _ = librosa.pyin(y, sr=sr, fmin=50, fmax=280)
    f0_vals = f0[~np.isnan(f0)]
    if len(f0_vals) > 0:
        f0_khz = f0_vals / 1000
        meanfun, minfun, maxfun = f0_khz.mean(), f0_khz.min(), f0_khz.max()
    else:
        meanfun = minfun = maxfun = 0.0

    D = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
    freqs_stft = librosa.fft_frequencies(sr=sr, n_fft=4096)
    max_bin = np.max(np.where(freqs_stft <= 280))
    D_low = D[:max_bin+1, :]
    dom_indices = np.argmax(D_low, axis=0)
    dom_freqs = freqs_stft[dom_indices]
    frame_energy = D_low.sum(axis=0)
    dom_freqs_active = dom_freqs[frame_energy > 1e-6 * frame_energy.max()]
    if len(dom_freqs_active) > 0:
        dom_khz = dom_freqs_active / 1000
        meandom, mindom, maxdom = dom_khz.mean(), dom_khz.min(), dom_khz.max()
    else:
        meandom = mindom = maxdom = 0.0
    dfrange = maxdom - mindom

    if len(f0_vals) > 1 and (f0_vals.max() - f0_vals.min()) > 0:
        modindx = np.abs(np.diff(f0_vals)).sum() / (f0_vals.max() - f0_vals.min())
    else:
        modindx = 0.0

    return {
        'meanfreq': meanfreq, 'sd': sd_f, 'median': median,
        'Q25': Q25, 'Q75': Q75, 'IQR': IQR, 'skew': skewness,
        'kurt': kurtosis, 'sp.ent': entropy, 'sfm': flatness,
        'meanfun': meanfun, 'minfun': minfun, 'maxfun': maxfun,
        'meandom': meandom, 'mindom': mindom, 'maxdom': maxdom,
        'dfrange': dfrange, 'modindx': modindx
    }

# --- 4. Predict gender ---
def predict_gender(model, feature_names, audio_path):
    feats = extract_features_from_audio(audio_path)
    # Put into a DataFrame with the right column names
    X_new = pd.DataFrame([feats], columns=feature_names)
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]
    conf = proba[list(model.classes_).index(pred)]
    return pred, conf


# --- Main script ---
if __name__ == "__main__":
    # Train once
    X, y = load_dataset()
    model, feat_names = train_model(X, y)

    # Get filename from command-line argument
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        print("Usage: python voice_gender.py <audiofile.wav|.mp3>")
        sys.exit(1)

    label, confidence = predict_gender(model, feat_names, test_file)
    print(f"Predicted Gender: {label} ({confidence*100:.1f}% confidence)")

