import sys
import numpy as np
import pandas as pd
import librosa
import kagglehub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# -------------------------
# 1. Load dataset
# -------------------------
def load_dataset():
    dataset_path = kagglehub.dataset_download("primaryobjects/voicegender")
    csv_path = f"{dataset_path}/voice.csv"
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['mode', 'centroid'], errors='ignore')  # drop redundant features
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y


# -------------------------
# 2. Train models (RF + XGBoost)
# -------------------------
def train_models(X, y):
    # Encode labels for XGBoost
    le = LabelEncoder()
    y_enc = le.fit_transform(y)  # 'female'->0, 'male'->1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # --- Random Forest with hyperparameter tuning ---
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [None, 20, 30],
        'max_features': ['sqrt', 'log2']
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    rf_acc = accuracy_score(y_test, best_rf.predict(X_test))
    print(f"[INFO] Random Forest best params: {grid.best_params_}")
    print(f"[INFO] Random Forest Test Accuracy: {rf_acc:.3f}")

    # --- XGBoost model ---
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",  # keep eval_metric
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"[INFO] XGBoost Test Accuracy: {xgb_acc:.3f}")

    # Pick the better model
    if xgb_acc >= rf_acc:
        print("[INFO] Using XGBoost as final model")
        return xgb, X.columns, le
    else:
        print("[INFO] Using Random Forest as final model")
        return best_rf, X.columns, le


# -------------------------
# 3. Extract features from audio
# -------------------------
def extract_features_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
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


# -------------------------
# 4. Predict gender
# -------------------------
def predict_gender(model, feature_names, audio_path, label_encoder):
    feats = extract_features_from_audio(audio_path)
    X_new = pd.DataFrame([feats], columns=feature_names)
    pred_enc = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]
    pred = label_encoder.inverse_transform([pred_enc])[0]
    conf = proba[pred_enc]
    return pred, conf


# -------------------------
# 5. Main script
# -------------------------
if __name__ == "__main__":
    X, y = load_dataset()
    model, feat_names, le = train_models(X, y)

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        print("Usage: python voice_gender1.py <audiofile.wav|.mp3>")
        sys.exit(1)

    label, confidence = predict_gender(model, feat_names, test_file, le)
    print(f"Predicted Gender: {label} ({confidence*100:.1f}% confidence)")
