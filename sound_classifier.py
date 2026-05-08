import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from signal_generator import SignalGenerator
from signal_processing import SignalProcessor


class SoundClassifier:

    CLASSES = ["speech", "clap", "noise", "whistle"]
    N_ESTIMATORS = 100

    def __init__(self, sample_rate: int = 44100, duration: float = 0.5):
        self.sample_rate = sample_rate
        self.duration    = duration

        self.model     = RandomForestClassifier(
            n_estimators=self.N_ESTIMATORS,
            max_depth=10,
            random_state=42
        )
        self.encoder   = LabelEncoder()
        self.is_trained = False

        self._sig_gen   = SignalGenerator(sample_rate, duration)
        self._processor = SignalProcessor(sample_rate)

    def train(self, samples_per_class: int = 120) -> dict:

        X, y = [], []

        for label in self.CLASSES:
            for _ in range(samples_per_class):
                raw     = self._sig_gen.generate(label)
                filtered= self._processor.bandpass_filter(raw)
                feats   = self._processor.extract_features(filtered)
                X.append(feats)
                y.append(label)

        X = np.array(X)
        y_enc = self.encoder.fit_transform(y)

        cv_scores = cross_val_score(self.model, X, y_enc, cv=5)
        self.model.fit(X, y_enc)
        self.is_trained = True

        result = {
            "cv_accuracy_mean"  : cv_scores.mean(),
            "cv_accuracy_std"   : cv_scores.std(),
            "feature_importances": dict(zip(
                self._feature_names(),
                self.model.feature_importances_
            )),
        }

        print(f"          CV Accuracy: {cv_scores.mean()*100:.1f}% "
              f"± {cv_scores.std()*100:.1f}%")
        return result

    def predict(self, features: np.ndarray):

        if not self.is_trained:
            raise RuntimeError("Call train() before predict().")

        proba   = self.model.predict_proba(features.reshape(1, -1))[0]
        idx     = np.argmax(proba)
        label   = self.encoder.inverse_transform([idx])[0]
        return label, float(proba[idx])

    @staticmethod
    def _feature_names() -> list:
        return [
            "rms", "zcr", "crest_factor", "temporal_centroid",
            "spectral_centroid", "spectral_spread", "spectral_rolloff",
            "spectral_flatness",
            "band_0_500", "band_500_2k", "band_2k_6k", "band_6k_plus",
            "attack_time",
        ]