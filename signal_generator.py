import numpy as np

class SignalGenerator:

    def __init__(self, sample_rate: int = 44100, duration: float = 0.5):
        self.sample_rate = sample_rate
        self.duration    = duration
        self.t           = np.linspace(0, duration,
                                       int(sample_rate * duration),
                                       endpoint=False)

    def generate(self, sound_type: str) -> np.ndarray:
        generators = {
            "speech" : self._speech,
            "clap"   : self._clap,
            "noise"  : self._noise,
            "whistle": self._whistle,
        }
        if sound_type not in generators:
            raise ValueError(f"Unknown sound type '{sound_type}'. "
                             f"Choose from {list(generators.keys())}")

        signal = generators[sound_type]()
        return self._normalise(signal)

    def _speech(self) -> np.ndarray:
        f0          = 150.0
        harmonics   = [1.0, 0.6, 0.3, 0.15]
        signal      = np.zeros_like(self.t)

        for k, amp in enumerate(harmonics, start=1):
            signal += amp * np.sin(2 * np.pi * f0 * k * self.t)

        am_rate  = 5.0
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * am_rate * self.t)
        signal  *= envelope

        signal += 0.05 * np.random.randn(len(self.t))
        return signal

    def _clap(self) -> np.ndarray:
        noise   = np.random.randn(len(self.t))
        attack  = 0.001
        decay   = 0.08

        n_attack = max(1, int(attack * self.sample_rate))
        envelope = np.zeros_like(self.t)
        envelope[:n_attack] = np.linspace(0, 1, n_attack)
        envelope[n_attack:] = np.exp(-5.0 * (self.t[n_attack:] - self.t[n_attack]) / decay)

        signal = noise * envelope

        fft_sig  = np.fft.rfft(signal)
        freqs    = np.fft.rfftfreq(len(signal), d=1.0/self.sample_rate)
        mask     = (freqs >= 200) & (freqs <= 8000)
        fft_sig  = fft_sig * mask
        signal   = np.fft.irfft(fft_sig, n=len(signal))
        return signal

    def _noise(self) -> np.ndarray:
        white_noise = np.random.randn(len(self.t))
        hum         = 0.1 * np.sin(2 * np.pi * 50 * self.t)
        return white_noise + hum

    def _whistle(self) -> np.ndarray:
        f_center = 2000.0
        vibrato  = 20.0 * np.sin(2 * np.pi * 6.0 * self.t)
        signal   = np.sin(2 * np.pi * (f_center + vibrato) * self.t)
        signal  += 0.02 * np.random.randn(len(self.t))
        return signal

    @staticmethod
    def _normalise(signal: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(signal))
        if peak < 1e-9:
            return signal
        return signal / peak