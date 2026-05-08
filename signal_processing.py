import numpy as np


class SignalProcessor:

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def bandpass_filter(self,
                        signal: np.ndarray,
                        low_hz: float = 80.0,
                        high_hz: float = 8000.0) -> np.ndarray:

        N      = len(signal)
        fft_s  = np.fft.rfft(signal)
        freqs  = np.fft.rfftfreq(N, d=1.0 / self.sample_rate)

        mask        = (freqs >= low_hz) & (freqs <= high_hz)
        fft_s      *= mask

        filtered    = np.fft.irfft(fft_s, n=N)
        return filtered

    def compute_fft(self, signal: np.ndarray):

        N      = len(signal)
        window = np.hanning(N)
        fft_s  = np.fft.rfft(signal * window)
        freqs  = np.fft.rfftfreq(N, d=1.0 / self.sample_rate)

        magnitude = np.abs(fft_s) / (np.sum(window) / 2.0)
        spectrum  = 20 * np.log10(magnitude + 1e-9)
        return freqs, spectrum

    def extract_features(self, signal: np.ndarray) -> np.ndarray:

        N      = len(signal)
        abs_s  = np.abs(signal)

        rms    = np.sqrt(np.mean(signal ** 2)) + 1e-9
        zcr    = np.sum(np.diff(np.sign(signal)) != 0) / N
        crest  = (np.max(abs_s) + 1e-9) / rms
        t_axis = np.arange(N) / self.sample_rate
        t_cent = np.sum(t_axis * abs_s) / (np.sum(abs_s) + 1e-9)

        window     = np.hanning(N)
        fft_s      = np.fft.rfft(signal * window)
        freqs      = np.fft.rfftfreq(N, d=1.0 / self.sample_rate)
        power      = np.abs(fft_s) ** 2 + 1e-9
        total_pwr  = np.sum(power)

        sp_cent    = np.sum(freqs * power) / total_pwr

        sp_spread  = np.sqrt(np.sum((freqs - sp_cent) ** 2 * power) / total_pwr)

        cum_pwr    = np.cumsum(power)
        rolloff_idx= np.searchsorted(cum_pwr, 0.85 * total_pwr)
        rolloff_idx= min(rolloff_idx, len(freqs) - 1)
        sp_rolloff = freqs[rolloff_idx]

        geo_mean   = np.exp(np.mean(np.log(power)))
        ari_mean   = np.mean(power)
        sp_flat    = geo_mean / ari_mean

        def _band_energy(flo, fhi):
            mask = (freqs >= flo) & (freqs < fhi)
            return np.sum(power[mask]) / total_pwr

        be_0_500   = _band_energy(0,    500)
        be_500_2k  = _band_energy(500,  2000)
        be_2k_6k   = _band_energy(2000, 6000)
        be_6k_plus = _band_energy(6000, self.sample_rate / 2)

        peak_val   = np.max(abs_s) + 1e-9
        half_peak  = 0.5 * peak_val
        attack_idx = np.argmax(abs_s >= half_peak)
        attack_t   = attack_idx / self.sample_rate

        features = np.array([
            rms, zcr, crest, t_cent,
            sp_cent, sp_spread, sp_rolloff, sp_flat,
            be_0_500, be_500_2k, be_2k_6k, be_6k_plus,
            attack_t,
        ], dtype=float)

        return features