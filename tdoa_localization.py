import numpy as np
from itertools import combinations


class TDOALocalizer:

    def __init__(self,
                 mic_positions: np.ndarray,
                 speed_sound:   float = 343.0,
                 sample_rate:   int   = 8000):

        self.positions = np.array(mic_positions, dtype=float)
        self.num_mics  = len(mic_positions)
        self.c         = speed_sound
        self.fs        = sample_rate
        self.pairs     = list(combinations(range(self.num_mics), 2))

        max_d = max(
            np.linalg.norm(self.positions[i] - self.positions[j])
            for i, j in self.pairs
        )
        self.max_lag = int(np.ceil(max_d / self.c * self.fs)) + 2

    def estimate_angle(self, mic_signals: list) -> float:

        gcc_dict = {}
        for i, j in self.pairs:
            gcc_dict[(i, j)] = self._gcc_phat_windowed(
                mic_signals[i], mic_signals[j]
            )

        angles     = np.arange(0, 360, 1.0)
        srp_scores = np.zeros(len(angles))
        centre     = len(gcc_dict[self.pairs[0]]) // 2

        for a_idx, theta in enumerate(angles):
            theta_rad = np.radians(theta)
            doa   = np.array([np.cos(theta_rad), np.sin(theta_rad)])
            score = 0.0

            for i, j in self.pairs:
                cc = gcc_dict[(i, j)]

                delta_d  = np.dot(self.positions[j] - self.positions[i], doa)
                exp_lag  = delta_d / self.c * self.fs
                lag_idx  = int(round(exp_lag)) + centre
                lag_idx  = max(0, min(lag_idx, len(cc) - 1))
                score   += cc[lag_idx]

            srp_scores[a_idx] = score

        return float(angles[np.argmax(srp_scores)])

    def _gcc_phat_windowed(self,
                           sig_i: np.ndarray,
                           sig_j: np.ndarray) -> np.ndarray:

        N  = len(sig_i)
        N2 = 1
        while N2 < 2 * N:
            N2 <<= 1

        Xi    = np.fft.rfft(sig_i, n=N2)
        Xj    = np.fft.rfft(sig_j, n=N2)
        G     = Xi * np.conj(Xj)
        G_hat = G / (np.abs(G) + 1e-10)

        cc     = np.fft.irfft(G_hat, n=N2)
        cc     = np.concatenate([cc[N2 // 2:], cc[:N2 // 2]])

        centre = N2 // 2
        mask   = np.zeros(len(cc))
        lo     = max(0, centre - self.max_lag)
        hi     = min(len(cc), centre + self.max_lag + 1)
        mask[lo:hi] = 1.0
        cc    *= mask

        return cc

    def gcc_peak_lag(self, sig_i: np.ndarray, sig_j: np.ndarray) -> int:

        cc     = self._gcc_phat_windowed(sig_i, sig_j)
        centre = len(cc) // 2
        return int(np.argmax(cc)) - centre