import numpy as np
from itertools import combinations


class TDOALocalizer:
    """
    Estimates the direction (and approximate range) of a sound source from
    multi-microphone recordings using GCC-PHAT time-delay estimation plus a
    Steered Response Power (SRP) search over candidate source positions.

    NOTE ON NEAR-FIELD VS. FAR-FIELD:
    A plane-wave (far-field) model assumes delay between two microphones
    depends only on the source's *direction*, not its distance. That
    assumption only holds when the source is much farther away than the
    array's own aperture (roughly >10x the array diagonal). This array's
    diagonal is ~2.8m, but the sources this project actually simulates sit
    2-6m away -- i.e. genuinely in the *near field*. Using a far-field model
    there was the root cause of large (sometimes >150 degree) angle errors:
    even at the true angle, the far-field-predicted delay didn't match the
    actual geometric delay closely enough to hit the right correlation peak.

    Fix: instead of scanning only over angle theta, we scan over candidate
    (angle, range) pairs and compute the *exact* Euclidean geometric delay
    from each candidate position to every microphone -- matching the same
    physics used to simulate the signals in microphone_array.py. This is
    the correct model for a room-scale array with nearby sources.
    """

    def __init__(self,
                 mic_positions: np.ndarray,
                 speed_sound:   float = 343.0,
                 sample_rate:   int   = 8000,
                 range_candidates: np.ndarray = None):

        self.positions = np.array(mic_positions, dtype=float)
        self.num_mics  = len(mic_positions)
        self.c         = speed_sound
        self.fs        = sample_rate
        self.pairs     = list(combinations(range(self.num_mics), 2))
        self.centre_xy = self.positions.mean(axis=0)

        # Candidate distances (metres) to search over when localizing a
        # near-field source. Covers typical room-scale placements.
        self.range_candidates = (
            range_candidates if range_candidates is not None
            else np.arange(0.5, 10.5, 0.5)
        )

        max_d = max(
            np.linalg.norm(self.positions[i] - self.positions[j])
            for i, j in self.pairs
        )
        self.max_lag = int(np.ceil(max_d / self.c * self.fs)) + 2

    def estimate_angle(self, mic_signals: list, return_range: bool = False):
        """
        Returns the estimated bearing (degrees, 0-360) of the source as
        seen from the array centroid. If return_range=True, also returns
        the estimated distance (metres) from the same search.
        """

        gcc_dict = {}
        for i, j in self.pairs:
            gcc_dict[(i, j)] = self._gcc_phat_windowed(
                mic_signals[i], mic_signals[j]
            )
        cc_centre = len(gcc_dict[self.pairs[0]]) // 2

        angles = np.arange(0, 360, 1.0)
        theta_grid, r_grid = np.meshgrid(
            np.radians(angles), self.range_candidates, indexing="ij"
        )  # shape: (num_angles, num_ranges)

        cand_x = self.centre_xy[0] + r_grid * np.cos(theta_grid)
        cand_y = self.centre_xy[1] + r_grid * np.sin(theta_grid)

        scores = np.zeros_like(cand_x)

        for i, j in self.pairs:
            cc = gcc_dict[(i, j)]

            dist_i = np.hypot(cand_x - self.positions[i, 0],
                               cand_y - self.positions[i, 1])
            dist_j = np.hypot(cand_x - self.positions[j, 0],
                               cand_y - self.positions[j, 1])

            exp_lag = (dist_i - dist_j) / self.c * self.fs
            lag_idx = np.round(exp_lag).astype(int) + cc_centre
            lag_idx = np.clip(lag_idx, 0, len(cc) - 1)

            scores += cc[lag_idx]

        best_flat = np.argmax(scores)
        best_a_idx, best_r_idx = np.unravel_index(best_flat, scores.shape)

        best_angle = float(angles[best_a_idx])
        best_range = float(self.range_candidates[best_r_idx])

        if return_range:
            return best_angle, best_range
        return best_angle

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
