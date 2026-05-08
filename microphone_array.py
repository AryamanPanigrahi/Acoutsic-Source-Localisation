import numpy as np

class MicrophoneArray:

    NOISE_STD = 0.001

    def __init__(self,
                 positions:   np.ndarray,
                 speed_sound: float = 343.0,
                 sample_rate: int   = 8000):

        self.positions   = np.array(positions, dtype=float)
        self.num_mics    = len(positions)
        self.speed_sound = speed_sound
        self.sample_rate = sample_rate

    def simulate(self,
                 signal: np.ndarray,
                 src_x:  float,
                 src_y:  float) -> list:

        src        = np.array([src_x, src_y])
        recordings = []

        for mic_pos in self.positions:

            distance = np.linalg.norm(src - mic_pos)
            distance = max(distance, 0.01)

            time_of_flight = distance / self.speed_sound
            sample_delay   = int(round(time_of_flight * self.sample_rate))

            attenuation = 1.0 / distance

            delayed = np.zeros(len(signal))
            if sample_delay < len(signal):
                delayed[sample_delay:] = signal[:len(signal) - sample_delay]

            noisy = (delayed * attenuation
                     + self.NOISE_STD * np.random.randn(len(signal)))
            recordings.append(noisy)

        return recordings

    def compute_delays(self, src_x: float, src_y: float) -> np.ndarray:

        src = np.array([src_x, src_y])
        return np.array([
            np.linalg.norm(src - mic) / self.speed_sound
            for mic in self.positions
        ])