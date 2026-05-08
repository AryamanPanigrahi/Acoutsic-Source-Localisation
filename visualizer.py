import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


COLOURS = {
    "speech" : "#4CAF50",
    "clap"   : "#F44336",
    "noise"  : "#9E9E9E",
    "whistle": "#2196F3",
    "mic"    : "#1A237E",
    "true"   : "#00BCD4",
    "est"    : "#FF5722",
}


class Visualizer:

    def __init__(self, mic_positions: np.ndarray):

        self.mic_positions = np.array(mic_positions)
        plt.style.use("dark_background")

    def plot_all(self, results: list):

        self._plot_array_layout(results)
        #self._plot_waveforms(results[0])
        self._plot_spectrum(results[0])
        self._plot_error_summary(results)
        #self._plot_classification(results)

    def _plot_array_layout(self, results: list):
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")

        for k, pos in enumerate(self.mic_positions):
            ax.scatter(*pos, s=200, color=COLOURS["mic"],
                       zorder=5, marker="^")
            ax.annotate(f"Mic {k}", xy=pos,
                        xytext=(pos[0] + 0.01, pos[1] + 0.015),
                        color="white", fontsize=9)

        for r in results:
            sx, sy = r["source"]
            true_angle = np.radians(r["true_angle"])
            est_angle  = np.radians(r["estimated_angle"])
            col = COLOURS.get(r["sound_type"], "white")

            ax.scatter(sx, sy, s=150, color=col, zorder=4, marker="*",
                       label=r["scenario"])
            ax.annotate(r["sound_type"],
                        xy=(sx, sy),
                        xytext=(sx + 0.05, sy + 0.05),
                        color=col, fontsize=8)

            L = 0.25
            ax.annotate("",
                xy=(L * np.cos(true_angle), L * np.sin(true_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->",
                                color=COLOURS["true"], lw=2))

            ax.annotate("",
                xy=(L * np.cos(est_angle), L * np.sin(est_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->",
                                color=COLOURS["est"], lw=2,
                                linestyle="dashed"))

        patches = [
            mpatches.Patch(color=COLOURS["true"], label="True direction"),
            mpatches.Patch(color=COLOURS["est"],  label="Estimated direction"),
            mpatches.Patch(color=COLOURS["mic"],  label="Microphone"),
        ]
        ax.legend(handles=patches, loc="upper right",
                  facecolor="#1C1C1C", edgecolor="grey", labelcolor="white")

        ax.set_title("Microphone Array Layout & Source Directions",
                     color="white", fontsize=13, pad=12)
        ax.set_xlabel("x (metres)", color="white")
        ax.set_ylabel("y (metres)", color="white")
        ax.tick_params(colors="white")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(color="#2C2C2C", linestyle="--", linewidth=0.5)
        ax.set_aspect("equal")
        plt.tight_layout()

    def _plot_waveforms(self, result: dict):

        signal = result["signal"]
        N      = len(signal)
        t      = np.linspace(0, 0.5, N)

        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle(f"Microphone Waveforms – {result['scenario']}",
                     color="white", fontsize=13)

        colours = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]
        for i, (ax, col) in enumerate(zip(axes, colours)):
            ax.set_facecolor("#0D1117")
            ax.plot(t[:2205], signal[:2205], color=col, linewidth=0.8)
            ax.set_ylabel(f"Mic {i}", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)
            ax.grid(color="#2C2C2C", linestyle="--", linewidth=0.4)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

        axes[-1].set_xlabel("Time (s)", color="white")
        plt.tight_layout()

    def _plot_spectrum(self, result: dict):
        freqs    = result["freqs"]
        spectrum = result["spectrum"]

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")

        mask = freqs <= 10000
        ax.fill_between(freqs[mask], spectrum[mask],
                        spectrum[mask].min() - 5,
                        alpha=0.4,
                        color=COLOURS.get(result["sound_type"], "cyan"))
        ax.plot(freqs[mask], spectrum[mask],
                color=COLOURS.get(result["sound_type"], "cyan"),
                linewidth=1.2)

        ax.set_title(f"FFT Magnitude Spectrum – {result['scenario']}",
                     color="white", fontsize=13)
        ax.set_xlabel("Frequency (Hz)", color="white")
        ax.set_ylabel("Magnitude (dB)", color="white")
        ax.tick_params(colors="white")
        ax.grid(color="#2C2C2C", linestyle="--", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        plt.tight_layout()

    def _plot_error_summary(self, results: list):
        labels = [r["scenario"].split("@")[0].strip() for r in results]
        true_a = [r["true_angle"]      for r in results]
        est_a  = [r["estimated_angle"] for r in results]
        errors = [r["error"]           for r in results]

        x      = np.arange(len(labels))
        width  = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0D1117")

        ax1.set_facecolor("#0D1117")
        bars1 = ax1.bar(x - width/2, true_a, width,
                        label="True angle", color=COLOURS["true"], alpha=0.85)
        bars2 = ax1.bar(x + width/2, est_a,  width,
                        label="Estimated angle", color=COLOURS["est"], alpha=0.85)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, color="white", fontsize=9, rotation=10)
        ax1.set_ylabel("Angle (°)", color="white")
        ax1.set_title("True vs Estimated Direction", color="white", fontsize=12)
        ax1.legend(facecolor="#1C1C1C", edgecolor="grey", labelcolor="white")
        ax1.tick_params(colors="white")
        ax1.grid(axis="y", color="#2C2C2C", linestyle="--", linewidth=0.5)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#333333")

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1,
                     f"{h:.0f}°", ha="center", color="white", fontsize=8)

        ax2.set_facecolor("#0D1117")
        bar_cols = [COLOURS.get(r["sound_type"], "white") for r in results]
        ax2.bar(labels, errors, color=bar_cols, alpha=0.85)
        ax2.set_ylabel("Angular Error (°)", color="white")
        ax2.set_title("TDOA Estimation Error per Scenario", color="white", fontsize=12)
        ax2.tick_params(colors="white")
        ax2.set_xticklabels(labels, color="white", fontsize=9, rotation=10)
        ax2.grid(axis="y", color="#2C2C2C", linestyle="--", linewidth=0.5)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333333")
        for i, (err, lbl) in enumerate(zip(errors, labels)):
            ax2.text(i, err + 0.5, f"{err:.1f}°",
                     ha="center", color="white", fontsize=9)

        plt.tight_layout()

    def _plot_classification(self, results: list):
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")

        labels  = [r["scenario"].split("@")[0].strip() for r in results]
        confs   = [r["confidence"] * 100 for r in results]
        correct = [r["sound_type"] == r["predicted_class"] for r in results]
        colours = [COLOURS["speech"] if ok else COLOURS["clap"] for ok in correct]

        bars = ax.barh(labels, confs, color=colours, alpha=0.85)
        ax.axvline(70, color="white", linestyle="--", linewidth=1, alpha=0.5,
                   label="70% threshold")

        for bar, r, conf in zip(bars, results, confs):
            status = "✓" if r["sound_type"] == r["predicted_class"] else "✗"
            ax.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{conf:.1f}%  {r['predicted_class']} {status}",
                    va="center", color="white", fontsize=9)

        ax.set_xlim(0, 115)
        ax.set_xlabel("Confidence (%)", color="white")
        ax.set_title("Sound Classification Confidence", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1C1C1C", edgecolor="grey", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        plt.tight_layout()