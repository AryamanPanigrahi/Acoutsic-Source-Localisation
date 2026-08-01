import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from signal_generator import SignalGenerator
from microphone_array  import MicrophoneArray
from signal_processing import SignalProcessor
from tdoa_localization import TDOALocalizer
from visualizer        import Visualizer

SAMPLE_RATE   = 8000
DURATION      = 1.0
SPEED_SOUND   = 343.0

MIC_POSITIONS = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [0.0, 2.0],
    [1.5, 1.0],
])

# Scenarios are defined by (name, true bearing in degrees, distance in
# metres, sound type). Source coordinates are DERIVED from the angle so the
# label always matches reality -- previously the source (x,y) values were
# hand-picked and didn't actually correspond to the angles in their labels.
SCENARIO_DEFS = [
    ("Speech",  45,  3.0, "speech"),
    ("Clap",    90,  2.5, "clap"),
    ("Noise",  180,  2.0, "noise"),
    ("Whistle", 315, 4.0, "whistle"),
]

TEST_SCENARIOS = []
for _name, _angle_deg, _dist, _sound in SCENARIO_DEFS:
    _rad  = np.radians(_angle_deg)
    _src_x = round(_dist * np.cos(_rad), 3)
    _src_y = round(_dist * np.sin(_rad), 3)
    TEST_SCENARIOS.append({
        "label":  f"{_name}  @  {_angle_deg} deg",
        "source": (_src_x, _src_y),
        "sound":  _sound,
    })

def angle_from_position(x, y):
    return np.degrees(np.arctan2(y, x)) % 360

def print_banner():
    print("=" * 63)
    print("  AI-Based Acoustic Source Localization & Classification")
    print("  Embedded Systems Software Simulation")
    print("=" * 63)

def run_simulation():
    print_banner()

    sig_gen    = SignalGenerator(sample_rate=SAMPLE_RATE, duration=DURATION)
    mic_array  = MicrophoneArray(MIC_POSITIONS, SPEED_SOUND, SAMPLE_RATE)
    processor  = SignalProcessor(SAMPLE_RATE)
    localizer  = TDOALocalizer(MIC_POSITIONS, SPEED_SOUND, SAMPLE_RATE)
    viz        = Visualizer(MIC_POSITIONS)

    print("\n[STEP 1]  Training sound classifier ...")
    print("          Classifier ready.\n")

    results = []

    for idx, scenario in enumerate(TEST_SCENARIOS):
        src_x, src_y = scenario["source"]
        sound_type   = scenario["sound"]
        true_angle   = angle_from_position(src_x, src_y)

        print(f"--- Scenario {idx+1}: {scenario['label']} ---")
        print(f"    Source position  : ({src_x:+.1f}, {src_y:+.1f}) m")
        print(f"    True angle       : {true_angle:.1f} deg")

        raw_signal = sig_gen.generate(sound_type)
        mic_signals = mic_array.simulate(raw_signal, src_x, src_y)
        filtered = [processor.bandpass_filter(s) for s in mic_signals]

        freqs, spectrum = processor.compute_fft(filtered[0])

        estimated_angle = localizer.estimate_angle(filtered)
        error = abs(estimated_angle - true_angle)
        if error > 180:
            error = 360 - error

        print(f"    Estimated angle  : {estimated_angle:.1f} deg  (error {error:.1f} deg)")

        results.append({
            "scenario"        : scenario["label"],
            "source"          : (src_x, src_y),
            "true_angle"      : true_angle,
            "estimated_angle" : estimated_angle,
            "error"           : error,
            "sound_type"      : sound_type,
            "signal"          : filtered[0],
            "freqs"           : freqs,
            "spectrum"        : spectrum,
        })

    print("=" * 63)
    print(f"  {'Scenario':<22} {'True':>7} {'Est.':>7} {'Err':>7}")
    print("-" * 63)

    for r in results:
        label = r['scenario'].split('@')[0].strip()
        print(f"  {label:<22} {r['true_angle']:>6.1f}d "
              f"{r['estimated_angle']:>6.1f}d {r['error']:>6.1f}d")

    mean_err = np.mean([r['error'] for r in results])
    print("-" * 63)
    print(f"  Mean angle error: {mean_err:.1f} degrees")
    print("=" * 63)

    print("\n[STEP 2]  Generating visualisations ...")
    viz.plot_all(results)
    print("          All plots displayed. Close windows to exit.\n")
    plt.show()

if __name__ == "__main__":
    run_simulation()