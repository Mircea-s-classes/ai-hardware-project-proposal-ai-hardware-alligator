import os
import numpy as np
import librosa

# ==============================
# CONFIG
# ==============================
WAV_DIR = "wavs"
OUTPUT_FILE = "bird_samples.h"

TARGET_SR = 44100
WINDOW_SAMPLES = 44100          # 1 second
ENERGY_FRAME = 1024
HOP_LENGTH = 512
TOP_WINDOWS_PER_FILE = 3
MIN_SEPARATION = 8000           # minimum samples between windows (~0.5s)
ENERGY_THRESHOLD_RATIO = 0.3    # ignore low-energy regions

# ==============================
# UTILS
# ==============================
def extract_distinct_windows(audio, n_windows):
    # compute smoothed energy
    energy = np.convolve(
        np.abs(audio),
        np.ones(ENERGY_FRAME),
        mode="same"
    )

    max_energy = np.max(energy)
    threshold = ENERGY_THRESHOLD_RATIO * max_energy

    candidate_idxs = np.where(energy > threshold)[0]
    if len(candidate_idxs) == 0:
        return []

    windows = []
    used_centers = []

    for idx in candidate_idxs:
        if len(windows) >= n_windows:
            break

        # enforce separation
        if any(abs(idx - c) < MIN_SEPARATION for c in used_centers):
            continue

        start = max(0, idx - WINDOW_SAMPLES // 2)
        end = start + WINDOW_SAMPLES

        segment = audio[start:end]
        if len(segment) < WINDOW_SAMPLES:
            segment = np.pad(segment, (0, WINDOW_SAMPLES - len(segment)))

        windows.append(segment)
        used_centers.append(idx)

    return windows

# ==============================
# LOAD + PROCESS WAV FILES
# ==============================
clips = []

for fname in sorted(os.listdir(WAV_DIR)):
    if not fname.lower().endswith(".wav"):
        continue

    path = os.path.join(WAV_DIR, fname)
    print(f"Processing {fname}")

    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    audio = np.clip(audio, -1.0, 1.0)

    windows = extract_distinct_windows(audio, TOP_WINDOWS_PER_FILE)

    for i, w in enumerate(windows):
        w_i16 = (w * 32767).astype(np.int16)
        clips.append((f"{fname}_win{i}", w_i16))

if not clips:
    raise RuntimeError("No usable audio windows extracted")

# ==============================
# WRITE HEADER
# ==============================
with open(OUTPUT_FILE, "w") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n")
    f.write("#include <stddef.h>\n\n")

    f.write(f"#define NUM_AUDIO_CLIPS {len(clips)}\n")
    f.write(f"#define AUDIO_CLIP_LENGTH {WINDOW_SAMPLES}\n\n")

    f.write("static const char *audio_clip_names[] = {\n")
    for name, _ in clips:
        f.write(f'    "{name}",\n')
    f.write("};\n\n")

    f.write("static const int16_t audio_clips[NUM_AUDIO_CLIPS][AUDIO_CLIP_LENGTH] = {\n")
    for _, data in clips:
        f.write("  {\n")
        for sample in data:
            f.write(f"    {sample},\n")
        f.write("  },\n")
    f.write("};\n")

print(f"\nSUCCESS: Generated {OUTPUT_FILE} with {len(clips)} DISTINCT windows")

