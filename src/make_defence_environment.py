import torch
import torchaudio
import math
from pathlib import Path
import soundfile as sf


# ============================================================
#                 USER SETTINGS
# ============================================================

# Clean speech
CLEAN_SPEECH = "voice3.wav"


# ------------------------------------------------------------
# SELECT ONE OR MULTIPLE DEFENCE NOISES
# ------------------------------------------------------------

NOISE_TYPES = [
    "helicopter",
]


# Examples:
#
# NOISE_TYPES = ["gunshot"]
#
# NOISE_TYPES = ["gunshot", "helicopter"]
#
# NOISE_TYPES = [
#     "gunshot",
#     "helicopter",
#     "vehicle",
# ]
#
# Heavy battlefield environment:
#
# NOISE_TYPES = [
#     "gunshot",
#     "shelling",
#     "vehicle",
#     "helicopter",
#     "fighter",
# ]


# Overall mixture SNR
TARGET_SNR_DB = 0


# Output file
OUTPUT_FILE = "h_ep_12_noise.wav"


# ============================================================
#                 DEFENCE NOISE DATABASE
# ============================================================

NOISE_DATABASE = {

    "communication":
        "defence_dataset/MAD_dataset/test/208/1.wav",

    "footsteps":
        "defence_dataset/MAD_dataset/test/148/0.wav",

    "gunshot":
        "defence_dataset/MAD_dataset/test/208/0.wav",

    "shelling":
        "defence_dataset/MAD_dataset/test/148/24.wav",

    "vehicle":
        "defence_dataset/MAD_dataset/test/022/0.wav",

    "helicopter":
        "defence_dataset/MAD_dataset/test/376/3.wav",

    "fighter":
        "defence_dataset/MAD_dataset/test/227/0.wav",
}


# ============================================================
#                 SETTINGS
# ============================================================

SAMPLE_RATE = 44100


# ============================================================
#                 CHECK CLEAN SPEECH
# ============================================================

if not Path(CLEAN_SPEECH).exists():

    raise FileNotFoundError(
        "\nClean speech not found:\n"
        + CLEAN_SPEECH
    )


# ============================================================
#                 CHECK NOISE TYPES
# ============================================================

for noise_type in NOISE_TYPES:

    if noise_type not in NOISE_DATABASE:

        print("\nERROR: Unknown noise type:")
        print(" ", noise_type)

        print("\nAvailable noise types:")

        for name in NOISE_DATABASE:

            print("  -", name)

        raise ValueError(
            "\nSelected noise: " + noise_type
        )


# ============================================================
#                 DISPLAY
# ============================================================

print("\n==========================================")
print("       DEFENCE ENVIRONMENT GENERATOR")
print("==========================================")

print("\nClean speech:")
print(" ", CLEAN_SPEECH)

print("\nDefence noises:")

for noise_type in NOISE_TYPES:
    print("  -", noise_type)

print("\nTarget overall SNR:")
print(" ", TARGET_SNR_DB, "dB")


# ============================================================
#                 LOAD SPEECH
# ============================================================

speech_np, speech_sr = sf.read(CLEAN_SPEECH, dtype="float32")
speech = torch.from_numpy(speech_np)

if speech.ndim == 1:
    speech = speech.unsqueeze(0)
else:
    speech = speech.mean(dim=1).unsqueeze(0)

print("\nSpeech sample rate:", speech_sr)


# ============================================================
#                 CONVERT SPEECH TO MONO
# ============================================================

if speech.shape[0] > 1:

    speech = speech.mean(
        dim=0,
        keepdim=True
    )


# ============================================================
#                 RESAMPLE SPEECH
# ============================================================

if speech_sr != SAMPLE_RATE:

    speech = torchaudio.functional.resample(
        speech,
        speech_sr,
        SAMPLE_RATE
    )


# ============================================================
#                 SPEECH LENGTH
# ============================================================

speech_length = speech.shape[1]


# ============================================================
#                 NORMALIZE SPEECH
# ============================================================

speech_peak = speech.abs().max()

if speech_peak > 1e-8:

    speech = speech / speech_peak


# ============================================================
#                 CREATE COMBINED NOISE
# ============================================================

combined_noise = torch.zeros_like(speech)


for noise_type in NOISE_TYPES:

    noise_file = NOISE_DATABASE[noise_type]

    print("\nLoading:", noise_type)
    print("File:", noise_file)

    if not Path(noise_file).exists():
        raise FileNotFoundError(
            "\nNoise file not found:\n"
            + noise_file
        )

    noise_np, noise_sr = sf.read(
        noise_file,
        dtype="float32"
    )

    noise = torch.from_numpy(noise_np)

    if noise.ndim == 1:
        noise = noise.unsqueeze(0)
    else:
        noise = noise.mean(dim=1).unsqueeze(0)

    print("Sample rate:", noise_sr)

    # --------------------------------------------------------
    # Convert to mono
    # --------------------------------------------------------

    if noise.shape[0] > 1:

        noise = noise.mean(
            dim=0,
            keepdim=True
        )

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    if noise_sr != SAMPLE_RATE:

        noise = torchaudio.functional.resample(
            noise,
            noise_sr,
            SAMPLE_RATE
        )

    # --------------------------------------------------------
    # Match speech length
    # --------------------------------------------------------

    noise_length = noise.shape[1]

    if noise_length < speech_length:

        repeat_count = math.ceil(
            speech_length / noise_length
        )

        noise = noise.repeat(
            1,
            repeat_count
        )

        noise = noise[:, :speech_length]

    else:

        noise = noise[:, :speech_length]

    # --------------------------------------------------------
    # Normalize individual noise
    # --------------------------------------------------------

    noise_peak = noise.abs().max()

    if noise_peak > 1e-8:

        noise = noise / noise_peak

    # --------------------------------------------------------
    # Add to combined noise
    # --------------------------------------------------------

    combined_noise = combined_noise + noise


# ============================================================
#       NORMALIZE COMBINED NOISE
# ============================================================

combined_noise_peak = combined_noise.abs().max()

if combined_noise_peak > 1e-8:

    combined_noise = (
        combined_noise /
        combined_noise_peak
    )


# ============================================================
#                 CALCULATE POWER
# ============================================================

speech_power = speech.pow(2).mean()

noise_power = combined_noise.pow(2).mean()


# ============================================================
#                 TARGET NOISE POWER
# ============================================================

target_noise_power = speech_power / (
    10 ** (TARGET_SNR_DB / 10)
)


# ============================================================
#                 SCALE NOISE
# ============================================================

scale = torch.sqrt(
    target_noise_power /
    (noise_power + 1e-12)
)


scaled_noise = combined_noise * scale


# ============================================================
#                 CREATE MIXTURE
# ============================================================

mixture = speech + scaled_noise


# ============================================================
#                 PREVENT CLIPPING
# ============================================================

peak = mixture.abs().max()

if peak > 0.99:

    mixture = (
        mixture / peak
    ) * 0.99


# ============================================================
#                 SAVE
# ============================================================

sf.write(
    OUTPUT_FILE,
    mixture.squeeze(0).numpy(),
    SAMPLE_RATE
)

# ============================================================
#                 FINISHED
# ============================================================

duration = mixture.shape[1] / SAMPLE_RATE


print("\n==========================================")
print("       DEFENCE ENVIRONMENT CREATED")
print("==========================================")

print("\nSpeech:")
print(" ", CLEAN_SPEECH)

print("\nDefence noises:")

for noise_type in NOISE_TYPES:
    print("  -", noise_type)

print("\nNumber of simultaneous noises:")
print(" ", len(NOISE_TYPES))

print("\nOverall SNR:")
print(" ", TARGET_SNR_DB, "dB")

print("\nOutput:")
print(" ", OUTPUT_FILE)

print("\nDuration:")
print(" ", round(duration, 2), "seconds")

print("\nSample rate:")
print(" ", SAMPLE_RATE)

print("\n==========================================")
print("       ENVIRONMENT READY")
print("==========================================")

print("\nNext command:")

enhanced_output = (
    Path(OUTPUT_FILE).stem +
    "_enhanced.wav"
)

print(
    "python demo.py "
    + OUTPUT_FILE
    + " "
    + enhanced_output
)

print("\n==========================================")
