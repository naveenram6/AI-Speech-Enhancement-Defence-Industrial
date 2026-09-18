import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import argparse


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 44100

# EPOCH 12 CHECKPOINT
CHECKPOINT = Path(
    "experiments/snr10/12.pt"
)

MODEL_PARAMS = {
    "label_len": 41,
    "L": 32,
    "enc_dim": 256,
    "num_enc_layers": 10,
    "dec_dim": 128,
    "num_dec_layers": 1,
    "dec_buf_len": 13,
    "dec_chunk_size": 13,
    "out_buf_len": 4,
}

# Speech = label 0
SPEECH_LABEL_INDEX = 0


# ============================================================
# IMPORT MODEL
# ============================================================

from src.training.dcc_tf import Net


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("======================================")
    print("DEFENCE SPEECH ENHANCEMENT")
    print("======================================")
    print()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Checkpoint:", CHECKPOINT)

    # ---------------------------------------------------------
    # Create exact training architecture
    # ---------------------------------------------------------

    model = Net(
        label_len=41,
        L=32,
        enc_dim=256,
        num_enc_layers=10,
        dec_dim=128,
        num_dec_layers=1,
        dec_buf_len=13,
        dec_chunk_size=13,
        out_buf_len=4
    )

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    print()
    print("Checkpoint loaded.")

    # ---------------------------------------------------------
    # Get model weights
    # ---------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # ---------------------------------------------------------
    # Remove DataParallel prefix if present
    # ---------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    # ---------------------------------------------------------
    # Load model weights
    # ---------------------------------------------------------

    model.load_state_dict(cleaned_state_dict)

    model.to(device)
    model.eval()

    print("Model weights loaded successfully.")
    print("Model is ready for inference.")
    print()

    return model, device


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(filename):

    print("Input audio:")
    print(filename)

    # Use SoundFile instead of torchaudio.load
    audio_np, original_sr = sf.read(
        filename,
        dtype="float32"
    )

    audio = torch.from_numpy(audio_np)

    # ---------------------------------------------------------
    # Convert to mono
    # ---------------------------------------------------------

    if audio.ndim == 1:

        audio = audio.unsqueeze(0)

    else:

        audio = audio.mean(
            dim=1,
            keepdim=True
        ).transpose(0, 1)

    print("Original sample rate:", original_sr)
    print("Original shape:", audio.shape)

    # ---------------------------------------------------------
    # Resample to 44.1 kHz
    # ---------------------------------------------------------

    if original_sr != SAMPLE_RATE:

        audio = torchaudio.functional.resample(
            audio,
            orig_freq=original_sr,
            new_freq=SAMPLE_RATE
        )

    print("Model sample rate:", SAMPLE_RATE)
    print("Model input shape:", audio.shape)

    return audio, original_sr


# ============================================================
# ENHANCE AUDIO
# ============================================================

def enhance(model, device, audio):

    # ---------------------------------------------------------
    # Normalize input
    # ---------------------------------------------------------

    peak = audio.abs().max()

    if peak > 1e-8:

        audio = audio / peak

    # ---------------------------------------------------------
    # Create Speech query
    # ---------------------------------------------------------

    query = torch.zeros(
        1,
        MODEL_PARAMS["label_len"]
    )

    query[0, SPEECH_LABEL_INDEX] = 1.0

    # ---------------------------------------------------------
    # Add batch dimension
    # ---------------------------------------------------------

    audio = audio.unsqueeze(0)

    query = query.to(device)
    audio = audio.to(device)

    print()
    print("Running DefenceWaveformer...")

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    with torch.inference_mode():

        output = model(
            audio,
            query
        )

    output = output.squeeze(0).cpu()

    # ---------------------------------------------------------
    # Prevent clipping
    # ---------------------------------------------------------

    peak = output.abs().max()

    if peak > 0.99:

        output = (
            output / peak
        ) * 0.99

    print("Enhancement completed.")

    return output


# ============================================================
# SAVE AUDIO
# ============================================================

def save_audio(filename, audio):

    # SoundFile avoids TorchCodec problem

    audio_np = audio.squeeze(0).numpy()

    sf.write(
        filename,
        audio_np,
        SAMPLE_RATE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Defence Speech Enhancement Demo"
    )

    parser.add_argument(
        "input",
        help="Input noisy speech WAV file"
    )

    parser.add_argument(
        "output",
        nargs="?",
        default="enhanced_output.wav",
        help="Output enhanced WAV file"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Check input
    # ---------------------------------------------------------

    if not Path(args.input).exists():

        print()
        print("ERROR:")
        print("Input file not found:")
        print(args.input)

        return

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model, device = load_model()

    # ---------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------

    audio, original_sr = load_audio(
        args.input
    )

    # ---------------------------------------------------------
    # Enhance
    # ---------------------------------------------------------

    enhanced = enhance(
        model,
        device,
        audio
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    save_audio(
        args.output,
        enhanced
    )

    # ---------------------------------------------------------
    # Final information
    # ---------------------------------------------------------

    print()
    print("======================================")
    print("OUTPUT READY")
    print("======================================")

    print()
    print("Input:")
    print(" ", args.input)

    print()
    print("Enhanced output:")
    print(" ", args.output)

    print()
    print("Model:")
    print(" DefenceWaveformer")

    print()
    print("Checkpoint:")
    print(" experiments/snr10/12.pt")

    print()
    print("Device:")
    print(" ", device)

    print()
    print("======================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
