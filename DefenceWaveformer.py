import argparse
import os

import torch
import torchaudio

from src.training.dcc_tf import Net as Waveformer


# Our custom training used label index 0 for Speech.
LABEL_LEN = 41
SPEECH_LABEL_INDEX = 0

# These are the exact parameters from:
# dcc_tf_ckpt_E256_10_D128_1
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


def main():
    parser = argparse.ArgumentParser(
        description="Run trained Defence Waveformer on an input WAV file."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input WAV file"
    )

    parser.add_argument(
        "output",
        type=str,
        help="Path to output enhanced WAV file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/snr10/12.pt",
        help="Path to trained Waveformer checkpoint"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("Loading checkpoint:", args.checkpoint)

    # ---------------------------------------------------------
    # Create model with EXACT training architecture
    # ---------------------------------------------------------
    model = Waveformer(**MODEL_PARAMS)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # Load input audio
    # ---------------------------------------------------------
    mixture, original_sr = torchaudio.load(args.input)

    print("Input:", args.input)
    print("Original sample rate:", original_sr)
    print("Input shape:", mixture.shape)

    # Waveformer expects 44.1 kHz
    if original_sr != 44100:
        mixture = torchaudio.functional.resample(
            mixture,
            orig_freq=original_sr,
            new_freq=44100
        )

    # Model expects:
    # [batch, channels, samples]
    mixture = mixture.unsqueeze(0).to(device)

    # ---------------------------------------------------------
    # Construct Speech query
    # ---------------------------------------------------------
    query = torch.zeros(
        1,
        LABEL_LEN,
        device=device
    )

    query[0, SPEECH_LABEL_INDEX] = 1.0

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    print("Running Waveformer...")

    with torch.inference_mode():
        output = model(
            mixture,
            query
        )

    output = output.squeeze(0).cpu()

    # ---------------------------------------------------------
    # Convert output back to original sample rate
    # ---------------------------------------------------------
    if original_sr != 44100:
        output = torchaudio.functional.resample(
            output,
            orig_freq=44100,
            new_freq=original_sr
        )

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    if os.path.exists(args.output):
        os.remove(args.output)

    torchaudio.save(
        args.output,
        output,
        original_sr
    )

    print("Inference completed.")
    print("Enhanced audio saved to:", args.output)


if __name__ == "__main__":
    main()