import argparse
import gc
import os
from pathlib import Path
import tempfile

import torch
import torchaudio

from sinlib import Tokenizer
from vocos import Vocos

from utils import process_audio, uniquify
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import prepare_audio_input
from zipvoice.zipvoice import ZipVoice
import numpy as np
import soundfile as sf


def tokenize_text(tokenizer: Tokenizer, text: str) -> list[list[int]]:
    return [tokenizer(text).input_ids]



def run_inference(
    checkpoint_path: Path,
    prompt_audio: Path,
    prompt_text: str,
    target_text: str,
    output_path: Path = Path("output.wav"),
    speed: float = 1.0,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    n_mels = int(os.getenv("N_MELS", "100"))
    hop_length = int(os.getenv("HOP_LENGTH", "256"))
    n_fft = int(os.getenv("N_FFT", "1024"))
    sample_rate = int(os.getenv("SAMPLE_RATE", "24000"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

    prompt_text_tokens = tokenize_text(tokenizer, prompt_text)
    target_text_tokens = tokenize_text(tokenizer, target_text)

    model = ZipVoice(feat_dim=n_mels, vocab_size=tokenizer.vocab_size).to(device)
    model.eval()
    model, _, _ = load_checkpoint(model, None, str(checkpoint_path))
    model.to(device)

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocos.eval()
    
    mel_mean = -1.7977
    mel_std = 2.0155

    feature_mel_spec = (
        process_audio(
            prompt_audio,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            sample_rate=sample_rate,
            mel_mean=mel_mean,
            mel_std=mel_std,
        )
        .unsqueeze(0)
        .to(device)
    )
    prompt_features, prompt_feature_lens = prepare_audio_input(
        feature_mel_spec, device=device
    )

    with torch.no_grad():
        generated_mel, _, _, _ = model.sample(
            tokens=target_text_tokens,
            prompt_tokens=prompt_text_tokens,
            prompt_features=prompt_features,
            prompt_feature_lens=prompt_feature_lens,
            speed=speed,
            device=device,
        )

        # Match vocoder expected shape: [B, n_mels, T]
        generated_mel = generated_mel.permute(0, 2, 1)

        # Denormalize (reverse training normalization)
        generated_mel = generated_mel * mel_std + mel_mean

        # Exponentiate to get back to linear scale
        generated_mel = torch.exp(generated_mel)

        audio = vocos.decode(generated_mel).cpu()  # usually returns [B, T]

        # debug print
        peak = audio.abs().max().item()
        rms = audio.pow(2).mean().sqrt().item()
        print(f"peak={peak:.4f}, rms={rms:.4f}")
        if peak < 1e-4:
            print("Warning: decoded audio is silent. Check mel denormalization.")

        # Normalize audio to [-1, 1] to prevent int16 clipping
        audio = audio / (audio.abs().max() + 1e-8)

        # Vocos returns [B, T]; squeeze batch if B==1, otherwise handle properly
        audio_np = audio.squeeze(0).numpy()  # [T] for single sample
        
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        # Save
        # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(str(output_path), audio_np, sample_rate, subtype="PCM_16")
        # return tmp.name


def save_audio(input_path: str, output_path: Path) -> None:
    output_path = Path(uniquify(str(output_path)))
    waveform, sample_rate = torchaudio.load(input_path)
    print(f"Generated audio sample rate {sample_rate} Hz")
    torchaudio.save(output_path, waveform, sample_rate=sample_rate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sinhala TTS inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/content/drive/My Drive/sinhala-tts-checkpoints/checkpoint_step_7500.pt"
        ),
    )
    parser.add_argument(
        "--prompt-audio", type=Path, default=Path("/content/drive/My Drive/audio.wav")
    )
    parser.add_argument("--prompt-text", type=str, default="ඒක නිසා මම")
    parser.add_argument("--target-text", type=str, default="ඒක නිසා මම")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/content/drive/My Drive/generated_audio/output.wav"),
    )
    parser.add_argument("--speed", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    output_audio = run_inference(
        checkpoint_path=args.checkpoint,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
        target_text=args.target_text,
        speed=args.speed,
    )
    save_audio(output_audio, args.output)
