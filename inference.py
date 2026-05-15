import argparse
import gc
from pathlib import Path
import tempfile

import librosa
import torch
import torchaudio
import torchaudio.transforms as T
from sinlib import Tokenizer
from vocos import Vocos

from utils import uniquify
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import prepare_audio_input
from zipvoice.zipvoice import ZipVoice
import soundfile as sf

SAMPLE_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
TOP_DB = 80.0
VOCOS_SAMPLE_RATE = 24000
N_MELS = 100


def tokenize_text(tokenizer: Tokenizer, text: str) -> list[list[int]]:
    return [tokenizer(text).input_ids]


def process_audio(file_path: Path, n_mels: int) -> torch.Tensor:
    waveform_np, sample_rate = librosa.load(file_path, sr=None)
    waveform = torch.from_numpy(waveform_np).float()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        power=2.0,
        normalized=True,
    )
    
    # 3. Compute mel-spectrogram
    mel_spec = mel_transform(waveform)  # [1, n_mels, time]
    
    mel_log = torch.log(torch.clamp(mel_spec, min=1e-7))  # [1, n_mels, time]
    mel_log = mel_log.squeeze(0)  # [n_mels, time]
    return mel_log


def normalize_audio_for_save(audio: torch.Tensor) -> torch.Tensor:
    peak = audio.abs().max()
    if torch.isclose(peak, torch.tensor(0.0, device=audio.device)):
        return audio
    return 0.95 * (audio / peak)


def run_inference(
    checkpoint_path: Path,
    prompt_audio: Path,
    prompt_text: str,
    target_text: str,
    speed: float = 1.0,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

    prompt_text_tokens = tokenize_text(tokenizer, prompt_text)
    target_text_tokens = tokenize_text(tokenizer, target_text)

    model = ZipVoice(feat_dim=N_MELS).to(device)
    model.eval()
    model, _, _ = load_checkpoint(model, None, str(checkpoint_path))
    model.to(device)

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocos.eval()

    feature_mel_spec = process_audio(prompt_audio, n_mels=N_MELS).unsqueeze(0).to(device)
    prompt_features, prompt_feature_lens = prepare_audio_input(feature_mel_spec, device=device)

    with torch.no_grad():
        generated_mel, _, _, _ = model.sample(
            tokens=target_text_tokens,
            prompt_tokens=prompt_text_tokens,
            prompt_features=prompt_features,
            prompt_feature_lens=prompt_feature_lens,
            speed=speed,
            device=device,
        )

        generated_mel = generated_mel.permute(0, 2, 1)
        audio = vocos.decode(generated_mel).cpu()

        peak = audio.abs().max().item()
        rms = audio.pow(2).mean().sqrt().item()
        print(f"Decoded audio stats: peak={peak:.6f}, rms={rms:.6f}")
        if peak < 1e-4:
            print(
                "Warning: decoded waveform is almost silent. This usually means the "
                "TTS model output and vocoder features do not match."
            )
        audio = normalize_audio_for_save(audio)
        audio_np = audio.squeeze(1).detach().cpu().numpy()
        
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        # Write to temp WAV file (Gradio serves this directly)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_np, 24000, subtype="PCM_16")
            return tmp.name  # <-- Gradio accepts str path

def save_audio(audio: torch.Tensor, sample_rate: int, output_path: Path) -> None:
    output_path = Path(uniquify(str(output_path)))
    torchaudio.save(output_path, audio.squeeze(1), sample_rate=sample_rate)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sinhala TTS inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/content/drive/My Drive/sinhala-tts-checkpoints/checkpoint_step_7500.pt"),
    )
    parser.add_argument("--prompt-audio", type=Path, default=Path("/content/drive/My Drive/audio.wav"))
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
    args = parser.parse_args(args=[])
    output_audio = run_inference(
        checkpoint_path=args.checkpoint,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
        target_text=args.target_text,
        speed=args.speed,
    )
    save_audio(output_audio, VOCOS_SAMPLE_RATE, args.output)
