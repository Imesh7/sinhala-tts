from pathlib import Path

import librosa
from sinlib import Tokenizer
from vocos import Vocos

from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import prepare_audio_input
from zipvoice.zipvoice import ZipVoice
import torch
import torchaudio.transforms as T


def inference():
    prompt_text = "ඉන්දියාවේ ගුවන් සේවා සමාගම්වලට සහනයක්"
    prompt_voice_file_path = ""
    target_text = "නයක්"
    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")
    # tokenize the text

    prompt_text_tokens = tokenizer(prompt_text, return_tensors="pt")
    target_text_tokens = tokenizer(target_text, return_tensors="pt")
    
    # load model from local checkpoint
    model = ZipVoice()
    model.eval()
    
    last_checkpoint = 500
    checkpoint_dir = Path("/content/drive/MyDrive/sinhala-tts-checkpoints")
    checkpoint_file_path = checkpoint_dir / f"checkpoint_step{last_checkpoint}.pth"
    model , _, _ = load_checkpoint(model, None, checkpoint_file_path)

    # vocoder
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocos.eval()
    
    feature_mel_spec = process_audio(prompt_voice_file_path)
    prompt_features, prompt_feature_lens = prepare_audio_input(feature_mel_spec)
    
    x1_wo_prompt, x1_wo_prompt_lens, x1_prompt, prompt_feature_lens = model.sample(
        tokens=target_text_tokens,
        prompt_tokens=prompt_text_tokens,
        prompt_features=prompt_features,
        prompt_feature_lens=prompt_feature_lens,
        speed=1.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    pred_features = pred_features.permute(0, 2, 1)
    
    # wav = vocos.decode(pred_features).squeeze(1).clamp(-1, 1)
    vocos.decode(x1_wo_prompt.cpu(), "output_with_prompt.wav")


def process_audio(file_path):
    # Load the audio file
    waveform, sample_rate = librosa.load(file_path)

    # Resample if necessary
    target_sample_rate = 22050
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    mel_spectrogram = T.MelSpectrogram(sample_rate=target_sample_rate, n_mels=100)
    mel_spec = mel_spectrogram(waveform)
    return mel_spec

if __name__ == "__main__":
    inference()
    
    
