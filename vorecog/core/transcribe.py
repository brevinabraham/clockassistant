import torch
import numpy as np
import sys
from vorecog.configs.config import *

device = "cuda" if torch.cuda.is_available() else "cpu"


if USE_FASTER_WHISPER:
    from faster_whisper import WhisperModel
    model_path = MODELS_DIR / "faster-whisper-medium"
    model = WhisperModel(model_size_or_path=str(model_path), device=device, compute_type="float16")
else:
    model_path = MODELS_DIR / "whisper-medium"
    sys.path.append(str(model_path.resolve()))
    import whisper
    model = whisper.load_model("medium", download_root=str(model_path))

print(f"🧠 Voice encoder using: {device}, Model used: {type(model).__name__}")

def transcribe_audio(audio_np):
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=0)

    if USE_FASTER_WHISPER:
        audio_np = torch.from_numpy(audio_np).detach().cpu().numpy()
        segments, _ = model.transcribe(audio_np, language="en", beam_size=5, without_timestamps=True)
        return ''.join(segment.text for segment in segments).strip()
    else:
        audio_np = torch.from_numpy(audio_np).to(device)
        audio_np = whisper.pad_or_trim(audio_np)
        mel = whisper.log_mel_spectrogram(audio_np).to(device)
        options = whisper.DecodingOptions(language="en", fp16=torch.cuda.is_available(), without_timestamps=True)
        result = whisper.decode(model, mel, options)
        return result.text.strip()
