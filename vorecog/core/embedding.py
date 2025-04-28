import numpy as np
import sounddevice as sd
import torch
import torchaudio
from pathlib import Path
from vorecog.configs.config import *
import time

from scipy.io.wavfile import write
from speechbrain.inference import EncoderClassifier

encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})

def cosine_similarity(a, b):
    a = torch.tensor(a).float().flatten()
    b = torch.tensor(b).float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def get_embedding(source):
    if isinstance(source, (Path, str)):
        signal, sr = torchaudio.load(str(source))
    else:
        signal = torch.tensor(source).unsqueeze(0)
        sr = SAMPLE_RATE
    if sr != SAMPLE_RATE:
        signal = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(signal)
    signal = (signal - signal.mean()) / (signal.std() + 1e-5)
    signal = torch.clamp(signal, -1.0, 1.0)
    return encoder.encode_batch(signal).squeeze(0).detach().cpu().numpy()

def generate_initial_embedding():
    samples = sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))
    if len(samples) < REQUIRED_SAMPLES:
        print("❗Please record more voice samples.")
        record_remaining_samples = REQUIRED_SAMPLES - len(samples)
        for i in range(record_remaining_samples):
            recording = sd.rec(int(SAMPLE_RATE * SAMPLE_DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            audio_data = (recording.flatten() * 32767).astype(np.int16)
            timestamp = int(time.time())
            sample_path = VOICE_FOLDER / f"my_voice_new_{timestamp}.wav"
            write(str(sample_path), SAMPLE_RATE, audio_data)
            print(f"✅ Sample {i+1}/{record_remaining_samples} recorded.")
            time.sleep(1)

    samples = sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))
    embeddings = [get_embedding(f) for f in samples]
    mean_emb = np.mean(embeddings, axis=0)
    np.save(EMBEDDING_PATH, mean_emb)
    print("✅ Initial embedding created.")
    return mean_emb

def save_and_train(audio_bytes, your_embedding):
    new_path = VOICE_FOLDER / f"my_voice_new_{int(time.time())}.wav"
    write(str(new_path), SAMPLE_RATE, np.frombuffer(audio_bytes, dtype=np.int16))
    print(f"✅ New sample saved: {new_path.name}")

    files = sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))
    scores = []
    for f in files:
        emb = get_embedding(f)
        sim = cosine_similarity(emb, your_embedding)
        scores.append((f, emb, sim))

    scores.sort(key=lambda x: x[2], reverse=True)

    while len(scores) > REQUIRED_SAMPLES:
        removable = None
        for f, _, _ in reversed(scores):
            if f != new_path:
                removable = f
                break
        if removable:
            print(f"🗑️ Removing {removable.name}")
            removable.unlink()
            scores = [(f, emb, sim) for f, emb, sim in scores if f != removable]
        else:
            break

    final_emb = np.mean([emb for _, emb, _ in scores], axis=0)
    np.save(EMBEDDING_PATH, final_emb)
    print("✅ Embedding updated.")
    return final_emb
