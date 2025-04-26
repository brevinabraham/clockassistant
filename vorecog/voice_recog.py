import sounddevice as sd
import numpy as np
import datetime as dt
import time
import queue
import json
import whisper
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from scipy.io.wavfile import write
from speechbrain.pretrained import EncoderClassifier

# ========== Configurations ==========
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000
CHANNELS = 1
RMS_THRESHOLD = 500.0
MIN_VOLUME_THRESHOLD = 10
SIMILARITY_PERCENTAGE = 0.45
MIN_WORDS_THRESHOLD = 2
REQUIRED_SAMPLES = 20
HISTORY_SIZE = 5
TRAINING_COOLDOWN = 6
SILENCE_DELAY_SECS = 2
MAX_AUDIO_SECONDS = 20  
MAX_AUDIO_BYTES = SAMPLE_RATE * MAX_AUDIO_SECONDS * 2  

# ========== Paths ==========
VOICE_FOLDER = Path("vorecog/voice_sample")
VOICE_FOLDER.mkdir(parents=True, exist_ok=True)
EMBEDDING_PATH = VOICE_FOLDER / "your_embedding.npy"
LOG_PATH = Path("vorecog/voice_log.json")

# ========== Models ==========
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🧠 Voice encoder using: {device}")
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": str(device)})
model = whisper.load_model("medium", download_root="vorecog/models/whisper-medium")

# ========== Audio Queue ==========
audio_queue = queue.Queue()

# ========== Helper Functions ==========
def cosine_similarity(a, b):
    a = torch.tensor(a).float().flatten()
    b = torch.tensor(b).float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()

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

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️ Audio Status:", status)
    audio_queue.put((indata.copy(), time.time()))

def generate_initial_embedding():
    samples = sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))
    if len(samples) < REQUIRED_SAMPLES:
        print("❗Please record more voice samples.")
        exit()
    embeddings = [get_embedding(f) for f in samples]
    mean_emb = np.mean(embeddings, axis=0)
    np.save(EMBEDDING_PATH, mean_emb)
    return mean_emb

def save_and_train(audio_bytes, similarity, your_embedding):
    path = VOICE_FOLDER / f"my_voice_new_{int(time.time())}.wav"
    write(str(path), SAMPLE_RATE, np.frombuffer(audio_bytes, dtype=np.int16))
    files = sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))
    scores = [(f, get_embedding(f), cosine_similarity(get_embedding(f), your_embedding)) for f in files]
    scores.sort(key=lambda x: x[2], reverse=True)
    for f, _, _ in scores[REQUIRED_SAMPLES:]:
        print(f"🗑️ Removed {f.name}")
        f.unlink()
    final_emb = np.mean([emb for _, emb, _ in scores[:REQUIRED_SAMPLES]], axis=0)
    np.save(EMBEDDING_PATH, final_emb)
    print(f"✅ Trained with sample (sim={similarity:.2f})")
    return final_emb

# ========== Main ==========
def main():
    print("🎤 Started.")
    your_embedding = np.load(EMBEDDING_PATH) if EMBEDDING_PATH.exists() else generate_initial_embedding()

    buffered_audio = b""
    similarity_history = []
    is_training = True
    last_training = time.time()
    last_state = 'silent'
    silence_announced = False
    last_speech = time.time()
    is_currently_speaking = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=audio_callback, blocksize=BLOCK_SIZE):
        while True:
            data, timestamp = audio_queue.get()
            rms = np.sqrt(np.mean((data * 32767).astype(np.float32) ** 2))
            volume = min((rms / 3000) * 100, 100)
            is_speaking = rms > RMS_THRESHOLD and volume >= MIN_VOLUME_THRESHOLD
            now = time.time()

            if is_speaking:
                buffered_audio += (data * 32767).astype(np.int16).tobytes()
                if len(buffered_audio) > MAX_AUDIO_BYTES:
                    buffered_audio = buffered_audio[-MAX_AUDIO_BYTES:]
                last_speech = now
                is_currently_speaking = True

                audio_np = np.frombuffer(buffered_audio, dtype=np.int16).astype(np.float32) / 32767.0
                emb = get_embedding(audio_np)
                sim = cosine_similarity(emb, your_embedding)

                if sim >= SIMILARITY_PERCENTAGE:
                    similarity_history.append(sim)
                    if len(similarity_history) > HISTORY_SIZE:
                        similarity_history.pop(0)

                avg_sim = sum(similarity_history) / len(similarity_history) if similarity_history else 0.0

                if is_training and sim >= SIMILARITY_PERCENTAGE and (now - last_training) > TRAINING_COOLDOWN:
                    your_embedding = save_and_train(buffered_audio, sim, your_embedding)
                    last_training = now

            else:
                if is_currently_speaking and (now - last_speech) > SILENCE_DELAY_SECS:
                    if len(buffered_audio) > 0:
                        audio_np = np.frombuffer(buffered_audio, dtype=np.int16).astype(np.float32) / 32767.0
                        temp_wav = "temp.wav"
                        write(temp_wav, SAMPLE_RATE, (audio_np * 32767).astype(np.int16))
                        result = model.transcribe(temp_wav, language="en", fp16=torch.cuda.is_available())
                        os.remove(temp_wav)

                        text = result['text'].strip()
                        if text and len(text.split()) >= MIN_WORDS_THRESHOLD:
                            avg_sim = sum(similarity_history) / len(similarity_history) if similarity_history else 0.0
                            label = "🟢 ME" if avg_sim >= SIMILARITY_PERCENTAGE else "🔵 OTHER"
                            print(f"{label} {dt.datetime.now().strftime('%H:%M')}: {text} [Sim={avg_sim:.2f}]")

                            log_entry = {
                                "timestamp": str(dt.datetime.now().isoformat()),
                                "text": text,
                                "volume": str(volume),
                                "similarity": str(avg_sim),
                                "is_me": str(avg_sim >= SIMILARITY_PERCENTAGE)
                            }
                            if LOG_PATH.exists():
                                logs = json.load(open(LOG_PATH))
                            else:
                                logs = []
                            logs.append(log_entry)
                            json.dump(logs, open(LOG_PATH, "w"), indent=2)

                        buffered_audio = b""
                        similarity_history.clear()
                        is_currently_speaking = False

            if is_speaking:
                if last_state != 'speaking':
                    last_state = 'speaking'
                silence_announced = False
            elif not silence_announced and (now - last_speech) > SILENCE_DELAY_SECS:
                print(f"🔴 {dt.datetime.now().strftime('%H:%M')}: ...silent")
                last_state = 'silent'
                silence_announced = True

# ========== Run ==========
if __name__ == "__main__":
    main()
