import sounddevice as sd
import numpy as np
import datetime as dt
import time
import queue
import json
import whisper
import os
from speechbrain.pretrained import EncoderClassifier
import torchaudio

from scipy.io.wavfile import write
from pathlib import Path

import torch
import torch.nn.functional as F

def cosine_similarity(a, b):
    a = torch.tensor(a).float().flatten()
    b = torch.tensor(b).float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()




# ========== Configurations ==========
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000
CHANNELS = 1
RMS_THRESHOLD = 500.0
SILENCE_DELAY_SECS = 2
RECORD_SECONDS = 6
REQUIRED_SAMPLES = 20
SIMILARITY_PERCENTAGE = 0.45
BUFFER_LIMIT_SECS = 5
MIN_VOLUME_THRESHOLD = 0
MIN_WORDS_THRESHOLD = 2
HISTORY_SIZE = 5

# ========== Initialize Models ==========
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("🧠 Voice encoder using:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("⚠️ CUDA not available, using CPU fallback.")

encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": str(device)})
model = whisper.load_model("medium", download_root="vorecog/models/whisper-medium")

# ========== Paths ==========
VOICE_FOLDER = Path("vorecog/voice_sample")
VOICE_FOLDER.mkdir(parents=True, exist_ok=True)
EMBEDDING_PATH = VOICE_FOLDER / "your_embedding.npy"
LOG_PATH = Path("vorecog/voice_log.json")


# ========== Audio Queue ==========
audio_queue = queue.Queue()

# ========== Helper Functions ==========
def record_voice_sample():
    print("🎙️ Recording sample... Please speak clearly")
    sample = sd.rec(int(SAMPLE_RATE * RECORD_SECONDS), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    timestamp = int(time.time())
    new_path = VOICE_FOLDER / f"my_voice_new_{timestamp}.wav"
    write(str(new_path), SAMPLE_RATE, sample)
    print(f"✅ Saved: {new_path.name}")
    return new_path

def ensure_voice_samples():
    existing = sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))
    needed = REQUIRED_SAMPLES - len(existing)
    for _ in range(needed):
        record_voice_sample()
    return sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))

def get_embedding(wav_path_or_array):
    if isinstance(wav_path_or_array, Path) or isinstance(wav_path_or_array, str):
        signal, sr = torchaudio.load(str(wav_path_or_array))
    else:
        signal = torch.tensor(wav_path_or_array).unsqueeze(0)
        sr = SAMPLE_RATE

    if sr != SAMPLE_RATE:
        signal = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(signal)
    
    # 🚿 Apply light denoising
    signal = (signal - signal.mean()) / (signal.std() + 1e-5)  # Normalize
    signal = torch.clamp(signal, min=-1.0, max=1.0)            # Clamp extreme values

    embedding = encoder.encode_batch(signal).squeeze(0).detach().cpu().numpy()
    return embedding


def generate_initial_embedding():
    voice_paths = ensure_voice_samples()
    embeddings = [get_embedding(p) for p in voice_paths]
    final_embedding = np.mean(embeddings, axis=0)
    np.save(EMBEDDING_PATH, final_embedding)
    print("✅ Voice profile initialized with 20 samples.")
    return final_embedding

def save_and_train(audio_bytes, similarity, your_embedding):
    timestamp = int(time.time())
    new_path = VOICE_FOLDER / f"my_voice_new_{timestamp}.wav"
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    write(str(new_path), SAMPLE_RATE, audio_np)

    voice_files = sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))
    file_scores = []
    for file in voice_files:
        emb = get_embedding(file)
        sim = cosine_similarity(emb, your_embedding)

        file_scores.append((file, emb, sim))

    file_scores.sort(key=lambda x: x[2], reverse=True)

    for file, _, score in file_scores[REQUIRED_SAMPLES:]:
        print(f"🗑️ Removed: {file.name} (sim={score:.2f})")
        file.unlink()

    final_embeddings = [emb for _, emb, _ in file_scores[:REQUIRED_SAMPLES]]
    final_embedding = np.mean(final_embeddings, axis=0)
    np.save(EMBEDDING_PATH, final_embedding)

    log_path = VOICE_FOLDER / "similarity_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "ranked_files": [
                {"file": str(file.name), "similarity": float(round(score, 4))}
                for file, _, score in file_scores[:REQUIRED_SAMPLES]
            ]
        }, f, indent=2)

    print(f"✅ Trained with new sample (sim={similarity:.2f})")
    return final_embedding

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio Status:", status)
    audio_int16 = (indata * 32767).astype(np.int16)
    audio_queue.put(audio_int16.tobytes())

def calculate_volume(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(audio_np ** 2))
    volume = min((rms / 3000.0) * 100, 100)
    return rms, volume

# ========== Main Logic ==========
def main():
    print("🎤 Started")

    if not EMBEDDING_PATH.exists():
        your_embedding = generate_initial_embedding()
    else:
        your_embedding = np.load(EMBEDDING_PATH)

    last_state = 'silent'
    last_speech_time = 0
    last_training_time = 0
    TRAINING_COOLDOWN_SECS = 6

    silence_announced = False
    is_training = True

    buffered_audio = b""
    buffer_duration = 0
    avg_similarity = 0.0
    similarity_history = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=audio_callback, blocksize=BLOCK_SIZE):
        while True:
            data = audio_queue.get()
            rms, volume = calculate_volume(data)
            current_time = time.time()

            is_speaking = rms > RMS_THRESHOLD and volume >= MIN_VOLUME_THRESHOLD

            if is_speaking and volume >= MIN_VOLUME_THRESHOLD:
                buffered_audio += data
                buffer_duration += len(data) / 2 / SAMPLE_RATE

                audio_np = np.frombuffer(buffered_audio, dtype=np.int16).astype(np.float32) / 32767.0
                live_embedding = get_embedding(audio_np)
                similarity = cosine_similarity(live_embedding, your_embedding)


                if similarity >= SIMILARITY_PERCENTAGE:
                    similarity_history.append(similarity)
                    if len(similarity_history) > HISTORY_SIZE:
                        similarity_history.pop(0)

                if similarity_history:
                    avg_similarity = sum(similarity_history) / len(similarity_history)
                else:
                    avg_similarity = 0.0


                if buffer_duration >= BUFFER_LIMIT_SECS and is_training:
                    if current_time - last_training_time >= TRAINING_COOLDOWN_SECS:
                        try:
                            print(f"similarity: {similarity:.2f}")
                            if similarity >= SIMILARITY_PERCENTAGE:
                                your_embedding = save_and_train(buffered_audio, similarity, your_embedding)
                                last_training_time = current_time  
                            else:
                                print(f"🧠 Not your voice. Similarity: {similarity:.2f}")
                            buffered_audio = b""
                            buffer_duration = 0
                        except Exception as e:
                            print("⚠️ Voice check failed:", e)
                            buffered_audio = b""
                            buffer_duration = 0
                    else:
                        buffered_audio = b""
                        buffer_duration = 0
            else:
                if buffered_audio:
                    buffered_audio = b""
                    buffer_duration = 0

            if len(buffered_audio) >= int(SAMPLE_RATE * BUFFER_LIMIT_SECS):
                audio_np = np.frombuffer(buffered_audio, dtype=np.int16).astype(np.float32) / 32767.0
                audio_tensor = torch.tensor(audio_np).unsqueeze(0)

                # Save temp WAV
                temp_wav = "temp.wav"
                write(temp_wav, SAMPLE_RATE, (audio_np * 32767).astype(np.int16))

                result = model.transcribe(temp_wav, language="en", fp16=torch.cuda.is_available())
                text = result['text'].strip()
                os.remove(temp_wav)

                if text and len(text.split()) >= MIN_WORDS_THRESHOLD:
                    label = "🟢 ME" if avg_similarity >= SIMILARITY_PERCENTAGE else "🔵 OTHER"
                    print(f"{label} {dt.datetime.now().strftime('%H:%M')}: {text} [Volume: {volume:.0f}/100] [Similarity: {avg_similarity:.2f}]")
                    
                    # Save to log
                    log_entry = {
                        "timestamp": str(dt.datetime.now().isoformat()),
                        "text": text,
                        "volume": str(volume),
                        "similarity": str(avg_similarity),
                        "is_me": str(avg_similarity >= SIMILARITY_PERCENTAGE)
                    }
                    if LOG_PATH.exists():
                        with open(LOG_PATH, "r") as f:
                            logs = json.load(f)
                    else:
                        logs = []
                    logs.append(log_entry)
                    with open(LOG_PATH, "w") as f:
                        json.dump(logs, f, indent=2)
                
                buffered_audio = b""
                buffer_duration = 0


            if is_speaking and volume >= MIN_VOLUME_THRESHOLD:
                if last_state != 'speaking':
                    last_state = 'speaking'
                silence_announced = False
                last_speech_time = current_time
            elif not silence_announced and (current_time - last_speech_time) > SILENCE_DELAY_SECS:
                print(f"🔴 {dt.datetime.now().strftime('%H:%M')}: ... [Volume: {volume:.0f}/100]")
                last_state = 'silent'
                silence_announced = True

# ========== Entry Point ==========
if __name__ == "__main__":
    main()
