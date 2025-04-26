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
from speechbrain.inference import EncoderClassifier
import tempfile
import io
import soundfile as sf
from faster_whisper import WhisperModel
from vorecog.configs.config import *


print(f"🛠️ Running Voice Recognition Project {VERSION}")

# ========== Models ==========
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🧠 Voice encoder using: {device}")
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": str(device)})
# model = whisper.load_model("medium", download_root="vorecog/models/whisper-medium")
model = WhisperModel("medium",download_root="vorecog/models/faster-whisper-medium", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

# ========== Audio Queue ==========
audio_queue = queue.Queue()

# ========== Helper Functions ==========
# def transcribe_in_memory(audio_np):#this is for the whisper medium model
#     if audio_np.ndim > 1:
#         audio_np = np.mean(audio_np, axis=0) 
#     audio_np = whisper.pad_or_trim(torch.from_numpy(audio_np).to(model.device))
#     mel = whisper.log_mel_spectrogram(audio_np).to(model.device)
#     options = whisper.DecodingOptions(language="en", fp16=torch.cuda.is_available(), without_timestamps=True)
#     result = whisper.decode(model, mel, options)
#     return result.text.strip()

def transcribe_in_memory(audio_np):#this is for the faster medium model
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=0)  # mono
    # audio_np = torch.from_numpy(audio_np).cpu().numpy()
    audio_np = torch.from_numpy(audio_np).detach().cpu().numpy()
    segments, _ = model.transcribe(audio_np, language="en", beam_size=5, without_timestamps=True)
    text = ''.join(segment.text for segment in segments)
    return text.strip()

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
    scores = []
    for f in files:
        emb = get_embedding(f)
        sim = cosine_similarity(emb, your_embedding)
        scores.append((f, emb, sim))

    scores.sort(key=lambda x: x[2], reverse=True)
    for f, _, _ in scores[REQUIRED_SAMPLES:]:
        # print(f"🗑️ Removed {f.name}")
        f.unlink()
    final_emb = np.mean([emb for _, emb, _ in scores[:REQUIRED_SAMPLES]], axis=0)
    np.save(EMBEDDING_PATH, final_emb)
    # print(f"✅ Trained with sample (sim={similarity:.2f})")
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
    # last_partial_transcribe = time.time()
    try:
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


                    #partial showing
                    # if now - last_partial_transcribe >= PARTIAL_TRANSCRIBE_INTERVAL:
                    #     if len(buffered_audio) > 0:
                    #         audio_np = np.frombuffer(buffered_audio, dtype=np.int16).astype(np.float32) / 32767.0
                    #         text = transcribe_in_memory(audio_np)

                    #         if text and len(text.split()) >= MIN_WORDS_THRESHOLD:
                    #             avg_sim = sum(similarity_history) / len(similarity_history) if similarity_history else 0.0
                    #             label = "🟡"
                    #             print(f"{label} {dt.datetime.now().strftime('%H:%M')}: {text} (Partial) [Sim={avg_sim:.2f}]", end='\r')

                    #     last_partial_transcribe = now

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
                            text = transcribe_in_memory(audio_np)

                            if text and len(text.split()) >= MIN_WORDS_THRESHOLD:
                                avg_sim = sum(similarity_history) / len(similarity_history) if similarity_history else 0.0
                                label = "🟢 ME" if avg_sim >= SIMILARITY_PERCENTAGE else "🔵 OTHER"
                                print(f"{label} {dt.datetime.now().strftime('%H:%M')}: {text} [rms={rms:.0f}, Vol={volume:.0f}, Sim={sim:.2f}]")

                                log_entry = {
                                    "timestamp": str(dt.datetime.now().isoformat()),
                                    "text": text,
                                    "volume": str(volume),
                                    "similarity": str(avg_sim),
                                    "is_me": str(avg_sim >= SIMILARITY_PERCENTAGE)
                                }
                                if LOG_PATH.exists():
                                    with open(LOG_PATH, "r") as f:
                                        logs = json.load(f)
                                else:
                                    logs = []
                                logs.append(log_entry)
                                if len(logs) > 500:
                                    logs = logs[-500:]
                                json.dump(logs, open(LOG_PATH, "w"), indent=2)
                            
                            if "stop training" in text.lower():
                                is_training = False
                                
                            if "start training" in text.lower():
                                is_training = True
                                

                            buffered_audio = b""
                            similarity_history.clear()
                            is_currently_speaking = False

                if is_speaking:
                    if last_state != 'speaking':
                        last_state = 'speaking'
                    silence_announced = False
                elif not silence_announced and (now - last_speech) > SILENCE_DELAY_SECS:
                    # print(f"🔴 {dt.datetime.now().strftime('%H:%M')}: ...silent")
                    last_state = 'silent'
                    silence_announced = True

    except Exception as e:
        print(f"❌ Error occurred: {e}")

# ========== Run ==========
if __name__ == "__main__":
    main()
