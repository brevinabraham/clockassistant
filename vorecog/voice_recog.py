import sounddevice as sd
import numpy as np
import datetime as dt
import time
import queue
import json
from vosk import Model, KaldiRecognizer

from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.io.wavfile import write
from pathlib import Path
import torch

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000
CHANNELS = 1
RMS_THRESHOLD = 500.0
SILENCE_DELAY_SECS = 2
RECORD_SECONDS = 4.5
REQUIRED_SAMPLES = 10
SIMILARITY_PERCENTAGE = 0.77
BUFFER_LIMIT_SECS = 3

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("🧠 Voice encoder using:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("⚠️ CUDA not available, using CPU fallback.")

encoder = VoiceEncoder(device=str(device))

VOICE_FOLDER = Path("vorecog/voice_sample")
VOICE_FOLDER.mkdir(parents=True, exist_ok=True)
EMBEDDING_PATH = VOICE_FOLDER / "your_embedding.npy"

model = Model("vorecog/models/vosk-model-en-us-0.22")
recogniser = KaldiRecognizer(model, SAMPLE_RATE)

audio_queue = queue.Queue()

def record_voice_sample():
    print("🎙️ Recording sample... Please speak clearly")
    sample = sd.rec(int(SAMPLE_RATE * RECORD_SECONDS), samplerate=SAMPLE_RATE,
                    channels=1, dtype='int16')
    sd.wait()
    timestamp = int(time.time())
    new_path = VOICE_FOLDER / f"my_voice_new_{timestamp}.wav"
    write(str(new_path), SAMPLE_RATE, sample)
    print(f"✅ Saved: {new_path.name}")
    return new_path

def ensure_10_voice_samples():
    existing = sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))
    needed = REQUIRED_SAMPLES - len(existing)
    for _ in range(needed):
        record_voice_sample()
    return sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))

def generate_initial_embedding():
    voice_paths = ensure_10_voice_samples()
    embeddings = [encoder.embed_utterance(preprocess_wav(p)) for p in voice_paths]
    final_embedding = np.mean(embeddings, axis=0)
    np.save(EMBEDDING_PATH, final_embedding)
    print("✅ Voice profile initialized with 10 samples.")
    return final_embedding

def save_and_train(audio_bytes, similarity, your_embedding):
    timestamp = int(time.time())
    new_path = VOICE_FOLDER / f"my_voice_new_{timestamp}.wav"
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    write(str(new_path), SAMPLE_RATE, audio_np)

    voice_files = sorted(VOICE_FOLDER.glob("my_voice_new_*.wav"))
    file_scores = []
    for file in voice_files:
        emb = encoder.embed_utterance(preprocess_wav(file))
        sim = np.dot(emb, your_embedding)
        file_scores.append((file, emb, sim))

    file_scores.sort(key=lambda x: x[2], reverse=True)


    for file, _, score in file_scores[10:]:
        print(f"🗑️ Removed: {file.name} (sim={score:.2f})")
        file.unlink()


    final_embeddings = [emb for _, emb, _ in file_scores[:10]]
    final_embedding = np.mean(final_embeddings, axis=0)
    np.save(EMBEDDING_PATH, final_embedding)

    # Save JSON similarity log
    log_path = VOICE_FOLDER / "similarity_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "ranked_files": [
                    {
                        "file": str(file.name),
                        "similarity": float(round(score, 4))
                    }
                    for file, _, score in file_scores[:10]
                ]
            },
            f,
            indent=2
        )

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

def main():
    global your_embedding
    print("🎤 Started")
    if not EMBEDDING_PATH.exists():
        your_embedding = generate_initial_embedding()
    else:
        your_embedding = np.load(EMBEDDING_PATH)

    last_state = 'silent'
    last_speech_time = 0
    silence_announced = False

    buffered_audio = b""
    buffer_duration = 0

    similarity = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype='float32', callback=audio_callback,
                        blocksize=BLOCK_SIZE):

        while True:
            data = audio_queue.get()
            rms, volume = calculate_volume(data)
            current_time = time.time()
            is_speaking = rms > RMS_THRESHOLD

            if is_speaking:
                buffered_audio += data
                buffer_duration += len(data) / 2 / SAMPLE_RATE

                if buffer_duration >= BUFFER_LIMIT_SECS:
                    try:
                        audio_np = np.frombuffer(buffered_audio, dtype=np.int16).astype(np.float32) / 32767.0
                        live_embedding = encoder.embed_utterance(audio_np)
                        similarity = np.dot(live_embedding, your_embedding)
                        print("similarity: ", similarity)
                        if similarity >= SIMILARITY_PERCENTAGE:
                            your_embedding = save_and_train(buffered_audio, similarity, your_embedding)
                        else:
                            print(f"🧠 Not your voice. Similarity: {similarity:.2f}")
                        buffered_audio = b""
                        buffer_duration = 0
                    except Exception as e:
                        print("⚠️ Voice check failed:", e)
                        buffered_audio = b""
                        buffer_duration = 0

            if recogniser.AcceptWaveform(data):
                result = json.loads(recogniser.Result())
                text = result.get("text", "").strip()
                if text:
                    label = "🟢 ME" if similarity >= SIMILARITY_PERCENTAGE else "🔵 OTHER"
                    print(f"{label} {dt.datetime.now().strftime('%H:%M')}: {text} [Volume: {volume:.0f}/100]")

            if is_speaking:
                if last_state != 'speaking':
                    last_state = 'speaking'
                silence_announced = False
                last_speech_time = current_time
            elif not silence_announced and (current_time - last_speech_time) > SILENCE_DELAY_SECS:
                print(f"🔴 {dt.datetime.now().strftime('%H:%M')}: ... [Volume: {volume:.0f}/100]")
                last_state = 'silent'
                silence_announced = True

if __name__ == "__main__":
    main()
