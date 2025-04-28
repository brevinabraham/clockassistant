import warnings
warnings.filterwarnings("ignore", module="speechbrain")
warnings.filterwarnings("ignore", module="torch")
import numpy as np
import time
import datetime as dt
from vorecog.configs.config import *
from vorecog.core.audio import start_stream, audio_queue
from vorecog.core.transcribe import transcribe_audio
from vorecog.core.embedding import generate_initial_embedding, save_and_train
from vorecog.core.recognise import recognise
from vorecog.core.logger import save_log




def main():
    print(f"🛠️ Running Voice Recognition Project {VERSION}")
    your_embedding = np.load(EMBEDDING_PATH) if EMBEDDING_PATH.exists() else generate_initial_embedding()

    buffered_audio = b""
    similarity_history = []
    is_training = True
    last_training = time.time()
    last_speech = time.time()
    is_currently_speaking = False

    try:
        with start_stream():
            while True:
                data, _ = audio_queue.get()
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
                    sim = recognise(audio_np, your_embedding)

                    if sim >= SIMILARITY_PERCENTAGE:
                        similarity_history.append(sim)
                        if len(similarity_history) > HISTORY_SIZE:
                            similarity_history.pop(0)

                    avg_sim = sum(similarity_history) / len(similarity_history) if similarity_history else 0.0

                    if is_training and len(buffered_audio) > MIN_SECONDS_AUDIO*(SAMPLE_RATE*2) and sim >= SIMILARITY_PERCENTAGE and (now - last_speech) > SAMPLE_DURATION and (now - last_training) > TRAINING_COOLDOWN:
                        your_embedding = save_and_train(buffered_audio, your_embedding)
                        last_training = now

                else:
                    if is_currently_speaking and (now - last_speech) > SILENCE_DELAY_SECS:
                        if len(buffered_audio) > MIN_SECONDS_AUDIO*(SAMPLE_RATE*2):
                            audio_np = np.frombuffer(buffered_audio, dtype=np.int16).astype(np.float32) / 32767.0
                            text = transcribe_audio(audio_np)
                            avg_sim = sum(similarity_history) / len(similarity_history) if similarity_history else 0.0
                            label = "🟢 ME" if avg_sim >= SIMILARITY_PERCENTAGE else "🔵 OTHER"
                            print(f"{label} {dt.datetime.now().strftime('%H:%M')}: {text} [S:{avg_sim*100:.0f}]")

                            save_log({
                                "timestamp": str(dt.datetime.now().isoformat()),
                                "text": text,
                                "volume": str(volume),
                                "similarity": str(avg_sim),
                                "is_me": str(avg_sim >= SIMILARITY_PERCENTAGE)
                            })

                        buffered_audio = b""
                        similarity_history.clear()
                        is_currently_speaking = False
                        last_speech = now

    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()
