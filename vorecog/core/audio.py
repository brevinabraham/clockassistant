import sounddevice as sd
import queue
from vorecog.configs.config import SAMPLE_RATE, BLOCK_SIZE, CHANNELS
import time

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️ Audio Status:", status)
    audio_queue.put((indata.copy(), time.time()))

def start_stream():
    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        callback=audio_callback,
        blocksize=BLOCK_SIZE
    )
