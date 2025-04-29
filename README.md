# 🧠 Voice Recognition Assistant (v1.0.0)

A real-time voice-controlled assistant that:
- 🗣️ Listens to your voice,
- 🧠 Uses local LLM (Mistral-7B via CTranslate2) for intent + answer generation,
- 🌐 Fetches real-time web results for dynamic questions,
- 🧬 Identifies your voice (vs others) using speaker embeddings,
- 🎙️ Powered by Whisper, Resemblyzer, CTranslate2, and Python async magic.

---

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Async Ready](https://img.shields.io/badge/asyncio-enabled-brightgreen)
![LLM](https://img.shields.io/badge/LLM-Mistral--7B-orange)
![Speech](https://img.shields.io/badge/speech-Whisper-lightgrey)
![Web Search](https://img.shields.io/badge/web--search-duckduckgo-purple)
![Speaker ID](https://img.shields.io/badge/speaker--recognition-resemblyzer-9cf)

---

## ⚙️ Features

- 🎧 **Real-time microphone input**
- 🤖 **Local LLM reasoning (Mistral-7B)** via CTranslate2
- 🌐 **Auto-detects web-based queries** and fetches live results
- 🎤 **Speaker diarization** with voice training and 10-sample pruning
- 🧠 **Fully async**, multi-thread safe architecture
- 🛠️ **Configurable**, easy to extend with your own models or search engines

---

## 🧰 Tech Stack

| Category        | Tool/Library        |
|----------------|---------------------|
| LLM Inference  | `ctranslate2` + `Mistral 7B` |
| Tokenizer      | `sentencepiece`      |
| Speech-to-Text | `openai-whisper` / `faster-whisper` |
| Speaker ID     | `resemblyzer`        |
| Audio I/O      | `pyaudio`            |
| Search         | `httpx` + `BeautifulSoup` |
| Async Runtime  | `asyncio`, `concurrent.futures` |

---

## 🚀 Run the Assistant

```bash
pip install -r requirements.txt
python main.py
