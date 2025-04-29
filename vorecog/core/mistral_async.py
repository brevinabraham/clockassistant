import ctranslate2
import sentencepiece as spm
import re
import asyncio
from vorecog.configs.config import MODELS_DIR
from vorecog.core.transcribe import device

# ========== Load Mistral-7B GGUF ==========
MISTRAL_MODEL_PATH = MODELS_DIR / "mistral-7b-instruct"
generator_mistral = ctranslate2.Generator(str(MISTRAL_MODEL_PATH))
tokenizer_mistral = spm.SentencePieceProcessor()
tokenizer_mistral.load(str(MISTRAL_MODEL_PATH / "tokenizer.model"))

print("✅ Mistral (ctranslate2.Generator) loaded!")

# ========== Helper Functions ==========

def sanitize_response(text):
    text = re.sub(r'(\W)\1{3,}', r'\1', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    return text.strip()

def build_prompt_mistral(user_text):
    return (
        "[INST] "
        "You are a precise assistant. "
        "Answer the user's question clearly. "
        "If the question is misunderstood due to text pick up from mic, infer from context. "
        "Only answer once. Do not add extra explanations unless requested. "
        "If you don't understand the question, reply exactly: 'I don't understand.' "
        "Always reply in English only. "
        f"Question: {user_text.strip()} "
        "[/INST]"
    )

async def handle_chat_mistral_async(text):
    prompt = build_prompt_mistral(text)
    tokens = tokenizer_mistral.encode(prompt, out_type=str)

    loop = asyncio.get_event_loop()

    def mistral_blocking_call():
        results = generator_mistral.generate_batch(
            [tokens],
            max_length=256,
            sampling_topk=1,
            include_prompt_in_result=False
        )
        output_tokens = results[0].sequences[0]
        return sanitize_response(tokenizer_mistral.decode(output_tokens))

    response = await loop.run_in_executor(None, mistral_blocking_call)
    return response
