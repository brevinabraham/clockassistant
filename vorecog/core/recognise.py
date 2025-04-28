import numpy as np
from vorecog.core.embedding import cosine_similarity, get_embedding

def recognise(audio_np, your_embedding):
    emb = get_embedding(audio_np)
    return cosine_similarity(emb, your_embedding)
