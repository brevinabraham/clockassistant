from vorecog.core.mistral_async import handle_chat_mistral_async

async def should_use_web_search_llm_async(text: str) -> bool:
    prompt = (
        "You are an assistant that decides whether a user's query needs to be web searched.\n"
        "Please make sure to reply with 'YES' or 'NO' and don't add anything else.\n"
        "\n"
        "If the Question is a basic fact, say NO.\n"
        "If the Question basic maths, say NO.\n"
        "If the Question depends on real-time events or recent updates, say YES.\n"
        "\n"
        f"User: {text.strip()}\n"
        "Answer:"
    )
    result = await handle_chat_mistral_async(prompt)
    return "YES" in result.strip().upper()

async def rewrite_query_for_search_async(text: str) -> str:
    prompt = (
        "You are an assistant that simplifies a user's query for a web search.\n"
        "Your goal is to rewrite the question as clearly and directly as possible, without adding or assuming any extra information.\n"
        "Focus only on exactly what the user is asking.\n"
        "\n"
        "Simplify the query for a direct search.\n"
        "\n"
        f"User: {text.strip()}\n"
        "Simplified Search Query:"
    )
    result = await handle_chat_mistral_async(prompt)
    return result.strip() or text.strip()

async def summarise_snippet_async(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return text

    prompt = f"Summarise this information briefly in 1-2 sentences or a paragraph depending on the context and what the user wants to know:\n\n{text}"
    return await handle_chat_mistral_async(prompt)
