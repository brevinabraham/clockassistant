from vorecog.core.llm_tools import (
    should_use_web_search_llm_async,
    rewrite_query_for_search_async,
    summarise_snippet_async,
)
from vorecog.core.mistral_async import handle_chat_mistral_async
from vorecog.core.search import web_search_async

async def handle_chat_async(text: str) -> str:
    try:
        if await should_use_web_search_llm_async(text):
            print("🌐 [LLM] Using web search based on intent...")
            # search_query = await rewrite_query_for_search_async(text)
            # print(search_query)
            result = await web_search_async(text)  
            if "Search error" in result or "No good results found" in result:
                return f"[Real-Time Info]\n{result}"
            
            summarised = await summarise_snippet_async(result)
            return f"[Real-Time Info]\n{summarised}"

        print("💡 Using local Mistral...")
        return await handle_chat_mistral_async(text)
    
    except KeyboardInterrupt:
        print("\n❗ Chat interrupted by user (Ctrl+C).")
        exit()
