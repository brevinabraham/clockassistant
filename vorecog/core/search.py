import httpx
from bs4 import BeautifulSoup
import asyncio
import random
from vorecog.core.chat_router import handle_chat_mistral_async

async def summarize_html_with_question(html_content: str, question: str) -> str:
    """Ask the AI to read HTML and answer the user's question."""
    prompt = (
        "[INST] "
        "You are an intelligent assistant. "
        "Given the following HTML content from a web search, find the most relevant information to answer the user's question. "
        "Only answer based on the information inside the HTML. "
        "If no answer is available, say 'No answer found.'\n\n"
        f"User Question: {question}\n\n"
        f"HTML Content:\n{html_content}\n"
        "[/INST]"
    )
    return await handle_chat_mistral_async(prompt)


async def web_search_async(query: str) -> str:
    try:
        await asyncio.sleep(random.uniform(0.8, 1.5))

        headers = {
            "User-Agent": "Chrome/113.0.0.0 Safari/537.36"
        }
        params = {"q": query}

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("https://lite.duckduckgo.com/lite", params=params, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            # sumarised_html_ai = await summarize_html_with_question(soup,query)
            # print(sumarised_html_ai)
            # Try to find the first result link
            result_link = soup.select_one("a.result-link")
            snippet_text = ""

            if result_link:
                parent_row = result_link.find_parent("tr")

                # Try finding snippet inside the same row
                if parent_row:
                    snippet_in_same_row = parent_row.select_one("td.result-snippet")
                    if snippet_in_same_row:
                        snippet_text = snippet_in_same_row.get_text(strip=True)

                    else:
                        # Otherwise, look at the next row
                        next_row = parent_row.find_next_sibling("tr")
                        if next_row:
                            snippet_in_next_row = next_row.select_one("td.result-snippet")
                            if snippet_in_next_row:
                                snippet_text = snippet_in_next_row.get_text(strip=True)

                # Build result
                title = result_link.get_text(strip=True)
                href = result_link.get("href")
                href = href if href.startswith("http") else "https:" + href

                return f"{title}\n{snippet_text}\n{href}" if snippet_text else f"{title}\n{href}"

            # Fallback: just any first snippet if no good result-link
            first_snippet = soup.select_one("td.result-snippet")
            if first_snippet:
                return first_snippet.get_text(strip=True)

            return "No good results found."

    except Exception as e:
        print(f"❗ Web search failed: {e}")
        return "Search error — please try again later."
