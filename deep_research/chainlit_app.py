import os
import asyncio
import chainlit as cl

from research import DeepResearch, system_prompt
from llm import LLMClient
from search import WebSearch
from content_fetcher import ContentFetcher

# Initialize required components.
api_key = os.getenv("GEMINI_API_KEY")  
llm_client = LLMClient(api_key=api_key)
search_client = WebSearch(max_results=5)
fetcher = ContentFetcher()

deep_researcher = DeepResearch(
    llm=llm_client,
    web_search=search_client,
    content_fetcher=fetcher,
    concurrency_limit=2,
    verbose=True,
    max_iterations=10,
)

@cl.on_chat_start
async def on_chat_start():
    """
    When a chat session starts, send a welcome message to the user.
    """
    welcome_msg = (
        "Welcome to the Deep Research Chat Interface!\n\n"
        "Enter your research query (for example, ask for market adoption rates, "
        "language learning statistics, and mobile penetration changes over the past 10 years) "
        "and I'll run our deep research workflow to generate a detailed report."
    )
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def on_message(message: cl.Message):
    """
    This function is called every time a user sends a message. The message content is treated as
    the research query. The deep research workflow is executed, and the final report is displayed.
    """
    original_query = message.content.strip()

    await cl.Message(content=f"Received query:\n\n`{original_query}`\n\nRunning deep research...").send()

    try:
        result = await deep_researcher.deep_research(original_query)
        final_report = await deep_researcher.write_final_report(
            original_query=original_query,
            learnings=result["learnings"],
            visited_urls=result["visited_urls"]
        )
    except Exception as e:
        error_msg = f"An error occurred while processing your query: {e}"
        await cl.Message(content=error_msg).send()
        return

    await cl.Message(content=final_report).send()
