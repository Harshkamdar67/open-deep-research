import asyncio
import os
import sys
import logging

from rich.console import Console
from rich.markdown import Markdown

from research import DeepResearch
from llm import LLMClient
from search import WebSearch
from content_fetcher import ContentFetcher

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

async def main():
    console = Console()

    console.print("[bold green]Welcome to the Deep Research Terminal Application![/bold green]\n")

    query = input("Please enter your research query: ").strip()
    if not query:
        logger.error("No query provided. Exiting.")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set. Please set the API key environment variable.")
        sys.exit(1)

    llm_client = LLMClient(api_key=api_key)
    search_client = WebSearch(max_results=5)
    fetcher = ContentFetcher()

    # Create the deep research orchestrator
    deep_researcher = DeepResearch(
        llm=llm_client,
        web_search=search_client,
        content_fetcher=fetcher,
        concurrency_limit=2,
        verbose=True, 
        max_iterations=10
    )

    logger.info("Starting deep research for your query...")
    research_result = await deep_researcher.deep_research(query)
    logger.info("Deep research completed. Generating final report...")

    final_report = await deep_researcher.write_final_report(
        original_query=query,
        learnings=research_result["learnings"],
        visited_urls=research_result["visited_urls"]
    )

    console.print("\n[bold blue]=== FINAL REPORT ===[/bold blue]\n")
    md = Markdown(final_report)
    console.print(md)

    if research_result["visited_urls"]:
        console.print("\n[bold]Visited URLs:[/bold]")
        for url in research_result["visited_urls"]:
            console.print(f"- {url}")

if __name__ == "__main__":
    asyncio.run(main())
