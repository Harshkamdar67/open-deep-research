# Open Deep Research

[Watch the video](https://youtu.be/fLB1cUaUdaY)

Open Deep Research is an open-source alternative to OpenAI’s Deep Research, aiming to provide a robust research and data collection pipeline for large language model (LLM) projects. This repository offers a modular Python-based approach for performing multi-step research tasks, scraping relevant content from the web, and generating comprehensive summaries or structured reports.

## Overview

Open Deep Research orchestrates various modules and utilities to create an end-to-end research pipeline:

- **Content Fetcher (`content_fetcher.py`)**: Fetches content asynchronously from a list of URLs using the `AsyncWebCrawler`.
- **LLM Clients (`llm.py` & `llm_groq.py`)**: Provides reusable LLM client classes for different backends like Groq or Google Gemini.
- **Research Orchestrator (`research.py`)**: Drives the research process by deciding how many search iterations to run and how many queries to execute. Uses the LLM to produce comprehensive final outputs.
- **Terminal Application (`run.py`)**: Offers a CLI that allows users to input queries, run deep research, and display the resulting report in Markdown.
- **Search Utility (`search.py`)**: Executes web searches using DuckDuckGo.
- **Text Processing (`text_processing.py`)**: Provides text splitting and prompt trimming utilities for large documents.

## Key Features

1. **Plug-and-Play**: Switch between different LLM backends (Groq, Google Gemini, or others) by simply updating the `llm.py` or `llm_groq.py` references.
2. **Async Content Fetching**: Scrape the web in parallel with minimal blocking, thanks to Python’s `asyncio`.
3. **Research Planning**: An LLM-based planner determines how many search queries to run, how many additional iterations are needed, and when to finalize the report.
4. **Modular Design**: Each component is self-contained, making it easy to maintain, upgrade, or replace.
5. **CLI Terminal Integration**: Launch a terminal app (`run.py`) that guides you step-by-step, and displays final results with beautiful Markdown formatting via `rich`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Harshkamdar67/open-deep-research.git
   cd open-deep-research
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: . venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Configure environment variables for your LLM API keys:
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key"
   export GROQ_API_KEY="your_groq_api_key"
   ```

## Usage

1. **Basic Terminal App**

   ```bash
   python run.py
   ```

   - Enter your research query.
   - The application will run multiple research steps and produce a final Markdown report.
2. **Programmatic Usage**

   - Import the relevant classes from `deep_research` and integrate them in your Python application:
     ```python
     from deep_research.research import DeepResearch
     from deep_research.llm import LLMClient
     from deep_research.search import WebSearch
     from deep_research.content_fetcher import ContentFetcher

     # Initialize clients
     llm = LLMClient(api_key="your_api_key")
     search = WebSearch(max_results=5)
     fetcher = ContentFetcher()

     # Orchestrate research
     deep_researcher = DeepResearch(llm, search, fetcher)
     result = await deep_researcher.deep_research("Your query here")

     final_report = await deep_researcher.write_final_report(
         original_query="Your query here",
         learnings=result["learnings"],
         visited_urls=result["visited_urls"]
     )
     print(final_report)
     ```

## Repository Structure

```
.
├── README.md
├── deep_research
│   ├── __init__.py
│   ├── content_fetcher.py
│   ├── llm.py
│   ├── llm_groq.py
│   ├── research.py
│   ├── run.py
│   ├── search.py
│   └── text_processing.py
├── requirements.txt
```

### `deep_research` Directory

- **`__init__.py`**: Makes the directory a Python package.
- **`content_fetcher.py`**: Contains `ContentFetcher` class to asynchronously fetch content from multiple URLs.
- **`llm.py`**: Generic LLM client for integration with the Gemini API.
- **`llm_groq.py`**: LLM client for the Groq API.
- **`research.py`**: Core orchestrator that combines the LLM and web search to iteratively gather info.
- **`run.py`**: Terminal app that lets you run the research process interactively.
- **`search.py`**: DuckDuckGo-based web search utility.
- **`text_processing.py`**: Text splitting and token trimming functions.

### `requirements.txt`

Lists necessary Python packages, including `requests`, `duckduckgo_search`, `rich`, `dotenv`, etc.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have:

- Found a bug
- Have a feature request
- Want to improve documentation

### Steps to Contribute

1. Fork the repo.
2. Create a new branch with your changes (`git checkout -b feature-branch`).
3. Commit and push your changes.
4. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute this software as stated in the LICENSE.

Happy researching! If you find this helpful, consider giving this repo a star to show your support.
