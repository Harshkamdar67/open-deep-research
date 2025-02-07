import asyncio
import json
import logging
import re
from typing import List, Dict, Optional, Any
from search import WebSearch
from llm import LLMClient
from content_fetcher import ContentFetcher
from text_processing import trim_prompt 
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def system_prompt() -> str:
    """
    Returns the system prompt string.
    """
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    return (
        f"You are an expert researcher. Today is {now}. Follow these instructions when responding:\n"
        "- Assume that the user is a highly experienced analyst.\n"
        "- Be as detailed, organized, and accurate as possible.\n"
        "- Always keep the original user query in context and do not deviate from it.\n"
        "- Use all previously gathered learnings to decide if further research is needed.\n"
        "- If you have gathered enough relevant information to fully answer the original query, set the depth to 0 and summarize the key findings.\n"
        "- Provide clear recommendations and structured outputs.\n"
    )

def extract_json_from_text(text: str) -> str:
        """
        Extracts the first JSON object found in the text.
        """
        json_regex = r"(\{(?:.|\n)*\})"
        match = re.search(json_regex, text)
        if match:
            return match.group(1).strip()
        return text.strip()
    
def extract_json_from_code_block(text: str) -> str:
    """
    Removes surrounding triple backticks (```), including optional language indicators.
    """
    cleaned = re.sub(r"^```\w*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return cleaned

class DeepResearch:
    """
    Orchestrates 'deep research' using an LLM for query generation,
    a web search client for SERP fetching, and a content fetcher.
    
    In each iteration, the LLM is given:
      - The original user query.
      - All previously gathered learnings.
    It is instructed to decide:
      - How many search queries (breadth) to run.
      - How many additional iterations (depth) are needed.
    If sufficient information has been gathered, it should set depth to 0.
    """

    def __init__(
        self,
        llm: LLMClient,
        web_search: WebSearch,
        content_fetcher: ContentFetcher,
        concurrency_limit: int = 2,
        verbose: bool = True,
        max_iterations: int = 10,
    ):
        self.llm = llm
        self.web_search = web_search
        self.content_fetcher = content_fetcher
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.verbose = verbose
        self.max_iterations = max_iterations

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    async def ask_llm_for_research_plan(
        self,
        original_query: str,
        learnings: List[str],
    ) -> Dict[str, Any]:
        """
        Ask the LLM for a research plan based on the original user query and previous learnings.
        Expected JSON:
        {
          "breadth": number,
          "depth": number,
          "queries": [
            {"query": "<SERP query>", "researchGoal": "<text>"},
            ...
          ]
        }
        If sufficient information has been gathered, the LLM should set depth to 0.
        """
        learnings_str = "\n".join(f"- {lrn}" for lrn in learnings) if learnings else "No prior learnings."
        prompt_text = (
            "You are deciding how to conduct further research.\n\n"
            f"Original query: {original_query}\n\n"
            f"Learnings from previous research steps:\n{learnings_str}\n\n"
            "Based on the original query and the learnings so far, determine how many new SERP queries to run "
            "and how many additional research iterations are needed. If you believe that sufficient information "
            "has been gathered to produce a final report that directly answers the original query, set depth to 0. "
            "Return your answer in valid JSON with the following structure:\n\n"
            "{\n"
            '  "breadth": <number of SERP queries to run this iteration>,\n'
            '  "depth": <number of additional iterations needed; set 0 if research is complete>,\n'
            '  "queries": [\n'
            '    {"query": "<SERP query>", "researchGoal": "<explain what this query should achieve>"},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Do not deviate from the original query and only request further research if it will add value."
        )

        if self.verbose:
            logger.debug("[ask_llm_for_research_plan] Sending to LLM:\n%s", prompt_text)

        response_data = self.llm.generate(
            system_prompt=system_prompt(),
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=1024,
            temperature=0.6
        )

        if not response_data.get("success"):
            logger.error("LLM failed to produce a research plan: %s", response_data.get("error"))
            return {"breadth": 0, "depth": 0, "queries": []}

        raw_response = response_data["response"].strip()
        if self.verbose:
            logger.debug("[ask_llm_for_research_plan] LLM raw response:\n%s", raw_response)

        cleaned_response = extract_json_from_code_block(raw_response)
        try:
            plan = json.loads(cleaned_response)
            # Validate minimal structure
            breadth = plan.get("breadth", 0)
            depth = plan.get("depth", 0)
            queries = plan.get("queries", [])
            if not isinstance(breadth, int) or not isinstance(depth, int) or not isinstance(queries, list):
                raise ValueError("Invalid data types in JSON plan.")
            return plan
        except Exception as e:
            logger.error("[ask_llm_for_research_plan] JSON parse error: %s\nCleaned response was: %s", e, cleaned_response)
            return {"breadth": 0, "depth": 0, "queries": []}

    async def process_serp_result(
        self,
        query: str,
        serp_contents: List[str],
    ) -> Dict[str, List[str]]:
        """
        Process SERP results by having the LLM extract key learnings and follow-up questions.
        Returns a dict:
        {
          "learnings": [...],
          "followUpQuestions": [...]
        }
        """
        contents_str = ""
        for content in serp_contents:
            chunk = trim_prompt(content, context_size=25000)
            contents_str += f"<content>\n{chunk}\n</content>\n"

        prompt_text = (
            "We have the following SERP results for this query:\n"
            f"<query>{query}</query>\n\n"
            f"<contents>\n{contents_str}</contents>\n\n"
            "Based on these contents, provide a JSON object with two arrays: 'learnings' and 'followUpQuestions'.\n"
            "The 'learnings' should contain the key insights from these results, and the 'followUpQuestions' "
            "should suggest further questions to clarify or expand on the original query if needed.\n"
            "Return valid JSON, for example:\n"
            "{\n"
            '  "learnings": ["...", "..."],\n'
            '  "followUpQuestions": ["...", "..."]\n'
            "}"
        )

        if self.verbose:
            logger.debug("[process_serp_result] Prompt to LLM:\n%s", prompt_text)

        response_data = self.llm.generate(
            system_prompt=system_prompt(),
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=2048,
            temperature=0.6
        )

        if not response_data.get("success"):
            logger.error("Failed to process SERP result: %s", response_data.get("error"))
            return {"learnings": [], "followUpQuestions": []}

        raw_response = response_data["response"].strip()
        if self.verbose:
            logger.debug("[process_serp_result] LLM raw response:\n%s", raw_response)

        cleaned_response = extract_json_from_code_block(raw_response)
        try:
            parsed = json.loads(cleaned_response)
            return {
                "learnings": parsed.get("learnings", []),
                "followUpQuestions": parsed.get("followUpQuestions", []),
            }
        except Exception as e:
            logger.error("[process_serp_result] JSON parse error: %s\nCleaned response was: %s", e, cleaned_response)
            return {"learnings": [], "followUpQuestions": []}
        
    

    async def write_final_report(
        self,
        original_query: str,
        learnings: List[str],
        visited_urls: List[str]
    ) -> str:
        """
        Ask the LLM to write a final detailed report based on the original query and the aggregated learnings.
        Returns the report in Markdown, with a Sources section appended.
        """
        learnings_str = "\n".join(f"<learning>\n{lrn}\n</learning>" for lrn in learnings)
        learnings_str = trim_prompt(learnings_str, context_size=150000)

        prompt_text = (
    "We have completed our deep research. Please respond to the user based on the original query and the compiled learnings.\n\n"
    "1. **If the user is asking a direct question or wants a brief answer**, provide a concise, clear response first.\n"
    "2. **If the user is asking for a detailed or final report**, produce a structured, in-depth report in proper Markdown format.\n"
    "   - The report should comprehensively address the original query.\n"
    "   - Incorporate all relevant insights from the compiled learnings.\n"
    "   - Include recommendations, key findings, and any other relevant details.\n\n"
    "3. **If the user's request is unclear**, politely ask for clarification.\n\n"
    "Include these elements in your response:\n"
    "- Directly answer any user question (if one was asked).\n"
    "- A structured summary or final report (if requested), with headings, bullet points, or tables as needed.\n"
    "- A final summary of key findings and recommendations, where applicable.\n\n"
    "Always return your final output in valid JSON under the key \"reportMarkdown\". For example:\n"
    "```json\n"
    "{\n"
    "  \"reportMarkdown\": \"Your answer or report in Markdown here\"\n"
    "}\n"
    "```\n\n"
    f"**Original query:** {original_query}\n\n"
    "### Compiled Learnings\n"
    f"<learnings>\n{learnings_str}\n</learnings>\n\n"
    "Be sure to use all relevant learnings and, if needed, clearly state any assumptions or remaining questions."
)


        if self.verbose:
            logger.debug("[write_final_report] Prompt to LLM:\n%s", prompt_text)

        response_data = self.llm.generate(
            system_prompt=system_prompt(),
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=10000,
            temperature=0.6
        )

        if not response_data.get("success"):
            logger.error("Failed to write final report: %s", response_data.get("error"))
            return "Error: LLM did not produce a final report."

        raw_response = response_data["response"].strip()
        if self.verbose:
            logger.debug("[write_final_report] LLM raw response:\n%s", raw_response)

        # Use the improved extraction function.
        cleaned_response = extract_json_from_text(raw_response)
        try:
            parsed = json.loads(cleaned_response)
            report_markdown = parsed.get("reportMarkdown", "")
            if visited_urls:
                sources_section = "\n\n## Sources\n" + "\n".join(f"- {u}" for u in visited_urls)
                report_markdown += sources_section
            return report_markdown
        except Exception as e:
            logger.error("[write_final_report] JSON parse error: %s\nCleaned response was: %s", e, cleaned_response)
            return cleaned_response

    async def deep_research(
        self,
        original_query: str,
        initial_learnings: Optional[List[str]] = None,
        initial_visited_urls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for the deep research workflow.
        The LLM is given the original query and all previously gathered learnings in each iteration.
        It should decide how many new SERP queries to run (breadth) and how many additional iterations (depth) are needed.
        If it has enough information to answer the query, it should set depth to 0.
        """
        learnings = initial_learnings[:] if initial_learnings else []
        visited_urls = initial_visited_urls[:] if initial_visited_urls else []
        iteration_count = 0

        while iteration_count < self.max_iterations:
            iteration_count += 1
            if self.verbose:
                logger.debug(f"--- Iteration {iteration_count} ---")
                logger.debug(f"Original query: {original_query}")
                logger.debug(f"Current learnings ({len(learnings)}): {learnings}")
                logger.debug(f"Visited URLs so far ({len(visited_urls)}): {visited_urls}")

            # Ask the LLM for a research plan, always including the original query
            plan = await self.ask_llm_for_research_plan(original_query, learnings)
            breadth = plan.get("breadth", 0)
            depth = plan.get("depth", 0)
            serp_queries = plan.get("queries", [])

            if self.verbose:
                logger.info(f"[deep_research] LLM says: breadth={breadth}, depth={depth}, #queries={len(serp_queries)}")

            # Stop if the LLM indicates no further research is needed.
            if breadth <= 0 or depth <= 0 or len(serp_queries) == 0:
                if self.verbose:
                    logger.info("[deep_research] LLM indicated no further research is needed. Stopping.")
                break

            # Limit the queries to the given breadth
            serp_queries = serp_queries[:breadth]

            async def run_single_query(qdict: Dict[str, str]) -> Dict[str, Any]:
                qtext = qdict.get("query", "")
                logger.debug(f"[run_single_query] Searching for: {qtext}")

                async with self.semaphore:
                    results = self.web_search.search(qtext, max_results=breadth)
                new_urls = [r["url"] for r in results if r.get("url")]

                logger.debug(f"[run_single_query] Found {len(new_urls)} URLs")

                contents = []
                if new_urls:
                    try:
                        async with self.semaphore:
                            fetched = await self.content_fetcher.fetch_content(new_urls)
                        for cd in fetched:
                            contents.append(cd["content"])
                    except Exception as e:
                        logger.error(f"[run_single_query] Error fetching content for {qtext}: {e}")

                results_dict = await self.process_serp_result(qtext, contents)
                return {
                    "urls": new_urls,
                    "learnings": results_dict["learnings"],
                    "followUps": results_dict["followUpQuestions"],
                }

            tasks = [run_single_query(qdict) for qdict in serp_queries]
            step_results = await asyncio.gather(*tasks)

            step_learnings = []
            step_urls = []

            for sr in step_results:
                step_learnings.extend(sr["learnings"])
                step_urls.extend(sr["urls"])

            # Deduplicate
            learnings = list(dict.fromkeys(learnings + step_learnings))
            visited_urls = list(dict.fromkeys(visited_urls + step_urls))

            if self.verbose:
                logger.debug(f"[deep_research] Iteration {iteration_count} done. Total learnings: {len(learnings)}; Total URLs: {len(visited_urls)}.")

            if depth <= 1:
                if self.verbose:
                    logger.info("[deep_research] LLM indicated final iteration. Stopping.")
                break

        return {
            "learnings": learnings,
            "visited_urls": visited_urls,
            "iterations": iteration_count,
        }