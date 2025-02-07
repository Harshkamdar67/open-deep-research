import os
import time
from typing import List, Dict, Any
from google import genai  
from dotenv import load_dotenv

class LLMClient:
    """
    A reusable client for interacting with the Gemini API.
    """

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        """
        Initialize the LLM client with the API key and model selection.
        The API key is read from the GEMINI_API_KEY environment variable if not provided.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("Gemini API key is required.")

        self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024, 
        temperature: float = 0.6,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a structured response from the Gemini API.

        Parameters:
            system_prompt (str): The system instructions for the LLM.
            messages (List[Dict[str, str]]): User and assistant messages in structured format.
            max_tokens (int): Maximum tokens for the response.
            temperature (float): Controls randomness.
            retries (int): Number of retry attempts.

        Returns:
            Dict[str, Any]: Structured output from the model.
        """
        user_contents = "\n".join(m["content"] for m in messages if m["role"] == "user")

        config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "system_instruction": system_prompt
        }

        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_contents,
                    config=config
                )
                return {
                    "success": True,
                    "response": response.text, 
                    "usage": {}  
                }
            except Exception as e:
                print(f"LLM API request failed: {e}. Retrying ({attempt+1}/{retries})...")
                time.sleep(2)

        return {"success": False, "error": "LLM API request failed after retries"}

