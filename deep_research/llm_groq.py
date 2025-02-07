import os
import time
from typing import List, Dict, Any
from groq import Groq

class LLMClient:
    """
    A reusable client for interacting with the Groq API.
    """

    def __init__(self, api_key: str = None, model: str = "deepseek-r1-distill-llama-70b"):
        """
        Initialize the LLM client with API key and model selection.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("Groq API key is required.")

        self.client = Groq(api_key=self.api_key)

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.6,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a structured response from the LLM.

        Parameters:
            system_prompt (str): The system instructions for the LLM.
            messages (List[Dict[str, str]]): User and assistant messages in structured format.
            max_tokens (int): Maximum tokens for response.
            temperature (float): Controls randomness.
            retries (int): Number of retry attempts.

        Returns:
            Dict[str, Any]: Structured output from the model.
        """

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    # reasoning_format="hidden"
                )

                return {
                    "success": True,
                    "response": response.choices[0].message.content if response.choices else "",
                    "usage": response.usage if hasattr(response, "usage") else {},
                }

            except Exception as e:
                print(f"LLM API request failed: {e}. Retrying ({attempt+1}/{retries})...")
                time.sleep(2)

        return {"success": False, "error": "LLM API request failed after retries"}

# Example Usage
if __name__ == "__main__":
    llm = LLMClient()
    result = llm.generate(
        system_prompt="You are an AI assistant that provides structured research outputs.",
        messages=[{"role": "user", "content": "Explain quantum computing."}]
    )
    print(result)
