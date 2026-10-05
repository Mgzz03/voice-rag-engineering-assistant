"""
LLM client for interacting with local phi-3-mini-4k-instruct model.

Connects to the model running on http://172.31.16.1:1234
"""

import requests
from typing import Optional, Dict, Any


class LLMClient:
    """Client for local LLM inference."""

    def __init__(self, base_url: str = "http://172.31.16.1:1234"):
        """
        Initialize LLM client.

        Args:
            base_url: Base URL where the LLM is running
        """
        self.base_url = base_url.rstrip('/')
        self.completions_endpoint = f"{self.base_url}/v1/chat/completions"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt/question
            system_prompt: Optional system prompt to guide the model
            conversation_history: Optional list of previous messages [{"role": "user"/"assistant", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional parameters for the API

        Returns:
            Generated text response

        Raises:
            Exception: If the API request fails
        """
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": prompt})

        # Prepare request payload
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        # Make request
        try:
            response = requests.post(
                self.completions_endpoint,
                json=payload,
                timeout=600  # Increased timeout for local LLM
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Could not connect to LLM at {self.base_url}. "
                "Make sure the model is running."
            )
        except requests.exceptions.Timeout:
            raise Exception("LLM request timed out")
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")

    def test_connection(self) -> bool:
        """
        Test if the LLM is reachable and responding.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.generate("Hello", max_tokens=10)
            return len(response) > 0
        except:
            return False


def main():
    """Test the LLM client."""
    print("Testing LLM Client...")
    print("="*80)

    client = LLMClient()

    # Test connection
    print("\n1. Testing connection to http://172.31.16.1:1234...")
    if not client.test_connection():
        print("❌ Connection failed!")
        print("Make sure phi-3-mini-4k-instruct is running on http://172.31.16.1:1234")
        return

    print("✓ Connection successful!")

    # Test basic generation
    print("\n2. Testing basic generation...")
    response = client.generate(
        "What is 2+2?",
        system_prompt="You are a helpful assistant. Be concise.",
        max_tokens=50
    )
    print(f"Response: {response}")

    # Test with context
    print("\n3. Testing with context...")
    context = "The University offers engineering programs in CIE, ENV, and SPC."
    question = "What programs does the university offer?"

    response = client.generate(
        f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        system_prompt="Answer based only on the provided context.",
        max_tokens=100
    )
    print(f"Response: {response}")

    print("\n" + "="*80)
    print("LLM Client test complete!")


if __name__ == "__main__":
    main()
