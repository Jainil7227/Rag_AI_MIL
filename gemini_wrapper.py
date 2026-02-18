import google.genai as genai
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiWrapper:
    """
    Wrapper for Google's Gemini AI API designed for RAG systems.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        verbose: bool = True
    ):
        """
        Initialize the Gemini wrapper.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env variable
            model_name: Gemini model to use
            temperature: Response randomness
            verbose: Whether to print initialization messages
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "No Gemini API key provided.\n"
                "Either pass api_key parameter or set GEMINI_API_KEY environment variable."
            )

        self.client = genai.Client(api_key=self.api_key)

        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose

        self.history = []
        self.persona = None

        if self.verbose:
            print(f"Gemini initialized: {model_name} (temp={temperature})")

    def set_persona(self, persona_description: str) -> None:
        """
        Set the AI's system persona/role.
        """
        self.persona = persona_description
        if self.verbose:
            preview = persona_description[:80] + "..." if len(persona_description) > 80 else persona_description
            print(f"Persona set: {preview}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate a response from Gemini.
        """
        full_prompt = f"SYSTEM: {self.persona}\n\nUSER: {prompt}" if self.persona else prompt
        temp = temperature if temperature is not None else self.temperature

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config={
                    "temperature": temp,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                },
            )

            text = ""
            if hasattr(resp, "text") and isinstance(resp.text, str):
                text = resp.text
            elif hasattr(resp, "candidates") and resp.candidates:
                for cand in resp.candidates:
                    if getattr(cand, "content", None):
                        parts = getattr(cand.content, "parts", [])
                        for p in parts:
                            if getattr(p, "text", None):
                                text += p.text
                text = text.strip()

            self.history.append({
                'prompt': prompt,
                'response': text,
                'temperature': temp,
                'model': self.model_name
            })

            return text or ""
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            if self.verbose:
                print(f"Error: {error_msg}")
            return error_msg

    def chat(self, message: str) -> str:
        """
        Simple chat using a running transcript.
        """
        if not hasattr(self, '_chat_transcript'):
            self._chat_transcript = []

        self._chat_transcript.append({"role": "user", "text": message})
        
        convo = []
        if self.persona:
            convo.append(f"SYSTEM: {self.persona}")
        for turn in self._chat_transcript[-10:]:
            prefix = "USER" if turn["role"] == "user" else "ASSISTANT"
            convo.append(f"{prefix}: {turn['text']}")
        convo.append("ASSISTANT:")
        prompt = "\n\n".join(convo)

        reply = self.generate(prompt)
        self._chat_transcript.append({"role": "assistant", "text": reply})
        return reply

    def clear_history(self) -> None:
        """
        Clear conversation history.
        """
        self.history = []
        if hasattr(self, '_chat_transcript'):
            self._chat_transcript = []
        if self.verbose:
            print("History cleared")

    def get_history(self) -> List[Dict]:
        """
        Get the conversation history.
        """
        return self.history

    def get_stats(self) -> Dict:
        """
        Get wrapper statistics.
        """
        return {
            'model': self.model_name,
            'temperature': self.temperature,
            'total_interactions': len(self.history),
            'has_persona': self.persona is not None
        }


def demo():
    """Run a simple demo of the Gemini wrapper."""
    print("\n=== GEMINI WRAPPER DEMO ===\n")

    try:
        llm = GeminiWrapper(temperature=0.7)

        print("1. Basic Generation")
        response = llm.generate("What is Python in one sentence?")
        print(f"Q: What is Python in one sentence?")
        print(f"A: {response}\n")

        print("2. With Persona")
        llm.set_persona("You are a helpful teacher who explains concepts using simple analogies.")
        response = llm.generate("What is machine learning?")
        print(f"Q: What is machine learning?")
        print(f"A: {response}\n")

    except ValueError as e:
        print(f"\nError: {e}\n")


if __name__ == "__main__":
    demo()