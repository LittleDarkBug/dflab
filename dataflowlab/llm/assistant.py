from typing import Any, Dict, Optional
from llama_cpp import Llama
from dataflowlab.utils.logger import get_logger

class LLMAssistant:
    """
    Assistant LLM local basé sur llama-cpp-python.
    """
    def __init__(self, model_path: str) -> None:
        self.logger = get_logger("LLMAssistant")
        self.model_path = model_path
        self.llm = Llama(model_path=model_path, n_ctx=2048)

    def ask(self, prompt: str, context: Optional[str] = None) -> str:
        try:
            full_prompt = prompt if context is None else f"{context}\n{prompt}"
            response = self.llm(full_prompt, max_tokens=256)
            return response["choices"][0]["text"].strip()
        except Exception as e:
            self.logger.error(f"Erreur LLM: {e}")
            return "Erreur lors de la génération de la réponse LLM."
