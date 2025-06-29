import pytest
from dataflowlab.llm.assistant import LLMAssistant

def test_llm_assistant_init(monkeypatch):
    class DummyLlama:
        def __init__(self, model_path, n_ctx):
            self.model_path = model_path
        def __call__(self, prompt, max_tokens):
            return {"choices": [{"text": "Réponse factice"}]}
    monkeypatch.setattr("dataflowlab.llm.assistant.Llama", DummyLlama)
    assistant = LLMAssistant(model_path="dummy.bin")
    response = assistant.ask("Test ?")
    assert "Réponse" in response
