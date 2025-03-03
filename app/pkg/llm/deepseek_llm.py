from pydantic import BaseModel

from app.pkg.llm.base_llm import BaseLLM


class DeepSeekLLM(BaseLLM):
    async def embed_documents(self, texts: list[str], dimensions: int = None) -> tuple[list[list[float]], float]:
        """
        Пока не умеет
        """
        pass

    async def embed_query(self, text: str, dimensions: int = None) -> tuple[list[float], float]:
        """
        Пока не умеет
        """
        pass

    async def invoke(self, prompt: str, intruction: str = None, response_format: dict = None, messages: list[dict] = None, **kwargs) -> dict | BaseModel:
        pass
