import asyncio
import json
import os
import tempfile
from typing import List, Tuple, Literal

import aiohttp
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.pkg.config import Config
from app.pkg.llm.base_llm import BaseLLM, BaseResult


class OpenAIResult(BaseResult):
    """
    Конкретная реализация BaseResult для OpenAI.
    """

    def __init__(self, content: str = None, usage: dict = None, **kwargs):
        super().__init__(content=content, usage=usage, **kwargs)


class OpenAILLM(BaseLLM):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.embeddings = OpenAIEmbeddings(model=self.cfg.openai.emb_model, dimensions=self.cfg.openai.emb_dimensions)
        self.embeddings.openai_api_key = self.cfg.openai.api_key

        self.llm = ChatOpenAI(model=self.cfg.openai.model, api_key=self.cfg.openai.api_key)
        self.client = AsyncOpenAI(api_key=self.cfg.openai.api_key)

    async def invoke(self, prompt: str, intruction: str = None, response_format: dict = None, messages: List[dict] = None, **kwargs) -> dict | BaseModel:
        """
        Генерирует ответ, комбинируя instruction (если задана) с prompt.
        """
        messages = messages or []

        if intruction:
            messages.insert(0, {"role": "system", "content": intruction})

        llm = self.llm
        if kwargs.get("max_tokens"):
            llm.max_tokens = kwargs.get("max_tokens")
        if kwargs.get("temperature"):
            llm.temperature = kwargs.get("temperature")
        if kwargs.get("top_p"):
            llm.top_p = kwargs.get("top_p")

        if response_format:
            if isinstance(response_format, dict):
                if response_format.get("type") == "json_object":
                    llm = llm.bind(response_format={"type": "json_object"})
                    messages.append({"role": "system", "content": f"Дай ответ в JSON формате:\n\n```\n{json.dumps(response_format.get("json_object"))}\n```"})
                elif response_format.get("type") == "json_schema":
                    json_schema_candidate = response_format.get("json_schema")
                    # Проверяем, что переданное значение является классом и наследуется от BaseModel
                    if isinstance(json_schema_candidate, type) and issubclass(json_schema_candidate, BaseModel):
                        json_schema = json_schema_candidate
                        llm = llm.with_structured_output(json_schema)
                    else:
                        llm = llm.with_structured_output(json_schema_candidate)

        messages.append({"role": "human", "content": prompt})

        return await llm.ainvoke(messages)

    async def embed_query(self, text: str, dimensions: int = None) -> Tuple[List[float], float]:
        """
        Возвращает векторное представление текста.
        Для этого используем OpenAIEmbeddings из LangChain.
        """
        self.embeddings.dimensions = dimensions
        vector = await asyncio.to_thread(self.embeddings.embed_query, text)
        price = 0.0  # Здесь можно рассчитать стоимость, если нужно
        return vector, price

    async def embed_documents(self, texts: List[str], dimensions: int = None) -> Tuple[List[List[float]], float]:
        """
        Возвращает векторные представления для списка текстов.
        """
        self.embeddings.dimensions = dimensions
        vectors = await asyncio.to_thread(self.embeddings.embed_documents, texts, chunk_size=10)
        price = 0.0
        return vectors, price

    async def speech_to_text(self, audio_path: str, is_local_file: bool = False):
        """
        Конвертирует речь в текст с использованием OpenAI Whisper через LangChain.

        Args:
            audio_path (str): URL аудиофайла или путь к локальному файлу.
            is_local_file (bool): Если True, audio_path интерпретируется как локальный путь к файлу.

        Returns:
            str: Транскрибированный текст из аудио.
        """

        temp_filename = None

        try:
            if not is_local_file:
                # Скачиваем аудиофайл, если это URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(audio_path) as response:
                        if response.status != 200:
                            raise Exception(f"Не удалось скачать аудиофайл с {audio_path}")

                        # Создаем временный файл для хранения аудио
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
                            temp_filename = temp_file.name
                            # Записываем содержимое аудио во временный файл
                            temp_file.write(await response.read())
            else:
                # Если это локальный файл, просто используем путь
                temp_filename = audio_path

            with open(temp_filename, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

                transcribed_text = response.text

            return transcribed_text
        finally:
            # Удаляем временный файл только если мы его создали (не локальный файл)
            if temp_filename and not is_local_file and os.path.exists(temp_filename):
                os.remove(temp_filename)

    async def save_file(self, file, purpose: Literal["assistants", "batch", "fine-tune", "vision"] = "assistants", file_path: str = None) -> str | None:
        try:
            result = await self.client.files.create(
                file=file if file else open(file_path, "rb"),
                purpose=purpose
            )
            file_id = result.id
        except Exception as e:
            file_id = None

        return file_id
