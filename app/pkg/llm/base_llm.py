import json
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from app.pkg.config import Config

ROLE_USER = "user"
ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"


class BaseResult(ABC):
    def __init__(self, content: str = None, usage: dict = None, **kwargs):
        self.content = content
        self.usage = usage
        self.kwargs = kwargs

        try:
            self.json = json.loads(content) if content else {}
        except:
            self.json = {}

        self.price = 0.0  # TODO Добавить расчет стоимости


class BaseLLM(ABC):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embeddings: Embeddings | None = None
        self.llm: BaseChatModel | None = None

    @abstractmethod
    async def invoke(self, prompt: str, intruction: str = None, response_format: dict = None, messages: list[dict] = None, **kwargs) -> dict | BaseModel:
        """
        Генерирует ответ на основе переданного промпта.
        :param prompt: Текст запроса, будет как текст User.
        :param intruction: Текст инструкции.
        :param response_format: Формат возвращаемого ответа.
        :param messages: Список предидущих сообщений.
        :param kwargs: Дополнительные параметры для разных LLM.
        :return: Сгенерированный ответ.
        """
        pass

    @abstractmethod
    async def embed_query(self, text: str, dimensions: int = None) -> tuple[list[float], float]:
        """
        Возвращает векторное представление текста.
        :param text: Текст для векторизации.
        :param dimensions: Размерность вектора.
        :return: Векторное представление текста.
        """
        pass

    @abstractmethod
    async def embed_documents(self, texts: list[str], dimensions: int = None) -> tuple[list[list[float]], float]:
        """
        Возвращает векторное представление текста.
        :param texts: Список текстов для векторизации.
        :param dimensions: Размерность вектора.
        :return: Векторное представление текста.
        """
        pass



    @abstractmethod
    async def speech_to_text(self, audio_path: str, is_local_file: bool = False):
        """
        Конвертирует речь в текст.

        Args:
            audio_path (str): URL аудиофайла или путь к локальному файлу.
            is_local_file (bool): Если True, audio_path интерпретируется как локальный путь к файлу.

        Returns:
            str: Транскрибированный текст из аудио.
        """
