import logging
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class ServerConfig(BaseModel):
    api_key: str
    debug: bool = False
    port: int


class DBConfig(BaseModel):
    url: str
    sync_url: str
    name: str
    pool_size: int = 20
    max_overflow: int = 5
    pool_timeout: int = 5
    pool_recycle: int = 30
    debug: bool = False


class OpenAIConfig(BaseModel):
    api_key: str
    model: str
    emb_model: str
    emb_dimensions: int | None = None
    debug: bool = False

class WebSocketConfig(BaseModel):
    """
    Связь для приложения naimi.kz
    """
    debug: Optional[bool] = False
    api_key: Optional[str]
    url: Optional[str]


class AgentConfig(BaseModel):
    id: int
    debug: bool = False
    callback_url: Optional[str]
    dataset_url: Optional[str]


class AppConfigSchema(BaseModel):
    server: ServerConfig
    db: DBConfig
    openai: OpenAIConfig
    ws: WebSocketConfig
    agent: AgentConfig


class Config:
    """
    Класс для управления конфигурацией приложения.
    Читает настройки из файла .env, если он существует, иначе использует переменные окружения.
    """

    def __init__(self):
        """
        Инициализация конфигурации приложения.
        """
        self.server: ServerConfig
        self.db: DBConfig
        self.openai: OpenAIConfig
        self.ws: WebSocketConfig
        self.agent: AgentConfig

        self._load_config()

    def _load_config(self):
        """
        Приватный метод для загрузки конфигурации.
        Сначала проверяет наличие файла конфигурации, и, если он найден, загружает из него настройки.
        В противном случае использует переменные окружения.
        """

        load_dotenv()

        raw_config = {
            "server": {
                "api_key": os.environ.get("SERVER_API_KEY"),
                "debug": bool(int(os.environ.get("SERVER_DEBUG", 1))),
                "port": int(os.environ.get("SERVER_PORT", 8000)),
            },
            "db": {
                "url": os.environ.get("DB_URL"),
                "sync_url": os.environ.get("DB_URL").replace("asyncpg", "psycopg"),
                "name": os.environ.get("DB_NAME"),
                "pool_size": int(os.environ.get("DB_POOL_SIZE", 20)),
                "max_overflow": int(os.environ.get("DB_MAX_OVERFLOW", 5)),
                "pool_timeout": int(os.environ.get("DB_POOL_TIMEOUT", 5)),
                "pool_recycle": int(os.environ.get("DB_POOL_RECYCLE", 30)),
                "debug": bool(int(os.environ.get("DB_DEBUG", 1))),
            },
            "openai": {
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "model": os.environ.get("OPENAI_MODEL"),
                "emb_model": os.environ.get("OPENAI_EMB_MODEL"),
                "emb_dimensions": int(os.environ.get("OPENAI_EMB_DIMENSIONS")) if os.environ.get("OPENAI_EMB_DIMENSIONS") else None,
                "debug": bool(int(os.environ.get("OPENAI_DEBUG", 1))),
            },
            "ws": {
                "debug": bool(int(os.environ.get("WS_DEBUG", 1))),
                "api_key": os.environ.get("WS_API_KEY"),
                "url": os.environ.get("WS_URL"),
            },
            "agent": {
                "id": os.environ.get("AGENT_ID"),
                "debug": bool(int(os.environ.get("AGENT_DEBUG", 1))),
                "callback_url": os.environ.get("AGENT_CALLBACK_URL"),
                "dataset_url": os.environ.get("AGENT_DATASET_URL"),
            },
        }

        try:
            validated_config = AppConfigSchema(**raw_config)
        except ValidationError as ve:
            logging.error(f"Конфигурация некорректна: {ve}")
            raise ValueError(f"Конфигурация некорректна: {ve}")

        self.server = validated_config.server
        self.db = validated_config.db
        self.openai = validated_config.openai
        self.ws = validated_config.ws
        self.agent = validated_config.agent

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def model_dump(self):
        """
        Возвращает словарь с конфигурацией.
        """

        return {
            "server": self.server.model_dump(),
            "db": self.db.model_dump(),
            "openai": self.openai.model_dump(),
            "ws": self.ws.model_dump(),
            "agent_instance": self.agent.model_dump(),
        }

    def logger(self, name: str) -> logging.Logger:
        """
        Создает логгер с указанным именем.
        """

        log = logging.getLogger(name)

        if not hasattr(self, name):
            log.setLevel(logging.DEBUG if self.__getattribute__("server").debug else logging.INFO)
        else:
            log.setLevel(logging.DEBUG if self.__getattribute__(name).debug else logging.INFO)

        return log
