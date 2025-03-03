from abc import ABC, abstractmethod

from app.pkg.config import Config
from app.pkg.entities.models import Dialog, AgentInstance
from app.pkg.event_router import EventRouter
from app.pkg.llm.base_llm import BaseLLM
from app.pkg.vector_store.vector_store import VectorStore


class BaseAgent(ABC):
    def __init__(self, agent_instance: AgentInstance, cfg: Config, vs: VectorStore, llm: BaseLLM):
        self.cfg = cfg
        self.logger = cfg.logger(self.__class__.__name__)
        self.agent_instance = agent_instance
        self.vs = vs
        self.llm = llm
        self.er = EventRouter(cfg)

    @abstractmethod
    async def build_asnwer(self, dialog: Dialog, question: str, media: list | None = None) -> tuple[str, list[str], dict]:
        pass

    @abstractmethod
    async def create_cloud(self) -> str:
        pass

    @abstractmethod
    async def create_thread(self, dialog_id: int, init_messages: list = None) -> str | None:
        pass

    @abstractmethod
    async def delete_thread(self, dialog_id: int, thread_id: str | None) -> bool:
        pass
