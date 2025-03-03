import ssl
from datetime import datetime

from app.pkg.agents import create_agent
from app.pkg.agents.base_agent import BaseAgent
from app.pkg.backend_client import BackendClient
from app.pkg.config import Config
from app.pkg.db import Database
from app.pkg.dialogs_manager import DialogsManager
from app.pkg.entities.api import OpenDialogRequest
from app.pkg.entities.models import Message, MSG_SENDER_USER, AgentInstance
from app.pkg.event_router import EventRouter
from app.pkg.llm import create_llm
from app.pkg.llm.base_llm import BaseLLM
from app.pkg.message_processor import MessageProcessor
from app.pkg.vector_store import create_vector_store
from app.pkg.vector_store.vector_store import VectorStore
from app.pkg.websocket import WebSocket

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


class DialogAgent:
    agent_instance: AgentInstance
    llm: BaseLLM
    vector_store: VectorStore
    ws: WebSocket
    bc: BackendClient
    db: Database
    dialogs_manager: DialogsManager
    message_processor: MessageProcessor
    event_router: EventRouter
    agent: BaseAgent

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = cfg.logger("agent_instance")

        self.reload_state = "loaded"

    async def reload_dataset(self, dataset: list[dict] | str | list[str] | None = None, group: str | None = None):
        self.reload_state = "in_progress"

        dataset = await self.bc.get_dataset() if not dataset else dataset

        try:
            await self.vector_store.build_dataset(dataset, group)
            self.agent_instance.config["vs_loaded"] = True
            await self.db.update_agent_instance(self.agent_instance.id, config=self.agent_instance.config)
            self.reload_state = "loaded"
        except Exception as e:
            self.logger.error(f"Ошибка при перезагрузке датасета: {e}")
            self.reload_state = "error"

        # Уведомляем внешний сервис о состоянии перезагрузки
        await self.bc.notify_reload_state(self.reload_state)
        self.logger.info("Датасет успешно перезагружен.")

    async def load(self):
        self.db = Database(
            logger=self.cfg.logger("db"),
            url=self.cfg.db.url,
            name=self.cfg.db.name,
            pool_size=self.cfg.db.pool_size,
            max_overflow=self.cfg.db.max_overflow,
            pool_timeout=self.cfg.db.pool_timeout,
            pool_recycle=self.cfg.db.pool_recycle,
        )

        # Инициализация клиентов для внешних API
        self.ws = WebSocket(self.cfg)
        self.bc = BackendClient(self.cfg)

        # Инициализация агента
        self.agent_instance = await self.db.get_agent_instance(self.cfg.agent.id)
        if not self.agent_instance:
            raise Exception(f"Агент с id={self.cfg.agent.id} не найден.")
        self.llm = create_llm(self.cfg, self.agent_instance.type)
        self.vector_store = create_vector_store(cfg=self.cfg, llm=self.llm, preset=self.agent_instance.config.get("vs_preset"))
        self.agent = create_agent(self.agent_instance, self.cfg, self.vector_store, self.llm)

        self.agent_instance.cloud_id = await self.agent.create_cloud()
        await self.db.update_agent_instance(self.agent_instance.id, cloud_id=self.agent_instance.cloud_id)

        # Создаем менеджеры для разделения логики
        self.dialogs_manager = DialogsManager(self.db, self.agent, self.logger)
        self.message_processor = MessageProcessor(self.cfg, self.agent, self.db, self.bc, self.ws)
        self.event_router = EventRouter(self.cfg)

        await self.dialogs_manager.load_active()

    async def unload(self):
        await self.dialogs_manager.update_active()

    async def add_new_message(self, external_id: int, message: Message, is_async: bool = True) -> dict:
        dialog = self.dialogs_manager.get_dialog(external_id)
        if not dialog:
            raise Exception(f"Диалог с external_id={external_id} не найден.")

        dialog.queue.append(message)

        message = await self.db.create_message(
            Message(
                dialog_id=dialog.id,
                sender=MSG_SENDER_USER,
                text=message.text,
                media=message.media,
                created_at=datetime.now(),
            )
        )

        if is_async:
            await self.message_processor.trigger_processing(dialog)
            return {"id": message.id}

        return await self.message_processor.process_messages_sync(dialog)

    async def close_dialog(self, external_id: int):
        await self.dialogs_manager.close_dialog(external_id)

    async def update_dialog(self, external_id: int, user_id: int):
        await self.dialogs_manager.update_dialog(external_id, user_id)

    async def open_dialog(self, external_id: int, request: OpenDialogRequest):
        dialog = await self.dialogs_manager.open_dialog(external_id, request.user_id, request.user_type, request.meta)
        self.logger.info(f"Открыт диалог: {external_id} -> {dialog.id}")
        return dialog

    async def process_typing(self, user_id: int, user_type: str):
        dialog_id = self.dialogs_manager.get_dialog_id_by_user(user_id, user_type)
        if dialog_id:
            dialog = self.dialogs_manager.get_dialog(dialog_id)
            if dialog:
                await self.message_processor.trigger_processing(dialog)

    async def create_cloud(self):
        cloud_id = await self.agent.create_cloud()
        if cloud_id != self.agent_instance.cloud_id:
            await self.db.update_agent_instance(self.agent_instance.id, cloud_id=self.agent_instance.cloud_id)

        self.agent_instance = await self.db.get_agent_instance(self.cfg.agent.id)
