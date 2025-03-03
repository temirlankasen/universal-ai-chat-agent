import asyncio
from datetime import datetime

from app.pkg.agents.base_agent import BaseAgent
from app.pkg.backend_client import BackendClient
from app.pkg.config import Config
from app.pkg.db import Database
from app.pkg.entities.models import Dialog
from app.pkg.entities.models import Message, MSG_SENDER_AGENT
from app.pkg.websocket import WebSocket


class MessageProcessor:
    def __init__(self, cfg: Config, agent: BaseAgent, db: Database, bc: BackendClient, ws: WebSocket):
        self.cfg = cfg
        self.db = db
        self.bc = bc
        self.agent = agent
        self.ws = ws
        self.logger = cfg.logger("message_processor")

    async def trigger_processing(self, dialog: Dialog):
        if not dialog.queue:
            return

        if dialog.timer and not dialog.timer.done() and not dialog.is_processing:
            dialog.timer.cancel()

        if not dialog.timer or dialog.timer.done():
            dialog.timer = asyncio.create_task(self._process_messages_after_delay(dialog, delay=5))
            dialog.timer.add_done_callback(lambda fut: asyncio.create_task(self.trigger_processing(dialog)))
            dialog.start_message_handle_at = datetime.now()

    async def _process_messages_after_delay(self, dialog: Dialog, delay: int):
        try:
            self.logger.debug(f"Запуск таймера обработки сообщений для external_id={dialog.external_id}")

            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.logger.debug(f"Таймер отменён для external_id={dialog.external_id}")
            return

        bot_is_typing = asyncio.create_task(self.ws.start_typing(dialog.user_id, dialog.user_type))

        dialog.is_processing = True

        question = ". ".join([message.text for message in dialog.queue])
        media = [message.media for message in dialog.queue if message.media]
        dialog.add_runtime_message(question)
        dialog.queue = []

        try:
            answer, actions, custom_data = await self.agent.build_asnwer(
                dialog,
                question,
                media,
            )
        except Exception as e:
            self.logger.error(f"Ошибка генерации ответа для external_id={dialog.external_id}: {e}")
            answer = ""
            actions = ["call_operator"]
            custom_data = {}

        await self.db.create_message(
            Message(
                dialog_id=dialog.id,
                sender=MSG_SENDER_AGENT,
                text=answer,
                media=None,
                meta={"actions": actions, "custom_data": custom_data, "start_at": dialog.start_message_handle_at.isoformat(), "end_at": datetime.now().isoformat()},
                created_at=datetime.now(),
            )
        )
        if custom_data.get("augmentations"):
            custom_data.pop("augmentations")

        bot_is_typing.cancel()

        await self.bc.send_answer(dialog.external_id, answer, actions, custom_data)

        dialog.is_processing = False

    async def process_messages_sync(self, dialog: Dialog):
        start_at = datetime.now()
        question = ". ".join([message.text for message in dialog.queue])
        media = [item for message in dialog.queue if message.media for item in message.media]
        dialog.add_runtime_message(question)
        dialog.queue = []
        try:
            answer, actions, custom_data = await self.agent.build_asnwer(
                dialog,
                question,
                media,
            )
        except Exception as e:
            self.logger.error(f"Ошибка генерации синхронного ответа для external_id={dialog.external_id}: {e}")
            answer = ""
            actions = ["call_operator"]
            custom_data = {}

        message = await self.db.create_message(
            Message(
                dialog_id=dialog.id,
                sender=MSG_SENDER_AGENT,
                text=answer,
                media=None,
                meta={"actions": actions, "custom_data": custom_data, "start_at": start_at.isoformat(), "end_at": datetime.now().isoformat()},
                created_at=datetime.now(),
            )
        )
        return message
