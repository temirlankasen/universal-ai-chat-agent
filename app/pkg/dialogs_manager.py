from datetime import datetime
from logging import Logger

from app.pkg.agents.base_agent import BaseAgent
from app.pkg.db import Database
from app.pkg.entities.models import Dialog
from app.pkg.helper import json_dumps


class DialogsManager:
    def __init__(self, db: Database, agent: BaseAgent, logger: Logger):
        self.db = db
        self.agent = agent
        self.logger = logger
        # todo вывести в кэш для вертикального масштабирования
        self.dialogs: dict[int, Dialog] = {}  # external_id -> Dialog
        self.user_dialogs_map: dict[str, dict[int, int]] = {}  # user_type -> user_id -> external_id

    async def load_active(self):
        dialogs: list[Dialog] = await self.db.get_active_dialogs()
        for dialog in dialogs:
            self.dialogs[dialog.external_id] = dialog
            self.user_dialogs_map.setdefault(dialog.user_type, {})[dialog.user_id] = dialog.external_id
            self.logger.info(f"Загружен активный диалог: {json_dumps(dialog.to_dict())}")
        self.logger.info("Активные диалоги успешно загружены.")

    async def update_active(self):
        for external_id, dialog in self.dialogs.items():
            await self.db.update_dialog(dialog.id, price=dialog.price)

    def get_dialog(self, external_id: int):
        return self.dialogs.get(external_id)

    def get_dialog_id_by_user(self, user_id: int, user_type: str):
        return self.user_dialogs_map.get(user_type, {}).get(user_id)

    def add_dialog(self, dialog: Dialog):
        self.dialogs[dialog.external_id] = dialog
        self.user_dialogs_map.setdefault(dialog.user_type, {})[dialog.user_id] = dialog.external_id

    async def open_dialog(self, external_id: int, user_id: int, user_type: str, meta: dict) -> Dialog:
        thread_id = await self.agent.create_thread(external_id, meta.get("init_messages", []))
        dialog: Dialog = await self.db.save_dialog(external_id, self.agent.agent_instance.id, user_id, user_type, meta, datetime.now(), thread_id)
        self.add_dialog(dialog)
        self.logger.info(f"Открыт диалог: {dialog}")

        return dialog

    async def close_dialog(self, external_id: int):
        dialog: Dialog | None = self.dialogs.pop(external_id, None)
        if dialog:
            self.user_dialogs_map[dialog.user_type].pop(dialog.user_id, None)
            await self.agent.delete_thread(dialog.id, dialog.thread_id)
            await self.db.update_dialog(dialog.id, ended_at=datetime.now(), price=dialog.price)
            self.logger.info(f"Диалог с external_id={external_id} удален.")
        else:
            self.logger.error(f"Диалог с external_id={external_id} не найден для удаления.")

    async def update_dialog(self, external_id: int, new_user_id: int):
        dialog: Dialog = self.get_dialog(external_id)
        if not dialog:
            self.logger.error(f"Диалог с external_id={external_id} не найден.")
            return
        old_user_id = dialog.user_id
        dialog.user_id = new_user_id
        self.user_dialogs_map[dialog.user_type].pop(old_user_id, None)
        self.user_dialogs_map[dialog.user_type][new_user_id] = external_id
        await self.db.update_dialog(dialog.id, user_id=new_user_id)
        self.logger.info(f"Информация о диалоге обновлена: {dialog}")
